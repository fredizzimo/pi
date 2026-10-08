import {
	type AnyModel,
	type Api,
	isModelType,
	type Model,
	type ModelsApiStreamOptions,
	type ModelsRequestTransforms,
	type ProviderRequestOptions,
	type StreamOptions,
} from "@earendil-works/pi-ai";
import type { ExtensionRunner } from "./extensions/runner.ts";
import type { ModelRegistry } from "./model-registry.ts";

/** Install once at an owned execution entry point, not at recursive runtime dispatch. */
export function composeProviderHooks<
	TModel extends AnyModel,
	TOptions extends ProviderRequestOptions<TModel> & ModelsRequestTransforms,
>(model: TModel, options: TOptions | undefined, runner: ExtensionRunner): TOptions {
	runner.assertActive();
	const { transformHeaders, onPayload, onResponse } = options ?? {};
	const streamOptions = options as (TOptions & Pick<StreamOptions, "onProviderStreamEvent">) | undefined;
	const onProviderStreamEvent = streamOptions?.onProviderStreamEvent;
	return {
		...options,
		transformHeaders: async (headers) => {
			const transformed = transformHeaders ? await transformHeaders(headers) : headers;
			runner.assertActive();
			return runner.hasHandlers("before_provider_headers")
				? runner.emitBeforeProviderHeaders(transformed)
				: transformed;
		},
		onPayload: async (payload, requestModel) => {
			const replacement = await onPayload?.(payload, requestModel);
			const transformed = replacement === undefined ? payload : replacement;
			runner.assertActive();
			return runner.hasHandlers("before_provider_request")
				? runner.emitBeforeProviderRequest(transformed)
				: transformed;
		},
		onResponse: async (response, requestModel) => {
			await onResponse?.(response, requestModel);
			runner.assertActive();
			if (!runner.hasHandlers("after_provider_response")) return;
			await runner.emit({ type: "after_provider_response", status: response.status, headers: response.headers });
		},
		...(isModelType(model, "chat")
			? {
					onProviderStreamEvent: async (data: unknown, requestModel: Model<Api>) => {
						await onProviderStreamEvent?.(data, requestModel);
						runner.assertActive();
						if (!runner.hasHandlers("provider_stream_event")) return;
						await runner.emit({
							type: "provider_stream_event",
							data,
							provider: requestModel.provider,
							api: requestModel.api,
							model: requestModel.id,
						});
					},
				}
			: {}),
	} as TOptions;
}

/** Wrap execution only; all other methods remain bound to the original facade. */
export function bindModelRegistry(registry: ModelRegistry, runner: ExtensionRunner): ModelRegistry {
	const execution: Pick<ModelRegistry, "stream" | "streamSimple" | "complete" | "classify" | "generateImages"> = {
		stream: <TApi extends Api>(
			model: Model<TApi>,
			context: Parameters<ModelRegistry["stream"]>[1],
			options?: ModelsApiStreamOptions<TApi>,
		) => registry.stream(model, context, composeProviderHooks(model, options, runner)),
		streamSimple: (model, context, options) =>
			registry.streamSimple(model, context, composeProviderHooks(model, options, runner)),
		complete: <TApi extends Api>(
			model: Model<TApi>,
			context: Parameters<ModelRegistry["complete"]>[1],
			options?: ModelsApiStreamOptions<TApi>,
		) => registry.complete(model, context, composeProviderHooks(model, options, runner)),
		classify: (model, context, options) =>
			registry.classify(model, context, composeProviderHooks(model, options, runner)),
		generateImages: (model, context, options) =>
			registry.generateImages(model, context, composeProviderHooks(model, options, runner)),
	};
	const methods = new Map<PropertyKey, unknown>();
	return new Proxy(registry, {
		get(target, property) {
			if (Object.hasOwn(execution, property)) return Reflect.get(execution, property);
			const value: unknown = Reflect.get(target, property);
			if (typeof value !== "function") return value;
			if (!methods.has(property)) methods.set(property, value.bind(target));
			return methods.get(property);
		},
	});
}
