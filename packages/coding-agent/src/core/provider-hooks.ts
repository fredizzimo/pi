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
import type { ModelRegistry } from "./model-registry.ts";

export type ProviderHooks = Pick<ProviderRequestOptions<AnyModel>, "onPayload" | "onResponse"> &
	Pick<StreamOptions, "onProviderStreamEvent"> &
	ModelsRequestTransforms;

/** Install once at an owned execution entry point, not at recursive runtime dispatch. */
export function composeProviderHooks<
	TModel extends AnyModel,
	TOptions extends ProviderRequestOptions<TModel> & ModelsRequestTransforms,
>(model: TModel, options: TOptions | undefined, hooks: ProviderHooks): TOptions {
	const { transformHeaders, onPayload, onResponse } = options ?? {};
	const streamOptions = options as (TOptions & Pick<StreamOptions, "onProviderStreamEvent">) | undefined;
	const onProviderStreamEvent = streamOptions?.onProviderStreamEvent;
	return {
		...options,
		transformHeaders: async (headers) => {
			const transformed = transformHeaders ? await transformHeaders(headers) : headers;
			return hooks.transformHeaders ? hooks.transformHeaders(transformed) : transformed;
		},
		onPayload: async (payload, requestModel) => {
			const replacement = await onPayload?.(payload, requestModel);
			const transformed = replacement === undefined ? payload : replacement;
			return hooks.onPayload ? hooks.onPayload(transformed, requestModel) : transformed;
		},
		onResponse: async (response, requestModel) => {
			await onResponse?.(response, requestModel);
			await hooks.onResponse?.(response, requestModel);
		},
		...(isModelType(model, "chat")
			? {
					onProviderStreamEvent: async (data: unknown, requestModel: Model<Api>) => {
						await onProviderStreamEvent?.(data, requestModel);
						await hooks.onProviderStreamEvent?.(data, requestModel);
					},
				}
			: {}),
	} as TOptions;
}

/** Wrap execution only; all other methods remain bound to the original facade. */
export function bindModelRegistry(
	registry: ModelRegistry,
	hooks: ProviderHooks,
	assertActive: () => void,
): ModelRegistry {
	function prepareOptions<
		TModel extends AnyModel,
		TOptions extends ProviderRequestOptions<TModel> & ModelsRequestTransforms,
	>(model: TModel, options: TOptions | undefined): TOptions {
		assertActive();
		return composeProviderHooks(model, options, hooks);
	}

	const execution: Pick<ModelRegistry, "stream" | "streamSimple" | "complete" | "classify" | "generateImages"> = {
		stream: <TApi extends Api>(
			model: Model<TApi>,
			context: Parameters<ModelRegistry["stream"]>[1],
			options?: ModelsApiStreamOptions<TApi>,
		) => registry.stream(model, context, prepareOptions(model, options)),
		streamSimple: (model, context, options) => registry.streamSimple(model, context, prepareOptions(model, options)),
		complete: <TApi extends Api>(
			model: Model<TApi>,
			context: Parameters<ModelRegistry["complete"]>[1],
			options?: ModelsApiStreamOptions<TApi>,
		) => registry.complete(model, context, prepareOptions(model, options)),
		classify: (model, context, options) => registry.classify(model, context, prepareOptions(model, options)),
		generateImages: (model, context, options) =>
			registry.generateImages(model, context, prepareOptions(model, options)),
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
