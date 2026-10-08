import {
	type Api,
	type ClassifierModel,
	createAssistantMessageEventStream,
	type ImageModel,
	type Model,
	type ModelsSimpleStreamOptions,
	type SimpleStreamOptions,
} from "@earendil-works/pi-ai";
import { beforeEach, describe, expect, it } from "vitest";
import { AuthStorage } from "../src/core/auth-storage.ts";
import type { ExtensionAPI } from "../src/core/extensions/types.ts";
import type { ModelRegistry } from "../src/core/model-registry.ts";
import type { ModelRuntime } from "../src/core/model-runtime.ts";
import { createInMemoryModelRegistry, getModelRuntime } from "./model-runtime-test-utils.ts";
import { createRunner, message, model } from "./provider-hook-test-utils.ts";

function recordHooks(pi: ExtensionAPI, calls: string[]) {
	pi.on("before_provider_headers", () => {
		calls.push("headers");
	});
	pi.on("before_provider_request", () => {
		calls.push("payload");
	});
	pi.on("after_provider_response", () => {
		calls.push("response");
	});
	pi.on("provider_stream_event", () => {
		calls.push("event");
	});
}

// #9773: public registry wiring uses real runner/registry/runtime components, not SDK sessions.
describe("bound registry execution", () => {
	let registry: ModelRegistry;
	let runtime: ModelRuntime;

	beforeEach(async () => {
		registry = await createInMemoryModelRegistry(AuthStorage.inMemory());
		runtime = getModelRuntime(registry);
	});

	it.each(["stream", "streamSimple", "complete"] as const)("installs chat hooks once for %s", async (method) => {
		const calls: string[] = [];
		const callerModels: Model<Api>[] = [];
		const dispatched: Model<Api>[] = [];
		runtime.registerProvider(model.provider, {
			api: model.api,
			apiKey: "hook-key",
			models: [model],
			streamSimple: (requestModel, _context, options) => {
				dispatched.push(requestModel);
				const result = createAssistantMessageEventStream();
				void (async () => {
					try {
						await options?.onPayload?.({}, requestModel);
						await options?.onResponse?.({ status: 200, headers: {} }, requestModel);
						await options?.onProviderStreamEvent?.({}, requestModel);
						result.end(message(requestModel));
					} catch (error) {
						result.end({ ...message(requestModel), stopReason: "error", errorMessage: String(error) });
					}
				})();
				return result;
			},
		});
		const runner = await createRunner((pi) => recordHooks(pi, calls), registry);
		const bound = runner.createContext().modelRegistry;
		const options = Object.freeze<ModelsSimpleStreamOptions>({
			transformHeaders: (headers) => {
				calls.push("caller:headers");
				return headers;
			},
			onPayload: (_payload, requestModel) => {
				callerModels.push(requestModel);
			},
			onResponse: (_response, requestModel) => {
				callerModels.push(requestModel);
			},
			onProviderStreamEvent: (_event, requestModel) => {
				callerModels.push(requestModel);
			},
		});

		const result =
			method === "complete"
				? await bound.complete(model, { messages: [] }, options)
				: await bound[method](model, { messages: [] }, options).result();

		expect(result.stopReason).toBe("stop");
		expect(calls).toEqual(["caller:headers", "headers", "payload", "response", "event"]);
		expect(callerModels).toEqual([model, model, model]);
		expect(dispatched).toEqual([model]);
		expect(options).not.toHaveProperty("headers");
		expect(bound).toBe(runner.createCommandContext().modelRegistry);
		runner.invalidate();
		expect(() => bound[method](model, { messages: [] })).toThrow("stale");
	});

	it("installs image hooks and forwards the model without a stream callback", async () => {
		const calls: string[] = [];
		const callerModels: ImageModel<string>[] = [];
		const image: ImageModel<string> = {
			type: "image",
			id: "image",
			name: "image",
			api: "hook-images",
			provider: "operations",
			baseUrl: model.baseUrl,
			input: ["text"],
			output: ["image"],
			cost: model.cost,
		};
		let streamCallbackInstalled: boolean | undefined;
		runtime.registerProvider("operations", {
			apiKey: "operations-key",
			models: [image],
			images: {
				"hook-images": {
					generateImages: async (requestModel, _context, options) => {
						streamCallbackInstalled = Object.hasOwn(options ?? {}, "onProviderStreamEvent");
						await options?.onPayload?.({}, requestModel);
						await options?.onResponse?.({ status: 201, headers: {} }, requestModel);
						return {
							api: requestModel.api,
							provider: requestModel.provider,
							model: requestModel.id,
							output: [],
							stopReason: "stop",
							timestamp: 0,
						};
					},
				},
			},
		});
		const runner = await createRunner((pi) => recordHooks(pi, calls), registry);

		const result = await runner.createContext().modelRegistry.generateImages(
			image,
			{ input: [{ type: "text", text: "circle" }] },
			{
				onPayload: (_payload, requestModel) => {
					callerModels.push(requestModel);
				},
				onResponse: (_response, requestModel) => {
					callerModels.push(requestModel);
				},
			},
		);

		expect(result.stopReason).toBe("stop");
		expect(calls).toEqual(["headers", "payload", "response"]);
		expect(callerModels).toEqual([image, image]);
		expect(streamCallbackInstalled).toBe(false);
		const bound = runner.createContext().modelRegistry;
		runner.invalidate();
		expect(() => bound.generateImages(image, { input: [] })).toThrow("stale");
	});

	it("installs classifier hooks and forwards the model without a stream callback", async () => {
		const calls: string[] = [];
		const callerModels: ClassifierModel<string>[] = [];
		const classifier: ClassifierModel<string> = {
			type: "classifier",
			id: "classifier",
			name: "classifier",
			api: "hook-classifier",
			provider: "operations",
			baseUrl: model.baseUrl,
			input: ["text"],
			cost: model.cost,
			contextWindow: 1000,
		};
		let streamCallbackInstalled: boolean | undefined;
		runtime.registerProvider("operations", {
			apiKey: "operations-key",
			models: [classifier],
			classifiers: {
				"hook-classifier": {
					classify: async (requestModel, _context, options) => {
						streamCallbackInstalled = Object.hasOwn(options ?? {}, "onProviderStreamEvent");
						await options?.onPayload?.({}, requestModel);
						await options?.onResponse?.({ status: 201, headers: {} }, requestModel);
						return {
							api: requestModel.api,
							provider: requestModel.provider,
							model: requestModel.id,
							answers: {},
							stopReason: "stop",
							timestamp: 0,
						};
					},
				},
			},
		});
		const runner = await createRunner((pi) => recordHooks(pi, calls), registry);

		const result = await runner.createContext().modelRegistry.classify(
			classifier,
			{ state: {}, questions: {} },
			{
				onPayload: (_payload, requestModel) => {
					callerModels.push(requestModel);
				},
				onResponse: (_response, requestModel) => {
					callerModels.push(requestModel);
				},
			},
		);

		expect(result.stopReason).toBe("stop");
		expect(calls).toEqual(["headers", "payload", "response"]);
		expect(callerModels).toEqual([classifier, classifier]);
		expect(streamCallbackInstalled).toBe(false);
		const bound = runner.createContext().modelRegistry;
		runner.invalidate();
		expect(() => bound.classify(classifier, { state: {}, questions: {} })).toThrow("stale");
	});

	// Issue 13: inherited facade methods use the same runtime; only owned execution is overridden.
	it("shares catalog, auth and registration without changing the original registry", async () => {
		const calls: string[] = [];
		const originalComplete = registry.complete;
		const dispatched = Promise.withResolvers<SimpleStreamOptions | undefined>();
		const runner = await createRunner((pi) => recordHooks(pi, calls), registry);
		const bound = runner.createContext().modelRegistry;
		bound.registerProvider(model.provider, {
			api: model.api,
			apiKey: "hook-key",
			models: [model],
			streamSimple: (_model, _context, options) => {
				dispatched.resolve(options);
				const result = createAssistantMessageEventStream();
				result.end(message());
				return result;
			},
		});

		expect(bound).not.toBe(registry);
		expect(bound.find(model.provider, model.id)).toEqual(registry.find(model.provider, model.id));
		expect(bound.getAll()).toEqual(registry.getAll());
		expect(bound.getAvailable()).toEqual(registry.getAvailable());
		expect(bound.getProvider(model.provider)).toEqual(registry.getProvider(model.provider));
		expect(bound.getRegisteredProviderConfig(model.provider)).toEqual(
			registry.getRegisteredProviderConfig(model.provider),
		);
		expect(bound.hasConfiguredAuth(model)).toBe(true);
		expect(await bound.getApiKeyAndHeaders(model)).toEqual(await registry.getApiKeyAndHeaders(model));
		expect(await bound.getApiKeyForProvider(model.provider)).toBe("hook-key");
		await bound.refresh();
		expect(bound.find(model.provider, model.id)).toEqual(registry.find(model.provider, model.id));

		runner.invalidate();
		registry.streamSimple(model, { messages: [] });
		const options = await dispatched.promise;
		expect(options?.onPayload).toBeUndefined();
		expect(options?.onResponse).toBeUndefined();
		expect(options?.onProviderStreamEvent).toBeUndefined();
		expect(calls).toEqual([]);
		expect(registry.complete).toBe(originalComplete);

		registry.registerVirtualModel({
			provider: "router",
			id: "auto",
			name: "Auto",
			route: () => ({ model, thinkingLevel: "off" }),
		});
		expect(bound.find("router", "auto")).toEqual(registry.find("router", "auto"));
		bound.unregisterVirtualModel("router", "auto");
		expect(registry.find("router", "auto")).toBeUndefined();
		bound.unregisterProvider(model.provider);
		expect(registry.find(model.provider, model.id)).toBeUndefined();
	});
});
