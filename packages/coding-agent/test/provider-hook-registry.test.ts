import {
	type ClassifierModel,
	createAssistantMessageEventStream,
	fauxAssistantMessage,
	type ImageModel,
	type Model,
	type ModelsSimpleStreamOptions,
	type ProviderHeaders,
} from "@earendil-works/pi-ai";
import { beforeEach, describe, expect, it } from "vitest";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { createEventBus } from "../src/core/event-bus.ts";
import { createExtensionRuntime, loadExtensionFromFactory } from "../src/core/extensions/loader.ts";
import { ExtensionRunner } from "../src/core/extensions/runner.ts";
import type { ExtensionAPI, ExtensionFactory } from "../src/core/extensions/types.ts";
import type { ModelRegistry } from "../src/core/model-registry.ts";
import { composeProviderHooks } from "../src/core/provider-hooks.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { createInMemoryModelRegistry } from "./model-runtime-test-utils.ts";

const model: Model<string> = {
	id: "chat",
	name: "Chat",
	api: "hooks",
	provider: "hooks",
	baseUrl: "https://hooks.invalid",
	reasoning: false,
	input: ["text"],
	contextWindow: 1000,
	maxTokens: 100,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
};

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

// #9773: real runner/registry/runtime wiring; only the provider boundary is synthetic.
describe("bound registry provider hooks", () => {
	let registry: ModelRegistry;
	let sent: Array<{ headers: ProviderHeaders | undefined; payload: unknown; maxTokens: number | undefined }>;

	beforeEach(async () => {
		registry = await createInMemoryModelRegistry(AuthStorage.inMemory());
		sent = [];
		registry.registerProvider(model.provider, {
			api: model.api,
			apiKey: "test-key",
			models: [model],
			streamSimple: (requestModel, _context, options) => {
				const stream = createAssistantMessageEventStream();
				const message = {
					...fauxAssistantMessage("ok"),
					api: requestModel.api,
					provider: requestModel.provider,
					model: requestModel.id,
				};
				void (async () => {
					try {
						const payload = await options?.onPayload?.({ original: true }, requestModel);
						sent.push({ headers: options?.headers, payload, maxTokens: options?.maxTokens });
						await options?.onResponse?.({ status: 201, headers: { "x-response": "yes" } }, requestModel);
						await options?.onProviderStreamEvent?.({ index: 1 }, requestModel);
						stream.end(message);
					} catch (error) {
						stream.end({ ...message, stopReason: "error", errorMessage: String(error) });
					}
				})();
				return stream;
			},
		});
	});

	async function createRunner(factory: ExtensionFactory) {
		const runtime = createExtensionRuntime();
		const extension = await loadExtensionFromFactory(factory, process.cwd(), createEventBus(), runtime);
		return new ExtensionRunner([extension], runtime, process.cwd(), SessionManager.inMemory(), registry);
	}

	it("completes with caller-first transformations and response events", async () => {
		const calls: string[] = [];
		const observed: unknown[] = [];
		const callerModels: Model<string>[] = [];
		const runner = await createRunner((pi) => {
			pi.on("before_provider_headers", (event) => {
				calls.push("headers");
				event.headers["x-hook"] = "yes";
			});
			pi.on("before_provider_request", (event) => {
				calls.push("payload");
				observed.push(event.payload);
				return { hook: true };
			});
			pi.on("after_provider_response", (event) => {
				calls.push("response");
				observed.push(event);
			});
			pi.on("provider_stream_event", (event) => {
				calls.push("event");
				observed.push(event);
			});
		});
		const options = Object.freeze<ModelsSimpleStreamOptions>({
			maxTokens: 17,
			transformHeaders: async () => {
				await Promise.resolve();
				calls.push("caller:headers");
				return { "x-caller": "yes" };
			},
			onPayload: async (_payload, requestModel) => {
				await Promise.resolve();
				calls.push("caller:payload");
				callerModels.push(requestModel);
				return { caller: true };
			},
			onResponse: async (_response, requestModel) => {
				await Promise.resolve();
				calls.push("caller:response");
				callerModels.push(requestModel);
			},
			onProviderStreamEvent: async (_event, requestModel) => {
				await Promise.resolve();
				calls.push("caller:event");
				callerModels.push(requestModel);
			},
		});
		const originalOptions = { ...options };

		const result = await runner.createContext().modelRegistry.complete(model, { messages: [] }, options);

		expect(result.stopReason).toBe("stop");
		expect(calls).toEqual([
			"caller:headers",
			"headers",
			"caller:payload",
			"payload",
			"caller:response",
			"response",
			"caller:event",
			"event",
		]);
		expect(sent).toEqual([
			{ headers: { "x-caller": "yes", "x-hook": "yes" }, payload: { hook: true }, maxTokens: 17 },
		]);
		expect(observed).toEqual([
			{ caller: true },
			{ type: "after_provider_response", status: 201, headers: { "x-response": "yes" } },
			{
				type: "provider_stream_event",
				data: { index: 1 },
				provider: model.provider,
				api: model.api,
				model: model.id,
			},
		]);
		expect(callerModels).toEqual([model, model, model]);
		expect(options).toEqual(originalOptions);
	});

	it("installs hooks for stream", async () => {
		const calls: string[] = [];
		const runner = await createRunner((pi) => recordHooks(pi, calls));

		const result = await runner.createContext().modelRegistry.stream(model, { messages: [] }).result();

		expect(result.stopReason).toBe("stop");
		expect(calls).toEqual(["headers", "payload", "response", "event"]);
	});

	it("installs hooks once when streamSimple routes to a physical model", async () => {
		const calls: string[] = [];
		const runner = await createRunner((pi) => recordHooks(pi, calls));
		const bound = runner.createContext().modelRegistry;
		bound.registerVirtualModel({
			provider: "router",
			id: "auto",
			name: "Auto",
			route: () => ({ model, thinkingLevel: "off" }),
		});

		const result = await bound.streamSimple(bound.find("router", "auto")!, { messages: [] }).result();

		expect(result.model).toBe(model.id);
		expect(result.stopReason).toBe("stop");
		expect(calls).toEqual(["headers", "payload", "response", "event"]);
	});

	it("installs image hooks without a stream callback", async () => {
		const image: ImageModel<string> = {
			type: "image",
			id: "image",
			name: "Image",
			api: "images",
			provider: "images",
			baseUrl: model.baseUrl,
			input: ["text"],
			output: ["image"],
			cost: model.cost,
		};
		const calls: string[] = [];
		let streamCallback: boolean | undefined;
		registry.registerProvider(image.provider, {
			apiKey: "test-key",
			models: [image],
			images: {
				images: {
					generateImages: async (requestModel, _context, options) => {
						streamCallback = Object.hasOwn(options ?? {}, "onProviderStreamEvent");
						await options?.onPayload?.({}, requestModel);
						await options?.onResponse?.({ status: 200, headers: {} }, requestModel);
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
		const runner = await createRunner((pi) => recordHooks(pi, calls));

		const result = await runner.createContext().modelRegistry.generateImages(image, { input: [] });

		expect(result.stopReason).toBe("stop");
		expect(calls).toEqual(["headers", "payload", "response"]);
		expect(streamCallback).toBe(false);
	});

	it("installs classifier hooks without a stream callback", async () => {
		const classifier: ClassifierModel<string> = {
			type: "classifier",
			id: "classifier",
			name: "Classifier",
			api: "classifier",
			provider: "classifier",
			baseUrl: model.baseUrl,
			input: ["text"],
			cost: model.cost,
			contextWindow: 1000,
		};
		const calls: string[] = [];
		let streamCallback: boolean | undefined;
		registry.registerProvider(classifier.provider, {
			apiKey: "test-key",
			models: [classifier],
			classifiers: {
				classifier: {
					classify: async (requestModel, _context, options) => {
						streamCallback = Object.hasOwn(options ?? {}, "onProviderStreamEvent");
						await options?.onPayload?.({}, requestModel);
						await options?.onResponse?.({ status: 200, headers: {} }, requestModel);
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
		const runner = await createRunner((pi) => recordHooks(pi, calls));

		const result = await runner.createContext().modelRegistry.classify(classifier, { state: {}, questions: {} });

		expect(result.stopReason).toBe("stop");
		expect(calls).toEqual(["headers", "payload", "response"]);
		expect(streamCallback).toBe(false);
	});

	it.each([
		{ replacement: null, expected: null },
		{ replacement: undefined, expected: { original: true } },
	])(
		"uses the original payload only when the caller returns undefined ($replacement)",
		async ({ replacement, expected }) => {
			const observed: unknown[] = [];
			const runner = await createRunner((pi) => {
				pi.on("before_provider_request", (event) => {
					observed.push(event.payload);
				});
			});
			const options = composeProviderHooks<Model<string>, ModelsSimpleStreamOptions>(
				model,
				{ onPayload: async () => replacement },
				runner,
			);

			const result = await options.onPayload!({ original: true }, model);

			expect(result).toEqual(expected);
			expect(observed).toEqual([expected]);
		},
	);

	it("propagates caller failure without dispatching the payload hook", async () => {
		const observed: unknown[] = [];
		const runner = await createRunner((pi) => {
			pi.on("before_provider_request", (event) => {
				observed.push(event.payload);
			});
		});
		const failure = new Error("caller failed");
		const options = composeProviderHooks<Model<string>, ModelsSimpleStreamOptions>(
			model,
			{
				onPayload: async () => {
					throw failure;
				},
			},
			runner,
		);

		await expect(options.onPayload!({}, model)).rejects.toBe(failure);
		expect(observed).toEqual([]);
	});

	it("rejects captured callbacks after invalidation even without handlers", async () => {
		const runner = await createRunner(() => {});
		const options = composeProviderHooks<Model<string>, ModelsSimpleStreamOptions>(model, undefined, runner);
		runner.invalidate();

		await expect(options.transformHeaders!({})).rejects.toThrow("stale");
		await expect(options.onPayload!({}, model)).rejects.toThrow("stale");
		await expect(options.onResponse!({ status: 200, headers: {} }, model)).rejects.toThrow("stale");
		await expect(options.onProviderStreamEvent!({}, model)).rejects.toThrow("stale");
	});

	it("allows an entered header hook to finish after invalidation", async () => {
		const entered = Promise.withResolvers<void>();
		const release = Promise.withResolvers<void>();
		const runner = await createRunner((pi) => {
			pi.on("before_provider_headers", async (event) => {
				entered.resolve();
				await release.promise;
				event.headers["x-finished"] = "yes";
			});
		});
		const options = composeProviderHooks<Model<string>, ModelsSimpleStreamOptions>(model, undefined, runner);

		const pending = options.transformHeaders!({});
		await entered.promise;
		runner.invalidate();
		release.resolve();

		await expect(pending).resolves.toEqual({ "x-finished": "yes" });
	});

	it("keeps stable runner bindings while sharing the original registry", async () => {
		const owners: string[] = [];
		const first = await createRunner((pi) => {
			pi.on("before_provider_request", () => {
				owners.push("A");
			});
		});
		const second = await createRunner((pi) => {
			pi.on("before_provider_request", () => {
				owners.push("B");
			});
		});
		const a = first.createContext().modelRegistry;
		const b = second.createContext().modelRegistry;
		expect(a).toBe(first.createCommandContext().modelRegistry);
		expect(a).not.toBe(b);
		a.registerVirtualModel({
			provider: "router",
			id: "shared",
			name: "Shared",
			route: () => ({ model, thinkingLevel: "off" }),
		});
		expect(b.find("router", "shared")).toBe(registry.find("router", "shared"));
		expect(b.find("router", "shared")).toBeDefined();

		await a.complete(model, { messages: [] });
		await b.complete(model, { messages: [] });
		await a.complete(model, { messages: [] });
		await registry.complete(model, { messages: [] });

		expect(owners).toEqual(["A", "B", "A"]);
		expect(sent).toHaveLength(4);
	});
});
