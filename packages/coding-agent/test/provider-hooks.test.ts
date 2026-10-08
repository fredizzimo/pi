import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	type Api,
	type AssistantMessage,
	type ClassifierModel,
	type ClassifierOptions,
	createAssistantMessageEventStream,
	type ImageModel,
	type ImagesOptions,
	type Model,
	type ModelsSimpleStreamOptions,
	normalizeContext,
	type SimpleStreamOptions,
} from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AgentSession } from "../src/core/agent-session.ts";
import { AuthStorage } from "../src/core/auth-storage.ts";
import type { ExtensionAPI, ExtensionFactory } from "../src/core/extensions/types.ts";
import type { ModelRuntime } from "../src/core/model-runtime.ts";
import { DefaultResourceLoader } from "../src/core/resource-loader.ts";
import { createAgentSession } from "../src/core/sdk.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import { createInMemoryModelRegistry, getModelRuntime } from "./model-runtime-test-utils.ts";

const model: Model<Api> = {
	id: "hook-model",
	name: "Hook Model",
	api: "hook-api",
	provider: "hook-provider",
	baseUrl: "https://hooks.invalid",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 10_000,
	maxTokens: 1000,
};

function message(): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text: "response" }],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: {
			input: 1,
			output: 1,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 2,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: Date.now(),
	};
}

// Regression coverage for #9773: exercise real SDK/runner/registry/runtime wiring.
// Only the provider boundary is synthetic, so no endpoint or paid request is used.
describe("session provider hooks", () => {
	let tempDir: string;
	const sessions: AgentSession[] = [];
	let sent: Array<{ payload: unknown; options?: SimpleStreamOptions }>;
	let runtime: ModelRuntime;

	beforeEach(async () => {
		tempDir = mkdtempSync(join(tmpdir(), "pi-provider-hooks-"));
		sent = [];
		const credentials = AuthStorage.inMemory();
		await credentials.modify(model.provider, async () => ({ type: "api_key", key: "hook-key" }));
		runtime = getModelRuntime(await createInMemoryModelRegistry(credentials));
		const stream = (requestModel: Model<Api>, _context: unknown, options?: SimpleStreamOptions) => {
			const result = createAssistantMessageEventStream();
			void (async () => {
				try {
					const payload = { original: true };
					const replacement = await options?.onPayload?.(payload, requestModel);
					sent.push({ payload: replacement === undefined ? payload : replacement, options });
					await options?.onResponse?.({ status: 200, headers: { "x-response": "yes" } }, requestModel);
					await options?.onProviderStreamEvent?.({ index: 1 }, requestModel);
					await options?.onProviderStreamEvent?.({ index: 2 }, requestModel);
					result.end(message());
				} catch (error) {
					result.end({ ...message(), stopReason: "error", errorMessage: String(error) });
				}
			})();
			return result;
		};
		runtime.registerProvider(model.provider, {
			api: model.api,
			models: [{ ...model }],
			headers: { "x-provider": "yes" },
			streamSimple: stream,
		});
	});

	afterEach(() => {
		for (const session of sessions.splice(0)) session.dispose();
		rmSync(tempDir, { recursive: true, force: true });
	});

	async function createSession(factory: ExtensionFactory) {
		const cwd = join(tempDir, `session-${sessions.length}`);
		mkdirSync(cwd);
		const settingsManager = SettingsManager.inMemory({
			compaction: { enabled: false },
			retry: { enabled: false },
		});
		const resourceLoader = new DefaultResourceLoader({
			cwd,
			agentDir: cwd,
			settingsManager,
			extensionFactories: [factory],
		});
		await resourceLoader.reload();
		const { session } = await createAgentSession({
			cwd,
			agentDir: cwd,
			model,
			modelRuntime: runtime,
			settingsManager,
			sessionManager: SessionManager.inMemory(cwd),
			resourceLoader,
			noTools: "all",
		});
		sessions.push(session);
		return session;
	}

	describe("registry and Agent wiring", () => {
		it.each(["stream", "streamSimple", "complete"] as const)(
			"composes all hooks once for registry %s",
			async (method) => {
				const calls: string[] = [];
				const session = await createSession((pi) => {
					pi.on("before_provider_headers", (event) => {
						calls.push("hook:headers");
						expect(event.headers).toEqual({ "x-caller": "yes" });
						event.headers["x-hook"] = "yes";
					});
					pi.on("before_provider_request", (event) => {
						calls.push("hook:payload");
						expect(event.payload).toEqual({ caller: true });
						return { hook: true };
					});
					pi.on("after_provider_response", () => {
						calls.push("hook:response");
					});
					pi.on("provider_stream_event", (event) => {
						calls.push(`hook:event:${(event.data as { index: number }).index}`);
					});
				});
				const registry = session.extensionRunner.createContext().modelRegistry;
				expect(registry).toBe(session.extensionRunner.createCommandContext().modelRegistry);
				const options: ModelsSimpleStreamOptions = Object.freeze({
					transformHeaders: async () => {
						calls.push("caller:headers");
						return { "x-caller": "yes" };
					},
					onPayload: async () => {
						calls.push("caller:payload");
						return { caller: true };
					},
					onResponse: async () => {
						calls.push("caller:response");
					},
					onProviderStreamEvent: async (data: unknown) => {
						calls.push(`caller:event:${(data as { index: number }).index}`);
					},
				});

				const response =
					method === "complete"
						? await registry.complete(model, normalizeContext({ messages: [] }), options)
						: await registry[method](model, normalizeContext({ messages: [] }), options).result();

				expect(response.stopReason).toBe("stop");
				expect(calls).toEqual([
					"caller:headers",
					"hook:headers",
					"caller:payload",
					"hook:payload",
					"caller:response",
					"hook:response",
					"caller:event:1",
					"hook:event:1",
					"caller:event:2",
					"hook:event:2",
				]);
				expect(sent).toHaveLength(1);
				expect(sent[0].payload).toEqual({ hook: true });
				expect(sent[0].options?.headers).toEqual({ "x-caller": "yes", "x-hook": "yes" });
				expect(options).not.toHaveProperty("headers");
			},
		);

		it("preserves exactly-once Agent hook delivery for normal turns", async () => {
			const calls: string[] = [];
			const session = await createSession((pi) => {
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
			});

			await session.prompt("turn");

			expect(calls).toEqual(["headers", "payload", "response", "event", "event"]);
			expect(sent).toHaveLength(1);
		});

		it("retains runner error reporting and payload chaining", async () => {
			const session = await createSession((pi) => {
				pi.on("before_provider_request", (event) => {
					(event.payload as Record<string, unknown>).mutated = true;
					throw new Error("extension failed");
				});
				pi.on("before_provider_request", (event) => ({ final: event.payload }));
			});
			const errors: string[] = [];
			session.extensionRunner.onError((error) => errors.push(error.error));

			const result = await session.extensionRunner.createContext().modelRegistry.complete(model, { messages: [] });

			expect(result.stopReason).toBe("stop");
			expect(sent[0].payload).toEqual({ final: { original: true, mutated: true } });
			expect(errors).toEqual(["extension failed"]);
		});

		it("isolates interleaved requests from sessions sharing a runtime without wire session ids", async () => {
			const observed: Array<{ owner: string; event: string }> = [];
			const first = await createSession((pi) => {
				pi.on("before_provider_request", () => {
					observed.push({ owner: "A", event: "payload" });
				});
				pi.on("provider_stream_event", () => {
					observed.push({ owner: "A", event: "event" });
				});
			});
			const second = await createSession((pi) => {
				pi.on("before_provider_request", () => {
					observed.push({ owner: "B", event: "payload" });
				});
				pi.on("provider_stream_event", () => {
					observed.push({ owner: "B", event: "event" });
				});
			});
			const entered = Promise.withResolvers<void>();
			const release = Promise.withResolvers<void>();
			const pending = first.extensionRunner.createContext().modelRegistry.complete(
				model,
				{ messages: [] },
				{
					onResponse: async () => {
						entered.resolve();
						await release.promise;
					},
				},
			);
			await entered.promise;
			await second.extensionRunner.createContext().modelRegistry.complete(model, { messages: [] });
			release.resolve();
			await pending;
			await runtime.complete(model, { messages: [] });

			expect(observed).toEqual([
				{ owner: "A", event: "payload" },
				{ owner: "B", event: "payload" },
				{ owner: "B", event: "event" },
				{ owner: "B", event: "event" },
				{ owner: "A", event: "event" },
				{ owner: "A", event: "event" },
			]);
			expect(sent.map((request) => request.options?.sessionId)).toEqual([undefined, undefined, undefined]);
		});

		it("keeps guarded nested registry requests hooked", async () => {
			const calls: string[] = [];
			let nested = false;
			const session = await createSession((pi) => {
				pi.on("before_provider_request", async (_event, ctx) => {
					calls.push(nested ? "nested:payload" : "outer:payload");
					if (!nested) {
						nested = true;
						await ctx.modelRegistry.complete(model, { messages: [] });
						nested = false;
					}
				});
				pi.on("after_provider_response", () => {
					calls.push(nested ? "nested:response" : "outer:response");
				});
			});

			await session.extensionRunner.createContext().modelRegistry.complete(model, { messages: [] });

			expect(calls).toEqual(["outer:payload", "nested:payload", "nested:response", "outer:response"]);
			expect(sent).toHaveLength(2);
		});

		it("keeps hook composition once through virtual stream routing", async () => {
			const calls: string[] = [];
			const session = await createSession((pi) => {
				pi.on("before_provider_headers", () => {
					calls.push("headers");
				});
				pi.on("provider_stream_event", (event) => {
					calls.push(event.model);
				});
			});
			const registry = session.extensionRunner.createContext().modelRegistry;
			registry.registerVirtualModel({
				provider: "router",
				id: "auto",
				name: "Auto",
				route: () => ({ model, thinkingLevel: "off" }),
			});
			const virtual = registry.find("router", "auto")!;

			const result = await registry.streamSimple(virtual, { messages: [] }).result();

			expect(result.stopReason).toBe("stop");
			expect(calls).toEqual(["headers", "hook-model", "hook-model"]);
			expect(sent).toHaveLength(1);
		});
	});

	describe("runner lifetime", () => {
		it("rejects stale registry callbacks after reload without retargeting", async () => {
			const calls: string[] = [];
			let generation = 0;
			const session = await createSession((pi) => {
				const owner = ++generation;
				pi.on("before_provider_request", () => {
					calls.push(`${owner}:payload`);
				});
				pi.on("after_provider_response", () => {
					calls.push(`${owner}:response`);
				});
				pi.on("provider_stream_event", () => {
					calls.push(`${owner}:event`);
				});
			});
			const entered = Promise.withResolvers<void>();
			const release = Promise.withResolvers<void>();
			const registry = session.extensionRunner.createContext().modelRegistry;
			const options = {
				onResponse: async () => {
					entered.resolve();
					await release.promise;
					calls.push("caller:response");
				},
			};
			const stream = registry.streamSimple(model, { messages: [] }, options);
			await entered.promise;
			await session.reload();
			expect(() => registry.streamSimple(model, { messages: [] })).toThrow("stale");
			expect(session.extensionRunner.createContext().modelRegistry).not.toBe(registry);
			release.resolve();
			const result = await stream.result();
			expect(result.stopReason).toBe("error");
			expect(result.errorMessage).toContain("stale");
			expect(calls).toEqual(["1:payload", "caller:response"]);

			await session.extensionRunner.createContext().modelRegistry.complete(model, { messages: [] });
			expect(calls).toEqual(["1:payload", "caller:response", "2:payload", "2:response", "2:event", "2:event"]);
		});

		// #9773: dispose during each caller callback, before its matching session hook enters.
		it.each([
			{
				callback: "transformHeaders",
				register: (pi: ExtensionAPI, handler: () => void) => pi.on("before_provider_headers", handler),
				sentCount: 0,
			},
			{
				callback: "onPayload",
				register: (pi: ExtensionAPI, handler: () => void) => pi.on("before_provider_request", handler),
				sentCount: 0,
			},
			{
				callback: "onResponse",
				register: (pi: ExtensionAPI, handler: () => void) => pi.on("after_provider_response", handler),
				sentCount: 1,
			},
			{
				callback: "onProviderStreamEvent",
				register: (pi: ExtensionAPI, handler: () => void) => pi.on("provider_stream_event", handler),
				sentCount: 1,
			},
		])("rejects disposed $callback after awaiting its caller callback", async ({ callback, register, sentCount }) => {
			const calls: string[] = [];
			const session = await createSession((pi) => {
				register(pi, () => {
					calls.push("hook");
				});
			});
			const entered = Promise.withResolvers<void>();
			const release = Promise.withResolvers<void>();
			const registry = session.extensionRunner.createContext().modelRegistry;
			const response = registry.complete(
				model,
				{ messages: [] },
				{
					[callback]: async (value: unknown) => {
						entered.resolve();
						await release.promise;
						calls.push("caller:finished");
						return value;
					},
				},
			);
			await entered.promise;
			session.dispose();
			release.resolve();

			const result = await response;
			expect(result.stopReason).toBe("error");
			expect(result.errorMessage).toContain("stale");
			expect(calls).toEqual(["caller:finished"]);
			expect(sent).toHaveLength(sentCount);
			expect(() => registry.complete(model, { messages: [] })).toThrow("stale");
			expect(() => registry.stream(model, { messages: [] })).toThrow("stale");
		});

		// #9773: the activity precondition also applies when no session handler exists.
		it.each([
			{ callback: "transformHeaders", nextCallback: "onPayload", sentCount: 0 },
			{ callback: "onPayload", nextCallback: "onResponse", sentCount: 0 },
			{ callback: "onResponse", nextCallback: "onProviderStreamEvent", sentCount: 1 },
		])(
			"rejects stale $callback before $nextCallback even without registered handlers",
			async ({ callback, nextCallback, sentCount }) => {
				const session = await createSession(() => {});
				const registry = session.extensionRunner.createContext().modelRegistry;
				const calls: string[] = [];

				const result = await registry.complete(
					model,
					{ messages: [] },
					{
						[callback]: async (value: unknown) => {
							calls.push("caller:dispose");
							session.dispose();
							return value;
						},
						[nextCallback]: async () => {
							calls.push("caller:next");
						},
					},
				);

				expect(result.stopReason).toBe("error");
				expect(result.errorMessage).toContain("stale");
				expect(calls).toEqual(["caller:dispose"]);
				expect(sent).toHaveLength(sentCount);
			},
		);

		// #9773: stream events repeat the same callback; no second caller entry may occur.
		it("rejects the first stale stream event even without registered handlers", async () => {
			const session = await createSession(() => {});
			const registry = session.extensionRunner.createContext().modelRegistry;
			const events: unknown[] = [];

			const result = await registry.complete(
				model,
				{ messages: [] },
				{
					onProviderStreamEvent: async (data) => {
						events.push(data);
						session.dispose();
					},
				},
			);

			expect(result.stopReason).toBe("error");
			expect(result.errorMessage).toContain("stale");
			expect(events).toEqual([{ index: 1 }]);
		});

		// #9773: an entered hook may finish; activity is checked on the next callback's entry.
		it.each([
			{
				hook: "before_provider_headers",
				register: (pi: ExtensionAPI, handler: () => Promise<void>) => pi.on("before_provider_headers", handler),
				nextCallback: "onPayload",
			},
			{
				hook: "before_provider_request",
				register: (pi: ExtensionAPI, handler: () => Promise<void>) => pi.on("before_provider_request", handler),
				nextCallback: "onResponse",
			},
			{
				hook: "after_provider_response",
				register: (pi: ExtensionAPI, handler: () => Promise<void>) => pi.on("after_provider_response", handler),
				nextCallback: "onProviderStreamEvent",
			},
		])("allows entered $hook to finish before stale $nextCallback fails", async ({ register, nextCallback }) => {
			const calls: string[] = [];
			const entered = Promise.withResolvers<void>();
			const release = Promise.withResolvers<void>();
			const session = await createSession((pi) => {
				register(pi, async () => {
					calls.push("hook:entered");
					entered.resolve();
					await release.promise;
					calls.push("hook:finished");
				});
			});
			const registry = session.extensionRunner.createContext().modelRegistry;
			const response = registry.complete(
				model,
				{ messages: [] },
				{
					[nextCallback]: async () => {
						calls.push("caller:next");
					},
				},
			);
			await entered.promise;
			session.dispose();
			release.resolve();

			const result = await response;
			expect(result.stopReason).toBe("error");
			expect(result.errorMessage).toContain("stale");
			expect(calls).toEqual(["hook:entered", "hook:finished", "caller:next"]);
		});

		// #9773: repeated stream events re-enter the same callback rather than advancing to another hook.
		it("finishes an entered stream hook but rejects the next stream event after disposal", async () => {
			const calls: string[] = [];
			const entered = Promise.withResolvers<void>();
			const release = Promise.withResolvers<void>();
			const session = await createSession((pi) => {
				pi.on("provider_stream_event", async (event) => {
					calls.push(`hook:entered:${(event.data as { index: number }).index}`);
					entered.resolve();
					await release.promise;
					calls.push("hook:finished");
				});
			});
			const pending = session.extensionRunner.createContext().modelRegistry.complete(
				model,
				{ messages: [] },
				{
					onProviderStreamEvent: async (data) => {
						calls.push(`caller:event:${(data as { index: number }).index}`);
					},
				},
			);
			await entered.promise;
			session.dispose();
			release.resolve();

			const result = await pending;
			expect(result.stopReason).toBe("error");
			expect(result.errorMessage).toContain("stale");
			expect(calls).toEqual(["caller:event:1", "hook:entered:1", "hook:finished", "caller:event:2"]);
		});

		it("keeps captured registries revoked when reload fails", async () => {
			const session = await createSession(() => {});
			const registry = session.extensionRunner.createContext().modelRegistry;
			const reload = vi.spyOn(session.resourceLoader, "reload").mockRejectedValueOnce(new Error("load failed"));

			await expect(session.reload()).rejects.toThrow("load failed");

			expect(() => registry.complete(model, { messages: [] })).toThrow("stale");
			expect(sent).toEqual([]);
			reload.mockRestore();
			await session.reload();
			expect(
				(await session.extensionRunner.createContext().modelRegistry.complete(model, { messages: [] })).stopReason,
			).toBe("stop");
		});

		it("rejects registry execution invalidated during authentication", async () => {
			const session = await createSession(() => {});
			const entered = Promise.withResolvers<void>();
			const release = Promise.withResolvers<void>();
			const getAuth = runtime.getAuth.bind(runtime);
			// Hold the actual auth boundary to invalidate between admission and physical dispatch.
			const auth = vi.spyOn(runtime, "getAuth").mockImplementation(async (request, overrides) => {
				entered.resolve();
				await release.promise;
				return typeof request === "string" ? getAuth(request, overrides) : getAuth(request, overrides);
			});
			const registry = session.extensionRunner.createContext().modelRegistry;
			const result = registry.complete(model, { messages: [] });
			await entered.promise;
			session.dispose();
			release.resolve();

			expect((await result).errorMessage).toContain("stale");
			expect(sent).toEqual([]);
			auth.mockRestore();
		});
	});

	describe("non-chat registry wiring", () => {
		// #9773: each non-chat entry point installs only header/payload/response callbacks.
		it("binds image execution without parsed-event callbacks", async () => {
			const observed: Array<{ payload: unknown; options?: ImagesOptions }> = [];
			const calls: string[] = [];
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
			runtime.registerProvider("operations", {
				apiKey: "operations-key",
				models: [image],
				images: {
					"hook-images": {
						generateImages: async (requestModel, _context, options) => {
							observed.push({
								payload: await options?.onPayload?.({ operation: "image" }, requestModel),
								options,
							});
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
			const session = await createSession((pi) => {
				pi.on("before_provider_headers", (event) => {
					calls.push("hook:headers");
					event.headers["x-hook"] = "yes";
				});
				pi.on("before_provider_request", (event) => {
					calls.push("hook:payload");
					return { wrapped: event.payload };
				});
				pi.on("after_provider_response", () => {
					calls.push("hook:response");
				});
				pi.on("provider_stream_event", () => {
					calls.push("hook:event");
				});
			});
			const callerModels: ImageModel<string>[] = [];

			const result = await session.extensionRunner.createContext().modelRegistry.generateImages(
				image,
				{ input: [{ type: "text", text: "circle" }] },
				{
					transformHeaders: async () => {
						calls.push("caller:headers");
						return { "x-caller": "yes" };
					},
					onPayload: async (payload, requestModel) => {
						calls.push("caller:payload");
						callerModels.push(requestModel);
						return { caller: payload };
					},
					onResponse: async (_response, requestModel) => {
						calls.push("caller:response");
						callerModels.push(requestModel);
					},
				},
			);

			expect(result.stopReason).toBe("stop");
			expect(callerModels).toEqual([image, image]);
			expect(observed).toHaveLength(1);
			expect(observed[0].payload).toEqual({ wrapped: { caller: { operation: "image" } } });
			expect(observed[0].options?.headers).toEqual({ "x-caller": "yes", "x-hook": "yes" });
			expect(observed[0].options).not.toHaveProperty("onProviderStreamEvent");
			expect(calls).toEqual([
				"caller:headers",
				"hook:headers",
				"caller:payload",
				"hook:payload",
				"caller:response",
				"hook:response",
			]);
		});

		// #9773: classifier callback/model forwarding is independent of image dispatch.
		it("binds classifier execution without parsed-event callbacks", async () => {
			const observed: Array<{ payload: unknown; options?: ClassifierOptions }> = [];
			const calls: string[] = [];
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
			runtime.registerProvider("operations", {
				apiKey: "operations-key",
				models: [classifier],
				classifiers: {
					"hook-classifier": {
						classify: async (requestModel, _context, options) => {
							observed.push({
								payload: await options?.onPayload?.({ operation: "classifier" }, requestModel),
								options,
							});
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
			const session = await createSession((pi) => {
				pi.on("before_provider_headers", (event) => {
					calls.push("hook:headers");
					event.headers["x-hook"] = "yes";
				});
				pi.on("before_provider_request", (event) => {
					calls.push("hook:payload");
					return { wrapped: event.payload };
				});
				pi.on("after_provider_response", () => {
					calls.push("hook:response");
				});
				pi.on("provider_stream_event", () => {
					calls.push("hook:event");
				});
			});
			const callerModels: ClassifierModel<string>[] = [];

			const result = await session.extensionRunner.createContext().modelRegistry.classify(
				classifier,
				{ state: {}, questions: {} },
				{
					transformHeaders: async () => {
						calls.push("caller:headers");
						return { "x-caller": "yes" };
					},
					onPayload: async (payload, requestModel) => {
						calls.push("caller:payload");
						callerModels.push(requestModel);
						return { caller: payload };
					},
					onResponse: async (_response, requestModel) => {
						calls.push("caller:response");
						callerModels.push(requestModel);
					},
				},
			);

			expect(result.stopReason).toBe("stop");
			expect(callerModels).toEqual([classifier, classifier]);
			expect(observed).toHaveLength(1);
			expect(observed[0].payload).toEqual({ wrapped: { caller: { operation: "classifier" } } });
			expect(observed[0].options?.headers).toEqual({ "x-caller": "yes", "x-hook": "yes" });
			expect(observed[0].options).not.toHaveProperty("onProviderStreamEvent");
			expect(calls).toEqual([
				"caller:headers",
				"hook:headers",
				"caller:payload",
				"hook:payload",
				"caller:response",
				"hook:response",
			]);
		});
	});
});
