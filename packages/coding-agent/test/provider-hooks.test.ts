import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	type AnyModel,
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
import type { ExtensionFactory } from "../src/core/extensions/types.ts";
import type { ModelRuntime } from "../src/core/model-runtime.ts";
import { DefaultResourceLoader } from "../src/core/resource-loader.ts";
import { createAgentSession } from "../src/core/sdk.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { type Settings, SettingsManager } from "../src/core/settings-manager.ts";
import { createInMemoryModelRegistry, getModelRuntime } from "./model-runtime-test-utils.ts";

const model: Model<Api> = {
	id: "hook-model",
	name: "Hook Model",
	api: "hook-api",
	provider: "hook-provider",
	baseUrl: "https://hooks.invalid",
	reasoning: false,
	input: ["text"],
	cost: { input: 10, output: 50, cacheRead: 0.25, cacheWrite: 12.5 },
	contextWindow: 1_000_000,
	maxTokens: 4096,
	promptCache: { short: 300 },
};

function message(): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text: "summary" }],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: {
			input: 100_000,
			output: 1,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 100_001,
			cost: { input: 1, output: 0, cacheRead: 0, cacheWrite: 0, total: 1 },
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
		vi.useRealTimers();
		rmSync(tempDir, { recursive: true, force: true });
	});

	async function createSession(factory: ExtensionFactory, settings: Partial<Settings> = {}) {
		const cwd = join(tempDir, `session-${sessions.length}`);
		mkdirSync(cwd);
		const settingsManager = SettingsManager.inMemory({
			compaction: { enabled: false },
			retry: { enabled: false },
			...settings,
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
			expect(sent[0].options?.apiKey).toBe("hook-key");
			expect(sent[0].options?.timeoutMs).toBeUndefined();
			expect(options).not.toHaveProperty("headers");
		},
	);

	it("delivers all hooks once for normal turns and default summaries", async () => {
		const calls: string[] = [];
		const session = await createSession(
			(pi) => {
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
			},
			{ compaction: { enabled: false, keepRecentTokens: 1, reserveTokens: 1024 } },
		);

		await session.prompt("first turn");
		expect(calls).toEqual(["headers", "payload", "response", "event", "event"]);
		const targetId = session.sessionManager
			.getEntries()
			.find((entry) => entry.type === "message" && entry.message.role === "assistant")?.id;
		expect(targetId).toBeDefined();
		await session.prompt("second turn");
		calls.length = 0;
		await session.navigateTree(targetId!, { summarize: true });
		expect(calls).toEqual(["headers", "payload", "response", "event", "event"]);
		await session.prompt("third turn");
		calls.length = 0;
		const beforeCompaction = sent.length;
		await session.compact();
		// keepRecentTokens splits the last turn: one checkpoint and one turn-prefix summary.
		expect(sent.length - beforeCompaction).toBe(2);
		expect(calls).toEqual([
			"headers",
			"payload",
			"response",
			"event",
			"event",
			"headers",
			"payload",
			"response",
			"event",
			"event",
		]);
		expect(sent.slice(beforeCompaction).map((request) => request.options?.cacheRetention)).toEqual(["none", "none"]);
	});

	it("composes explicit Agent callbacks rather than suppressing session hooks", async () => {
		const calls: string[] = [];
		const session = await createSession((pi) => {
			pi.on("before_provider_request", (event) => {
				calls.push(`hook:${JSON.stringify(event.payload)}`);
				return false;
			});
			pi.on("after_provider_response", () => {
				calls.push("hook:response");
			});
			pi.on("provider_stream_event", () => {
				calls.push("hook:event");
			});
		});
		session.agent.onPayload = async () => {
			calls.push("caller:payload");
			return null;
		};
		session.agent.onResponse = async () => {
			calls.push("caller:response");
		};
		session.agent.onProviderStreamEvent = async () => {
			calls.push("caller:event");
		};

		await session.prompt("turn");

		expect(calls).toEqual([
			"caller:payload",
			"hook:null",
			"caller:response",
			"hook:response",
			"caller:event",
			"hook:event",
			"caller:event",
			"hook:event",
		]);
		expect(sent[0].payload).toBe(false);
	});

	it.each([null, false, 0, "", undefined])("preserves caller payload replacement %s", async (replacement) => {
		const observed: unknown[] = [];
		const session = await createSession((pi) => {
			pi.on("before_provider_request", (event) => {
				observed.push(event.payload);
			});
		});
		const registry = session.extensionRunner.createContext().modelRegistry;

		await registry.complete(
			model,
			{ messages: [] },
			{
				onPayload: async (payload) => {
					(payload as Record<string, unknown>).mutated = true;
					return replacement;
				},
			},
		);

		const expected = replacement === undefined ? { original: true, mutated: true } : replacement;
		expect(observed).toEqual([expected]);
		expect(sent[0].payload).toEqual(expected);
	});

	it.each([
		["transformHeaders", []],
		["onPayload", ["headers"]],
		["onResponse", ["headers", "payload"]],
		["onProviderStreamEvent", ["headers", "payload", "response"]],
	] as const)("preserves fail-fast caller errors in %s", async (callback, expectedHooks) => {
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

		const response = await session.extensionRunner.createContext().modelRegistry.complete(
			model,
			{ messages: [] },
			{
				[callback]: async () => {
					throw new Error("caller failed");
				},
			},
		);

		expect(response.stopReason).toBe("error");
		expect(response.errorMessage).toContain("caller failed");
		expect(calls).toEqual(expectedHooks);
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

		const result = await registry
			.streamSimple(virtual, { messages: [] }, { headers: { "x-virtual": "secret" } })
			.result();

		expect(result.stopReason).toBe("stop");
		expect(calls).toEqual(["headers", "hook-model", "hook-model"]);
		expect(sent).toHaveLength(1);
		expect(sent[0].options?.headers).not.toHaveProperty("x-virtual");
		const unrouted = await registry.complete(virtual, { messages: [] });
		expect(unrouted.stopReason).toBe("error");
		expect(sent).toHaveLength(1);
	});

	it.each(["registry", "sdk"] as const)(
		"suppresses later %s hook emissions after reload without retargeting",
		async (path) => {
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
			const stream =
				path === "registry"
					? registry.streamSimple(model, { messages: [] }, options)
					: await session.agent.streamFunction(model, normalizeContext({ messages: [] }), options);
			await entered.promise;
			await session.reload();
			expect(() => registry.streamSimple(model, { messages: [] })).toThrow("stale");
			expect(session.extensionRunner.createContext().modelRegistry).not.toBe(registry);
			release.resolve();
			expect((await stream.result()).stopReason).toBe("stop");
			expect(calls).toEqual(["1:payload", "caller:response"]);

			await session.extensionRunner.createContext().modelRegistry.complete(model, { messages: [] });
			expect(calls).toEqual(["1:payload", "caller:response", "2:payload", "2:response", "2:event", "2:event"]);
		},
	);

	it("suppresses disposed request hooks but still invokes caller observers", async () => {
		const calls: string[] = [];
		const session = await createSession((pi) => {
			pi.on("before_provider_request", () => {
				calls.push("hook:payload");
			});
			pi.on("after_provider_response", () => {
				calls.push("hook:response");
			});
			pi.on("provider_stream_event", () => {
				calls.push("hook:event");
			});
		});
		const entered = Promise.withResolvers<void>();
		const release = Promise.withResolvers<void>();
		const registry = session.extensionRunner.createContext().modelRegistry;
		const response = registry.complete(
			model,
			{ messages: [] },
			{
				onResponse: async () => {
					entered.resolve();
					await release.promise;
					calls.push("caller:response");
				},
				onProviderStreamEvent: async () => {
					calls.push("caller:event");
				},
			},
		);
		await entered.promise;
		session.dispose();
		release.resolve();

		expect((await response).stopReason).toBe("stop");
		expect(calls).toEqual(["hook:payload", "caller:response", "caller:event", "caller:event"]);
		expect(() => registry.complete(model, { messages: [] })).toThrow("stale");
	});

	it("keeps captured registries revoked when reload fails", async () => {
		const session = await createSession(() => {});
		const registry = session.extensionRunner.createContext().modelRegistry;
		const reload = vi.spyOn(session.resourceLoader, "reload").mockRejectedValueOnce(new Error("load failed"));

		await expect(session.reload()).rejects.toThrow("load failed");

		expect(() => registry.complete(model, { messages: [] })).toThrow("stale");
		await expect(session.agent.streamFunction(model, normalizeContext({ messages: [] }))).rejects.toThrow("stale");
		expect(sent).toEqual([]);
		reload.mockRestore();
		await session.reload();
		expect(
			(await session.extensionRunner.createContext().modelRegistry.complete(model, { messages: [] })).stopReason,
		).toBe("stop");
	});

	it("rejects execution invalidated while a caller header transform awaits", async () => {
		const hooks: string[] = [];
		const session = await createSession((pi) => {
			pi.on("before_provider_headers", () => {
				hooks.push("headers");
			});
		});
		const entered = Promise.withResolvers<void>();
		const release = Promise.withResolvers<void>();
		const registry = session.extensionRunner.createContext().modelRegistry;
		const response = registry.complete(
			model,
			{ messages: [] },
			{
				transformHeaders: async (headers) => {
					entered.resolve();
					await release.promise;
					return headers;
				},
			},
		);
		await entered.promise;
		session.dispose();
		release.resolve();

		expect((await response).errorMessage).toContain("stale");
		expect(sent).toEqual([]);
		expect(hooks).toEqual([]);
		expect(() => registry.stream(model, { messages: [] })).toThrow("stale");
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

	it("binds existing image and classifier execution without parsed-event callbacks", async () => {
		const observed: Array<{ payload: unknown; model: AnyModel; options?: ImagesOptions | ClassifierOptions }> = [];
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
			models: [image, classifier],
			images: {
				"hook-images": {
					generateImages: async (requestModel, _context, options) => {
						observed.push({
							model: requestModel,
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
			classifiers: {
				"hook-classifier": {
					classify: async (requestModel, _context, options) => {
						observed.push({
							model: requestModel,
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
				calls.push("headers");
				event.headers["x-hook"] = "yes";
			});
			pi.on("before_provider_request", (event) => {
				calls.push("payload");
				return { wrapped: event.payload };
			});
			pi.on("after_provider_response", () => {
				calls.push("response");
			});
			pi.on("provider_stream_event", () => {
				calls.push("event");
			});
		});
		const registry = session.extensionRunner.createContext().modelRegistry;
		const callerModels: AnyModel[] = [];

		const images = await registry.generateImages(
			image,
			{ input: [{ type: "text", text: "circle" }] },
			{
				onPayload: async (payload, requestModel) => {
					callerModels.push(requestModel);
					return payload;
				},
				onResponse: async (_response, requestModel) => {
					callerModels.push(requestModel);
				},
			},
		);
		const answers = await registry.classify(
			classifier,
			{ state: {}, questions: {} },
			{
				onPayload: async (payload, requestModel) => {
					callerModels.push(requestModel);
					return payload;
				},
				onResponse: async (_response, requestModel) => {
					callerModels.push(requestModel);
				},
			},
		);

		expect(images.stopReason).toBe("stop");
		expect(answers.stopReason).toBe("stop");
		expect(callerModels.map((entry) => entry.type)).toEqual(["image", "image", "classifier", "classifier"]);
		expect(observed.map((entry) => entry.payload)).toEqual([
			{ wrapped: { operation: "image" } },
			{ wrapped: { operation: "classifier" } },
		]);
		expect(observed.map((entry) => entry.options?.headers)).toEqual([{ "x-hook": "yes" }, { "x-hook": "yes" }]);
		for (const entry of observed) expect(entry.options).not.toHaveProperty("onProviderStreamEvent");
		expect(calls).toEqual(["headers", "payload", "response", "headers", "payload", "response"]);
	});

	it("delivers provider hooks through automatic threshold compaction", async () => {
		const calls: string[] = [];
		const session = await createSession(
			(pi) => {
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
			},
			{ compaction: { enabled: true, keepRecentTokens: 1, reserveTokens: 990_000 } },
		);

		await session.prompt("turn");

		expect(session.sessionManager.getEntries().filter((entry) => entry.type === "compaction")).toHaveLength(1);
		// With no earlier turns, only the split-turn prefix needs summarizing.
		expect(sent).toHaveLength(2);
		expect(calls).toEqual([
			"headers",
			"payload",
			"response",
			"event",
			"event",
			"headers",
			"payload",
			"response",
			"event",
			"event",
		]);
	});

	it("replays warming hooks once and cancels warming on reload", async () => {
		vi.useFakeTimers();
		const calls: string[] = [];
		const session = await createSession(
			(pi) => {
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
			},
			{ cacheWarming: "idle" },
		);
		await session.prompt("turn");
		calls.length = 0;
		await vi.advanceTimersByTimeAsync(270_000);
		await vi.advanceTimersByTimeAsync(270_000);

		expect(sent).toHaveLength(3);
		expect(sent.slice(1).map((request) => [request.options?.maxTokens, request.options?.maxRetries])).toEqual([
			[1, 0],
			[1, 0],
		]);
		expect(calls).toEqual([
			"headers",
			"payload",
			"response",
			"event",
			"event",
			"headers",
			"payload",
			"response",
			"event",
			"event",
		]);
		await session.reload();
		await vi.advanceTimersByTimeAsync(270_000);
		expect(sent).toHaveLength(3);
		expect(session.cacheWarmingStatus?.state).toBe("inactive");
	});
});
