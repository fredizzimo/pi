import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	type Api,
	createAssistantMessageEventStream,
	type Model,
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
import { SettingsManager } from "../src/core/settings-manager.ts";
import { createInMemoryModelRegistry, getModelRuntime } from "./model-runtime-test-utils.ts";
import { message, model } from "./provider-hook-test-utils.ts";

// #9773: SDK integration covers session ownership and lifecycle, not callback-local composition rules.
// Only the provider boundary is synthetic; no endpoint or paid request is used.
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
			models: [model],
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

	describe("Agent and session ownership", () => {
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

	describe("session lifetime", () => {
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
			const stream = registry.streamSimple(
				model,
				{ messages: [] },
				{
					onResponse: async () => {
						entered.resolve();
						await release.promise;
						calls.push("caller:response");
					},
				},
			);
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

		// #9773: retain one real-session disposal race; the four-callback contract is tested locally.
		it("revokes captured registry admission and pending callbacks on disposal", async () => {
			const calls: string[] = [];
			const session = await createSession((pi) => {
				pi.on("before_provider_headers", () => {
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
					transformHeaders: async (headers) => {
						entered.resolve();
						await release.promise;
						calls.push("caller:finished");
						return headers;
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
			expect(sent).toEqual([]);
			expect(() => registry.complete(model, { messages: [] })).toThrow("stale");
			expect(() => registry.stream(model, { messages: [] })).toThrow("stale");
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
			const result = session.extensionRunner.createContext().modelRegistry.complete(model, { messages: [] });
			await entered.promise;
			session.dispose();
			release.resolve();

			expect((await result).errorMessage).toContain("stale");
			expect(sent).toEqual([]);
			auth.mockRestore();
		});
	});
});
