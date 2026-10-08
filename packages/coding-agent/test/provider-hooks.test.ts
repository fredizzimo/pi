import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createAssistantMessageEventStream, fauxAssistantMessage, type Model } from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { AgentSession } from "../src/core/agent-session.ts";
import { AuthStorage } from "../src/core/auth-storage.ts";
import type { ExtensionFactory } from "../src/core/extensions/types.ts";
import { DefaultResourceLoader } from "../src/core/resource-loader.ts";
import { createAgentSession } from "../src/core/sdk.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import { createInMemoryModelRegistry, getModelRuntime } from "./model-runtime-test-utils.ts";

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

// #9773: SDK wiring evidence only; composition decisions are tested at the registry boundary.
describe("session provider hook binding", () => {
	let cwd: string;
	let session: AgentSession;

	beforeEach(() => {
		cwd = mkdtempSync(join(tmpdir(), "pi-provider-hooks-"));
	});
	afterEach(() => {
		session?.dispose();
		rmSync(cwd, { recursive: true, force: true });
	});

	async function createSession(factory: ExtensionFactory) {
		const registry = await createInMemoryModelRegistry(AuthStorage.inMemory());
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
						await options?.onPayload?.({}, requestModel);
						await options?.onResponse?.({ status: 200, headers: {} }, requestModel);
						await options?.onProviderStreamEvent?.({ index: 1 }, requestModel);
						stream.end(message);
					} catch (error) {
						stream.end({ ...message, stopReason: "error", errorMessage: String(error) });
					}
				})();
				return stream;
			},
		});
		const settingsManager = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } });
		const resourceLoader = new DefaultResourceLoader({
			cwd,
			agentDir: cwd,
			settingsManager,
			extensionFactories: [factory],
		});
		await resourceLoader.reload();
		({ session } = await createAgentSession({
			cwd,
			agentDir: cwd,
			model,
			modelRuntime: getModelRuntime(registry),
			settingsManager,
			sessionManager: SessionManager.inMemory(cwd),
			resourceLoader,
			noTools: "all",
		}));
	}

	it("delivers all four hooks once during an Agent turn", async () => {
		const calls: string[] = [];
		await createSession((pi) => {
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

		await session.prompt("hello");

		expect(calls).toEqual(["headers", "payload", "response", "event"]);
	});

	it("revokes old registry requests on reload without retargeting their callbacks", async () => {
		const calls: string[] = [];
		let generation = 0;
		await createSession((pi) => {
			const owner = ++generation;
			pi.on("after_provider_response", () => {
				calls.push(`${owner}:response`);
			});
		});
		const oldRegistry = session.extensionRunner.createContext().modelRegistry;
		const entered = Promise.withResolvers<void>();
		const release = Promise.withResolvers<void>();
		const pending = oldRegistry.complete(
			model,
			{ messages: [] },
			{
				onResponse: async () => {
					entered.resolve();
					await release.promise;
					calls.push("caller:finished");
				},
			},
		);

		await entered.promise;
		await session.reload();
		release.resolve();
		const result = await pending;

		expect(result.stopReason).toBe("error");
		expect(result.errorMessage).toContain("stale");
		expect(calls).toEqual(["caller:finished"]);
		expect(() => oldRegistry.complete(model, { messages: [] })).toThrow("stale");
		const newRegistry = session.extensionRunner.createContext().modelRegistry;
		expect(newRegistry).not.toBe(oldRegistry);
		expect((await newRegistry.complete(model, { messages: [] })).stopReason).toBe("stop");
		expect(calls).toEqual(["caller:finished", "2:response"]);
	});
});
