import type { Api, Model, ModelsSimpleStreamOptions } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { createEventBus } from "../src/core/event-bus.ts";
import { createExtensionRuntime, loadExtensionFromFactory } from "../src/core/extensions/loader.ts";
import { ExtensionRunner } from "../src/core/extensions/runner.ts";
import type { ExtensionFactory } from "../src/core/extensions/types.ts";
import { composeProviderHooks } from "../src/core/provider-hooks.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { createInMemoryModelRegistry } from "./model-runtime-test-utils.ts";

const model: Model<Api> = {
	id: "hook-model",
	name: "Hook Model",
	api: "hook-api",
	provider: "hook-provider",
	baseUrl: "https://hooks.invalid",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 1000,
	maxTokens: 100,
};

async function createRunner(factory: ExtensionFactory): Promise<ExtensionRunner> {
	const cwd = process.cwd();
	const runtime = createExtensionRuntime();
	const extension = await loadExtensionFromFactory(factory, cwd, createEventBus(), runtime);
	const registry = await createInMemoryModelRegistry(AuthStorage.inMemory());
	return new ExtensionRunner([extension], runtime, cwd, SessionManager.inMemory(cwd), registry);
}

const callbackCases = [
	{
		callback: "transformHeaders",
		invoke: (options: ModelsSimpleStreamOptions) => options.transformHeaders!({}),
	},
	{
		callback: "onPayload",
		invoke: (options: ModelsSimpleStreamOptions) => options.onPayload!({}, model),
	},
	{
		callback: "onResponse",
		invoke: (options: ModelsSimpleStreamOptions) => options.onResponse!({ status: 200, headers: {} }, model),
	},
	{
		callback: "onProviderStreamEvent",
		invoke: (options: ModelsSimpleStreamOptions) => options.onProviderStreamEvent!({ index: 1 }, model),
	},
];

// #9773: exercise composition with local inputs and a real runner, without SDK sessions or provider dispatch.
describe("provider hook composition", () => {
	it("passes replacement headers to the hook without modifying caller options", async () => {
		const original = { "x-original": "yes" };
		const replacement = { "x-caller": "yes" };
		const observed: unknown[] = [];
		const runner = await createRunner((pi) => {
			pi.on("before_provider_headers", (event) => {
				observed.push(event.headers);
				event.headers["x-hook"] = "yes";
			});
		});
		const callerOptions: ModelsSimpleStreamOptions = Object.freeze({
			transformHeaders: async () => replacement,
		});
		const composed = composeProviderHooks(model, callerOptions, runner);

		const result = await composed.transformHeaders!(original);

		expect(result).toEqual({ "x-caller": "yes", "x-hook": "yes" });
		expect(observed).toEqual([replacement]);
		expect(observed[0]).toBe(replacement);
		expect(original).toEqual({ "x-original": "yes" });
		expect(composed).not.toBe(callerOptions);
		expect(callerOptions).not.toHaveProperty("onPayload");
	});

	it.each([null, false, 0, "", undefined])("preserves caller payload replacement %s", async (replacement) => {
		const payload = { original: true, mutated: false };
		const observed: unknown[] = [];
		const runner = await createRunner((pi) => {
			pi.on("before_provider_request", (event) => {
				observed.push(event.payload);
				return event.payload;
			});
		});
		const composed = composeProviderHooks<Model<Api>, ModelsSimpleStreamOptions>(
			model,
			{
				onPayload: async () => {
					payload.mutated = true;
					return replacement;
				},
			},
			runner,
		);

		const result = await composed.onPayload!(payload, model);

		const expected = replacement === undefined ? payload : replacement;
		expect(result).toBe(expected);
		expect(observed).toEqual([expected]);
		expect(payload.mutated).toBe(true);
	});

	it("returns the hook's payload replacement after the caller replacement", async () => {
		const observed: unknown[] = [];
		const runner = await createRunner((pi) => {
			pi.on("before_provider_request", (event) => {
				observed.push(event.payload);
				return { hook: true };
			});
		});
		const composed = composeProviderHooks<Model<Api>, ModelsSimpleStreamOptions>(
			model,
			{ onPayload: () => ({ caller: true }) },
			runner,
		);

		const result = await composed.onPayload!({ original: true }, model);

		expect(observed).toEqual([{ caller: true }]);
		expect(result).toEqual({ hook: true });
	});

	it.each(callbackCases)("awaits the caller before $callback hook delivery", async ({ invoke }) => {
		const calls: string[] = [];
		const entered = Promise.withResolvers<void>();
		const release = Promise.withResolvers<void>();
		const caller = async () => {
			calls.push("caller:entered");
			entered.resolve();
			await release.promise;
			calls.push("caller:finished");
		};
		const runner = await createRunner((pi) => {
			const handler = () => {
				calls.push("hook");
			};
			pi.on("before_provider_headers", handler);
			pi.on("before_provider_request", handler);
			pi.on("after_provider_response", handler);
			pi.on("provider_stream_event", handler);
		});
		const composed = composeProviderHooks<Model<Api>, ModelsSimpleStreamOptions>(
			model,
			{
				transformHeaders: async (headers) => {
					await caller();
					return headers;
				},
				onPayload: async (payload) => {
					await caller();
					return payload;
				},
				onResponse: caller,
				onProviderStreamEvent: caller,
			},
			runner,
		);

		const pending = invoke(composed);
		await entered.promise;
		expect(calls).toEqual(["caller:entered"]);
		release.resolve();
		await pending;

		expect(calls).toEqual(["caller:entered", "caller:finished", "hook"]);
	});

	it.each(callbackCases)("propagates caller failure before $callback hook delivery", async ({ invoke }) => {
		const calls: string[] = [];
		const failure = new Error("caller failed");
		const caller = async () => {
			throw failure;
		};
		const runner = await createRunner((pi) => {
			const handler = () => {
				calls.push("hook");
			};
			pi.on("before_provider_headers", handler);
			pi.on("before_provider_request", handler);
			pi.on("after_provider_response", handler);
			pi.on("provider_stream_event", handler);
		});
		const composed = composeProviderHooks<Model<Api>, ModelsSimpleStreamOptions>(
			model,
			{ transformHeaders: caller, onPayload: caller, onResponse: caller, onProviderStreamEvent: caller },
			runner,
		);

		await expect(invoke(composed)).rejects.toBe(failure);

		expect(calls).toEqual([]);
	});
});
