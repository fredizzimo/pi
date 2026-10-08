import type { Api, Model, ModelsSimpleStreamOptions } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import { composeProviderHooks } from "../src/core/provider-hooks.ts";

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

// #9773: detailed composition rules need only local callbacks, not an SDK session.
describe("provider hook composition", () => {
	it("passes replacement headers to the hook without modifying caller options", async () => {
		const original = { "x-original": "yes" };
		const replacement = { "x-caller": "yes" };
		const observed: unknown[] = [];
		const callerOptions: ModelsSimpleStreamOptions = Object.freeze({
			transformHeaders: async () => replacement,
		});
		const composed = composeProviderHooks(model, callerOptions, {
			transformHeaders: (headers) => {
				observed.push(headers);
				return { ...headers, "x-hook": "yes" };
			},
		});

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
		const composed = composeProviderHooks<Model<Api>, ModelsSimpleStreamOptions>(
			model,
			{
				onPayload: async () => {
					payload.mutated = true;
					return replacement;
				},
			},
			{
				onPayload: (value, requestModel) => {
					observed.push(value, requestModel);
					return value;
				},
			},
		);

		const result = await composed.onPayload!(payload, model);

		const expected = replacement === undefined ? payload : replacement;
		expect(result).toBe(expected);
		expect(observed).toEqual([expected, model]);
		expect(payload.mutated).toBe(true);
	});

	it("returns the hook's payload replacement after the caller replacement", async () => {
		const observed: unknown[] = [];
		const composed = composeProviderHooks<Model<Api>, ModelsSimpleStreamOptions>(
			model,
			{ onPayload: () => ({ caller: true }) },
			{
				onPayload: (payload) => {
					observed.push(payload);
					return { hook: true };
				},
			},
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
			{
				transformHeaders: (headers) => {
					calls.push("hook");
					return headers;
				},
				onPayload: (payload) => {
					calls.push("hook");
					return payload;
				},
				onResponse: () => {
					calls.push("hook");
				},
				onProviderStreamEvent: () => {
					calls.push("hook");
				},
			},
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
		const composed = composeProviderHooks<Model<Api>, ModelsSimpleStreamOptions>(
			model,
			{ transformHeaders: caller, onPayload: caller, onResponse: caller, onProviderStreamEvent: caller },
			{
				transformHeaders: (headers) => {
					calls.push("hook");
					return headers;
				},
				onPayload: (payload) => {
					calls.push("hook");
					return payload;
				},
				onResponse: () => {
					calls.push("hook");
				},
				onProviderStreamEvent: () => {
					calls.push("hook");
				},
			},
		);

		await expect(invoke(composed)).rejects.toBe(failure);

		expect(calls).toEqual([]);
	});
});
