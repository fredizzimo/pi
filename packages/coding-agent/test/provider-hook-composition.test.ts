import type { Api, Model, ModelsSimpleStreamOptions } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import type { ExtensionAPI } from "../src/core/extensions/types.ts";
import { composeProviderHooks } from "../src/core/provider-hooks.ts";
import { createRunner, model } from "./provider-hook-test-utils.ts";

const callbackCases = [
	{
		callback: "transformHeaders",
		register: (pi: ExtensionAPI, handler: () => Promise<void>) => pi.on("before_provider_headers", handler),
		invoke: (options: ModelsSimpleStreamOptions) => options.transformHeaders!({}),
	},
	{
		callback: "onPayload",
		register: (pi: ExtensionAPI, handler: () => Promise<void>) => pi.on("before_provider_request", handler),
		invoke: (options: ModelsSimpleStreamOptions) => options.onPayload!({}, model),
	},
	{
		callback: "onResponse",
		register: (pi: ExtensionAPI, handler: () => Promise<void>) => pi.on("after_provider_response", handler),
		invoke: (options: ModelsSimpleStreamOptions) => options.onResponse!({ status: 200, headers: {} }, model),
	},
	{
		callback: "onProviderStreamEvent",
		register: (pi: ExtensionAPI, handler: () => Promise<void>) => pi.on("provider_stream_event", handler),
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

	it.each(callbackCases)("awaits the caller before $callback hook delivery", async ({ invoke, register }) => {
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
			register(pi, async () => {
				calls.push("hook");
			});
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

	it.each(callbackCases)("propagates caller failure before $callback hook delivery", async ({ invoke, register }) => {
		const calls: string[] = [];
		const failure = new Error("caller failed");
		const caller = async () => {
			throw failure;
		};
		const runner = await createRunner((pi) => {
			register(pi, async () => {
				calls.push("hook");
			});
		});
		const composed = composeProviderHooks<Model<Api>, ModelsSimpleStreamOptions>(
			model,
			{ transformHeaders: caller, onPayload: caller, onResponse: caller, onProviderStreamEvent: caller },
			runner,
		);

		await expect(invoke(composed)).rejects.toBe(failure);

		expect(calls).toEqual([]);
	});

	// #9773: callback lifetime is local to the captured runner, independent of provider sequencing.
	it.each(callbackCases)(
		"rejects $callback invalidated while its caller awaits",
		async ({ callback, invoke, register }) => {
			const calls: string[] = [];
			const entered = Promise.withResolvers<void>();
			const release = Promise.withResolvers<void>();
			const runner = await createRunner((pi) => {
				register(pi, async () => {
					calls.push("hook");
				});
			});
			const composed = composeProviderHooks<Model<Api>, ModelsSimpleStreamOptions>(
				model,
				{
					[callback]: async (value: unknown) => {
						entered.resolve();
						await release.promise;
						calls.push("caller:finished");
						return value;
					},
				},
				runner,
			);

			const pending = invoke(composed);
			await entered.promise;
			runner.invalidate();
			release.resolve();

			await expect(pending).rejects.toThrow("stale");
			expect(calls).toEqual(["caller:finished"]);
		},
	);

	it.each(callbackCases)("rejects stale $callback even without handlers", async ({ callback, invoke }) => {
		const runner = await createRunner(() => {});
		const composed = composeProviderHooks<Model<Api>, ModelsSimpleStreamOptions>(
			model,
			{
				[callback]: async (value: unknown) => {
					runner.invalidate();
					return value;
				},
			},
			runner,
		);

		await expect(invoke(composed)).rejects.toThrow("stale");
	});

	it.each(callbackCases)(
		"lets an entered $callback hook finish without a post-await guard",
		async ({ invoke, register }) => {
			const calls: string[] = [];
			const entered = Promise.withResolvers<void>();
			const release = Promise.withResolvers<void>();
			const runner = await createRunner((pi) => {
				register(pi, async () => {
					calls.push("hook:entered");
					entered.resolve();
					await release.promise;
					calls.push("hook:finished");
				});
			});
			const composed = composeProviderHooks<Model<Api>, ModelsSimpleStreamOptions>(model, {}, runner);

			const pending = invoke(composed);
			await entered.promise;
			runner.invalidate();
			release.resolve();

			await pending;
			expect(calls).toEqual(["hook:entered", "hook:finished"]);
			await expect(invoke(composed)).rejects.toThrow("stale");
			expect(calls).toEqual(["hook:entered", "hook:finished"]);
		},
	);

	it("retains runner error reporting and payload chaining", async () => {
		const runner = await createRunner((pi) => {
			pi.on("before_provider_request", (event) => {
				(event.payload as Record<string, unknown>).mutated = true;
				throw new Error("extension failed");
			});
			pi.on("before_provider_request", (event) => ({ final: event.payload }));
		});
		const errors: string[] = [];
		runner.onError((error) => errors.push(error.error));

		const result = await runner.emitBeforeProviderRequest({ original: true });

		expect(result).toEqual({ final: { original: true, mutated: true } });
		expect(errors).toEqual(["extension failed"]);
	});
});
