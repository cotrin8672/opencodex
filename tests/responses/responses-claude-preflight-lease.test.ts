// The Claude usage preflight runs before a claude-cli turn is dispatched. When the request also holds
// the web-search sidecar's cooldown-recovery probe lease, every exit of that preflight must hand the
// lease back: the outer cleanup does not own it, so a leaked lease blocks that account's probe.
import { afterEach, beforeEach, expect, mock, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ProviderAdapter } from "../../src/adapters/base";
import type { AdapterEvent, OcxConfig, OcxProviderConfig } from "../../src/types";
import { setClaudeUsagePreflightForTests } from "../../src/adapters/claude-cli/usage-admission";
import { isNonReplayableResponse } from "../../src/lib/upstream-retry";
import type { RequestLogContext } from "../../src/server/request-log";
import { acquireOwnedSpendHome } from "../helpers/owned-spend-home";
import { removeTreeWithRetry } from "../helpers/remove-tree";

let turns = 0;
let failure: Extract<AdapterEvent, { type: "error" }> | undefined;
let requestLog: RequestLogContext;
function claudeFixture(provider: OcxProviderConfig): ProviderAdapter {
  return {
    name: "claude-cli",
    buildRequest: () => ({ url: provider.baseUrl, method: "POST", headers: {}, body: "" }),
    async *parseStream() { yield { type: "done" } as AdapterEvent; },
    async runTurn(_parsed, _incoming, emit) {
      turns++;
      if (failure) { emit(failure); return; }
      emit({ type: "text_delta", text: "ok" } as AdapterEvent);
      emit({ type: "done" } as AdapterEvent);
    },
  };
}
const resolver = await import("../../src/server/adapter-resolve");
const resolveAdapter = resolver.resolveAdapter;
mock.module("../../src/server/adapter-resolve", () => ({ ...resolver,
  resolveAdapter: (provider: OcxProviderConfig, cache?: "none" | "short" | "long") =>
    provider.adapter === "claude-cli" ? claudeFixture(provider) : resolveAdapter(provider, cache),
}));
let released = 0;
const sidecarAuth = await import("../../src/server/responses/request-sidecar-auth");
mock.module("../../src/server/responses/request-sidecar-auth", () => ({ ...sidecarAuth,
  prepareResponsesSidecarAuth: async () => ({
    routedCompaction: false,
    openAiSidecar: { releaseProbeLease: () => { released++; } },
  }) as Awaited<ReturnType<typeof sidecarAuth.prepareResponsesSidecarAuth>>,
}));
const { handleResponses } = await import("../../src/server/responses");

const originalHome = process.env.OPENCODEX_HOME;
let home = "";
let release: (() => void) | undefined;
beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), "ocx-claude-preflight-lease-"));
  process.env.OPENCODEX_HOME = home;
  release = acquireOwnedSpendHome();
  turns = 0;
  failure = undefined;
  released = 0;
});
afterEach(() => {
  setClaudeUsagePreflightForTests();
  release?.();
  if (originalHome === undefined) delete process.env.OPENCODEX_HOME;
  else process.env.OPENCODEX_HOME = originalHome;
  removeTreeWithRetry(home);
});

async function send(stream = false, comboAttempt = false): Promise<Response> {
  const config = { port: 0, defaultProvider: "claude-cli", providers: {
    "claude-cli": { adapter: "claude-cli", baseUrl: "https://api.anthropic.com", authMode: "key", selectedModels: ["claude-sonnet-5-5"] },
  } } as OcxConfig;
  requestLog = { model: "", provider: "" };
  return handleResponses(new Request("http://localhost/v1/responses", {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: "claude-cli/claude-sonnet-5-5", input: "hello", stream }),
  }), config, requestLog, { comboAttempt });
}

test("a preflight that throws hands the probe lease back and starts no turn", async () => {
  setClaudeUsagePreflightForTests(async () => { throw new Error("quota cache unreadable"); });
  // The error itself keeps its existing path (it is rethrown); only the lease is settled first.
  await expect(send()).rejects.toThrow("quota cache unreadable");
  expect(released).toBeGreaterThanOrEqual(1);
  expect(turns).toBe(0);
});

test.each([true, false])("an exhausted preflight releases the lease and returns a non-replayable quota refusal (stream=%s)", async stream => {
  const resetAt = Date.now() + 60_001;
  setClaudeUsagePreflightForTests(async () => ({ state: "exhausted", checkedAt: Date.now(), resetAt, message: "exhausted" }));
  const res = await send(stream);
  expect(res.status).toBe(429);
  expect(isNonReplayableResponse(res)).toBe(true);
  expect(await res.json()).toEqual({ error: {
    type: "usage_limit_reached", code: "usage_limit_reached", message: "exhausted", resets_at: Math.ceil(resetAt / 1000),
  } });
  expect(released).toBeGreaterThanOrEqual(1);
  expect(turns).toBe(0);
});

test("an exhausted preflight with no known reset invents neither a reset nor a retry delay", async () => {
  setClaudeUsagePreflightForTests(async () => ({ state: "exhausted", checkedAt: Date.now() }));
  const res = await send();
  expect(res.status).toBe(429);
  expect(res.headers.get("retry-after")).toBeNull();
  expect(await res.json()).toEqual({ error: {
    type: "usage_limit_reached", code: "usage_limit_reached", message: "Claude subscription usage is exhausted.",
  } });
  expect(isNonReplayableResponse(res)).toBe(true);
  expect(released).toBeGreaterThanOrEqual(1);
  expect(turns).toBe(0);
});

test("an available preflight dispatches the turn", async () => {
  setClaudeUsagePreflightForTests(async () => ({ state: "available", checkedAt: Date.now() }));
  const res = await send();
  await res.text();
  expect(res.status).toBe(200);
  expect(turns).toBe(1);
});

test.each([true, false].flatMap(stream => [true, false].map(resetKnown => ({ stream, resetKnown }))))(
  "a first adapter quota failure retains its HTTP quota verdict (stream=$stream, resetKnown=$resetKnown)", async ({ stream, resetKnown }) => {
    setClaudeUsagePreflightForTests(async () => ({ state: "available", checkedAt: Date.now() }));
    const resetAt = resetKnown ? Date.now() + 90_001 : undefined;
    failure = { type: "error", status: 429, errorType: "usage_limit_reached", code: "usage_limit_reached",
      message: "This turn is stopped.", retryable: false, usageLimitResetAt: resetAt };
    const server = Bun.serve({ port: 0, hostname: "127.0.0.1", fetch: () => send(stream, true) });
    try {
      const res = await fetch(`http://127.0.0.1:${server.port}/v1/responses`, { method: "POST" });
      expect(res.status).toBe(429);
      expect(await res.json()).toEqual({ error: { type: "usage_limit_reached", code: "usage_limit_reached",
        message: "This turn is stopped.", ...(resetAt !== undefined ? { resets_at: Math.ceil(resetAt / 1000) } : {}) } });
      if (resetKnown) {
        expect(Number(res.headers.get("retry-after"))).toBeGreaterThan(60);
        expect(Number(res.headers.get("retry-after"))).toBeLessThanOrEqual(91);
      } else expect(res.headers.get("retry-after")).toBeNull();
      expect(requestLog.errorCode).toBe("usage_limit_reached");
      expect(turns).toBe(1);
      expect(released).toBeGreaterThanOrEqual(1);
    } finally { await server.stop(true); }
  },
);

test.each([true, false])("an unconfirmed first adapter 429 stays a rate limit (stream=%s)", async stream => {
  setClaudeUsagePreflightForTests(async () => ({ state: "available", checkedAt: Date.now() }));
  failure = { type: "error", status: 429, errorType: "rate_limit_error", code: "rate_limit_exceeded",
    message: "Too many requests", retryable: true };
  const res = await send(stream, true);
  expect(res.status).toBe(429);
  expect((await res.json() as { error: unknown }).error).toMatchObject({ type: "rate_limit_error", code: "rate_limit_exceeded" });
  expect(turns).toBe(1);
});

test.each([true, false])("a quota-shaped non-429 error keeps its original status (stream=%s)", async stream => {
  setClaudeUsagePreflightForTests(async () => ({ state: "available", checkedAt: Date.now() }));
  failure = { type: "error", status: 503, errorType: "usage_limit_reached", code: "usage_limit_reached",
    message: "Upstream unavailable", usageLimitResetAt: Date.now() + 90_000 };
  const res = await send(stream, true);
  expect(res.status).toBe(503);
  expect((await res.json() as { error: unknown }).error).not.toHaveProperty("resets_at");
  expect(turns).toBe(1);
});
