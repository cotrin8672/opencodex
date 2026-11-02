import { afterEach, describe, expect, test } from "bun:test";
import type { ChildProcess } from "node:child_process";
import { EventEmitter } from "node:events";
import { Readable, Writable } from "node:stream";
import { createClaudeCliAdapter, withClaudeLoginHint } from "../../src/adapters/claude-cli/adapter";
import { clearClaudeCliBinaryCache } from "../../src/adapters/claude-cli/profiles";
import { setClaudeUsagePreflightForTests } from "../../src/adapters/claude-cli/usage-admission";
import type { ProviderAdapter } from "../../src/adapters/base";
import { getAdapterDefinition } from "../../src/adapters/registry";
import { bridgeToResponsesSSE, buildResponseJSON } from "../../src/bridge";
import { adapterFailureFromEvent } from "../../src/bridge/internal";
import { chatCompletionsErrorBody, chatCompletionsErrorResponse, chatCompletionsStreamErrorPayload, responsesSseToChatCompletionsSse } from "../../src/chat/outbound";
import { createTranslatorBudget } from "../../src/lib/translator-budget";
import { encodeChatCompletionSse, foldChatCompletion } from "../../src/protocols/encoders/chat";
import { handleResponses } from "../../src/server/responses";
import { handleChatCompletions } from "../../src/server/chat-completions";
import { resolveResponsesApiAuth } from "../../src/server/auth-cors";
import type { AdapterEvent, OcxConfig } from "../../src/types";
import { nativeBinary, nativeSession } from "../fixtures/claude-session-native-api";
import { turn } from "../fixtures/claude-session-worker";
import { collectSse, replay } from "../helpers/responses-conformance";
import { acquireOwnedSpendHome } from "../helpers/owned-spend-home";

const provider = { adapter: "claude-cli", baseUrl: "https://api.anthropic.com" };
const available = async () => ({ state: "available" as const, checkedAt: 0 });
const limits = ["You've hit your 5-hour usage limit · resets 3:30pm (UTC)",
  "You've hit your weekly usage limit · resets 3:30pm (UTC)"];
const quotaType = "usage_limit_reached";
type Failure = Extract<AdapterEvent, { type: "error" }>;

afterEach(clearClaudeCliBinaryCache);

async function coreHttp(adapter: ProviderAdapter, resident: boolean, wire = "responses", stream = true) {
  const definition = getAdapterDefinition("claude-cli")!;
  const mutable = definition as { create: typeof definition.create };
  const original = mutable.create;
  const release = acquireOwnedSpendHome();
  const key = "ocx_fixture_quota_wire_only";
  const config = { port: 0, defaultProvider: "claude-cli",
    apiKeys: [{ id: "quota-wire", name: "Offline quota regression", key, createdAt: "2026-11-02T00:00:00Z" }], providers: { "claude-cli": {
    ...provider, authMode: "key", selectedModels: ["claude-opus-4-6"],
  } } } as OcxConfig;
  setClaudeUsagePreflightForTests(available);
  mutable.create = () => adapter;
  const server = Bun.serve({ hostname: "127.0.0.1", port: 0,
    fetch: req => wire === "chat/completions" ? handleChatCompletions(req, config, { model: "", provider: "" }, {
      requestId: crypto.randomUUID(), start: Date.now(), admission: resolveResponsesApiAuth(req, config)!,
    }) : handleResponses(req, config, { model: "", provider: "" }, {
      admission: resolveResponsesApiAuth(req, config)!, comboAttempt: true,
    }) });
  try {
    const response = await fetch(`http://127.0.0.1:${server.port}/v1/${wire}`, {
      method: "POST", headers: { "content-type": "application/json", "x-opencodex-api-key": key,
        ...(resident ? { "thread-id": "quota-wire-native", "idempotency-key": crypto.randomUUID() } : {}) },
      body: JSON.stringify({ model: "claude-cli/claude-opus-4-6", stream, store: false,
        ...(wire === "chat/completions" ? { messages: [{ role: "user", content: "Offline quota regression" }] } : { input: "Offline quota regression" }),
        ...(resident ? { tools: [{ type: "function", name: "read_file", parameters: { type: "object", properties: {} } }] } : {}) }),
    });
    return { status: response.status, contentType: response.headers.get("content-type"),
      retryAfter: response.headers.get("retry-after"), text: await response.text() };
  } finally { await server.stop(true); mutable.create = original; setClaudeUsagePreflightForTests(); release(); }
}

// Compute the expected next UTC clock independently of the production reset parser.
function nextReset(now: number) {
  const date = new Date(now);
  const reset = Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate(), 15, 30);
  return reset > now ? reset : reset + 86_400_000;
}

function child(lines: unknown[]): ChildProcess {
  const proc = new EventEmitter() as ChildProcess;
  proc.stdout = Readable.from(lines.map(line => Buffer.from(JSON.stringify(line) + "\n")));
  proc.stderr = Readable.from([]);
  proc.stdin = new Writable({ write(_chunk, _encoding, done) { done(); } });
  proc.kill = () => true;
  proc.exitCode = null;
  setTimeout(() => { proc.exitCode = 1; proc.emit("close", 1); }, 10);
  return proc;
}

function quotaError(error: unknown, reset: number) {
  expect(error).toMatchObject({ type: quotaType, code: quotaType, resets_at: reset / 1000 });
  expect(error).not.toHaveProperty("plan_type");
}

async function responsesContract(events: AdapterEvent[], reset: number) {
  const failure = events.find((event): event is Failure => event.type === "error");
  expect(failure).toMatchObject({ status: 429, errorType: quotaType, code: quotaType,
    retryable: false, usageLimitResetAt: reset });
  expect(events.filter(event => event.type === "error")).toHaveLength(1);
  expect(events.some(event => event.type === "done")).toBe(false);
  const buffered = buildResponseJSON(structuredClone(events), "offline-model");
  expect(buffered).toMatchObject({ status: "failed", retryable: false });
  quotaError(buffered.error, reset);
  quotaError(buffered.last_error, reset);
  const frames = await collectSse(bridgeToResponsesSSE(replay(structuredClone(events)), "offline-model"));
  expect(frames[0]?.event).toBe("response.created");
  expect(frames.filter(frame => frame.event === "response.failed")).toHaveLength(1);
  expect(frames.some(frame => frame.event === "response.completed")).toBe(false);
  const streamed = frames.find(frame => frame.event === "response.failed")!.data.response as Record<string, unknown>;
  expect(streamed).toMatchObject({ status: "failed", retryable: false });
  quotaError(streamed.error, reset);
  quotaError(streamed.last_error, reset);
  if (failure?.usage) {
    expect(streamed.usage).toEqual(buffered.usage);
    expect(buffered.usage).toMatchObject({ input_tokens: failure.usage.inputTokens, output_tokens: failure.usage.outputTokens });
  }
}

test.each(limits)("one-shot CLI exhaustion stops retries and survives Responses wires: %s", async message => {
  const input = turn();
  input.parsed.context.tools = [];
  let launches = 0;
  const adapter = createClaudeCliAdapter(provider, {
    which: () => process.execPath, usageAdmission: available,
    usageObserver: () => ({ observe() {}, finish() {} }), usageRefusal: async () => {},
    spawn: () => { launches++; return child([
      { type: "stream_event", event: { type: "content_block_delta", delta: { type: "text_delta", text: "partial answer" } } },
      { type: "result", subtype: "error_during_execution", is_error: true, errors: [message],
        result: `Rate limit reached: ${message}`, usage: { input_tokens: 7, output_tokens: 2, cache_read_input_tokens: 11 } },
    ]); }, killGraceMs: 10, reapTimeoutMs: 100,
  });
  const events: AdapterEvent[] = [];
  const reset = nextReset(Date.now());
  await adapter.runTurn!(input.parsed, { ...input.incoming, headers: new Headers() }, event => events.push(event));
  expect(launches).toBe(1);
  expect(events).toContainEqual({ type: "text_delta", text: "partial answer" });
  expect(events.find(event => event.type === "error")?.usage).toMatchObject({ inputTokens: 18, outputTokens: 2 });
  await responsesContract(events, reset);
  const http = await coreHttp(adapter, false);
  expect(http.status).toBe(200);
  expect(http.contentType).toContain("text/event-stream");
  const frames = await collectSse(new Response(http.text).body!);
  expect(frames.some(frame => frame.event === "response.output_text.delta" && frame.data.delta === "partial answer")).toBe(true);
  const failed = frames.find(frame => frame.event === "response.failed")!.data.response as Record<string, unknown>;
  expect(failed).toMatchObject({ status: "failed", retryable: false, usage: { input_tokens: 18, output_tokens: 2 } });
  quotaError(failed.error, reset);
  expect(launches).toBe(2);
});

test.each(["SSE", "JSON", "Responses SSE"])("Chat %s preserves the typed quota error and reset time", async wire => {
  const reset = 2_000_000_000_000;
  const events: AdapterEvent[] = [{ type: "text_delta", text: "partial answer" },
    { type: "error", status: 429, errorType: quotaType, code: quotaType, message: "Subscription exhausted",
      retryable: false, usageLimitResetAt: reset, usage: { inputTokens: 18, outputTokens: 2, totalTokens: 20 } }];
  const budget = createTranslatorBudget();
  try {
    if (wire !== "JSON") {
      const stream = wire === "SSE" ? encodeChatCompletionSse(replay(events), { model: "offline-model", translatorBudget: budget })
        : responsesSseToChatCompletionsSse(bridgeToResponsesSSE(replay(events), "offline-model"), "offline-model", { translatorBudget: budget });
      const frames = await collectSse(stream);
      expect(frames.some(frame => JSON.stringify(frame.data).includes("partial answer"))).toBe(true);
      quotaError(frames.find(frame => frame.data.error)?.data.error, reset);
      expect(frames.find(frame => frame.data.error)?.data.error).toMatchObject({ retryable: false });
    } else {
      const response = await foldChatCompletion(replay(events), { model: "offline-model", translatorBudget: budget });
      expect(response.status).toBe(429);
      const body = await response.json() as { error: unknown };
      quotaError(body.error, reset);
      expect(body.error).toMatchObject({ retryable: false });
    }
  } finally { budget.dispose(); }
});

test.each([false, true])("live Chat handler preserves post-text CLI exhaustion (stream=%s)", async stream => {
  let launches = 0;
  const adapter = createClaudeCliAdapter(provider, {
    which: () => process.execPath, usageAdmission: available,
    usageObserver: () => ({ observe() {}, finish() {} }), usageRefusal: async () => {},
    spawn: () => { launches++; return child([
      { type: "stream_event", event: { type: "content_block_delta", delta: { type: "text_delta", text: "partial answer" } } },
      { type: "result", subtype: "error_during_execution", is_error: true, result: `Rate limit reached: ${limits[0]}`,
        usage: { input_tokens: 7, output_tokens: 2, cache_read_input_tokens: 11 } },
    ]); }, killGraceMs: 10, reapTimeoutMs: 100,
  });
  const started = Date.now();
  const reset = nextReset(started);
  const http = await coreHttp(adapter, false, "chat/completions", stream);
  expect(launches).toBe(1);
  if (stream) {
    expect(http.status).toBe(200);
    expect(http.contentType).toContain("text/event-stream");
    expect(http.text).not.toContain("data: [DONE]");
    const frames = await collectSse(new Response(http.text).body!);
    expect(frames.some(frame => JSON.stringify(frame.data).includes("partial answer"))).toBe(true);
    const error = frames.find(frame => frame.data.error)?.data.error;
    quotaError(error, reset);
    expect(error).toMatchObject({ retryable: false });
  } else {
    expect(http.status).toBe(429);
    expect(http.contentType).toContain("application/json");
    quotaError(JSON.parse(http.text).error, reset);
    expect(JSON.parse(http.text).error).toMatchObject({ retryable: false });
    expect(Number(http.retryAfter)).toBeGreaterThanOrEqual(Math.max(1, Math.ceil((reset - Date.now()) / 1000)));
    expect(Number(http.retryAfter)).toBeLessThanOrEqual(Math.max(1, Math.ceil((reset - started) / 1000)));
  }
});

test.each([false, true])("live Chat handler invents no reset or retry delay for unknown exhaustion (stream=%s)", async stream => {
  let launches = 0;
  const adapter = createClaudeCliAdapter(provider, {
    which: () => process.execPath,
    usageAdmission: async () => ({ state: "exhausted", checkedAt: 0, message: "Subscription exhausted" }),
    usageObserver: () => ({ observe() {}, finish() {} }), usageRefusal: async () => {},
    spawn: () => { launches++; throw new Error("Unknown-reset admission must not launch a CLI"); },
  });
  const http = await coreHttp(adapter, false, "chat/completions", stream);
  expect(http.status).toBe(stream ? 200 : 429);
  expect(http.retryAfter).toBeNull();
  expect(launches).toBe(0);
  const error = stream ? (await collectSse(new Response(http.text).body!)).find(frame => frame.data.error)?.data.error
    : JSON.parse(http.text).error;
  expect(error).toMatchObject({ type: quotaType, code: quotaType });
  expect(error).not.toHaveProperty("resets_at");
  expect(error).not.toHaveProperty("plan_type");
});

test.each([
  { status: 429, type: "rate_limit_error", code: "rate_limit_exceeded", message: "Too many requests" },
  { status: 401, type: "authentication_error", code: "invalid_api_key", message: "Unauthorized" },
  { status: 529, type: "upstream_error", code: "overloaded_error", message: "Overloaded" },
])("Chat does not attach quota metadata to $type", fields => {
  const { status, type, code, message } = fields;
  expect(chatCompletionsErrorBody(status, message, type, code, 2_000_000_000))
    .toEqual(chatCompletionsErrorBody(status, message, type, code));
  expect(chatCompletionsStreamErrorPayload(message, { ...fields, resets_at: 2_000_000_000 }))
    .toEqual(chatCompletionsStreamErrorPayload(message, fields));
  expect(chatCompletionsErrorResponse(status, message, type, code, 2_000_000_000).headers.get("retry-after")).toBeNull();
});

test.each([undefined, NaN, Infinity, -Infinity, 0, -1, 8.64e12 + 1])("Chat rejects invalid reset seconds: %s", resets_at => {
  const details = { status: 429, type: quotaType, code: quotaType, resets_at };
  for (const body of [chatCompletionsErrorBody(429, "Subscription exhausted", quotaType, quotaType, resets_at),
    chatCompletionsStreamErrorPayload("Subscription exhausted", details).payload]) {
    expect(body.error).toMatchObject({ type: quotaType, code: quotaType, retryable: false });
    expect(body.error).not.toHaveProperty("resets_at");
    expect(body.error).not.toHaveProperty("plan_type");
  }
  expect(chatCompletionsErrorResponse(429, "Subscription exhausted", quotaType, quotaType, resets_at).headers.get("retry-after")).toBeNull();
});

describe.skipIf(!nativeBinary)("real Claude SDK/native worker quota failures", () => {
  test.each(limits)("core HTTP prelude preserves the first native 429 with available preflight: %s", async message => {
    const h = nativeSession([]);
    let requests = 0;
    const api = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch(req) {
      const path = new URL(req.url).pathname;
      if (path === "/v1/messages/count_tokens") return Response.json({ input_tokens: 12 });
      expect(path).toBe("/v1/messages");
      expect(req.headers.get("x-api-key")).toBe("offline-native-fixture-key");
      requests++;
      return Response.json({ type: "error", error: { type: "rate_limit_error", message } },
        { status: 429, headers: { "retry-after": "0" } });
    } });
    const base = `http://127.0.0.1:${api.port}`;
    const run = h.manager.run.bind(h.manager);
    h.manager.run = (input, emit) => run({ ...input, env: { ...h.input.env,
      ANTHROPIC_BASE_URL: base, HTTP_PROXY: base, HTTPS_PROXY: base, ALL_PROXY: base, CLAUDE_CODE_MAX_RETRIES: "0",
    } }, emit);
    const adapter = createClaudeCliAdapter(provider, {
      which: () => nativeBinary!, usageAdmission: available,
      usageObserver: () => ({ observe() {}, finish() {} }), usageRefusal: async () => {},
      sessionManager: h.manager,
    });
    try {
      const reset = nextReset(Date.now());
      const http = await coreHttp(adapter, true);
      expect(http).toMatchObject({ status: 429 });
      expect(requests).toBe(1);
      expect(h.children).toHaveLength(1);
      expect(h.unexpected).toEqual([]);
      expect(http.status).toBe(429);
      expect(http.contentType).toContain("application/json");
      quotaError(JSON.parse(http.text).error, reset);
    } finally { await h.close(); await api.stop(true); }
  }, 25_000);
});

test.each([
  { status: 429, errorType: "rate_limit_error", code: "rate_limit_exceeded", message: "Too many requests", retryable: true },
  { status: 529, errorType: "upstream_error", code: "overloaded_error", message: "Overloaded", retryable: true },
  { status: 401, errorType: "authentication_error", code: "invalid_api_key", message: "Invalid API key", retryable: false },
  { status: 429, errorType: "rate_limit_error", code: "rate_limit_exceeded", message: "Usage limit reached; reset time unknown", retryable: true },
])("unconfirmed failures retain their existing retry verdict: $message", fields => {
  const event: Failure = { type: "error", ...fields };
  const output: AdapterEvent[] = [];
  withClaudeLoginHint(value => output.push(value), "offline-model")(event);
  expect(output).toEqual([event]);
  expect(adapterFailureFromEvent(event).error).not.toHaveProperty("resets_at");
});

test("a caller model id carrying a reset phrase cannot establish a quota verdict", () => {
  const model = "offline-model resets 3:30pm (UTC)";
  const event: Failure = { type: "error", status: 429, errorType: "rate_limit_error",
    code: "rate_limit_exceeded", message: `Usage limit reached for model ${model}`, retryable: true };
  const events: AdapterEvent[] = [];
  withClaudeLoginHint(value => events.push(value), model)(event);
  expect(events).toEqual([event]);
  expect(adapterFailureFromEvent(event).error).not.toHaveProperty("resets_at");
});

test("shared mapper carries reset seconds only on an explicit typed quota verdict", () => {
  const reset = 2_000_000_000_123;
  const event: Failure = { type: "error", status: 429, errorType: quotaType, code: quotaType,
    message: "Subscription exhausted", retryable: false, usageLimitResetAt: reset };
  const mapped = adapterFailureFromEvent(event);
  expect(mapped.httpStatus).toBe(429);
  quotaError(mapped.error, Math.ceil(reset / 1000) * 1000);
  for (const fields of [
    { errorType: "rate_limit_error", code: "rate_limit_exceeded" },
    { errorType: "rate_limit_error", code: quotaType },
    { errorType: quotaType, code: "rate_limit_exceeded" },
    { errorType: undefined, code: undefined },
  ]) expect(adapterFailureFromEvent({ ...event, ...fields }).error).not.toHaveProperty("resets_at");
});

test.each([undefined, NaN, Infinity, -Infinity, 0, -1, 8.64e15 + 1])("invalid quota reset cannot enter the client payload: %s", usageLimitResetAt => {
  const failure = adapterFailureFromEvent({ type: "error", status: 429, errorType: quotaType,
    code: quotaType, message: "Subscription exhausted", retryable: false, usageLimitResetAt });
  expect(failure.error).toMatchObject({ type: quotaType, code: quotaType });
  expect(failure.error).not.toHaveProperty("resets_at");
  expect(failure.error).not.toHaveProperty("plan_type");
});
