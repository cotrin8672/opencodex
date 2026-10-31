import { afterEach, describe, expect, test } from "bun:test";
import type { NonNullableUsage, SDKAssistantMessage, SDKAssistantMessageError, SDKPartialAssistantMessage, SDKResultError, SDKResultSuccess } from "@anthropic-ai/claude-agent-sdk";
import { createClaudeSdkStreamState, mapClaudeSdkMessage, releaseClaudeSdkBlocks, type ClaudeSdkStreamState } from "../../src/adapters/claude-cli/session-events";
import { CodingAgentProtocolError, MAX_TOOL_BLOCK_STARTS, releaseOpenToolBlocks } from "../../src/adapters/coding-agent/protocol";
import { createTranslatorBudget, TranslatorBudgetExceededError } from "../../src/lib/translator-budget";

const NAME = "mcp__opencodex__lookup";
const envelope = { uuid: "00000000-0000-4000-8000-000000000001", session_id: "offline-session", parent_tool_use_id: null } as const;
const usage = (changes: Partial<NonNullableUsage> = {}): NonNullableUsage => ({
  input_tokens: 0, output_tokens: 0, cache_read_input_tokens: 0, cache_creation_input_tokens: 0,
  cache_creation: { ephemeral_1h_input_tokens: 0, ephemeral_5m_input_tokens: 0 },
  fallback_credit: null, inference_geo: "us", iterations: [], output_tokens_details: { thinking_tokens: 0 },
  server_tool_use: { web_search_requests: 0, web_fetch_requests: 0 }, service_tier: "standard", speed: "standard", ...changes,
});
const text = (value: string) => ({ type: "text", text: value, citations: null } as const);
const thinking = (value: string) => ({ type: "thinking", thinking: value, signature: "signature" } as const);
const tool = (id = "toolu_1", input: unknown = {}, name = NAME) => ({ type: "tool_use", id, name, input } as const);
function assistant(content: SDKAssistantMessage["message"]["content"], changes: Partial<SDKAssistantMessage["message"]> = {}): SDKAssistantMessage {
  return { ...envelope, type: "assistant", message: { id: "msg_1", type: "message", role: "assistant", model: "claude-test",
    content, container: null, context_management: null, diagnostics: null, stop_details: null,
    stop_reason: null, stop_sequence: null, usage: usage(), ...changes } };
}
const stream = (event: SDKPartialAssistantMessage["event"]): SDKPartialAssistantMessage => ({ ...envelope, type: "stream_event", event });
const start = (id = "msg_1", counters = usage()) => stream({ type: "message_start", message: assistant([], { id, usage: counters }).message });
const stop = (index: number) => stream({ type: "content_block_stop", index });
const messageStop = () => stream({ type: "message_stop" });
const toolStart = (index = 0, id = "toolu_1", input: unknown = {}, name = NAME) => stream({ type: "content_block_start", index, content_block: tool(id, input, name) });
const args = (partial_json: string, index = 0) => stream({ type: "content_block_delta", index, delta: { type: "input_json_delta", partial_json } });
function result(changes: Partial<SDKResultSuccess> = {}): SDKResultSuccess {
  return { ...envelope, type: "result", subtype: "success", duration_ms: 1, duration_api_ms: 1, is_error: false,
    num_turns: 1, result: "", stop_reason: "end_turn", total_cost_usd: 0, usage: usage(), modelUsage: {}, permission_denials: [], ...changes };
}
const states: ClaudeSdkStreamState[] = [];
function state(options: Parameters<typeof createClaudeSdkStreamState>[0] = {}) {
  const value = createClaudeSdkStreamState({ toolBridgeNames: new Map([[NAME, "lookup"]]), ...options });
  states.push(value);
  return value;
}
afterEach(() => { for (const value of states.splice(0)) { releaseClaudeSdkBlocks(value); value.translatorBudget?.dispose(); } });

describe("Claude SDK session events", () => {
  test("completed assistant frames deliver one block at a time under a shared response ID", () => {
    const s = state();
    mapClaudeSdkMessage(start(), s);
    mapClaudeSdkMessage(stream({ type: "content_block_start", index: 0, content_block: text("") }), s);
    expect(mapClaudeSdkMessage(stream({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "first" } }), s))
      .toEqual([{ type: "text_delta", text: "first" }]);
    mapClaudeSdkMessage(stop(0), s);
    expect(mapClaudeSdkMessage(assistant([text("first")]), s)).toEqual([]);
    expect(mapClaudeSdkMessage(assistant([text("second")]), s)).toEqual([{ type: "text_delta", text: "second" }]);
    mapClaudeSdkMessage(stream({ type: "content_block_start", index: 2, content_block: thinking("") }), s);
    expect(mapClaudeSdkMessage(stream({ type: "content_block_delta", index: 2, delta: { type: "thinking_delta", thinking: "reason", estimated_tokens: null } }), s))
      .toEqual([{ type: "thinking_delta", thinking: "reason" }]);
    expect(mapClaudeSdkMessage(assistant([thinking("reason")]), s)).toEqual([]);
    expect(mapClaudeSdkMessage(assistant([thinking("more")]), s)).toEqual([{ type: "thinking_delta", thinking: "more" }]);
    mapClaudeSdkMessage(messageStop(), s);
    mapClaudeSdkMessage(start("msg_2"), s);
    expect(s.sawMessageStop).toBe(false);
    expect(mapClaudeSdkMessage(assistant([text("next"), thinking("next reason")], { id: "msg_2" }), s))
      .toEqual([{ type: "text_delta", text: "next" }, { type: "thinking_delta", thinking: "next reason" }]);
  });

  test("an empty partial block does not suppress completed content or unrelated kinds", () => {
    const s = state();
    mapClaudeSdkMessage(start(), s);
    mapClaudeSdkMessage(stream({ type: "content_block_start", index: 0, content_block: text("") }), s);
    mapClaudeSdkMessage(stop(0), s);
    expect(mapClaudeSdkMessage(assistant([text("fallback"), thinking("reason")]), s))
      .toEqual([{ type: "text_delta", text: "fallback" }, { type: "thinking_delta", thinking: "reason" }]);
    expect(mapClaudeSdkMessage(assistant([text("another")]), s)).toEqual([{ type: "text_delta", text: "another" }]);
  });

  test("usage takes response maxima, sums distinct responses, and accepts an authoritative result", () => {
    const s = state();
    const first = usage({ input_tokens: 10, cache_read_input_tokens: 20, cache_creation_input_tokens: 5,
      cache_creation: { ephemeral_1h_input_tokens: 5, ephemeral_5m_input_tokens: 0 }, output_tokens: 1 });
    mapClaudeSdkMessage(start("msg_1", first), s);
    mapClaudeSdkMessage(stream({ type: "message_delta", delta: { stop_reason: "end_turn", stop_sequence: null, container: null, stop_details: null },
      context_management: null, usage: usage({ output_tokens: 7 }) }), s);
    mapClaudeSdkMessage(assistant([text("a")], { usage: first }), s);
    mapClaudeSdkMessage(assistant([text("b")], { usage: first }), s);
    expect(s.partialUsage).toMatchObject({ inputTokens: 35, outputTokens: 7, totalTokens: 42, cachedInputTokens: 20, cacheCreation1hInputTokens: 5 });
    mapClaudeSdkMessage(messageStop(), s);
    mapClaudeSdkMessage(start("msg_2", usage({ input_tokens: 4, output_tokens: 2 })), s);
    expect(s.partialUsage).toMatchObject({ inputTokens: 39, outputTokens: 9, totalTokens: 48 });
    const events = mapClaudeSdkMessage(result({ usage: usage({ input_tokens: 3, output_tokens: 1 }) }), s);
    expect(events).toEqual([{ type: "done", usage: { inputTokens: 3, outputTokens: 1, totalTokens: 4 }, stopReason: "stop" }]);
    expect(s.partialUsage).toEqual({ inputTokens: 3, outputTokens: 1, totalTokens: 4 });
    expect(s.sawTerminalResult).toBe(true);
  });

  test("cache-only partial usage survives until a zero authoritative result replaces it", () => {
    const s = state();
    mapClaudeSdkMessage(start("msg_1", usage({ cache_creation_input_tokens: 8 })), s);
    expect(s.partialUsage).toMatchObject({ inputTokens: 8, cacheCreationInputTokens: 8 });
    expect(mapClaudeSdkMessage(result(), s)).toEqual([{ type: "done", stopReason: "stop", usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 } }]);
    expect(s.partialUsage?.inputTokens).toBe(0);
  });

  const errors: Array<[SDKAssistantMessageError, number, string, boolean]> = [
    ["rate_limit", 429, "rate_limit_exceeded", true], ["overloaded", 529, "overloaded_error", true],
    ["authentication_failed", 401, "invalid_api_key", false], ["cloud_credential_error", 401, "invalid_api_key", false],
    ["invalid_request", 400, "invalid_request_error", false], ["model_not_found", 400, "model_not_found", false],
    ["billing_error", 429, "insufficient_quota", false], ["oauth_org_not_allowed", 403, "permission_denied", false],
    ["account_on_hold", 403, "permission_denied", false], ["verification_required", 403, "permission_denied", false],
    ["server_error", 500, "upstream_error", true], ["unknown", 502, "upstream_error", false], ["max_output_tokens", 502, "upstream_error", false],
  ];
  test.each(errors)("structured assistant error %s has status %i", (error, status, code, retryable) => {
    const s = state();
    expect(mapClaudeSdkMessage({ ...assistant([text("upstream detail")], { usage: usage({ input_tokens: 2 }) }), error }, s))
      .toEqual([{ type: "error", message: "upstream detail", status, code, errorType: expect.any(String), retryable,
        usage: { inputTokens: 2, outputTokens: 0, totalTokens: 2 } }]);
  });

  test.each([429, 529, 401, 400])("result API status %i overrides ambiguous error prose", status => {
    const [event] = mapClaudeSdkMessage(result({ is_error: true, result: "service failure", api_error_status: status }), state());
    expect(event).toMatchObject({ type: "error", status, retryable: status === 429 || status === 529 });
  });
  test.each([418, 200, 0, -1, NaN, Infinity, null])("unexpected SDK API status %s is ignored", api_error_status => {
    expect(mapClaudeSdkMessage(result({ is_error: true, result: "service failure", api_error_status }), state())[0])
      .toMatchObject({ type: "error", status: 502, code: "upstream_error", retryable: false });
  });
  test("result-only errors preserve normalized statuses and authoritative usage", () => {
    for (const [detail, status, code] of [["too many requests", 429, "rate_limit_exceeded"], ["not logged in", 401, "invalid_api_key"],
      ["invalid model", 400, "model_not_found"], ["credit usage limit", 429, "insufficient_quota"]] as const) {
      const error: SDKResultError = { ...result({ usage: usage({ output_tokens: 3 }) }), subtype: "error_during_execution", errors: ["", detail] };
      const s = state();
      expect(mapClaudeSdkMessage(error, s)[0]).toMatchObject({ type: "error", status, code, usage: { outputTokens: 3 } });
      expect(s.sawTerminalResult).toBe(true);
    }
    for (const subtype of ["error_max_turns", "error_max_budget_usd", "error_max_structured_output_retries"] as const) {
      expect(mapClaudeSdkMessage({ ...result(), subtype, errors: [] }, state())[0]).toMatchObject({ type: "error", status: 502 });
    }
  });

  test("tool capture waits for complete indexed arguments and never supplies dispatch or terminal completion", () => {
    const s = state();
    mapClaudeSdkMessage(start(), s);
    expect(mapClaudeSdkMessage(toolStart(1), s)).toEqual([]);
    expect(mapClaudeSdkMessage(args('{"query":"日', 1), s)).toEqual([]);
    expect(mapClaudeSdkMessage(args('本\\n語"}', 1), s)).toEqual([]);
    expect(mapClaudeSdkMessage(assistant([tool("toolu_1", { query: "日本\n語" })]), s)).toEqual([]);
    expect(s.completedToolCalls ?? 0).toBe(0);
    expect(mapClaudeSdkMessage(stop(0), s)).toEqual([]);
    expect(mapClaudeSdkMessage(stop(1), s)).toEqual([
      { type: "tool_call_start", id: "toolu_1", name: NAME },
      { type: "tool_call_delta", arguments: '{"query":"日本\\n語"}' }, { type: "tool_call_end" },
    ]);
    expect(s.completedToolCalls).toBe(1);
    expect(s.sawMessageStop).toBe(false);
    expect(mapClaudeSdkMessage(assistant([tool("toolu_1", { query: "日本\n語" })]), s)).toEqual([]);
    expect(mapClaudeSdkMessage(messageStop(), s)).toEqual([]);
    expect(s.sawMessageStop).toBe(true);
    expect(s.sawTerminalResult).toBe(false);
  });

  test("interleaved tool blocks emit atomic batches and completed assistant tools are deduplicated", () => {
    const s = state();
    mapClaudeSdkMessage(start(), s);
    mapClaudeSdkMessage(toolStart(0, "toolu_1"), s);
    mapClaudeSdkMessage(toolStart(1, "toolu_2"), s);
    mapClaudeSdkMessage(args('{"a":', 0), s);
    mapClaudeSdkMessage(args('{"b":2}', 1), s);
    mapClaudeSdkMessage(args("1}", 0), s);
    expect(mapClaudeSdkMessage(stop(1), s)).toEqual([{ type: "tool_call_start", id: "toolu_2", name: NAME },
      { type: "tool_call_delta", arguments: '{"b":2}' }, { type: "tool_call_end" }]);
    expect(mapClaudeSdkMessage(stop(0), s)[0]).toEqual({ type: "tool_call_start", id: "toolu_1", name: NAME });
    expect(mapClaudeSdkMessage(assistant([tool("toolu_2", { b: 2 })]), s)).toEqual([]);
    expect(s.completedToolCalls).toBe(2);
  });

  test("a complete assistant tool is a structured fallback and empty streamed arguments become an object", () => {
    const s = state();
    expect(mapClaudeSdkMessage(assistant([tool()]), s)).toEqual([{ type: "tool_call_start", id: "toolu_1", name: NAME },
      { type: "tool_call_delta", arguments: "{}" }, { type: "tool_call_end" }]);
    expect(mapClaudeSdkMessage(assistant([tool()]), s)).toEqual([]);
    mapClaudeSdkMessage(toolStart(0, "toolu_2"), s);
    expect(mapClaudeSdkMessage(stop(0), s)[1]).toEqual({ type: "tool_call_delta", arguments: "{}" });
    expect(s.completedToolCalls).toBe(2);
    expect(s.sawMessageStop).toBe(false);
  });

  test("the first assistant ID attaches to partial blocks even without a message_start snapshot", () => {
    const s = state();
    mapClaudeSdkMessage(stream({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "once" } }), s);
    mapClaudeSdkMessage(toolStart(1), s);
    mapClaudeSdkMessage(args('{"a":1}', 1), s);
    expect(mapClaudeSdkMessage(assistant([text("once"), tool("toolu_1", { a: 1 })]), s)).toEqual([]);
    expect(mapClaudeSdkMessage(stop(1), s)[0]).toEqual({ type: "tool_call_start", id: "toolu_1", name: NAME });
  });

  test("tool identities, catalog membership, and object arguments fail closed", () => {
    for (const id of ["", " padded", "padded ", "a\nb", "x".repeat(257)]) {
      expect(() => mapClaudeSdkMessage(toolStart(0, id), state())).toThrow(CodingAgentProtocolError);
      expect(() => mapClaudeSdkMessage(assistant([tool(id)]), state())).toThrow(CodingAgentProtocolError);
    }
    for (const input of [null, [], "{}", 2]) {
      expect(() => mapClaudeSdkMessage(assistant([tool("toolu_1", input)]), state())).toThrow(CodingAgentProtocolError);
    }
    expect(() => mapClaudeSdkMessage(toolStart(0, "toolu_1", {}, "unknown"), state())).toThrow(CodingAgentProtocolError);
    expect(() => mapClaudeSdkMessage(assistant([tool("toolu_1", {}, "unknown")]), state())).toThrow(CodingAgentProtocolError);
    expect(() => mapClaudeSdkMessage(toolStart(), state({ toolBridgeNames: new Map() }))).toThrow(CodingAgentProtocolError);
  });

  test("incomplete/non-object JSON, mismatched indices, and reused identities cannot complete a tool batch", () => {
    for (const fragment of ['{"a":', "[]", "null"]) {
      const s = state();
      mapClaudeSdkMessage(toolStart(), s);
      mapClaudeSdkMessage(args(fragment), s);
      expect(() => mapClaudeSdkMessage(stop(0), s)).toThrow(CodingAgentProtocolError);
      expect(s.completedToolCalls ?? 0).toBe(0);
    }
    const s = state();
    mapClaudeSdkMessage(toolStart(), s);
    expect(() => mapClaudeSdkMessage(args("{}", 1), s)).toThrow(CodingAgentProtocolError);
    expect(() => mapClaudeSdkMessage(toolStart(0, "toolu_2"), s)).toThrow(CodingAgentProtocolError);
    expect(() => mapClaudeSdkMessage(stream({ type: "content_block_start", index: 0, content_block: text("") }), s)).toThrow(CodingAgentProtocolError);
    expect(() => mapClaudeSdkMessage(toolStart(1), s)).toThrow(CodingAgentProtocolError);
    mapClaudeSdkMessage(args('{"a":1}'), s);
    expect(() => mapClaudeSdkMessage(assistant([tool("toolu_1", { a: 2 })]), s)).toThrow(CodingAgentProtocolError);
    mapClaudeSdkMessage(stop(0), s);
    expect(() => mapClaudeSdkMessage(assistant([tool("toolu_1", { a: 2 })]), s)).toThrow(CodingAgentProtocolError);
    const incomplete = state();
    mapClaudeSdkMessage(toolStart(), incomplete);
    mapClaudeSdkMessage(messageStop(), incomplete);
    expect(incomplete.uncapturedToolUse).toBe(true);
    expect(() => mapClaudeSdkMessage(result(), incomplete)).toThrow(CodingAgentProtocolError);
  });

  test("configured and shared hard call limits reject allocation before retaining an extra block", () => {
    for (const [maximum, limit] of [[0, 0], [1, 1], [1.5, 1], [MAX_TOOL_BLOCK_STARTS + 10, MAX_TOOL_BLOCK_STARTS], [NaN, MAX_TOOL_BLOCK_STARTS]]) {
      const s = state({ maxToolBlockStarts: maximum });
      for (let i = 0; i < limit; i++) mapClaudeSdkMessage(assistant([tool(`toolu_${i}`)]), s);
      const before = s.translatorBudget!.snapshot().currentBytes;
      expect(mapClaudeSdkMessage(toolStart(0, "toolu_extra"), s)).toEqual([]);
      expect(s.toolCallLimitExceeded).toBe(true);
      expect(s.toolBlockStarts ?? 0).toBe(limit);
      expect(s.translatorBudget!.snapshot().currentBytes).toBe(before);
    }
  });

  test("per-call UTF-8 budget is exact, reservations survive emit, and shared cleanup releases them", () => {
    const json = '{"x":"日"}';
    const bytes = Buffer.byteLength("toolu_1" + NAME + json);
    for (const maxCallArgumentBytes of [bytes, bytes - 1]) {
      const budget = createTranslatorBudget({ maxCallArgumentBytes, maxTurnBytes: 1024 });
      const s = state({ translatorBudget: budget });
      mapClaudeSdkMessage(toolStart(), s);
      if (maxCallArgumentBytes < bytes) expect(() => mapClaudeSdkMessage(args(json), s)).toThrow(TranslatorBudgetExceededError);
      else {
        mapClaudeSdkMessage(args(json), s);
        mapClaudeSdkMessage(stop(0), s);
        expect(budget.snapshot().currentBytes).toBe(bytes);
      }
      releaseOpenToolBlocks(s);
      expect(budget.snapshot()).toMatchObject({ currentBytes: 0, activeCalls: 0 });
    }
  });

  test("complete tools and multiple buffered calls obey the turn budget and cleanup after failure", () => {
    const bytes = Buffer.byteLength("toolu_1" + NAME + "{}");
    const budget = createTranslatorBudget({ maxCallArgumentBytes: 1024, maxTurnBytes: bytes });
    const s = state({ translatorBudget: budget });
    mapClaudeSdkMessage(assistant([tool()]), s);
    expect(() => mapClaudeSdkMessage(toolStart(1, "toolu_2"), s)).toThrow(TranslatorBudgetExceededError);
    releaseClaudeSdkBlocks(s);
    expect(budget.snapshot()).toMatchObject({ currentBytes: 0, activeCalls: 0 });
    const small = state({ translatorBudget: createTranslatorBudget({ maxCallArgumentBytes: bytes - 1 }) });
    expect(() => mapClaudeSdkMessage(assistant([tool()]), small)).toThrow(TranslatorBudgetExceededError);
    releaseClaudeSdkBlocks(small);
    expect(small.translatorBudget!.snapshot().currentBytes).toBe(0);
  });
});
