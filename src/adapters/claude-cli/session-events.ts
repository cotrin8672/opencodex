import type { SDKAssistantMessageError, SDKMessage, SDKPartialAssistantMessage } from "@anthropic-ai/claude-agent-sdk";
import { createTranslatorBudget } from "../../lib/translator-budget";
import type { AdapterEvent, OcxUsage } from "../../types";
import { CodingAgentProtocolError, MAX_TOOL_BLOCK_STARTS, releaseOpenToolBlocks, usageFromResult, type StreamParseState } from "../coding-agent/protocol";
import { isValidCodingAgentToolNamePart } from "../coding-agent/tool-catalog";

type ContentKind = "text" | "thinking";
type SdkEvent = SDKPartialAssistantMessage["event"];
type ErrorEvent = Extract<AdapterEvent, { type: "error" }>;

export interface ClaudeSdkStreamState extends StreamParseState {
  sdkResponseId?: string;
  sdkContentBlocks?: Map<number, { kind: ContentKind; streamed: boolean; claimed: boolean }>;
  sdkResponseUsage?: Map<string, OcxUsage>;
}

export function createClaudeSdkStreamState(options: Pick<StreamParseState,
  "toolBridgeNames" | "translatorBudget" | "maxToolBlockStarts"> = {}): ClaudeSdkStreamState {
  return { sawPartialText: false, sawPartialThinking: false, sawTerminalResult: false, sawMessageStop: false,
    strictToolBlockCapture: true, completeAssistantToolUse: true, openToolBlocks: new Map(),
    partialToolCallIds: new Set(), ...options, translatorBudget: options.translatorBudget ?? createTranslatorBudget() };
}

export function releaseClaudeSdkBlocks(state: ClaudeSdkStreamState): void {
  releaseOpenToolBlocks(state);
  state.sdkContentBlocks?.clear();
  state.sdkResponseUsage?.clear();
}

function response(state: ClaudeSdkStreamState, id: string): void {
  if (state.sdkResponseId === id) return;
  if (state.sdkResponseId === undefined) { state.sdkResponseId = id; return; }
  if (state.openToolBlocks?.size) throw new CodingAgentProtocolError("Claude began a response with unfinished tool arguments.");
  state.sdkResponseId = id;
  state.sdkContentBlocks = new Map();
  state.sawPartialText = false;
  state.sawPartialThinking = false;
  state.sawMessageStop = false;
}

function foldUsage(a: OcxUsage, b: OcxUsage, combine: (a: number, b: number) => number): OcxUsage {
  const read = combine(a.cacheReadInputTokens ?? 0, b.cacheReadInputTokens ?? 0);
  const write = combine(a.cacheCreationInputTokens ?? 0, b.cacheCreationInputTokens ?? 0);
  const write1h = Math.min(write, combine(a.cacheCreation1hInputTokens ?? 0, b.cacheCreation1hInputTokens ?? 0));
  const fresh = (u: OcxUsage) => Math.max(0, u.inputTokens - (u.cacheReadInputTokens ?? 0) - (u.cacheCreationInputTokens ?? 0));
  const inputTokens = combine(fresh(a), fresh(b)) + read + write;
  const outputTokens = combine(a.outputTokens, b.outputTokens);
  return { inputTokens, outputTokens, totalTokens: inputTokens + outputTokens,
    ...(read ? { cachedInputTokens: read, cacheReadInputTokens: read } : {}),
    ...(write ? { cacheCreationInputTokens: write } : {}), ...(write1h ? { cacheCreation1hInputTokens: write1h } : {}) };
}

const ZERO_USAGE: OcxUsage = { inputTokens: 0, outputTokens: 0, totalTokens: 0 };

function observeUsage(state: ClaudeSdkStreamState, usage: object): void {
  const next = usageFromResult({ usage });
  if (!next) return;
  const snapshots = state.sdkResponseUsage ??= new Map();
  const id = state.sdkResponseId ?? "";
  snapshots.set(id, foldUsage(snapshots.get(id) ?? ZERO_USAGE, next, Math.max));
  state.partialUsage = [...snapshots.values()].reduce((a, b) => foldUsage(a, b, (x, y) => x + y), ZERO_USAGE);
}

function failure(message: string, status = 502, code = "upstream_error", errorType = "upstream_error", retryable = false): ErrorEvent {
  return { type: "error", message, status, code, errorType, retryable };
}

function statusFailure(message: string, status: number | null | undefined): ErrorEvent | undefined {
  switch (status) {
    case 400: return failure(message, 400, "invalid_request_error", "invalid_request_error");
    case 401: return failure(message, 401, "invalid_api_key", "authentication_error");
    case 403: return failure(message, 403, "permission_denied", "permission_error");
    case 404: return failure(message, 404, "model_not_found", "not_found_error");
    case 413: return failure(message, 413, "request_too_large", "request_too_large");
    case 429: return failure(message, 429, "rate_limit_exceeded", "rate_limit_error", true);
    case 529: return failure(message, 529, "overloaded_error", "overloaded_error", true);
    case 500: case 502: case 503: case 504: return failure(message, status, "upstream_error", "upstream_error", true);
    default: return undefined;
  }
}

function assistantFailure(message: string, error: SDKAssistantMessageError): ErrorEvent {
  switch (error) {
    case "authentication_failed": case "cloud_credential_error": return statusFailure(message, 401)!;
    case "oauth_org_not_allowed": case "account_on_hold": case "verification_required": return statusFailure(message, 403)!;
    case "billing_error": return failure(message, 429, "insufficient_quota", "insufficient_quota");
    case "rate_limit": return statusFailure(message, 429)!;
    case "overloaded": return statusFailure(message, 529)!;
    case "invalid_request": return statusFailure(message, 400)!;
    case "model_not_found": return failure(message, 400, "model_not_found", "invalid_request_error");
    case "server_error": return statusFailure(message, 500)!;
    default: return failure(message);
  }
}

function resultFailure(message: string): ErrorEvent {
  if (/credit usage limit/i.test(message)) return failure(message, 429, "insufficient_quota", "insufficient_quota");
  if (/not logged in|invalid (?:personal access )?token|authentication (?:failed|error|required)|unauthorized/i.test(message)) return statusFailure(message, 401)!;
  if (/rate limit|too many requests/i.test(message)) return statusFailure(message, 429)!;
  if (/model (?:is )?(?:not found|unavailable|unsupported)|invalid model/i.test(message)) return failure(message, 400, "model_not_found", "invalid_request_error");
  return failure(message);
}

function argumentsJson(input: unknown): string {
  if (!input || typeof input !== "object" || Array.isArray(input)) throw new CodingAgentProtocolError("Claude returned non-object tool arguments.");
  return JSON.stringify(input);
}

function parseArguments(parts: string[]): string {
  let input: unknown;
  try { input = JSON.parse(parts.length ? parts.join("") : "{}"); }
  catch { throw new CodingAgentProtocolError("Claude ended a tool block with incomplete JSON arguments."); }
  return argumentsJson(input);
}

function validateTool(state: ClaudeSdkStreamState, id: string, name: string): void {
  if (!id || id.length > 256 || id.trim() !== id || !isValidCodingAgentToolNamePart(id)
    || !isValidCodingAgentToolNamePart(name) || !state.toolBridgeNames?.has(name)) {
    throw new CodingAgentProtocolError("Claude returned an invalid identity or undeclared tool name.");
  }
}

function isRepeat(state: ClaudeSdkStreamState, id: string, name: string, json: string): boolean {
  const previous = state.emittedToolCalls?.get(id);
  if (!previous) return false;
  if (previous.name !== name || previous.argumentsJson !== json) throw new CodingAgentProtocolError("Claude reused a tool ID with conflicting name or arguments.");
  return true;
}

let nextLease = 0;

function startTool(state: ClaudeSdkStreamState, index: number, id: string, name: string, input: unknown): void {
  validateTool(state, id, name);
  if (!Number.isInteger(index) || index < 0 || state.openToolBlocks?.has(index)
    || [...state.openToolBlocks?.values() ?? []].some(block => block.id === id)) {
    throw new CodingAgentProtocolError("Claude reused an open tool index or identity.");
  }
  const configuredLimit = state.maxToolBlockStarts ?? MAX_TOOL_BLOCK_STARTS;
  const limit = Number.isFinite(configuredLimit) ? Math.max(0, Math.min(MAX_TOOL_BLOCK_STARTS, Math.floor(configuredLimit))) : MAX_TOOL_BLOCK_STARTS;
  if ((state.toolBlockStarts ?? 0) >= limit) { state.toolCallLimitExceeded = true; return; }
  const json = argumentsJson(input);
  const argParts = json === "{}" ? [] : [json];
  const budget = state.translatorBudget ??= createTranslatorBudget();
  const budgetCallId = `claude-sdk:${++nextLease}`;
  budget.openCall(budgetCallId);
  try {
    budget.chargeRetained(Buffer.byteLength(id) + Buffer.byteLength(name) + (argParts.length ? Buffer.byteLength(json) : 0), { kind: "tool_args", callId: budgetCallId });
  } catch (error) { budget.closeCall(budgetCallId); throw error; }
  (state.openToolBlocks ??= new Map()).set(index, { id, name, argParts, indexed: true, budgetCallId });
  (state.partialToolCallIds ??= new Set()).add(id);
  state.toolBlockStarts = (state.toolBlockStarts ?? 0) + 1;
}

function closeTool(state: ClaudeSdkStreamState, index: number): AdapterEvent[] {
  const block = state.openToolBlocks?.get(index);
  if (!block) return [];
  const json = parseArguments(block.argParts);
  const repeat = isRepeat(state, block.id, block.name, json);
  if (!repeat) {
    const difference = Buffer.byteLength(json) - block.argParts.reduce((n, part) => n + Buffer.byteLength(part), 0);
    if (difference > 0) state.translatorBudget?.chargeRetained(difference, { kind: "tool_args", callId: block.budgetCallId });
    if (difference < 0) state.translatorBudget?.releaseRetained(-difference, { kind: "tool_args", callId: block.budgetCallId });
    (state.emittedToolCalls ??= new Map()).set(block.id, { name: block.name, argumentsJson: json,
      budgetCallIds: block.budgetCallId ? [block.budgetCallId] : [] });
  } else if (block.budgetCallId) state.translatorBudget?.closeCall(block.budgetCallId);
  state.openToolBlocks!.delete(index);
  if (repeat) return [];
  state.completedToolCalls = (state.completedToolCalls ?? 0) + 1;
  return [{ type: "tool_call_start", id: block.id, name: block.name },
    { type: "tool_call_delta", arguments: json }, { type: "tool_call_end" }];
}

function streamEvent(event: SdkEvent, state: ClaudeSdkStreamState): AdapterEvent[] {
  switch (event.type) {
    case "message_start":
      response(state, event.message.id);
      state.sawMessageStop = false;
      observeUsage(state, event.message.usage);
      return [];
    case "message_delta": observeUsage(state, event.usage); return [];
    case "message_stop":
      state.sawMessageStop = true;
      if (state.openToolBlocks?.size) state.uncapturedToolUse = true;
      return [];
    case "content_block_start": {
      if (state.openToolBlocks?.has(event.index)) throw new CodingAgentProtocolError("Claude reused an unfinished tool block index.");
      const block = event.content_block;
      if (block.type === "tool_use") startTool(state, event.index, block.id, block.name, block.input);
      if (block.type === "text" || block.type === "thinking") {
        (state.sdkContentBlocks ??= new Map()).set(event.index, { kind: block.type, streamed: false, claimed: false });
      }
      return [];
    }
    case "content_block_stop": return closeTool(state, event.index);
    case "content_block_delta": {
      const delta = event.delta;
      if (delta.type === "input_json_delta") {
        const block = state.openToolBlocks?.get(event.index);
        if (!block) throw new CodingAgentProtocolError("Claude sent arguments without a matching open tool block.");
        if (delta.partial_json) {
          state.translatorBudget?.chargeRetained(Buffer.byteLength(delta.partial_json), { kind: "tool_args", callId: block.budgetCallId });
          block.argParts.push(delta.partial_json);
        }
        return [];
      }
      if (delta.type !== "text_delta" && delta.type !== "thinking_delta") return [];
      const kind = delta.type === "text_delta" ? "text" : "thinking";
      const text = delta.type === "text_delta" ? delta.text : delta.thinking;
      if (!text) return [];
      const blocks = state.sdkContentBlocks ??= new Map();
      const block = blocks.get(event.index) ?? { kind, streamed: false, claimed: false };
      block.streamed = true;
      blocks.set(event.index, block);
      if (kind === "text") state.sawPartialText = true;
      else state.sawPartialThinking = true;
      return kind === "text" ? [{ type: "text_delta", text }] : [{ type: "thinking_delta", thinking: text }];
    }
  }
  return [];
}

/** SDK assistant frames deliver completed blocks, often sharing one response ID. */
export function mapClaudeSdkMessage(message: SDKMessage, state: ClaudeSdkStreamState): AdapterEvent[] {
  if (message.type === "stream_event") return streamEvent(message.event, state);
  if (message.type === "assistant") {
    response(state, message.message.id);
    observeUsage(state, message.message.usage);
    if (message.error) {
      const detail = message.message.content.filter(block => block.type === "text").map(block => block.text).join("\n") || "Claude API request failed.";
      return [{ ...assistantFailure(detail, message.error), ...(state.partialUsage ? { usage: state.partialUsage } : {}) }];
    }
    if (message.message.stop_reason === "tool_use") state.sawAssistantToolUseStop = true;
    const events: AdapterEvent[] = [];
    for (const block of message.message.content) {
      if (block.type === "tool_use") {
        validateTool(state, block.id, block.name);
        const json = argumentsJson(block.input);
        if (isRepeat(state, block.id, block.name, json)) continue;
        const open = [...state.openToolBlocks?.values() ?? []].find(item => item.id === block.id);
        if (open) {
          if (open.name !== block.name || parseArguments(open.argParts) !== json) throw new CodingAgentProtocolError("Claude's completed tool block disagrees with its streamed arguments.");
          continue;
        }
        const index = state.openToolBlocks?.size ? Math.max(...state.openToolBlocks.keys()) + 1 : 0;
        startTool(state, index, block.id, block.name, block.input);
        if (!state.toolCallLimitExceeded) events.push(...closeTool(state, index));
      } else if (block.type === "text" || block.type === "thinking") {
        const partial = [...state.sdkContentBlocks?.values() ?? []].find(item => item.kind === block.type && !item.claimed);
        if (partial) partial.claimed = true;
        if (partial?.streamed) continue;
        if (block.type === "text" && block.text) events.push({ type: "text_delta", text: block.text });
        if (block.type === "thinking" && block.thinking) events.push({ type: "thinking_delta", thinking: block.thinking });
      }
    }
    return events;
  }
  if (message.type === "result") {
    const usage = usageFromResult({ usage: message.usage }) ?? { ...ZERO_USAGE };
    state.partialUsage = usage;
    state.sawTerminalResult = true;
    if (message.subtype !== "success" || message.is_error) {
      const detail = message.subtype === "success" ? message.result : message.errors.find(error => error.trim());
      const text = detail || "Claude ended the turn with an execution error.";
      const error = (message.subtype === "success" ? statusFailure(text, message.api_error_status) : undefined) ?? resultFailure(text);
      return [{ ...error, usage }];
    }
    if (state.openToolBlocks?.size || state.uncapturedToolUse || state.toolCallLimitExceeded) throw new CodingAgentProtocolError("Claude ended the turn with an incomplete tool batch.");
    return [{ type: "done", usage, stopReason: "stop" }];
  }
  return [];
}
