import { afterEach } from "bun:test";
import type { AdapterEvent, OcxMessage, OcxParsedRequest } from "../../src/types";
import { claudeSessionScope } from "../../src/adapters/claude-cli/session-context";
import { ClaudeSessionManager, type ClaudePersistentTurn, type ClaudeSessionManagerDeps } from "../../src/adapters/claude-cli/session-manager";
import type { ClaudeSessionJournal, ClaudeSessionSnapshot } from "../../src/adapters/claude-cli/session-journal";
import type { ClaudeSessionWorker, ClaudeWorkerOptions, ClaudeWorkerToolCall } from "../../src/adapters/claude-cli/session-worker";
import { stableClaudeToolBridge } from "../../src/adapters/claude-cli/stable-replay";
import { buildCodeBuddyToolBridge, CODEBUDDY_MCP_TOOL_PREFIX } from "../../src/adapters/codebuddy/tool-bridge";
import { createTestTranslatorBudget } from "../helpers/translator-budget";

export class MemoryJournal implements ClaudeSessionJournal {
  records = new Map<string, ClaudeSessionSnapshot>();
  rejectSave?: (snapshot: ClaudeSessionSnapshot) => boolean;
  closes = 0;
  load(scope: string) { return structuredClone(this.records.get(scope)); }
  save(scope: string, snapshot: ClaudeSessionSnapshot) {
    if (this.rejectSave?.(snapshot)) throw new Error("injected journal failure");
    this.records.set(scope, structuredClone(snapshot));
  }
  close() { this.closes++; }
}

export const user = (content: string): OcxMessage => ({ role: "user", content, timestamp: 0 });
export const assistant = (text: string): OcxMessage => ({ role: "assistant", content: [{ type: "text", text }], timestamp: 1 });
export function toolAssistant(calls: ClaudeWorkerToolCall[], name = "read_file"): OcxMessage {
  return { role: "assistant", timestamp: 1, content: calls.map(call => ({
    type: "toolCall", id: call.id, name, arguments: structuredClone(call.arguments),
  })) };
}
export function result(id: string, content: string, isError = false): OcxMessage {
  return { role: "toolResult", toolCallId: id, toolName: "read_file", content, isError, timestamp: 2 };
}

export function turn(messages: OcxMessage[] = [user("read both files")]): ClaudePersistentTurn {
  const parsed: OcxParsedRequest = {
    modelId: "offline-model", stream: true, options: {}, context: { messages, tools: [{
      name: "read_file", description: "Client-owned file read", parameters: {
        type: "object", properties: { path: { type: "string" } }, required: ["path"], additionalProperties: false,
      },
    }] },
  };
  const incoming = { headers: new Headers({ "thread-id": "offline-thread", "idempotency-key": "offline-operation" }), providerName: "claude-cli", translatorBudget: createTestTranslatorBudget() };
  const provider = { adapter: "claude-cli", baseUrl: "claude-cli://offline" };
  return {
    scope: claudeSessionScope(parsed, incoming, provider)!, parsed, incoming, provider,
    binary: "never-executed-claude", env: {}, systemPrompt: "Offline session test",
    bridge: stableClaudeToolBridge(buildCodeBuddyToolBridge(parsed)),
  };
}
export function withMessages(input: ClaudePersistentTurn, messages: OcxMessage[]): ClaudePersistentTurn {
  return { ...input, parsed: { ...input.parsed, context: { ...input.parsed.context, messages } } };
}
export function call(input: ClaudePersistentTurn, id: string, args = { path: "same.txt" }): ClaudeWorkerToolCall {
  return { id, name: input.bridge.tools[0]!.name, arguments: args };
}
export async function until(check: () => boolean): Promise<void> {
  for (let i = 0; i < 200; i++) {
    if (check()) return;
    await Bun.sleep(5);
  }
  throw new Error("Offline worker did not reach the expected state within 1s");
}

const envelope = { uuid: "00000000-0000-4000-8000-000000000001", session_id: "offline-session", parent_tool_use_id: null };
export class ControlledWorker implements ClaudeSessionWorker {
  private responseId = "msg_offline";
  private responseOrdinal = 0;
  sends: string[] = [];
  sendOptions: Array<{ uuid?: string; shouldQuery?: boolean; priority?: "now" | "next" | "later" } | undefined> = [];
  updates: ClaudeWorkerOptions["tools"][] = [];
  stops = 0;
  interrupts = 0;
  constructor(readonly options: ClaudeWorkerOptions) {}
  async send(text: string, options?: { uuid?: string; shouldQuery?: boolean; priority?: "now" | "next" | "later" }) {
    this.sends.push(text); this.sendOptions.push(options);
  }
  async interrupt() {
    this.interrupts++;
    await this.emit({ type: "result", subtype: "success", is_error: false, result: "", usage: { input_tokens: 0, output_tokens: 0 } });
  }
  async updateTools(tools: ClaudeWorkerOptions["tools"]) { this.updates.push(structuredClone(tools)); }
  stop() { this.stops++; }
  async emit(message: Record<string, unknown>) {
    if (message.type === "assistant") message = { ...message, message: {
      id: this.responseId, type: "message", model: this.options.model, stop_reason: null, stop_sequence: null,
      usage: { input_tokens: 3, output_tokens: 4 }, ...message.message as object,
    } };
    await this.options.onMessage({ ...envelope, uuid: crypto.randomUUID(),
      session_id: this.options.sessionId ?? this.options.resume ?? envelope.session_id, ...message } as Parameters<ClaudeWorkerOptions["onMessage"]>[0]);
  }
  async raw(event: Record<string, unknown>) { await this.emit({ type: "stream_event", event }); }
  async begin() {
    this.responseId = `msg_offline_${++this.responseOrdinal}`;
    await this.raw({ type: "message_start", message: {
      id: this.responseId, type: "message", role: "assistant", model: "offline-model", content: [],
      stop_reason: null, stop_sequence: null, usage: { input_tokens: 3, output_tokens: 0 },
    } });
  }
  mcp(call: ClaudeWorkerToolCall, signal = new AbortController().signal) {
    const pending = this.options.onToolCall(call, signal);
    void pending.catch(() => {});
    return pending;
  }
  async blocks(calls: ClaudeWorkerToolCall[], stop = true) {
    await this.begin();
    for (const [index, call] of calls.entries()) {
      await this.raw({ type: "content_block_start", index, content_block: {
        type: "tool_use", id: call.id, name: CODEBUDDY_MCP_TOOL_PREFIX + call.name, input: {},
      } });
      const json = JSON.stringify(call.arguments);
      const split = Math.floor(json.length / 2);
      for (const partial_json of [json.slice(0, split), json.slice(split)]) {
        await this.raw({ type: "content_block_delta", index, delta: { type: "input_json_delta", partial_json } });
      }
      await this.raw({ type: "content_block_stop", index });
    }
    if (stop) await this.endBatch(calls);
  }
  async endBatch(calls: ClaudeWorkerToolCall[]) {
    await this.raw({ type: "message_delta", delta: { stop_reason: "tool_use", stop_sequence: null }, usage: { output_tokens: 4 } });
    await this.raw({ type: "message_stop" });
    await this.emit({ type: "assistant", message: { role: "assistant", stop_reason: "tool_use", content: calls.map(call => ({
      type: "tool_use", id: call.id, name: CODEBUDDY_MCP_TOOL_PREFIX + call.name, input: call.arguments,
    })) } });
  }
  async finish(text: string, streamed = true) {
    if (streamed) {
      await this.begin();
      await this.raw({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
      await this.raw({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text } });
      await this.raw({ type: "content_block_stop", index: 0 });
      await this.raw({ type: "message_delta", delta: { stop_reason: "end_turn", stop_sequence: null }, usage: { output_tokens: 4 } });
      await this.raw({ type: "message_stop" });
    }
    await this.emit({ type: "assistant", message: { role: "assistant", stop_reason: "end_turn", content: [{ type: "text", text }] } });
    await this.emit({ type: "result", subtype: "success", is_error: false, result: text,
      duration_ms: 1, duration_api_ms: 1, num_turns: 1, total_cost_usd: 0,
      usage: { input_tokens: 3, output_tokens: 4 }, modelUsage: {}, permission_denials: [], stop_reason: "end_turn" });
  }
}

const managers: ClaudeSessionManager[] = [];
afterEach(async () => { await Promise.all(managers.splice(0).map(manager => manager.close())); });
export function harness(journal = new MemoryJournal(), deps: Omit<ClaudeSessionManagerDeps, "journal" | "createWorker"> = {}) {
  const workers: ControlledWorker[] = [];
  const manager = new ClaudeSessionManager({ journal, nativeIO: {
    getSessionMessages: async () => [], forkSession: async () => { throw new Error("No simulated transcript was registered"); },
  }, ...deps, createWorker: async options => {
    const worker = new ControlledWorker(options);
    workers.push(worker);
    return worker;
  } });
  managers.push(manager);
  function run(input: ClaudePersistentTurn, emit?: (event: AdapterEvent) => void) {
    const events: AdapterEvent[] = [];
    const done = manager.run(input, event => { events.push(event); emit?.(event); });
    void done.catch(() => {});
    return { events, done };
  }
  return {
    manager, workers, journal, run,
    async start(input = turn(), index = workers.length) {
      const response = run(input);
      await until(() => !!workers[index]?.sends.length);
      await Bun.sleep(0);
      return { ...response, worker: workers[index]! };
    },
  };
}
