import { describe, expect, test } from "bun:test";
import type { AdapterEvent, OcxMessage } from "../../src/types";
import { buildCodeBuddyToolBridge } from "../../src/adapters/codebuddy/tool-bridge";
import { stableClaudeToolBridge } from "../../src/adapters/claude-cli/stable-replay";
import { assistant, call, result, toolAssistant, user, withMessages } from "../fixtures/claude-session-worker";
import { gate, nativeBinary, nativeSession, nativeUntil, type NativeBlock } from "../fixtures/claude-session-native-api";

const text = (text: string): NativeBlock => ({ type: "text", text });
const calls = (events: AdapterEvent[]) => events.filter(event => event.type === "tool_call_start");
const output = (events: AdapterEvent[]) => events.filter(event => event.type === "text_delta").map(event => event.text).join("");
function final(events: AdapterEvent[], answer: string) {
  expect(events.filter(event => event.type === "error")).toEqual([]);
  expect(calls(events)).toEqual([]);
  expect(output(events)).toBe(answer);
  expect(events.at(-1)).toMatchObject({ type: "done" });
}

// Skip only the optional native package; an installed but broken binary must fail.
describe.skipIf(!nativeBinary)("Claude sessions through SDK 0.3.314 and local native API", () => {
  test.each([2, 3])("hands off %i unsafe tools one at a time, then continues and retains the next user history", async count => {
    const replies = [{ content: [] as NativeBlock[] }, { content: [text("all files complete")] }, { content: [text("follow-up complete")] }];
    const h = nativeSession(replies);
    try {
      const actions = Array.from({ length: count }, (_, index) => call(h.input, `toolu_native_${index}`, { path: `${index}.txt` }));
      replies[0]!.content = actions.map(action => ({ type: "tool_use", id: action.id,
        name: `mcp__opencodex__${action.name}`, input: action.arguments }));
      for (const tool of h.input.bridge.tools) expect(tool).not.toHaveProperty("annotations");
      let history: OcxMessage[] = [...h.input.parsed.context.messages];
      let response = h.run(h.input);
      const delivered: string[] = [];
      for (const [index, action] of actions.entries()) {
        await response.done;
        expect(response.events.filter(event => event.type === "error")).toEqual([]);
        expect(calls(response.events)).toEqual([{ type: "tool_call_start", id: action.id, name: "read_file" }]);
        expect(response.events.filter(event => event.type === "tool_call_delta").map(event => event.arguments).join(""))
          .toBe(JSON.stringify(action.arguments));
        expect(response.events.filter(event => event.type === "tool_call_end")).toHaveLength(1);
        expect(response.events.at(-1)).toMatchObject({ type: "done", stopReason: "tool_use", endTurn: false });
        delivered.push(action.id);
        // All calls were generated together, but B cannot reach MCP before A's client result.
        expect(h.requested).toEqual(actions.slice(0, index + 1).map(action => action.id));
        expect(h.resolved).toEqual(actions.slice(0, index).map(action => action.id));
        expect(h.requests).toHaveLength(1);
        history.push(toolAssistant([action]), result(action.id, `RESULT_${index}`));
        response = h.run(withMessages(h.input, [...history]));
      }
      await response.done;
      final(response.events, "all files complete");
      expect(delivered).toEqual(actions.map(action => action.id));
      expect(new Set(delivered).size).toBe(count);
      expect(h.requested).toEqual(delivered);
      expect(h.resolved).toEqual(delivered);
      expect(h.maxActive).toBe(1);
      expect(h.requests).toHaveLength(2);
      const continuation = JSON.stringify(h.requests[1]!.messages);
      for (const [index, action] of actions.entries()) {
        expect(continuation).toContain(action.id);
        expect(continuation).toContain(`RESULT_${index}`);
      }
      history.push(assistant("all files complete"), user("ask again with the previous results"));
      const next = h.run(withMessages(h.input, history));
      await next.done;
      final(next.events, "follow-up complete");
      expect(h.requests).toHaveLength(3);
      const nextHistory = JSON.stringify(h.requests[2]!.messages);
      for (const item of ["read both files", "all files complete", "ask again with the previous results", "RESULT_0"]) expect(nextHistory).toContain(item);
      expect(h.children).toHaveLength(1);
      expect(h.children[0]!.exitCode).toBeNull();
      expect(h.children[0]!.signalCode).toBeNull();
      expect(h.unexpected).toEqual([]);
    } finally { await h.close(); }
  }, 45_000);

  test("same-input reconnect keeps the active native generation", async () => {
    const held = gate();
    const h = nativeSession([{ content: [text("reconnected answer")], wait: held.wait }]);
    try {
      const abort = new AbortController();
      const first = h.run({ ...h.input, incoming: { ...h.input.incoming, abortSignal: abort.signal } });
      await nativeUntil(() => h.requests.length === 1, "first inference begins");
      abort.abort();
      await first.done;
      const reconnect = h.run(h.input);
      held.release();
      await reconnect.done;
      final(reconnect.events, "reconnected answer");
      expect(first.events).toEqual([]);
      expect(h.requests).toHaveLength(1);
      expect(h.children).toHaveLength(1);
      expect(h.unexpected).toEqual([]);
    } finally { held.release(); await h.close(); }
  }, 45_000);

  test.each([false, true])("newer input interrupts the active turn while preserving native history (disconnected=%s)", async disconnected => {
    const held = gate();
    const h = nativeSession([
      { content: [text("earlier answer")] },
      { content: [text("STALE_OLD_GENERATION")], wait: held.wait },
      { content: [text("replacement answer")] },
    ]);
    try {
      const earlier = h.run(h.input);
      await earlier.done;
      final(earlier.events, "earlier answer");
      const history = [...h.input.parsed.context.messages, assistant("earlier answer"), user("old slow instruction")];
      const abort = new AbortController();
      const slowInput = withMessages(h.input, history);
      const old = h.run({ ...slowInput, incoming: { ...slowInput.incoming, abortSignal: abort.signal } });
      await nativeUntil(() => h.requests.length === 2, "old native inference waits for API");
      expect(h.children).toHaveLength(1);
      if (disconnected) { abort.abort(); await old.done; }
      const replacement = h.run(withMessages(h.input, [...history, user("new urgent instruction")]));
      await replacement.done;
      await old.done;
      final(replacement.events, "replacement answer");
      expect(h.children[0]!.exitCode).toBeNull();
      expect(h.children[0]!.signalCode).toBeNull();
      expect(h.children).toHaveLength(1);
      expect(h.requests).toHaveLength(3);
      const bootstrap = JSON.stringify(h.requests[2]!.messages);
      for (const item of ["read both files", "earlier answer", "old slow instruction", "new urgent instruction"]) expect(bootstrap).toContain(item);
      const eventsBeforeLateReply = structuredClone(replacement.events);
      held.release();
      await nativeUntil(() => h.replied.includes(2), "late old API response is released");
      expect(replacement.events).toEqual(eventsBeforeLateReply);
      expect(output([...old.events, ...replacement.events])).not.toContain("STALE_OLD_GENERATION");
      expect(h.requests).toHaveLength(3);
      expect(h.unexpected).toEqual([]);
    } finally { held.release(); await h.close(); }
  }, 45_000);

  test("a different caller turn identity can regenerate identical input", async () => {
    const h = nativeSession([{ content: [text("first answer")] }, { content: [text("regenerated answer")] }]);
    try {
      const identified = (id: string) => ({ ...h.input, incoming: { ...h.input.incoming,
        headers: new Headers({ "thread-id": "offline-thread", "x-codex-turn-metadata": JSON.stringify({ turn_id: id }) }),
      } });
      const first = h.run(identified("turn-first"));
      await first.done;
      final(first.events, "first answer");
      const retry = h.run(identified("turn-first"));
      await retry.done;
      final(retry.events, "first answer");
      expect(h.requests).toHaveLength(1);
      const regenerated = h.run(identified("turn-second"));
      await regenerated.done;
      final(regenerated.events, "regenerated answer");
      expect(h.requests).toHaveLength(2);
      expect(JSON.stringify(h.requests[1]!.messages)).toContain("read both files");
      expect(h.unexpected).toEqual([]);
    } finally { await h.close(); }
  }, 45_000);

  test("a model switch bootstraps earlier context and namespaced tool history", async () => {
    const replies = [{ content: [] as NativeBlock[] }, { content: [text("new model answer")] }];
    const h = nativeSession(replies);
    try {
      h.input.parsed.context.tools![0]!.namespace = "client_files";
      h.input.bridge = stableClaudeToolBridge(buildCodeBuddyToolBridge(h.input.parsed));
      const action = call(h.input, "toolu_before_model_switch", { path: "retained.txt" });
      const toolUse: NativeBlock = { type: "tool_use", id: action.id, name: `mcp__opencodex__${action.name}`, input: action.arguments };
      // This tool really reaches native MCP and the client before its result triggers the model change.
      replies[0]!.content = [toolUse];
      const history = [...h.input.parsed.context.messages];
      const first = h.run(h.input);
      await first.done;
      expect(calls(first.events)).toEqual([{ type: "tool_call_start", id: action.id, name: "client_files__read_file" }]);
      history.push({ role: "assistant", timestamp: 1, content: [{ type: "toolCall", id: action.id,
        namespace: "client_files", name: "read_file", arguments: action.arguments }] }, result(action.id, "RETAINED_TOOL_OUTPUT"));
      const switched = withMessages(h.input, history);
      switched.parsed.modelId = "claude-sonnet-4-6";
      const next = h.run(switched);
      await next.done;
      final(next.events, "new model answer");
      expect(h.children).toHaveLength(2);
      await nativeUntil(() => h.children[0]!.exitCode !== null || h.children[0]!.signalCode !== null, "old model child exits");
      expect(h.requests).toHaveLength(2);
      expect(h.requests.map(request => request.model)).toEqual(["claude-opus-4-6", "claude-sonnet-4-6"]);
      const bootstrap = JSON.stringify(h.requests[1]!.messages);
      for (const item of ["read both files", "retained.txt", action.id, toolUse.name, "RETAINED_TOOL_OUTPUT"]) expect(bootstrap).toContain(item);
      expect(h.requested).toEqual([action.id]);
      expect(h.unexpected).toEqual([]);
    } finally { await h.close(); }
  }, 45_000);

  test.each(["same-input", "new-input"] as const)("generation timeout forks safely and recovers with %s", async mode => {
    const held = gate();
    const h = nativeSession([
      { content: [text("CLEAN_A")] },
      { content: [text("NEVER_DELIVERED_B")], wait: held.wait },
      { content: [text("RECOVERED")] },
    ], { nativePersistence: true });
    const identified = (messages: OcxMessage[], operation: string) => {
      const next = withMessages(h.input, messages);
      next.incoming = { ...next.incoming, headers: new Headers(next.incoming.headers) };
      next.incoming.headers.set("idempotency-key", operation);
      return next;
    };
    try {
      const initial = [user("A initial question")];
      const seed = h.run(identified(initial, "A"));
      await seed.done;
      final(seed.events, "CLEAN_A");
      const priorNative = h.journal.load(h.input.scope)!.native!;
      const checkpoint = priorNative.checkpoints.at(-1)!;
      expect(priorNative.dirty).toBe(false);
      expect(checkpoint).toBeDefined();

      (h.manager as any).deps.timeoutMs = 300;
      const history = [...initial, assistant("CLEAN_A"), user("B slow request")];
      const slowInput = identified(history, "B");
      const slow = h.run(slowInput);
      await nativeUntil(() => h.requests.length === 2, "B dispatches before its inactivity timeout", 2_000);
      await nativeUntil(() => slow.events.at(-1)?.type === "error", "B reaches its terminal timeout", 2_000);
      await slow.done;
      expect(slow.events.at(-1)).toMatchObject({ type: "error", code: "claude_session_timeout", status: 504, retryable: true });
      expect(h.journal.load(h.input.scope)!.native!.dirty).toBe(true);
      expect(h.requests).toHaveLength(2);

      (h.manager as any).deps.timeoutMs = 8_000;
      const retryInput = mode === "same-input" ? slowInput : identified([...history, user("C recover and answer")], "C");
      const retry = h.run(retryInput);
      await nativeUntil(() => retry.events.some(event => event.type === "done" || event.type === "error"), "native retry terminates", 10_000);
      await retry.done;
      final(retry.events, "RECOVERED");
      expect(h.children).toHaveLength(2);
      await nativeUntil(() => h.children[0]!.exitCode !== null || h.children[0]!.signalCode !== null, "timed-out child exits");
      expect(h.requests).toHaveLength(3);
      const recoveredNative = h.journal.load(h.input.scope)!.native!;
      expect(recoveredNative.sessionId).not.toBe(priorNative.sessionId);
      expect(recoveredNative.dirty).toBe(false);
      const remapped = recoveredNative.checkpoints.find(point => point.length === checkpoint.length);
      expect(remapped).toBeDefined();
      expect(remapped!.messageId).not.toBe(checkpoint.messageId);
      const bootstrap = JSON.stringify(h.requests[2]!.messages);
      for (const item of ["A initial question", "CLEAN_A", "B slow request"]) expect(bootstrap).toContain(item);
      if (mode === "new-input") expect(bootstrap).toContain("C recover and answer");
      expect(bootstrap).not.toContain("NEVER_DELIVERED_B");

      const beforeLateReply = structuredClone(retry.events);
      held.release();
      await nativeUntil(() => h.replied.includes(2), "late B API response is released");
      expect(retry.events).toEqual(beforeLateReply);
      expect(output([...slow.events, ...retry.events])).not.toContain("NEVER_DELIVERED_B");
      const replay = h.run(retryInput);
      await replay.done;
      final(replay.events, "RECOVERED");
      expect(h.requests).toHaveLength(3);
      expect(h.children).toHaveLength(2);
      expect(h.requested).toEqual([]);
      expect(h.unexpected).toEqual([]);
    } finally { held.release(); await h.close(); }
  }, 30_000);

  test("queued native instruction wait ends explicitly and the original MCP resumes on the same child", async () => {
    const replies = [{ content: [] as NativeBlock[] }, { content: [text("original result and saved B applied")] }];
    const h = nativeSession(replies, { nativePersistence: true });
    const action = call(h.input, "toolu_native_delayed_result", { path: "original.txt" });
    replies[0]!.content = [{ type: "tool_use", id: action.id, name: `mcp__opencodex__${action.name}`, input: action.arguments }];
    try {
      const first = h.run(h.input);
      await first.done;
      expect(calls(first.events)).toEqual([{ type: "tool_call_start", id: action.id, name: "read_file" }]);
      expect(first.events.at(-1)).toMatchObject({ type: "done", stopReason: "tool_use" });
      const original = h.children[0]!;
      const history = [...h.input.parsed.context.messages, toolAssistant([action])];
      const instructionB = "NATIVE_SAVED_B_AFTER_MISSING_RESULT_150";
      (h.manager as any).deps.setupTimeoutMs = 150;
      const waiting = h.run(withMessages(h.input, [...history, user(instructionB)]));
      await nativeUntil(() => waiting.events.some(event => event.type === "error"), "B receives a bounded pending-result error", 2_000);
      await waiting.done;
      expect(waiting.events.filter(event => event.type === "error")).toEqual([
        expect.objectContaining({ code: "claude_tool_results_pending", status: 409, retryable: false }),
      ]);
      expect(calls(waiting.events)).toEqual([]);
      expect(h.journal.load(h.input.scope)!.instructions).toEqual([
        expect.objectContaining({ phase: "queued", texts: [instructionB] }),
      ]);
      expect(h.requested).toEqual([action.id]);
      expect(h.resolved).toEqual([]);
      expect(h.requests).toHaveLength(1);
      expect(h.children).toHaveLength(1);
      expect(original.exitCode).toBeNull();
      expect(original.signalCode).toBeNull();
      expect(h.interrupts).toBe(0);

      (h.manager as any).deps.setupTimeoutMs = 5_000;
      const actualResult = "ACTUAL_ORIGINAL_NATIVE_TOOL_RESULT";
      const continuation = withMessages(h.input, [...history, result(action.id, actualResult)]);
      continuation.incoming = { ...continuation.incoming, headers: new Headers(continuation.incoming.headers) };
      continuation.incoming.headers.set("idempotency-key", "native-delayed-result-C");
      expect(JSON.stringify(continuation.parsed.context.messages)).not.toContain(instructionB);
      const resumed = h.run(continuation);
      await nativeUntil(() => resumed.events.some(event => event.type === "done" || event.type === "error"), "C applies the original result and saved B", 8_000);
      await resumed.done;
      final(resumed.events, "original result and saved B applied");
      const messages = JSON.stringify(h.requests[1]!.messages);
      expect(messages.split(instructionB)).toHaveLength(2);
      expect(messages.split(actualResult)).toHaveLength(2);
      expect(messages).toContain(action.id);
      expect(h.journal.load(h.input.scope)!.instructions).toEqual([
        expect.objectContaining({ phase: "completed", texts: [instructionB] }),
      ]);
      const replay = h.run(continuation);
      await replay.done;
      final(replay.events, "original result and saved B applied");
      expect(h.requested).toEqual([action.id]);
      expect(h.resolved).toEqual([action.id]);
      expect(h.maxActive).toBe(1);
      expect(h.requests).toHaveLength(2);
      expect(h.children).toHaveLength(1);
      expect(h.children[0]).toBe(original);
      expect(original.exitCode).toBeNull();
      expect(original.signalCode).toBeNull();
      expect(h.interrupts).toBe(0);
      expect(h.unexpected).toEqual([]);
    } finally { await h.close(); }
  }, 30_000);
});
