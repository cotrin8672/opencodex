import { describe, expect, test } from "bun:test";
import { getSessionMessages } from "@anthropic-ai/claude-agent-sdk";
import { claudeHistory, claudeSessionScope } from "../../src/adapters/claude-cli/session-context";
import type { ClaudePersistentTurn } from "../../src/adapters/claude-cli/session-manager";
import type { OcxMessage } from "../../src/types";
import { assistant, call, result, toolAssistant, user, withMessages } from "../fixtures/claude-session-worker";
import { gate, nativeBinary, nativeSession, nativeUntil, type NativeBlock } from "../fixtures/claude-session-native-api";

type Harness = ReturnType<typeof nativeSession>;
const text = (text: string): NativeBlock => ({ type: "text", text });
const replies = (...answers: string[]) => answers.map(answer => ({ content: [text(answer)] }));
const occurrences = (body: string, marker: string) => body.split(marker).length - 1;

function operation(h: Harness, messages: OcxMessage[], id: string): ClaudePersistentTurn {
  const input = withMessages(h.input, messages);
  input.incoming = { ...input.incoming, headers: new Headers(input.incoming.headers) };
  input.incoming.headers.set("idempotency-key", id);
  return input;
}

function native(h: Harness, scope = h.input.scope) {
  const record = h.journal.load(scope)?.native;
  expect(record).toBeDefined();
  return record!;
}

function containsOnce(body: string, markers: string[], absent: string[] = []) {
  for (const marker of markers) expect(occurrences(body, marker)).toBe(1);
  for (const marker of absent) expect(body).not.toContain(marker);
}

async function complete(h: Harness, input: ClaudePersistentTurn, answer: string) {
  const response = h.run(input);
  await response.done;
  expect(response.events.filter(event => event.type === "error" || event.type === "tool_call_start")).toEqual([]);
  expect(response.events.filter(event => event.type === "text_delta").map(event => event.text).join("")).toBe(answer);
  expect(response.events.at(-1)).toMatchObject({ type: "done", stopReason: "stop" });
  expect(native(h, input.scope).dirty).toBe(false);
  expect(h.unexpected).toEqual([]);
}

async function transcript(h: Harness, sessionId: string) {
  const messages = await getSessionMessages(sessionId, { dir: h.input.env.HOME });
  expect(messages.length).toBeGreaterThan(0);
  expect(messages.every(message => message.session_id === sessionId)).toBe(true);
  return messages;
}

// Every session uses the real SDK and packaged native CLI; only localhost inference is scripted.
describe.skipIf(!nativeBinary)("Claude native continuity across workers and client histories", () => {
  test("an active native partial followed by a foreign answer forks and imports the full suffix exactly once", async () => {
    const h = nativeSession(replies("ACTIVE_SEED_ANSWER", "ACTIVE_NATIVE_PARTIAL", "ACTIVE_FOREIGN_C_ANSWER"), { nativePersistence: true });
    const held = gate();
    const upstream = h.input.env.ANTHROPIC_BASE_URL!;
    let ordinal = 0;
    // Delay only the second response's terminal SSE frames, keeping the partial output genuinely native.
    const proxy = Bun.serve({ hostname: "127.0.0.1", port: 0, async fetch(request) {
      const url = new URL(request.url);
      const number = url.pathname === "/v1/messages" ? ++ordinal : 0;
      const response = await fetch(new Request(new URL(url.pathname + url.search, upstream), request));
      if (number !== 2) return response;
      const body = await response.text();
      const boundary = body.indexOf("event: message_delta\n");
      expect(boundary).toBeGreaterThan(0);
      let cancelled = false;
      const encoder = new TextEncoder();
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(encoder.encode(body.slice(0, boundary)));
          void held.wait.then(() => {
            if (!cancelled) { controller.enqueue(encoder.encode(body.slice(boundary))); controller.close(); }
          }).catch(() => {});
        },
        cancel() { cancelled = true; },
      });
      const headers = new Headers(response.headers);
      headers.delete("content-length");
      return new Response(stream, { headers });
    } });
    h.input.env.ANTHROPIC_BASE_URL = `http://127.0.0.1:${proxy.port}`;
    try {
      const prefix = [user("ACTIVE_SEED_INPUT")];
      await complete(h, operation(h, prefix, "active-seed"), "ACTIVE_SEED_ANSWER");
      const settled = native(h);
      prefix.push(assistant("ACTIVE_SEED_ANSWER"), user("ACTIVE_NATIVE_A_INPUT"));
      const active = h.run(operation(h, prefix, "active-native-A"));
      await nativeUntil(() => active.events.some(event => event.type === "text_delta" && event.text.includes("ACTIVE_NATIVE_PARTIAL")),
        "native CLI delivers partial A before its terminal result");
      expect(h.requests).toHaveLength(2);
      const history = [...prefix, assistant("ACTIVE_NATIVE_PARTIAL"), assistant("ACTIVE_FOREIGN_ANSWER"), user("ACTIVE_FOREIGN_C_INPUT")];
      await complete(h, operation(h, history, "active-foreign-C"), "ACTIVE_FOREIGN_C_ANSWER");
      await active.done;
      const forked = native(h);
      expect(forked.sessionId).not.toBe(settled.sessionId);
      expect(forked.checkpoints[0]!.messageId).not.toBe(settled.checkpoints[0]!.messageId);
      expect(h.children).toHaveLength(2);
      expect(h.children[0]!.exitCode !== null || h.children[0]!.signalCode !== null).toBe(true);
      expect(h.requests).toHaveLength(3);
      const markers = ["ACTIVE_SEED_INPUT", "ACTIVE_SEED_ANSWER", "ACTIVE_NATIVE_A_INPUT", "ACTIVE_NATIVE_PARTIAL", "ACTIVE_FOREIGN_ANSWER", "ACTIVE_FOREIGN_C_INPUT"];
      containsOnce(JSON.stringify(h.requests[2]!.messages), markers);
      expect(h.requested).toEqual([]);
      await h.manager.close();
      containsOnce(JSON.stringify(await transcript(h, forked.sessionId)), [...markers, "ACTIVE_FOREIGN_C_ANSWER"]);
    } finally { held.release(); try { await h.close(); } finally { await proxy.stop(true); } }
  }, 60_000);

  test("a pending native MCP result plus foreign history forks without repeating the original external action", async () => {
    const script = replies("PENDING_SEED_ANSWER", "unused", "PENDING_FOREIGN_C_ANSWER");
    const h = nativeSession(script, { nativePersistence: true });
    try {
      const prefix = [user("PENDING_SEED_INPUT")];
      await complete(h, operation(h, prefix, "pending-seed"), "PENDING_SEED_ANSWER");
      const settled = native(h);
      prefix.push(assistant("PENDING_SEED_ANSWER"), user("PENDING_FOREIGN_A_INPUT"));
      const action = call(h.input, "toolu_native_foreign_executed", { path: "pending-foreign-original.txt" });
      script[1]!.content = [{ type: "tool_use", id: action.id, name: `mcp__opencodex__${action.name}`, input: action.arguments }];
      const pending = h.run(operation(h, prefix, "pending-foreign-A"));
      await pending.done;
      expect(pending.events.filter(event => event.type === "error")).toEqual([]);
      expect(pending.events.filter(event => event.type === "tool_call_start")).toEqual([{ type: "tool_call_start", id: action.id, name: "read_file" }]);
      expect(native(h).dirty).toBe(true);
      const history = [...prefix, toolAssistant([action]), result(action.id, "PENDING_FOREIGN_ACTUAL_RESULT"),
        assistant("PENDING_FOREIGN_ASSISTANT"), user("PENDING_FOREIGN_C_INPUT")];
      await complete(h, operation(h, history, "pending-foreign-C"), "PENDING_FOREIGN_C_ANSWER");
      const forked = native(h);
      expect(forked.sessionId).not.toBe(settled.sessionId);
      expect(forked.checkpoints[0]!.messageId).not.toBe(settled.checkpoints[0]!.messageId);
      expect(h.children).toHaveLength(2);
      expect(h.children[0]!.exitCode !== null || h.children[0]!.signalCode !== null).toBe(true);
      expect(h.requests).toHaveLength(3);
      expect(h.requested).toEqual([action.id]);
      expect(h.resolved).toEqual([]);
      const markers = ["PENDING_SEED_INPUT", "PENDING_SEED_ANSWER", "PENDING_FOREIGN_A_INPUT", "pending-foreign-original.txt",
        "PENDING_FOREIGN_ACTUAL_RESULT", "PENDING_FOREIGN_ASSISTANT", "PENDING_FOREIGN_C_INPUT"];
      containsOnce(JSON.stringify(h.requests[2]!.messages), markers);
      expect(h.journal.load(h.input.scope)!.turns.flatMap(turn => turn.calls).find(call => call.id === action.id)?.result?.content)
        .toBe("PENDING_FOREIGN_ACTUAL_RESULT");
      await h.manager.close();
      const persisted = await transcript(h, forked.sessionId);
      containsOnce(JSON.stringify(persisted), [...markers, "PENDING_FOREIGN_C_ANSWER"], ["The tool call was interrupted before a result was received"]);
      expect(persisted.flatMap(message => (message.message as { content: Array<{ type: string }> }).content)
        .filter(block => block?.type === "tool_use")).toEqual([]);
    } finally { await h.close(); }
  }, 60_000);

  test("clean A survives restart in the same native session on a new OS process without duplicate context", async () => {
    const h = nativeSession(replies("CLEAN_A_ANSWER", "COLD_C_ANSWER"), { nativePersistence: true });
    try {
      const history = [user("CLEAN_A_INPUT")];
      await complete(h, operation(h, history, "clean-A"), "CLEAN_A_ANSWER");
      const settled = native(h);
      expect(settled.history).toEqual(claudeHistory([...history, assistant("CLEAN_A_ANSWER")]));
      const pid = h.children[0]!.pid;
      expect(pid).toBeGreaterThan(0);
      await h.restart();
      expect(h.children[0]!.exitCode !== null || h.children[0]!.signalCode !== null).toBe(true);
      const persisted = await transcript(h, settled.sessionId);
      expect(persisted.at(-1)!.uuid).toBe(settled.checkpoints.at(-1)!.messageId);
      history.push(assistant("CLEAN_A_ANSWER"), user("COLD_C_CURRENT_INPUT"));
      await complete(h, operation(h, history, "cold-C"), "COLD_C_ANSWER");
      expect(native(h).sessionId).toBe(settled.sessionId);
      expect(h.children).toHaveLength(2);
      expect(h.children[1]!.pid).toBeGreaterThan(0);
      expect(h.children[1]!.pid).not.toBe(pid);
      expect(h.requests).toHaveLength(2);
      containsOnce(JSON.stringify(h.requests[1]!.messages), ["CLEAN_A_INPUT", "CLEAN_A_ANSWER", "COLD_C_CURRENT_INPUT"]);
      await h.manager.close();
      containsOnce(JSON.stringify(await transcript(h, settled.sessionId)),
        ["CLEAN_A_INPUT", "CLEAN_A_ANSWER", "COLD_C_CURRENT_INPUT", "COLD_C_ANSWER"]);
    } finally { await h.close(); }
  }, 60_000);

  test("GPT and Gemini full-history catch-up imports only missing markers across warm and cold resumes", async () => {
    const h = nativeSession(replies("MIX_CLAUDE_A", "MIX_CLAUDE_B_ANSWER", "MIX_CLAUDE_C_ANSWER"), { nativePersistence: true });
    try {
      const history = [user("MIX_ORIGINAL_INPUT")];
      await complete(h, operation(h, history, "mixed-A"), "MIX_CLAUDE_A");
      const id = native(h).sessionId;
      history.push(assistant("MIX_CLAUDE_A"), user("MIX_GPT_QUESTION"), assistant("MIX_GPT_ANSWER"), user("MIX_CLAUDE_B_INPUT"));
      await complete(h, operation(h, [...history], "mixed-B"), "MIX_CLAUDE_B_ANSWER");
      expect(native(h).sessionId).toBe(id);
      expect(h.children).toHaveLength(1);
      const retained = ["MIX_ORIGINAL_INPUT", "MIX_CLAUDE_A", "MIX_GPT_QUESTION", "MIX_GPT_ANSWER", "MIX_CLAUDE_B_INPUT"];
      containsOnce(JSON.stringify(h.requests[1]!.messages), retained);
      history.push(assistant("MIX_CLAUDE_B_ANSWER"), user("MIX_GEMINI_QUESTION"), assistant("MIX_GEMINI_ANSWER"), user("MIX_CLAUDE_C_INPUT"));
      await h.restart();
      await complete(h, operation(h, history, "mixed-C"), "MIX_CLAUDE_C_ANSWER");
      expect(native(h).sessionId).toBe(id);
      expect(h.children).toHaveLength(2);
      expect(h.requests).toHaveLength(3);
      retained.push("MIX_CLAUDE_B_ANSWER", "MIX_GEMINI_QUESTION", "MIX_GEMINI_ANSWER", "MIX_CLAUDE_C_INPUT");
      containsOnce(JSON.stringify(h.requests[2]!.messages), retained);
      await h.manager.close();
      containsOnce(JSON.stringify(await transcript(h, id)), [...retained, "MIX_CLAUDE_C_ANSWER"]);
    } finally { await h.close(); }
  }, 60_000);

  test("editing and Undo fork from the earlier clean checkpoint and exclude discarded later answers", async () => {
    const h = nativeSession(replies("BRANCH_A_ANSWER", "DISCARDED_B_ANSWER", "EDITED_B_ANSWER", "UNDO_C_ANSWER"), { nativePersistence: true });
    try {
      const prefix = [user("BRANCH_A_INPUT")];
      await complete(h, operation(h, prefix, "branch-A"), "BRANCH_A_ANSWER");
      prefix.push(assistant("BRANCH_A_ANSWER"));
      const first = native(h);
      await complete(h, operation(h, [...prefix, user("DISCARDED_B_INPUT")], "branch-B"), "DISCARDED_B_ANSWER");
      expect(native(h).sessionId).toBe(first.sessionId);
      expect(native(h).checkpoints).toHaveLength(2);
      await h.restart();
      await complete(h, operation(h, [...prefix, user("EDITED_B_INPUT")], "branch-edit"), "EDITED_B_ANSWER");
      const edited = native(h);
      expect(edited.sessionId).not.toBe(first.sessionId);
      expect(edited.checkpoints[0]!.length).toBe(2);
      expect(edited.checkpoints[0]!.messageId).not.toBe(first.checkpoints[0]!.messageId);
      containsOnce(JSON.stringify(h.requests[2]!.messages), ["BRANCH_A_INPUT", "BRANCH_A_ANSWER", "EDITED_B_INPUT"],
        ["DISCARDED_B_INPUT", "DISCARDED_B_ANSWER"]);
      await complete(h, operation(h, [...prefix, user("UNDO_C_INPUT")], "branch-undo"), "UNDO_C_ANSWER");
      const undone = native(h);
      expect(new Set([first.sessionId, edited.sessionId, undone.sessionId]).size).toBe(3);
      expect(undone.checkpoints[0]!.length).toBe(2);
      expect(undone.checkpoints[0]!.messageId).not.toBe(edited.checkpoints[0]!.messageId);
      containsOnce(JSON.stringify(h.requests[3]!.messages), ["BRANCH_A_INPUT", "BRANCH_A_ANSWER", "UNDO_C_INPUT"],
        ["DISCARDED_B_INPUT", "DISCARDED_B_ANSWER", "EDITED_B_INPUT", "EDITED_B_ANSWER"]);
      expect(h.requests).toHaveLength(4);
      await h.manager.close();
      containsOnce(JSON.stringify(await transcript(h, undone.sessionId)),
        ["BRANCH_A_INPUT", "BRANCH_A_ANSWER", "UNDO_C_INPUT", "UNDO_C_ANSWER"], ["DISCARDED_B_ANSWER", "EDITED_B_ANSWER"]);
      expect(JSON.stringify(await transcript(h, first.sessionId))).toContain("DISCARDED_B_ANSWER");
    } finally { await h.close(); }
  }, 60_000);

  test("compacted history with no common prefix rebuilds under a fresh native session ID", async () => {
    const h = nativeSession(replies("PRECOMPACTION_ANSWER", "COMPACTED_ANSWER"), { nativePersistence: true });
    try {
      await complete(h, operation(h, [user("PRECOMPACTION_INPUT")], "precompaction"), "PRECOMPACTION_ANSWER");
      const previous = native(h).sessionId;
      await h.restart();
      const compacted = [user("COMPACTED_SUMMARY_INPUT"), user("COMPACTED_CURRENT_INPUT")];
      await complete(h, operation(h, compacted, "compacted"), "COMPACTED_ANSWER");
      const rebuilt = native(h);
      expect(rebuilt.sessionId).not.toBe(previous);
      expect(rebuilt.checkpoints).toHaveLength(1);
      expect(rebuilt.history).toEqual(claudeHistory([...compacted, assistant("COMPACTED_ANSWER")]));
      expect(h.requests).toHaveLength(2);
      containsOnce(JSON.stringify(h.requests[1]!.messages), ["COMPACTED_SUMMARY_INPUT", "COMPACTED_CURRENT_INPUT"],
        ["PRECOMPACTION_INPUT", "PRECOMPACTION_ANSWER"]);
      await h.manager.close();
      containsOnce(JSON.stringify(await transcript(h, rebuilt.sessionId)),
        ["COMPACTED_SUMMARY_INPUT", "COMPACTED_CURRENT_INPUT", "COMPACTED_ANSWER"], ["PRECOMPACTION_INPUT", "PRECOMPACTION_ANSWER"]);
    } finally { await h.close(); }
  }, 60_000);

  test("independent thread scopes own distinct native IDs and isolated transcripts", async () => {
    const h = nativeSession(replies("THREAD_LEFT_ANSWER", "THREAD_RIGHT_ANSWER"), { nativePersistence: true });
    try {
      const left = operation(h, [user("THREAD_LEFT_INPUT")], "thread-left");
      const right = operation(h, [user("THREAD_RIGHT_INPUT")], "thread-right");
      right.incoming.headers.set("thread-id", "independent-right-thread");
      right.scope = claudeSessionScope(right.parsed, right.incoming, right.provider)!;
      expect(right.scope).not.toBe(left.scope);
      await complete(h, left, "THREAD_LEFT_ANSWER");
      await complete(h, right, "THREAD_RIGHT_ANSWER");
      const leftId = native(h, left.scope).sessionId;
      const rightId = native(h, right.scope).sessionId;
      expect(leftId).not.toBe(rightId);
      expect(h.children).toHaveLength(2);
      expect(h.requests).toHaveLength(2);
      containsOnce(JSON.stringify(h.requests[1]!.messages), ["THREAD_RIGHT_INPUT"], ["THREAD_LEFT_INPUT", "THREAD_LEFT_ANSWER"]);
      await h.manager.close();
      containsOnce(JSON.stringify(await transcript(h, leftId)), ["THREAD_LEFT_INPUT", "THREAD_LEFT_ANSWER"], ["THREAD_RIGHT_INPUT"]);
      containsOnce(JSON.stringify(await transcript(h, rightId)), ["THREAD_RIGHT_INPUT", "THREAD_RIGHT_ANSWER"], ["THREAD_LEFT_INPUT"]);
    } finally { await h.close(); }
  }, 60_000);

  test("pending external tools survive cold restart without reissue and actual results import after a safe dirty fork", async () => {
    const script = replies("TOOL_SAFE_A_ANSWER", "unused", "TOOL_RECOVERED_ANSWER");
    const h = nativeSession(script, { nativePersistence: true });
    try {
      const prefix = [user("TOOL_SAFE_A_INPUT")];
      await complete(h, operation(h, prefix, "tool-safe-A"), "TOOL_SAFE_A_ANSWER");
      const settled = native(h);
      prefix.push(assistant("TOOL_SAFE_A_ANSWER"), user("TOOL_PENDING_REQUEST"));
      const action = call(h.input, "toolu_continuity_pending", { path: "pending-original.txt" });
      script[1]!.content = [{ type: "tool_use", id: action.id, name: `mcp__opencodex__${action.name}`, input: action.arguments }];
      const pendingInput = operation(h, prefix, "tool-pending");
      const pending = h.run(pendingInput);
      await pending.done;
      expect(pending.events.filter(event => event.type === "error")).toEqual([]);
      expect(pending.events.filter(event => event.type === "tool_call_start")).toEqual([{ type: "tool_call_start", id: action.id, name: "read_file" }]);
      expect(pending.events.at(-1)).toMatchObject({ type: "done", stopReason: "tool_use", endTurn: false });
      expect(native(h).dirty).toBe(true);
      expect(native(h).checkpoints).toEqual(settled.checkpoints);
      expect(native(h).history).toEqual(settled.history);
      expect(h.requests).toHaveLength(2);
      await h.restart();
      expect(h.requests).toHaveLength(2);
      expect(native(h).dirty).toBe(true);
      const reconnect = h.run(pendingInput);
      await expect(reconnect.done).rejects.toMatchObject({ code: "claude_tool_delivery_uncertain" });
      expect(reconnect.events.filter(event => event.type === "tool_call_start")).toEqual([]);
      expect(h.requests).toHaveLength(2);
      expect(h.requested).toEqual([action.id]);
      const recovered = [...prefix, toolAssistant([action]), result(action.id, "ACTUAL_COLD_EXTERNAL_RESULT")];
      await complete(h, operation(h, recovered, "tool-actual-result"), "TOOL_RECOVERED_ANSWER");
      const forked = native(h);
      expect(forked.sessionId).not.toBe(settled.sessionId);
      expect(forked.checkpoints[0]!.messageId).not.toBe(settled.checkpoints[0]!.messageId);
      expect(h.requested).toEqual([action.id]);
      expect(h.resolved).toEqual([]);
      expect(h.requests).toHaveLength(3);
      const discarded = ["The tool call was interrupted before a result was received"];
      containsOnce(JSON.stringify(h.requests.at(-1)!.messages),
        ["TOOL_SAFE_A_INPUT", "TOOL_SAFE_A_ANSWER", "TOOL_PENDING_REQUEST", "pending-original.txt", "ACTUAL_COLD_EXTERNAL_RESULT"], discarded);
      expect(h.journal.load(h.input.scope)!.turns.flatMap(turn => turn.calls).find(call => call.id === action.id)?.result?.content)
        .toBe("ACTUAL_COLD_EXTERNAL_RESULT");
      await h.manager.close();
      const persisted = await transcript(h, forked.sessionId);
      containsOnce(JSON.stringify(persisted), ["ACTUAL_COLD_EXTERNAL_RESULT", "TOOL_RECOVERED_ANSWER"], discarded);
      expect(persisted.flatMap(message => (message.message as { content: Array<{ type: string }> }).content)
        .filter(block => block?.type === "tool_use")).toEqual([]);
      expect(JSON.stringify(await transcript(h, settled.sessionId))).toContain(action.id);
    } finally { await h.close(); }
  }, 60_000);
});
