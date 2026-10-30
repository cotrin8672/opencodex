import { expect, test } from "bun:test";
import { claudeDigest, claudeHistory, claudeResult } from "../../src/adapters/claude-cli/session-context";
import type { ClaudeSessionSnapshot, ClaudeTurnReceipt } from "../../src/adapters/claude-cli/session-journal";
import { MemoryJournal, call, harness, result, toolAssistant, turn, user, withMessages } from "../fixtures/claude-session-worker";

const MAX_RECORD_BYTES = 16 * 1024 * 1024;
const scope = claudeDigest("retention regression");
const bytes = (snapshot: ClaudeSessionSnapshot): number => Buffer.byteLength(JSON.stringify(snapshot));

function completed(index: number): ClaudeTurnReceipt {
  return {
    inputHash: claudeDigest(index), phase: "completed", events: [], updatedAt: 0, deliveryStarted: true,
    calls: [{ id: `call-${index}`, name: "read_file", arguments: "{}", result: claudeResult(result(`call-${index}`, "saved output")) }],
  };
}

// Exercise retention directly so the byte boundary and failed write do not depend on timing or hashes.
type RetentionSession = {
  scope: string;
  snapshot: ClaudeSessionSnapshot;
  pending: Map<string, { settled: boolean }>;
  active?: { receipt: ClaudeTurnReceipt };
};
function save(h: ReturnType<typeof harness>, session: RetentionSession, pinned: ClaudeTurnReceipt[] = []): void {
  (h.manager as unknown as {
    save(session: RetentionSession, snapshot: ClaudeSessionSnapshot, pinned: ClaudeTurnReceipt[]): void;
  }).save(session, session.snapshot, pinned);
}

function savedRecovery() {
  const journal = new MemoryJournal();
  const input = turn();
  const action = call(input, "already_executed", { path: "one-call.txt" });
  const origin = toolAssistant([action]);
  const saved: ClaudeSessionSnapshot = {
    version: 1, history: claudeHistory([...input.parsed.context.messages, origin]),
    turns: [{
      inputHash: claudeDigest("original tool request"), phase: "tools", events: [], updatedAt: 0, deliveryStarted: true,
      calls: [{ id: action.id, name: "read_file", arguments: JSON.stringify(action.arguments),
        result: claudeResult(result(action.id, "UNIQUE_COMMITTED_RECOVERY_OUTPUT")) }],
    }],
  };
  journal.save(input.scope, saved);
  const rebased = withMessages(input, [...input.parsed.context.messages, origin, user("continue")]);
  return { journal, input, action, rebased };
}

test("identical raw-history reconnect keeps the restored generation and its duplicate-action guard", async () => {
  const { journal, input, action, rebased } = savedRecovery();
  const h = harness(journal);
  const recovered = await h.start(rebased);
  expect(recovered.worker.sends[0]).toContain("UNIQUE_COMMITTED_RECOVERY_OUTPUT");
  expect(journal.load(input.scope)!.turns.at(-1)!.requestHash).toMatch(/^[0-9a-f]{64}$/);
  const reconnect = h.run(rebased);
  await recovered.done;
  expect(h.workers).toHaveLength(1);
  expect(recovered.worker.stops).toBe(0);
  expect(recovered.worker.sends).toHaveLength(1);
  const repeated = { ...action, id: "fresh_repeated_id" };
  recovered.worker.mcp(repeated);
  await recovered.worker.blocks([repeated]);
  await reconnect.done;
  expect(reconnect.events.some(event => event.type === "tool_call_start")).toBe(false);
  expect(reconnect.events.at(-1)).toMatchObject({ type: "error", code: "claude_recovery_repeat_refused" });
});

test("completed restored responses replay for the raw request both before and after restart", async () => {
  const { journal, rebased } = savedRecovery();
  const h = harness(journal);
  const recovered = await h.start(rebased);
  await recovered.worker.finish("recovered final answer");
  await recovered.done;
  const replay = h.run(rebased);
  await replay.done;
  expect(replay.events).toContainEqual({ type: "text_delta", text: "recovered final answer" });
  expect(h.workers).toHaveLength(1);
  expect(recovered.worker.sends).toHaveLength(1);
  h.manager.close();
  const restarted = harness(journal);
  const coldReplay = restarted.run(rebased);
  await coldReplay.done;
  expect(coldReplay.events).toContainEqual({ type: "text_delta", text: "recovered final answer" });
  expect(restarted.workers).toHaveLength(0);
});

test("a crashed restored continuation retains omitted output and duplicate protection on retry", async () => {
  const { journal, input, action, rebased } = savedRecovery();
  const h = harness(journal);
  const recovered = await h.start(rebased);
  recovered.worker.options.onError(new Error("offline continuation crash"));
  await recovered.done;
  expect(journal.load(input.scope)!.turns[0]!.phase).toBe("completed");
  const retried = await h.start(rebased);
  expect(retried.worker.sends[0]).toContain("UNIQUE_COMMITTED_RECOVERY_OUTPUT");
  const repeated = { ...action, id: "fresh_id_after_crash" };
  retried.worker.mcp(repeated);
  await retried.worker.blocks([repeated]);
  await retried.done;
  expect(retried.events.some(event => event.type === "tool_call_start")).toBe(false);
  expect(retried.events.at(-1)).toMatchObject({ type: "error", code: "claude_recovery_repeat_refused" });
});

test("a failed retention write preserves live receipts, call references and pending entries until commit", () => {
  const h = harness();
  const snapshot: ClaudeSessionSnapshot = { version: 1, history: [], turns: Array.from({ length: 17 }, (_, index) => completed(index)) };
  const oldest = snapshot.turns[0]!;
  const calls = oldest.calls;
  const pending = { settled: true };
  const session: RetentionSession = { scope, snapshot, pending: new Map([[calls[0]!.id, pending]]) };
  const before = structuredClone(snapshot);
  h.journal.save(scope, snapshot);
  h.journal.rejectSave = () => true;
  expect(() => save(h, session)).toThrow("injected journal failure");
  expect(session.snapshot).toBe(snapshot);
  expect(session.snapshot).toEqual(before);
  expect(oldest.calls).toBe(calls);
  expect(session.pending.get(calls[0]!.id)).toBe(pending);
  expect(h.journal.load(scope)).toEqual(before);
  h.journal.rejectSave = undefined;
  save(h, session);
  expect(session.snapshot.turns[0]).toBe(oldest);
  expect(oldest.phase).toBe("expired");
  expect(oldest.calls).toEqual([]);
  expect(session.pending.size).toBe(0);
  expect(h.journal.load(scope)).toEqual(session.snapshot);
});

test.each(["pinned turn", "unresolved MCP entry"] as const)("16 MiB retention cannot erase an undelivered result protected by %s", protection => {
  const h = harness();
  const pendingTurn = completed(0);
  pendingTurn.phase = "tools";
  const running: ClaudeTurnReceipt = { inputHash: claudeDigest("active continuation"), phase: "running", calls: [], events: [], updatedAt: 0 };
  const snapshot: ClaudeSessionSnapshot = { version: 1, history: [], turns: [pendingTurn, running] };
  const output = pendingTurn.calls[0]!.result!;
  output.content = "";
  output.content = "x".repeat(MAX_RECORD_BYTES - bytes(snapshot) - 1);
  const contentLength = output.content.length;
  output.hash = claudeResult(result(pendingTurn.calls[0]!.id, output.content)).hash;
  expect(bytes(snapshot)).toBe(MAX_RECORD_BYTES - 1);
  h.journal.save(scope, snapshot);
  const calls = pendingTurn.calls;
  const entry = { settled: false };
  const session: RetentionSession = {
    scope, snapshot, active: { receipt: running },
    pending: protection === "unresolved MCP entry" ? new Map([[calls[0]!.id, entry]]) : new Map(),
  };
  h.journal.rejectSave = candidate => bytes(candidate) > MAX_RECORD_BYTES;
  pendingTurn.phase = "completed";
  expect(bytes(snapshot)).toBe(MAX_RECORD_BYTES + 3);
  expect(() => save(h, session, protection === "pinned turn" ? [pendingTurn] : [])).toThrow("injected journal failure");
  expect(pendingTurn.calls).toBe(calls);
  expect(pendingTurn.calls[0]!.result).toBe(output);
  expect(output.content).toHaveLength(contentLength);
  expect(pendingTurn.phase).toBe("completed");
  expect(session.active!.receipt).toBe(running);
  if (protection === "unresolved MCP entry") expect(session.pending.get(calls[0]!.id)).toBe(entry);
  const durable = h.journal.load(scope)!;
  expect(durable.turns[0]!.phase).toBe("tools");
  expect(durable.turns[0]!.calls[0]!.result?.content).toBe(output.content);
}, 10_000);
