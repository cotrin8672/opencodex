import { describe, expect, test } from "bun:test";
import type { SessionMessage } from "@anthropic-ai/claude-agent-sdk";
import { randomUUID } from "node:crypto";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { claudeDigest, claudeSessionScope } from "../../src/adapters/claude-cli/session-context";
import type { ClaudeNativeSessionIO } from "../../src/adapters/claude-cli/session-native";
import { ControlledWorker, MemoryJournal, assistant, call, harness, result, toolAssistant, turn, until, user, withMessages } from "../fixtures/claude-session-worker";

const directory = resolve(tmpdir(), "ocx-offline-claude-continuity");

function nativeCache() {
  const files = new Map<string, SessionMessage[]>();
  const reads: string[] = [];
  const forks: Array<{ source: string; checkpoint: string; destination: string }> = [];
  let failRead = false;
  const io: ClaudeNativeSessionIO = {
    async getSessionMessages(id, options) {
      expect(options?.dir).toBe(directory);
      reads.push(id);
      if (failRead) { failRead = false; throw new Error("temporary transcript read failure"); }
      return structuredClone(files.get(id) ?? []);
    },
    async forkSession(id, options) {
      expect(options?.dir).toBe(directory);
      const source = files.get(id)!;
      const end = source.findIndex(message => message.uuid === options?.upToMessageId);
      expect(end).toBeGreaterThanOrEqual(0);
      const sessionId = randomUUID();
      files.set(sessionId, source.slice(0, end + 1).map(message => ({ ...structuredClone(message), uuid: randomUUID(), session_id: sessionId })));
      forks.push({ source: id, checkpoint: options!.upToMessageId!, destination: sessionId });
      return { sessionId };
    },
  };
  return { files, reads, forks, io, failNextRead() { failRead = true; },
    capture(worker: ControlledWorker) {
      const deliver = worker.options.onMessage;
      worker.options.onMessage = async message => {
        if (message.type === "assistant") {
          const transcript = files.get(message.session_id) ?? [];
          transcript.push({ type: "assistant", uuid: message.uuid, session_id: message.session_id,
            message: structuredClone(message.message), parent_tool_use_id: null, parent_agent_id: null });
          files.set(message.session_id, transcript);
        }
        await deliver(message);
      };
    },
  };
}

async function settled(cache: ReturnType<typeof nativeCache>) {
  const journal = new MemoryJournal();
  const h = harness(journal, { nativeIO: cache.io, nativeDirectory: directory });
  const input = turn([user("remember one")]);
  const first = await h.start(input);
  cache.capture(first.worker);
  await first.worker.finish("settled answer");
  await first.done;
  const messages = [...input.parsed.context.messages, assistant("settled answer")];
  const native = journal.load(input.scope)!.native!;
  expect(native.dirty).toBe(false);
  expect(native.checkpoints.at(-1)?.messageId).toBe(cache.files.get(native.sessionId)!.at(-1)!.uuid);
  return { h, journal, input, first, messages, native };
}

describe("Claude manager native continuity with offline SDK IO", () => {
  test("active partial plus an unproven foreign assistant forks and imports the whole unseen suffix once", async () => {
    const cache = nativeCache();
    const s = await settled(cache);
    const activeInput = withMessages(s.input, [...s.messages, user("ACTIVE_FOREIGN_A_INPUT")]);
    const active = s.h.run(activeInput);
    await until(() => s.first.worker.sends.length === 2);
    await s.first.worker.begin();
    await s.first.worker.raw({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
    await s.first.worker.raw({ type: "content_block_delta", index: 0,
      delta: { type: "text_delta", text: "PROVEN_CLAUDE_PARTIAL" } });
    expect(active.events).toContainEqual({ type: "text_delta", text: "PROVEN_CLAUDE_PARTIAL" });
    const history = [...activeInput.parsed.context.messages, assistant("PROVEN_CLAUDE_PARTIAL"),
      assistant("UNPROVEN_FOREIGN_ANSWER"), user("ACTIVE_FOREIGN_C_INPUT")];
    const next = s.h.run(withMessages(s.input, history));
    await until(() => !!s.h.workers[1]?.sends.length || s.first.worker.sends.length === 3);
    expect(s.first.worker.stops).toBe(1);
    expect(s.first.worker.interrupts).toBe(0);
    expect(s.first.worker.sends).toHaveLength(2);
    const replacement = s.h.workers[1]!;
    cache.capture(replacement);
    expect(cache.forks).toEqual([{ source: s.native.sessionId, checkpoint: s.native.checkpoints[0]!.messageId,
      destination: replacement.options.resume! }]);
    expect(replacement.options.resume).not.toBe(s.native.sessionId);
    for (const marker of ["ACTIVE_FOREIGN_A_INPUT", "PROVEN_CLAUDE_PARTIAL", "UNPROVEN_FOREIGN_ANSWER", "ACTIVE_FOREIGN_C_INPUT"]) {
      expect(replacement.sends[0]!.split(marker)).toHaveLength(2);
    }
    expect(replacement.sends[0]).not.toContain("settled answer");
    await replacement.finish("FOREIGN_C_COMPLETED");
    await next.done;
    await active.done;
    expect(next.events.some(event => event.type.startsWith("tool_call_") || event.type === "error")).toBe(false);
    const saved = s.journal.load(s.input.scope)!.native!;
    expect(saved.dirty).toBe(false);
    expect(saved.checkpoints[0]!.messageId).not.toBe(s.native.checkpoints[0]!.messageId);
    await s.h.manager.close();
  });

  test("actual pending MCP result plus a foreign assistant cold-catches up without reissuing the executed call", async () => {
    const cache = nativeCache();
    const s = await settled(cache);
    const actionInput = withMessages(s.input, [...s.messages, user("MCP_FOREIGN_A_INPUT")]);
    const active = s.h.run(actionInput);
    await until(() => s.first.worker.sends.length === 2);
    const action = call(actionInput, "toolu_foreign_already_executed", { path: "foreign-original.txt" });
    const pending = s.first.worker.mcp(action);
    await s.first.worker.blocks([action]);
    await active.done;
    expect(active.events.filter(event => event.type === "tool_call_start")).toHaveLength(1);
    const history = [...actionInput.parsed.context.messages, toolAssistant([action]), result(action.id, "MCP_REAL_RESULT"),
      assistant("MCP_FOREIGN_ANSWER"), user("MCP_FOREIGN_C_INPUT")];
    const next = s.h.run(withMessages(s.input, history));
    await until(() => !!s.h.workers[1]?.sends.length || s.first.worker.sends.length === 3);
    expect(s.first.worker.stops).toBe(1);
    await expect(pending).rejects.toThrow("worker stopped");
    expect(s.first.worker.sends).toHaveLength(2);
    const replacement = s.h.workers[1]!;
    cache.capture(replacement);
    expect(cache.forks).toEqual([{ source: s.native.sessionId, checkpoint: s.native.checkpoints[0]!.messageId,
      destination: replacement.options.resume! }]);
    expect(replacement.options.resume).not.toBe(s.native.sessionId);
    for (const marker of ["MCP_FOREIGN_A_INPUT", "foreign-original.txt", "MCP_REAL_RESULT", "MCP_FOREIGN_ANSWER", "MCP_FOREIGN_C_INPUT"]) {
      expect(replacement.sends[0]!.split(marker)).toHaveLength(2);
    }
    await replacement.finish("MCP_FOREIGN_C_COMPLETED");
    await next.done;
    expect(next.events.some(event => event.type.startsWith("tool_call_") || event.type === "error")).toBe(false);
    const receipt = s.journal.load(s.input.scope)!.turns.flatMap(turn => turn.calls);
    expect(receipt).toMatchObject([{ id: action.id, result: { content: "MCP_REAL_RESULT" } }]);
    expect(s.journal.load(s.input.scope)!.native!.dirty).toBe(false);
    await s.h.manager.close();
  });

  test("replacement waits for worker stop/flush before reading, creating, or sending", async () => {
    const cache = nativeCache();
    const s = await settled(cache);
    let flush!: () => void;
    const flushed = new Promise<void>(resolve => { flush = resolve; });
    s.first.worker.stop = () => { s.first.worker.stops++; return flushed; };
    const changed = { ...withMessages(s.input, [...s.messages, user("new input")]), systemPrompt: "changed system prompt" };
    const next = s.h.run(changed);
    try {
      await until(() => s.first.worker.stops === 1);
      await Bun.sleep(0);
      expect(cache.reads).toEqual([]);
      expect(cache.forks).toEqual([]);
      expect(s.h.workers).toHaveLength(1);
      expect(s.first.worker.sends).toHaveLength(1);
    } finally { flush(); }
    await until(() => s.h.workers[1]?.sends.length === 1);
    const replacement = s.h.workers[1]!;
    cache.capture(replacement);
    expect(cache.reads).toEqual([s.native.sessionId]);
    expect(replacement.options.resume).toBe(s.native.sessionId);
    expect(replacement.sends).toEqual(["new input"]);
    await replacement.finish("replacement answer");
    await next.done;
    await s.h.manager.close();
  });

  test("clean close/reopen resumes the same native ID and sends only new input", async () => {
    const cache = nativeCache();
    const s = await settled(cache);
    await s.h.manager.close();
    expect(s.first.worker.stops).toBe(1);
    const reopened = harness(s.journal, { nativeIO: cache.io, nativeDirectory: directory });
    const next = await reopened.start(withMessages(s.input, [...s.messages, user("only this is new")]));
    cache.capture(next.worker);
    expect(next.worker.options.resume).toBe(s.native.sessionId);
    expect(next.worker.options.sessionId).toBeUndefined();
    expect(next.worker.sends).toEqual(["only this is new"]);
    expect(cache.reads).toEqual([s.native.sessionId]);
    expect(cache.forks).toEqual([]);
    await next.worker.finish("resumed answer");
    await next.done;
    const native = s.journal.load(s.input.scope)!.native!;
    expect(native.sessionId).toBe(s.native.sessionId);
    expect(native.checkpoints).toHaveLength(2);
    expect(native.dirty).toBe(false);
    await reopened.manager.close();
  });

  test("dirty tool generation forks the settled checkpoint and never reissues a delivered action", async () => {
    const cache = nativeCache();
    const s = await settled(cache);
    const actionInput = withMessages(s.input, [...s.messages, user("read the file")]);
    const generating = s.h.run(actionInput);
    await until(() => s.first.worker.sends.length === 2);
    const action = call(actionInput, "toolu_already_issued");
    const pending = s.first.worker.mcp(action);
    await s.first.worker.blocks([action]);
    await generating.done;
    expect(generating.events.filter(event => event.type === "tool_call_start")).toHaveLength(1);
    const dirty = s.journal.load(s.input.scope)!;
    expect(dirty.native).toEqual({ ...s.native, dirty: true });
    expect(dirty.turns.at(-1)).toMatchObject({ phase: "tools", deliveryStarted: true });
    await s.h.manager.close();
    await expect(pending).rejects.toThrow("worker stopped");
    const reopened = harness(s.journal, { nativeIO: cache.io, nativeDirectory: directory });
    const replay = reopened.run(actionInput);
    await expect(replay.done).rejects.toMatchObject({ code: "claude_tool_delivery_uncertain" });
    expect(replay.events).toEqual([]);
    expect(reopened.workers).toEqual([]);
    expect(cache.reads).toEqual([]);
    const history = [...actionInput.parsed.context.messages, toolAssistant([action]), result(action.id, "actual client output")];
    const next = await reopened.start(withMessages(s.input, history));
    cache.capture(next.worker);
    expect(cache.forks).toEqual([{ source: s.native.sessionId, checkpoint: s.native.checkpoints[0]!.messageId,
      destination: next.worker.options.resume! }]);
    expect(next.worker.options.resume).not.toBe(s.native.sessionId);
    expect(cache.files.get(next.worker.options.resume!)!).toHaveLength(1);
    expect(next.worker.sends[0]).toContain("actual client output");
    await next.worker.finish("used the supplied output");
    await next.done;
    expect(next.events.some(event => event.type.startsWith("tool_call_"))).toBe(false);
    const saved = s.journal.load(s.input.scope)!;
    expect(saved.turns.flatMap(receipt => receipt.calls)).toMatchObject([{ id: action.id, result: { content: "actual client output" } }]);
    await reopened.manager.close();
  });

  test("a failed checkpoint commit retains dirty history and the previous settled checkpoint", async () => {
    const cache = nativeCache();
    const s = await settled(cache);
    const next = s.h.run(withMessages(s.input, [...s.messages, user("second question")]));
    await until(() => s.first.worker.sends.length === 2);
    s.journal.rejectSave = snapshot => snapshot.native?.dirty === false && snapshot.native.checkpoints.length > 1;
    await s.first.worker.finish("must not become a clean checkpoint");
    await next.done;
    expect(next.events.at(-1)).toMatchObject({ type: "error", code: "claude_journal_write_failed" });
    expect(s.journal.load(s.input.scope)!.native).toEqual({ ...s.native, dirty: true });
    expect(s.journal.load(s.input.scope)!.turns.at(-1)?.phase).toBe("interrupted");
    expect(cache.files.get(s.native.sessionId)).toHaveLength(2);
    s.journal.rejectSave = undefined;
    await s.h.manager.close();
  });

  test("separate caller threads keep distinct native IDs across close/reopen", async () => {
    const cache = nativeCache();
    const journal = new MemoryJournal();
    const h = harness(journal, { nativeIO: cache.io, nativeDirectory: directory });
    const inputs = [turn([user("thread one")]), turn([user("thread two")])];
    inputs[1]!.incoming.headers.set("thread-id", "a-different-thread");
    inputs[1]!.scope = claudeSessionScope(inputs[1]!.parsed, inputs[1]!.incoming, inputs[1]!.provider)!;
    const ids: string[] = [];
    for (const input of inputs) {
      const response = await h.start(input);
      cache.capture(response.worker);
      await response.worker.finish("settled");
      await response.done;
      ids.push(journal.load(input.scope)!.native!.sessionId);
    }
    expect(inputs[0]!.scope).not.toBe(inputs[1]!.scope);
    expect(new Set(ids).size).toBe(2);
    await h.manager.close();
    const reopened = harness(journal, { nativeIO: cache.io, nativeDirectory: directory });
    for (const [index, input] of inputs.entries()) {
      const response = await reopened.start(withMessages(input, [...input.parsed.context.messages, assistant("settled"), user(`new for ${index}`)]));
      expect(response.worker.options.resume).toBe(ids[index]);
      expect(response.worker.sends).toEqual([`new for ${index}`]);
      await response.worker.finish("done");
      await response.done;
    }
    expect(cache.reads).toEqual(ids);
    expect(cache.forks).toEqual([]);
    await reopened.manager.close();
  });

  test("a transient transcript read fails before inference without changing queued instructions or receipts", async () => {
    const cache = nativeCache();
    const s = await settled(cache);
    await s.h.manager.close();
    const before = s.journal.load(s.input.scope)!;
    before.instructions = [{ inputHash: claudeDigest("accepted reminder"), history: before.history,
      texts: ["accepted reminder"], phase: "queued", updatedAt: 1 }];
    s.journal.save(s.input.scope, before);
    const reopened = harness(s.journal, { nativeIO: cache.io, nativeDirectory: directory });
    const input = withMessages(s.input, [...s.messages, user("accepted reminder"), user("new question")]);
    cache.failNextRead();
    const failed = reopened.run(input);
    await expect(failed.done).rejects.toMatchObject({ code: "claude_native_history_unavailable", status: 503, retryable: true });
    expect(failed.events).toEqual([]);
    expect(reopened.workers).toEqual([]);
    expect(cache.forks).toEqual([]);
    expect(s.journal.load(s.input.scope)).toEqual(before);
    const retry = await reopened.start(input);
    expect(retry.worker.options.resume).toBe(s.native.sessionId);
    expect(retry.worker.sends[0]!.match(/accepted reminder/g)).toHaveLength(1);
    await retry.worker.finish("recovered");
    await retry.done;
    expect(cache.reads).toEqual([s.native.sessionId, s.native.sessionId]);
    expect(s.journal.load(s.input.scope)!.instructions?.[0]?.phase).toBe("completed");
    await reopened.manager.close();
  });
});
