import { expect, test } from "bun:test";
import type { SessionMessage } from "@anthropic-ai/claude-agent-sdk";
import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, realpathSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { claudeHistory } from "../../src/adapters/claude-cli/session-context";
import type { ClaudeNativeSession } from "../../src/adapters/claude-cli/session-journal";
import { planClaudeNativeSession, type ClaudeNativeSessionIO } from "../../src/adapters/claude-cli/session-native";
import type { OcxMessage } from "../../src/types";
import { removeTreeWithRetry } from "../helpers/remove-tree";
import { repoPath } from "../helpers/repo-root";

const directory = resolve(tmpdir(), "ocx-native-worker");
const identity = "a".repeat(64);
const user = (content: string): OcxMessage => ({ role: "user", content, timestamp: 0 });
const assistant = (text: string): OcxMessage => ({ role: "assistant", content: [{ type: "text", text }], timestamp: 0 });
const initial = [user("one"), assistant("Claude one")];

function fixture(history = initial) {
  const sessionId = randomUUID();
  const frame = (type: "user" | "assistant", content: unknown, id = sessionId): SessionMessage => ({
    type, uuid: randomUUID(), session_id: id, message: { role: type, content, stop_reason: type === "assistant" ? "end_turn" : null },
    parent_tool_use_id: null, parent_agent_id: null,
  });
  const source = [frame("user", "bootstrap"), frame("assistant", [{ type: "text", text: "Claude one" }])];
  const files = new Map([[sessionId, source]]);
  const reads: string[] = [];
  const forks: Array<{ id: string; upToMessageId?: string }> = [];
  const io: ClaudeNativeSessionIO = {
    async getSessionMessages(id, options) { expect(options).toEqual({ dir: directory }); reads.push(id); return structuredClone(files.get(id) ?? []); },
    async forkSession(id, options) {
      expect(options?.dir).toBe(directory);
      forks.push({ id, upToMessageId: options?.upToMessageId });
      const source = files.get(id)!;
      const end = source.findIndex(message => message.uuid === options?.upToMessageId);
      const sessionId = randomUUID();
      files.set(sessionId, source.slice(0, end + 1).map(message => ({ ...structuredClone(message), uuid: randomUUID(), session_id: sessionId })));
      return { sessionId };
    },
  };
  const record: ClaudeNativeSession = { sessionId, directory, identity, history: claudeHistory(history),
    checkpoints: [{ length: claudeHistory(history).length, messageId: source[1]!.uuid }], dirty: false };
  const plan = (messages = [...history, user("next")], native = record, customIO = io) =>
    planClaudeNativeSession({ record: native, messages, identity, directory }, customIO);
  return { record, source, files, reads, forks, io, frame, plan };
}

test("fresh sessions use crypto UUIDs and no SDK IO", async () => {
  const f = fixture();
  const input = { messages: initial, identity, directory };
  const first = await planClaudeNativeSession(input, f.io);
  const second = await planClaudeNativeSession(input, f.io);
  expect(first).toEqual({ native: { sessionId: expect.any(String), identity, directory, history: [], checkpoints: [], dirty: false }, start: 0 });
  expect(first.native.sessionId).not.toBe(second.native.sessionId);
  expect(f.reads).toEqual([]);
});

test("Claude/GPT/Claude/Gemini/Claude full histories catch up each foreign turn only once", async () => {
  const f = fixture();
  let record = f.record;
  const messages = [...initial];
  for (const provider of ["GPT", "Gemini"]) {
    messages.push(user(`ask ${provider}`), assistant(`${provider} answer`), user("ask Claude"));
    const plan = await f.plan(messages, record);
    expect(plan.resume).toBe(record.sessionId);
    expect(plan.start).toBe(messages.length - 3);
    expect(messages.slice(plan.start).map(message => message.role)).toEqual(["user", "assistant", "user"]);
    messages.push(assistant("Claude answer"));
    const end = f.frame("assistant", [{ type: "text", text: "Claude answer" }]);
    f.source.push(f.frame("user", "catchup"), end);
    record = { ...plan.native, history: claudeHistory(messages), checkpoints: [...record.checkpoints,
      { length: claudeHistory(messages).length, messageId: end.uuid }] };
    const duplicate = await f.plan(messages, record);
    expect(duplicate.start).toBe(messages.length);
    expect(duplicate.resume).toBe(record.sessionId);
  }
  expect(f.forks).toEqual([]);
});

test("raw offsets skip developer/thinking and preserve collapsed adjacent assistant fragments", async () => {
  const messages: OcxMessage[] = [user("one"), { role: "developer", content: "policy", timestamp: 0 },
    assistant("Claude "), assistant("one"), user("next")];
  const f = fixture();
  expect((await f.plan(messages)).start).toBe(4);
});

test.each([false, true])("edits and Undo fork at the last matching safe checkpoint, dirty=%s", async dirty => {
  const messages = [...initial, user("old"), assistant("old answer")];
  const f = fixture(messages);
  const last = f.frame("assistant", [{ type: "text", text: "old answer" }]);
  f.source.push(f.frame("user", "old"), last);
  f.record.checkpoints.unshift({ length: 2, messageId: f.source[1]!.uuid });
  f.record.checkpoints[1] = { length: 4, messageId: last.uuid };
  f.record.dirty = dirty;
  for (const edited of [[...initial, user("edited")], initial]) {
    const plan = await f.plan(edited);
    expect(plan.start).toBe(2);
    expect(plan.resume).not.toBe(f.record.sessionId);
    expect(plan.native.history).toEqual(claudeHistory(initial));
    expect(plan.native.dirty).toBe(false);
    expect(plan.native.checkpoints).toEqual([{ length: 2, messageId: f.files.get(plan.resume!)![1]!.uuid }]);
  }
  expect(f.forks.map(fork => fork.upToMessageId)).toEqual([f.source[1]!.uuid, f.source[1]!.uuid]);
});

test.each([false, true])("pending tool tails are never resumed, dirty=%s", async dirty => {
  const f = fixture();
  f.record.dirty = dirty;
  f.source.push(f.frame("assistant", [{ type: "tool_use", id: "pending", name: "external", input: { amount: 7 } }]));
  const plan = await f.plan();
  expect(plan.start).toBe(2);
  expect(plan.resume).not.toBe(f.record.sessionId);
  expect(f.files.get(plan.resume!)!).toHaveLength(2);
  expect(f.record.dirty).toBe(dirty);
});

test("cold dirty records fork even when the visible file ends at the settled checkpoint", async () => {
  const f = fixture();
  f.record.dirty = true;
  expect((await f.plan()).resume).not.toBe(f.record.sessionId);
  expect(f.forks).toHaveLength(1);
});

test.each(["missing", "no-checkpoint", "unknown-anchor", "unsafe-anchor", "changed-prefix"])("%s starts fresh", async reason => {
  const f = fixture();
  if (reason === "missing") f.files.clear();
  if (reason === "no-checkpoint") f.record.checkpoints = [];
  if (reason === "unknown-anchor") f.record.checkpoints[0]!.messageId = randomUUID();
  if (reason === "unsafe-anchor") f.source[1]!.message = { content: [{ type: "tool_use", id: "pending" }] };
  const plan = await f.plan(reason === "changed-prefix" ? [user("changed")] : undefined);
  expect(plan.start).toBe(0);
  expect(plan.resume).toBeUndefined();
  expect(plan.native.history).toEqual([]);
  expect(plan.native.sessionId).not.toBe(f.record.sessionId);
  expect(f.forks).toEqual([]);
});

test.each(["identity", "directory"] as const)("%s isolates native history without reading a different cache", async field => {
  const f = fixture();
  f.record[field] = field === "identity" ? "b".repeat(64) : join(directory, "other");
  expect((await f.plan()).resume).toBeUndefined();
  expect(f.reads).toEqual([]);
});

test.each(["wrong-session", "duplicate-uuid", "invalid-uuid"])("counterfeit source %s is refused", async reason => {
  const f = fixture();
  if (reason === "wrong-session") f.source[0]!.session_id = randomUUID();
  if (reason === "duplicate-uuid") f.source[0]!.uuid = f.source[1]!.uuid;
  if (reason === "invalid-uuid") f.source[0]!.uuid = "not-uuid";
  await expect(f.plan()).rejects.toMatchObject({ code: "claude_native_history_invalid", status: 503 });
});

test.each(["body", "type", "tool-arguments", "missing", "extra", "reused-uuid", "missing-mapping"])("fork %s changes fail closed", async reason => {
  const f = fixture();
  f.record.dirty = true;
  f.source[0]!.message = { content: [{ type: "tool_use", id: "historical", name: "external", input: { uuid: randomUUID(), amount: 7 } },
    { type: "tool_result", tool_use_id: "historical", content: "actual output" }] };
  if (reason === "missing-mapping") f.record.checkpoints.unshift({ length: 0, messageId: randomUUID() });
  const io: ClaudeNativeSessionIO = { ...f.io, async forkSession(id, options) {
    const fork = await f.io.forkSession(id, options);
    const file = f.files.get(fork.sessionId)!;
    if (reason === "body") file[1]!.message = { content: "counterfeit" };
    if (reason === "type") file[1]!.type = "user";
    if (reason === "tool-arguments") (file[0]!.message as any).content[0].input.amount = 8;
    if (reason === "missing") file.pop();
    if (reason === "extra") file.push(f.frame("assistant", [], fork.sessionId));
    if (reason === "reused-uuid") file[0]!.uuid = f.source[0]!.uuid;
    return fork;
  } };
  await expect(f.plan(undefined, f.record, io)).rejects.toMatchObject({ code: "claude_native_history_invalid", status: 503 });
});

test.each(["read", "fork"])("actual %s failures are retryable 503s", async operation => {
  const f = fixture();
  f.record.dirty = true;
  const io = { ...f.io, [operation === "read" ? "getSessionMessages" : "forkSession"]: async () => { throw new Error("filesystem denied"); } };
  await expect(f.plan(undefined, f.record, io)).rejects.toMatchObject({ code: "claude_native_history_unavailable", status: 503, retryable: true });
});

test("public SDK 0.3.314 reads and forks offline in an isolated process-global profile", () => {
  const root = realpathSync(mkdtempSync(join(tmpdir(), "ocx-native-sync-")));
  try {
    const dir = join(root, "workspace");
    const profile = join(root, "profile");
    const project = join(profile, "projects", dir.replace(/[^a-zA-Z0-9]/g, "-"));
    mkdirSync(project, { recursive: true }); mkdirSync(dir);
    const f = fixture();
    const rows = f.source.map((message, index) => ({ type: message.type, uuid: message.uuid, sessionId: f.record.sessionId,
      parentUuid: index ? f.source[index - 1]!.uuid : null, message: message.message, cwd: dir, isSidechain: false,
      timestamp: "2026-01-01T00:00:00.000Z", version: "2.1.314" }));
    writeFileSync(join(project, `${f.record.sessionId}.jsonl`), rows.map(row => JSON.stringify(row)).join("\n") + "\n");
    const script = `import { planClaudeNativeSession } from ${JSON.stringify(pathToFileURL(repoPath("src", "adapters", "claude-cli", "session-native.ts")).href)};
      const input = ${JSON.stringify({ record: { ...f.record, directory: dir }, messages: [...initial, user("next")], identity, directory: dir })};
      const resumed = await planClaudeNativeSession(input);
      if (resumed.resume !== input.record.sessionId || resumed.start !== 2) throw Error("resume mismatch");
      input.record.dirty = true;
      const forked = await planClaudeNativeSession(input);
      if (!forked.resume || forked.resume === resumed.resume || forked.start !== 2
        || forked.native.checkpoints[0].messageId === input.record.checkpoints[0].messageId) throw Error("fork mapping mismatch");
      console.log("offline SDK resume/fork verified");`;
    const child = Bun.spawnSync([process.execPath, "-e", script], { cwd: repoPath(),
      env: { ...process.env, CLAUDE_CONFIG_DIR: profile }, stdout: "pipe", stderr: "pipe", timeout: 20_000 });
    expect(child.stderr.toString()).toBe("");
    expect(child.exitCode).toBe(0);
    expect(child.stdout.toString()).toContain("offline SDK resume/fork verified");
  } finally { removeTreeWithRetry(root); }
});
