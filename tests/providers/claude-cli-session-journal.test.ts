import { Database } from "bun:sqlite";
import { afterEach, beforeEach, expect, test } from "bun:test";
import { existsSync, linkSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, statSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { openClaudeSessionJournal, type ClaudeSessionJournal, type ClaudeSessionSnapshot } from "../../src/adapters/claude-cli/session-journal";
import { protectedHomeForTests } from "../../src/lib/test-home-guard";
import { resetHardenedStateForTests, setIcaclsRunnerForTests, setPlatformForTests } from "../../src/lib/windows-secret-acl";
import { setWindowsPrincipalRunnerForTests } from "../../src/lib/windows-user-principal";
import type { AdapterEvent } from "../../src/types";
import { removeTreeWithRetry } from "../helpers/remove-tree";

const scope = "a".repeat(64);
const otherScope = "b".repeat(64);
const MAX_RECORD_BYTES = 16 * 1024 * 1024;
const aclSuccess = { success: true, exitCode: 0, timedOut: false, stdout: "" };
let root: string;
let path: string;
let journal: ClaudeSessionJournal | undefined;
let database: Database | undefined;
let previousHome: string | undefined;

beforeEach(() => {
  root = realpathSync(mkdtempSync(join(tmpdir(), "ocx-claude-session-journal-")));
  path = join(root, "claude-cli-sessions.sqlite");
  previousHome = process.env.OPENCODEX_HOME;
  process.env.OPENCODEX_HOME = root;
  // Most cases test SQLite and validation; ACL cases below exercise the Windows helpers.
  setPlatformForTests("linux");
  resetHardenedStateForTests();
});

afterEach(() => {
  database?.close(); database = undefined;
  journal?.close(); journal = undefined;
  setIcaclsRunnerForTests(null);
  setWindowsPrincipalRunnerForTests(null);
  setPlatformForTests(null);
  resetHardenedStateForTests();
  if (previousHome === undefined) delete process.env.OPENCODEX_HOME;
  else process.env.OPENCODEX_HOME = previousHome;
  removeTreeWithRetry(root);
});

function snapshot(): ClaudeSessionSnapshot {
  return {
    version: 1, history: ["1".repeat(64), "2".repeat(64)],
    turns: [{
      inputHash: "3".repeat(64), phase: "tools", updatedAt: 123,
      events: [
        { type: "text_delta", text: "private text", phase: "commentary", anthropicThinkingSignature: "signed" },
        { type: "tool_call_start", id: "call-1", name: "read_file" },
        { type: "tool_call_delta", arguments: '{"path":"example.txt"}' },
        { type: "tool_call_end" },
      ],
      calls: [
        { id: "call-1", name: "read_file", arguments: '{"path":"example.txt"}', result: { hash: "4".repeat(64), content: "private output", isError: false } },
        { id: "call-2", name: "read_file", arguments: "{}" },
      ],
    }],
  };
}

function open(): ClaudeSessionJournal {
  journal = openClaudeSessionJournal(path);
  return journal;
}

test("commits before returning, reopens durably, and isolates scopes", () => {
  const first = open();
  expect(first.load(scope)).toBeUndefined();
  const saved = snapshot();
  first.save(scope, saved);
  database = new Database(path);
  expect(JSON.parse(database.query<{ payload: string }, [string]>("SELECT payload FROM claude_sessions WHERE scope = ?").get(scope)!.payload)).toEqual(saved);
  expect(database.query("PRAGMA journal_mode").get()).toEqual({ journal_mode: "wal" });
  expect(first.load(otherScope)).toBeUndefined();
  const second: ClaudeSessionSnapshot = { version: 1, history: [], turns: [] };
  first.save(otherScope, second);
  first.close();
  const reopened = open();
  expect(reopened.load(scope)).toEqual(saved);
  expect(reopened.load(otherScope)).toEqual(second);
  saved.turns[0]!.calls.length = 0;
  const loaded = reopened.load(scope)!;
  loaded.turns.length = 0;
  expect(reopened.load(scope)!.turns[0]!.calls).toHaveLength(2);
});

test.each([undefined, false, true])("preserves deliveryStarted=%s across reopen with unacknowledged calls", deliveryStarted => {
  const saved = snapshot();
  if (deliveryStarted !== undefined) saved.turns[0]!.deliveryStarted = deliveryStarted;
  open().save(scope, saved);
  journal!.close();
  expect(open().load(scope)).toEqual(saved);
});

test.each([undefined, "a".repeat(64), "B".repeat(64)])("preserves optional v1 operationId=%s across reopen", operationId => {
  const saved = snapshot();
  if (operationId !== undefined) saved.turns[0]!.operationId = operationId;
  open().save(scope, saved);
  journal!.close();
  expect(open().load(scope)).toEqual(saved);
});

test.each([undefined, "a".repeat(64), "B".repeat(64)])("preserves optional v1 requestHash=%s across reopen", requestHash => {
  const saved = snapshot();
  if (requestHash !== undefined) saved.turns[0]!.requestHash = requestHash;
  open().save(scope, saved);
  journal!.close();
  expect(open().load(scope)).toEqual(saved);
});

test.each(["queued", "sent", "completed"] as const)("round-trips optional v1 %s instructions and replay aliases", phase => {
  const saved = snapshot();
  saved.instructions = [{
    inputHash: scope, operationId: otherScope, history: saved.history, texts: ["Keep it local", "Then summarize"], phase, updatedAt: 456,
  }];
  saved.turns[0]!.requestHashes = [scope, otherScope];
  open().save(scope, saved);
  journal!.close();
  expect(open().load(scope)).toEqual(saved);
  saved.instructions[0]!.texts.push("changed");
  const loaded = journal!.load(scope)!;
  loaded.instructions![0]!.texts.length = 0;
  loaded.turns[0]!.requestHashes!.length = 0;
  expect(journal!.load(scope)!.instructions![0]!.texts).toEqual(["Keep it local", "Then summarize"]);
  expect(journal!.load(scope)!.turns[0]!.requestHashes).toEqual([scope, otherScope]);
});

test("accepts bounded instruction queues without operation metadata and empty optional v1 arrays", () => {
  const saved = snapshot();
  saved.instructions = [];
  saved.turns[0]!.requestHashes = [];
  open().save(scope, saved);
  expect(journal!.load(scope)).toEqual(saved);
  saved.instructions = Array.from({ length: 64 }, () => ({
    inputHash: scope, history: [], texts: Array.from({ length: 64 }, () => "literal text"), phase: "queued", updatedAt: 0,
  }));
  saved.instructions.push(...Array.from({ length: 16 }, () => ({
    inputHash: otherScope, history: [], texts: ["completed instruction"], phase: "completed" as const, updatedAt: 0,
  })));
  expect(saved.instructions).toHaveLength(80);
  saved.turns[0]!.requestHashes = Array.from({ length: 64 }, () => otherScope);
  journal!.save(scope, saved);
  journal!.close();
  expect(open().load(scope)).toEqual(saved);
});

test.each(["running", "tools", "completed", "interrupted", "expired"] as const)("accepts %s receipts without delivery metadata", phase => {
  const saved = snapshot();
  saved.turns[0]!.phase = phase;
  if (phase === "expired") { saved.turns[0]!.events = []; saved.turns[0]!.calls = []; }
  open().save(scope, saved);
  expect(journal!.load(scope)).toEqual(saved);
});

test("round-trips the AdapterEvent wire variants and optional metadata", () => {
  const events: AdapterEvent[] = [
    { type: "heartbeat", replayUnsafe: true, preflightReady: true },
    { type: "thinking_delta", thinking: "thinking" },
    { type: "thinking_signature", signature: "signature" },
    { type: "redacted_thinking", data: "opaque" },
    { type: "kiro_redacted_reasoning", data: "opaque" },
    { type: "reasoning_raw_delta", text: "raw" },
    { type: "tool_call_start", id: "call", name: "tool", providerMetadata: { google: { thoughtSignature: "signature" } } },
    { type: "tool_call_end", incomplete: true },
    { type: "assistant_boundary" },
    { type: "web_search_call_begin", id: "search", itemId: "item" },
    { type: "web_search_call_end", id: "search", itemId: "item", queries: ["example"], status: "completed", sources: [{ url: "https://example.com", title: "Example" }], anthropic: { use: { type: "server_tool_use" }, result: { content: [] } } },
    { type: "done", usage: { inputTokens: 2, outputTokens: 1, upstreamReported: true, rawUsage: { extra: [1, false, null] } }, stopReason: "end_turn", endTurn: true, compactionEncryptedContent: "opaque", providerState: { cursor: { conversationId: "private", checkpointUsable: false }, kiro: { conversationId: "private" }, claude: { opaque: "value" }, __ocxOwner: { version: 1, providerName: "claude", providerDestinationIdentity: "destination", adapterName: "claude-cli", modelId: "model", credentialIdentity: "digest" } } },
    { type: "incomplete", reason: "cancelled", message: "stopped", retryable: true, endTurn: false },
    { type: "error", message: "failed", status: 429, errorType: "rate_limit", code: "busy", retryable: true, apiKeyRecovery: true, replaySafeBeforeOutput: false },
  ];
  const saved = snapshot();
  saved.turns[0]!.events.push(...events);
  open().save(scope, saved);
  expect(journal!.load(scope)).toEqual(saved);
});

test("rejects invalid scopes for both reads and writes", () => {
  const opened = open();
  for (const invalid of ["", "a".repeat(63), "a".repeat(65), "g".repeat(64), "../session", `${scope}\n`, null, 123]) {
    expect(() => opened.load(invalid as string)).toThrow("scope");
    expect(() => opened.save(invalid as string, snapshot())).toThrow("scope");
  }
  expect(opened.load(scope)).toBeUndefined();
});

test("rejects malformed, unsupported, and out-of-bounds snapshots on save and load", () => {
  const opened = open();
  const valid = snapshot();
  const turn = valid.turns[0]!;
  const withTurn = (patch: Record<string, unknown>) => ({ ...valid, turns: [{ ...turn, ...patch }] });
  const instruction = { inputHash: scope, history: valid.history, texts: ["Keep it local"], phase: "queued", updatedAt: 0 };
  const withInstruction = (patch: Record<string, unknown>) => ({ ...valid, instructions: [{ ...instruction, ...patch }] });
  const invalid: unknown[] = [
    null, [], {}, { ...valid, version: 2 }, { ...valid, extra: "unexpected" },
    { ...valid, history: ["not a hash"] }, { ...valid, history: "history" }, { ...valid, turns: {} },
    withTurn({ inputHash: "invalid" }), withTurn({ phase: "pending" }),
    ...["", "a".repeat(63), "a".repeat(65), "g".repeat(64), null, 123].map(operationId => withTurn({ operationId })),
    ...["", "a".repeat(63), "a".repeat(65), "g".repeat(64), null, 123].map(requestHash => withTurn({ requestHash })),
    ...[null, "hash", {}, ["bad"], [null], Array.from({ length: 65 }, () => scope)].map(requestHashes => withTurn({ requestHashes })),
    ...[null, {}, "instructions", [null], Array.from({ length: 81 }, () => instruction)].map(instructions => ({ ...valid, instructions })),
    withInstruction({ extra: true }), withInstruction({ phase: "pending" }),
    ...["", "a".repeat(63), "a".repeat(65), "g".repeat(64), null, 123].flatMap(value => [
      withInstruction({ inputHash: value }), withInstruction({ operationId: value }), withInstruction({ history: [value] }),
      withTurn({ requestHashes: [value] }),
    ]),
    withInstruction({ history: null }), withInstruction({ history: "history" }),
    ...[null, "text", [], [""], [123], Array.from({ length: 65 }, () => "text")].map(texts => withInstruction({ texts })),
    ...[-1, 0.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1].map(updatedAt => withInstruction({ updatedAt })),
    withTurn({ updatedAt: -1 }), withTurn({ updatedAt: NaN }), withTurn({ updatedAt: Infinity }),
    withTurn({ updatedAt: 0.5 }), withTurn({ updatedAt: Number.MAX_SAFE_INTEGER + 1 }),
    withTurn({ deliveryStarted: "true" }), withTurn({ deliveryStarted: null }),
    withTurn({ events: null }), withTurn({ calls: null }),
    withTurn({ events: [{ type: "unknown" }] }), withTurn({ events: [{ type: "text_delta", text: 1 }] }),
    withTurn({ events: [{ type: "text_delta", text: "text", phase: "invalid" }] }),
    withTurn({ events: [{ type: "done", usage: { inputTokens: -1, outputTokens: 1 } }] }),
    withTurn({ events: [{ type: "done", providerState: { cursor: { conversationId: 1 } } }] }),
    withTurn({ events: [{ type: "tool_call_start", id: "id", name: "tool", providerMetadata: { google: { thoughtSignature: false } } }] }),
    withTurn({ events: [{ type: "error", message: "failed", status: "429" }] }),
    withTurn({ calls: [{ id: "", name: "tool", arguments: "{}" }] }),
    withTurn({ calls: [{ id: "id", name: "tool", arguments: {} }] }),
    withTurn({ calls: [{ ...turn.calls[0], result: { hash: "bad", content: "text", isError: false } }] }),
    withTurn({ calls: [{ ...turn.calls[0], result: { hash: scope, content: 1, isError: false } }] }),
    withTurn({ calls: [{ ...turn.calls[0], result: { hash: scope, content: "text", isError: "false" } }] }),
  ];
  opened.save(scope, valid);
  database = new Database(path);
  const replace = database.query("UPDATE claude_sessions SET payload = ? WHERE scope = ?");
  for (const value of invalid) {
    replace.run(JSON.stringify(valid), scope);
    expect(() => opened.save(scope, value as ClaudeSessionSnapshot)).toThrow();
    expect(opened.load(scope)).toEqual(valid);
    replace.run(JSON.stringify(value), scope);
    expect(() => opened.load(scope)).toThrow("load failed");
  }
  for (const payload of ["{", "", "undefined"]) {
    replace.run(payload, scope);
    expect(() => opened.load(scope)).toThrow("load failed");
  }
  const cyclic = snapshot();
  (cyclic as unknown as Record<string, unknown>).cycle = cyclic;
  expect(() => opened.save(scope, cyclic)).toThrow();
});

test("enforces the 16 MiB UTF-8 boundary on save and load", () => {
  const saved = snapshot();
  const event = { type: "text_delta" as const, text: "" };
  saved.turns[0]!.events = [event];
  event.text = "x".repeat(MAX_RECORD_BYTES - Buffer.byteLength(JSON.stringify(saved)));
  expect(Buffer.byteLength(JSON.stringify(saved))).toBe(MAX_RECORD_BYTES);
  open().save(scope, saved);
  expect(journal!.load(scope)!.turns[0]!.events).toEqual([event]);
  const length = event.text.length;
  event.text += "é";
  expect(() => journal!.save(scope, saved)).toThrow("save failed");
  expect((journal!.load(scope)!.turns[0]!.events[0] as typeof event).text.length).toBe(length);
  database = new Database(path);
  database.run("UPDATE claude_sessions SET payload = ? WHERE scope = ?", [JSON.stringify(saved), scope]);
  expect(() => journal!.load(scope)).toThrow("load failed");
});

test("counts private instruction text toward the existing 16 MiB UTF-8 record boundary", () => {
  const saved = snapshot();
  saved.instructions = [{ inputHash: scope, history: [], texts: [""], phase: "sent", updatedAt: 0 }];
  const instruction = saved.instructions[0]!;
  instruction.texts[0] = "x".repeat(MAX_RECORD_BYTES - Buffer.byteLength(JSON.stringify(saved)));
  expect(Buffer.byteLength(JSON.stringify(saved))).toBe(MAX_RECORD_BYTES);
  open().save(scope, saved);
  expect(journal!.load(scope)).toEqual(saved);
  instruction.texts[0] += "é";
  expect(() => journal!.save(scope, saved)).toThrow("save failed");
  expect(Buffer.byteLength(JSON.stringify(journal!.load(scope)))).toBe(MAX_RECORD_BYTES);
  database = new Database(path);
  database.run("UPDATE claude_sessions SET payload = ? WHERE scope = ?", [JSON.stringify(saved), scope]);
  expect(() => journal!.load(scope)).toThrow("load failed");
});

test("database write failures throw and retain the last committed receipt", () => {
  const opened = open();
  const valid = snapshot();
  opened.save(scope, valid);
  const changed = snapshot();
  changed.turns[0]!.deliveryStarted = true;
  database = new Database(path);
  database.exec("BEGIN IMMEDIATE");
  expect(() => opened.save(scope, changed)).toThrow("save failed");
  database.exec("ROLLBACK");
  expect(opened.load(scope)).toEqual(valid);
  database.exec("CREATE TRIGGER reject_receipt BEFORE UPDATE ON claude_sessions BEGIN SELECT RAISE(ABORT, 'private failure'); END");
  expect(() => opened.save(scope, changed)).toThrow("save failed");
  expect(opened.load(scope)).toEqual(valid);
  database.exec("DROP TRIGGER reject_receipt");
  opened.save(scope, changed);
  expect(opened.load(scope)).toEqual(changed);
  opened.close();
  expect(() => opened.load(scope)).toThrow();
  expect(() => opened.save(scope, valid)).toThrow();
});

test("refuses a corrupt database without replacing it", () => {
  writeFileSync(path, "not a SQLite database");
  expect(() => open()).toThrow("open failed");
  expect(readFileSync(path, "utf8")).toBe("not a SQLite database");
});

test("uses the isolated config home by default and guards the real home before mutation", () => {
  journal = openClaudeSessionJournal();
  expect(existsSync(path)).toBe(true);
  expect(() => openClaudeSessionJournal(join(protectedHomeForTests(), "claude-cli-sessions.sqlite"))).toThrow("refusing to write the real OpenCodex home");
});

test("requires an existing parent and refuses directories and hard-linked sidecars", () => {
  expect(() => openClaudeSessionJournal(join(root, "missing", "journal.sqlite"))).toThrow("open failed");
  expect(existsSync(join(root, "missing"))).toBe(false);
  mkdirSync(path);
  expect(() => open()).toThrow("open failed");
  path = join(root, "other.sqlite");
  const target = join(root, "target");
  writeFileSync(target, "untouched");
  linkSync(target, `${path}-wal`);
  expect(() => open()).toThrow("open failed");
  expect(readFileSync(target, "utf8")).toBe("untouched");
});

test.skipIf(process.platform === "win32")("keeps POSIX database and sidecars owner-only and rejects symlinks", () => {
  open().save(scope, snapshot());
  expect(statSync(root).mode & 0o777).toBe(0o700);
  for (const suffix of ["", "-wal", "-shm"]) expect(statSync(path + suffix).mode & 0o777).toBe(0o600);
  journal!.close(); journal = undefined;
  const linked = join(root, "linked.sqlite");
  symlinkSync(path, linked);
  expect(() => openClaudeSessionJournal(linked)).toThrow("open failed");
});

test("hardens the Windows database and sidecars and fails closed on ACL refusal", () => {
  setPlatformForTests("win32");
  setWindowsPrincipalRunnerForTests(() => ({ ...aclSuccess, stdout: "S-1-5-21-1-2-3-1001\nSYNTHETIC\\owner\n" }));
  const paths = new Set<string>();
  setIcaclsRunnerForTests(args => { paths.add(args[0]!); return aclSuccess; });
  open().save(scope, snapshot());
  for (const entry of [root, path, `${path}-wal`, `${path}-shm`]) expect(paths.has(entry)).toBe(true);
  resetHardenedStateForTests();
  setIcaclsRunnerForTests(() => ({ ...aclSuccess, success: false, exitCode: 1, stderr: "denied" }));
  const delivered = snapshot();
  delivered.turns[0]!.deliveryStarted = true;
  expect(() => journal!.save(scope, delivered)).toThrow("save failed");
  expect(journal!.load(scope)).toEqual(snapshot());
  journal!.close(); journal = undefined;
  expect(() => open()).toThrow("open failed");
});
