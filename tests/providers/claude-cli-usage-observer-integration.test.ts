import { expect, test } from "bun:test";
import { EventEmitter } from "node:events";
import type { ChildProcess } from "node:child_process";
import { Readable, Writable } from "node:stream";
import { createClaudeCliAdapter } from "../../src/adapters/claude-cli/adapter";
import { clearClaudeCliBinaryCache } from "../../src/adapters/claude-cli/profiles";
import type { ClaudeUsageObserver } from "../../src/adapters/claude-cli/usage-admission";
import type { AdapterEvent } from "../../src/types";
import { harness, result, toolAssistant, turn, until, withMessages } from "../fixtures/claude-session-worker";

const available = async () => ({ state: "available" as const, checkedAt: 0 });
const provider = { adapter: "claude-cli", baseUrl: "https://api.anthropic.com" };
const quota = { type: "rate_limit_event", rate_limit_info: { status: "allowed", rateLimitType: "five_hour", utilization: 0.37, resetsAt: 2_000_000_000 } };

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void; let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

test("resident quota observation survives HTTP tool handoff and cached replay never creates another observer", async () => {
  clearClaudeCliBinaryCache();
  const h = harness(); const input = turn();
  const messages: unknown[] = []; let created = 0; let finished = 0;
  const observer: ClaudeUsageObserver = { observe: message => { messages.push(message); }, finish: () => { finished++; } };
  const adapter = createClaudeCliAdapter(provider, { which: () => "/synthetic/claude.exe", sessionManager: h.manager,
    usageAdmission: available, usageObserver: () => { created++; return observer; } });
  const run = (next = input) => {
    const events: AdapterEvent[] = [];
    const done = adapter.runTurn!(next.parsed, next.incoming, event => events.push(event));
    void done.catch(() => {});
    return { events, done };
  };
  try {
    const first = run(); await until(() => !!h.workers[0]?.sends.length);
    const worker = h.workers[0]!;
    const action = { id: "toolu_quota_observer", name: worker.options.tools[0]!.name, arguments: { path: "fixture.txt" } };
    const pending = worker.mcp(action);
    await worker.blocks([action]); await first.done;
    expect(first.events.at(-1)).toMatchObject({ type: "done", stopReason: "tool_use" });
    await worker.emit(quota);
    expect(messages.at(-1)).toMatchObject(quota);
    expect(finished).toBe(0);
    const next = withMessages(input, [...input.parsed.context.messages, toolAssistant([action]), result(action.id, "actual result")]);
    const continuation = run(next);
    expect((await pending).content[0]!.text).toBe("actual result");
    await worker.finish("complete"); await continuation.done;
    expect(continuation.events.at(-1)).toMatchObject({ type: "done" });
    const count = messages.length;
    const replay = run(next); await replay.done;
    expect(replay.events.at(-1)).toMatchObject({ type: "done", usage: { totalTokens: 0 } });
    expect(messages).toHaveLength(count); expect(created).toBe(1);
    await h.manager.close();
    expect(finished).toBe(1); expect(worker.stops).toBe(1);
    await worker.emit(quota);
    expect(messages).toHaveLength(count);
  } finally { await h.manager.close(); clearClaudeCliBinaryCache(); }
});

test("optional quota observer failures cannot terminate a resident inference or its shutdown", async () => {
  const h = harness(); const input = turn();
  input.createUsageObserver = () => ({ observe() { throw Error("observation failed"); }, finish() { throw Error("finish failed"); } });
  const response = await h.start(input);
  await response.worker.emit(quota);
  await response.worker.finish("still completes"); await response.done;
  expect(response.events.at(-1)).toMatchObject({ type: "done" });
  expect(response.events.some(event => event.type === "error")).toBe(false);
  await h.manager.close(); expect(response.worker.stops).toBe(1);
});

test("resident launch waits for its quota account binding before starting the worker", async () => {
  const h = harness(); const input = turn(); const binding = deferred<void>();
  let created = false;
  input.createUsageObserver = () => { created = true; return { ready: binding.promise, observe() {}, finish() {} }; };
  const response = h.run(input);
  try {
    await until(() => created); await Bun.sleep(5);
    expect(h.workers).toHaveLength(0);
    binding.resolve(); await until(() => !!h.workers[0]?.sends.length);
    await h.workers[0]!.finish("bound before launch"); await response.done;
    expect(response.events.at(-1)).toMatchObject({ type: "done" });
  } finally { binding.resolve(); await h.manager.close(); }
});

test("a stalled quota binding times out preparation without a late worker launch", async () => {
  const h = harness(undefined, { setupTimeoutMs: 30 }); const input = turn(); const binding = deferred<void>();
  let finished = 0;
  input.createUsageObserver = () => ({ ready: binding.promise, observe() {}, finish() { finished++; } });
  const response = h.run(input);
  try {
    const error = await response.done.then(() => undefined, error => error);
    expect(error).toMatchObject({ code: "claude_session_setup_timeout" });
    binding.resolve(); await Bun.sleep(5);
    expect(h.workers).toHaveLength(0); expect(finished).toBe(1);
  } finally { binding.resolve(); await h.manager.close(); }
});

function child(lines: unknown[]): ChildProcess {
  const proc = new EventEmitter() as ChildProcess;
  proc.stdout = Readable.from(lines.map(line => Buffer.from(JSON.stringify(line) + "\n")));
  proc.stderr = Readable.from([]);
  proc.stdin = new Writable({ write(_chunk, _encoding, done) { done(); } });
  proc.kill = () => true;
  proc.exitCode = null;
  setTimeout(() => { proc.exitCode = 0; proc.emit("close", 0); }, 3);
  return proc;
}

test("one-shot launch waits for its quota account binding and ignores a rejected binding", async () => {
  for (const reject of [false, true]) {
    clearClaudeCliBinaryCache(); const input = turn(); input.parsed.context.tools = [];
    const binding = deferred<void>(); let observing = false; let spawned = 0; let finished = 0;
    const adapter = createClaudeCliAdapter(provider, { which: () => process.execPath, usageAdmission: available,
      usageObserver: () => { observing = true; return { ready: binding.promise, observe() {}, finish() { finished++; } }; },
      spawn: () => { spawned++; return child([{ type: "result", subtype: "success", is_error: false,
        usage: { input_tokens: 3, output_tokens: 2 } }]); }, killGraceMs: 10, reapTimeoutMs: 50 });
    const events: AdapterEvent[] = [];
    const done = adapter.runTurn!(input.parsed, { ...input.incoming, headers: new Headers() }, event => events.push(event));
    try {
      await until(() => observing); await Bun.sleep(5); expect(spawned).toBe(0);
      if (reject) binding.reject(Error("synthetic identity failure")); else binding.resolve();
      await done; expect(spawned).toBe(1); expect(finished).toBe(1);
      expect(events.at(-1)).toMatchObject({ type: "done" });
    } finally { binding.resolve(); clearClaudeCliBinaryCache(); }
  }
});

test.each([false, true])("one-shot observes raw quota frames and finishes without changing output (observer throws=%s)", async throws => {
  clearClaudeCliBinaryCache();
  const input = turn(); input.parsed.context.tools = [];
  const messages: unknown[] = []; let finished = 0;
  const adapter = createClaudeCliAdapter(provider, { which: () => process.execPath, usageAdmission: available,
    usageObserver: () => ({ observe: message => { messages.push(message); if (throws) throw Error("synthetic observer failure"); },
      finish: () => { finished++; if (throws) throw Error("synthetic finish failure"); } }),
    spawn: () => child([quota, { type: "assistant", message: { role: "assistant", content: [{ type: "text", text: "one-shot answer" }] } },
      { type: "result", subtype: "success", is_error: false, usage: { input_tokens: 3, output_tokens: 2 } }]),
    killGraceMs: 10, reapTimeoutMs: 50,
  });
  const events: AdapterEvent[] = [];
  try {
    await adapter.runTurn!(input.parsed, { ...input.incoming, headers: new Headers() }, event => events.push(event));
    expect(messages[0]).toMatchObject(quota);
    expect(finished).toBe(1);
    expect(events).toContainEqual({ type: "text_delta", text: "one-shot answer" });
    expect(events.at(-1)).toMatchObject({ type: "done", usage: { inputTokens: 3, outputTokens: 2 } });
    expect(events.some(event => event.type === "error")).toBe(false);
  } finally { clearClaudeCliBinaryCache(); }
});
