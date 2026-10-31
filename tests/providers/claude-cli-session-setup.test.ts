import { afterEach, expect, test } from "bun:test";
import { ClaudeSessionManager, type ClaudeSessionManagerDeps } from "../../src/adapters/claude-cli/session-manager";
import { ClaudeSessionError } from "../../src/adapters/claude-cli/session-errors";
import type { AdapterEvent } from "../../src/types";
import { ControlledWorker, MemoryJournal, assistant, call, harness, result, toolAssistant, turn, until, user, withMessages } from "../fixtures/claude-session-worker";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
// Bun 1.4.4 can stop pumping timers inside .rejects; settle before entering the matcher.
async function rejection(pending: Promise<unknown>): Promise<Error> {
  try { await pending; } catch (error) { return error as Error; }
  throw new Error("Expected the promise to reject");
}
const managers: ClaudeSessionManager[] = [];
const cleanup: Array<() => void> = [];
afterEach(async () => {
  for (const release of cleanup.splice(0)) release();
  await Promise.all(managers.splice(0).map(manager => manager.close()));
});
function custom(factory: NonNullable<ClaudeSessionManagerDeps["createWorker"]>, setupTimeoutMs = 80, timeoutMs = 2_000) {
  const manager = new ClaudeSessionManager({ journal: new MemoryJournal(), createWorker: factory, setupTimeoutMs, timeoutMs,
    nativeIO: { getSessionMessages: async () => [], forkSession: async () => { throw Error("Unexpected fork"); } } });
  managers.push(manager);
  const run = (input = turn()) => {
    const events: AdapterEvent[] = [];
    const done = manager.run(input, event => events.push(event));
    void done.catch(() => {});
    return { done, events };
  };
  return { manager, run };
}

test("native output refreshes inactivity rather than imposing a total generation deadline", async () => {
  const h = harness(undefined, { timeoutMs: 100 });
  const first = await h.start();
  await first.worker.begin();
  await first.worker.raw({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
  for (let n = 0; n < 5; n++) {
    await Bun.sleep(35);
    await first.worker.raw({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "progress" } });
  }
  expect(first.events.some(event => event.type === "error")).toBe(false);
  await first.worker.finish("finished after the former total deadline");
  await first.done;
  expect(first.events.at(-1)).toMatchObject({ type: "done" });
});

test.each(["heartbeat", "stale-turn"])("%s messages cannot keep a stalled generation alive", async mode => {
  const h = harness(undefined, { timeoutMs: 80 });
  const first = await h.start();
  for (let n = 0; n < 5; n++) {
    await Bun.sleep(25);
    if (mode === "heartbeat") await first.worker.emit({ type: "keep_alive" });
    else await first.worker.emit({ type: "assistant", user_message_uuid: "obsolete-turn", message: {
      role: "assistant", content: [{ type: "text", text: "stale" }],
    } });
  }
  await first.done;
  expect(first.events.at(-1)).toMatchObject({ type: "error", code: "claude_session_timeout" });
  expect(first.worker.stops).toBe(1);
  const retry = await h.start();
  await retry.worker.finish("recovered"); await retry.done;
  expect(retry.events.at(-1)).toMatchObject({ type: "done" });
});

test.each([false, true])("native history wait releases setup on timeout or cancellation (cancel=%s)", async cancel => {
  const held = deferred<[]>(); cleanup.push(() => held.resolve([]));
  let reads = 0;
  const h = harness(undefined, { setupTimeoutMs: 80, timeoutMs: 2_000, nativeIO: {
    getSessionMessages: async () => ++reads === 1 ? held.promise : [],
    forkSession: async () => { throw Error("Cancelled history must not fork"); },
  } });
  const input = turn([user("seed")]);
  const first = await h.start(input); await first.worker.finish("seed answer"); await first.done;
  const next = withMessages(input, [...input.parsed.context.messages, assistant("seed answer"), user("next")]);
  next.systemPrompt += " changed";
  const abort = new AbortController();
  const blocked = h.run({ ...next, incoming: { ...next.incoming, abortSignal: abort.signal } });
  void blocked.done.catch(() => {});
  await until(() => reads === 1);
  if (cancel) { abort.abort(); await blocked.done; }
  else expect(await rejection(blocked.done)).toMatchObject({ code: "claude_session_setup_timeout", status: 504, retryable: true });
  const retry = await h.start(next);
  held.resolve([]);
  await first.worker.finish("late obsolete output");
  await retry.worker.finish("new answer"); await retry.done;
  expect(retry.events.at(-1)).toMatchObject({ type: "done" });
  expect(retry.events.some(event => event.type === "text_delta" && event.text.includes("obsolete"))).toBe(false);
  expect(h.workers).toHaveLength(2);
});

test("a cancelled queued request cannot let its successor overtake the preparation owner", async () => {
  const gate = deferred<ControlledWorker>(); let creates = 0;
  let old!: ControlledWorker; const workers: ControlledWorker[] = [];
  const h = custom(async options => {
    creates++;
    const worker = new ControlledWorker(options); workers.push(worker);
    if (creates === 1) { old = worker; return gate.promise; }
    return worker;
  }, 2_000);
  cleanup.push(() => gate.resolve(old));
  const a = h.run(); await until(() => creates === 1);
  const abort = new AbortController();
  const input = turn([user("successor")]);
  const b = h.run({ ...input, incoming: { ...input.incoming, abortSignal: abort.signal } });
  abort.abort(); await b.done;
  const c = h.run(input);
  await Bun.sleep(25); expect(creates).toBe(1);
  gate.resolve(old);
  await until(() => workers.length === 2 && !!workers[1]!.sends.length);
  await workers[1]!.finish("successor answer"); await a.done; await c.done;
  expect(old.stops).toBe(1);
  expect(c.events.at(-1)).toMatchObject({ type: "done" });
});

test("a timed-out factory remains fenced until its late worker is confirmed stopped", async () => {
  const gate = deferred<ControlledWorker>(); const stopped = deferred<void>();
  const workers: ControlledWorker[] = []; let late!: ControlledWorker;
  const h = custom(async options => {
    const worker = new ControlledWorker(options); workers.push(worker);
    if (workers.length === 1) {
      late = worker;
      worker.stop = () => { worker.stops++; return stopped.promise; };
      return gate.promise;
    }
    return worker;
  });
  cleanup.push(() => { gate.resolve(late); stopped.resolve(); });
  const first = h.run();
  expect(await rejection(first.done)).toMatchObject({ code: "claude_session_setup_timeout" });
  expect(first.events).toEqual([]);
  const second = h.run();
  expect(await rejection(second.done)).toMatchObject({ code: "claude_session_setup_timeout" });
  expect(workers).toHaveLength(1);
  gate.resolve(late); await until(() => late.stops === 1);
  const third = h.run(); await Bun.sleep(25); expect(workers).toHaveLength(1);
  stopped.resolve();
  await until(() => !!workers[1]?.sends.length);
  await late.finish("obsolete factory response");
  await workers[1]!.finish("only new inference"); await third.done;
  expect(late.sends).toEqual([]);
  expect(late.stops).toBe(1);
  expect(third.events.at(-1)).toMatchObject({ type: "done" });
  expect(third.events.some(event => event.type === "text_delta" && event.text.includes("obsolete"))).toBe(false);
});

test("tool-catalog setup timeout keeps actual tool results and never reissues the delivered action", async () => {
  const held = deferred<void>(); cleanup.push(() => held.resolve());
  const h = harness(undefined, { setupTimeoutMs: 80, timeoutMs: 2_000 });
  const input = turn(); const first = await h.start(input);
  const action = call(input, "toolu_setup_once"); const pending = first.worker.mcp(action);
  await first.worker.blocks([action]); await first.done;
  first.worker.updateTools = () => held.promise;
  const next = withMessages(input, [...input.parsed.context.messages, toolAssistant([action]), result(action.id, "ACTUAL_RESULT")]);
  next.bridge = { ...next.bridge, tools: next.bridge.tools.map(tool => ({ ...tool, description: tool.description + " updated" })) };
  const blocked = h.run(next); void blocked.done.catch(() => {});
  expect(await rejection(blocked.done)).toMatchObject({ code: "claude_session_setup_timeout" });
  expect(await rejection(pending)).toMatchObject({ message: expect.stringContaining("worker stopped") });
  expect(h.journal.load(input.scope)!.turns[0]!.calls[0]!.result?.content).toBe("ACTUAL_RESULT");
  const retry = await h.start(next); held.resolve();
  expect(retry.worker.sends[0]).toContain("ACTUAL_RESULT");
  await retry.worker.finish("used the existing result"); await retry.done;
  expect(retry.events.some(event => event.type === "tool_call_start")).toBe(false);
  expect(h.workers).toHaveLength(2);
});

test("generation failure interrupts a pending SDK write before a consumer is attached", async () => {
  const held = deferred<void>(); cleanup.push(() => held.resolve());
  const workers: ControlledWorker[] = [];
  const h = custom(async options => {
    const worker = new ControlledWorker(options); workers.push(worker);
    if (workers.length === 1) {
      const send = worker.send.bind(worker);
      worker.send = async (...args) => { await send(...args); await held.promise; };
    }
    return worker;
  }, 1_000, 80);
  const first = h.run();
  expect(await rejection(first.done)).toMatchObject({ code: "claude_session_timeout" });
  const retry = h.run(); await until(() => !!workers[1]?.sends.length);
  held.reject(Error("late write rejection"));
  await workers[1]!.finish("recovered after write timeout"); await retry.done;
  expect(workers[0]!.stops).toBe(1);
  expect(retry.events.at(-1)).toMatchObject({ type: "done" });
});

test("a queued instruction reports missing results without stopping or repeating the original tool", async () => {
  const h = harness(undefined, { setupTimeoutMs: 80, timeoutMs: 2_000 });
  const input = turn(); const first = await h.start(input);
  const action = call(input, "toolu_followup_once");
  const pending = first.worker.mcp(action);
  await first.worker.blocks([action]); await first.done;
  const original = [...input.parsed.context.messages, toolAssistant([action])];
  const waiting = h.run(withMessages(input, [...original, user("apply this saved instruction")]));
  await Bun.sleep(130); await waiting.done;
  expect(waiting.events.at(-1)).toMatchObject({ type: "error", code: "claude_tool_results_pending", status: 409, retryable: false });
  expect(first.worker.stops).toBe(0);
  expect(first.worker.sends).toHaveLength(1);
  expect(h.journal.load(input.scope)!.instructions!.map(entry => entry.phase)).toEqual(["queued"]);
  const continuation = h.run(withMessages(input, [...original, result(action.id, "ACTUAL_LATE_RESULT")]));
  expect((await pending).content[0]!.text).toBe("ACTUAL_LATE_RESULT");
  expect(first.worker.sends[1]).toBe("apply this saved instruction");
  expect(first.worker.sendOptions[1]).toMatchObject({ shouldQuery: false });
  await first.worker.finish("applied saved instruction"); await continuation.done;
  expect(continuation.events.some(event => event.type === "tool_call_start")).toBe(false);
  expect(h.workers).toHaveLength(1); expect(first.worker.stops).toBe(0);
  expect(h.journal.load(input.scope)!.instructions!.map(entry => entry.phase)).toEqual(["completed"]);
});

test("shutdown cancels a queued setup without waiting for its normal deadline", async () => {
  const held = deferred<ControlledWorker>(); let worker!: ControlledWorker;
  const h = custom(async options => { worker = new ControlledWorker(options); return held.promise; }, 5_000);
  cleanup.push(() => held.resolve(worker));
  const first = h.run(); await until(() => !!worker);
  const queued = h.run();
  const closing = h.manager.close();
  expect(await rejection(first.done)).toMatchObject({ code: "claude_session_shutdown" });
  expect(await rejection(queued.done)).toMatchObject({ code: "claude_session_shutdown" });
  held.resolve(worker); await closing;
  expect(worker.stops).toBe(1); expect(worker.sends).toEqual([]);
});

test("a launch rejected after cleanup can retry without retaining a false ownership fence", async () => {
  let creates = 0; let worker!: ControlledWorker;
  const h = custom(async options => {
    if (++creates === 1) throw Error("sanitized launch failure");
    worker = new ControlledWorker(options); return worker;
  });
  expect(await rejection(h.run().done)).toMatchObject({ message: expect.stringContaining("launch failure") });
  const retry = h.run(); await until(() => !!worker?.sends.length);
  await worker.finish("recovered"); await retry.done;
  expect(creates).toBe(2); expect(retry.events.at(-1)).toMatchObject({ type: "done" });
});

test("unverified launch cleanup refuses a second process and closes with an explicit error", async () => {
  let creates = 0;
  const h = custom(async () => { creates++; throw new ClaudeSessionError("unconfirmed process exit", "claude_worker_shutdown_failed", 503, true); });
  expect(await rejection(h.run().done)).toMatchObject({ code: "claude_worker_shutdown_failed" });
  expect(await rejection(h.run().done)).toMatchObject({ code: "claude_worker_shutdown_failed" });
  expect(creates).toBe(1);
  expect(await rejection(h.manager.close())).toMatchObject({ code: "claude_worker_shutdown_failed" });
  managers.splice(managers.indexOf(h.manager), 1);
});

test("late original-turn output cannot clear or refresh a waiting instruction deadline", async () => {
  const h = harness(undefined, { setupTimeoutMs: 150, timeoutMs: 2_000 });
  const input = turn(); const first = await h.start(input);
  const action = call(input, "toolu_late_output_wait");
  const pending = first.worker.mcp(action);
  let settled = false;
  void pending.then(() => { settled = true; }, () => { settled = true; });
  await first.worker.blocks([action]); await first.done;
  const history = [...input.parsed.context.messages, toolAssistant([action])];
  const waiting = h.run(withMessages(input, [...history, user("retain this instruction despite late output")]));
  await until(() => waiting.events.some(event => event.type === "heartbeat"));
  for (let n = 0; n < 6; n++) {
    await Bun.sleep(35);
    if (n % 2 === 0) await first.worker.emit({ type: "assistant", user_message_uuid: first.worker.sendOptions[0]!.uuid,
      message: { role: "assistant", content: [{ type: "text", text: "late original output" }] } });
    else await first.worker.raw({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "late original output" } });
  }
  expect(waiting.events.at(-1)).toMatchObject({ type: "error", code: "claude_tool_results_pending", status: 409, retryable: false });
  await waiting.done;
  expect(waiting.events.filter(event => event.type === "error")).toHaveLength(1);
  expect(waiting.events.some(event => event.type === "text_delta" || event.type === "incomplete")).toBe(false);
  expect(first.worker.stops).toBe(0); expect(settled).toBe(false);
  expect(h.journal.load(input.scope)!.instructions!.map(entry => entry.phase)).toEqual(["queued"]);
  const continuation = h.run(withMessages(input, [...history, result(action.id, "ACTUAL_RESULT_AFTER_LATE_OUTPUT")]));
  expect((await pending).content[0]!.text).toBe("ACTUAL_RESULT_AFTER_LATE_OUTPUT");
  expect(first.worker.sends[1]).toBe("retain this instruction despite late output");
  await first.worker.finish("continued after the waiting deadline"); await continuation.done;
  expect(continuation.events.at(-1)).toMatchObject({ type: "done" });
  expect(continuation.events.some(event => event.type === "tool_call_start")).toBe(false);
  expect(h.workers).toHaveLength(1); expect(first.worker.stops).toBe(0);
  expect(h.journal.load(input.scope)!.instructions!.map(entry => entry.phase)).toEqual(["completed"]);
});

test("cancellation between factory resolution and adoption fences successors until orphan stop completes", async () => {
  const factory = deferred<ControlledWorker>(); const stopped = deferred<void>();
  const workers: ControlledWorker[] = [];
  const h = custom(options => {
    const worker = new ControlledWorker(options); workers.push(worker);
    if (workers.length === 1) {
      worker.stop = () => { worker.stops++; return stopped.promise; };
      return factory.promise;
    }
    return Promise.resolve(worker);
  }, 1_000);
  cleanup.push(() => { factory.resolve(workers[0]!); stopped.resolve(); });
  const input = turn(); const abort = new AbortController();
  const first = h.run({ ...input, incoming: { ...input.incoming, abortSignal: abort.signal } });
  await until(() => workers.length === 1);
  factory.resolve(workers[0]!);
  queueMicrotask(() => abort.abort());
  await first.done;
  await until(() => workers[0]!.stops === 1);
  expect(first.events).toEqual([]); expect(workers[0]!.sends).toEqual([]);
  const retry = h.run(input);
  await Bun.sleep(25);
  expect(workers).toHaveLength(1);
  stopped.resolve();
  await until(() => !!workers[1]?.sends.length);
  await workers[0]!.finish("obsolete unadopted worker output");
  await workers[1]!.finish("successor after verified stop"); await retry.done;
  expect(workers[0]!.stops).toBe(1);
  expect(retry.events.at(-1)).toMatchObject({ type: "done" });
  expect(retry.events.some(event => event.type === "error" || event.type === "text_delta" && event.text.includes("obsolete"))).toBe(false);
});

test("typed provider error wins over a pending SDK write rejected synchronously by stop", async () => {
  const write = deferred<void>(); cleanup.push(() => write.resolve());
  let worker!: ControlledWorker;
  const h = custom(async options => {
    worker = new ControlledWorker(options);
    const send = worker.send.bind(worker);
    worker.send = (...args) => { void send(...args); return write.promise; };
    worker.stop = () => { worker.stops++; write.reject(new Error("Claude session worker stopped")); };
    return worker;
  }, 1_000);
  const first = h.run();
  await until(() => !!worker?.sends.length);
  const original = new ClaudeSessionError("Synthetic provider throttling", "claude_cli_rate_limit", 429, true);
  worker.options.onError(original);
  const delivered = await rejection(first.done);
  expect(delivered).toBe(original);
  expect(delivered).toMatchObject({ code: "claude_cli_rate_limit", status: 429, retryable: true });
  expect(worker.stops).toBe(1); expect(first.events).toEqual([]);
});
