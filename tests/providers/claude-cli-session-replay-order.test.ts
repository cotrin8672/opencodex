import { describe, expect, test } from "bun:test";
import type { ClaudePersistentTurn } from "../../src/adapters/claude-cli/session-manager";
import type { ControlledWorker } from "../fixtures/claude-session-worker";
import { assistant, call, harness, result, toolAssistant, turn, until, user, withMessages } from "../fixtures/claude-session-worker";

function operation(input: ClaudePersistentTurn, id: string, messages = input.parsed.context.messages): ClaudePersistentTurn {
  const next = withMessages(input, messages);
  next.incoming = { ...input.incoming, headers: new Headers(input.incoming.headers) };
  next.incoming.headers.set("idempotency-key", id);
  return next;
}

async function beginNewer(worker: ControlledWorker) {
  await worker.begin();
  await worker.raw({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
  await worker.raw({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "B before retry; " } });
}

async function finishNewer(h: ReturnType<typeof harness>, scope: string, worker: ControlledWorker,
  newer: ReturnType<ReturnType<typeof harness>["run"]>, sends = 2) {
  expect(worker.stops).toBe(0);
  expect(h.workers).toEqual([worker]);
  expect(h.journal.load(scope)!.turns.at(-1)!.phase).toBe("running");
  expect(newer.events).toEqual([{ type: "text_delta", text: "B before retry; " }]);
  await worker.raw({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "B after retry" } });
  await worker.raw({ type: "content_block_stop", index: 0 });
  await worker.raw({ type: "message_stop" });
  await worker.finish("B before retry; B after retry", false);
  await newer.done;
  expect(newer.events.filter(event => event.type === "text_delta")).toEqual([
    { type: "text_delta", text: "B before retry; " }, { type: "text_delta", text: "B after retry" },
  ]);
  expect(newer.events.filter(event => event.type === "error" || event.type.startsWith("tool_call_"))).toEqual([]);
  expect(newer.events.at(-1)).toMatchObject({ type: "done", stopReason: "stop" });
  expect(h.journal.load(scope)!.turns.at(-1)).toMatchObject({ phase: "completed", events: newer.events });
  expect(worker.sends).toHaveLength(sends);
  expect(worker.stops).toBe(0);
}

async function completedTextWithNewer() {
  const h = harness();
  const input = operation(turn([user("request A")]), "text-A");
  const first = await h.start(input);
  await first.worker.finish("answer A");
  await first.done;
  const nextInput = operation(input, "text-B", [...input.parsed.context.messages, assistant("answer A"), user("request B")]);
  const newer = h.run(nextInput);
  await until(() => first.worker.sends.length === 2);
  await beginNewer(first.worker);
  return { h, input, first, newer };
}

async function consumedToolsWithNewer() {
  const h = harness();
  const input = operation(turn([user("request A: read a.txt")]), "tool-A");
  const first = await h.start(input);
  const action = call(input, "toolu_A", { path: "a.txt" });
  const pending = first.worker.mcp(action);
  await first.worker.blocks([action]);
  await first.done;
  const history = [...input.parsed.context.messages, toolAssistant([action]), result(action.id, "actual A output")];
  const continuedInput = operation(input, "tool-A-results", history);
  const continued = h.run(continuedInput);
  expect((await pending).content).toEqual([{ type: "text", text: "actual A output" }]);
  await first.worker.finish("A finished");
  await continued.done;
  const newer = h.run(operation(input, "tool-B", [...history, assistant("A finished"), user("request B")]));
  await until(() => first.worker.sends.length === 2);
  await beginNewer(first.worker);
  return { h, input, first, action, continuedInput, newer };
}

describe("Claude historical replay does not own a newer generation", () => {
  test("replays completed text A while B keeps its worker and consumer", async () => {
    const { h, input, first, newer } = await completedTextWithNewer();
    const before = h.journal.load(input.scope);
    const replay = h.run(input);
    await replay.done;
    expect(replay.events.filter(event => event.type !== "done")).toEqual(first.events.filter(event => event.type !== "done"));
    expect(replay.events.at(-1)).toMatchObject({ type: "done", usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 } });
    expect(h.journal.load(input.scope)).toEqual(before);
    await finishNewer(h, input.scope, first.worker, newer);
  });

  test("refuses consumed tool A without duplicating its action or cancelling B", async () => {
    const { h, input, first, newer } = await consumedToolsWithNewer();
    const before = h.journal.load(input.scope);
    const replay = h.run(input);
    await expect(replay.done).rejects.toMatchObject({ code: "claude_tool_delivery_uncertain" });
    expect(replay.events).toEqual([]);
    expect(h.journal.load(input.scope)).toEqual(before);
    await finishNewer(h, input.scope, first.worker, newer);
  });

  test("a rejected A replay emitter cannot cancel B's worker or consumer", async () => {
    const { h, input, first, newer } = await completedTextWithNewer();
    const before = h.journal.load(input.scope);
    const replay = h.run(input, () => { throw new Error("old replay consumer disconnected"); });
    await expect(replay.done).rejects.toThrow("old replay consumer disconnected");
    expect(h.journal.load(input.scope)).toEqual(before);
    await finishNewer(h, input.scope, first.worker, newer);
  });

  test.each(["content", "error-bit"] as const)("normalizes stale A's completed %s and replays its continuation while B continues", async conflict => {
    const { h, input, first, action, continuedInput, newer } = await consumedToolsWithNewer();
    const stale = withMessages(continuedInput, [...input.parsed.context.messages, toolAssistant([action]),
      result(action.id, conflict === "content" ? "changed output" : "actual A output", conflict === "error-bit")]);
    const before = h.journal.load(input.scope);
    const replay = h.run(stale);
    await replay.done;
    expect(replay.events).toContainEqual({ type: "text_delta", text: "A finished" });
    expect(replay.events.filter(event => event.type.startsWith("tool_call_"))).toEqual([]);
    expect(replay.events.at(-1)).toMatchObject({ type: "done", usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 } });
    expect(h.journal.load(input.scope)).toEqual(before);
    await finishNewer(h, input.scope, first.worker, newer);
  });

  test("rejects stale A's conflicting tool identity while B continues", async () => {
    const { h, input, first, action, continuedInput, newer } = await consumedToolsWithNewer();
    const stale = withMessages(continuedInput, [...input.parsed.context.messages,
      toolAssistant([{ ...action, arguments: { path: "changed.txt" } }]), result(action.id, "actual A output")]);
    const before = h.journal.load(input.scope);
    const replay = h.run(stale);
    await expect(replay.done).rejects.toMatchObject({ code: "claude_tool_id_conflict" });
    expect(replay.events).toEqual([]);
    expect(h.journal.load(input.scope)).toEqual(before);
    await finishNewer(h, input.scope, first.worker, newer);
  });

  test("an expired A receipt refuses replay while B continues", async () => {
    const h = harness();
    const input = operation(turn([user("request A")]), "expired-A");
    const first = await h.start(input);
    await first.worker.finish("answer A");
    await first.done;
    let history = [...input.parsed.context.messages, assistant("answer A")];
    for (let index = 0; index < 16; index++) {
      const messages = [...history, user(`intervening request ${index}`)];
      const intervening = h.run(operation(input, `intervening-${index}`, messages));
      await until(() => first.worker.sends.length === index + 2);
      await first.worker.finish(`intervening answer ${index}`);
      await intervening.done;
      history = [...messages, assistant(`intervening answer ${index}`)];
    }
    expect(h.journal.load(input.scope)!.turns[0]!.phase).toBe("expired");
    const newer = h.run(operation(input, "expired-B", [...history, user("request B")]));
    await until(() => first.worker.sends.length === 18);
    await beginNewer(first.worker);
    const before = h.journal.load(input.scope);
    const replay = h.run(input);
    await expect(replay.done).rejects.toMatchObject({ code: "claude_replay_expired" });
    expect(replay.events).toEqual([]);
    expect(h.journal.load(input.scope)).toEqual(before);
    expect(first.worker.stops).toBe(0);
    await finishNewer(h, input.scope, first.worker, newer, 18);
  });

  test("reading A's text receipt needs no snapshot write even when writes fail, and leaves B alive", async () => {
    const { h, input, first, newer } = await completedTextWithNewer();
    let writes = 0;
    h.journal.rejectSave = () => { writes++; return true; };
    const before = h.journal.load(input.scope);
    const replay = h.run(input);
    await replay.done;
    expect(replay.events).toContainEqual({ type: "text_delta", text: "answer A" });
    expect(replay.events.at(-1)).toMatchObject({ type: "done", stopReason: "stop" });
    expect(writes).toBe(0);
    expect(h.journal.load(input.scope)).toEqual(before);
    h.journal.rejectSave = undefined;
    await finishNewer(h, input.scope, first.worker, newer);
  });
});
