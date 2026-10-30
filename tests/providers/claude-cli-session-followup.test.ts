import { describe, expect, spyOn, test } from "bun:test";
import { claudeHistory } from "../../src/adapters/claude-cli/session-context";
import type { ClaudePersistentTurn } from "../../src/adapters/claude-cli/session-manager";
import { resetProviderRequestPacingForTest, waitForProviderRequestSlot, type ProviderRequestSlot } from "../../src/providers/request-pacing";
import { assistant, call, harness, result, toolAssistant, turn, until, user, withMessages } from "../fixtures/claude-session-worker";

const instructionB = "Keep B local before continuing";

function operation(input: ClaudePersistentTurn, id: string, messages = input.parsed.context.messages): ClaudePersistentTurn {
  const next = withMessages(input, messages);
  next.incoming = { ...input.incoming, headers: new Headers(input.incoming.headers) };
  next.incoming.headers.set("idempotency-key", id);
  return next;
}

async function handoff(count = 1) {
  const h = harness();
  const input = operation(turn(), "operation-A");
  const first = await h.start(input);
  const actions = Array.from({ length: count }, (_, index) => call(input, `call-A-${index}`, { path: `a-${index}.txt` }));
  const pending = actions.map(action => first.worker.mcp(action));
  await first.worker.blocks(actions);
  await first.done;
  const history = [...input.parsed.context.messages, toolAssistant(actions)];
  const bInput = operation(input, "operation-B", [...history, user(instructionB)]);
  return { h, input, first, actions, pending, history, bInput };
}

async function queued(h: ReturnType<typeof harness>, input: ClaudePersistentTurn) {
  const response = h.run(input);
  await until(() => response.events.some(event => event.type === "heartbeat"));
  expect(response.events).toEqual([{ type: "heartbeat", replayUnsafe: true }]);
  expect(h.journal.load(input.scope)!.instructions).toContainEqual(expect.objectContaining({ texts: [instructionB], phase: "queued" }));
  return response;
}

describe("Claude durable follow-up ownership", () => {
  test.each([false, true])("sent B survives C omitting interrupted partial A and B itself (cold=%s)", async cold => {
    const h = harness();
    const input = operation(turn(), "operation-A");
    const first = await h.start(input);
    const partial = "Unfinished partial A that the client later omits";
    await first.worker.begin();
    await first.worker.raw({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
    await first.worker.raw({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: partial } });
    expect(first.events).toContainEqual({ type: "text_delta", text: partial });
    const b = h.run(operation(input, "operation-B", [...input.parsed.context.messages, assistant(partial), user(instructionB)]));
    await until(() => first.worker.sends.length === 2);
    await first.done;
    expect(first.worker.sends[1]).toBe(instructionB);
    const saved = h.journal.load(input.scope)!.instructions![0]!;
    expect(saved).toMatchObject({ history: claudeHistory(input.parsed.context.messages), texts: [instructionB], phase: "sent" });

    if (cold) h.manager.close();
    const owner = cold ? harness(h.journal) : h;
    const c = await owner.start(operation(input, "operation-C", [...input.parsed.context.messages, user("Continue with C")]));
    await b.done;
    expect(c.worker.sends[0]!.split(instructionB)).toHaveLength(2);
    expect(c.worker.sends[0]).toContain("Continue with C");
    expect(c.worker.sends[0]).not.toContain(partial);
    await c.worker.finish("C retained B without partial A");
    await c.done;
    expect(c.events.some(event => event.type === "error")).toBe(false);
    expect(c.events).toContainEqual({ type: "text_delta", text: "C retained B without partial A" });
    expect(owner.journal.load(input.scope)!.instructions).toContainEqual(expect.objectContaining({ inputHash: saved.inputHash, phase: "completed" }));
    expect(owner.journal.load(input.scope)!.turns.at(-1)!.requestHashes).toContain(saved.inputHash);
  });

  test("cold raw B retry replaces its interrupted receipt, waits for A, then C applies B and adopts its replay alias", async () => {
    const { h, input, first, actions, pending, history, bInput } = await handoff();
    const b = await queued(h, bInput);
    const inputHash = h.journal.load(input.scope)!.instructions![0]!.inputHash;
    h.manager.close();
    await b.done;
    await expect(pending[0]!).rejects.toThrow("worker stopped");
    expect(first.worker.stops).toBe(1);
    expect(h.journal.load(input.scope)!.turns.find(receipt => receipt.inputHash === inputHash)!.phase).toBe("interrupted");

    const reopened = harness(h.journal);
    const retried = await queued(reopened, bInput);
    const receipts = reopened.journal.load(input.scope)!.turns.filter(receipt => receipt.inputHash === inputHash);
    expect(receipts).toHaveLength(1);
    expect(receipts[0]!.phase).toBe("running");
    expect(reopened.workers).toHaveLength(0);
    let settled = false;
    void retried.done.then(() => { settled = true; });
    await Bun.sleep(0);
    expect(settled).toBe(false);

    const c = await reopened.start(operation(input, "operation-C", [...history, result(actions[0]!.id, "actual A output")]));
    expect(c.worker.sends[0]!.split(instructionB)).toHaveLength(2);
    expect(c.worker.sends[0]).toContain("actual A output");
    await retried.done;
    expect(retried.events.some(event => event.type === "error")).toBe(false);
    await c.worker.finish("C applied B");
    await c.done;
    const completed = reopened.journal.load(input.scope)!;
    expect(completed.instructions).toContainEqual(expect.objectContaining({ inputHash, texts: [instructionB], phase: "completed" }));
    expect(completed.turns.at(-1)!.requestHashes).toContain(inputHash);
    const replay = reopened.run(bInput);
    await replay.done;
    expect(replay.events).toContainEqual({ type: "text_delta", text: "C applied B" });
    expect(replay.events.at(-1)).toMatchObject({ type: "done", usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 } });
    expect(c.worker.sends).toHaveLength(1);
  });

  test("a failed initial queue commit leaves A's generation and consumer untouched", async () => {
    const h = harness();
    const input = operation(turn(), "operation-A");
    const first = await h.start(input);
    const before = h.journal.load(input.scope);
    let settled = false;
    void first.done.then(() => { settled = true; });
    h.journal.rejectSave = snapshot => snapshot.instructions?.some(entry => entry.texts.includes(instructionB)) === true;
    const b = h.run(operation(input, "operation-B", [...input.parsed.context.messages, user(instructionB)]));
    await expect(b.done).rejects.toThrow("injected journal failure");
    expect(b.events).toEqual([]);
    expect(h.journal.load(input.scope)).toEqual(before);
    expect(first.worker.interrupts).toBe(0);
    expect(first.worker.stops).toBe(0);
    expect(first.worker.sends).toHaveLength(1);
    expect(settled).toBe(false);
    h.journal.rejectSave = undefined;
    await first.worker.finish("A kept its consumer");
    await first.done;
    expect(first.events).toContainEqual({ type: "text_delta", text: "A kept its consumer" });
    expect(first.events.at(-1)).toMatchObject({ type: "done", stopReason: "stop" });
    expect(first.events.some(event => event.type === "error" || event.type === "incomplete")).toBe(false);
  });

  test.each([false, true])("a conflicting partial A result retains queued B and its current consumer (error-bit=%s)", async errorBit => {
    const { h, input, first, actions, pending, history, bInput } = await handoff(2);
    const b = await queued(h, bInput);
    const partialInput = operation(input, "partial-A", [...history, result(actions[0]!.id, "actual A0 output")]);
    const waiting = await queued(h, partialInput);
    await b.done;
    const before = h.journal.load(input.scope);
    let settled = false;
    void waiting.done.then(() => { settled = true; });
    const conflict = h.run(operation(input, "conflicting-A", [...history,
      result(actions[0]!.id, errorBit ? "actual A0 output" : "changed output", errorBit)]));
    await expect(conflict.done).rejects.toMatchObject({ code: "claude_tool_result_conflict" });
    expect(conflict.events).toEqual([]);
    expect(h.journal.load(input.scope)).toEqual(before);
    expect(settled).toBe(false);
    expect(first.worker.interrupts).toBe(0);
    expect(first.worker.stops).toBe(0);
    expect(first.worker.sends).toHaveLength(1);

    const c = h.run(operation(input, "operation-C", [...history,
      result(actions[0]!.id, "actual A0 output"), result(actions[1]!.id, "actual A1 output")]));
    expect((await Promise.all(pending)).map(output => output.content[0]!.text)).toEqual(["actual A0 output", "actual A1 output"]);
    await waiting.done;
    expect(waiting.events.some(event => event.type === "error")).toBe(false);
    expect(first.worker.sends).toEqual([first.worker.sends[0]!, instructionB]);
    await first.worker.finish("C retained B after refusal");
    await c.done;
    expect(c.events).toContainEqual({ type: "text_delta", text: "C retained B after refusal" });
  });

  test("initial waiting B and its reconnect each release their own cap-one pacing slot exactly once", async () => {
    const { h, input, first, actions, pending, history, bInput } = await handoff();
    const provider = { ...input.provider, requestPacing: { enabled: true, maxConcurrentRequests: 1 } };
    const slots: ProviderRequestSlot[] = [];
    const acquire = async () => {
      const slot = await waitForProviderRequestSlot("claude-followup-cap-one", provider, input.parsed.modelId, AbortSignal.timeout(1000));
      slots.push(slot);
      return slot;
    };
    const initialSlot = await acquire();
    expect(initialSlot.leased).toBe(true);
    const initialRelease = spyOn(initialSlot, "release");
    const releaseSpies = [initialRelease];
    try {
      const b = await queued(h, { ...bInput, incoming: { ...bInput.incoming, pacingSlot: initialSlot } });
      expect(initialRelease).toHaveBeenCalledTimes(1);
      expect(initialSlot.released).toBe(true);
      const reconnectSlot = await acquire();
      const reconnectRelease = spyOn(reconnectSlot, "release");
      releaseSpies.push(reconnectRelease);
      const reconnect = await queued(h, { ...bInput, incoming: { ...bInput.incoming, pacingSlot: reconnectSlot } });
      await b.done;
      expect(initialRelease).toHaveBeenCalledTimes(1);
      expect(reconnectRelease).toHaveBeenCalledTimes(1);
      expect(reconnectSlot.released).toBe(true);

      const cSlot = await acquire();
      const cInput = operation(input, "operation-C", [...history, result(actions[0]!.id, "actual A output")]);
      const c = h.run({ ...cInput, incoming: { ...cInput.incoming, pacingSlot: cSlot } });
      expect((await pending[0]!).content[0]!.text).toBe("actual A output");
      await reconnect.done;
      await first.worker.finish("cap one continued");
      await c.done;
      expect(c.events).toContainEqual({ type: "text_delta", text: "cap one continued" });
      expect(initialRelease).toHaveBeenCalledTimes(1);
      expect(reconnectRelease).toHaveBeenCalledTimes(1);
    } finally {
      for (const release of releaseSpies) release.mockRestore();
      for (const slot of slots) slot.release();
      resetProviderRequestPacingForTest();
    }
  });

  test("an old A MCP signal aborted after SDK interruption cannot stop B", async () => {
    const h = harness();
    const input = operation(turn(), "operation-A");
    const first = await h.start(input);
    const abort = new AbortController();
    const pendingA = first.worker.mcp(call(input, "unfinished-A"), abort.signal);
    const b = h.run(operation(input, "operation-B", [...input.parsed.context.messages, user(instructionB)]));
    await until(() => first.worker.sends.length === 2);
    await first.done;
    expect(first.worker.interrupts).toBe(1);
    abort.abort();
    await expect(pendingA).rejects.toThrow();
    expect(first.worker.stops).toBe(0);
    expect(h.workers).toEqual([first.worker]);
    await first.worker.finish("B survived A's cancelled MCP");
    await b.done;
    expect(b.events).toContainEqual({ type: "text_delta", text: "B survived A's cancelled MCP" });
    expect(b.events.some(event => event.type === "error")).toBe(false);
    expect(first.worker.sends).toEqual([first.worker.sends[0]!, instructionB]);
  });

  test("SDK interrupt acknowledgement cannot send B before A's terminal frame drains", async () => {
    const h = harness();
    const input = operation(turn(), "operation-A");
    const first = await h.start(input);
    const acknowledged = Promise.withResolvers<void>();
    first.worker.interrupt = async () => { first.worker.interrupts++; acknowledged.resolve(); };
    let firstSettled = false;
    void first.done.then(() => { firstSettled = true; });
    const b = h.run(operation(input, "operation-B", [...input.parsed.context.messages, user(instructionB)]));
    await acknowledged.promise;
    await Bun.sleep(0);
    expect(first.worker.sends).toHaveLength(1);
    expect(firstSettled).toBe(false);
    expect(h.journal.load(input.scope)!.instructions).toContainEqual(expect.objectContaining({ texts: [instructionB], phase: "queued" }));
    await first.worker.emit({ type: "result", user_message_uuid: first.worker.sendOptions[0]!.uuid,
      subtype: "success", is_error: false, result: "", usage: { input_tokens: 0, output_tokens: 0 } });
    await until(() => first.worker.sends.length === 2);
    await first.done;
    expect(first.events.at(-1)).toMatchObject({ type: "incomplete", reason: "superseded" });
    expect(first.worker.sends[1]).toBe(instructionB);
    expect(first.worker.stops).toBe(0);
    await first.worker.finish("B began after A drained");
    await b.done;
    expect(b.events).toContainEqual({ type: "text_delta", text: "B began after A drained" });
    expect(b.events.some(event => event.type === "error")).toBe(false);
  });

  test("an unexpected old-A terminal while B waits fails recoverably and retains the instruction queue", async () => {
    const { h, input, first, pending, bInput } = await handoff();
    const b = await queued(h, bInput);
    const before = h.journal.load(input.scope)!.instructions;
    await first.worker.emit({ type: "result", user_message_uuid: first.worker.sendOptions[0]!.uuid,
      subtype: "success", is_error: false, result: "unexpected A completion", usage: { input_tokens: 0, output_tokens: 0 } });
    await b.done;
    expect(b.events.at(-1)).toMatchObject({ type: "error", code: "claude_tool_continuation_lost", status: 502, retryable: true });
    expect(h.journal.load(input.scope)!.instructions).toEqual(before);
    expect(h.journal.load(input.scope)!.turns[0]!.phase).toBe("tools");
    expect(first.worker.sends).toHaveLength(1);
    expect(first.worker.stops).toBe(1);
    await expect(pending[0]!).rejects.toThrow("worker stopped");
  });

  test("a late A result cannot drain interrupted B or start C before B's own terminal", async () => {
    const h = harness();
    const input = operation(turn(), "operation-A");
    const first = await h.start(input);
    const aUuid = first.worker.sendOptions[0]!.uuid;
    const bInput = operation(input, "operation-B", [...input.parsed.context.messages, user(instructionB)]);
    const b = h.run(bInput);
    await until(() => first.worker.sends.length === 2);
    await first.done;
    const bUuid = first.worker.sendOptions[1]!.uuid;
    expect(bUuid).toBeDefined();
    expect(bUuid).not.toBe(aUuid);
    const acknowledged = Promise.withResolvers<void>();
    first.worker.interrupt = async () => { first.worker.interrupts++; acknowledged.resolve(); };
    let bSettled = false;
    void b.done.then(() => { bSettled = true; });
    const c = h.run(operation(input, "operation-C", [...bInput.parsed.context.messages, user("Begin C only after B drains")]));
    await acknowledged.promise;
    await Bun.sleep(0);
    expect(first.worker.sends).toHaveLength(2);
    await first.worker.emit({ type: "result", user_message_uuid: aUuid, subtype: "success", is_error: false,
      result: "late A terminal", usage: { input_tokens: 100, output_tokens: 50 } });
    await Bun.sleep(0);
    expect(bSettled).toBe(false);
    expect(first.worker.sends).toHaveLength(2);
    expect(c.events).toEqual([]);
    expect(first.worker.stops).toBe(0);

    await first.worker.emit({ type: "result", user_message_uuid: bUuid, subtype: "success", is_error: false,
      result: "B terminal", usage: { input_tokens: 7, output_tokens: 2 } });
    await until(() => first.worker.sends.length === 3);
    await b.done;
    expect(b.events.at(-1)).toMatchObject({ type: "incomplete", reason: "superseded", usage: { inputTokens: 7, outputTokens: 2 } });
    expect(first.worker.sends[2]).toBe("Begin C only after B drains");
    expect(h.workers).toEqual([first.worker]);
    await first.worker.finish("C followed B's own terminal");
    await c.done;
    expect(c.events).toContainEqual({ type: "text_delta", text: "C followed B's own terminal" });
    expect(c.events.some(event => event.type === "error")).toBe(false);
  });

  test("interrupted A saves full terminal usage and stable A replay reports zero incomplete usage", async () => {
    const h = harness();
    const input = operation(turn(), "operation-A");
    const first = await h.start(input);
    await first.worker.begin();
    await first.worker.raw({ type: "message_delta", delta: { stop_reason: null, stop_sequence: null }, usage: { output_tokens: 2 } });
    first.worker.interrupt = async () => {
      first.worker.interrupts++;
      await first.worker.emit({ type: "result", user_message_uuid: first.worker.sendOptions[0]!.uuid,
        subtype: "error_during_execution", is_error: true, errors: ["interrupted"],
        usage: { input_tokens: 41, output_tokens: 17, cache_read_input_tokens: 7, cache_creation_input_tokens: 5 } });
    };
    const b = h.run(operation(input, "operation-B", [...input.parsed.context.messages, user(instructionB)]));
    await until(() => first.worker.sends.length === 2);
    await first.done;
    expect(first.events.at(-1)).toMatchObject({ type: "incomplete", reason: "superseded", usage: {
      inputTokens: 53, outputTokens: 17, totalTokens: 70, cachedInputTokens: 7, cacheReadInputTokens: 7, cacheCreationInputTokens: 5,
    } });
    const saved = h.journal.load(input.scope)!;
    expect(saved.turns[0]!.phase).toBe("interrupted");
    expect(saved.turns[0]!.events.at(-1)).toEqual(first.events.at(-1));
    const replay = h.run(input);
    await replay.done;
    expect(replay.events.at(-1)).toEqual({ ...first.events.at(-1)!, usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 } });
    expect(h.journal.load(input.scope)).toEqual(saved);
    expect(first.worker.sends).toHaveLength(2);
    expect(first.worker.stops).toBe(0);
    await first.worker.finish("B kept running after A usage replay");
    await b.done;
    expect(b.events).toContainEqual({ type: "text_delta", text: "B kept running after A usage replay" });
    expect(b.events.some(event => event.type === "error")).toBe(false);
  });
});
