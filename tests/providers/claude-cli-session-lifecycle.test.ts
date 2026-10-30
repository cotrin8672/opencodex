import { describe, expect, test } from "bun:test";
import { MemoryJournal, assistant, call, harness, result, toolAssistant, turn, until, user, withMessages } from "../fixtures/claude-session-worker";

describe("Claude session interruption and recovery", () => {
  test("a cancelled native MCP request cannot be handed to the external client", async () => {
    const h = harness();
    const input = turn();
    const first = await h.start(input);
    const action = call(input, "cancelled_mcp");
    const abort = new AbortController();
    const pending = first.worker.mcp(action, abort.signal);
    abort.abort();
    await expect(pending).rejects.toMatchObject({ code: "claude_mcp_call_cancelled" });
    await first.worker.blocks([action]);
    await first.done;
    expect(first.events.some(event => event.type === "tool_call_start")).toBe(false);
    expect(first.events.at(-1)).toMatchObject({ type: "error", code: "claude_mcp_call_cancelled" });
    expect(first.worker.stops).toBe(1);
  });

  test("hands sequential native MCP calls to the client without waiting for the next call", async () => {
    const h = harness();
    const input = turn();
    let sends = 0;
    input.incoming = { ...input.incoming, onPhysicalSend: () => { sends++; } };
    const first = await h.start(input);
    const a = call(input, "sequential_a", { path: "a.txt" });
    const b = call(input, "sequential_b", { path: "b.txt" });
    const pendingA = first.worker.mcp(a);
    await first.worker.blocks([a, b]);
    await first.done;
    expect(first.events.filter(event => event.type === "tool_call_start").map(event => event.id)).toEqual([a.id]);
    const historyA = [...input.parsed.context.messages, toolAssistant([a]), result(a.id, "output a")];
    const second = h.run(withMessages(input, historyA));
    expect((await pendingA).content[0]!.text).toBe("output a");
    const pendingB = first.worker.mcp(b);
    await second.done;
    expect(second.events.filter(event => event.type === "tool_call_start").map(event => event.id)).toEqual([b.id]);
    expect(second.events.at(-1)).not.toHaveProperty("usage");
    expect(sends).toBe(1);
    const historyB = [...historyA, toolAssistant([b]), result(b.id, "output b")];
    const final = h.run(withMessages(input, historyB));
    expect((await pendingB).content[0]!.text).toBe("output b");
    await first.worker.finish("both outputs received");
    await final.done;
    expect(sends).toBe(2);
    expect(h.workers).toHaveLength(1);
    const next = h.run(withMessages(input, [...historyB, assistant("both outputs received"), user("next")]));
    await until(() => first.worker.sends.length === 2);
    await first.worker.finish("next answer");
    await next.done;
    expect(first.worker.sends[1]).toBe("next");
  });

  test("supersedes active generation, retains all new instructions, and ignores stale worker events", async () => {
    const h = harness();
    const input = turn([user("remember the original context")]);
    const first = await h.start(input);
    await first.worker.begin();
    const replacement = h.run(withMessages(input, [...input.parsed.context.messages, user("new steering instruction")]));
    await until(() => h.workers[1]?.sends.length === 1);
    await first.done;
    expect(first.events.at(-1)).toMatchObject({ type: "error", code: "claude_turn_superseded" });
    expect(first.worker.stops).toBe(1);
    expect(h.workers[1]!.sends[0]).toContain("remember the original context");
    expect(h.workers[1]!.sends[0]).toContain("new steering instruction");
    first.worker.options.onError(new Error("late error from replaced worker"));
    await first.worker.finish("stale output");
    await h.workers[1]!.finish("new output");
    await replacement.done;
    expect(replacement.events).toContainEqual({ type: "text_delta", text: "new output" });
    expect(replacement.events.some(event => event.type === "error" || (event.type === "text_delta" && event.text === "stale output"))).toBe(false);
  });

  test("a reconnect takes ownership of the same generation without another send", async () => {
    const h = harness();
    const input = turn();
    const first = await h.start(input);
    const reconnect = h.run(input);
    await first.done;
    await first.worker.finish("same generation");
    await reconnect.done;
    expect(first.worker.sends).toHaveLength(1);
    expect(h.workers).toHaveLength(1);
    expect(reconnect.events).toContainEqual({ type: "text_delta", text: "same generation" });
  });

  test.each([false, true])("a new prompt discards only an unpublished batch (cold=%s)", async cold => {
    const journal = new MemoryJournal();
    let h = harness(journal);
    const input = turn();
    const abort = new AbortController();
    const first = await h.start({ ...input, incoming: { ...input.incoming, abortSignal: abort.signal } });
    abort.abort();
    await first.done;
    const action = call(input, "never_delivered");
    first.worker.mcp(action);
    await first.worker.blocks([action]);
    expect(journal.load(input.scope)!.turns[0]!.deliveryStarted).not.toBe(true);
    if (cold) { h.manager.close(); h = harness(journal); }
    const next = await h.start(withMessages(input, [user("replacement prompt")]));
    await next.worker.finish("replacement answer");
    await next.done;
    expect(next.worker.sends[0]).toContain("replacement prompt");
    expect(next.events.some(event => event.type === "tool_call_start")).toBe(false);
  });

  test("an instruction inserted before a tool result forces a complete history bootstrap", async () => {
    const h = harness();
    const input = turn();
    const first = await h.start(input);
    const action = call(input, "steering_tool");
    first.worker.mcp(action);
    await first.worker.blocks([action]);
    await first.done;
    const next = await h.start(withMessages(input, [...input.parsed.context.messages, toolAssistant([action]),
      user("use this additional instruction"), result(action.id, "actual tool output")]));
    expect(first.worker.stops).toBe(1);
    expect(next.worker.sends[0]).toContain("use this additional instruction");
    expect(next.worker.sends[0]).toContain("actual tool output");
    await next.worker.finish("instruction included");
    await next.done;
  });

  test.each(["different", "absent"])("equal input is a new completed operation when its stable ID is %s", async mode => {
    const h = harness();
    const input = turn();
    if (mode === "absent") input.incoming.headers.delete("idempotency-key");
    const first = await h.start(input);
    await first.worker.finish("first answer");
    await first.done;
    const nextInput = { ...input, incoming: { ...input.incoming, headers: new Headers(input.incoming.headers) } };
    if (mode === "different") nextInput.incoming.headers.set("idempotency-key", "different-operation");
    const next = await h.start(nextInput);
    await next.worker.finish("new answer");
    await next.done;
    expect(next.events).toContainEqual({ type: "text_delta", text: "new answer" });
    expect(next.events).not.toContainEqual({ type: "text_delta", text: "first answer" });
  });

  test("preserves rate-limit retryability, classification, and usage", async () => {
    const h = harness();
    const first = await h.start();
    await first.worker.emit({ type: "result", subtype: "error_during_execution", is_error: true,
      errors: ["Rate limit exceeded"], usage: { input_tokens: 100, output_tokens: 7 } });
    await first.done;
    expect(first.events.at(-1)).toMatchObject({ type: "error", status: 429, retryable: true,
      errorType: "rate_limit_error", usage: { inputTokens: 100, outputTokens: 7 } });
  });

  test("a failed result commit leaves the live receipt untouched for a corrected retry", async () => {
    const h = harness();
    const input = turn();
    const first = await h.start(input);
    const action = call(input, "atomic_result");
    const pending = first.worker.mcp(action);
    await first.worker.blocks([action]);
    await first.done;
    h.journal.rejectSave = snapshot => snapshot.turns.some(turn => turn.calls.some(call => !!call.result));
    const history = [...input.parsed.context.messages, toolAssistant([action])];
    await expect(h.run(withMessages(input, [...history, result(action.id, "not committed")])).done).rejects.toThrow("journal failure");
    expect(h.journal.load(input.scope)!.turns[0]!.calls[0]!.result).toBeUndefined();
    h.journal.rejectSave = undefined;
    const retry = h.run(withMessages(input, [...history, result(action.id, "corrected result")]));
    expect((await pending).content[0]!.text).toBe("corrected result");
    await first.worker.finish("corrected");
    await retry.done;
  });

  test("restores a saved output after dispatch refusal and rejects repetition under a fresh call ID", async () => {
    const journal = new MemoryJournal();
    const h = harness(journal);
    const input = turn();
    const first = await h.start(input);
    const action = call(input, "saved_original");
    const pending = first.worker.mcp(action);
    await first.worker.blocks([action]);
    await first.done;
    const history = [...input.parsed.context.messages, toolAssistant([action])];
    const next = withMessages(input, [...history, result(action.id, "UNIQUE_SAVED_OUTPUT")]);
    await expect(h.run({ ...next, incoming: { ...next.incoming, beforePhysicalSend: () => { throw new Error("dispatch refused"); } } }).done)
      .rejects.toThrow("dispatch refused");
    await expect(pending).rejects.toThrow("worker stopped");
    expect(journal.load(input.scope)!.turns[0]!.phase).toBe("tools");
    h.manager.close();
    const reopened = harness(journal);
    const recovered = await reopened.start(withMessages(input, [...history, user("continue")]));
    expect(recovered.worker.sends[0]).toContain("UNIQUE_SAVED_OUTPUT");
    expect(recovered.worker.sends[0]).toContain("continue");
    const repeated = call(input, "fresh_repeated");
    recovered.worker.mcp(repeated);
    await recovered.worker.blocks([repeated]);
    await recovered.done;
    expect(recovered.events.some(event => event.type === "tool_call_start")).toBe(false);
    expect(recovered.events.at(-1)).toMatchObject({ type: "error", code: "claude_recovery_repeat_refused" });
  });
});
