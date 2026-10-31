import { describe, expect, test } from "bun:test";
import { claudeDigest, claudeHistory, claudeSessionScope } from "../../src/adapters/claude-cli/session-context";
import { stableClaudeToolBridge } from "../../src/adapters/claude-cli/stable-replay";
import { buildCodeBuddyToolBridge } from "../../src/adapters/codebuddy/tool-bridge";
import { MemoryJournal, assistant, call, harness, result, toolAssistant, turn, until, user, withMessages } from "../fixtures/claude-session-worker";

describe("Claude persistent session manager (offline adversarial)", () => {
  test.each(["assistant", "result"] as const)("observes an accepted %s error once without losing usage or classification", async frame => {
    for (const observerThrows of [false, true]) {
      const h = harness();
      const input = turn();
      const observed: string[] = [];
      const detail = "Rate limit exceeded · resets 3pm (Asia/Tokyo)";
      input.onCliOutputError = message => {
        observed.push(message);
        if (observerThrows) throw new Error("injected usage observer failure");
      };
      const response = await h.start(input);
      await response.worker.begin();
      const terminal = { type: "result", subtype: "error_during_execution", is_error: true,
        errors: [detail], usage: { input_tokens: 100, output_tokens: 7 } };
      const failure = frame === "assistant" ? { type: "assistant", error: "rate_limit", message: {
        role: "assistant", content: [{ type: "text", text: detail }], usage: { input_tokens: 100, output_tokens: 7 },
      } } : terminal;
      await response.worker.emit({ ...failure, user_message_uuid: crypto.randomUUID() });
      expect(observed).toEqual([]);
      await response.worker.emit(failure);
      await response.done;
      expect(observed).toEqual([detail]);
      expect(response.events).toEqual([{ type: "error", message: detail, status: 429,
        code: "rate_limit_exceeded", retryable: true, errorType: "rate_limit_error",
        usage: { inputTokens: 100, outputTokens: 7, totalTokens: 107 } }]);
      await response.worker.emit(terminal);
      expect(observed).toEqual([detail]);
      expect(response.events).toHaveLength(1);
      expect(response.worker.stops).toBe(1);
      await h.manager.close();
    }
  });

  test("reuses one native worker through identical parallel calls, results, final text and another user", async () => {
    const h = harness();
    const input = turn();
    const first = await h.start(input);
    const calls = [call(input, "toolu_left"), call(input, "toolu_right")];
    const right = first.worker.mcp(calls[1]!);
    const left = first.worker.mcp(calls[0]!);
    expect(first.worker.mcp(calls[0]!)).toBe(left);
    await expect(first.worker.mcp(call(input, "toolu_left", { path: "changed.txt" })))
      .rejects.toMatchObject({ code: "claude_tool_id_conflict" });
    await first.worker.blocks(calls);
    await first.done;
    expect(first.events.filter(event => event.type === "tool_call_start")).toEqual(calls.map(call => ({
      type: "tool_call_start", id: call.id, name: "read_file",
    })));
    expect(first.events.filter(event => event.type === "tool_call_delta").map(event => event.arguments).join(""))
      .toBe(calls.map(call => JSON.stringify(call.arguments)).join(""));
    expect(first.events.at(-1)).toMatchObject({ type: "done", stopReason: "tool_use", endTurn: false });
    const history = [...input.parsed.context.messages, toolAssistant(calls), result(calls[1]!.id, "right"), result(calls[0]!.id, "left")];
    const continued = h.run(withMessages(input, history));
    expect(await left).toEqual({ content: [{ type: "text", text: "left" }], isError: false });
    expect(await right).toEqual({ content: [{ type: "text", text: "right" }], isError: false });
    await first.worker.finish("both read");
    await continued.done;
    expect(continued.events.filter(event => event.type === "text_delta")).toEqual([{ type: "text_delta", text: "both read" }]);
    expect(continued.events.filter(event => event.type.startsWith("tool_call_"))).toEqual([]);
    const next = h.run(withMessages(input, [...history, assistant("both read"), user("another question")]));
    await until(() => first.worker.sends.length === 2);
    await first.worker.finish("next answer", false);
    await next.done;
    expect(h.workers).toHaveLength(1);
    expect(first.worker.sends[1]).toBe("another question");
    expect(first.worker.stops).toBe(0);
  });

  test.each(["stream-first", "mcp-first"])("stages the whole batch until message_stop and exact MCP IDs match (%s)", async order => {
    const h = harness();
    const input = turn();
    const response = await h.start(input);
    const calls = [call(input, "toolu_a"), call(input, "toolu_b")];
    if (order === "mcp-first") calls.forEach(call => response.worker.mcp(call));
    await response.worker.blocks(calls, false);
    expect(response.events).toEqual([]);
    await response.worker.endBatch(calls);
    if (order === "stream-first") {
      expect(response.events).toEqual([]);
      response.worker.mcp(calls[1]!);
      expect(response.events).toEqual([]);
      response.worker.mcp(calls[0]!);
    }
    await response.done;
    expect(response.events.filter(event => event.type === "tool_call_start").map(event => event.id)).toEqual(["toolu_a", "toolu_b"]);
  });

  test("refuses an MCP identity absent from the generated batch before releasing any calls", async () => {
    const h = harness();
    const input = turn();
    const response = await h.start(input);
    response.worker.mcp(call(input, "toolu_unknown"));
    response.worker.mcp(call(input, "toolu_expected"));
    await response.worker.blocks([call(input, "toolu_expected")]);
    await response.done;
    expect(response.events.some(event => event.type.startsWith("tool_call_"))).toBe(false);
    expect(response.events.at(-1)).toMatchObject({ type: "error", code: "claude_tool_id_conflict" });
  });

  test("refuses a streamed/MCP argument mismatch without releasing tool calls", async () => {
    const h = harness();
    const input = turn();
    const response = await h.start(input);
    response.worker.mcp(call(input, "toolu_a", { path: "different.txt" }));
    await response.worker.blocks([call(input, "toolu_a")]);
    await response.done;
    expect(response.events.filter(event => event.type.startsWith("tool_call_"))).toEqual([]);
    expect(response.events.at(-1)).toMatchObject({ type: "error", code: "claude_tool_id_conflict" });
  });

  test("a zero-argument streamed tool produces a valid empty JSON object", async () => {
    const h = harness();
    const input = turn();
    input.parsed.context.tools![0]!.parameters = { type: "object", properties: {}, additionalProperties: false };
    input.bridge = stableClaudeToolBridge(buildCodeBuddyToolBridge(input.parsed));
    const response = await h.start(input);
    const action = { id: "toolu_empty", name: input.bridge.tools[0]!.name, arguments: {} };
    response.worker.mcp(action);
    await response.worker.begin();
    await response.worker.raw({ type: "content_block_start", index: 0, content_block: {
      type: "tool_use", id: action.id, name: `mcp__opencodex__${action.name}`, input: {},
    } });
    await response.worker.raw({ type: "content_block_stop", index: 0 });
    await response.worker.raw({ type: "message_stop" });
    await response.done;
    expect(response.events).toContainEqual({ type: "tool_call_delta", arguments: "{}" });
    expect(response.events.at(-1)).toMatchObject({ type: "done", stopReason: "tool_use" });
  });

  test("an undelivered batch can reconnect after a temporary delivery-marker write failure", async () => {
    const h = harness();
    const input = turn();
    const abort = new AbortController();
    const first = await h.start({ ...input, incoming: { ...input.incoming, abortSignal: abort.signal } });
    abort.abort();
    await first.done;
    const action = call(input, "toolu_retry_marker");
    first.worker.mcp(action);
    await first.worker.blocks([action]);
    h.journal.rejectSave = snapshot => snapshot.turns.some(turn => turn.deliveryStarted === true);
    await expect(h.run(input).done).rejects.toThrow("journal failure");
    h.journal.rejectSave = undefined;
    const reconnected = h.run(input);
    await reconnected.done;
    expect(reconnected.events.filter(event => event.type === "tool_call_start")).toHaveLength(1);
    expect(reconnected.events.at(-1)).toMatchObject({ type: "done", stopReason: "tool_use" });
    expect(first.worker.sends).toHaveLength(1);
  });

  test("a durable caller identity survives changes in process-local replay identities", () => {
    const input = turn();
    const incoming = { ...input.incoming, durableClientPrincipalId: claudeDigest("stable admitted caller") };
    const parsed = { ...input.parsed, _reasoningReplayScope: { clientThreadId: "thread", clientPrincipalId: "before-restart" } };
    const before = claudeSessionScope(parsed, incoming, input.provider);
    const after = claudeSessionScope({ ...parsed, _reasoningReplayScope: { ...parsed._reasoningReplayScope, clientPrincipalId: "after-restart" } }, incoming, input.provider);
    expect(before).toBe(after);
    expect(claudeSessionScope(parsed, { ...incoming, durableClientPrincipalId: undefined }, input.provider)).toBeUndefined();
  });

  test("keeps the per-call argument bound when the worker outlives the HTTP request", async () => {
    const h = harness();
    const input = turn();
    const response = await h.start(input);
    await response.worker.blocks([call(input, "toolu_large", { path: "x".repeat(2 * 1024 * 1024 + 1) })]);
    await response.done;
    expect(response.events.some(event => event.type.startsWith("tool_call_"))).toBe(false);
    expect(response.events.at(-1)).toMatchObject({ type: "error", code: "translation_buffer_limit" });
  });

  test("a model change cannot discard an unacknowledged external tool", async () => {
    const h = harness();
    const input = turn();
    const first = await h.start(input);
    const action = call(input, "toolu_model_change");
    first.worker.mcp(action);
    await first.worker.blocks([action]);
    await first.done;
    const switched = { ...input, parsed: { ...input.parsed, modelId: "another-model" } };
    expect(claudeSessionScope(switched.parsed, switched.incoming, switched.provider)).toBe(input.scope);
    await expect(h.run(switched).done).rejects.toMatchObject({ code: "claude_tool_delivery_uncertain" });
    expect(h.workers).toHaveLength(1);
    expect(first.worker.sends).toHaveLength(1);
  });

  test("an evicted idle conversation replays its receipt without another inference", async () => {
    const h = harness();
    const firstInput = turn([user("conversation zero")]);
    for (let index = 0; index < 17; index++) {
      const input = index === 0 ? firstInput : turn([user(`conversation ${index}`)]);
      if (index > 0) {
        input.incoming.headers.set("thread-id", `thread-${index}`);
        input.scope = claudeSessionScope(input.parsed, input.incoming, input.provider)!;
      }
      const response = await h.start(input);
      await response.worker.finish(`answer ${index}`);
      await response.done;
    }
    expect(h.workers[0]!.stops).toBe(1);
    const replay = h.run(firstInput);
    await replay.done;
    expect(replay.events).toContainEqual({ type: "text_delta", text: "answer 0" });
    expect(h.workers).toHaveLength(17);
  });

  test("large completed payloads expire before the cache blocks a new response", async () => {
    const h = harness();
    const firstInput = turn([user("large response 0")]);
    const text = "x".repeat(2 * 1024 * 1024);
    for (let index = 0; index < 9; index++) {
      const input = index === 0 ? firstInput : turn([user(`large response ${index}`)]);
      const response = await h.start(input);
      await response.worker.finish(text);
      await response.done;
      expect(response.events.at(-1)).toMatchObject({ type: "done" });
    }
    const snapshot = h.journal.load(firstInput.scope)!;
    expect(Buffer.byteLength(JSON.stringify(snapshot))).toBeLessThanOrEqual(16 * 1024 * 1024);
    await expect(h.run(firstInput).done).rejects.toMatchObject({ code: "claude_replay_expired" });
  });

  test.each(["before-output", "after-output"])("abort/reconnect preserves the inference without a second send (%s)", async timing => {
    const h = harness();
    const input = turn();
    const abort = new AbortController();
    const response = await h.start({ ...input, incoming: { ...input.incoming, abortSignal: abort.signal } });
    await response.worker.begin();
    await response.worker.raw({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
    const firstDelta = { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "first " } };
    if (timing === "after-output") await response.worker.raw(firstDelta);
    abort.abort();
    await response.done;
    if (timing === "before-output") await response.worker.raw(firstDelta);
    await response.worker.raw({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "second" } });
    const reconnected = h.run(input);
    await response.worker.raw({ type: "content_block_stop", index: 0 });
    await response.worker.raw({ type: "message_stop" });
    await response.worker.emit({ type: "assistant", message: { role: "assistant", content: [{ type: "text", text: "first second" }] } });
    await response.worker.emit({ type: "result", subtype: "success", is_error: false });
    await reconnected.done;
    expect(response.events).toEqual(timing === "after-output" ? [{ type: "text_delta", text: "first " }] : []);
    expect(reconnected.events.filter(event => event.type === "text_delta")).toEqual([
      { type: "text_delta", text: "first " }, { type: "text_delta", text: "second" },
    ]);
    expect(reconnected.events.at(-1)).toMatchObject({ type: "done", stopReason: "stop" });
    expect(h.workers).toHaveLength(1);
    expect(response.worker.sends).toHaveLength(1);
    expect(response.worker.stops).toBe(0);
  });

  test("replays a completed text receipt after restart without creating a worker", async () => {
    const journal = new MemoryJournal();
    const h = harness(journal);
    const input = turn();
    const response = await h.start(input);
    await response.worker.finish("cached answer");
    await response.done;
    await h.manager.close();
    const reopened = harness(journal);
    const replay = reopened.run(input);
    await replay.done;
    expect(replay.events.filter(event => event.type !== "done")).toEqual(response.events.filter(event => event.type !== "done"));
    expect(replay.events.at(-1)).toEqual({ type: "done", stopReason: "stop", usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 } });
    expect(journal.load(input.scope)!.turns[0]!.events.at(-1)).toMatchObject({ usage: { inputTokens: 3, outputTokens: 4 } });
    expect(reopened.workers).toEqual([]);
    expect(journal.closes).toBe(1);
  });

  test("ambiguous partial delivery refuses to reemit an external action", async () => {
    const h = harness();
    const input = turn();
    const first = h.run(input, event => { if (event.type === "tool_call_start") throw new Error("connection lost"); });
    await until(() => !!h.workers[0]?.sends.length);
    const worker = h.workers[0]!;
    const c = call(input, "toolu_once");
    worker.mcp(c);
    await worker.blocks([c]);
    await first.done;
    expect(h.journal.load(input.scope)!.turns[0]!.deliveryStarted).toBe(true);
    const replay = h.run(input);
    await expect(replay.done).rejects.toMatchObject({ code: "claude_tool_delivery_uncertain" });
    expect(replay.events).toEqual([]);
    expect(worker.sends).toHaveLength(1);
    await h.manager.close();
    const reopened = harness(h.journal);
    const durableRefusal = reopened.run(input);
    await expect(durableRefusal.done).rejects.toMatchObject({ code: "claude_tool_delivery_uncertain" });
    expect(durableRefusal.events).toEqual([]);
    expect(reopened.workers).toEqual([]);
  });

  test("a batch completed while disconnected is delivered once on reconnect", async () => {
    const h = harness();
    const input = turn();
    const abort = new AbortController();
    const first = await h.start({ ...input, incoming: { ...input.incoming, abortSignal: abort.signal } });
    abort.abort();
    await first.done;
    const c = call(input, "toolu_offline");
    first.worker.mcp(c);
    await first.worker.blocks([c]);
    expect(first.events).toEqual([]);
    expect(h.journal.load(input.scope)!.turns[0]!.deliveryStarted).not.toBe(true);
    const reconnect = h.run(input);
    await reconnect.done;
    expect(reconnect.events.filter(event => event.type === "tool_call_start")).toHaveLength(1);
    const duplicate = h.run(input);
    await expect(duplicate.done).rejects.toMatchObject({ code: "claude_tool_delivery_uncertain" });
    expect(duplicate.events).toEqual([]);
    expect(first.worker.sends).toHaveLength(1);
  });

  test("a caught reconnect delivery exception preserves the durable refusal marker", async () => {
    const h = harness();
    const input = turn();
    const abort = new AbortController();
    const first = await h.start({ ...input, incoming: { ...input.incoming, abortSignal: abort.signal } });
    abort.abort();
    await first.done;
    const c = call(input, "toolu_delivery_exception");
    first.worker.mcp(c);
    await first.worker.blocks([c]);
    const delivery = h.run(input, event => { if (event.type === "tool_call_start") throw new Error("consumer disconnected"); });
    await expect(delivery.done).rejects.toThrow("consumer disconnected");
    expect(h.journal.load(input.scope)!.turns[0]!.deliveryStarted).toBe(true);
    await h.manager.close();
    const reopened = harness(h.journal);
    const retry = reopened.run(input);
    await expect(retry.done).rejects.toMatchObject({ code: "claude_tool_delivery_uncertain" });
    expect(retry.events).toEqual([]);
    expect(reopened.workers).toEqual([]);
  });

  test("results are durable, immutable and rejected when their content or error bit changes", async () => {
    const h = harness();
    const input = turn();
    const response = await h.start(input);
    const calls = [call(input, "toolu_a"), call(input, "toolu_b")];
    calls.forEach(call => response.worker.mcp(call));
    await response.worker.blocks(calls);
    await response.done;
    const base = [...input.parsed.context.messages, toolAssistant(calls)];
    const partial = h.run(withMessages(input, [...base, result("toolu_a", "accepted")]));
    await expect(partial.done).rejects.toMatchObject({ code: "claude_tool_results_pending" });
    const saved = h.journal.load(input.scope)!;
    expect(saved.turns[0]!.calls[0]!.result?.content).toBe("accepted");
    saved.turns[0]!.calls[0]!.result!.content = "mutated copy";
    for (const changed of [result("toolu_a", "changed"), result("toolu_a", "accepted", true)]) {
      const conflict = h.run(withMessages(input, [...base, changed, result("toolu_b", "b")]));
      await expect(conflict.done).rejects.toMatchObject({ code: "claude_tool_result_conflict" });
      expect(conflict.events).toEqual([]);
    }
    expect(h.journal.load(input.scope)!.turns[0]!.calls[0]!.result?.content).toBe("accepted");
    expect(response.worker.sends).toHaveLength(1);
  });

  test("reopens the journal with every supplied external result and resumes without duplicate calls", async () => {
    const journal = new MemoryJournal();
    const h = harness(journal);
    const input = turn();
    const response = await h.start(input);
    const calls = [call(input, "toolu_a"), call(input, "toolu_b")];
    calls.forEach(call => response.worker.mcp(call));
    await response.worker.blocks(calls);
    await response.done;
    await h.manager.close();
    const reopened = harness(journal);
    const history = [...input.parsed.context.messages, toolAssistant(calls), result("toolu_a", "ALPHA"), result("toolu_b", "BETA", true)];
    const recovered = await reopened.start(withMessages(input, history));
    expect(recovered.worker.sends[0]).toContain("ALPHA");
    expect(recovered.worker.sends[0]).toContain("BETA");
    expect(recovered.worker.sends[0]).toContain("toolu_a");
    expect(recovered.worker.sends[0]).toContain("toolu_b");
    await recovered.worker.finish("recovered");
    await recovered.done;
    expect(recovered.events.filter(event => event.type === "tool_call_start")).toEqual([]);
    expect(journal.load(input.scope)!.turns[0]!.calls.map(call => call.result?.content)).toEqual(["ALPHA", "BETA"]);
    expect(reopened.workers).toHaveLength(1);
  });

  test("retrying a consumed tool response never reemits the already executed action", async () => {
    const h = harness();
    const input = turn();
    const first = await h.start(input);
    const c = call(input, "toolu_executed");
    const pending = first.worker.mcp(c);
    await first.worker.blocks([c]);
    await first.done;
    const continued = h.run(withMessages(input, [...input.parsed.context.messages, toolAssistant([c]), result(c.id, "executed")]));
    await pending;
    await first.worker.finish("finished");
    await continued.done;
    const replay = h.run(input);
    const error = await replay.done.then(() => undefined, error => error);
    expect(replay.events.filter(event => event.type.startsWith("tool_call_"))).toEqual([]);
    expect(error).toMatchObject({ code: "claude_tool_delivery_uncertain" });
    expect(first.worker.sends).toHaveLength(1);
    await h.manager.close();
    const reopened = harness(h.journal);
    const changedCatalog = { ...input, bridge: { ...input.bridge, tools: input.bridge.tools.map(tool => ({ ...tool, description: "changed" })) } };
    for (const changed of [
      { ...input, parsed: { ...input.parsed, modelId: "another-model" } },
      { ...input, parsed: { ...input.parsed, options: { ...input.parsed.options, temperature: 0.5 } } },
      changedCatalog,
    ]) {
      const retry = reopened.run(changed);
      await expect(retry.done).rejects.toMatchObject({ code: "claude_tool_delivery_uncertain" });
      expect(retry.events).toEqual([]);
    }
    expect(reopened.workers).toHaveLength(0);
  });

  test("a cold recovery refuses reuse of an already recorded tool call ID", async () => {
    const journal = new MemoryJournal();
    const h = harness(journal);
    const input = turn();
    const first = await h.start(input);
    const oldCall = call(input, "toolu_original");
    first.worker.mcp(oldCall);
    await first.worker.blocks([oldCall]);
    await first.done;
    await h.manager.close();
    const reopened = harness(journal);
    const recovered = await reopened.start(withMessages(input, [...input.parsed.context.messages,
      toolAssistant([oldCall]), result(oldCall.id, "already executed")]));
    const repeated = call(input, oldCall.id);
    recovered.worker.mcp(repeated);
    await recovered.worker.blocks([repeated]);
    await recovered.done;
    expect(recovered.events.some(event => event.type.startsWith("tool_call_"))).toBe(false);
    expect(recovered.events.at(-1)).toMatchObject({ type: "error", code: "claude_tool_id_conflict" });
  });

  test("a worker crash before tool output is safely retried with a fresh worker", async () => {
    const h = harness();
    const input = turn();
    const response = await h.start(input);
    response.worker.options.onError(new Error("offline worker crash"));
    await response.done;
    expect(response.events).toEqual([expect.objectContaining({ type: "error", retryable: true })]);
    const retry = await h.start(input);
    await retry.worker.finish("retry answer");
    await retry.done;
    expect(h.workers).toHaveLength(2);
    expect(response.worker.stops).toBe(1);
    expect(retry.events.some(event => event.type === "error")).toBe(false);
  });

  test("abort during worker creation prevents dispatch and a later retry sends only once", async () => {
    const h = harness();
    const input = turn();
    const abort = new AbortController();
    const cancelled = h.run({ ...input, incoming: { ...input.incoming, abortSignal: abort.signal } });
    abort.abort();
    await cancelled.done;
    expect(h.workers.flatMap(worker => worker.sends)).toEqual([]);
    expect(h.workers.every(worker => worker.stops === 1)).toBe(true);
    expect(cancelled.events).toEqual([]);
    const retry = await h.start(input);
    await retry.worker.finish("only inference");
    await retry.done;
    expect(h.workers.flatMap(worker => worker.sends)).toHaveLength(1);
  });

  test("serializes concurrent setup and gives the newest input ownership", async () => {
    const h = harness();
    const input = turn();
    const first = h.run(input);
    const conflict = h.run(withMessages(input, [user("a conflicting instruction")]));
    await until(() => !!h.workers[1]?.sends.length);
    await first.done;
    expect(first.events.at(-1)).toMatchObject({ type: "error", code: "claude_turn_superseded" });
    expect(h.workers[0]!.stops).toBe(1);
    const another = h.run(withMessages(input, [user("a conflicting instruction")]));
    await conflict.done;
    await h.workers[1]!.finish("owned response");
    await another.done;
    expect(h.workers).toHaveLength(2);
    expect(another.events.at(-1)).toMatchObject({ type: "done" });
  });

  test.each(["receipt", "delivery"])("journal %s failure releases no external action", async stage => {
    const h = harness();
    const input = turn();
    const response = await h.start(input);
    h.journal.rejectSave = snapshot => snapshot.turns.at(-1)?.phase === "tools"
      && (stage === "receipt" || snapshot.turns.at(-1)?.deliveryStarted === true);
    const c = call(input, "toolu_not_committed");
    response.worker.mcp(c);
    await response.worker.blocks([c]);
    await response.done;
    expect(response.events.filter(event => event.type.startsWith("tool_call_"))).toEqual([]);
    expect(response.events.at(-1)).toMatchObject({ type: "error", code: "claude_journal_write_failed", status: 503 });
    expect(response.worker.stops).toBe(1);
  });

  test("an initial journal failure prevents worker creation and inference", async () => {
    const h = harness();
    h.journal.rejectSave = () => true;
    const response = h.run(turn());
    await expect(response.done).rejects.toThrow("injected journal failure");
    expect(h.workers).toEqual([]);
    expect(response.events).toEqual([]);
  });

  test("an aborted inference expires after bounded grace and stops its worker", async () => {
    const h = harness(new MemoryJournal(), { reconnectGraceMs: 20, idleMs: 10_000, timeoutMs: 10_000 });
    const input = turn();
    const abort = new AbortController();
    const response = await h.start({ ...input, incoming: { ...input.incoming, abortSignal: abort.signal } });
    abort.abort();
    await response.done;
    await until(() => response.worker.stops === 1);
    const receipt = h.journal.load(input.scope)!.turns[0]!;
    expect(receipt.phase).toBe("interrupted");
    expect(receipt.events.at(-1)).toMatchObject({ type: "error", code: "claude_reconnect_expired", status: 499 });
    expect(response.events).toEqual([]);
  });

  test("updates a dynamic tool catalog on the same worker and maps its new wire name", async () => {
    const h = harness();
    const input = turn();
    const first = await h.start(input);
    await first.worker.finish("ready");
    await first.done;
    const changed = withMessages(input, [...input.parsed.context.messages, assistant("ready"), user("use the new tool")]);
    changed.parsed.context.tools = [{ name: "inspect_file", description: "New catalog", parameters: { type: "object", properties: {} } }];
    changed.bridge = stableClaudeToolBridge(buildCodeBuddyToolBridge(changed.parsed));
    const next = h.run(changed);
    await until(() => first.worker.sends.length === 2);
    expect(first.worker.updates).toEqual([changed.bridge.tools]);
    await expect(first.worker.mcp(call(input, "toolu_removed"))).rejects.toMatchObject({ code: "undeclared_tool_call" });
    const c = call(changed, "toolu_new");
    first.worker.mcp(c);
    await first.worker.blocks([c]);
    await next.done;
    expect(next.events).toContainEqual({ type: "tool_call_start", id: c.id, name: "inspect_file" });
    expect(h.workers).toHaveLength(1);
  });

  test("session-context hashes isolate provider, principal and child thread in the manager", async () => {
    const base = turn();
    const principal = turn();
    principal.parsed._cursorIdentityScope = "another-principal";
    const provider = turn();
    provider.incoming.providerName = "another-claude-provider";
    const thread = turn();
    thread.parsed._codexOwnThreadId = "child-thread";
    const destination = turn();
    destination.provider.baseUrl = "claude-cli://another-profile";
    const requests = [base, principal, provider, thread, destination];
    const h = harness();
    for (const request of requests) request.scope = claudeSessionScope(request.parsed, request.incoming, request.provider)!;
    expect(new Set(requests.map(request => request.scope)).size).toBe(requests.length);
    for (const request of requests) {
      const response = await h.start(request);
      await response.worker.finish(request.scope);
      await response.done;
      expect(request.scope).toMatch(/^[a-f0-9]{64}$/);
    }
    expect(h.workers).toHaveLength(requests.length);
    expect(h.journal.records.size).toBe(requests.length);
    expect(claudeDigest({ b: 2, a: 1 })).toBe(claudeDigest({ a: 1, b: 2 }));
    expect(claudeHistory([assistant("ab")])).toEqual(claudeHistory([assistant("a"), assistant("b")]));
  });

  test("missing or invalid thread identity fails closed instead of using a shared session", () => {
    const input = turn();
    input.parsed._clientThreadId = "cache-cohort";
    input.parsed._reasoningReplayScope = { clientThreadId: "parent-thread", clientPrincipalId: "caller" };
    for (const id of [null, " ", "x".repeat(4097), "thread\u0001"]) {
      const headers = new Headers();
      if (id !== null) headers.set("thread-id", id);
      expect(claudeSessionScope(input.parsed, { ...input.incoming, headers }, input.provider)).toBeUndefined();
    }
    for (const name of ["thread-id", "session_id", "session-id"]) {
      expect(claudeSessionScope(input.parsed, { ...input.incoming, headers: new Headers({ [name]: "own-thread" }) }, input.provider)).toMatch(/^[a-f0-9]{64}$/);
    }
  });
});
