import { describe, expect, test } from "bun:test";
import { responsesUsage } from "../../src/bridge/internal";
import { claudeSessionScope } from "../../src/adapters/claude-cli/session-context";
import { claudeSdkResultUsage, createClaudeSdkStreamState, mapClaudeSdkMessage, releaseClaudeSdkBlocks } from "../../src/adapters/claude-cli/session-events";
import type { SDKMessage } from "@anthropic-ai/claude-agent-sdk";
import type { AdapterEvent, OcxMessage } from "../../src/types";
import { assistant, call, harness, result, toolAssistant, turn, until, user, withMessages } from "../fixtures/claude-session-worker";
import { nativeBinary, nativeSession, type NativeBlock } from "../fixtures/claude-session-native-api";

const text = (text: string): NativeBlock => ({ type: "text", text });
function terminal(events: AdapterEvent[]) {
  expect(events.filter(event => event.type === "error")).toEqual([]);
  const event = events.at(-1);
  if (event?.type !== "done") throw new Error("Expected completed native response");
  return event;
}

describe("Claude per-leg result accounting", () => {
  test.each(["success", "error_during_execution"] as const)("%s result subtracts earlier handoffs without message_stop", subtype => {
    const state = createClaudeSdkStreamState();
    state.sdkReportedUsage = { inputTokens: 45, outputTokens: 8, totalTokens: 53,
      cachedInputTokens: 20, cacheReadInputTokens: 20, cacheCreationInputTokens: 5, cacheCreation1hInputTokens: 3 };
    try {
      for (const message of [
        { type: "stream_event", event: { type: "message_start", message: { id: "current", usage: { input_tokens: 4, output_tokens: 0 } } } },
        { type: "stream_event", event: { type: "message_delta", usage: { output_tokens: 1 } } },
      ]) mapClaudeSdkMessage(message as SDKMessage, state);
      const [event] = mapClaudeSdkMessage({ type: "result", subtype, is_error: subtype !== "success", errors: ["interrupted"],
        usage: { input_tokens: 24, output_tokens: 11, cache_read_input_tokens: 26, cache_creation_input_tokens: 7,
          cache_creation: { ephemeral_1h_input_tokens: 4, ephemeral_5m_input_tokens: 3 } } } as SDKMessage, state);
      expect(event?.type).toBe(subtype === "success" ? "done" : "error");
      expect(state.partialUsage).toEqual({ inputTokens: 12, outputTokens: 3, totalTokens: 15,
        cachedInputTokens: 6, cacheReadInputTokens: 6, cacheCreationInputTokens: 2, cacheCreation1hInputTokens: 1 });
    } finally { releaseClaudeSdkBlocks(state); state.translatorBudget?.dispose(); }
  });

  test("completed frames remain the fallback for zero/missing aggregates; partial zero results stay authoritative", () => {
    const state = createClaudeSdkStreamState();
    try {
      mapClaudeSdkMessage({ type: "stream_event", event: { type: "message_start", message: {
        id: "current", usage: { input_tokens: 12, output_tokens: 4 },
      } } } as SDKMessage, state);
      const zero = { usage: { input_tokens: 0, output_tokens: 0 } } as Parameters<typeof claudeSdkResultUsage>[0];
      state.sawMessageStop = true;
      expect(claudeSdkResultUsage(zero, state)).toEqual({ inputTokens: 12, outputTokens: 4, totalTokens: 16 });
      expect(claudeSdkResultUsage({} as typeof zero, state)).toEqual({ inputTokens: 12, outputTokens: 4, totalTokens: 16 });
      state.sawMessageStop = false;
      expect(claudeSdkResultUsage(zero, state)).toEqual({ inputTokens: 0, outputTokens: 0, totalTokens: 0 });
    } finally { releaseClaudeSdkBlocks(state); state.translatorBudget?.dispose(); }
  });

  test("steering after a charged handoff reports only the interrupted remainder and resets the next native turn", async () => {
    const h = harness();
    const input = turn();
    const first = await h.start(input);
    const action = call(input, "toolu_steer_usage");
    const pending = first.worker.mcp(action);
    await first.worker.blocks([action]); await first.done;
    expect(terminal(first.events).usage).toEqual({ inputTokens: 3, outputTokens: 4, totalTokens: 7 });
    const history = [...input.parsed.context.messages, toolAssistant([action]), result(action.id, "ACTUAL_RESULT")];
    const current = h.run(withMessages(input, history));
    await pending; await Bun.sleep(0);
    await first.worker.begin();
    await first.worker.raw({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "partial" } });
    await first.worker.raw({ type: "message_delta", delta: { stop_reason: null }, usage: { output_tokens: 1 } });
    first.worker.interrupt = async () => {
      first.worker.interrupts++;
      // Usage can advance while output is being suppressed during the SDK interrupt/drain.
      await first.worker.raw({ type: "message_delta", delta: { stop_reason: null }, usage: { output_tokens: 5 } });
      await first.worker.emit({ type: "result", subtype: "error_during_execution", is_error: true,
        errors: ["interrupted"], usage: { input_tokens: 6, output_tokens: 9 } });
    };
    const steered = h.run(withMessages(input, [...history, assistant("partial"), user("Change direction now")]));
    await until(() => first.worker.sends.length === 2); await current.done;
    expect(current.events.at(-1)).toMatchObject({ type: "incomplete", usage: { inputTokens: 3, outputTokens: 5, totalTokens: 8 } });
    await first.worker.finish("steered answer", false); await steered.done;
    expect(terminal(steered.events).usage).toEqual({ inputTokens: 3, outputTokens: 4, totalTokens: 7 });
    expect(h.workers).toHaveLength(1);
  });
});

describe.skipIf(!nativeBinary)("Claude native compaction context", () => {
  test("a native tool loop reports each HTTP leg once instead of repeating the SDK turn's aggregate usage", async () => {
    const replies = [{ content: [] as NativeBlock[] }, { content: [] as NativeBlock[] },
      { content: [text("LOOP_COMPLETE")] }, { content: [text("NEXT_COMPLETE")] }];
    const h = nativeSession(replies, { nativePersistence: true });
    try {
      const actions = [call(h.input, "toolu_context_A", { path: "A.txt" }), call(h.input, "toolu_context_B", { path: "B.txt" })];
      for (const [index, action] of actions.entries()) replies[index]!.content = [{ type: "tool_use", id: action.id,
        name: `mcp__opencodex__${action.name}`, input: action.arguments }];
      const history: OcxMessage[] = [...h.input.parsed.context.messages];
      const usages = [];
      let response = h.run(h.input);
      for (const action of actions) {
        await response.done;
        expect(response.events.filter(event => event.type === "tool_call_start")).toEqual([
          { type: "tool_call_start", id: action.id, name: "read_file" },
        ]);
        usages.push(terminal(response.events).usage!);
        history.push(toolAssistant([action]), result(action.id, `ACTUAL_${action.id}`));
        response = h.run(withMessages(h.input, [...history]));
      }
      await response.done;
      const usage = terminal(response.events).usage;
      // The real CLI result totals 36/30, including the already reported tool handoffs.
      expect(usage).toEqual({ inputTokens: 12, outputTokens: 10, totalTokens: 22 });
      usages.push(usage!);
      expect(usages.reduce((sum, value) => sum + value.inputTokens, 0)).toBe(36);
      expect(usages.reduce((sum, value) => sum + value.outputTokens, 0)).toBe(30);
      expect(responsesUsage(usage).total_tokens).toBe(22);
      expect(h.requested).toEqual(actions.map(action => action.id));
      expect(h.resolved).toEqual(h.requested);
      history.push(assistant("LOOP_COMPLETE"), user("NEXT_INPUT"));
      const next = h.run(withMessages(h.input, history));
      await next.done;
      expect(terminal(next.events).usage).toEqual({ inputTokens: 12, outputTokens: 10, totalTokens: 22 });
      expect(h.children).toHaveLength(1);
      expect(h.requests).toHaveLength(4);
      expect(h.unexpected).toEqual([]);
    } finally { await h.close(); }
  }, 60_000);

  test("a compacted client history replaces native context and the next turn retains only that replacement", async () => {
    const h = nativeSession(["PRE_COMPACT_ANSWER", "SUMMARY_ANSWER", "POST_COMPACT_ANSWER", "FOLLOWUP_ANSWER"]
      .map(answer => ({ content: [text(answer)] })), { nativePersistence: true });
    try {
      const original = withMessages(h.input, [user("PRE_COMPACT_PRIVATE_HISTORY")]);
      const first = h.run(original);
      await first.done;
      terminal(first.events);
      const oldId = h.journal.load(h.input.scope)!.native!.sessionId;
      const compact = withMessages(h.input, [...original.parsed.context.messages, assistant("PRE_COMPACT_ANSWER"), user("SUMMARIZE_NOW")]);
      compact.parsed._compactionRequest = true;
      compact.scope = claudeSessionScope(compact.parsed, compact.incoming, compact.provider)!;
      expect(compact.scope).not.toBe(original.scope);
      const summary = h.run(compact);
      await summary.done;
      terminal(summary.events);
      const replacement = [user("CHECKPOINT_SUMMARY"), user("POST_COMPACT_INPUT")];
      const after = h.run(withMessages(h.input, replacement));
      await after.done;
      terminal(after.events);
      expect(h.journal.load(h.input.scope)!.native!.sessionId).not.toBe(oldId);
      replacement.push(assistant("POST_COMPACT_ANSWER"), user("FOLLOWUP_INPUT"));
      const followup = h.run(withMessages(h.input, replacement));
      await followup.done;
      terminal(followup.events);
      for (const request of h.requests.slice(2)) {
        const body = JSON.stringify(request.messages);
        expect(body).not.toContain("PRE_COMPACT_PRIVATE_HISTORY");
        expect(body).not.toContain("SUMMARIZE_NOW");
        expect(body.split("CHECKPOINT_SUMMARY")).toHaveLength(2);
      }
      expect(h.unexpected).toEqual([]);
    } finally { await h.close(); }
  }, 60_000);
});
