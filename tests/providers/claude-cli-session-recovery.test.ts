import { expect, test } from "bun:test";
import { claudeDigest, claudeResult } from "../../src/adapters/claude-cli/session-context";
import { ClaudeSessionError } from "../../src/adapters/claude-cli/session-errors";
import type { ClaudeSessionSnapshot, ClaudeToolReceipt } from "../../src/adapters/claude-cli/session-journal";
import { prepareClaudeToolResults } from "../../src/adapters/claude-cli/session-recovery";
import { parseRequest } from "../../src/responses/parser";
import type { OcxMessage, OcxToolCall } from "../../src/types";

const user = (content: string): OcxMessage => ({ role: "user", content, timestamp: 0 });
const result = (id: string, content: string, isError = false): OcxMessage => ({
  role: "toolResult", toolCallId: id, toolName: "read_file", content, isError, timestamp: 2,
});
function receipt(id: string, content?: string, isError = false): ClaudeToolReceipt {
  return {
    id, name: "read_file", arguments: JSON.stringify({ path: `${id}.txt` }),
    ...(content === undefined ? {} : { result: claudeResult(result(id, content, isError)) }),
  };
}
function snapshot(calls = [receipt("A"), receipt("B")]): ClaudeSessionSnapshot {
  return {
    version: 1, history: [claudeDigest("original input")],
    turns: [{ inputHash: claudeDigest("turn"), operationId: claudeDigest("operation"), phase: "tools", events: [], calls, updatedAt: 123 }],
  };
}
function assistant(calls: ClaudeToolReceipt[]): OcxMessage {
  return { role: "assistant", timestamp: 1, content: calls.map(call => ({
    type: "toolCall", id: call.id, name: call.name, arguments: JSON.parse(call.arguments),
  })) };
}
function history(saved: ClaudeSessionSnapshot): OcxMessage[] {
  return [user("read files"), assistant(saved.turns[0]!.calls)];
}
function expectCode(action: () => unknown, code: string): void {
  let caught: unknown;
  try { action(); } catch (error) { caught = error; }
  expect(caught).toBeInstanceOf(ClaudeSessionError);
  expect(caught).toMatchObject({ code, status: 409, retryable: false });
}

test("stages a partial batch on detached state so the manager can commit before its missing-result guard", () => {
  const saved = snapshot();
  const messages = [...history(saved), result("A", "saved A")];
  const before = structuredClone({ saved, messages });
  const prepared = prepareClaudeToolResults(saved, messages);
  expect({ saved, messages }).toEqual(before);
  expect(prepared.changed).toBe(true);
  expect(prepared.pendingTurns).toEqual(prepared.snapshot.turns);
  expect(prepared.pendingTurns[0]).toBe(prepared.snapshot.turns[0]);
  expect(prepared.pendingTurns[0]!.phase).toBe("tools");
  expect(prepared.pendingTurns[0]!.deliveryStarted).toBe(true);
  expect(prepared.pendingTurns[0]!.calls[0]!.result).toEqual(claudeResult(result("A", "saved A")));
  expect(prepared.pendingTurns.some(turn => turn.calls.some(call => !call.result))).toBe(true);
  expect(prepared.recoveredCalls).toEqual(new Set(["A"]));
  prepared.snapshot.history.push("detached");
  prepared.pendingTurns[0]!.events.push({ type: "text_delta", text: "detached" });
  prepared.messages[0]!.timestamp++;
  expect({ saved, messages }).toEqual(before);
});

test("a discarded stage after a failed journal save does not poison a corrected retry", () => {
  const saved = snapshot([receipt("A")]);
  const base = history(saved);
  prepareClaudeToolResults(saved, [...base, result("A", "not committed")]);
  const corrected = prepareClaudeToolResults(saved, [...base, result("A", "corrected")]);
  expect(saved.turns[0]!.calls[0]!.result).toBeUndefined();
  expect(corrected.pendingTurns[0]!.calls[0]!.result?.content).toBe("corrected");
});

test.each([result("B", "wrong B"), result("B", "saved B", true)])("B conflict leaves no staged A or delivery mutation in the original (%j)", badB => {
  const saved = snapshot([receipt("A"), receipt("B", "saved B")]);
  const messages = [...history(saved), result("A", "rejected A"), badB];
  const before = structuredClone({ saved, messages });
  expectCode(() => prepareClaudeToolResults(saved, messages), "claude_tool_result_conflict");
  expect({ saved, messages }).toEqual(before);
  const corrected = prepareClaudeToolResults(saved, [...history(saved), result("A", "valid A"), result("B", "saved B")]);
  expect(corrected.pendingTurns[0]!.calls[0]!.result?.content).toBe("valid A");
});

test("validates every result occurrence and allows identical text/content-part duplicates", () => {
  const saved = snapshot([receipt("A")]);
  const text = result("A", "same");
  const parts: OcxMessage = { ...text, content: [{ type: "text", text: "same" }] };
  const prepared = prepareClaudeToolResults(saved, [...history(saved), text, parts]);
  expect(prepared.pendingTurns[0]!.calls[0]!.result?.content).toBe("same");
  expectCode(() => prepareClaudeToolResults(saved, [...history(saved), text, parts, result("A", "different")]), "claude_tool_result_conflict");
  expect(saved.turns[0]!.calls[0]!.result).toBeUndefined();
});

test("unknown or similar IDs cannot fill an outstanding result slot", () => {
  const saved = snapshot([receipt("A")]);
  const prepared = prepareClaudeToolResults(saved, [...history(saved), result("prefix-A", "unrelated history")]);
  expect(prepared.changed).toBe(false);
  expect(prepared.pendingTurns[0]!.calls[0]!.result).toBeUndefined();
  expect(prepared.recoveredCalls.size).toBe(0);
});

test("restores saved A beside its original call even with B and a final user continue", () => {
  const saved = snapshot([receipt("A", "saved A", true), receipt("B")]);
  saved.turns[0]!.deliveryStarted = true;
  const messages = [...history(saved), result("B", "new B"), user("continue")];
  const prepared = prepareClaudeToolResults(saved, messages);
  expect(prepared.messages).toEqual([
    messages[0], messages[1],
    { role: "toolResult", toolCallId: "A", toolName: "read_file", content: "saved A", isError: true, timestamp: 123 },
    messages[2], messages[3],
  ]);
  expect(prepared.recoveredCalls).toEqual(new Set(["A", "B"]));
  expect(prepared.pendingTurns[0]!.calls.every(call => call.result)).toBe(true);
  expect(prepared.pendingTurns[0]!.phase).toBe("tools");
  expect(messages).toHaveLength(4);
  expect(saved.turns[0]!.calls[1]!.result).toBeUndefined();
});

test("one-call result survives a beforePhysicalSend refusal and restores into a rebased continue", () => {
  const saved = snapshot([receipt("A")]);
  const base = history(saved);
  const staged = prepareClaudeToolResults(saved, [...base, result("A", "committed before send")]);
  // Model the durable result commit followed by an admission failure, without completing the tool turn.
  const committed = structuredClone(staged.snapshot);
  const rebased = [...base, user("continue")];
  const prepared = prepareClaudeToolResults(committed, rebased);
  expect(prepared.messages.map(message => message.role)).toEqual(["user", "assistant", "toolResult", "user"]);
  expect(prepared.messages[2]).toMatchObject({ toolCallId: "A", content: "committed before send" });
  expect(prepared.recoveredCalls).toEqual(staged.recoveredCalls);
  expect(prepared.pendingTurns[0]!.phase).toBe("tools");
  expect(prepared.changed).toBe(false);
  expect(committed).toEqual(staged.snapshot);
});

test("a restored saved result stages the delivery marker without mutating the original", () => {
  const saved = snapshot([receipt("A", "saved A")]);
  const prepared = prepareClaudeToolResults(saved, [...history(saved), user("continue")]);
  expect(prepared.changed).toBe(true);
  expect(prepared.pendingTurns[0]!.deliveryStarted).toBe(true);
  expect(saved.turns[0]!.deliveryStarted).toBeUndefined();
});

test.each([false, true])("rejects a missing saved result whose original call is absent (new result=%s)", newResult => {
  const saved = snapshot([receipt("A", "saved A"), receipt("B")]);
  const messages = [user("continue"), ...(newResult ? [result("B", "new B")] : [])];
  const before = structuredClone({ saved, messages });
  expectCode(() => prepareClaudeToolResults(saved, messages), "claude_recovery_history_missing");
  expect({ saved, messages }).toEqual(before);
});

test("rejects ambiguous origins rather than choosing a replay location", () => {
  const saved = snapshot([receipt("A", "saved A")]);
  const messages = [...history(saved), assistant(saved.turns[0]!.calls), user("continue")];
  expectCode(() => prepareClaudeToolResults(saved, messages), "claude_recovery_history_missing");
});

test.each(["name", "arguments", "namespace"] as const)("validates historical call %s even when the saved result is supplied", field => {
  const saved = snapshot([receipt("A", "saved A")]);
  const origin = assistant(saved.turns[0]!.calls);
  if (origin.role !== "assistant") throw new Error("expected assistant");
  const call = origin.content[0] as OcxToolCall;
  if (field === "arguments") call.arguments = { path: "different.txt" };
  else call[field] = "changed";
  const messages = [user("read"), origin, result("A", "saved A")];
  const before = structuredClone({ saved, messages });
  expectCode(() => prepareClaudeToolResults(saved, messages), "claude_tool_id_conflict");
  expect({ saved, messages }).toEqual(before);
});

test("preserves namespace in reconstructed output and compares arguments canonically", () => {
  const call = receipt("A", "namespaced output");
  call.name = "mcp__files__read_file";
  call.arguments = '{"z":1,"nested":{"b":2,"a":1}}';
  const saved = snapshot([call]);
  const origin: OcxMessage = { role: "assistant", timestamp: 1, content: [{
    type: "toolCall", id: "A", name: "read_file", namespace: "mcp__files", arguments: { nested: { a: 1, b: 2 }, z: 1 },
  }] };
  const prepared = prepareClaudeToolResults(saved, [user("read"), origin, user("continue")]);
  expect(prepared.messages[2]).toMatchObject({ toolCallId: "A", toolName: "read_file", toolNamespace: "mcp__files", content: "namespaced output" });
  expect(prepared.recoveredCalls).toEqual(new Set(["A"]));
});

test("collects all outstanding available call IDs across turns, regardless of trailing role", () => {
  const saved = snapshot([receipt("A", "A")]);
  saved.turns[0]!.deliveryStarted = true;
  saved.turns.push({ ...structuredClone(saved.turns[0]!), inputHash: claudeDigest("second turn"), calls: [receipt("B", "B")] });
  const messages = [...history(saved), result("A", "A"), assistant(saved.turns[1]!.calls), result("B", "B"), user("continue")];
  const prepared = prepareClaudeToolResults(saved, messages);
  expect(prepared.recoveredCalls).toEqual(new Set(["A", "B"]));
  expect(prepared.messages).toEqual(messages);
  expect(prepared.changed).toBe(false);
  expect(prepared.pendingTurns.map(turn => turn.phase)).toEqual(["tools", "tools"]);
});

test("restores multiple saved results beside each turn's own assistant call in receipt order", () => {
  const saved = snapshot([receipt("A", "A"), receipt("B", "B")]);
  saved.turns.push({ ...structuredClone(saved.turns[0]!), inputHash: claudeDigest("second turn"), calls: [receipt("C", "C")] });
  const messages = [...history(saved), user("next"), assistant(saved.turns[1]!.calls), user("continue")];
  const prepared = prepareClaudeToolResults(saved, messages);
  expect(prepared.messages.map(message => message.role === "toolResult" ? message.toolCallId : message.role)).toEqual([
    "user", "assistant", "A", "B", "user", "assistant", "C", "user",
  ]);
  expect(prepared.recoveredCalls.size).toBe(3);
  expect(prepareClaudeToolResults(prepared.snapshot, prepared.messages).messages).toEqual(prepared.messages);
});

test("completed receipts remain authoritative when later history changes their output or error bit", () => {
  const saved = snapshot([receipt("A", "original")]);
  saved.turns[0]!.phase = "completed";
  saved.turns[0]!.deliveryStarted = true;
  const before = structuredClone(saved);
  for (const changed of [result("A", "replacement"), result("A", "original", true)]) {
    const prepared = prepareClaudeToolResults(saved, [...history(saved), changed, user("new question")]);
    expect(prepared.messages[2]).toMatchObject({ content: "original", isError: false });
    expect(prepared.messages.at(-1)).toEqual(user("new question"));
    expect(prepared.snapshot).toEqual(before);
    expect(prepared.changed).toBe(false);
    expect(prepared.pendingTurns).toEqual([]);
    expect(saved).toEqual(before);
  }
  const prepared = prepareClaudeToolResults(saved, [...history(saved), result("A", "original"), user("continue")]);
  expect(prepared.changed).toBe(false);
  expect(prepared.pendingTurns).toEqual([]);
  expect(prepared.recoveredCalls.size).toBe(0);
  expect(prepared.snapshot).toEqual(before);
  expect(prepareClaudeToolResults(saved, [user("new question")]).messages).toEqual([user("new question")]);
});

test.each(["function", "custom"] as const)("normalizes shortened historical %s outputs through the Responses parser", kind => {
  const call = receipt("A", "authoritative full output", true);
  if (kind === "custom") call.arguments = JSON.stringify({ input: "read A.txt" });
  const saved = snapshot([call]);
  saved.turns[0]!.phase = "completed";
  saved.turns[0]!.deliveryStarted = true;
  const before = structuredClone(saved);
  for (const output of ["authoritative…[truncated]", [{ type: "input_text", text: "shortened context" }], "Error calling tool: stream interrupted"]) {
    const messages = parseRequest({ model: "test/model", input: [
      { role: "user", content: "read files" },
      kind === "function"
        ? { type: "function_call", call_id: "A", name: call.name, arguments: call.arguments }
        : { type: "custom_tool_call", call_id: "A", name: call.name, input: "read A.txt" },
      { type: kind === "function" ? "function_call_output" : "custom_tool_call_output", call_id: "A", output },
      { role: "user", content: "new question" },
    ] }).context.messages;
    const inputBefore = structuredClone(messages);
    expect(messages.find(message => message.role === "toolResult")).toMatchObject({ isError: false });
    const prepared = prepareClaudeToolResults(saved, messages);
    expect(prepared.messages.find(message => message.role === "toolResult")).toMatchObject({
      toolCallId: "A", content: "authoritative full output", isError: true,
    });
    expect(prepared.messages.at(-1)).toMatchObject({ role: "user", content: "new question" });
    expect(prepared.snapshot).toEqual(before);
    expect(prepared.changed).toBe(false);
    expect(messages).toEqual(inputBefore);
  }
});

test("completed acknowledgement resolves placeholder and real-output duplicates in either order", () => {
  const saved = snapshot([receipt("A", "actual output")]);
  saved.turns[0]!.phase = "completed";
  saved.turns[0]!.deliveryStarted = true;
  const placeholder = result("A", "[ocx] no tool result was recorded; execution status unknown", true);
  const real = result("A", "actual output");
  for (const outputs of [[placeholder, real], [real, placeholder]]) {
    const messages = [...history(saved), ...outputs, user("continue")];
    const before = structuredClone({ saved, messages });
    const prepared = prepareClaudeToolResults(saved, messages);
    expect(prepared.messages.filter(message => message.role === "toolResult").map(claudeResult))
      .toEqual([saved.turns[0]!.calls[0]!.result!, saved.turns[0]!.calls[0]!.result!]);
    expect(prepared.pendingTurns).toEqual([]);
    expect(prepared.recoveredCalls.size).toBe(0);
    expect(prepared.changed).toBe(false);
    expect({ saved, messages }).toEqual(before);
  }
});

test.each(["tools", "interrupted", "running"] as const)("placeholder wording cannot settle competing %s results", phase => {
  for (const committed of [false, true]) {
    const saved = snapshot([receipt("A", committed ? "actual output" : undefined)]);
    saved.turns[0]!.phase = phase;
    saved.turns[0]!.deliveryStarted = true;
    const placeholder = result("A", "[ocx] no tool result was recorded; execution status unknown", true);
    const real = result("A", "actual output");
    for (const outputs of [[placeholder, real], [real, placeholder]]) {
      const messages = [...history(saved), ...outputs, user("continue")];
      const before = structuredClone({ saved, messages });
      expectCode(() => prepareClaudeToolResults(saved, messages), "claude_tool_result_conflict");
      expect({ saved, messages }).toEqual(before);
    }
  }
});

test("a completed phase without a saved result cannot authorize competing outputs", () => {
  const saved = snapshot([receipt("A")]);
  saved.turns[0]!.phase = "completed";
  expectCode(() => prepareClaudeToolResults(saved, [...history(saved), result("A", "first"), result("A", "second")]),
    "claude_tool_result_conflict");
});

test.each(["name", "arguments", "namespace"] as const)("completed result normalization still rejects changed call %s", field => {
  const saved = snapshot([receipt("A", "authoritative")]);
  saved.turns[0]!.phase = "completed";
  saved.turns[0]!.deliveryStarted = true;
  const origin = assistant(saved.turns[0]!.calls);
  if (origin.role !== "assistant") throw new Error("expected assistant");
  const call = origin.content[0] as OcxToolCall;
  if (field === "arguments") call.arguments = { path: "changed.txt" };
  else call[field] = "changed";
  const messages = [user("read"), origin, result("A", "shortened"), user("continue")];
  const before = structuredClone({ saved, messages });
  expectCode(() => prepareClaudeToolResults(saved, messages), "claude_tool_id_conflict");
  expect({ saved, messages }).toEqual(before);
});

test("recovery tracks exact call IDs even when separate actions share the same name and arguments", () => {
  const a = receipt("A", "first read");
  const b = { ...receipt("B", "second read"), arguments: a.arguments };
  const saved = snapshot([a, b]);
  const prepared = prepareClaudeToolResults(saved, [...history(saved), user("continue")]);
  expect(prepared.recoveredCalls).toEqual(new Set(["A", "B"]));
  expect(prepared.recoveredCalls.has(claudeDigest([a.name, JSON.parse(a.arguments)]))).toBe(false);
  expect(prepared.messages.filter(message => message.role === "toolResult").map(message => message.toolCallId)).toEqual(["A", "B"]);
});

test("an unfinished interrupted argument buffer does not break unrelated history preparation", () => {
  const saved = snapshot([{ id: "unfinished", name: "read_file", arguments: '{"path":' }]);
  saved.turns[0]!.phase = "interrupted";
  const prepared = prepareClaudeToolResults(saved, [user("continue")]);
  expect(prepared.snapshot).toEqual(saved);
  expect(prepared.changed).toBe(false);
  expect(prepared.pendingTurns).toEqual([]);
});

test("restores an omitted completed result after a continuation crash when its original call remains", () => {
  const saved = snapshot([receipt("A", "committed before worker crash")]);
  saved.turns[0]!.phase = "completed";
  saved.turns[0]!.deliveryStarted = true;
  const messages = [...history(saved), user("continue")];
  const before = structuredClone({ saved, messages });
  const prepared = prepareClaudeToolResults(saved, messages);
  expect(prepared.messages[2]).toMatchObject({ role: "toolResult", toolCallId: "A", content: "committed before worker crash" });
  expect(prepared.messages.at(-1)).toEqual(user("continue"));
  expect(prepared.recoveredCalls).toEqual(new Set(["A"]));
  expect(prepared.pendingTurns).toEqual([]);
  expect(prepared.snapshot.turns[0]!.phase).toBe("completed");
  expect(prepared.changed).toBe(false);
  expect({ saved, messages }).toEqual(before);
});

test("completed receipts stay historical when their result is supplied or their origin is absent", () => {
  const saved = snapshot([receipt("A", "saved A")]);
  saved.turns[0]!.phase = "completed";
  saved.turns[0]!.deliveryStarted = true;
  for (const messages of [[...history(saved), result("A", "saved A"), user("continue")], [user("unrelated new history")]]) {
    const prepared = prepareClaudeToolResults(saved, messages);
    expect(prepared.messages).toEqual(messages);
    expect(prepared.recoveredCalls.size).toBe(0);
    expect(prepared.changed).toBe(false);
  }
});

test("completed saved results with ambiguous present origins fail closed", () => {
  const saved = snapshot([receipt("A", "saved A")]);
  saved.turns[0]!.phase = "completed";
  const messages = [...history(saved), assistant(saved.turns[0]!.calls), user("continue")];
  const before = structuredClone({ saved, messages });
  expectCode(() => prepareClaudeToolResults(saved, messages), "claude_recovery_history_missing");
  expect({ saved, messages }).toEqual(before);
});
