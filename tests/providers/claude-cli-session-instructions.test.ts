import { describe, expect, spyOn, test } from "bun:test";
import * as sessionContext from "../../src/adapters/claude-cli/session-context";
import { claudeDigest, claudeHistory } from "../../src/adapters/claude-cli/session-context";
import { ClaudeSessionError } from "../../src/adapters/claude-cli/session-errors";
import { extractClaudeInstructions, restoreClaudeInstructions } from "../../src/adapters/claude-cli/session-instructions";
import type { ClaudeQueuedInstruction } from "../../src/adapters/claude-cli/session-journal";
import type { OcxMessage } from "../../src/types";

const user = (content: string): OcxMessage => ({ role: "user", content, timestamp: 0 });
const assistant = (text: string): OcxMessage => ({ role: "assistant", content: [{ type: "text", text }], timestamp: 1 });
const result: OcxMessage = { role: "toolResult", toolCallId: "call-A", toolName: "read_file", content: "literal output", isError: false, timestamp: 2 };
const prefix: OcxMessage[] = [user("Read the report"), {
  role: "assistant", timestamp: 1, content: [{ type: "toolCall", id: "call-A", name: "read_file", arguments: {} }],
}];
const history = claudeHistory(prefix);
function entry(texts = ["Keep it local"], patch: Partial<ClaudeQueuedInstruction> = {}): ClaudeQueuedInstruction {
  return { inputHash: claudeDigest(texts), history, texts, phase: "queued", updatedAt: 3, ...patch };
}
const texts = (messages: OcxMessage[]) => messages.filter(message => message.role === "user").map(message => message.content);

describe("extractClaudeInstructions", () => {
  test.each([false, true])("keeps all user instructions with results first=%s", resultsFirst => {
    const suffix = resultsFirst ? [result, user("First"), assistant("partial"), user("Second")]
      : [user("First"), assistant("partial"), result, user("Second")];
    expect(extractClaudeInstructions([...prefix, ...suffix], history)).toEqual(["First", "Second"]);
  });

  test("finds an exact raw boundary before collapsed assistant partials", () => {
    const initial = [user("Question"), assistant("Start")];
    expect(extractClaudeInstructions([...initial, assistant(" more"), user("New instruction")], claudeHistory(initial))).toEqual(["New instruction"]);
    expect(extractClaudeInstructions([user("Question"), assistant("Start more"), user("New instruction")], claudeHistory(initial))).toBeUndefined();
  });

  test("requires a proven prefix and a new nonempty user text", () => {
    for (const messages of [prefix, [...prefix, result, assistant("partial")], [user("Changed"), ...prefix.slice(1), user("new")], [user("new")]]) {
      expect(extractClaudeInstructions(messages, history)).toBeUndefined();
    }
    expect(extractClaudeInstructions([], [])).toBeUndefined();
    expect(extractClaudeInstructions([...prefix, user("")], history)).toBeUndefined();
    expect(extractClaudeInstructions([user("New")], [])).toEqual(["New"]);
  });

  test("reads literal text parts only, leaving developer instructions to the system prompt", () => {
    const messages: OcxMessage[] = [...prefix,
      { role: "developer", content: "system only", timestamp: 0 },
      { role: "user", timestamp: 4, content: [{ type: "text", text: " first " }, { type: "image", imageUrl: "https://example.com/image.png" }, { type: "text", text: "second\n" }] },
      { role: "user", timestamp: 4, content: [{ type: "image", imageUrl: "https://example.com/image.png" }] },
    ];
    expect(extractClaudeInstructions(messages, history)).toEqual([" first \nsecond\n"]);
  });

  test("does not truncate arbitrary multiple user messages", () => {
    const all = Array.from({ length: 100 }, (_, index) => `Instruction ${index}`);
    expect(extractClaudeInstructions([...prefix, ...all.map(user)], history)).toEqual(all);
  });

  test("hashes only candidate boundaries after cheaply counting merged and skipped history items", () => {
    const initial: OcxMessage[] = Array.from({ length: 1000 }, (_, index) => user(`Old ${index}`));
    initial.push(assistant("Start"), { role: "developer", content: "system", timestamp: 0 },
      { role: "assistant", content: [{ type: "thinking", thinking: "private" }], timestamp: 0 }, assistant(" end"));
    const hashes = claudeHistory(initial);
    const hashHistory = spyOn(sessionContext, "claudeHistory");
    try {
      expect(extractClaudeInstructions([...initial, user("New")], hashes)).toEqual(["New"]);
      expect(hashHistory).toHaveBeenCalledTimes(2);
      expect(hashHistory.mock.calls.map(([messages]) => messages.length)).toEqual([1001, 1004]);
    } finally { hashHistory.mockRestore(); }
  });
});

describe("restoreClaudeInstructions", () => {
  test.each(["queued", "sent", "completed"] as const)("replays %s text absent from client history before the following user", phase => {
    const messages = [...prefix, result, user("Now summarize")];
    const before = structuredClone(messages);
    const saved = entry(["Keep it local", "Do not upload"], { phase });
    const restored = restoreClaudeInstructions(messages, [saved]);
    expect(texts(restored)).toEqual(["Read the report", "Keep it local", "Do not upload", "Now summarize"]);
    expect(restored.slice(2, 3)).toEqual([result]);
    expect(restored[3]).toEqual({ role: "user", content: "Keep it local", timestamp: 3 });
    expect(messages).toEqual(before);
    expect(restoreClaudeInstructions(restored, [saved])).toEqual(restored);
  });

  test.each([false, true])("restores completed B before C's answer and new D, with C user present=%s", hasCUser => {
    const original = [...prefix, assistant("Earlier partial A")];
    const following = [...(hasCUser ? [user("C")] : []), assistant("Completed answer to C"), user("D")];
    const messages = [...original, result, ...following];
    const saved = entry(["B"], { phase: "completed", history: claudeHistory(original) });
    const restored = restoreClaudeInstructions(messages, [saved]);
    expect(restored).toEqual([...original, result, { role: "user", content: "B", timestamp: 3 }, ...following]);
    expect(restored.find(message => message.role === "toolResult")).toBe(result);
    expect(restoreClaudeInstructions(restored, [saved])).toEqual(restored);
    expect(messages).toEqual([...original, result, ...following]);
  });

  test("inserts missing completed text before intervening answers even when a later accepted text is present", () => {
    const messages = [...prefix, result, assistant("C's completed answer"), user("D"), user("E")];
    const saved = [entry(["B"], { phase: "completed" }), entry(["D"], { phase: "completed" })];
    expect(restoreClaudeInstructions(messages, saved)).toEqual([
      ...prefix, result, { role: "user", content: "B", timestamp: 3 }, assistant("C's completed answer"), user("D"), user("E"),
    ]);
  });

  test.each([false, true])("preserves result content and order when results follow the user=%s", resultsLast => {
    const messages = [...prefix, ...(resultsLast ? [user("New"), result] : [result, user("New")])];
    const restored = restoreClaudeInstructions(messages, [entry()]);
    expect(restored).toEqual([...prefix, ...(resultsLast ? [user("Keep it local"), user("New"), result] : [result, user("Keep it local"), user("New")])].map(message =>
      message.role === "user" && message.content === "Keep it local" ? { ...message, timestamp: 3 } : message));
    expect(restored.find(message => message.role === "toolResult")).toBe(result);
  });

  test("recognizes already supplied texts after their history prefix, including across results and partial assistants", () => {
    const saved = entry(["One", "Two"]);
    const messages = [...prefix, user("One"), result, assistant("partial"), user("Two"), user("New")];
    expect(restoreClaudeInstructions(messages, [saved])).toEqual(messages);
    expect(texts(restoreClaudeInstructions([...prefix, user("Two"), user("New")], [saved]))).toEqual(["Read the report", "One", "Two", "New"]);
    expect(texts(restoreClaudeInstructions([...prefix, user("One"), user("New")], [saved]))).toEqual(["Read the report", "One", "Two", "New"]);
  });

  test("keeps acceptance order when a later queued instruction is already in history", () => {
    const saved = [entry(["One"], { updatedAt: 9 }), entry(["Two"], { updatedAt: 4 }), entry(["Three"])];
    const restored = restoreClaudeInstructions([...prefix, user("Two"), result, user("New")], saved);
    expect(texts(restored)).toEqual(["Read the report", "One", "Two", "Three", "New"]);
    expect(restoreClaudeInstructions(restored, saved)).toEqual(restored);
  });

  test("rejects out-of-order accepted texts rather than duplicating supplied instructions", () => {
    const messages = [...prefix, user("Two"), result, user("One"), user("New")];
    for (const entries of [[entry(["One", "Two"])], [entry(["One"]), entry(["Two"])]]) {
      expect(() => restoreClaudeInstructions(messages, entries)).toThrow("out of order");
    }
    expect(texts(messages)).toEqual(["Read the report", "Two", "One", "New"]);
  });

  test("proves later prefixes that include an earlier restored instruction", () => {
    const saved = [entry(["One"]), entry(["Two"], { history: claudeHistory([...prefix, user("One")]) })];
    expect(texts(restoreClaudeInstructions([...prefix, user("New")], saved))).toEqual(["Read the report", "One", "Two", "New"]);
  });

  test("deduplicates a same-hash retry despite optional operation metadata", () => {
    const first = entry(["One", "Two"]);
    const retry = { ...first, operationId: "a".repeat(64), phase: "sent" as const, updatedAt: 99 };
    expect(texts(restoreClaudeInstructions([...prefix, user("New")], [first, retry]))).toEqual(["Read the report", "One", "Two", "New"]);
    const independent = { ...first, inputHash: "b".repeat(64), operationId: "b".repeat(64) };
    expect(texts(restoreClaudeInstructions([...prefix, user("New")], [first, independent]))).toEqual(["Read the report", "One", "Two", "One", "Two", "New"]);
  });

  test("does not mistake the old prefix's identical text for an accepted instruction", () => {
    const old = [user("Same")];
    expect(texts(restoreClaudeInstructions([...old, user("New")], [entry(["Same"], { history: claudeHistory(old) })]))).toEqual(["Same", "Same", "New"]);
  });

  test("skips unproven completed histories but rejects unproven pending histories without clearing their queue", () => {
    const messages = [...prefix, result, user("New")];
    expect(restoreClaudeInstructions(messages, [entry(["Done"], { phase: "completed", history: ["a".repeat(64)] })])).toEqual(messages);
    for (const phase of ["queued", "sent"] as const) {
      const entries = [entry(), entry(["Unproven"], { phase, history: ["a".repeat(64)] })];
      const before = structuredClone({ messages, entries });
      expect(() => restoreClaudeInstructions(messages, entries)).toThrow(ClaudeSessionError);
      try { restoreClaudeInstructions(messages, entries); }
      catch (error) {
        expect(error).toMatchObject({ code: "claude_instruction_history_missing", status: 409, retryable: true });
      }
      expect({ messages, entries }).toEqual(before);
    }
  });

  test("appends after results if there is no new user and preserves repeated literal texts", () => {
    const all = Array.from({ length: 100 }, () => " repeated \ntext ");
    const restored = restoreClaudeInstructions([...prefix, result], [entry(all)]);
    expect(restored.slice(0, 3)).toEqual([...prefix, result]);
    expect(texts(restored)).toEqual(["Read the report", ...all]);
    expect(restoreClaudeInstructions(restored, [entry(all)])).toEqual(restored);
  });
});
