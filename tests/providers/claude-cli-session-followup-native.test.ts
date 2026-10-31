import { describe, expect, test } from "bun:test";
import type { AdapterEvent } from "../../src/types";
import type { ClaudePersistentTurn } from "../../src/adapters/claude-cli/session-manager";
import { MemoryJournal, assistant, call, result, toolAssistant, user, withMessages } from "../fixtures/claude-session-worker";
import { gate, nativeBinary, nativeSession, nativeUntil, type NativeBlock } from "../fixtures/claude-session-native-api";

const text = (text: string): NativeBlock => ({ type: "text", text });
const output = (events: AdapterEvent[]) => events.filter(event => event.type === "text_delta").map(event => event.text).join("");
const calls = (events: AdapterEvent[]) => events.filter(event => event.type === "tool_call_start");
const occurrences = (text: string, marker: string) => text.split(marker).length - 1;
const instructionB = "NATIVE_ADDITIONAL_INSTRUCTION_B_74";
const instructionC = "NATIVE_ADDITIONAL_INSTRUCTION_C_85";
const actualResultA = "ACTUAL_EXTERNAL_RESULT_A_96";

function operation(input: ClaudePersistentTurn, id: string, messages = input.parsed.context.messages) {
  const next = withMessages(input, messages);
  next.incoming = { ...input.incoming, headers: new Headers(input.incoming.headers) };
  next.incoming.headers.set("idempotency-key", id);
  return next;
}

function final(events: AdapterEvent[], answer: string) {
  expect(events.filter(event => event.type === "error")).toEqual([]);
  expect(calls(events)).toEqual([]);
  expect(output(events)).toBe(answer);
  expect(events.at(-1)).toMatchObject({ type: "done", stopReason: "stop" });
}

function live(h: ReturnType<typeof nativeSession>) {
  expect(h.children).toHaveLength(1);
  expect(h.children[0]!.exitCode).toBeNull();
  expect(h.children[0]!.signalCode).toBeNull();
  expect(h.unexpected).toEqual([]);
}

function queuedB(h: ReturnType<typeof nativeSession>) {
  return h.journal.load(h.input.scope)?.instructions?.some(instruction => instruction.phase === "queued"
    && instruction.texts.some(text => text.includes(instructionB))) === true;
}

async function holdToolThenQueueB(h: ReturnType<typeof nativeSession>, action: ReturnType<typeof call>) {
  const first = h.run(operation(h.input, "native-tool-A"));
  await first.done;
  expect(first.events.filter(event => event.type === "error")).toEqual([]);
  expect(calls(first.events)).toEqual([{ type: "tool_call_start", id: action.id, name: "read_file" }]);
  expect(first.events.at(-1)).toMatchObject({ type: "done", stopReason: "tool_use" });
  const history = [...h.input.parsed.context.messages, toolAssistant([action])];
  const pending = h.run(operation(h.input, "native-instruction-B", [...history, user(instructionB)]));
  let settled = false;
  let failure: unknown;
  void pending.done.then(() => { settled = true; }, error => { settled = true; failure = error; });
  await nativeUntil(() => settled || queuedB(h), "B is durably queued while the actual tool awaits its client result");
  expect(failure).toBeUndefined();
  expect(settled).toBe(false);
  expect(pending.events.every(event => event.type === "heartbeat")).toBe(true);
  expect(queuedB(h)).toBe(true);
  expect(h.requested).toEqual([action.id]);
  expect(h.resolved).toEqual([]);
  expect(h.requests).toHaveLength(1);
  live(h);
  return { pending, history };
}

describe.skipIf(!nativeBinary)("Claude native additional-instruction acceptance", () => {
  test("B interrupts API-held A through SDK control and finishes on the same live child", async () => {
    const held = gate();
    const h = nativeSession([
      { content: [text("STALE_A_NATIVE_OUTPUT")], wait: held.wait },
      { content: [text("B completed on original child")] },
    ]);
    try {
      const first = h.run(operation(h.input, "native-active-A"));
      await nativeUntil(() => h.requests.length === 1, "A reaches the held local API");
      const original = h.children[0]!;
      const newer = h.run(operation(h.input, "native-active-B", [...h.input.parsed.context.messages, user(instructionB)]));
      await newer.done;
      await first.done;
      final(newer.events, "B completed on original child");
      expect(h.interrupts).toBe(1);
      expect(h.acknowledgedInterrupts).toBe(1);
      live(h);
      expect(h.children[0]).toBe(original);
      expect(h.requests).toHaveLength(2);
      const messages = JSON.stringify(h.requests[1]!.messages);
      expect(messages).toContain("read both files");
      expect(occurrences(messages, instructionB)).toBe(1);
      const before = structuredClone(newer.events);
      held.release();
      await nativeUntil(() => h.replied.includes(1), "old A API frames are released after B completes");
      await Bun.sleep(50);
      expect(newer.events).toEqual(before);
      expect(output([...first.events, ...newer.events])).not.toContain("STALE_A_NATIVE_OUTPUT");
      expect(h.requests).toHaveLength(2);
      live(h);
    } finally { held.release(); await h.close(); }
  }, 45_000);

  test("B and C arriving during a real interrupt retain both instructions and ignore late A frames", async () => {
    const held = gate();
    const interrupted = gate();
    const h = nativeSession([
      { content: [text("STALE_A_DURING_QUEUED_INSTRUCTIONS")], wait: held.wait },
      { content: [text("queued instructions completed")] },
      { content: [text("queued instructions completed")] },
    ], { interruptWait: interrupted.wait });
    try {
      const first = h.run(operation(h.input, "native-queued-A"));
      await nativeUntil(() => h.requests.length === 1, "A reaches the held API");
      const historyB = [...h.input.parsed.context.messages, user(instructionB)];
      const b = h.run(operation(h.input, "native-queued-B", historyB));
      await nativeUntil(() => h.acknowledgedInterrupts === 1 || h.children.length > 1 || b.events.some(event => event.type === "error"),
        "the real SDK acknowledges interrupt without replacing the child");
      expect(h.acknowledgedInterrupts).toBe(1);
      live(h);
      const c = h.run(operation(h.input, "native-queued-C", [...historyB, user(instructionC)]));
      await Bun.sleep(30);
      interrupted.release();
      await c.done;
      await b.done;
      await first.done;
      final(c.events, "queued instructions completed");
      const messages = JSON.stringify(h.requests.at(-1)!.messages);
      expect(occurrences(messages, instructionB)).toBe(1);
      expect(occurrences(messages, instructionC)).toBe(1);
      expect(messages).toContain("read both files");
      expect(h.interrupts).toBeGreaterThanOrEqual(1);
      expect(h.acknowledgedInterrupts).toBe(h.interrupts);
      live(h);
      const before = structuredClone(c.events);
      const requestCount = h.requests.length;
      held.release();
      await nativeUntil(() => h.replied.includes(1), "late A response is released");
      await Bun.sleep(50);
      expect(c.events).toEqual(before);
      expect(output([...first.events, ...b.events, ...c.events])).not.toContain("STALE_A_DURING_QUEUED_INSTRUCTIONS");
      expect(h.requests).toHaveLength(requestCount);
      live(h);
    } finally { interrupted.release(); held.release(); await h.close(); }
  }, 45_000);

  test("B waits for actual MCP result A, then reshaped C and cached next-user D retain B on the same child", async () => {
    const replies = [{ content: [] as NativeBlock[] }, { content: [text("real A result and B applied")] },
      { content: [text("next user retains B and A")] }];
    const h = nativeSession(replies);
    const action = call(h.input, "toolu_native_deferred_A", { path: "a.txt" });
    replies[0]!.content = [{ type: "tool_use", id: action.id, name: `mcp__opencodex__${action.name}`, input: action.arguments }];
    try {
      const { pending, history } = await holdToolThenQueueB(h, action);
      const original = h.children[0]!;
      const c = h.run(operation(h.input, "native-result-C", [...history, result(action.id, actualResultA)]));
      await c.done;
      await pending.done;
      final(c.events, "real A result and B applied");
      expect(pending.events.filter(event => event.type === "error")).toEqual([]);
      expect(h.requests).toHaveLength(2);
      const messages = JSON.stringify(h.requests[1]!.messages);
      expect(occurrences(messages, instructionB)).toBe(1);
      expect(occurrences(messages, actualResultA)).toBe(1);
      expect(messages).toContain(action.id);
      expect(h.requested).toEqual([action.id]);
      expect(h.resolved).toEqual([action.id]);
      expect(h.maxActive).toBe(1);
      expect(h.children[0]).toBe(original);
      live(h);
      const cachedHistory = [...history, result(action.id, actualResultA), assistant("real A result and B applied")];
      const nextInput = operation(h.input, "native-next-user-D", [...cachedHistory, user("NATIVE_NEXT_USER_D_107")]);
      expect(JSON.stringify(nextInput.parsed.context.messages)).not.toContain(instructionB);
      const d = h.run(nextInput);
      await d.done;
      final(d.events, "next user retains B and A");
      expect(h.requests).toHaveLength(3);
      const nextMessages = JSON.stringify(h.requests[2]!.messages);
      for (const marker of [instructionB, actualResultA, "real A result and B applied", "NATIVE_NEXT_USER_D_107"]) {
        expect(occurrences(nextMessages, marker)).toBe(1);
      }
      expect(nextMessages).toContain(action.id);
      expect(h.requested).toEqual([action.id]);
      expect(h.resolved).toEqual([action.id]);
      expect(h.children[0]).toBe(original);
      live(h);
    } finally { await h.close(); }
  }, 45_000);

  test("queued B survives manager close and cold recovery when C omits it", async () => {
    const journal = new MemoryJournal();
    const replies = [{ content: [] as NativeBlock[] }];
    const h = nativeSession(replies, { journal });
    const action = call(h.input, "toolu_native_cold_A", { path: "a.txt" });
    replies[0]!.content = [{ type: "tool_use", id: action.id, name: `mcp__opencodex__${action.name}`, input: action.arguments }];
    let oldPid: number | undefined;
    try {
      const { pending } = await holdToolThenQueueB(h, action);
      oldPid = h.children[0]!.pid;
      await h.manager.close();
      await pending.done;
      expect(queuedB(h)).toBe(true);
      expect(h.resolved).toEqual([]);
    } finally { await h.close(); }
    const reopened = nativeSession([{ content: [text("cold A result and restored B applied")] }], { journal });
    try {
      const c = reopened.run(operation(reopened.input, "native-cold-result-C", [...reopened.input.parsed.context.messages,
        toolAssistant([action]), result(action.id, actualResultA)]));
      await c.done;
      final(c.events, "cold A result and restored B applied");
      expect(reopened.requests).toHaveLength(1);
      const messages = JSON.stringify(reopened.requests[0]!.messages);
      expect(occurrences(messages, instructionB)).toBe(1);
      expect(occurrences(messages, actualResultA)).toBe(1);
      expect(reopened.requested).toEqual([]);
      expect(reopened.resolved).toEqual([]);
      expect(reopened.children[0]!.pid).not.toBe(oldPid);
      live(reopened);
    } finally { await reopened.close(); }
  }, 45_000);

  test("a new user may deliberately repeat A's arguments with a fresh native tool ID", async () => {
    const replies = [{ content: [] as NativeBlock[] }, { content: [text("A completed")] },
      { content: [] as NativeBlock[] }, { content: [text("deliberate repeat completed")] }];
    const h = nativeSession(replies);
    const a = call(h.input, "toolu_native_original", { path: "same.txt" });
    const repeat = { ...a, id: "toolu_native_deliberate_repeat" };
    for (const [index, action] of [[0, a], [2, repeat]] as const) replies[index]!.content = [{
      type: "tool_use", id: action.id, name: `mcp__opencodex__${action.name}`, input: action.arguments,
    }];
    try {
      const first = h.run(operation(h.input, "native-repeat-A"));
      await first.done;
      expect(calls(first.events).map(event => event.id)).toEqual([a.id]);
      const historyA = [...h.input.parsed.context.messages, toolAssistant([a]), result(a.id, actualResultA)];
      const afterA = h.run(operation(h.input, "native-repeat-A-result", historyA));
      await afterA.done;
      final(afterA.events, "A completed");
      const historyRepeat = [...historyA, assistant("A completed"), user("Deliberately read same.txt again now")];
      const again = h.run(operation(h.input, "native-repeat-new-user", historyRepeat));
      await again.done;
      expect(again.events.filter(event => event.type === "error")).toEqual([]);
      expect(calls(again.events).map(event => event.id)).toEqual([repeat.id]);
      expect(again.events.at(-1)).toMatchObject({ type: "done", stopReason: "tool_use" });
      const completed = h.run(operation(h.input, "native-repeat-result", [...historyRepeat,
        toolAssistant([repeat]), result(repeat.id, "ACTUAL_REPEAT_RESULT")]));
      await completed.done;
      final(completed.events, "deliberate repeat completed");
      expect(h.requested).toEqual([a.id, repeat.id]);
      expect(h.resolved).toEqual([a.id, repeat.id]);
      expect(h.requests).toHaveLength(4);
      const messages = JSON.stringify(h.requests[3]!.messages);
      expect(messages).toContain("Deliberately read same.txt again now");
      expect(messages).toContain("ACTUAL_REPEAT_RESULT");
      live(h);
    } finally { await h.close(); }
  }, 45_000);
});
