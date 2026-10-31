import { describe, expect, test } from "bun:test";
import { nativeWorker, workerGate, workerNativeBinary, workerUntil } from "../fixtures/claude-session-worker-native-api";

const firstUuid = "11111111-1111-4111-8111-111111111111";
const appendUuid = "22222222-2222-4222-8222-222222222222";
const nextUuid = "33333333-3333-4333-8333-333333333333";
const text = (text: string) => ({ type: "text" as const, text });

describe.skipIf(!workerNativeBinary)("Claude worker native follow-up contract (SDK 0.3.314 / CLI 2.1.314)", () => {
  test.each(["reject", "resolve", "abort"])("stop during pending MCP never starts another inference (settlement=%s)", async settlement => {
    let started = false;
    let signal!: AbortSignal;
    let resolve!: (value: { content: Array<{ type: "text"; text: string }> }) => void;
    let reject!: (error: Error) => void;
    const receipt = new Promise<{ content: Array<{ type: "text"; text: string }> }>((done, fail) => { resolve = done; reject = fail; });
    const h = await nativeWorker([
      { content: [{ type: "tool_use", id: "toolu_worker_shutdown", name: "mcp__opencodex__run", input: {} }] },
    ], async (_call, value) => { signal = value; started = true; return receipt; });
    try {
      await h.worker.send("pending external operation", { uuid: firstUuid });
      await workerUntil(() => started, "pending shutdown tool");
      expect(h.requests).toHaveLength(1);
      const closing = h.worker.stop();
      if (settlement === "reject") reject(new Error("Detached manager pending promise rejected"));
      if (settlement === "resolve") resolve({ content: [text("Late external result")] });
      await closing;
      expect(signal.aborted).toBe(true);
      expect(h.requests).toHaveLength(1);
      expect(h.unexpected).toEqual([]);
      expect(h.children[0]!.exitCode).not.toBeNull();
      expect(h.children[0]!.killed).toBe(false);
      expect(h.children[0]!.signalCode).toBeNull();
      expect(h.errors).toEqual([]);
    } finally { resolve({ content: [] }); await h.close(); }
  }, 15_000);

  test("stop allows native stdin shutdown to exit cleanly before resolving", async () => {
    const h = await nativeWorker([{ content: [text("ready to close")] }]);
    try {
      await h.worker.send("one turn", { uuid: firstUuid });
      await workerUntil(() => h.messages.some(frame => frame.type === "result"), "native result");
      const closing = h.worker.stop();
      expect(h.worker.stop()).toBe(closing);
      await closing;
      expect(h.children).toHaveLength(1);
      expect(h.children[0]!.exitCode).toBe(0);
      expect(h.children[0]!.signalCode).toBeNull();
      expect(h.errors).toEqual([]);
    } finally { await h.close(); }
  }, 15_000);

  test.each([false, true])("proves shouldQuery:true queue correlation (pending MCP=%s)", async pendingMcp => {
    const held = workerGate();
    let toolStarted = false;
    const h = await nativeWorker([
      pendingMcp
        ? { content: [{ type: "tool_use", id: "toolu_worker_queued", name: "mcp__opencodex__run", input: {} }] }
        : { content: [text("ORIGINAL_REPLY")], wait: held.wait },
      { content: [text("QUEUED_OR_FOLDED_REPLY")] },
      { content: [text("SEPARATE_QUEUED_REPLY")] },
    ], async () => {
      toolStarted = true;
      await held.wait;
      return { content: [text("ACTUAL_QUEUED_TOOL_RECEIPT")] };
    });
    try {
      h.worker.send("ORIGINAL_PROMPT", { uuid: firstUuid });
      await workerUntil(() => pendingMcp ? toolStarted : h.requests.length === 1, "active native work");
      h.worker.send("QUEUED_QUERYING_USER", { uuid: appendUuid, shouldQuery: true, priority: "next" });
      await h.barrier(appendUuid);
      expect(h.requests).toHaveLength(1);
      held.release();
      await workerUntil(() => {
        const results = h.raw.filter(frame => frame.type === "result");
        return results.length === 2 || results.some(frame => (frame.user_message_uuids as string[] | undefined)?.includes(appendUuid));
      }, "queued user is consumed");
      expect(h.requests).toHaveLength(2);
      expect(JSON.stringify(h.requests[1]!.messages)).toContain("QUEUED_QUERYING_USER");
      expect(h.raw.filter(frame => frame.type === "result")).toMatchObject(pendingMcp ? [
        { subtype: "success", result: "QUEUED_OR_FOLDED_REPLY",
          user_message_uuid: firstUuid, user_message_uuids: [firstUuid, appendUuid] },
      ] : [
        { subtype: "success", result: "ORIGINAL_REPLY", user_message_uuid: firstUuid, user_message_uuids: [firstUuid] },
        { subtype: "success", result: "QUEUED_OR_FOLDED_REPLY", user_message_uuid: appendUuid, user_message_uuids: [appendUuid] },
      ]);
      expect(h.children).toHaveLength(1);
      expect(h.errors).toEqual([]);
      expect(h.unexpected).toEqual([]);
    } finally { held.release(); await h.close(); }
  }, 30_000);

  test("direct follow-ups keep leading slash commands and file mentions literal", async () => {
    const h = await nativeWorker([
      { content: [text("first reply")] }, { content: [text("literal slash reply")] }, { content: [text("literal mention reply")] },
    ]);
    try {
      h.worker.send("bootstrap", { uuid: firstUuid });
      await workerUntil(() => h.messages.some(frame => frame.type === "result"), "bootstrap result");
      for (const [index, prompt] of ["/init", "@definitely-missing-worker-proof.txt follow the literal instruction"].entries()) {
        h.worker.send(prompt, { uuid: index === 0 ? appendUuid : nextUuid });
        await workerUntil(() => h.messages.filter(frame => frame.type === "result").length === index + 2, "literal follow-up result");
        expect(JSON.stringify(h.requests[index + 1]!.messages)).toContain(prompt);
      }
      expect(h.requests).toHaveLength(3);
      expect(h.children).toHaveLength(1);
      expect(h.raw.some(frame => frame.type === "local_command_output")).toBe(false);
      expect(h.errors).toEqual([]);
      expect(h.unexpected).toEqual([]);
    } finally { await h.close(); }
  }, 30_000);

  test.each([false, true])("proves shouldQuery:false and UUID correlation while MCP is pending (awaitWrite=%s)", async awaitWrite => {
    const receipt = workerGate();
    let started = false;
    let settled = false;
    let signal!: AbortSignal;
    const h = await nativeWorker([
      { content: [{ type: "tool_use", id: "toolu_worker_pending", name: "mcp__opencodex__run", input: {} }] },
      { content: [text("after actual tool receipt")] },
      { content: [text("after querying follow-up")] },
    ], async (call, value) => {
      expect(call.id).toBe("toolu_worker_pending");
      signal = value; started = true;
      await receipt.wait;
      settled = true;
      return { content: [text("ACTUAL_EXTERNAL_RECEIPT")] };
    });
    try {
      expect(h.version()).toMatch(/^2\.1\.314\b/);
      h.worker.send("FIRST_USER", { uuid: firstUuid });
      await workerUntil(() => started, "external MCP call starts");
      const sent = h.worker.send("APPENDED_WHILE_MCP_PENDING", { uuid: appendUuid, shouldQuery: false });
      if (awaitWrite) await sent;
      expect(h.requests).toHaveLength(1);
      expect(settled).toBe(false);
      expect(signal.aborted).toBe(false);
      receipt.release();
      await workerUntil(() => h.messages.some(frame => frame.type === "result"), "first terminal result");
      expect(h.requests).toHaveLength(2);
      expect(JSON.stringify(h.requests[1]!.messages)).toContain("ACTUAL_EXTERNAL_RECEIPT");
      expect(JSON.stringify(h.requests[1]!.messages)).toContain("APPENDED_WHILE_MCP_PENDING");
      // A transcript append joins this tool continuation without becoming a new turn.
      expect(h.raw.filter(frame => frame.type === "result")).toMatchObject([{
        subtype: "success", result: "after actual tool receipt",
        user_message_uuid: firstUuid, user_message_uuids: [firstUuid, appendUuid],
      }]);
      const assistants = h.raw.filter(frame => frame.type === "assistant");
      expect(assistants[0]).toMatchObject({ user_message_uuid: firstUuid, user_message_uuids: [firstUuid] });
      expect(assistants[1]).not.toHaveProperty("user_message_uuid");
      expect(assistants[1]).not.toHaveProperty("user_message_uuids");
      h.worker.send("QUERYING_FOLLOW_UP", { uuid: nextUuid });
      await workerUntil(() => h.messages.filter(frame => frame.type === "result").length === 2, "next terminal result");
      expect(h.requests).toHaveLength(3);
      expect(JSON.stringify(h.requests[2]!.messages)).toContain("APPENDED_WHILE_MCP_PENDING");
      expect(h.raw.filter(frame => frame.user_message_uuid).map(frame => ({
        type: frame.type, user_message_uuid: frame.user_message_uuid, user_message_uuids: frame.user_message_uuids,
      }))).toEqual([
        { type: "stream_event", user_message_uuid: firstUuid, user_message_uuids: [firstUuid] },
        { type: "assistant", user_message_uuid: firstUuid, user_message_uuids: [firstUuid] },
        { type: "result", user_message_uuid: firstUuid, user_message_uuids: [firstUuid, appendUuid] },
        { type: "stream_event", user_message_uuid: nextUuid, user_message_uuids: [nextUuid] },
        { type: "assistant", user_message_uuid: nextUuid, user_message_uuids: [nextUuid] },
        { type: "result", user_message_uuid: nextUuid, user_message_uuids: [nextUuid] },
      ]);
      for (const result of h.raw.filter(frame => frame.type === "result")) {
        expect(result.uuid).toBeString();
        expect([firstUuid, appendUuid, nextUuid]).not.toContain(result.uuid);
      }
      expect(h.children).toHaveLength(1);
      expect(h.children[0]!.exitCode).toBeNull();
      expect(h.errors).toEqual([]);
      expect(h.unexpected).toEqual([]);
    } finally { receipt.release(); await h.close(); }
  }, 30_000);

  test("native interrupt acknowledges before terminal result and the same child accepts the next query", async () => {
    const held = workerGate();
    const h = await nativeWorker([
      { content: [text("STALE_HELD_REPLY")], wait: held.wait },
      { content: [text("SAME_PROCESS_FOLLOW_UP")] },
    ]);
    try {
      h.worker.send("slow user", { uuid: firstUuid });
      await workerUntil(() => h.requests.length === 1, "inference begins");
      const boundary = h.raw.length;
      await h.worker.interrupt();
      await workerUntil(() => h.messages.some(frame => frame.type === "result"), "interrupted terminal result");
      const tail = h.raw.slice(boundary);
      const ack = tail.findIndex(frame => frame.type === "control_response");
      const result = tail.findIndex(frame => frame.type === "result");
      expect(ack).toBeGreaterThanOrEqual(0);
      expect(result).toBeGreaterThan(ack);
      h.worker.send("new user", { uuid: nextUuid, priority: "now" });
      await workerUntil(() => h.messages.filter(frame => frame.type === "result").length === 2, "follow-up terminal result");
      expect(h.raw.filter(frame => frame.type === "result")).toMatchObject([
        { subtype: "error_during_execution", is_error: true, terminal_reason: "aborted_streaming",
          user_message_uuid: firstUuid, user_message_uuids: [firstUuid] },
        { subtype: "success", is_error: false, terminal_reason: "completed",
          user_message_uuid: nextUuid, user_message_uuids: [nextUuid] },
      ]);
      expect(h.messages.filter(frame => frame.type === "result").at(-1)).toMatchObject({ result: "SAME_PROCESS_FOLLOW_UP" });
      expect(h.children).toHaveLength(1);
      expect(h.children[0]!.exitCode).toBeNull();
      expect(h.errors).toEqual([]);
      expect(h.unexpected).toEqual([]);
    } finally { held.release(); await h.close(); }
  }, 30_000);
});
