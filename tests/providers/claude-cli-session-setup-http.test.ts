import { expect, test } from "bun:test";
import type { SessionMessage } from "@anthropic-ai/claude-agent-sdk";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createClaudeCliAdapter } from "../../src/adapters/claude-cli/adapter";
import { clearClaudeCliBinaryCache } from "../../src/adapters/claude-cli/profiles";
import type { ClaudeSessionManagerDeps } from "../../src/adapters/claude-cli/session-manager";
import type { ClaudeNativeSessionIO } from "../../src/adapters/claude-cli/session-native";
import { setClaudeUsagePreflightForTests } from "../../src/adapters/claude-cli/usage-admission";
import { getAdapterDefinition } from "../../src/adapters/registry";
import { flushConfigDirHardeningForTests } from "../../src/config/paths";
import { clearResponseStateForTests, flushResponseState } from "../../src/responses/state";
import { resolveResponsesApiAuth } from "../../src/server/auth-cors";
import { handleResponses } from "../../src/server/responses";
import type { AdapterEvent, OcxConfig, OcxProviderConfig } from "../../src/types";
import { harness, until } from "../fixtures/claude-session-worker";
import { acquireOwnedSpendHome } from "../helpers/owned-spend-home";
import { removeTreeAfterBackgroundWork } from "../helpers/remove-tree";

type Frame = Record<string, any>;
const setupTimeoutMs = 150;
const timeoutMs = 350;
const tools = [{ type: "function", name: "read_file", description: "Client-owned fixture read", parameters: {
  type: "object", properties: { path: { type: "string" } }, required: ["path"], additionalProperties: false,
} }];
const available = async () => ({ state: "available" as const, checkedAt: 0 });
const input = (content: string) => ({ input: [{ role: "user", content }] });

function gate() {
  let release!: () => void;
  const pending = new Promise<void>(resolve => { release = resolve; });
  return { pending, release };
}

async function bounded<T>(pending: Promise<T>, budget = 3_000): Promise<T> {
  let timer!: ReturnType<typeof setTimeout>;
  try {
    return await Promise.race([pending, new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(new Error(`HTTP fixture exceeded ${budget}ms`)), budget);
    })]);
  } finally { clearTimeout(timer); }
}

async function collect(response: Response) {
  const text = await bounded(response.text());
  const frames: Frame[] = text.split(/\r?\n/).filter(line => line.startsWith("data:"))
    .map(line => line.slice(5).trim()).filter(line => line && line !== "[DONE]").map(line => JSON.parse(line));
  return { status: response.status, contentType: response.headers.get("content-type"), frames,
    json: frames.length ? undefined : JSON.parse(text) as Frame };
}
type Reply = Awaited<ReturnType<typeof collect>>;
const externalCalls = (reply: Reply): Frame[] => reply.json?.output?.filter((item: Frame) => item.type === "function_call")
  ?? reply.frames.filter(frame => frame.type === "response.output_item.done" && frame.item.type === "function_call").map(frame => frame.item);

function completed(reply: Reply): Frame {
  expect(reply.status).toBe(200);
  if (reply.json) { expect(reply.json.status).toBe("completed"); return reply.json; }
  expect(reply.contentType).toContain("text/event-stream");
  const terminals = reply.frames.filter(frame => ["response.completed", "response.failed", "response.incomplete"].includes(frame.type));
  expect(terminals).toHaveLength(1);
  expect(terminals[0]!.type).toBe("response.completed");
  return terminals[0]!.response;
}

function failed(reply: Reply, stream: boolean, code: string) {
  if (stream) {
    expect(reply.status).toBe(200);
    expect(reply.contentType).toContain("text/event-stream");
    const terminals = reply.frames.filter(frame => ["response.completed", "response.failed", "response.incomplete"].includes(frame.type));
    expect(terminals).toHaveLength(1);
    expect(terminals[0]!.type).toBe("response.failed");
    expect(terminals[0]!.response.error.code).toBe(code);
  } else {
    expect(reply.contentType).toContain("application/json");
    expect(reply.json?.error?.code).toBe(code);
    expect(reply.status === 504 || reply.status === 200 && reply.json?.status === "failed").toBe(true);
  }
  expect(externalCalls(reply)).toEqual([]);
}

async function withHttp(run: (http: {
  h: ReturnType<typeof harness>;
  post(body: Frame, signal?: AbortSignal): Promise<Response>;
  events: AdapterEvent[];
  signals: AbortSignal[];
  turns: Promise<void>[];
}) => Promise<void>, deps: Omit<ClaudeSessionManagerDeps, "journal" | "createWorker"> = {}) {
  const previousHome = process.env.OPENCODEX_HOME;
  const home = mkdtempSync(join(tmpdir(), "ocx-claude-setup-http-"));
  process.env.OPENCODEX_HOME = home;
  const definition = getAdapterDefinition("claude-cli")! as { create: NonNullable<ReturnType<typeof getAdapterDefinition>>["create"] };
  const originalCreate = definition.create;
  const h = harness(undefined, { timeoutMs, setupTimeoutMs, reconnectGraceMs: 2_000, ...deps });
  const events: AdapterEvent[] = [];
  const signals: AbortSignal[] = [];
  const turns: Promise<void>[] = [];
  let releaseSpend = () => {};
  let server: ReturnType<typeof Bun.serve> | undefined;
  let oneShotSpawns = 0;
  try {
    clearResponseStateForTests();
    releaseSpend = acquireOwnedSpendHome();
    clearClaudeCliBinaryCache();
    setClaudeUsagePreflightForTests(available);
    definition.create = (provider: OcxProviderConfig) => {
      const adapter = createClaudeCliAdapter(provider, {
        which: () => "/claude.exe", usageAdmission: available, usageRefusal: () => {}, sessionManager: h.manager,
        spawn: () => { oneShotSpawns++; throw new Error("Scoped HTTP request entered one-shot dispatch"); },
      });
      const runTurn = adapter.runTurn!;
      adapter.runTurn = (parsed, incoming, emit) => {
        signals.push(incoming.abortSignal!);
        const pending = runTurn(parsed, incoming, event => { events.push(event); emit(event); });
        turns.push(pending);
        return pending;
      };
      return adapter;
    };
    const config = { port: 0, stallTimeoutSec: 0, defaultProvider: "claude-cli", emptyCompletionRetry: true,
      providers: { "claude-cli": { adapter: "claude-cli", baseUrl: "https://api.anthropic.com", authMode: "key",
        selectedModels: ["claude-sonnet-5-5"],
      } } } as OcxConfig;
    server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: req => {
      const admission = resolveResponsesApiAuth(req, config);
      return admission ? handleResponses(req, config, { model: "", provider: "" }, { admission }) : new Response("Unauthorized", { status: 401 });
    } });
    const url = `http://127.0.0.1:${server.port}/v1/responses`;
    await run({ h, events, signals, turns, post: (body, signal) => {
      const pending = fetch(url, { method: "POST",
        headers: { "content-type": "application/json", "thread-id": "setup-http-thread", "idempotency-key": "setup-http-operation" },
        body: JSON.stringify({ model: "claude-cli/claude-sonnet-5-5", stream: true, store: true, tools, ...body }),
        signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(3_000)]) : AbortSignal.timeout(3_000),
      });
      void pending.catch(() => {});
      return pending;
    } });
    expect(oneShotSpawns).toBe(0);
  } finally {
    try { await h.manager.close(); }
    finally {
      try {
        await server?.stop(true);
        await flushResponseState();
        clearResponseStateForTests();
        await flushConfigDirHardeningForTests();
      } finally {
        try { releaseSpend(); }
        finally {
          definition.create = originalCreate;
          setClaudeUsagePreflightForTests();
          clearClaudeCliBinaryCache();
          try { await removeTreeAfterBackgroundWork(home); }
          finally {
            if (previousHome === undefined) delete process.env.OPENCODEX_HOME;
            else process.env.OPENCODEX_HOME = previousHome;
          }
        }
      }
    }
  }
}

for (const stream of [true, false]) {
  for (const operation of ["read", "fork"] as const) {
    test(`HTTP stream=${stream}: hanging native ${operation} terminates within setup budget and retry succeeds`, async () => {
      const held = gate();
      let entered = 0;
      const files = new Map<string, SessionMessage[]>();
      const nativeIO: ClaudeNativeSessionIO = {
        async getSessionMessages(id) {
          if (operation === "read" && ++entered === 1) await held.pending;
          return structuredClone(files.get(id) ?? []);
        },
        async forkSession(id, options) {
          if (operation === "fork" && ++entered === 1) await held.pending;
          const source = files.get(id)!;
          const end = source.findIndex(message => message.uuid === options?.upToMessageId);
          const sessionId = crypto.randomUUID();
          files.set(sessionId, source.slice(0, end + 1).map(message => ({ ...message, uuid: crypto.randomUUID(), session_id: sessionId })));
          return { sessionId };
        },
      };
      await withHttp(async ({ h, post, events }) => {
        try {
          const seed = post({ ...input("seed question"), stream }).then(collect);
          await until(() => !!h.workers[0]?.sends.length);
          await h.workers[0]!.finish("seed answer");
          const first = completed(await seed);
          const native = [...h.journal.records.values()][0]!.native!;
          expect(native.checkpoints.length).toBeGreaterThan(0);
          files.set(native.sessionId, native.checkpoints.map(point => ({ type: "assistant", uuid: point.messageId,
            session_id: native.sessionId, parent_tool_use_id: null, parent_agent_id: null,
            message: { role: "assistant", content: [{ type: "text", text: "seed answer" }], stop_reason: "end_turn" },
          })));
          const body = { previous_response_id: first.id, ...input("continue after native timeout"), stream };
          failed(await collect(await post(body)), stream, "claude_session_timeout");
          expect(h.workers[0]!.stops).toBe(1);
          const started = Date.now();
          const stalled = post(body).then(collect);
          await until(() => entered === 1);
          failed(await bounded(stalled, setupTimeoutMs + 800), stream, "claude_session_setup_timeout");
          expect(Date.now() - started).toBeLessThan(setupTimeoutMs + 800);
          expect(events).toContainEqual(expect.objectContaining({ type: "error", code: "claude_session_setup_timeout", status: 504, retryable: true }));
          expect(h.workers).toHaveLength(1);

          const retry = post(body).then(collect);
          await until(() => !!h.workers[1]?.sends.length);
          await h.workers[1]!.finish("recovered answer");
          const recovered = await retry;
          expect(JSON.stringify(completed(recovered).output)).toContain("recovered answer");
          expect(externalCalls(recovered)).toEqual([]);
          held.release();
          await Bun.sleep(20);
          const replay = await collect(await post(body));
          expect(completed(replay).output.map((item: Frame) => item.content)).toEqual(completed(recovered).output.map((item: Frame) => item.content));
          expect(completed(replay).usage).toMatchObject({ input_tokens: 0, output_tokens: 0, total_tokens: 0 });
          expect(externalCalls(replay)).toEqual([]);
          expect(h.workers).toHaveLength(2);
          expect(h.workers[1]!.sends).toHaveLength(1);
          expect(h.workers[1]!.stops).toBe(0);
        } finally { held.release(); }
      }, { nativeIO });
    }, 10_000);
  }

  test(`HTTP stream=${stream}: aborted native setup releases ownership for the next retry`, async () => {
    const held = gate();
    let reads = 0;
    await withHttp(async ({ h, post, events, signals, turns }) => {
      try {
        const seed = post({ ...input("seed before cancellation"), stream }).then(collect);
        await until(() => !!h.workers[0]?.sends.length);
        await h.workers[0]!.finish("seed answer");
        const first = completed(await seed);
        const body = { previous_response_id: first.id, ...input("retry cancelled setup"), stream };
        failed(await collect(await post(body)), stream, "claude_session_timeout");
        const abort = new AbortController();
        const eventCount = events.length;
        const cancelled = post(body, abort.signal).then(collect);
        void cancelled.catch(() => {});
        await until(() => reads === 1);
        abort.abort();
        await cancelled.catch(() => undefined);
        await until(() => signals[2]?.aborted === true);
        await bounded(turns[2]!, setupTimeoutMs + 800);
        expect(events.slice(eventCount).filter(event => event.type === "error")).toEqual([]);
        const retry = post(body).then(collect);
        await until(() => !!h.workers[1]?.sends.length);
        await h.workers[1]!.finish("answer after cancelled setup");
        const reply = await retry;
        expect(JSON.stringify(completed(reply).output)).toContain("answer after cancelled setup");
        expect(externalCalls(reply)).toEqual([]);
        held.release();
        await Bun.sleep(20);
        expect(h.workers).toHaveLength(2);
        expect(h.workers[1]!.stops).toBe(0);
      } finally { held.release(); }
    }, { nativeIO: { async getSessionMessages() { if (++reads === 1) await held.pending; return []; },
      async forkSession() { throw new Error("Unexpected native fork"); },
    } });
  }, 10_000);

  test(`HTTP stream=${stream}: hanging catalog update times out without repeating the client's tool`, async () => {
    const held = gate();
    await withHttp(async ({ h, post, events }) => {
      try {
        const seed = post({ ...input("Read fixture.txt once."), stream }).then(collect);
        await until(() => !!h.workers[0]?.sends.length);
        const worker = h.workers[0]!;
        const call = { id: "toolu_setup_http_read", name: worker.options.tools[0]!.name, arguments: { path: "fixture.txt" } };
        const mcp = worker.mcp(call);
        await worker.blocks([call]);
        const first = await seed;
        const response = completed(first);
        expect(externalCalls(first)).toEqual([expect.objectContaining({ call_id: call.id, name: "read_file" })]);
        let clientExecutions = 0;
        const executeOnClient = () => { clientExecutions++; return "AUTHORITATIVE_FILE_CONTENT"; };
        let updates = 0;
        worker.updateTools = async () => { updates++; await held.pending; };
        const body = { previous_response_id: response.id, stream,
          tools: tools.map(tool => ({ ...tool, description: "Updated client-owned fixture read" })),
          input: [{ type: "function_call_output", call_id: call.id, output: executeOnClient() }],
        };
        const stalled = post(body).then(collect);
        await until(() => updates === 1);
        failed(await bounded(stalled, setupTimeoutMs + 800), stream, "claude_session_setup_timeout");
        expect(events).toContainEqual(expect.objectContaining({ type: "error", code: "claude_session_setup_timeout", status: 504, retryable: true }));
        await expect(mcp).rejects.toThrow("worker stopped");
        expect(worker.stops).toBe(1);
        const retry = post(body).then(collect);
        await until(() => !!h.workers[1]?.sends.length);
        expect(h.workers[1]!.sends[0]).toContain("AUTHORITATIVE_FILE_CONTENT");
        await h.workers[1]!.finish("Read: AUTHORITATIVE_FILE_CONTENT");
        const recovered = await retry;
        expect(JSON.stringify(completed(recovered).output)).toContain("Read: AUTHORITATIVE_FILE_CONTENT");
        expect(externalCalls(recovered)).toEqual([]);
        held.release();
        await Bun.sleep(20);
        const replay = await collect(await post(body));
        expect(completed(replay).output.map((item: Frame) => item.content)).toEqual(completed(recovered).output.map((item: Frame) => item.content));
        expect(completed(replay).usage).toMatchObject({ input_tokens: 0, output_tokens: 0, total_tokens: 0 });
        expect(externalCalls(replay)).toEqual([]);
        expect(clientExecutions).toBe(1);
        expect(h.workers).toHaveLength(2);
        expect(h.workers[1]!.sends).toHaveLength(1);
      } finally { held.release(); }
    });
  }, 10_000);

  test(`HTTP stream=${stream}: queued follow-up ends explicitly while retaining the original MCP wait`, async () => {
    await withHttp(async ({ h, post, events }) => {
      const prompt = "Read fixture.txt once before applying more instructions.";
      const seed = post({ ...input(prompt), stream }).then(collect);
      await until(() => !!h.workers[0]?.sends.length);
      const worker = h.workers[0]!;
      const call = { id: "toolu_delayed_result_http", name: worker.options.tools[0]!.name, arguments: { path: "fixture.txt" } };
      const mcp = worker.mcp(call);
      let mcpSettled = false;
      void mcp.then(() => { mcpSettled = true; }, () => { mcpSettled = true; });
      await worker.blocks([call]);
      const first = await seed;
      const response = completed(first);
      expect(externalCalls(first)).toEqual([expect.objectContaining({ call_id: call.id, name: "read_file" })]);

      const instruction = "B: include the saved instruction when the original tool returns.";
      const eventCount = events.length;
      const started = Date.now();
      const queued = post({ stream, input: [{ role: "user", content: prompt }, ...response.output,
        { role: "user", content: instruction }],
      }).then(collect);
      await until(() => [...h.journal.records.values()].some(snapshot => snapshot.instructions?.some(entry => entry.phase === "queued")));
      const reply = await bounded(queued, setupTimeoutMs + 800);
      expect(Date.now() - started).toBeGreaterThanOrEqual(setupTimeoutMs);
      expect(Date.now() - started).toBeLessThan(setupTimeoutMs + 800);
      expect(reply.status).toBe(200);
      if (stream) {
        expect(reply.contentType).toContain("text/event-stream");
        const terminals = reply.frames.filter(frame => ["response.completed", "response.failed", "response.incomplete"].includes(frame.type));
        expect(terminals).toHaveLength(1);
        expect(terminals[0]!.type).toBe("response.failed");
      } else expect(reply.contentType).toContain("application/json");
      const failure = reply.json ?? reply.frames.find(frame => frame.type === "response.failed")!.response;
      expect(failure).toMatchObject({ status: "failed", retryable: false, error: { code: "claude_tool_results_pending" } });
      expect(failure.error.message).toContain("instructions were saved");
      expect(failure.error.message).toContain("original tool results");
      expect(events.slice(eventCount).filter(event => event.type === "error")).toEqual([
        expect.objectContaining({ code: "claude_tool_results_pending", status: 409, retryable: false }),
      ]);
      expect(externalCalls(reply)).toEqual([]);
      expect(mcpSettled).toBe(false);
      expect(worker.stops).toBe(0);
      expect(worker.interrupts).toBe(0);
      expect(worker.sends).toHaveLength(1);
      expect([...h.journal.records.values()][0]!.instructions).toEqual([
        expect.objectContaining({ phase: "queued", texts: [instruction] }),
      ]);

      let clientExecutions = 0;
      const executeOnClient = () => { clientExecutions++; return "ORIGINAL_CLIENT_RESULT"; };
      const continuation = { previous_response_id: response.id, stream, input: [
        { type: "function_call_output", call_id: call.id, output: executeOnClient() },
      ] };
      const resumed = post(continuation).then(collect);
      expect(await bounded(mcp)).toEqual({ content: [{ type: "text", text: "ORIGINAL_CLIENT_RESULT" }], isError: false });
      expect(worker.sends[1]).toBe(instruction);
      expect(worker.sendOptions[1]).toMatchObject({ shouldQuery: false });
      await worker.finish("C: ORIGINAL_CLIENT_RESULT; saved B instruction applied.");
      const recovered = await resumed;
      expect(JSON.stringify(completed(recovered).output)).toContain("saved B instruction applied");
      expect(externalCalls(recovered)).toEqual([]);
      const replay = await collect(await post(continuation));
      expect(completed(replay).output.map((item: Frame) => item.content)).toEqual(completed(recovered).output.map((item: Frame) => item.content));
      expect(externalCalls(replay)).toEqual([]);
      expect(clientExecutions).toBe(1);
      expect(h.workers).toHaveLength(1);
      expect(worker.sends).toHaveLength(2);
      expect(worker.stops).toBe(0);
      expect(worker.interrupts).toBe(0);
      expect([...h.journal.records.values()][0]!.instructions).toEqual([
        expect.objectContaining({ phase: "completed", texts: [instruction] }),
      ]);
    });
  }, 10_000);

  test(`HTTP stream=${stream}: genuine SDK progress keeps a response alive beyond the inactivity budget`, async () => {
    await withHttp(async ({ h, post }) => {
      const reply = post({ ...input("Stream a long answer"), stream }).then(collect);
      await until(() => !!h.workers[0]?.sends.length);
      const worker = h.workers[0]!;
      await worker.begin();
      await worker.raw({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
      const started = Date.now();
      for (let i = 0; i < 5; i++) {
        await Bun.sleep(100);
        await worker.raw({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "progress " } });
      }
      expect(Date.now() - started).toBeGreaterThan(timeoutMs);
      await worker.raw({ type: "content_block_stop", index: 0 });
      await worker.finish("long answer complete");
      expect(JSON.stringify(completed(await reply).output)).toContain("long answer complete");
      expect(worker.stops).toBe(0);
      expect(h.workers).toHaveLength(1);
    });
  }, 10_000);
}
