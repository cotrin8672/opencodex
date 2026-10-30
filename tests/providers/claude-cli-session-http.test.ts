import { expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createClaudeCliAdapter } from "../../src/adapters/claude-cli/adapter";
import { clearClaudeCliBinaryCache } from "../../src/adapters/claude-cli/profiles";
import { setClaudeUsagePreflightForTests } from "../../src/adapters/claude-cli/usage-admission";
import { getAdapterDefinition } from "../../src/adapters/registry";
import { flushConfigDirHardeningForTests } from "../../src/config/paths";
import { clearResponseStateForTests, flushResponseState } from "../../src/responses/state";
import { handleResponses } from "../../src/server/responses";
import { resolveResponsesApiAuth } from "../../src/server/auth-cors";
import type { OcxConfig, OcxProviderConfig } from "../../src/types";
import { harness, until } from "../fixtures/claude-session-worker";
import { acquireOwnedSpendHome } from "../helpers/owned-spend-home";
import { removeTreeAfterBackgroundWork } from "../helpers/remove-tree";

type Frame = Record<string, any>;
const tools = [{ type: "function", name: "read_file", description: "Client-owned fixture read", parameters: {
  type: "object", properties: { path: { type: "string" } }, required: ["path"], additionalProperties: false,
} }];
const available = async () => ({ state: "available" as const, checkedAt: 0 });

async function bounded<T>(pending: Promise<T>): Promise<T> {
  let timer!: ReturnType<typeof setTimeout>;
  try {
    return await Promise.race([pending, new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(new Error("HTTP session test timed out after 5s")), 5_000);
    })]);
  } finally { clearTimeout(timer); }
}

async function collect(response: Response) {
  const text = await bounded(response.text());
  const frames: Frame[] = text.split(/\r?\n/).filter(line => line.startsWith("data:"))
    .map(line => line.slice(5).trim()).filter(line => line && line !== "[DONE]").map(line => JSON.parse(line));
  return { status: response.status, frames, json: frames.length ? undefined : JSON.parse(text) as Frame };
}
const completed = (reply: Awaited<ReturnType<typeof collect>>): Frame => {
  expect(reply.status).toBe(200);
  expect(reply.frames.some(frame => frame.type === "response.failed")).toBe(false);
  const terminal = reply.frames.filter(frame => frame.type === "response.completed");
  expect(terminal).toHaveLength(1);
  return terminal[0]!.response;
};
const externalCalls = (reply: Awaited<ReturnType<typeof collect>>) => reply.frames
  .filter(frame => frame.type === "response.output_item.done" && frame.item.type === "function_call").map(frame => frame.item);

async function withHttp(run: (http: {
  h: ReturnType<typeof harness>;
  post(body: Frame, signal?: AbortSignal): Promise<Response>;
  signals: AbortSignal[];
}) => Promise<void>) {
  const previousHome = process.env.OPENCODEX_HOME;
  const home = mkdtempSync(join(tmpdir(), "ocx-claude-session-http-"));
  process.env.OPENCODEX_HOME = home;
  const definition = getAdapterDefinition("claude-cli")! as { create: NonNullable<ReturnType<typeof getAdapterDefinition>>["create"] };
  const originalCreate = definition.create;
  const h = harness(undefined, { timeoutMs: 5_000, reconnectGraceMs: 2_000 });
  const signals: AbortSignal[] = [];
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
        spawn: () => { oneShotSpawns++; throw new Error("A scoped HTTP request entered one-shot dispatch"); },
      });
      const runTurn = adapter.runTurn!;
      adapter.runTurn = (parsed, incoming, emit) => {
        signals.push(incoming.abortSignal!);
        return runTurn(parsed, incoming, emit);
      };
      return adapter;
    };
    const config = { port: 0, defaultProvider: "claude-cli", emptyCompletionRetry: true, providers: { "claude-cli": {
      adapter: "claude-cli", baseUrl: "https://api.anthropic.com", authMode: "key", selectedModels: ["claude-sonnet-5-5"],
    } } } as OcxConfig;
    server = Bun.serve({ hostname: "127.0.0.1", port: 0,
      fetch: req => {
        const admission = resolveResponsesApiAuth(req, config);
        return admission ? handleResponses(req, config, { model: "", provider: "" }, { admission }) : new Response("Unauthorized", { status: 401 });
      },
    });
    const url = `http://127.0.0.1:${server.port}/v1/responses`;
    await run({ h, signals, post: (body, signal) => fetch(url, {
      method: "POST", headers: { "content-type": "application/json", "thread-id": "own-http-thread", "idempotency-key": "offline-http-operation" },
      body: JSON.stringify({ model: "claude-cli/claude-sonnet-5-5", stream: true, store: true, tools, ...body }),
      signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(5_000)]) : AbortSignal.timeout(5_000),
    }) });
    expect(oneShotSpawns).toBe(0);
  } finally {
    h.manager.close();
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

test("scoped HTTP Responses reuses the worker for client results and refuses duplicate external execution", async () => {
  await withHttp(async ({ h, post }) => {
    const original = { input: [{ role: "user", content: "Read fixture.txt with read_file." }] };
    const firstRequest = post(original);
    void firstRequest.catch(() => {});
    await until(() => !!h.workers[0]?.sends.length);
    const worker = h.workers[0]!;
    expect(worker.options.binary).toBe("/claude.exe");
    const firstResponse = await bounded(firstRequest);
    expect(firstResponse.headers.get("content-type")).toContain("text/event-stream");
    const firstBody = collect(firstResponse);
    void firstBody.catch(() => {});
    const call = { id: "toolu_http_read", name: worker.options.tools[0]!.name, arguments: { path: "fixture.txt" } };
    const mcpResult = worker.mcp(call);
    await worker.blocks([call]);
    const first = await firstBody;
    const firstCompleted = completed(first);
    expect(externalCalls(first)).toEqual([expect.objectContaining({
      name: "read_file", call_id: call.id, arguments: JSON.stringify(call.arguments),
    })]);
    let clientExecutions = 0;
    const executeOnClient = () => { clientExecutions++; return "LOCAL_FILE_CONTENT_42"; };
    const continuation = { previous_response_id: firstCompleted.id, input: [{
      type: "function_call_output", call_id: externalCalls(first)[0]!.call_id, output: executeOnClient(),
    }] };
    const secondRequest = post(continuation);
    void secondRequest.catch(() => {});
    expect(await bounded(mcpResult)).toEqual({ content: [{ type: "text", text: "LOCAL_FILE_CONTENT_42" }], isError: false });
    expect(h.workers).toHaveLength(1);
    expect(worker.sends).toHaveLength(1);
    await worker.finish("Read: LOCAL_FILE_CONTENT_42");
    const second = await collect(await bounded(secondRequest));
    const final = completed(second);
    expect(JSON.stringify(final.output)).toContain("Read: LOCAL_FILE_CONTENT_42");
    expect(externalCalls(second)).toEqual([]);

    const duplicate = await collect(await bounded(post(original)));
    const failure = duplicate.frames.find(frame => frame.type === "response.failed");
    expect(duplicate.status === 409 || failure !== undefined).toBe(true);
    expect(duplicate.json?.error?.code ?? failure?.response.error.code).toBe("claude_tool_delivery_uncertain");
    expect(externalCalls(duplicate)).toEqual([]);

    const replay = await collect(await bounded(post(continuation)));
    const replayedFinal = completed(replay);
    expect(replayedFinal.output.map((item: Frame) => item.content)).toEqual(final.output.map((item: Frame) => item.content));
    expect(replayedFinal.usage).toMatchObject({ input_tokens: 0, output_tokens: 0, total_tokens: 0 });
    expect(externalCalls(replay)).toEqual([]);
    expect(h.workers).toHaveLength(1);
    expect(worker.sends).toHaveLength(1);
    expect(worker.stops).toBe(0);
    expect(clientExecutions).toBe(1);
  });
}, 20_000);

test("HTTP follow-up survives a cancelled consumer, missing replay text, and reshaped historical results", async () => {
  await withHttp(async ({ h, post }) => {
    const prompt = "Read fixture.txt once.";
    const initial = post({ input: [{ role: "user", content: prompt }] });
    await until(() => !!h.workers[0]?.sends.length);
    const worker = h.workers[0]!;
    const firstBody = collect(await bounded(initial));
    const action = { id: "toolu_http_queued", name: worker.options.tools[0]!.name, arguments: { path: "fixture.txt" } };
    const pending = worker.mcp(action);
    await worker.blocks([action]);
    const first = completed(await firstBody);
    const instruction = "Keep the queued instruction when the tool returns.";
    const abort = new AbortController();
    const queuedResponse = await bounded(post({ input: [
      { role: "user", content: prompt }, ...first.output, { role: "user", content: instruction },
    ] }, abort.signal));
    const queuedBody = collect(queuedResponse);
    void queuedBody.catch(() => {});
    await until(() => [...h.journal.records.values()].some(snapshot => snapshot.instructions?.some(entry => entry.phase === "queued")));
    expect(worker.sends).toHaveLength(1);
    abort.abort();
    await queuedBody.catch(() => undefined);
    const continuation = post({ previous_response_id: first.id, input: [{
      type: "function_call_output", call_id: action.id, output: "AUTHORITATIVE_FULL_OUTPUT",
    }] });
    expect((await bounded(pending)).content[0]!.text).toBe("AUTHORITATIVE_FULL_OUTPUT");
    expect(worker.sends[1]).toBe(instruction);
    expect(worker.sendOptions[1]).toMatchObject({ shouldQuery: false });
    await worker.finish("Queued instruction applied.");
    const resumed = completed(await collect(await bounded(continuation)));
    expect(JSON.stringify(resumed.output)).toContain("Queued instruction applied.");
    const nextRequest = post({ previous_response_id: resumed.id, input: [{ role: "user", content: "next question" }] });
    await until(() => worker.sends.length === 3 || h.workers.length > 1);
    expect(h.workers).toHaveLength(1);
    expect(worker.sends[2]).toBe("next question");
    const nextBody = collect(await bounded(nextRequest));
    const stale = await collect(await bounded(post({ input: [
      { role: "user", content: prompt }, ...first.output,
      { type: "function_call_output", call_id: action.id, output: "[shortened historical output]" },
    ] })));
    expect(JSON.stringify(completed(stale).output)).toContain("Queued instruction applied.");
    expect(worker.stops).toBe(0);
    await worker.finish("next answer");
    expect(JSON.stringify(completed(await nextBody).output)).toContain("next answer");
    expect(h.workers).toHaveLength(1);
    expect(worker.interrupts).toBe(0);
  });
}, 15_000);

test("cancelling open HTTP SSE before tools reattaches the same inference and delivers each call once", async () => {
  await withHttp(async ({ h, post, signals }) => {
    const input = { input: [{ role: "user", content: "Read fixture.txt with read_file." }] };
    const abort = new AbortController();
    const firstRequest = post(input, abort.signal);
    void firstRequest.catch(() => {});
    await until(() => !!h.workers[0]?.sends.length);
    const worker = h.workers[0]!;
    const firstResponse = await bounded(firstRequest);
    expect(firstResponse.headers.get("content-type")).toContain("text/event-stream");
    const reader = firstResponse.body!.getReader();
    await worker.begin();
    await worker.raw({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
    await worker.raw({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Working on it." } });
    await worker.raw({ type: "content_block_stop", index: 0 });
    let received = "";
    const decoder = new TextDecoder();
    for (let i = 0; i < 20 && !received.includes("response.output_text.delta"); i++) {
      const chunk = await bounded(reader.read());
      if (chunk.done) break;
      received += decoder.decode(chunk.value, { stream: true });
    }
    expect(received).toContain("response.output_text.delta");
    expect(received).not.toContain('"type":"function_call"');
    abort.abort();
    await reader.cancel().catch(() => {});
    await until(() => signals[0]?.aborted === true);
    expect(worker.stops).toBe(0);

    const reconnect = await bounded(post(input));
    await until(() => signals.length === 2);
    const reconnectedBody = collect(reconnect);
    void reconnectedBody.catch(() => {});
    const call = { id: "toolu_after_reconnect", name: worker.options.tools[0]!.name, arguments: { path: "fixture.txt" } };
    worker.mcp(call);
    await worker.blocks([call]);
    const recovered = await reconnectedBody;
    completed(recovered);
    expect(externalCalls(recovered)).toEqual([expect.objectContaining({ name: "read_file", call_id: call.id })]);
    expect(recovered.frames.filter(frame => frame.type === "response.output_text.delta").map(frame => frame.delta).join("")).toBe("Working on it.");
    expect(h.workers).toHaveLength(1);
    expect(worker.sends).toHaveLength(1);
    expect(worker.stops).toBe(0);
  });
}, 20_000);
