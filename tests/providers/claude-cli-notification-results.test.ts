import { expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createClaudeCliAdapter } from "../../src/adapters/claude-cli/adapter";
import { clearClaudeCliBinaryCache } from "../../src/adapters/claude-cli/profiles";
import { setClaudeUsagePreflightForTests } from "../../src/adapters/claude-cli/usage-admission";
import { getAdapterDefinition } from "../../src/adapters/registry";
import { claudeDigest } from "../../src/adapters/claude-cli/session-context";
import type { ClaudeSessionSnapshot } from "../../src/adapters/claude-cli/session-journal";
import { prepareClaudeToolResults } from "../../src/adapters/claude-cli/session-recovery";
import { stableClaudeToolBridge } from "../../src/adapters/claude-cli/stable-replay";
import { buildCodeBuddyToolBridge } from "../../src/adapters/codebuddy/tool-bridge";
import { parseRequest } from "../../src/responses/parser";
import { clearResponseStateForTests, flushResponseState } from "../../src/responses/state";
import { flushConfigDirHardeningForTests } from "../../src/config/paths";
import { handleResponses } from "../../src/server/responses";
import { resolveResponsesApiAuth } from "../../src/server/auth-cors";
import type { OcxConfig, OcxProviderConfig } from "../../src/types";
import { harness, turn, until } from "../fixtures/claude-session-worker";
import { acquireOwnedSpendHome } from "../helpers/owned-spend-home";
import { removeTreeAfterBackgroundWork } from "../helpers/remove-tree";

const callId = "toolu_notification_fixture";
const code = 'notify("migration running"); text("migration complete");';
const final = "Script completed\nWall time: 344.7 seconds\nOutput:\nstore.sqlite=751054848";
const notifications = ["MATLAB migration running: store.sqlite=724348928", "MATLAB migration still running: store.sqlite=739246080"];
const tools = [{ type: "custom", name: "exec", description: "Offline fixture", format: { type: "text" } }];
const origin = { type: "custom_tool_call", call_id: callId, name: "exec", input: code };
const outputs = () => [
  { type: "custom_tool_call_output", call_id: callId, output: [{ type: "input_text", text: final }] },
  ...notifications.map(output => ({ type: "custom_tool_call_output", call_id: callId, output })),
];
function parsed(input: unknown[]) {
  return parseRequest({ model: "offline-model", stream: true, tools, input });
}
function messages(items = outputs()) {
  return parsed([{ role: "user", content: "Migrate the fixture once." }, origin, ...items]).context.messages;
}
function snapshot(): ClaudeSessionSnapshot {
  return { version: 1, history: [], turns: [{
    inputHash: claudeDigest("notification fixture"), phase: "tools", deliveryStarted: true,
    events: [], updatedAt: 123, calls: [{ id: callId, name: "exec", arguments: JSON.stringify({ input: code }) }],
  }] };
}
function expectAllOutput(content: string) {
  for (const text of [final, ...notifications]) {
    expect(content).toContain(text);
    expect(content.split(text)).toHaveLength(2);
  }
}

test("Responses custom tool final content plus two notify outputs becomes one durable result without losing text", () => {
  const saved = snapshot();
  const input = messages();
  expect(input.filter(message => message.role === "toolResult")).toHaveLength(3);
  expect(input.filter(message => message.role === "toolResult").every(message => message.isCustomToolOutput === true)).toBe(true);
  const before = structuredClone({ saved, input });
  const prepared = prepareClaudeToolResults(saved, input);
  const result = prepared.snapshot.turns[0]!.calls[0]!.result!;
  expectAllOutput(result.content);
  expect(result.isError).toBe(false);
  expect(prepared.pendingTurns).toHaveLength(1);
  expect(prepared.snapshot.turns[0]!.deliveryStarted).toBe(true);
  expect(prepared.changed).toBe(true);
  expect(prepared.messages.filter(message => message.role === "toolResult")).toHaveLength(1);
  expect(result.content).toBe([final, ...notifications].join("\n\n"));
  expect({ saved, input }).toEqual(before);

  const retried = prepareClaudeToolResults(prepared.snapshot, structuredClone(input));
  expect(retried.snapshot).toEqual(prepared.snapshot);
  expect(retried.messages).toEqual(prepared.messages);
  expect(retried.changed).toBe(false);
});

test("accepted pending notification results still reject changed text and changed error bits", () => {
  const accepted = prepareClaudeToolResults(snapshot(), messages()).snapshot;
  const before = structuredClone(accepted);
  const changed = outputs();
  changed[0] = { ...changed[0]!, output: [{ type: "input_text", text: "a different final result" }] };
  expect(() => prepareClaudeToolResults(accepted, messages(changed)))
    .toThrow(expect.objectContaining({ code: "claude_tool_result_conflict" }));
  const errorMessages = messages();
  for (const message of errorMessages) if (message.role === "toolResult") message.isError = true;
  expect(() => prepareClaudeToolResults(accepted, errorMessages))
    .toThrow(expect.objectContaining({ code: "claude_tool_result_conflict" }));
  expect(accepted).toEqual(before);
});

test("completed custom output groups normalize differing historical error bits to the saved result exactly once", () => {
  const saved = prepareClaudeToolResults(snapshot(), messages()).snapshot;
  saved.turns[0]!.phase = "completed";
  const authoritative = saved.turns[0]!.calls[0]!.result!;
  for (const firstIsError of [false, true]) {
    const input = messages();
    const results = input.filter(message => message.role === "toolResult");
    results[0]!.isError = firstIsError;
    results[1]!.isError = !firstIsError;
    const before = structuredClone({ saved, input });
    const prepared = prepareClaudeToolResults(saved, input);
    const normalized = prepared.messages.filter(message => message.role === "toolResult");
    expect(normalized).toHaveLength(1);
    expect(normalized[0]).toMatchObject({ content: authoritative.content, isError: authoritative.isError });
    expectAllOutput(normalized[0]!.content as string);
    expect(prepared.snapshot).toEqual(saved);
    expect(prepared.changed).toBe(false);
    expect({ saved, input }).toEqual(before);
    expect(prepareClaudeToolResults(saved, prepared.messages).messages).toEqual(prepared.messages);
  }
});

test.each(["name", "arguments", "namespace"] as const)("notification aggregation still rejects changed custom call %s", field => {
  const input = messages();
  for (const message of input) if (message.role === "assistant") {
    for (const part of message.content) if (part.type === "toolCall") {
      if (field === "name") part.name = "different";
      else if (field === "namespace") part.namespace = "different";
      else part.arguments = { input: "different" };
    }
  }
  expect(() => prepareClaudeToolResults(snapshot(), input))
    .toThrow(expect.objectContaining({ code: "claude_tool_id_conflict" }));
});

test("ordinary function outputs with competing contents remain a conflict", () => {
  const input = parseRequest({ model: "offline-model", input: [
    { role: "user", content: "Read once." },
    { type: "function_call", call_id: callId, name: "exec", arguments: JSON.stringify({ input: code }) },
    { type: "function_call_output", call_id: callId, output: "first" },
    { type: "function_call_output", call_id: callId, output: "different" },
  ] }).context.messages;
  expect(() => prepareClaudeToolResults(snapshot(), input))
    .toThrow(expect.objectContaining({ code: "claude_tool_result_conflict" }));
});

test("custom fragments deduplicate identical notifications but preserve conflicting error and mixed-output guards", () => {
  const duplicate = outputs();
  duplicate.push({ ...duplicate[1]! });
  const prepared = prepareClaudeToolResults(snapshot(), messages(duplicate));
  expectAllOutput(prepared.snapshot.turns[0]!.calls[0]!.result!.content);
  for (const field of ["isError", "isCustomToolOutput"] as const) {
    const input = messages();
    const result = input.find(message => message.role === "toolResult")!;
    if (result.role !== "toolResult") throw new Error("Missing fixture result");
    if (field === "isError") result.isError = true;
    else delete result.isCustomToolOutput;
    expect(() => prepareClaudeToolResults(snapshot(), input))
      .toThrow(expect.objectContaining({ code: "claude_tool_result_conflict" }));
  }
});

test("native MCP consumes the combined custom output once and warm/cold retries replay without another tool execution", async () => {
  const h = harness();
  const initialParsed = parsed([{ role: "user", content: "Migrate the fixture once." }]);
  const input = { ...turn(), parsed: initialParsed, bridge: stableClaudeToolBridge(buildCodeBuddyToolBridge(initialParsed)) };
  const first = await h.start(input);
  const action = { id: callId, name: input.bridge.tools[0]!.name, arguments: { input: code } };
  let mcpCompletions = 0;
  const pending = first.worker.mcp(action).then(result => { mcpCompletions++; return result; });
  void pending.catch(() => {});
  await first.worker.blocks([action]);
  await first.done;
  expect(first.events.filter(event => event.type === "tool_call_start")).toHaveLength(1);

  const continuation = { ...input, parsed: parsed([
    { role: "user", content: "Migrate the fixture once." }, origin, ...outputs(),
  ]) };
  const resumed = h.run(continuation);
  const mcpResult = await Promise.race([pending, resumed.done.then(() => { throw new Error("Continuation ended before accepting the tool output"); })]);
  expectAllOutput(mcpResult.content[0]!.text);
  expect(mcpResult.isError).toBe(false);
  expect(mcpCompletions).toBe(1);
  expect(h.workers).toHaveLength(1);
  expect(first.worker.sends).toHaveLength(1);
  await first.worker.finish("Migration complete; no rerun.");
  await resumed.done;
  expect(resumed.events.some(event => event.type === "error" || event.type === "tool_call_start")).toBe(false);

  const retry = h.run(continuation);
  await retry.done;
  expect(retry.events).toContainEqual({ type: "text_delta", text: "Migration complete; no rerun." });
  expect(retry.events.some(event => event.type === "tool_call_start")).toBe(false);
  expect(first.worker.sends).toHaveLength(1);
  await h.manager.close();

  const cold = harness(h.journal);
  const replay = cold.run(continuation);
  await replay.done;
  expect(replay.events).toContainEqual({ type: "text_delta", text: "Migration complete; no rerun." });
  expect(replay.events.some(event => event.type === "error" || event.type === "tool_call_start")).toBe(false);
  expect(cold.workers).toHaveLength(0);
  expect(mcpCompletions).toBe(1);
}, 10_000);

test("cold recovery resumes an already saved custom group as history without emitting the original call again", async () => {
  const h = harness();
  const continuationParsed = parsed([{ role: "user", content: "Migrate the fixture once." }, origin, ...outputs()]);
  const input = { ...turn(), parsed: continuationParsed, bridge: stableClaudeToolBridge(buildCodeBuddyToolBridge(continuationParsed)) };
  const saved = prepareClaudeToolResults(snapshot(), continuationParsed.context.messages).snapshot;
  h.journal.save(input.scope, saved);
  const resumed = await h.start(input);
  for (const fragment of [final, ...notifications]) {
    expect(resumed.worker.sends[0]).toContain(fragment);
    expect(resumed.worker.sends[0]!.split(fragment)).toHaveLength(2);
  }
  expect(h.journal.load(input.scope)!.turns[0]!.phase).toBe("completed");
  expect(h.journal.load(input.scope)!.turns[0]!.calls[0]!.result).toEqual(saved.turns[0]!.calls[0]!.result);
  await resumed.worker.finish("Recovered saved migration output.");
  await resumed.done;
  expect(resumed.events.some(event => event.type === "error" || event.type === "tool_call_start")).toBe(false);
  expect(h.workers).toHaveLength(1);
  expect(resumed.worker.sends).toHaveLength(1);
}, 10_000);

test("HTTP Responses preserves custom notification provenance through MCP continuation and cold replay", async () => {
  const previousHome = process.env.OPENCODEX_HOME;
  const home = mkdtempSync(join(tmpdir(), "ocx-claude-notifications-"));
  process.env.OPENCODEX_HOME = home;
  const definition = getAdapterDefinition("claude-cli")! as { create: NonNullable<ReturnType<typeof getAdapterDefinition>>["create"] };
  const originalCreate = definition.create;
  const available = async () => ({ state: "available" as const, checkedAt: 0 });
  const h = harness();
  let active = h;
  let releaseSpend = () => {};
  let server: ReturnType<typeof Bun.serve> | undefined;
  try {
    clearResponseStateForTests();
    releaseSpend = acquireOwnedSpendHome();
    clearClaudeCliBinaryCache();
    setClaudeUsagePreflightForTests(available);
    definition.create = (provider: OcxProviderConfig) => createClaudeCliAdapter(provider, {
      which: () => "/claude.exe", usageAdmission: available, usageRefusal: () => {}, sessionManager: active.manager,
      usageObserver: () => ({ observe() {}, finish() {} }),
      spawn: () => { throw new Error("HTTP fixture must never launch a real CLI"); },
    });
    const config = { port: 0, defaultProvider: "claude-cli", emptyCompletionRetry: true, providers: { "claude-cli": {
      adapter: "claude-cli", baseUrl: "https://api.anthropic.com", authMode: "key", selectedModels: ["offline-model"],
    } } } as OcxConfig;
    server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: request => {
      const admission = resolveResponsesApiAuth(request, config);
      return admission ? handleResponses(request, config, { model: "", provider: "" }, { admission }) : new Response("Unauthorized", { status: 401 });
    } });
    const post = async (input: unknown[]) => {
      const response = await fetch(`http://127.0.0.1:${server!.port}/v1/responses`, {
        method: "POST", headers: { "content-type": "application/json", "thread-id": "offline-notifications", "idempotency-key": "offline-notification-operation" },
        body: JSON.stringify({ model: "claude-cli/offline-model", stream: true, store: true, tools, input }),
        signal: AbortSignal.timeout(5_000),
      });
      const text = await response.text();
      const frames = text.split(/\r?\n/).filter(line => line.startsWith("data:"))
        .map(line => line.slice(5).trim()).filter(line => line && line !== "[DONE]").map(line => JSON.parse(line));
      expect(response.status).toBe(200);
      expect(frames.some(frame => frame.type === "response.failed")).toBe(false);
      expect(frames.filter(frame => frame.type === "response.completed")).toHaveLength(1);
      return frames;
    };
    const first = post([{ role: "user", content: "Migrate the fixture once." }]);
    void first.catch(() => {});
    await until(() => !!h.workers[0]?.sends.length);
    const worker = h.workers[0]!;
    const action = { id: callId, name: worker.options.tools[0]!.name, arguments: { input: code } };
    let completions = 0;
    const pending = worker.mcp(action).then(result => { completions++; return result; });
    void pending.catch(() => {});
    await worker.blocks([action]);
    const firstFrames = await first;
    expect(firstFrames.filter(frame => frame.type === "response.output_item.done" && frame.item.type === "custom_tool_call")).toHaveLength(1);

    const input = [{ role: "user", content: "Migrate the fixture once." }, origin, ...outputs()];
    const resumed = post(input);
    void resumed.catch(() => {});
    const result = await Promise.race([pending, resumed.then(() => { throw new Error("HTTP continuation ended before the MCP result"); })]);
    expectAllOutput(result.content[0]!.text);
    expect(completions).toBe(1);
    await worker.finish("HTTP migration complete; no rerun.");
    const resumedFrames = await resumed;
    expect(JSON.stringify(resumedFrames)).toContain("HTTP migration complete; no rerun.");
    expect(resumedFrames.some(frame => frame.type === "response.output_item.done" && frame.item.type === "custom_tool_call")).toBe(false);
    expect(h.workers).toHaveLength(1);
    expect(worker.sends).toHaveLength(1);

    await h.manager.close();
    active = harness(h.journal);
    const replay = await post(input);
    expect(JSON.stringify(replay)).toContain("HTTP migration complete; no rerun.");
    expect(replay.some(frame => frame.type === "response.output_item.done" && frame.item.type === "custom_tool_call")).toBe(false);
    expect(active.workers).toHaveLength(0);
    expect(completions).toBe(1);
  } finally {
    await active.manager.close();
    await h.manager.close();
    try {
      await server?.stop(true);
      await flushResponseState();
      clearResponseStateForTests();
      await flushConfigDirHardeningForTests();
    } finally {
      releaseSpend();
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
}, 15_000);
