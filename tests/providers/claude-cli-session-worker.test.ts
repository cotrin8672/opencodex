import { query as officialQuery, type Options, type Query, type SDKMessage, type SDKUserMessage } from "@anthropic-ai/claude-agent-sdk";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { CallToolResultSchema, ToolListChangedNotificationSchema } from "@modelcontextprotocol/sdk/types.js";
import { afterEach, describe, expect, test } from "bun:test";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { EventEmitter } from "node:events";
import { PassThrough, Writable } from "node:stream";
import { createClaudeSessionWorker, type ClaudeSessionWorker, type ClaudeWorkerOptions } from "../../src/adapters/claude-cli/session-worker";
import type { CodingAgentMcpTool } from "../../src/adapters/coding-agent/mcp-server";
import { fixturePath } from "../helpers/repo-root";

const tool: CodingAgentMcpTool = {
  name: "run",
  description: "Client-owned operation",
  inputSchema: {
    type: "object",
    properties: { value: { anyOf: [{ type: "string" }, { type: "integer" }] } },
    required: ["value"], additionalProperties: false,
  },
};
const active: ClaudeSessionWorker[] = [];
afterEach(async () => { await Promise.all(active.splice(0).map(worker => worker.stop())); });

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(done => { resolve = done; });
  return { promise, resolve };
}

async function until(check: () => boolean): Promise<void> {
  for (let i = 0; i < 800; i++) {
    if (check()) return;
    await Bun.sleep(5);
  }
  expect(check()).toBe(true);
}

async function harness(overrides: Partial<ClaudeWorkerOptions> = {}, onInterrupt = async () => {}) {
  let options!: Options;
  let input!: AsyncIterator<SDKUserMessage>;
  let controller!: ReadableStreamDefaultController<SDKMessage>;
  let outputClosed = false;
  let queryCalls = 0;
  let interruptCalls = 0;
  let closeCalls = 0;
  const client = new Client({ name: "offline-worker-test", version: "1.0.0" });
  const errors: Error[] = [];
  const messages: SDKMessage[] = [];
  let ready!: Promise<void>;
  const output = new ReadableStream<SDKMessage>({ start(value) { controller = value; } });
  const fakeQuery: typeof officialQuery = params => {
    queryCalls++;
    options = params.options!;
    input = (params.prompt as AsyncIterable<SDKUserMessage>)[Symbol.asyncIterator]();
    const config = options.mcpServers!.opencodex;
    if (config.type !== "sdk") throw new Error("Expected in-process MCP server");
    const [serverSide, clientSide] = InMemoryTransport.createLinkedPair();
    ready = Promise.all([config.instance.connect(serverSide), client.connect(clientSide)]).then(() => {});
    const stream = (async function* () {
      await ready;
      for await (const message of output) yield message;
    })();
    return Object.assign(stream, {
      async interrupt() { interruptCalls++; await onInterrupt(); },
      close() {
        closeCalls++;
        if (!outputClosed) { outputClosed = true; controller.close(); }
        void client.close();
      },
    }) as Query;
  };
  const worker = await createClaudeSessionWorker({
    binary: process.execPath,
    model: "offline-model",
    systemPrompt: "Private system prompt: keep off argv",
    env: { PATH: "scoped-path", USERPROFILE: "scoped-profile" },
    effort: "high",
    tools: [tool],
    onMessage: message => { messages.push(message); },
    onToolCall: async () => ({ content: [{ type: "text", text: "completed" }] }),
    onError: error => { errors.push(error); },
    query: fakeQuery,
    ...overrides,
  });
  active.push(worker);
  await ready;
  return {
    worker, options, input, client, errors, messages,
    queryCalls: () => queryCalls,
    interruptCalls: () => interruptCalls,
    closeCalls: () => closeCalls,
    emit(message: Record<string, unknown>) { controller.enqueue(message as SDKMessage); },
    end() { outputClosed = true; controller.close(); },
    error(error: Error) { outputClosed = true; controller.error(error); },
    call(id: unknown, name = "run", args: Record<string, unknown> = { value: "same" }) {
      return client.request({ method: "tools/call", params: {
        name, arguments: args, _meta: { "claudecode/toolUseId": id },
      } }, CallToolResultSchema);
    },
  };
}

describe("Claude SDK persistent session worker", () => {
  test("send acknowledges only after the SDK resumes input, independently of results and onMessage", async () => {
    const blockedMessage = deferred<void>();
    let entered = false;
    const h = await harness({ onMessage: () => { entered = true; return blockedMessage.promise; } });
    h.emit({ type: "assistant" });
    await until(() => entered);
    let written = false;
    const sent = h.worker.send("ordered input").then(() => { written = true; });
    expect((await h.input.next()).value.message.content).toBe("ordered input");
    await Bun.sleep(0);
    expect(written).toBe(false);
    const next = h.input.next(); // SDK calls next() after its transport write completes.
    await sent;
    expect(written).toBe(true);
    expect(h.messages).toEqual([]);
    h.worker.stop();
    expect((await next).done).toBe(true);
    blockedMessage.resolve();
  });

  test("stop rejects both the yielded write and queued writes", async () => {
    const h = await harness();
    const first = h.worker.send("yielded");
    const queued = h.worker.send("queued");
    expect((await h.input.next()).value.message.content).toBe("yielded");
    h.worker.stop();
    await expect(first).rejects.toThrow("stopped");
    await expect(queued).rejects.toThrow("stopped");
    expect((await h.input.next()).done).toBe(true);
    expect(h.errors).toEqual([]);
  });

  test("sends only explicit supported SDK follow-up fields", async () => {
    const h = await harness();
    const uuid = "aabbccdd-1234-4567-890a-aabbccddeeff";
    for (const priority of ["now", "next", "later"] as const) {
      h.worker.send("follow-up", { uuid, shouldQuery: false, priority, isSynthetic: true } as Parameters<ClaudeSessionWorker["send"]>[1]);
      expect((await h.input.next()).value).toEqual({
        type: "user", message: { role: "user", content: "follow-up" }, parent_tool_use_id: null,
        uuid, shouldQuery: false, priority,
      });
    }
    h.worker.send("query", { shouldQuery: true });
    expect((await h.input.next()).value.shouldQuery).toBe(true);
    h.worker.send("default", { uuid: undefined, shouldQuery: undefined, priority: undefined });
    expect((await h.input.next()).value).toEqual({
      type: "user", message: { role: "user", content: "default" }, parent_tool_use_id: null,
    });
  });

  test.each([{ uuid: "bad\nframe" }, { uuid: 42 }, { shouldQuery: "false" }, { priority: "immediate" }])(
    "rejects invalid follow-up options before enqueueing: %j", async options => {
      const h = await harness();
      expect(() => h.worker.send("invalid", options as Parameters<ClaudeSessionWorker["send"]>[1])).toThrow("Invalid Claude worker");
      h.worker.send("valid");
      expect((await h.input.next()).value.message.content).toBe("valid");
    },
  );

  test("interrupt awaits only control ack, retains pending MCP and input, and can run inside onMessage", async () => {
    const ack = deferred<void>();
    const interrupted = deferred<void>();
    const receipt = deferred<Awaited<ReturnType<ClaudeWorkerOptions["onToolCall"]>>>();
    let signal: AbortSignal | undefined;
    let callDone = false;
    const seen: string[] = [];
    const h = await harness({
      onToolCall: async (_call, value) => { signal = value; return receipt.promise; },
      onMessage: async message => {
        seen.push(String(message.type));
        if (message.type === "assistant") { await h.worker.interrupt(); interrupted.resolve(); }
      },
    }, () => ack.promise);
    const call = h.call("toolu_interrupt_pending");
    void call.then(() => { callDone = true; });
    await until(() => signal !== undefined);
    h.emit({ type: "assistant" });
    await until(() => h.interruptCalls() === 1);
    let acked = false;
    void interrupted.promise.then(() => { acked = true; });
    await Bun.sleep(0);
    expect(acked).toBe(false);
    ack.resolve();
    await interrupted.promise;
    expect(seen).toEqual(["assistant"]);
    expect(signal!.aborted).toBe(false);
    expect(callDone).toBe(false);
    expect(h.closeCalls()).toBe(0);
    h.emit({ type: "result", subtype: "success" });
    await until(() => seen.length === 2);
    h.worker.send("same stream after interrupt");
    expect((await h.input.next()).value.message.content).toBe("same stream after interrupt");
    receipt.resolve({ content: [{ type: "text", text: "actual external receipt" }] });
    expect((await call).content).toEqual([{ type: "text", text: "actual external receipt" }]);
    expect(h.queryCalls()).toBe(1);
    expect(h.errors).toEqual([]);
  });

  test("interrupt errors propagate without shutting down the worker; stopped interrupts reject", async () => {
    const h = await harness({}, async () => { throw new Error("SDK interrupt rejected"); });
    await expect(h.worker.interrupt()).rejects.toThrow("Claude CLI interrupt failed");
    expect(h.closeCalls()).toBe(0);
    h.worker.send("still usable");
    expect((await h.input.next()).value.message.content).toBe("still usable");
    expect(h.errors).toEqual([]);
    h.worker.stop();
    await expect(h.worker.interrupt()).rejects.toThrow("stopped");
    expect(h.interruptCalls()).toBe(1);
  });

  test("keeps one input stream alive across two results and delivers messages sequentially", async () => {
    const gate = deferred<void>();
    const seen: SDKMessage[] = [];
    const h = await harness({ onMessage: async message => {
      seen.push(message);
      if (seen.length === 1) await gate.promise;
    } });
    h.worker.send("first");
    expect((await h.input.next()).value).toEqual({
      type: "user", message: { role: "user", content: "first" }, parent_tool_use_id: null,
    });
    h.emit({ type: "result", result: "one" });
    h.emit({ type: "result", result: "two" });
    await until(() => seen.length === 1);
    await Bun.sleep(0);
    expect(seen).toHaveLength(1);
    gate.resolve();
    await until(() => seen.length === 2);
    const next = h.input.next();
    let ended = false;
    void next.then(() => { ended = true; });
    await Bun.sleep(0);
    expect(ended).toBe(false);
    h.worker.send("second");
    expect((await next).value.message.content).toBe("second");
    expect(h.queryCalls()).toBe(1);
    expect(h.errors).toEqual([]);
  });

  test("preserves raw JSON Schema and correlates parallel identical calls by exact model IDs", async () => {
    const first = deferred<Awaited<ReturnType<ClaudeWorkerOptions["onToolCall"]>>>();
    const second = deferred<Awaited<ReturnType<ClaudeWorkerOptions["onToolCall"]>>>();
    const ids: string[] = [];
    const h = await harness({ onToolCall: async call => {
      ids.push(call.id);
      expect(call).toEqual({ id: call.id, name: "run", arguments: { value: "same" } });
      return call.id === "toolu_first" ? first.promise : second.promise;
    } });
    expect((await h.client.listTools()).tools).toEqual([tool]);
    const a = h.call("toolu_first");
    const b = h.call("toolu_second");
    await until(() => ids.length === 2);
    expect(ids).toEqual(["toolu_first", "toolu_second"]);
    let aDone = false;
    void a.then(() => { aDone = true; });
    second.resolve({ content: [{ type: "text", text: "second receipt" }] });
    expect((await b).content).toEqual([{ type: "text", text: "second receipt" }]);
    expect(aDone).toBe(false);
    first.resolve({ content: [{ type: "text", text: "first receipt" }] });
    expect((await a).content).toEqual([{ type: "text", text: "first receipt" }]);
  });

  test("delivers omitted arguments as an empty object for a zero-argument tool", async () => {
    let received: Record<string, unknown> | undefined;
    const h = await harness({
      tools: [{ ...tool, inputSchema: { type: "object", properties: {}, additionalProperties: false } }],
      onToolCall: async call => {
        expect(call.id).toBe("toolu_zero_args");
        received = call.arguments;
        return { content: [{ type: "text", text: "zero-argument receipt" }] };
      },
    });
    const result = await h.client.request({ method: "tools/call", params: {
      name: "run", _meta: { "claudecode/toolUseId": "toolu_zero_args" },
    } }, CallToolResultSchema);
    expect(received).toEqual({});
    expect(result.content).toEqual([{ type: "text", text: "zero-argument receipt" }]);
    expect(h.errors).toEqual([]);
  });

  test.each([null, []])("rejects non-object arguments %j without invoking the tool", async args => {
    let executed = false;
    const h = await harness({ onToolCall: async () => { executed = true; return { content: [] }; } });
    await expect(h.client.request({ method: "tools/call", params: {
      name: "run", arguments: args, _meta: { "claudecode/toolUseId": "toolu_bad_args" },
    } }, CallToolResultSchema)).rejects.toThrow();
    expect(executed).toBe(false);
  });

  test.each([undefined, "", " ", "x".repeat(257), 42])("fails closed on missing or invalid model ID %j", async id => {
    let executed = false;
    const h = await harness({ onToolCall: async () => {
      executed = true;
      return { content: [] };
    } });
    await expect(h.call(id)).rejects.toThrow();
    await until(() => h.errors.length === 1);
    expect(executed).toBe(false);
    expect(() => h.worker.send("after failure")).toThrow("stopped");
    await Bun.sleep(0);
    expect(h.errors).toHaveLength(1);
  });

  test("rejects a name outside the current catalog without executing it", async () => {
    let executed = false;
    const h = await harness({ onToolCall: async () => { executed = true; return { content: [] }; } });
    await expect(h.call("toolu_unknown", "not_advertised")).rejects.toThrow();
    expect(executed).toBe(false);
    expect(h.errors).toHaveLength(1);
  });

  test("updates the live tool catalog and notifies list_changed", async () => {
    const h = await harness();
    let notifications = 0;
    h.client.setNotificationHandler(ToolListChangedNotificationSchema, () => { notifications++; });
    const updated = { ...tool, name: "replacement" };
    await h.worker.updateTools([updated]);
    await until(() => notifications === 1);
    expect((await h.client.listTools()).tools).toEqual([updated]);
    expect((await h.call("toolu_new", "replacement")).content).toEqual([{ type: "text", text: "completed" }]);
    await expect(h.call("toolu_old", "run")).rejects.toThrow();
    expect(h.errors).toHaveLength(1);
  });

  test("preserves SDK-prepared launch env within the scoped environment", async () => {
    const h = await harness({ env: { ONLY_SCOPED: "yes", CLAUDECODE: "parent-proxy" } });
    expect(h.options).toMatchObject({
      pathToClaudeCodeExecutable: process.execPath,
      model: "offline-model", effort: "high",
      systemPrompt: "Private system prompt: keep off argv",
      tools: [], includePartialMessages: true, settingSources: [], strictMcpConfig: true, persistSession: false,
      verbatimPrompts: true,
      allowedTools: ["mcp__opencodex__*"],
    });
    expect(h.options.permissionMode).toBeUndefined();
    expect(h.options.allowDangerouslySkipPermissions).toBeUndefined();
    expect(Object.keys(h.options.mcpServers!)).toEqual(["opencodex"]);
    const env = h.options.env!;
    expect(env.CLAUDE_CODE_ENTRYPOINT).toBeUndefined();
    expect(env.CLAUDE_AGENT_SDK_VERSION).toBeUndefined();
    expect(env.CLAUDE_CODE_SDK_READS_SESSION_STATE).toBeUndefined();
    expect(env.CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS).toBe("0");
    expect(Number(env.MCP_TOOL_TIMEOUT)).toBeGreaterThanOrEqual(3_600_000);
    expect(Number(env.MCP_TOOL_TIMEOUT)).toBeLessThanOrEqual(86_400_000);
    const child = h.options.spawnClaudeCodeProcess!({
      command: process.execPath,
      args: ["-e", "process.stdout.write(JSON.stringify(process.env))"],
      env: { ...env, SDK_PREPARED: "honored", CLAUDECODE: "proxy",
        CLAUDE_CODE_ENTRYPOINT: "sdk-ts", CLAUDE_AGENT_SDK_VERSION: "sdk-owned-version",
        CLAUDE_CODE_SDK_READS_SESSION_STATE: "1" },
      signal: new AbortController().signal,
    });
    const text = await new Response(child.stdout as unknown as ReadableStream).text();
    const actual = JSON.parse(text);
    expect(actual.ONLY_SCOPED).toBe("yes");
    expect(actual.ANTHROPIC_API_KEY).toBeUndefined();
    expect(actual.CLAUDE_CODE_OAUTH_TOKEN).toBeUndefined();
    expect(actual.CLAUDECODE).toBeUndefined();
    expect(actual.HTTP_PROXY).toBeUndefined();
    expect(actual.OCX_TEST_PRELOAD_PID).toBeUndefined();
    expect(actual.SDK_PREPARED).toBe("honored");
    expect(actual.CLAUDE_CODE_ENTRYPOINT).toBe("sdk-ts");
    expect(actual.CLAUDE_AGENT_SDK_VERSION).toBe("sdk-owned-version");
    expect(actual.CLAUDE_CODE_SDK_READS_SESSION_STATE).toBe("1");
    for (const [key, value] of Object.entries(env)) expect(actual[key]).toBe(value);
    const guarded = await harness();
    expect(() => guarded.options.spawnClaudeCodeProcess!({
      command: process.execPath, args: ["--system-prompt=private"], env: {},
      signal: new AbortController().signal,
    })).toThrow("stdin");
  });

  test.each([{}, { sessionId: "11111111-1111-4111-8111-111111111111" }, { resume: "22222222-2222-4222-8222-222222222222" }])(
    "passes explicit cwd and persistence/session options to the SDK: %j", async identity => {
      const h = await harness({ cwd: fixturePath("."), persistSession: true, ...identity });
      expect(h.options).toMatchObject({ cwd: fixturePath("."), persistSession: true, ...identity });
    },
  );

  test("rejects sessionId plus resume before calling the SDK", async () => {
    let launched = false;
    await expect(harness({ sessionId: "private-new", resume: "private-resume", query: () => {
      launched = true; throw new Error("must not launch");
    } })).rejects.toThrow("mutually exclusive");
    expect(launched).toBe(false);
  });

  test("sanitizes synchronous SDK launch failures", async () => {
    const errors: Error[] = [];
    await expect(harness({ query: () => { throw new Error("private prompt and credentials"); },
      onError: error => { errors.push(error); },
    })).rejects.toThrow("Claude CLI session launch failed");
    expect(errors.map(error => error.message)).toEqual(["Claude CLI session launch failed"]);
  });

  test("sanitizes rejected external tool calls", async () => {
    const h = await harness({ onToolCall: async () => { throw new Error("private tool arguments and credentials"); } });
    const result = await h.call("toolu_failed").then(value => JSON.stringify(value), error => String(error));
    expect(result).not.toContain("private tool");
    expect(h.errors.map(error => error.message)).toEqual(["Claude MCP tool call failed"]);
  });

  test("honors the SDK launch signal without aborting unrelated children", async () => {
    const h = await harness();
    const controller = new AbortController();
    const unrelated = spawn(process.execPath, ["-e", "setInterval(() => {}, 1000)"], { windowsHide: true, stdio: "ignore" });
    try {
      const child = h.options.spawnClaudeCodeProcess!({ command: process.execPath,
        args: ["-e", "process.stdout.write('ready'); setInterval(() => {}, 1000)"],
        env: {}, signal: controller.signal,
      });
      await new Promise<void>(resolve => { child.stdout.once("data", () => resolve()); });
      const exited = new Promise<void>(resolve => { child.once("exit", () => resolve()); });
      controller.abort(new Error("private abort reason"));
      await exited;
      await h.worker.stop();
      expect(unrelated.exitCode).toBeNull();
      expect(unrelated.signalCode).toBeNull();
      expect(h.errors.map(error => error.message)).toEqual(["Claude CLI process failed"]);
    } finally { unrelated.kill("SIGKILL"); }
  });

  test.each([false, true])("stop is concurrent/reentrant-safe and waits for owned exit (hung=%s)", async hung => {
    const h = await harness();
    const code = hung
      ? "process.stdout.write('ready'); setInterval(() => {}, 1000)"
      : "process.stdin.resume(); process.stdin.on('end', () => setTimeout(() => process.exit(0), 100)); process.stdout.write('ready')";
    const child = h.options.spawnClaudeCodeProcess!({ command: process.execPath, args: ["-e", code],
      env: {}, signal: new AbortController().signal,
    });
    await new Promise<void>(resolve => { child.stdout.once("data", () => resolve()); });
    const started = Date.now();
    const closing = h.worker.stop();
    expect(h.worker.stop()).toBe(closing);
    child.stderr.emit("error", new Error("private cleanup recursion"));
    expect(h.worker.stop()).toBe(closing);
    expect(child.exitCode).toBeNull();
    await closing;
    expect(child.exitCode !== null || child.signalCode !== null).toBe(true);
    expect(h.closeCalls()).toBe(1);
    expect(h.errors).toEqual([]);
    expect(h.worker.stop()).toBe(closing);
    if (hung) {
      expect(Date.now() - started).toBeGreaterThanOrEqual(1_900);
      expect(Date.now() - started).toBeLessThan(6_000);
    } else {
      expect(child.exitCode).toBe(0);
      expect(child.signalCode).toBeNull();
    }
  }, 10_000);

  test.skipIf(process.platform !== "win32")("force-kills only the owned PID tree, including its descendant", async () => {
    const h = await harness();
    const unrelated = spawn(process.execPath, ["-e", "setInterval(() => {}, 1000)"], { windowsHide: true, stdio: "ignore" });
    let descendant: number | undefined;
    const exists = (pid: number) => { try { process.kill(pid, 0); return true; } catch { return false; } };
    try {
      const child = h.options.spawnClaudeCodeProcess!({ command: process.execPath, args: ["-e",
        "const {spawn}=require('node:child_process'); const child=spawn(process.execPath,['-e','setInterval(() => {}, 1000)'],{stdio:'ignore',windowsHide:true}); process.stdout.write(String(child.pid)); setInterval(() => {}, 1000)",
      ], env: {}, signal: new AbortController().signal });
      descendant = Number(await new Promise<string>(resolve => { child.stdout.once("data", chunk => resolve(String(chunk))); }));
      expect(Number.isSafeInteger(descendant) && descendant > 0).toBe(true);
      expect(exists(descendant)).toBe(true);
      expect(() => h.options.spawnClaudeCodeProcess!({ command: process.execPath, args: [], env: {},
        signal: new AbortController().signal,
      })).toThrow("already owns a process");
      await h.worker.stop();
      await until(() => !exists(descendant!));
      expect(unrelated.exitCode).toBeNull();
      expect(unrelated.signalCode).toBeNull();
      expect(h.errors).toEqual([]);
    } finally {
      await h.worker.stop();
      unrelated.kill("SIGKILL");
      if (descendant && exists(descendant)) process.kill(descendant, "SIGKILL");
    }
  }, 10_000);

  test("stop aborts pending tool results, ends stdin and emits no error", async () => {
    let callSignal: AbortSignal | undefined;
    const h = await harness({ onToolCall: async (_call, signal) => {
      callSignal = signal;
      return new Promise(() => {});
    } });
    const result = h.call("toolu_pending");
    const rejected = result.then(() => null, error => error);
    await until(() => callSignal !== undefined);
    const inputEnd = h.input.next();
    const closing = h.worker.stop();
    expect(h.worker.stop()).toBe(closing);
    await closing;
    expect(callSignal!.aborted).toBe(true);
    expect((await inputEnd).done).toBe(true);
    expect(await rejected).toBeInstanceOf(Error);
    await Bun.sleep(0);
    expect(h.errors).toEqual([]);
    await expect(h.worker.updateTools([])).rejects.toThrow("stopped");
  });

  test("stop gates detached MCP rejection until native interrupt acknowledges and SDK input closes", async () => {
    const ack = deferred<void>();
    let reject!: (error: Error) => void;
    let signal: AbortSignal | undefined;
    const receipt = new Promise<Awaited<ReturnType<ClaudeWorkerOptions["onToolCall"]>>>((_resolve, fail) => { reject = fail; });
    const h = await harness({ onToolCall: async (_call, value) => { signal = value; return receipt; } }, () => ack.promise);
    let settled = false;
    const call = h.call("toolu_stop_order").then(() => { settled = true; }, () => { settled = true; });
    await until(() => signal !== undefined);
    const closing = h.worker.stop();
    reject(new Error("Detached manager settlement"));
    await until(() => h.interruptCalls() === 1);
    await Bun.sleep(0);
    expect(settled).toBe(false);
    expect(h.closeCalls()).toBe(0);
    expect(signal!.aborted).toBe(false);
    ack.resolve();
    await closing;
    await call;
    expect(h.closeCalls()).toBe(1);
    expect(signal!.aborted).toBe(true);
    expect(h.errors).toEqual([]);
  });

  test.each(["error", "timeout"])("pending MCP shutdown stays bounded when native interrupt %s prevents acknowledgement", async mode => {
    let signal: AbortSignal | undefined;
    const h = await harness({ onToolCall: async (_call, value) => { signal = value; return new Promise(() => {}); } },
      async () => { if (mode === "error") throw new Error("Private control failure"); await new Promise(() => {}); });
    const call = h.call("toolu_stop_no_ack").catch(error => error);
    await until(() => signal !== undefined);
    const started = Date.now();
    await h.worker.stop();
    expect(Date.now() - started).toBeLessThan(2_500);
    expect(await call).toBeInstanceOf(Error);
    expect(signal!.aborted).toBe(true);
    expect(h.closeCalls()).toBe(1);
    expect(h.errors).toEqual([]);
  });

  test("interrupt preserves its owned running native process and only stop shuts it down", async () => {
    const h = await harness();
    const child = h.options.spawnClaudeCodeProcess!({
      command: process.execPath,
      args: ["-e", "process.stdout.write('ready'); setInterval(() => {}, 1000)"],
      env: {}, signal: new AbortController().signal,
    });
    await new Promise<void>(resolve => { child.stdout.once("data", () => resolve()); });
    let exited = false;
    child.once("exit", () => { exited = true; });
    await h.worker.interrupt();
    await Bun.sleep(0);
    expect(exited).toBe(false);
    expect(child.exitCode).toBeNull();
    expect(h.closeCalls()).toBe(0);
    h.worker.stop();
    await until(() => exited);
    expect(h.errors).toEqual([]);
  });

  test("the installed SDK uses native interrupt control without closing input and keeps the prompt off argv", async () => {
    const prompt = "Private offline initialize marker";
    let initializePrompt: unknown;
    let argv: string[] = [];
    const messages: SDKMessage[] = [];
    const errors: Error[] = [];
    const sent: Record<string, unknown>[] = [];
    const controls: string[] = [];
    let rejectInterrupt = false;
    let closed = false;
    const fakeQuery: typeof officialQuery = params => officialQuery({
      ...params,
      options: {
        ...params.options,
        spawnClaudeCodeProcess: launch => {
          argv = launch.args;
          const stdout = new PassThrough();
          const stderr = new PassThrough();
          const frame = (value: unknown) => stdout.write(JSON.stringify(value) + "\n");
          let buffered = "";
          const stdin = new Writable({
            write(chunk, _encoding, done) {
              buffered += String(chunk);
              let newline: number;
              while ((newline = buffered.indexOf("\n")) >= 0) {
                const message = JSON.parse(buffered.slice(0, newline));
                buffered = buffered.slice(newline + 1);
                if (message.type === "control_request" && message.request.subtype === "initialize") {
                  initializePrompt = message.request.systemPrompt;
                  frame({ type: "control_response", response: {
                    subtype: "success", request_id: message.request_id, response: {},
                  } });
                } else if (message.type === "control_request" && message.request.subtype === "interrupt") {
                  controls.push(message.request.subtype);
                  frame({ type: "control_response", response: {
                    subtype: rejectInterrupt ? "error" : "success", request_id: message.request_id,
                    ...(rejectInterrupt ? { error: "native interrupt refused" } : { response: { still_queued: [] } }),
                  } });
                } else if (message.type === "user") {
                  sent.push(message);
                  frame({ type: "result", subtype: "success", result: "offline", session_id: "offline" });
                }
              }
              done();
            },
            final(done) {
              closed = true;
              stdout.end(); stderr.end();
              child.exitCode = 0;
              child.emit("exit", 0, null);
              done();
            },
          });
          const child = Object.assign(new EventEmitter(), {
            stdin, stdout, stderr, exitCode: null as number | null, killed: false,
            kill() { this.killed = true; stdin.end(); return true; },
          });
          return child;
        },
      },
    });
    const worker = await createClaudeSessionWorker({
      binary: process.execPath, model: "offline-model", systemPrompt: prompt, env: {}, tools: [],
      query: fakeQuery,
      onMessage: message => { messages.push(message); },
      onToolCall: async () => { throw new Error("No tool should run"); },
      onError: error => { errors.push(error); },
    });
    active.push(worker);
    worker.send("offline user frame");
    await until(() => messages.some(message => message.type === "result"));
    expect(initializePrompt).toEqual([prompt]);
    expect(argv.join(" ")).not.toContain(prompt);
    expect(argv.some(arg => /^--(?:append-)?system-prompt(?:=|$)/.test(arg))).toBe(false);
    await worker.interrupt();
    expect(controls).toEqual(["interrupt"]);
    expect(messages.filter(message => message.type === "result")).toHaveLength(1);
    expect(closed).toBe(false);
    rejectInterrupt = true;
    await expect(worker.interrupt()).rejects.toThrow("Claude CLI interrupt failed");
    expect(closed).toBe(false);
    worker.send("offline follow-up", { uuid: "aabbccdd-1234-4567-890a-aabbccddeeff", shouldQuery: false, priority: "next" });
    await until(() => sent.length === 2);
    expect(sent[1]).toEqual({ type: "user", message: { role: "user", content: "offline follow-up" }, client_composed: true,
      parent_tool_use_id: null, uuid: "aabbccdd-1234-4567-890a-aabbccddeeff", shouldQuery: false, priority: "next" });
    expect(errors).toEqual([]);
  });

  test("drains a real owned child's stderr through the installed SDK without blocking results", async () => {
    let child!: ChildProcessWithoutNullStreams;
    const result = deferred<SDKMessage>();
    const errors: Error[] = [];
    const worker = await createClaudeSessionWorker({
      binary: process.execPath, model: "offline-model", systemPrompt: "Private offline prompt", env: {}, tools: [],
      query: params => officialQuery({
        ...params,
        options: { ...params.options, spawnClaudeCodeProcess: launch => {
          child = params.options!.spawnClaudeCodeProcess!({
            ...launch, command: process.execPath, args: [fixturePath("claude-session-stderr-child.cjs")],
          }) as ChildProcessWithoutNullStreams;
          return child;
        } },
      }),
      onMessage: message => { if (message.type === "result") result.resolve(message); },
      onToolCall: async () => { throw new Error("No tools should run"); },
      onError: error => { errors.push(error); },
    });
    active.push(worker);
    worker.send("offline stderr probe");
    expect(await result.promise).toMatchObject({ result: "stderr drained", stderr_bytes: 16 * 1024 * 1024 });
    expect(child.exitCode).toBeNull();
    expect(child.signalCode).toBeNull();
    expect(child.stderr.readableLength).toBeLessThan(64 * 1024);
    let exited = false;
    child.once("exit", () => { exited = true; });
    worker.stop();
    await until(() => exited);
    expect(errors).toEqual([]);
  }, 10_000);

  test.each(["stderr", "process"])("handles owned %s errors once without exposing private diagnostics", async source => {
    const h = await harness();
    const child = h.options.spawnClaudeCodeProcess!({
      command: process.execPath, args: ["-e", "process.stdout.write('ready'); setInterval(() => {}, 1000)"],
      env: {}, signal: new AbortController().signal,
    }) as ChildProcessWithoutNullStreams;
    await new Promise<void>(resolve => { child.stdout.once("data", () => resolve()); });
    let exited = false;
    child.once("exit", () => { exited = true; });
    const target = source === "stderr" ? child.stderr : child;
    target.emit("error", new Error("Private prompt, argv, and environment diagnostics"));
    await until(() => exited);
    expect(h.errors.map(error => error.message)).toEqual([
      source === "stderr" ? "Claude CLI stderr read failed" : "Claude CLI process failed",
    ]);
    expect(() => h.worker.send("after failure")).toThrow("stopped");
    target.emit("error", new Error("Late private diagnostic"));
    h.worker.stop();
    expect(h.errors).toHaveLength(1);
  });

  test("handles an asynchronous spawn failure without exposing its command or arguments", async () => {
    const h = await harness();
    h.options.spawnClaudeCodeProcess!({
      command: fixturePath("missing-private-claude-executable"), args: ["private argument"],
      env: {}, signal: new AbortController().signal,
    });
    await until(() => h.errors.length === 1);
    expect(h.errors[0].message).toBe("Claude CLI process failed");
    expect(() => h.worker.send("after failure")).toThrow("stopped");
  });

  test.each(["end", "error"])("reports unexpected SDK %s exactly once", async mode => {
    const h = await harness();
    if (mode === "end") h.end();
    else h.error(new Error("SDK process exited with code 7"));
    await until(() => h.errors.length === 1);
    expect(h.errors[0].message).toBe(mode === "end" ? "Claude CLI session ended unexpectedly" : "Claude CLI session failed");
    h.worker.stop();
    await Bun.sleep(0);
    expect(h.errors).toHaveLength(1);
  });
});
