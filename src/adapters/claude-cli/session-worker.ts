import { query as sdkQuery, type Query, type SDKMessage, type SDKUserMessage } from "@anthropic-ai/claude-agent-sdk";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import { execFileSync, spawn, type ChildProcess } from "node:child_process";
import type { CodingAgentMcpTool } from "../coding-agent/mcp-server";
import { CodingAgentProtocolError } from "../coding-agent/protocol";
import { ClaudeSessionError } from "./session-errors";

export interface ClaudeWorkerToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface ClaudeWorkerOptions {
  binary: string;
  model: string;
  systemPrompt: string;
  env: Record<string, string>;
  cwd?: string;
  sessionId?: string;
  resume?: string;
  persistSession?: boolean;
  effort?: string;
  tools: CodingAgentMcpTool[];
  onMessage: (message: SDKMessage) => void | Promise<void>;
  onToolCall: (call: ClaudeWorkerToolCall, signal: AbortSignal) => Promise<{
    content: Array<{ type: "text"; text: string }>;
    isError?: boolean;
  }>;
  onError: (error: Error) => void;
  query?: typeof sdkQuery;
}

export interface ClaudeSessionWorker {
  /**
   * Resolves when the SDK resumes input after its awaited transport write, not on a result
   * or native consumption. Pending writes reject on shutdown. shouldQuery:false appends
   * without starting a turn; consumption is confirmed by result.user_message_uuids.
   */
  send(text: string, options?: { uuid?: string; shouldQuery?: boolean; priority?: "now" | "next" | "later" }): Promise<void>;
  /**
   * Resolves on the SDK control acknowledgment, which can precede the terminal result.
   * The owner must drain onMessage through that result before replacing the active turn;
   * waiting for it here can deadlock an interrupt awaited inside onMessage. This does not
   * shut down the owned process/input stream or synthesize external tool results.
   */
  interrupt(): Promise<void>;
  updateTools(tools: CodingAgentMcpTool[]): Promise<void>;
  stop(): void | Promise<void>;
}

const SERVER_NAME = "opencodex";
const BRIDGE_ENV = {
  CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS: "0",
  MCP_TOOL_TIMEOUT: "86400000",
};

export async function createClaudeSessionWorker(options: ClaudeWorkerOptions): Promise<ClaudeSessionWorker> {
  if (options.sessionId !== undefined && options.resume !== undefined) {
    throw new Error("Claude worker sessionId and resume are mutually exclusive");
  }
  const effort = options.effort;
  if (effort !== undefined && effort !== "low" && effort !== "medium" && effort !== "high"
    && effort !== "xhigh" && effort !== "max") throw new Error("Unsupported Claude worker effort");

  const env = {
    ...Object.fromEntries(Object.entries(options.env).filter(([key]) =>
      key.toUpperCase() !== "CLAUDECODE" && !(key.toUpperCase() in BRIDGE_ENV))),
    ...BRIDGE_ENV,
  };
  // Cancel MCP waits independently so SDK shutdown retains its stdin grace.
  const abort = new AbortController();
  const input: Array<{ frame: SDKUserMessage; resolve: () => void; reject: (error: Error) => void }> = [];
  const pendingWrites = new Set<(error: Error) => void>();
  let wake: (() => void) | undefined;
  let stopped = false;
  let closing: Promise<void> | undefined;
  let activeMcp = 0;
  let session: Query | undefined;
  let child: ChildProcess | undefined;
  let tools = [...options.tools];
  let names = new Set(tools.map(tool => tool.name));
  const mcp = new McpServer({ name: SERVER_NAME, version: "1.0.0" }, {
    capabilities: { tools: { listChanged: true } },
  });

  function alive(owned: ChildProcess | undefined): owned is ChildProcess & { pid: number } {
    return !!owned?.pid && owned.exitCode === null && owned.signalCode === null;
  }

  async function waitForExit(owned: ChildProcess | undefined): Promise<void> {
    if (!alive(owned)) return;
    await new Promise<void>(resolve => {
      const done = () => {
        clearTimeout(timer);
        owned.off("exit", done);
        owned.off("close", done);
        resolve();
      };
      const timer = setTimeout(done, 2_000);
      owned.once("exit", done);
      owned.once("close", done);
    });
  }

  function stop(): Promise<void> {
    if (closing) return closing;
    stopped = true;
    // Publish the promise before cleanup can synchronously re-enter stop().
    closing = Promise.resolve().then(async () => {
      if (activeMcp && session) {
        let timer: ReturnType<typeof setTimeout> | undefined;
        try {
          await Promise.race([session.interrupt(), new Promise<void>(resolve => {
            timer = setTimeout(resolve, 1_000);
          })]);
        } catch { /* Broken control transport still needs bounded shutdown. */ }
        finally { clearTimeout(timer); }
      }
      try { session?.close(); } catch { /* Still close input if SDK cleanup fails. */ }
      const owned = child;
      try { owned?.stdin?.end(); } catch { /* The input may already be closed. */ }
      abort.abort(new Error("Claude session worker stopped"));
      void mcp.close().catch(() => {});
      await waitForExit(owned);
      // Only force our still-live process tree after stdin EOF has had its grace.
      if (alive(owned)) {
        try {
          if (process.platform === "win32") {
            const taskkill = `${process.env.SystemRoot ?? "C:\\Windows"}\\System32\\taskkill.exe`;
            execFileSync(taskkill, ["/PID", String(owned.pid), "/T", "/F"], {
              env, stdio: "ignore", windowsHide: true, timeout: 2_000,
            });
          } else {
            process.kill(-owned.pid, "SIGKILL");
          }
        } catch {
          if (alive(owned)) {
            try { owned.kill("SIGKILL"); } catch { /* The owned child already exited. */ }
          }
        }
        await waitForExit(owned);
        if (alive(owned)) throw new Error("Claude CLI process shutdown failed");
      }
    });
    // Compatible fire-and-forget owners must not cause an unhandled rejection.
    void closing.catch(() => {});
    const error = new Error("Claude session worker stopped");
    for (const reject of pendingWrites) reject(error);
    pendingWrites.clear();
    input.length = 0;
    wake?.();
    wake = undefined;
    return closing;
  }

  function fail(error: Error): void {
    if (stopped) return;
    stop();
    options.onError(error);
  }

  mcp.server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools }));
  mcp.server.setRequestHandler(CallToolRequestSchema, async (request, extra) => {
    const id = request.params._meta?.["claudecode/toolUseId"];
    const args = request.params.arguments ?? {};
    if (typeof id !== "string" || !id.length || id.length > 256 || id.trim() !== id
      || !names.has(request.params.name) || request.params.arguments === null || typeof args !== "object" || Array.isArray(args)) {
      const error = new CodingAgentProtocolError("Invalid isolated Claude MCP tool call; a supported native Claude CLI with tool-use identity metadata is required");
      fail(error);
      throw error;
    }
    const signal = AbortSignal.any([abort.signal, extra.signal]);
    if (signal.aborted) throw signal.reason;
    let onAbort: () => void;
    const cancelled = new Promise<never>((_resolve, reject) => {
      onAbort = () => reject(signal.reason);
      signal.addEventListener("abort", onAbort, { once: true });
    });
    activeMcp++;
    try {
      const result = await Promise.race([
        options.onToolCall({ id, name: request.params.name, arguments: args }, signal),
        cancelled,
      ]);
      // Detached owner settlements must not become CLI tool results during shutdown.
      if (stopped) await closing;
      return result;
    } catch {
      if (stopped) await closing;
      const error = new Error("Claude MCP tool call failed");
      if (!signal.aborted) fail(error);
      throw error;
    } finally {
      activeMcp--;
      signal.removeEventListener("abort", onAbort!);
    }
  });

  async function* userFrames(): AsyncGenerator<SDKUserMessage> {
    while (!stopped) {
      const message = input.shift();
      if (message) {
        yield message.frame;
        // SDK streamInput resumes only after awaiting transport.write for this frame.
        if (pendingWrites.delete(message.reject)) message.resolve();
      }
      else await new Promise<void>(resolve => { wake = resolve; });
    }
  }

  try {
    session = (options.query ?? sdkQuery)({
      prompt: userFrames(),
      options: {
        pathToClaudeCodeExecutable: options.binary,
        cwd: options.cwd,
        sessionId: options.sessionId,
        resume: options.resume,
        model: options.model,
        // SDK 0.3.314 sends this through initialize on stdin, never argv.
        systemPrompt: options.systemPrompt,
        verbatimPrompts: true,
        effort,
        env: { ...env },
        tools: [],
        includePartialMessages: true,
        strictMcpConfig: true,
        settingSources: [],
        allowedTools: [`mcp__${SERVER_NAME}__*`],
        persistSession: options.persistSession ?? false,
        mcpServers: { [SERVER_NAME]: { type: "sdk", name: SERVER_NAME, instance: mcp } },
        spawnClaudeCodeProcess: launch => {
          if (stopped) throw new Error("Claude session worker stopped");
          if (child) throw new Error("Claude worker already owns a process");
          if (launch.args.some(arg => /^--(?:append-)?system-prompt(?:=|$)/.test(arg))) {
            throw new Error("Claude worker system prompt must travel through stdin");
          }
          const ownedChild = spawn(launch.command, launch.args, {
            cwd: launch.cwd,
            env: {
              ...Object.fromEntries(Object.entries(launch.env).filter(([key]) =>
                key.toUpperCase() !== "CLAUDECODE" && !(key.toUpperCase() in BRIDGE_ENV))),
              ...BRIDGE_ENV,
            },
            signal: launch.signal,
            stdio: "pipe",
            windowsHide: true,
            detached: process.platform !== "win32",
          });
          child = ownedChild;
          ownedChild.on("error", () => fail(new Error("Claude CLI process failed")));
          ownedChild.stdin.on("error", () => fail(new Error("Claude CLI input write failed")));
          ownedChild.stderr.on("error", () => fail(new Error("Claude CLI stderr read failed")));
          // Custom SDK spawns must drain stderr; discard it to keep diagnostics private.
          ownedChild.stderr.resume();
          return ownedChild;
        },
      },
    });
  } catch {
    const error = new Error("Claude CLI session launch failed");
    fail(error);
    try { await stop(); }
    catch { throw new ClaudeSessionError("Claude CLI launch cleanup could not confirm process exit.", "claude_worker_shutdown_failed", 503, true); }
    throw error;
  }

  void (async () => {
    try {
      for await (const message of session!) {
        if (stopped) continue;
        await options.onMessage(message);
      }
      fail(new Error("Claude CLI session ended unexpectedly"));
    } catch {
      fail(new Error("Claude CLI session failed"));
    }
  })();

  return {
    send(text, options) {
      if (stopped) throw new Error("Claude session worker stopped");
      const frame: SDKUserMessage = { type: "user", message: { role: "user", content: text }, parent_tool_use_id: null };
      if (options?.uuid !== undefined) {
        if (typeof options.uuid !== "string" || !/^[\da-f]{8}(?:-[\da-f]{4}){3}-[\da-f]{12}$/i.test(options.uuid)) {
          throw new Error("Invalid Claude worker message UUID");
        }
        frame.uuid = options.uuid as SDKUserMessage["uuid"];
      }
      if (options?.shouldQuery !== undefined) {
        if (typeof options.shouldQuery !== "boolean") throw new Error("Invalid Claude worker shouldQuery");
        frame.shouldQuery = options.shouldQuery;
      }
      if (options?.priority !== undefined) {
        if (options.priority !== "now" && options.priority !== "next" && options.priority !== "later") {
          throw new Error("Invalid Claude worker priority");
        }
        frame.priority = options.priority;
      }
      const written = new Promise<void>((resolve, reject) => {
        pendingWrites.add(reject);
        input.push({ frame, resolve, reject });
      });
      // Existing fire-and-forget callers must not create an unhandled rejection on stop.
      void written.catch(() => {});
      wake?.();
      wake = undefined;
      return written;
    },
    async interrupt() {
      if (stopped) throw new Error("Claude session worker stopped");
      try { await session!.interrupt(); }
      catch { throw new Error("Claude CLI interrupt failed"); }
    },
    async updateTools(updated) {
      if (stopped) throw new Error("Claude session worker stopped");
      tools = [...updated];
      names = new Set(tools.map(tool => tool.name));
      if (mcp.isConnected()) await mcp.server.sendToolListChanged();
    },
    stop,
  };
}
