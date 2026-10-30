import { query as sdkQuery, type Query, type SDKUserMessage } from "@anthropic-ai/claude-agent-sdk";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import { execFileSync, spawn, type ChildProcess } from "node:child_process";
import type { CodingAgentMcpTool } from "../coding-agent/mcp-server";
import { CodingAgentProtocolError } from "../coding-agent/protocol";

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
  effort?: string;
  tools: CodingAgentMcpTool[];
  onMessage: (message: Record<string, unknown>) => void | Promise<void>;
  onToolCall: (call: ClaudeWorkerToolCall, signal: AbortSignal) => Promise<{
    content: Array<{ type: "text"; text: string }>;
    isError?: boolean;
  }>;
  onError: (error: Error) => void;
  query?: typeof sdkQuery;
}

export interface ClaudeSessionWorker {
  send(text: string): void;
  updateTools(tools: CodingAgentMcpTool[]): Promise<void>;
  stop(): void;
}

const SERVER_NAME = "opencodex";
const SDK_ENV = {
  CLAUDE_CODE_ENTRYPOINT: "sdk-ts",
  CLAUDE_AGENT_SDK_VERSION: "0.3.314",
  CLAUDE_CODE_SDK_READS_SESSION_STATE: "1",
  CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS: "0",
  MCP_TOOL_TIMEOUT: "86400000",
};

export async function createClaudeSessionWorker(options: ClaudeWorkerOptions): Promise<ClaudeSessionWorker> {
  const effort = options.effort;
  if (effort !== undefined && effort !== "low" && effort !== "medium" && effort !== "high"
    && effort !== "xhigh" && effort !== "max") throw new Error("Unsupported Claude worker effort");

  const env = {
    ...Object.fromEntries(Object.entries(options.env).filter(([key]) =>
      key.toUpperCase() !== "CLAUDECODE" && !(key.toUpperCase() in SDK_ENV))),
    ...SDK_ENV,
  };
  const abort = new AbortController();
  const input: SDKUserMessage[] = [];
  let wake: (() => void) | undefined;
  let stopped = false;
  let session: Query | undefined;
  let child: ChildProcess | undefined;
  let tools = [...options.tools];
  let names = new Set(tools.map(tool => tool.name));
  const mcp = new McpServer({ name: SERVER_NAME, version: "1.0.0" }, {
    capabilities: { tools: { listChanged: true } },
  });

  function stop(): void {
    if (stopped) return;
    stopped = true;
    input.length = 0;
    wake?.();
    wake = undefined;
    abort.abort(new Error("Claude session worker stopped"));
    // Kill only our still-live child (and its descendants), never a discovered user CLI.
    if (child?.pid && child.exitCode === null && child.signalCode === null) {
      try {
        if (process.platform === "win32") {
          const taskkill = `${process.env.SystemRoot ?? "C:\\Windows"}\\System32\\taskkill.exe`;
          execFileSync(taskkill, ["/PID", String(child.pid), "/T", "/F"], {
            env, stdio: "ignore", windowsHide: true, timeout: 2_000,
          });
        } else {
          process.kill(-child.pid, "SIGKILL");
        }
      } catch {
        try { child.kill("SIGKILL"); } catch { /* The owned child already exited. */ }
      }
    }
    try { session?.close(); } catch { /* Cancellation must finish even if SDK cleanup fails. */ }
    void mcp.close().catch(() => {});
  }

  function fail(error: unknown): void {
    if (stopped) return;
    stop();
    options.onError(error instanceof Error ? error : new Error(String(error)));
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
    try {
      return await Promise.race([
        options.onToolCall({ id, name: request.params.name, arguments: args }, signal),
        cancelled,
      ]);
    } catch (error) {
      if (!signal.aborted) fail(error);
      throw error;
    } finally {
      signal.removeEventListener("abort", onAbort!);
    }
  });

  async function* userFrames(): AsyncGenerator<SDKUserMessage> {
    while (!stopped) {
      const message = input.shift();
      if (message) yield message;
      else await new Promise<void>(resolve => { wake = resolve; });
    }
  }

  try {
    session = (options.query ?? sdkQuery)({
      prompt: userFrames(),
      options: {
        pathToClaudeCodeExecutable: options.binary,
        model: options.model,
        // SDK 0.3.314 sends this through initialize on stdin, never argv.
        systemPrompt: options.systemPrompt,
        effort,
        env: { ...env },
        abortController: abort,
        tools: [],
        includePartialMessages: true,
        strictMcpConfig: true,
        settingSources: [],
        allowedTools: [`mcp__${SERVER_NAME}__*`],
        persistSession: false,
        mcpServers: { [SERVER_NAME]: { type: "sdk", name: SERVER_NAME, instance: mcp } },
        spawnClaudeCodeProcess: launch => {
          if (stopped) throw new Error("Claude session worker stopped");
          if (launch.args.some(arg => /^--(?:append-)?system-prompt(?:=|$)/.test(arg))) {
            throw new Error("Claude worker system prompt must travel through stdin");
          }
          const ownedChild = spawn(launch.command, launch.args, {
            cwd: launch.cwd,
            env: { ...env },
            stdio: "pipe",
            windowsHide: true,
            detached: process.platform !== "win32",
          });
          child = ownedChild;
          ownedChild.on("error", () => fail(new Error("Claude CLI process failed")));
          ownedChild.stderr.on("error", () => fail(new Error("Claude CLI stderr read failed")));
          // Custom SDK spawns must drain stderr; discard it to keep diagnostics private.
          ownedChild.stderr.resume();
          return ownedChild;
        },
      },
    });
  } catch (error) {
    fail(error);
    throw error;
  }

  void (async () => {
    try {
      for await (const message of session!) {
        if (stopped) return;
        await options.onMessage(message as unknown as Record<string, unknown>);
      }
      fail(new Error("Claude CLI session ended unexpectedly"));
    } catch (error) {
      fail(error);
    }
  })();

  return {
    send(text) {
      if (stopped) throw new Error("Claude session worker stopped");
      input.push({ type: "user", message: { role: "user", content: text }, parent_tool_use_id: null });
      wake?.();
      wake = undefined;
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
