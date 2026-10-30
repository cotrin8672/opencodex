import assert from "node:assert/strict";
import type { ChildProcess } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { query } from "@anthropic-ai/claude-agent-sdk";
import { ClaudeSessionManager, type ClaudePersistentTurn } from "../../src/adapters/claude-cli/session-manager";
import { createClaudeSessionWorker } from "../../src/adapters/claude-cli/session-worker";
import { isTestHomeGuardArmed } from "../../src/lib/test-home-guard";
import type { AdapterEvent } from "../../src/types";
import { repoPath, repoRoot } from "../helpers/repo-root";
import { MemoryJournal, turn } from "./claude-session-worker";

const packageName = `@anthropic-ai/claude-agent-sdk-${process.platform}-${process.arch}`;
export const nativeBinary = [packageName, `${packageName}-musl`]
  .map(name => repoPath("node_modules", name, process.platform === "win32" ? "claude.exe" : "claude"))
  .find(existsSync);

export type NativeBlock = { type: "text"; text: string }
  | { type: "tool_use"; id: string; name: string; input: Record<string, unknown> };
export type NativeRequest = {
  model: string;
  stream?: boolean;
  messages: Array<{ role: string; content: unknown }>;
  tools?: Array<{ name: string }>;
};
export type NativeReply = { content: NativeBlock[]; wait?: Promise<void> };

export function gate() {
  let release!: () => void;
  const wait = new Promise<void>(resolve => { release = resolve; });
  return { wait, release };
}

export async function nativeUntil(check: () => boolean, description: string, timeoutMs = 12_000) {
  const deadline = Date.now() + timeoutMs;
  while (!check()) {
    if (Date.now() >= deadline) throw new Error(`Native Claude fixture timed out: ${description}`);
    await Bun.sleep(10);
  }
}

function response(body: NativeRequest, reply: NativeReply, ordinal: number): Response {
  const stop_reason = reply.content.some(block => block.type === "tool_use") ? "tool_use" : "end_turn";
  const message = {
    id: `msg_native_${ordinal}`, type: "message", role: "assistant", model: body.model,
    content: reply.content, stop_reason, stop_sequence: null, usage: { input_tokens: 12, output_tokens: 10 },
  };
  if (!body.stream) return Response.json(message);
  const frames: Record<string, unknown>[] = [{ type: "message_start", message: {
    ...message, content: [], stop_reason: null, usage: { input_tokens: 12, output_tokens: 0 },
  } }];
  for (const [index, block] of reply.content.entries()) {
    frames.push({ type: "content_block_start", index, content_block: block.type === "tool_use"
      ? { ...block, input: {} } : { type: "text", text: "" } });
    frames.push({ type: "content_block_delta", index, delta: block.type === "tool_use"
      ? { type: "input_json_delta", partial_json: JSON.stringify(block.input) }
      : { type: "text_delta", text: block.text } });
    frames.push({ type: "content_block_stop", index });
  }
  frames.push({ type: "message_delta", delta: { stop_reason, stop_sequence: null }, usage: { output_tokens: 10 } },
    { type: "message_stop" });
  return new Response(frames.map(frame => `event: ${frame.type}\ndata: ${JSON.stringify(frame)}\n\n`).join(""), {
    headers: { "content-type": "text/event-stream", "cache-control": "no-cache", "request-id": `req_native_${ordinal}` },
  });
}

/** Real SDK + packaged native scheduler; only Anthropic inference is scripted. */
export function nativeSession(replies: NativeReply[]) {
  assert(nativeBinary, "The packaged native Claude executable is absent");
  assert(isTestHomeGuardArmed() && process.env.OCX_TEST_PRELOAD_PID === String(process.pid), "Load tests/preload.ts from the repo CWD");
  assert.equal(realpathSync(process.cwd()), realpathSync(repoRoot()), "Run native tests from the repository CWD");
  assert.equal(JSON.parse(readFileSync(repoPath("node_modules/@anthropic-ai/claude-agent-sdk/package.json"), "utf8")).version, "0.3.314");
  const tempRoot = realpathSync(tmpdir());
  const home = mkdtempSync(join(tempRoot, "ocx-claude-native-"));
  for (const dir of [".claude", "AppData/Local", "AppData/Roaming", "tmp"]) mkdirSync(join(home, dir), { recursive: true });
  const requests: NativeRequest[] = [];
  const replied: number[] = [];
  const unexpected: string[] = [];
  const children: ChildProcess[] = [];
  const requested: string[] = [];
  const resolved: string[] = [];
  const active = new Set<string>();
  let maxActive = 0;
  const shutdown = gate();
  const server = Bun.serve({ hostname: "127.0.0.1", port: 0, async fetch(req) {
    const url = new URL(req.url);
    if (url.hostname !== "127.0.0.1") {
      unexpected.push(req.url);
      return new Response("External network refused", { status: 403 });
    }
    if (url.pathname === "/v1/messages/count_tokens") return Response.json({ input_tokens: 12 });
    if (url.pathname !== "/v1/messages") {
      unexpected.push(url.pathname);
      return new Response("Unexpected endpoint", { status: 404 });
    }
    if (req.headers.get("x-api-key") !== "offline-native-fixture-key") {
      unexpected.push("Inference did not use the dummy API key");
      return new Response("Dummy API key required", { status: 401 });
    }
    const body = await req.json() as NativeRequest;
    const index = requests.push(body) - 1;
    const reply = replies[index];
    if (!reply) {
      unexpected.push(`Unscripted inference ${index + 1}`);
      return Response.json({ type: "error", error: { type: "invalid_request_error", message: "Unscripted inference" } }, { status: 400 });
    }
    if (reply.wait) await Promise.race([reply.wait, shutdown.wait]);
    replied.push(index + 1);
    return response(body, reply, index + 1);
  } });
  const baseUrl = `http://127.0.0.1:${server.port}`;
  // Allowlist environment: never inherit OAuth, API keys, user PATH, or profile settings.
  const env: Record<string, string> = {
    HOME: home, USERPROFILE: home, CLAUDE_CONFIG_DIR: join(home, ".claude"),
    APPDATA: join(home, "AppData/Roaming"), LOCALAPPDATA: join(home, "AppData/Local"),
    TEMP: join(home, "tmp"), TMP: join(home, "tmp"), TMPDIR: join(home, "tmp"),
    PATH: process.platform === "win32" ? `${process.env.SystemRoot ?? "C:\\Windows"}\\System32` : "/usr/bin:/bin",
    ANTHROPIC_BASE_URL: baseUrl, ANTHROPIC_API_KEY: "offline-native-fixture-key",
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: "1", DISABLE_TELEMETRY: "1", DISABLE_ERROR_REPORTING: "1", DISABLE_AUTOUPDATER: "1",
    HTTP_PROXY: baseUrl, HTTPS_PROXY: baseUrl, ALL_PROXY: baseUrl, NO_PROXY: "127.0.0.1,localhost",
  };
  if (process.platform === "win32") Object.assign(env, {
    SystemRoot: process.env.SystemRoot ?? "C:\\Windows", WINDIR: process.env.WINDIR ?? "C:\\Windows",
    HOMEDRIVE: home.slice(0, 2), HOMEPATH: home.slice(2),
  });
  const manager = new ClaudeSessionManager({
    journal: new MemoryJournal(), timeoutMs: 15_000, reconnectGraceMs: 30_000,
    createWorker: options => createClaudeSessionWorker({ ...options,
      query: params => {
        assert.deepEqual(params.options?.tools, []);
        assert.deepEqual(params.options?.settingSources, []);
        return query({ ...params, options: { ...params.options, cwd: home,
          spawnClaudeCodeProcess: launch => {
            assert.equal(launch.command, nativeBinary);
            const child = params.options!.spawnClaudeCodeProcess!(launch);
            children.push(child as ChildProcess);
            return child;
          },
        } });
      },
      onToolCall: async (call, signal) => {
        requested.push(call.id);
        active.add(call.id);
        maxActive = Math.max(maxActive, active.size);
        try {
          const result = await options.onToolCall(call, signal);
          resolved.push(call.id);
          return result;
        } finally { active.delete(call.id); }
      },
    }),
  });
  const input = turn();
  input.binary = nativeBinary;
  input.env = env;
  input.parsed.modelId = "claude-opus-4-6";
  return {
    input, requests, replied, unexpected, children, requested, resolved, get maxActive() { return maxActive; },
    run(input: ClaudePersistentTurn) {
      const events: AdapterEvent[] = [];
      const done = manager.run(input, event => { events.push(event); });
      void done.catch(() => {});
      return { events, done };
    },
    async close() {
      try { manager.close(); } finally {
        shutdown.release();
        await server.stop(true);
        for (const child of children) {
          if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
        }
        await nativeUntil(() => children.every(child => child.exitCode !== null || child.signalCode !== null), "owned native processes exit", 3_000);
        assert.equal(dirname(realpathSync(home)), tempRoot, "Cleanup must stay inside the owned temporary root");
        assert(home.startsWith(join(tempRoot, "ocx-claude-native-")));
        rmSync(home, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
      }
    },
  };
}
