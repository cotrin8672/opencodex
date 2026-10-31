import assert from "node:assert/strict";
import { execFileSync, type ChildProcess } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { query, type Query, type SDKMessage, type SDKUserMessage } from "@anthropic-ai/claude-agent-sdk";
import { createClaudeSessionWorker, type ClaudeWorkerOptions } from "../../src/adapters/claude-cli/session-worker";
import { isTestHomeGuardArmed } from "../../src/lib/test-home-guard";
import { repoPath, repoRoot } from "../helpers/repo-root";

const packageName = `@anthropic-ai/claude-agent-sdk-${process.platform}-${process.arch}`;
export const workerNativeBinary = [packageName, `${packageName}-musl`]
  .map(name => repoPath("node_modules", name, process.platform === "win32" ? "claude.exe" : "claude"))
  .find(existsSync);
export function workerGate() {
  let release!: () => void;
  const wait = new Promise<void>(resolve => { release = resolve; });
  return { wait, release };
}
export async function workerUntil(check: () => boolean, description: string, timeoutMs = 10_000) {
  const deadline = Date.now() + timeoutMs;
  while (!check()) {
    if (Date.now() >= deadline) throw new Error(`Native worker timed out: ${description}`);
    await Bun.sleep(10);
  }
}

type Block = { type: "text"; text: string }
  | { type: "tool_use"; id: string; name: string; input: Record<string, unknown> };
type Reply = { content: Block[]; wait?: Promise<void> };
type Request = { model: string; stream?: boolean; messages: unknown[] };
function response(body: Request, reply: Reply, ordinal: number) {
  const stop_reason = reply.content.some(block => block.type === "tool_use") ? "tool_use" : "end_turn";
  const message = { id: `msg_worker_${ordinal}`, type: "message", role: "assistant", model: body.model,
    content: reply.content, stop_reason, stop_sequence: null, usage: { input_tokens: 12, output_tokens: 10 } };
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
    headers: { "content-type": "text/event-stream", "request-id": `req_worker_${ordinal}` },
  });
}

/** Direct worker proof: real SDK/CLI/control frames, dummy loopback inference, no manager. */
export async function nativeWorker(replies: Reply[], onToolCall: ClaudeWorkerOptions["onToolCall"] = async () => {
  throw new Error("Unscripted external tool call");
}) {
  assert(workerNativeBinary, "Packaged native Claude is absent");
  assert(isTestHomeGuardArmed() && process.env.OCX_TEST_PRELOAD_PID === String(process.pid), "Load repository test preload");
  assert.equal(realpathSync(process.cwd()), realpathSync(repoRoot()));
  assert.equal(JSON.parse(readFileSync(repoPath("node_modules/@anthropic-ai/claude-agent-sdk/package.json"), "utf8")).version, "0.3.314");
  const tempRoot = realpathSync(tmpdir());
  const home = mkdtempSync(join(tempRoot, "ocx-worker-native-"));
  for (const dir of [".claude", "AppData/Local", "AppData/Roaming", "tmp"]) mkdirSync(join(home, dir), { recursive: true });
  const requests: Request[] = [];
  const unexpected: string[] = [];
  const children: ChildProcess[] = [];
  const raw: Record<string, unknown>[] = [];
  const messages: SDKMessage[] = [];
  const errors: Error[] = [];
  const written: SDKUserMessage[] = [];
  const shutdown = workerGate();
  const server = Bun.serve({ hostname: "127.0.0.1", port: 0, async fetch(req) {
    const url = new URL(req.url);
    if (url.hostname !== "127.0.0.1") { unexpected.push("External request refused"); return new Response(null, { status: 403 }); }
    if (url.pathname === "/v1/messages/count_tokens") return Response.json({ input_tokens: 12 });
    if (url.pathname !== "/v1/messages" || req.headers.get("x-api-key") !== "offline-worker-fixture-key") {
      unexpected.push(url.pathname); return new Response(null, { status: 403 });
    }
    const body = await req.json() as Request;
    const ordinal = requests.push(body);
    const reply = replies[ordinal - 1];
    if (!reply) { unexpected.push(`Unscripted inference ${ordinal}`); return new Response(null, { status: 400 }); }
    if (reply.wait) await Promise.race([reply.wait, shutdown.wait]);
    return response(body, reply, ordinal);
  } });
  const baseUrl = `http://127.0.0.1:${server.port}`;
  const env: Record<string, string> = {
    HOME: home, USERPROFILE: home, CLAUDE_CONFIG_DIR: join(home, ".claude"),
    APPDATA: join(home, "AppData/Roaming"), LOCALAPPDATA: join(home, "AppData/Local"),
    TEMP: join(home, "tmp"), TMP: join(home, "tmp"), TMPDIR: join(home, "tmp"),
    PATH: process.platform === "win32" ? `${process.env.SystemRoot ?? "C:\\Windows"}\\System32` : "/usr/bin:/bin",
    ANTHROPIC_BASE_URL: baseUrl, ANTHROPIC_API_KEY: "offline-worker-fixture-key",
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: "1", DISABLE_TELEMETRY: "1", DISABLE_ERROR_REPORTING: "1", DISABLE_AUTOUPDATER: "1",
    HTTP_PROXY: baseUrl, HTTPS_PROXY: baseUrl, ALL_PROXY: baseUrl, NO_PROXY: "127.0.0.1,localhost",
  };
  if (process.platform === "win32") Object.assign(env, {
    SystemRoot: process.env.SystemRoot ?? "C:\\Windows", WINDIR: process.env.WINDIR ?? "C:\\Windows",
    HOMEDRIVE: home.slice(0, 2), HOMEPATH: home.slice(2),
  });
  let session!: Query;
  const worker = await createClaudeSessionWorker({
    binary: workerNativeBinary, model: "claude-opus-4-6", systemPrompt: "Offline worker proof", env, cwd: home,
    tools: [{ name: "run", description: "Client-owned operation", inputSchema: { type: "object", properties: {}, additionalProperties: false } }],
    onToolCall, onMessage: message => { messages.push(message); }, onError: error => { errors.push(error); },
    query: params => {
      assert.deepEqual(params.options?.tools, []);
      assert.deepEqual(params.options?.settingSources, []);
      session = query({ ...params, prompt: (async function* () {
        for await (const frame of params.prompt as AsyncIterable<SDKUserMessage>) {
          yield frame;
          written.push(frame); // The SDK resumed the iterator after awaiting transport.write.
        }
      })(), options: { ...params.options, spawnClaudeCodeProcess: launch => {
        assert.equal(launch.cwd, home);
        assert.equal(launch.env.CLAUDE_CODE_ENTRYPOINT, "sdk-ts");
        assert.equal(launch.env.CLAUDE_AGENT_SDK_VERSION, "0.3.314");
        assert.equal(launch.env.CLAUDE_CODE_SDK_READS_SESSION_STATE, "1");
        const child = params.options!.spawnClaudeCodeProcess!(launch);
        children.push(child as ChildProcess);
        let buffer = "";
        child.stdout.on("data", chunk => {
          buffer += String(chunk);
          let end: number;
          while ((end = buffer.indexOf("\n")) >= 0) {
            const line = buffer.slice(0, end);
            buffer = buffer.slice(end + 1);
            if (line.trim()) raw.push(JSON.parse(line));
          }
        });
        return child;
      } } });
      return session;
    },
  });
  return {
    worker, requests, unexpected, children, messages, raw, errors,
    version() { return execFileSync(workerNativeBinary, ["--version"], { env, cwd: home, timeout: 5_000, windowsHide: true }).toString().trim(); },
    async barrier(uuid: string) {
      await workerUntil(() => written.some(frame => frame.uuid === uuid), "SDK input write");
      await session.mcpServerStatus(); // Native control ack after the earlier stdin frame.
    },
    async close() {
      await worker.stop(); shutdown.release();
      await server.stop(true);
      await workerUntil(() => children.every(child => child.exitCode !== null || child.signalCode !== null), "owned child exit", 3_000);
      assert.equal(dirname(realpathSync(home)), tempRoot);
      assert(home.startsWith(join(tempRoot, "ocx-worker-native-")));
      rmSync(home, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
    },
  };
}
