import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { getAdapterDefinition } from "../../src/adapters/registry";
import { setClaudeUsagePreflightForTests } from "../../src/adapters/claude-cli/usage-admission";
import { flushConfigDirHardeningForTests } from "../../src/config/paths";
import { clearResponseStateForTests, flushResponseState } from "../../src/responses/state";
import {
  resolveApiAuth, resolveContextPrincipal, resolveDurableReplayPrincipal,
  resolveDurableSessionPrincipal, resolveResponsesApiAuth, type DataPlaneAdmission,
} from "../../src/server/auth-cors";
import { handleChatCompletions } from "../../src/server/chat-completions";
import { handleClaudeMessages } from "../../src/server/claude-messages";
import { handleResponses } from "../../src/server/responses";
import { buildResponsesWsData, selectForwardHeaders } from "../../src/server/ws-bridge";
import type { OcxConfig } from "../../src/types";
import { acquireOwnedSpendHome } from "../helpers/owned-spend-home";
import { removeTreeAfterBackgroundWork } from "../helpers/remove-tree";

const keyA = "ocx_fixture_session_principal_a";
const keyB = "ocx_fixture_session_principal_b";
let previousToken: string | undefined;

function config(hostname = "0.0.0.0"): OcxConfig {
  return {
    port: 0, hostname, defaultProvider: "claude-cli",
    apiKeys: [keyA, keyB].map((key, index) => ({ id: `key-${index}`, name: "Fixture", key, createdAt: "2026-10-30T00:00:00Z" })),
    providers: { "claude-cli": { adapter: "claude-cli", baseUrl: "https://api.anthropic.com", authMode: "key", selectedModels: ["claude-sonnet-5-5"] } },
  } as OcxConfig;
}

function request(headers: HeadersInit = {}): Request {
  return new Request("http://localhost/v1/responses", { headers });
}

beforeEach(() => {
  previousToken = process.env.OPENCODEX_API_AUTH_TOKEN;
  delete process.env.OPENCODEX_API_AUTH_TOKEN;
});
afterEach(() => {
  if (previousToken === undefined) delete process.env.OPENCODEX_API_AUTH_TOKEN;
  else process.env.OPENCODEX_API_AUTH_TOKEN = previousToken;
});

test("durable identity survives a fresh context-principal salt", async () => {
  const cfg = config();
  const req = request({ "x-opencodex-api-key": keyA });
  const admission = resolveResponsesApiAuth(req, cfg)!;
  // A fresh process resets the private salt without changing the production resolver.
  const child = Bun.spawn([process.execPath, "--eval", `
    const auth = await import(${JSON.stringify(new URL("../../src/server/auth-cors.ts", import.meta.url).href)});
    const cfg = ${JSON.stringify(cfg)};
    const req = new Request("http://localhost/", { headers: { "x-opencodex-api-key": ${JSON.stringify(keyA)} } });
    const admission = auth.resolveResponsesApiAuth(req, cfg);
    console.log(JSON.stringify({ local: auth.resolveContextPrincipal(req, cfg, admission),
      durable: auth.resolveDurableSessionPrincipal(req, cfg, admission) }));
  `], { stdout: "pipe", stderr: "pipe" });
  const watchdog = setTimeout(() => child.kill(), 5_000);
  try {
    const [stdout, stderr, exitCode] = await Promise.all([
      new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited,
    ]);
    expect(exitCode, stderr).toBe(0);
    const restarted = JSON.parse(stdout);
    expect(restarted.local).not.toBe(resolveContextPrincipal(req, cfg, admission));
    const durable = resolveDurableSessionPrincipal(req, cfg, admission);
    expect(durable).toMatch(/^[0-9a-f]{64}$/);
    expect(restarted.durable).toBe(durable);
    expect(durable).not.toBe(keyA);
  } finally { clearTimeout(watchdog); if (child.exitCode === null) child.kill(); await child.exited; }
}, 10_000);

test("header presentation does not rename an admitted key, and different keys stay isolated", () => {
  const cfg = config();
  const principals = [
    { "x-opencodex-api-key": keyA }, { authorization: `Bearer ${keyA}` }, { "x-api-key": keyA },
  ].map(headers => {
    const req = request(headers);
    return resolveDurableSessionPrincipal(req, cfg, resolveApiAuth(req, cfg)!);
  });
  expect(new Set(principals).size).toBe(1);
  const other = request({ "x-opencodex-api-key": keyB });
  expect(resolveDurableSessionPrincipal(other, cfg, resolveApiAuth(other, cfg)!)).not.toBe(principals[0]);
  const renamed = config();
  renamed.apiKeys![0]!.id = "another-key-id";
  const sameSecret = request({ "x-opencodex-api-key": keyA });
  expect(resolveDurableSessionPrincipal(sameSecret, renamed, resolveApiAuth(sameSecret, renamed)!)).not.toBe(principals[0]);
  process.env.OPENCODEX_API_AUTH_TOKEN = keyA;
  expect(resolveDurableSessionPrincipal(sameSecret, cfg, resolveApiAuth(sameSecret, cfg)!)).not.toBe(principals[0]);
});

test("credentialless loopback ignores rotating upstream bearer and untrusted principal headers", () => {
  const cfg = config("127.0.0.1");
  const baseline = request();
  const operator = resolveDurableSessionPrincipal(baseline, cfg, resolveApiAuth(baseline, cfg)!);
  expect(operator).toMatch(/^[0-9a-f]{64}$/);
  for (const token of ["upstream-openai-before", "upstream-openai-after"]) {
    const req = request({ authorization: `Bearer ${token}`, "x-api-key": "unrecognized-upstream-key",
      "x-opencodex-client-principal-id": token, "x-opencodex-api-key": "not-a-local-key" });
    expect(resolveDurableSessionPrincipal(req, cfg, resolveApiAuth(req, cfg)!)).toBe(operator);
  }
  expect(resolveDurableSessionPrincipal(baseline, cfg, undefined)).toBeUndefined();
});

test("real dedicated local keys take precedence and isolate local callers", () => {
  const cfg = config("127.0.0.1");
  const req = request({ "x-opencodex-api-key": keyA, authorization: `Bearer ${keyB}`, "x-api-key": keyB });
  const dedicated = resolveDurableSessionPrincipal(req, cfg, resolveApiAuth(req, cfg)!);
  const remote = config();
  const remoteReq = request({ "x-opencodex-api-key": keyA });
  expect(dedicated).toBe(resolveDurableSessionPrincipal(remoteReq, remote, resolveApiAuth(remoteReq, remote)!));
  const second = request({ "x-opencodex-api-key": keyB });
  expect(resolveDurableSessionPrincipal(second, cfg, resolveApiAuth(second, cfg)!)).not.toBe(dedicated);
  const noKey = request();
  expect(resolveDurableSessionPrincipal(noKey, cfg, resolveApiAuth(noKey, cfg)!)).not.toBe(dedicated);
});

test("remote ownership requires the actual admitted credential, source, kind, key ID and proof", () => {
  const cfg = config();
  const req = request({ "x-opencodex-api-key": keyA });
  const admitted = resolveApiAuth(req, cfg)!;
  const badOwner = request({ "x-opencodex-api-key": keyB, "x-opencodex-client-principal-id": "trusted-looking" });
  expect(resolveDurableSessionPrincipal(badOwner, cfg, admitted)).toBeUndefined();
  expect(resolveDurableSessionPrincipal(req, cfg, { kind: "environment", source: "dedicated" })).toBeUndefined();
  expect(resolveDurableSessionPrincipal(req, cfg, { ...admitted, contextPrincipalId: "forged" } as DataPlaneAdmission)).toBeUndefined();
  const wrongSource = request({ authorization: `Bearer ${keyA}` });
  expect(resolveDurableSessionPrincipal(wrongSource, cfg, admitted)).toBeUndefined();
  for (const headers of [{}, { authorization: "Bearer upstream-only" }, { "x-opencodex-client-principal-id": "forged" }]) {
    const unknown = request(headers);
    expect(resolveApiAuth(unknown, cfg)).toBeNull();
    expect(resolveDurableSessionPrincipal(unknown, cfg, admitted)).toBeUndefined();
    expect(resolveDurableSessionPrincipal(unknown, cfg, undefined)).toBeUndefined();
  }
});

test("environment admission must match and rotating or expiring a secret drops continuity", () => {
  const cfg = config();
  process.env.OPENCODEX_API_AUTH_TOKEN = "ocx_fixture_environment_before";
  const req = request({ authorization: `Bearer ${process.env.OPENCODEX_API_AUTH_TOKEN}` });
  const admission = resolveApiAuth(req, cfg)!;
  const original = resolveDurableSessionPrincipal(req, cfg, admission);
  process.env.OPENCODEX_API_AUTH_TOKEN = "ocx_fixture_environment_after";
  expect(resolveDurableSessionPrincipal(req, cfg, admission)).toBeUndefined();
  const next = request({ authorization: `Bearer ${process.env.OPENCODEX_API_AUTH_TOKEN}` });
  expect(resolveDurableSessionPrincipal(next, cfg, resolveApiAuth(next, cfg)!)).not.toBe(original);
  expect(resolveDurableSessionPrincipal(next, cfg, admission)).toBeUndefined();
  cfg.apiKeys![0]!.pendingRotation = { key: "ocx_fixture_pending", expiresAt: new Date(Date.now() + 60_000).toISOString() };
  const current = request({ "x-opencodex-api-key": keyA });
  const currentAdmission = resolveApiAuth(current, cfg)!;
  const pending = request({ "x-opencodex-api-key": "ocx_fixture_pending" });
  const pendingAdmission = resolveApiAuth(pending, cfg)!;
  expect(resolveDurableSessionPrincipal(pending, cfg, pendingAdmission)).not.toBe(resolveDurableSessionPrincipal(current, cfg, currentAdmission));
  expect(resolveDurableSessionPrincipal(pending, cfg, currentAdmission)).toBeUndefined();
  cfg.apiKeys![0]!.pendingRotation.expiresAt = new Date(0).toISOString();
  expect(resolveApiAuth(pending, cfg)).toBeNull();
  expect(resolveDurableSessionPrincipal(pending, cfg, pendingAdmission)).toBeUndefined();
});

test("the existing process-local and structural ownership contracts stay independent", () => {
  const cfg = config();
  const req = request({ "x-opencodex-api-key": keyA });
  const admission = resolveApiAuth(req, cfg)!;
  const processLocal = resolveContextPrincipal(req, cfg, admission);
  const signingKey = new Uint8Array(32).fill(7);
  const structural = resolveDurableReplayPrincipal(req, cfg, admission, signingKey);
  const durable = resolveDurableSessionPrincipal(req, cfg, admission);
  expect(new Set([processLocal, structural, durable]).size).toBe(3);
  expect(resolveContextPrincipal(req, cfg, admission)).toBe(processLocal);
  expect(resolveDurableReplayPrincipal(req, cfg, admission, signingKey)).toBe(structural);
});

test("WebSocket metadata retains the trusted owner after local credentials are filtered", () => {
  const cfg = config();
  const req = request({ "x-opencodex-api-key": keyA, "thread-id": "own-session" });
  const admission = resolveResponsesApiAuth(req, cfg)!;
  const principal = resolveDurableSessionPrincipal(req, cfg, admission);
  const headers = selectForwardHeaders(req.headers);
  expect(headers.has("x-opencodex-api-key")).toBe(false);
  const data = buildResponsesWsData(headers, admission, undefined, undefined, principal);
  expect(data.durableClientPrincipalId).toBe(principal);
  expect(data.admission).toBe(admission);
});

test.each(["responses", "chat", "messages", "combo", "captured-unknown"] as const)("%s runTurn preserves the original trusted owner", async surface => {
  const priorHome = process.env.OPENCODEX_HOME;
  const home = mkdtempSync(join(tmpdir(), "ocx-session-principal-"));
  process.env.OPENCODEX_HOME = home;
  const definition = getAdapterDefinition("claude-cli")!;
  const mutable = definition as { create: typeof definition.create };
  const originalCreate = mutable.create;
  const originalFetch = globalThis.fetch;
  let releaseSpend = () => {};
  const seen: Array<string | undefined> = [];
  try {
    clearResponseStateForTests();
    releaseSpend = acquireOwnedSpendHome();
    setClaudeUsagePreflightForTests(async () => ({ state: "available", checkedAt: 0 }));
    globalThis.fetch = (() => { throw new Error("Principal tests must not call an upstream"); }) as typeof fetch;
    mutable.create = () => ({
      name: "claude-cli",
      buildRequest() { throw new Error("Principal tests use runTurn only"); },
      async *parseStream() {},
      async runTurn(_parsed, incoming, emit) {
        expect(incoming.headers.has("x-opencodex-api-key")).toBe(false);
        seen.push(incoming.durableClientPrincipalId);
        emit({ type: "text_delta", text: "fixture answer" });
        emit({ type: "done", usage: { inputTokens: 1, outputTokens: 1 } });
      },
    });
    const cfg = config();
    const responses = surface !== "chat" && surface !== "messages";
    if (surface === "combo") cfg.combos = { durable: { strategy: "failover", targets: [{ provider: "claude-cli", model: "claude-sonnet-5-5" }] } };
    const body = responses ? { input: "hello" } : { messages: [{ role: "user", content: "hello" }], max_tokens: 64 };
    const req = new Request(`http://localhost/v1/${responses ? "responses" : surface === "chat" ? "chat/completions" : surface}`, {
      method: "POST", headers: { "content-type": "application/json", "x-opencodex-api-key": keyA,
        authorization: "Bearer fixture", "thread-id": "own-session" },
      body: JSON.stringify({ model: surface === "combo" ? "combo/durable" : "claude-cli/claude-sonnet-5-5", stream: false, ...body }),
    });
    const admission = resolveResponsesApiAuth(req, cfg)!;
    const principal = resolveDurableSessionPrincipal(req, cfg, admission);
    expect(principal).toMatch(/^[0-9a-f]{64}$/);
    const expected = surface === "captured-unknown" ? undefined : principal;
    const log = { model: "", provider: "" };
    const ids = { requestId: `principal-${surface}`, start: Date.now(), admission };
    const response = responses ? await handleResponses(req, cfg, log, { admission,
      ...(surface === "captured-unknown" ? { durableClientPrincipalId: undefined } : {}),
    })
      : surface === "chat" ? await handleChatCompletions(req, cfg, log, ids)
      : await handleClaudeMessages(req, cfg, log, ids);
    const text = await response.text();
    expect(response.status, text).toBe(200);
    expect(seen).toEqual([expected]);
  } finally {
    mutable.create = originalCreate;
    globalThis.fetch = originalFetch;
    setClaudeUsagePreflightForTests();
    try {
      await flushResponseState();
      clearResponseStateForTests();
      await flushConfigDirHardeningForTests();
    } finally {
      try { releaseSpend(); await removeTreeAfterBackgroundWork(home); }
      finally {
        if (priorHome === undefined) delete process.env.OPENCODEX_HOME;
        else process.env.OPENCODEX_HOME = priorHome;
      }
    }
  }
}, 15_000);
