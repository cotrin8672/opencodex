import { describe, expect, test } from "bun:test";
import type { OcxMessage } from "../../src/types";
import { buildClaudeSessionBootstrap, buildClaudeSessionCatchUp, claudeHistory, claudeOperationIdentity, claudeSessionScope } from "../../src/adapters/claude-cli/session-context";
import { buildCodeBuddyToolBridge } from "../../src/adapters/codebuddy/tool-bridge";
import { assistant, result, turn, user } from "../fixtures/claude-session-worker";

describe("Claude stable operation identity", () => {
  test("separates operations A/B without changing thread/provider/principal scope", () => {
    const a = turn(), b = turn();
    a.incoming.headers.delete("idempotency-key");
    b.incoming.headers.delete("idempotency-key");
    a.parsed._rawBody = { client_metadata: { turn_id: "operation-A" } };
    b.parsed._rawBody = { client_metadata: { turn_id: "operation-B" } };
    const identity = claudeOperationIdentity(a.parsed, a.incoming);
    expect(identity).toMatch(/^[a-f0-9]{64}$/);
    expect(identity).not.toBe(claudeOperationIdentity(b.parsed, b.incoming));
    expect(claudeSessionScope(a.parsed, a.incoming, a.provider)).toBe(claudeSessionScope(b.parsed, b.incoming, b.provider));
    a.incoming.headers.set("x-client-request-id", "request-2");
    a.incoming.headers.set("x-request-id", "request-3");
    a.incoming.headers.set("x-codex-turn-state", "server-state-2");
    expect(claudeOperationIdentity(a.parsed, a.incoming)).toBe(identity);
  });

  test("prefers an explicit idempotency key, then body turn ID, then header turn ID", () => {
    const input = turn();
    input.incoming.headers.delete("idempotency-key");
    input.incoming.headers.set("x-codex-turn-metadata", JSON.stringify({ turn_id: "header-turn" }));
    const header = claudeOperationIdentity(input.parsed, input.incoming);
    input.parsed._rawBody = { client_metadata: { turn_id: "header-turn" } };
    expect(claudeOperationIdentity(input.parsed, input.incoming)).toBe(header);
    input.parsed._rawBody = { client_metadata: { turn_id: "body-turn" } };
    const body = claudeOperationIdentity(input.parsed, input.incoming);
    expect(body).not.toBe(header);
    input.incoming.headers.set("idempotency-key", "explicit-operation");
    const key = claudeOperationIdentity(input.parsed, input.incoming);
    expect(key).not.toBe(body);
    input.parsed._rawBody = { client_metadata: { turn_id: "changed-turn" } };
    input.incoming.headers.set("x-codex-turn-metadata", "malformed");
    expect(claudeOperationIdentity(input.parsed, input.incoming)).toBe(key);
    input.incoming.headers.set("idempotency-key", "operation-B");
    expect(claudeOperationIdentity(input.parsed, input.incoming)).not.toBe(key);
  });

  test("rejects invalid body/header ID types, lengths and control characters", () => {
    const input = turn();
    input.incoming.headers.delete("idempotency-key");
    for (const id of [undefined, null, 7, false, {}, [], "", "  ", "x".repeat(4097), "a\nb", "a\0b", "a\u007fb", "a\u0085b"]) {
      input.parsed._rawBody = { client_metadata: { turn_id: id } };
      input.incoming.headers.set("x-codex-turn-metadata", JSON.stringify({ turn_id: id }));
      expect(claudeOperationIdentity(input.parsed, input.incoming)).toBeUndefined();
    }
  });

  test("bounds keys and falls back from invalid metadata without inventing an ID", () => {
    const input = turn();
    input.incoming.headers.set("idempotency-key", "x".repeat(4096));
    expect(claudeOperationIdentity(input.parsed, input.incoming)).toMatch(/^[a-f0-9]{64}$/);
    input.incoming.headers.set("idempotency-key", "x".repeat(4097));
    expect(claudeOperationIdentity(input.parsed, input.incoming)).toBeUndefined();
    input.parsed._rawBody = { client_metadata: { turn_id: 42 } };
    input.incoming.headers.set("x-codex-turn-metadata", JSON.stringify({ turn_id: "fallback" }));
    expect(claudeOperationIdentity(input.parsed, input.incoming)).toMatch(/^[a-f0-9]{64}$/);
  });

  test("missing/malformed metadata, request IDs and server turn state supply no operation ID", () => {
    const input = turn();
    input.incoming.headers.delete("idempotency-key");
    input.incoming.headers.set("x-client-request-id", "request-A");
    input.incoming.headers.set("x-request-id", "request-B");
    input.incoming.headers.set("x-codex-turn-state", "server-turn-A");
    for (const body of [undefined, null, "turn-A", {}, { client_metadata: null }, { client_metadata: "turn-A" }]) {
      input.parsed._rawBody = body;
      for (const header of ["", "{", "null", "[]", '"turn-A"', JSON.stringify({ turn_id: "x".repeat(8192) })]) {
        input.incoming.headers.set("x-codex-turn-metadata", header);
        expect(claudeOperationIdentity(input.parsed, input.incoming)).toBeUndefined();
      }
    }
  });
});

describe("Claude session bootstrap", () => {
  test("catch-up sends only the unseen foreign model history and current user input", () => {
    const input = turn([user("SEEN_USER_A"), assistant("SEEN_CLAUDE_A"), user("FOREIGN_USER_B"),
      assistant("FOREIGN_GPT_B"), user("CURRENT_USER_C")]);
    const text = buildClaudeSessionCatchUp(input.parsed, input.provider, input.bridge, 2);
    expect(text).not.toContain("SEEN_USER_A");
    expect(text).not.toContain("SEEN_CLAUDE_A");
    for (const marker of ["FOREIGN_USER_B", "FOREIGN_GPT_B", "CURRENT_USER_C"]) expect(text.split(marker)).toHaveLength(2);
    expect(text).toContain("ASSISTANT");
  });

  test("ordinary native continuation preserves the new user text verbatim", () => {
    const input = turn([user("prior"), assistant("answer"), user("/init @literal-file next")]);
    expect(buildClaudeSessionCatchUp(input.parsed, input.provider, input.bridge, 2)).toBe("/init @literal-file next");
  });

  test("projects historical same-name tools into distinct advertised MCP names", () => {
    const input = turn();
    input.parsed.context.tools = ["db_primary", "db_archive"].map(namespace => ({
      namespace, name: "lookup", description: namespace, parameters: { type: "object", properties: {} },
    }));
    input.bridge = buildCodeBuddyToolBridge(input.parsed);
    const bootstrap = (namespace: string) => {
      input.parsed.context.messages = [user("Look up account"), { role: "assistant", timestamp: 0, content: [
        { type: "toolCall", id: "old-call", namespace, name: "lookup", arguments: { z: 2, a: 1 } },
      ] }, result("old-call", "value"), user("Which database did you use?")];
      return buildClaudeSessionBootstrap(input.parsed, input.provider, input.bridge);
    };
    const primary = bootstrap("db_primary"), archive = bootstrap("db_archive");
    const names = new Map([...input.bridge.emittedNameMap].map(([emitted, wire]) => [wire, emitted]));
    expect(primary).toContain(`Tool call: ${names.get("db_primary__lookup")}`);
    expect(archive).toContain(`Tool call: ${names.get("db_archive__lookup")}`);
    expect(primary).not.toBe(archive);
    expect(primary).toContain('with args: {"a":1,"z":2}');
    input.bridge = { tools: [], emittedNameMap: new Map(), requireToolCall: false };
    expect(bootstrap("db_primary")).toContain("Tool call: db_primary__lookup");
    expect(bootstrap("db_archive")).toContain("Tool call: db_archive__lookup");
  });

  test("retains ordinary model-switch conversation text and the current input", () => {
    const input = turn([user("Remember ORCHID"), assistant("Remembered ORCHID"), user("What was the codeword?")]);
    input.parsed.modelId = "different-model";
    expect(buildClaudeSessionBootstrap(input.parsed, input.provider, input.bridge)).toBe(
      "USER:\nRemember ORCHID\n\nASSISTANT:\nRemembered ORCHID\n\nUSER:\nWhat was the codeword?",
    );
  });

  test("uses the model context window before the provider window and default history cap", () => {
    const input = turn([user("EARLY-MARKER" + "x".repeat(250_000)), assistant("prior response"), user("current input")]);
    const bootstrap = () => buildClaudeSessionBootstrap(input.parsed, input.provider, input.bridge);
    expect(bootstrap()).not.toContain("EARLY-MARKER");
    input.provider.contextWindow = 128_000;
    expect(bootstrap()).toContain("EARLY-MARKER");
    input.provider.modelContextWindows = { "offline-model": 64_000 };
    expect(bootstrap()).not.toContain("EARLY-MARKER");
    input.provider.modelContextWindows = { "offline-model": 1_000_000 };
    input.provider.contextWindow = 64_000;
    expect(bootstrap()).toContain("EARLY-MARKER");
    expect(bootstrap()).toContain("current input");
  });
});
