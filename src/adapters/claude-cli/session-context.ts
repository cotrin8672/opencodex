import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { getConfigDir } from "../../config/paths";
import { isTestHomeGuardArmed } from "../../lib/test-home-guard";
import { namespacedToolName, type OcxMessage, type OcxParsedRequest, type OcxProviderConfig } from "../../types";
import type { IncomingMeta } from "../base";
import { canonicalClaudeJson } from "./stable-replay";

export function claudeDigest(value: unknown): string {
  // Parsed requests can contain optional undefined properties, unlike their JSON wire form.
  const json = JSON.parse(JSON.stringify(value));
  return createHash("sha256").update(JSON.stringify(canonicalClaudeJson(json))).digest("hex");
}

/** A parent thread and a prompt-cache cohort must never merge parallel child conversations. */
export function claudeSessionScope(parsed: OcxParsedRequest, incoming: IncomingMeta, provider: OcxProviderConfig): string | undefined {
  const thread = [parsed._codexOwnThreadId, incoming.headers.get("thread-id"),
    incoming.headers.get("session_id"), incoming.headers.get("session-id")].find(value => value?.trim());
  if (!thread?.trim() || thread.length > 4096 || /[\x00-\x1f\x7f]/.test(thread)) return undefined;
  if ("durableClientPrincipalId" in incoming && !incoming.durableClientPrincipalId) return undefined;
  const principal = incoming.durableClientPrincipalId ?? parsed._reasoningReplayScope?.clientPrincipalId ?? parsed._cursorIdentityScope
    ?? claudeDigest([incoming.headers.get("authorization"), incoming.headers.get("x-api-key")]);
  return claudeDigest([
    getConfigDir(), principal, incoming.providerName ?? "claude-cli", provider.baseUrl,
    thread,
    parsed._compactionRequest === true ? "compaction" : "conversation",
  ]);
}

export function claudeAccountIdentity(): string | undefined {
  if (isTestHomeGuardArmed()) return undefined;
  const path = process.env.CLAUDE_CONFIG_DIR?.trim();
  try {
    const account = JSON.parse(readFileSync(path ? join(path, ".claude.json") : join(homedir(), ".claude.json"), "utf8"))?.oauthAccount?.accountUuid;
    return typeof account === "string" ? claudeDigest(account) : undefined;
  } catch { return undefined; }
}

type HistoryItem = { role: string; content: unknown[]; callId?: string; isError?: boolean };

function textParts(content: unknown): unknown[] {
  if (typeof content === "string") return [{ type: "text", text: content }];
  return content as unknown[];
}

/** Ignore transport timestamps, signatures and the client's optional thinking display. */
export function claudeHistory(messages: OcxMessage[]): string[] {
  const items: HistoryItem[] = [];
  for (const message of messages) {
    if (message.role === "developer") continue;
    if (message.role !== "assistant") {
      items.push({ role: message.role, content: textParts(message.content),
        ...(message.role === "toolResult" ? { callId: message.toolCallId, isError: message.isError === true } : {}) });
      continue;
    }
    const content: Array<{ type: string; text?: string; id?: string; name?: string; arguments?: unknown }> = [];
    for (const part of message.content) {
      if (part.type === "thinking") continue;
      if (part.type === "text") {
        if (!part.text) continue;
        const previous = content.at(-1);
        if (previous?.type === "text") previous.text += part.text;
        else content.push({ type: "text", text: part.text });
      } else if (part.type === "toolCall") {
        content.push({ type: "toolCall", id: part.id, name: namespacedToolName(part.namespace, part.name), arguments: part.arguments ?? {} });
      }
    }
    if (!content.length) continue;
    const previous = items.at(-1);
    if (previous?.role === "assistant") {
      const tail = previous.content.at(-1) as { type: string; text?: string } | undefined;
      const first = content[0];
      if (tail?.type === "text" && first?.type === "text") { tail.text += first.text!; content.shift(); }
      previous.content.push(...content);
    } else items.push({ role: "assistant", content });
  }
  return items.map(claudeDigest);
}

export function historyExtends(history: string[], prefix: string[]): boolean {
  return history.length >= prefix.length && prefix.every((hash, index) => history[index] === hash);
}

export function claudeUserText(message: OcxMessage): string {
  if (typeof message.content === "string") return message.content;
  return message.content.map(part => "text" in part ? part.text : `[${part.type}]`).join("\n");
}

export function claudeResult(message: OcxMessage): { hash: string; content: string; isError: boolean } {
  if (message.role !== "toolResult") throw new Error("Expected an external tool result.");
  const content = claudeUserText(message);
  const isError = message.isError === true;
  return { hash: claudeDigest([content, isError]), content, isError };
}
