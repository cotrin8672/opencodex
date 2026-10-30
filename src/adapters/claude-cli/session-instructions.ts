import type { OcxMessage } from "../../types";
import { claudeHistory, historyExtends } from "./session-context";
import { ClaudeSessionError } from "./session-errors";
import type { ClaudeQueuedInstruction } from "./session-journal";

function prefixBoundary(messages: OcxMessage[], history: string[]): number | undefined {
  if (!history.length) return 0;
  let count = 0;
  let previousAssistant = false;
  for (let end = 1; end <= messages.length; end++) {
    const message = messages[end - 1]!;
    if (message.role === "developer") continue;
    if (message.role === "assistant") {
      if (!message.content.some(part => part.type === "toolCall" || (part.type === "text" && part.text.length > 0))) continue;
      if (!previousAssistant) count++;
      previousAssistant = true;
    } else {
      count++;
      previousAssistant = false;
    }
    if (count > history.length) break;
    if (count !== history.length) continue;
    const prefix = claudeHistory(messages.slice(0, end));
    if (historyExtends(prefix, history)) return end;
  }
  return undefined;
}

function userText(message: OcxMessage): string {
  if (message.role !== "user") return "";
  return typeof message.content === "string" ? message.content
    : message.content.filter(part => part.type === "text").map(part => part.text).join("\n");
}

function unprovenHistory(): never {
  throw new ClaudeSessionError("The accepted instruction's history is missing or out of order; supply its original history before continuing.",
    "claude_instruction_history_missing", 409, true);
}

/** Compare raw boundaries because adjacent assistant fragments collapse in claudeHistory. */
export function extractClaudeInstructions(messages: OcxMessage[], prefixHistory: string[]): string[] | undefined {
  const start = prefixBoundary(messages, prefixHistory);
  if (start === undefined) return undefined;
  const texts = messages.slice(start).map(userText).filter(text => text.length > 0);
  return texts.length ? texts : undefined;
}

/** Unproven pending histories fail closed; callers must retain their journal entries for recovery. */
export function restoreClaudeInstructions(messages: OcxMessage[], entries: ClaudeQueuedInstruction[]): OcxMessage[] {
  const restored = [...messages];
  const seen = new Set<string>();
  const matched = new Set<OcxMessage>();
  let cursor = 0;
  let floor = 0;
  let missing: OcxMessage[] = [];
  const insertionPoint = (): number => {
    for (let index = Math.max(cursor, floor); index < restored.length; index++) {
      if (restored[index]!.role === "user" || restored[index]!.role === "assistant") return index;
    }
    return restored.length;
  };
  const insert = (index: number): number => {
    const count = missing.length;
    for (const message of missing) matched.add(message);
    restored.splice(index, 0, ...missing);
    missing = [];
    cursor = index + count;
    floor = 0;
    return count;
  };
  for (const entry of entries) {
    if (seen.has(entry.inputHash)) continue;
    seen.add(entry.inputHash);
    let start = prefixBoundary(restored, entry.history);
    if (start === undefined && missing.length) {
      // A later accepted history may include an earlier instruction omitted by the client.
      insert(insertionPoint());
      start = prefixBoundary(restored, entry.history);
    }
    if (start === undefined) {
      if (entry.phase === "completed") continue;
      unprovenHistory();
    }
    for (const text of entry.texts) {
      const match = restored.findIndex((message, index) => index >= Math.max(start, cursor, floor) && userText(message) === text);
      if (match >= 0) {
        const count = missing.length ? insert(Math.min(match, insertionPoint())) : 0;
        cursor = match + count + 1;
        matched.add(restored[cursor - 1]!);
      } else {
        if (restored.some((message, index) => index >= start && userText(message) === text && !matched.has(message))) unprovenHistory();
        floor = Math.max(floor, start);
        missing.push({ role: "user", content: text, timestamp: entry.updatedAt });
      }
    }
  }
  if (missing.length) insert(insertionPoint());
  return restored;
}
