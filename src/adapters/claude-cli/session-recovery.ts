import { namespacedToolName, type OcxMessage, type OcxToolCall } from "../../types";
import { claudeDigest, claudeResult } from "./session-context";
import { ClaudeSessionError } from "./session-errors";
import type { ClaudeSessionSnapshot, ClaudeTurnReceipt } from "./session-journal";

/** changed tracks receipt edits; the manager commits before adoption and checks for missing results. */
export function prepareClaudeToolResults(snapshot: ClaudeSessionSnapshot, messages: OcxMessage[]): {
  snapshot: ClaudeSessionSnapshot;
  messages: OcxMessage[];
  pendingTurns: ClaudeTurnReceipt[];
  recoveredCalls: Set<string>;
  changed: boolean;
} {
  snapshot = structuredClone(snapshot);
  messages = structuredClone(messages);
  const results = new Map<string, ReturnType<typeof claudeResult>>();
  const origins = new Map<string, Array<{ index: number; call: OcxToolCall }>>();
  const conflict = (): never => {
    throw new ClaudeSessionError("A tool result changed for an already recorded call ID.", "claude_tool_result_conflict");
  };
  const sameResult = (left: ReturnType<typeof claudeResult>, right: ReturnType<typeof claudeResult>): boolean =>
    left.hash === right.hash && left.content === right.content && left.isError === right.isError;

  for (const [index, message] of messages.entries()) {
    if (message.role === "toolResult") {
      const result = claudeResult(message);
      const previous = results.get(message.toolCallId);
      if (previous && !sameResult(previous, result)) conflict();
      results.set(message.toolCallId, result);
    } else if (message.role === "assistant") {
      for (const call of message.content) {
        if (call.type !== "toolCall") continue;
        const matches = origins.get(call.id) ?? [];
        matches.push({ index, call });
        origins.set(call.id, matches);
      }
    }
  }

  const pendingTurns = snapshot.turns.filter(turn => turn.phase === "tools");
  const recoveredCalls = new Set<string>();
  const restored = new Map<number, OcxMessage[]>();
  let changed = false;
  for (const turn of snapshot.turns) for (const call of turn.calls) {
    const matches = origins.get(call.id) ?? [];
    // A running/interrupted receipt can contain an unfinished argument buffer.
    let signature: string | undefined;
    if (matches.length || (turn.phase === "tools" && call.result)) {
      signature = claudeDigest([call.name, JSON.parse(call.arguments)]);
      for (const origin of matches) {
        if (claudeDigest([namespacedToolName(origin.call.namespace, origin.call.name), origin.call.arguments ?? {}]) !== signature) {
          throw new ClaudeSessionError("Historical tool identity disagrees with its recorded call ID.", "claude_tool_id_conflict");
        }
      }
    }
    const result = results.get(call.id);
    if (result) {
      if (call.result && !sameResult(call.result, result)) conflict();
      if (!call.result) { call.result = result; changed = true; }
    }
    if ((result || (turn.phase === "tools" && call.result)) && !turn.deliveryStarted) {
      turn.deliveryStarted = true;
      changed = true;
    }
    const restoreCompleted = turn.phase === "completed" && !result && matches.length > 0;
    if ((turn.phase !== "tools" && !restoreCompleted) || !call.result) continue;
    recoveredCalls.add(signature ?? claudeDigest([call.name, JSON.parse(call.arguments)]));
    if (result) continue;
    const origin = matches[0];
    if (matches.length !== 1 || !origin) {
      throw new ClaudeSessionError("The original assistant tool call is missing or ambiguous; its saved result cannot be safely restored.", "claude_recovery_history_missing");
    }
    const additions = restored.get(origin.index) ?? [];
    additions.push({
      role: "toolResult", toolCallId: call.id,
      toolName: origin.call.namespace ? origin.call.name : call.name,
      ...(origin.call.namespace ? { toolNamespace: origin.call.namespace } : {}),
      content: call.result.content, isError: call.result.isError, timestamp: turn.updatedAt,
    });
    restored.set(origin.index, additions);
  }

  return {
    snapshot, messages: messages.flatMap((message, index) => [message, ...(restored.get(index) ?? [])]),
    pendingTurns, recoveredCalls, changed,
  };
}
