import type { forkSession, getSessionMessages, SessionMessage } from "@anthropic-ai/claude-agent-sdk";
import { randomUUID } from "node:crypto";
import { claudeDigest, claudeHistory } from "./session-context";
import { ClaudeSessionError } from "./session-errors";
import { claudeHistoryBoundary } from "./session-instructions";
import { claudeNativeSessionSchema, type ClaudeNativeSession } from "./session-journal";
import type { OcxMessage } from "../../types";

export interface ClaudeNativeSessionIO {
  getSessionMessages: typeof getSessionMessages;
  forkSession: typeof forkSession;
}

const sdkIO: ClaudeNativeSessionIO = {
  // Public helpers use the process-global CLI profile, just as the production worker does.
  async getSessionMessages(id, options) { return (await import("@anthropic-ai/claude-agent-sdk")).getSessionMessages(id, options); },
  async forkSession(id, options) { return (await import("@anthropic-ai/claude-agent-sdk")).forkSession(id, options); },
};
const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function refused(): never {
  throw new ClaudeSessionError("Claude native history could not be verified.", "claude_native_history_invalid", 503, true);
}

function verifyMessages(messages: SessionMessage[], sessionId: string): void {
  const seen = new Set<string>();
  for (const message of messages) {
    if (!uuid.test(message.uuid) || seen.has(message.uuid) || message.session_id !== sessionId
      || !["user", "assistant", "system"].includes(message.type) || !message.message || typeof message.message !== "object"
      || message.parent_tool_use_id != null || message.parent_agent_id != null) refused();
    seen.add(message.uuid);
  }
}

/** Receipts establish completion; the transcript additionally must have no unresolved tool calls. */
function safeBoundaries(messages: SessionMessage[]): Set<string> {
  const safe = new Set<string>();
  const pending = new Set<string>();
  for (const message of messages) {
    const body = message.message as { content?: unknown; stop_reason?: unknown };
    if (Array.isArray(body.content)) for (const block of body.content) {
      if (!block || typeof block !== "object") continue;
      if (block.type === "tool_use") {
        if (typeof block.id !== "string" || pending.has(block.id)) refused();
        pending.add(block.id);
      } else if (block.type === "tool_result") {
        if (typeof block.tool_use_id !== "string" || !pending.delete(block.tool_use_id)) refused();
      }
    }
    if (message.type === "assistant" && !pending.size && body.stop_reason !== "tool_use"
      && body.stop_reason !== "max_tokens" && body.stop_reason !== "pause_turn") safe.add(message.uuid);
  }
  return safe;
}

/** The caller must await the previous worker's close/flush before planning native history. */
export async function planClaudeNativeSession(
  input: { record?: ClaudeNativeSession; messages: OcxMessage[]; identity: string; directory: string },
  io: ClaudeNativeSessionIO = sdkIO,
  signal?: AbortSignal,
): Promise<{ native: ClaudeNativeSession; start: number; resume?: string }> {
  signal?.throwIfAborted();
  const { messages, identity, directory } = input;
  const fresh = (): { native: ClaudeNativeSession; start: number } => ({
    native: claudeNativeSessionSchema.parse({ sessionId: randomUUID(), directory, identity, history: [], checkpoints: [], dirty: false }), start: 0,
  });
  const record = input.record && claudeNativeSessionSchema.parse(input.record);
  if (!record || record.identity !== identity || record.directory !== directory) return fresh();
  const history = claudeHistory(messages);
  let matched = 0;
  while (matched < record.history.length && history[matched] === record.history[matched]) matched++;
  const eligible = record.checkpoints.filter(checkpoint => checkpoint.length <= matched)
    .sort((a, b) => b.length - a.length);
  if (!eligible.length) return fresh();
  try {
    const source = await io.getSessionMessages(record.sessionId, { dir: directory });
    signal?.throwIfAborted();
    if (!source.length) return fresh();
    verifyMessages(source, record.sessionId);
    const safe = safeBoundaries(source);
    const checkpoint = eligible.find(point => safe.has(point.messageId));
    if (!checkpoint) return fresh();
    const start = claudeHistoryBoundary(messages, record.history.slice(0, checkpoint.length));
    if (start === undefined) return fresh();
    const end = source.findIndex(message => message.uuid === checkpoint.messageId);
    if (!record.dirty && matched === record.history.length && checkpoint.length === matched && end === source.length - 1) {
      return { native: structuredClone(record), start, resume: record.sessionId };
    }
    const { sessionId } = await io.forkSession(record.sessionId, { dir: directory, upToMessageId: checkpoint.messageId });
    signal?.throwIfAborted();
    if (!uuid.test(sessionId) || sessionId === record.sessionId) refused();
    const fork = await io.getSessionMessages(sessionId, { dir: directory });
    signal?.throwIfAborted();
    verifyMessages(fork, sessionId);
    if (fork.length !== end + 1) refused();
    const mapping = new Map<string, string>();
    const sourceIds = new Set(source.map(message => message.uuid));
    for (let index = 0; index < fork.length; index++) {
      const before = source[index]!;
      const after = fork[index]!;
      if (sourceIds.has(after.uuid) || before.type !== after.type
        || before.parent_tool_use_id !== after.parent_tool_use_id || before.parent_agent_id !== after.parent_agent_id
        || claudeDigest(before.message) !== claudeDigest(after.message)) refused();
      mapping.set(before.uuid, after.uuid);
    }
    const checkpoints = record.checkpoints.filter(point => point.length <= checkpoint.length).map(point => {
      const messageId = mapping.get(point.messageId);
      if (!messageId || !safe.has(point.messageId)) refused();
      return { length: point.length, messageId };
    });
    return { native: { ...record, sessionId, history: record.history.slice(0, checkpoint.length), checkpoints, dirty: false }, start, resume: sessionId };
  } catch (error) {
    signal?.throwIfAborted();
    if (error instanceof ClaudeSessionError) throw error;
    throw new ClaudeSessionError("Claude native history is unavailable.", "claude_native_history_unavailable", 503, true);
  }
}
