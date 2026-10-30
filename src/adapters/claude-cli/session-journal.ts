import { Database } from "bun:sqlite";
import { chmodSync, closeSync, lstatSync, openSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import * as z from "zod/v4";
import { getConfigDir } from "../../config/paths";
import { assertNotRealHomeUnderTest } from "../../lib/test-home-guard";
import { hardenSecretDir, hardenSecretPath, windowsSecretAclApplies } from "../../lib/windows-secret-acl";
import type { AdapterEvent } from "../../types";

export interface ClaudeToolReceipt {
  id: string;
  name: string;
  arguments: string;
  result?: { hash: string; content: string; isError: boolean };
}

export interface ClaudeTurnReceipt {
  inputHash: string;
  requestHash?: string;
  operationId?: string;
  historyHash?: string;
  phase: "running" | "tools" | "completed" | "interrupted" | "expired";
  events: AdapterEvent[];
  calls: ClaudeToolReceipt[];
  updatedAt: number;
  deliveryStarted?: boolean;
}

export interface ClaudeSessionSnapshot {
  version: 1;
  history: string[];
  turns: ClaudeTurnReceipt[];
}

export interface ClaudeSessionJournal {
  load(scope: string): ClaudeSessionSnapshot | undefined;
  save(scope: string, snapshot: ClaudeSessionSnapshot): void;
  close(): void;
}

const MAX_RECORD_BYTES = 16 * 1024 * 1024;
const hash = z.string().length(64).regex(/^[0-9a-f]+$/i);
const count = z.number().nonnegative().max(Number.MAX_SAFE_INTEGER);
const usage = z.strictObject({
  inputTokens: count, outputTokens: count,
  providerCredits: count.optional(), contextTotalTokens: count.optional(), totalTokens: count.optional(),
  cachedInputTokens: count.optional(), cacheReadInputTokens: count.optional(),
  cacheCreationInputTokens: count.optional(), cacheCreation1hInputTokens: count.optional(),
  reasoningOutputTokens: count.optional(), estimated: z.boolean().optional(),
  upstreamReported: z.literal(true).optional(), rawUsage: z.record(z.string(), z.json()).optional(),
});
const providerState = z.object({
  cursor: z.object({
    conversationId: z.string().optional(), checkpointUsable: z.boolean().optional(), checkpointRef: z.string().optional(),
  }).catchall(z.json()).optional(),
  kiro: z.object({ conversationId: z.string().optional() }).catchall(z.json()).optional(),
  __ocxOwner: z.object({
    version: z.literal(1), providerName: z.string(), providerDestinationIdentity: z.string(),
    adapterName: z.string(), modelId: z.string(), credentialIdentity: z.string(),
  }).catchall(z.union([z.string(), z.number()])).optional(),
}).catchall(z.record(z.string(), z.json()).optional());
const terminal = { usage: usage.optional(), endTurn: z.boolean().optional(), providerState: providerState.optional() };
const eventSchemas = {
  heartbeat: z.strictObject({ type: z.literal("heartbeat"), replayUnsafe: z.literal(true).optional(), preflightReady: z.literal(true).optional() }),
  text_delta: z.strictObject({ type: z.literal("text_delta"), text: z.string(), phase: z.enum(["commentary", "final_answer"]).optional(), anthropicThinkingSignature: z.string().optional() }),
  thinking_delta: z.strictObject({ type: z.literal("thinking_delta"), thinking: z.string() }),
  thinking_signature: z.strictObject({ type: z.literal("thinking_signature"), signature: z.string() }),
  redacted_thinking: z.strictObject({ type: z.literal("redacted_thinking"), data: z.string() }),
  kiro_redacted_reasoning: z.strictObject({ type: z.literal("kiro_redacted_reasoning"), data: z.string() }),
  reasoning_raw_delta: z.strictObject({ type: z.literal("reasoning_raw_delta"), text: z.string() }),
  tool_call_start: z.strictObject({ type: z.literal("tool_call_start"), id: z.string(), name: z.string(), providerMetadata: z.strictObject({ google: z.strictObject({ thoughtSignature: z.string().optional() }).optional() }).optional() }),
  tool_call_delta: z.strictObject({ type: z.literal("tool_call_delta"), arguments: z.string() }),
  tool_call_end: z.strictObject({ type: z.literal("tool_call_end"), incomplete: z.literal(true).optional() }),
  assistant_boundary: z.strictObject({ type: z.literal("assistant_boundary") }),
  web_search_call_begin: z.strictObject({ type: z.literal("web_search_call_begin"), id: z.string(), itemId: z.string().optional() }),
  web_search_call_end: z.strictObject({
    type: z.literal("web_search_call_end"), id: z.string(), itemId: z.string().optional(),
    queries: z.array(z.string()), status: z.enum(["completed", "failed"]).optional(),
    sources: z.array(z.strictObject({ url: z.string(), title: z.string().optional() })).optional(),
    anthropic: z.strictObject({ use: z.record(z.string(), z.json()), result: z.record(z.string(), z.json()) }).optional(),
  }),
  done: z.strictObject({ type: z.literal("done"), ...terminal, compactionEncryptedContent: z.string().optional(), stopReason: z.string().optional() }),
  incomplete: z.strictObject({ type: z.literal("incomplete"), ...terminal, reason: z.string(), message: z.string().optional(), retryable: z.boolean().optional() }),
  error: z.strictObject({
    type: z.literal("error"), message: z.string(), usage: usage.optional(), status: z.number().int().min(100).max(599).optional(),
    errorType: z.string().optional(), code: z.string().optional(), apiKeyRecovery: z.literal(true).optional(),
    retryable: z.boolean().optional(), replaySafeBeforeOutput: z.boolean().optional(),
  }),
} satisfies { [Type in AdapterEvent["type"]]: z.ZodType<Extract<AdapterEvent, { type: Type }>> };
const snapshotSchema: z.ZodType<ClaudeSessionSnapshot> = z.strictObject({
  version: z.literal(1), history: z.array(hash),
  turns: z.array(z.strictObject({
    inputHash: hash, requestHash: hash.optional(), operationId: hash.optional(), historyHash: hash.optional(), phase: z.enum(["running", "tools", "completed", "interrupted", "expired"]),
    events: z.array(z.union(Object.values(eventSchemas))),
    calls: z.array(z.strictObject({
      id: z.string().min(1), name: z.string().min(1), arguments: z.string(),
      result: z.strictObject({ hash, content: z.string(), isError: z.boolean() }).optional(),
    })),
    updatedAt: z.number().int().nonnegative(),
    deliveryStarted: z.boolean().optional(),
  })),
});

function validateSnapshot(value: unknown): ClaudeSessionSnapshot {
  const parsed = snapshotSchema.safeParse(value);
  if (!parsed.success) throw new Error("Invalid Claude session journal snapshot");
  return parsed.data;
}

function hardenFiles(path: string): void {
  for (const suffix of ["", "-wal", "-shm", "-journal"]) {
    const file = path + suffix;
    let stat;
    try { stat = lstatSync(file); }
    catch (error) {
      if (suffix && (error as NodeJS.ErrnoException).code === "ENOENT") continue;
      throw error;
    }
    if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1
      || (process.platform !== "win32" && stat.uid !== process.getuid!())) {
      throw new Error("Unsafe Claude session journal file");
    }
    if (windowsSecretAclApplies()) {
      if (!hardenSecretPath(file, { required: true }).ok) throw new Error("Claude session journal ACL refused");
    } else chmodSync(file, 0o600);
  }
}

export function openClaudeSessionJournal(path = join(getConfigDir(), "claude-cli-sessions.sqlite")): ClaudeSessionJournal {
  path = resolve(path);
  const dir = dirname(path);
  assertNotRealHomeUnderTest(dir);
  let database: Database | undefined;
  try {
    const stat = lstatSync(dir);
    if (!stat.isDirectory() || stat.isSymbolicLink()
      || (process.platform !== "win32" && stat.uid !== process.getuid!())) {
      throw new Error("Unsafe Claude session journal directory");
    }
    if (windowsSecretAclApplies()) {
      if (!hardenSecretDir(dir, { required: true }).ok) throw new Error("Claude session journal ACL refused");
    } else chmodSync(dir, 0o700);
    try { closeSync(openSync(path, "wx", 0o600)); }
    catch (error) { if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error; }
    hardenFiles(path);
    database = new Database(path, { create: true, strict: true });
    if (database.query<{ journal_mode: string }, []>("PRAGMA journal_mode = WAL").get()?.journal_mode !== "wal") {
      throw new Error("Claude session journal WAL unavailable");
    }
    database.exec(`PRAGMA synchronous = FULL; PRAGMA busy_timeout = 0;
      CREATE TABLE IF NOT EXISTS claude_sessions (scope TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT, WITHOUT ROWID`);
    hardenFiles(path);
    const db = database;
    const read = db.query<{ payload: unknown }, [number, string]>(
      "SELECT CASE WHEN length(CAST(payload AS BLOB)) <= ? THEN payload END AS payload FROM claude_sessions WHERE scope = ?",
    );
    const write = db.query("INSERT INTO claude_sessions (scope, payload) VALUES (?, ?) ON CONFLICT(scope) DO UPDATE SET payload = excluded.payload");
    const save = db.transaction((scope: string, payload: string) => {
      // BEGIN IMMEDIATE creates any sidecars before private receipt bytes are written.
      hardenFiles(path);
      write.run(scope, payload);
    });
    return {
      load(scope) {
        if (!hash.safeParse(scope).success) throw new Error("Invalid Claude session journal scope");
        try {
          const row = read.get(MAX_RECORD_BYTES, scope);
          if (!row) return undefined;
          if (typeof row.payload !== "string" || Buffer.byteLength(row.payload, "utf8") > MAX_RECORD_BYTES) {
            throw new Error("Invalid Claude session journal record size");
          }
          return validateSnapshot(JSON.parse(row.payload));
        } catch { throw new Error("Claude session journal load failed"); }
      },
      save(scope, snapshot) {
        if (!hash.safeParse(scope).success) throw new Error("Invalid Claude session journal scope");
        try {
          const payload = JSON.stringify(validateSnapshot(snapshot));
          if (Buffer.byteLength(payload, "utf8") > MAX_RECORD_BYTES) throw new Error("Claude session journal record too large");
          save.immediate(scope, payload);
        } catch { throw new Error("Claude session journal save failed"); }
      },
      close() { db.close(); },
    };
  } catch {
    try { database?.close(); } catch { /* preserve the opening failure */ }
    throw new Error("Claude session journal open failed");
  }
}
