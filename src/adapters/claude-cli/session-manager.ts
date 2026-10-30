import type { AdapterEvent, OcxMessage, OcxParsedRequest, OcxProviderConfig } from "../../types";
import type { IncomingMeta } from "../base";
import type { CodeBuddyToolBridge } from "../codebuddy/tool-bridge";
import { CODEBUDDY_MCP_TOOL_PREFIX, CODEBUDDY_TOOL_LIMITS } from "../codebuddy/tool-bridge";
import { CodingAgentProtocolError, mapStreamMessageToEvents, projectedHistoryCharLimit, releaseOpenToolBlocks, type StreamParseState } from "../coding-agent/protocol";
import { modelRecordValue } from "../../reasoning-effort";
import { SendBudgetExhaustedError } from "../../lib/upstream-retry";
import { getConfigDir } from "../../config/paths";
import { createTranslatorBudget, isTranslatorBudgetExceededError } from "../../lib/translator-budget";
import { registerOptionalShutdownHook, didRunOptionalShutdownHooks } from "../../lib/optional-shutdown-hooks";
import { claudeAccountIdentity, claudeDigest, claudeHistory, claudeResult, claudeUserText, historyExtends } from "./session-context";
import { openClaudeSessionJournal, type ClaudeSessionJournal, type ClaudeSessionSnapshot, type ClaudeToolReceipt, type ClaudeTurnReceipt } from "./session-journal";
import { buildStableClaudeConversationInput } from "./stable-replay";
import type { ClaudeSessionWorker, ClaudeWorkerOptions, ClaudeWorkerToolCall } from "./session-worker";

export class ClaudeSessionError extends Error {
  constructor(message: string, readonly code: string, readonly status = 409, readonly retryable = false) { super(message); }
}

export interface ClaudePersistentTurn {
  scope: string;
  parsed: OcxParsedRequest;
  incoming: IncomingMeta;
  provider: OcxProviderConfig;
  binary: string;
  env: Record<string, string>;
  systemPrompt: string;
  bridge: CodeBuddyToolBridge;
  effort?: string;
  onCliOutputError?: (message: string) => void;
}

export interface ClaudeSessionManagerDeps {
  journal?: ClaudeSessionJournal;
  createWorker?: (options: ClaudeWorkerOptions) => Promise<ClaudeSessionWorker>;
  timeoutMs?: number;
  reconnectGraceMs?: number;
  idleMs?: number;
}

type Pending = {
  signature: string;
  promise: Promise<{ content: Array<{ type: "text"; text: string }>; isError?: boolean }>;
  resolve: (value: { content: Array<{ type: "text"; text: string }>; isError?: boolean }) => void;
  reject: (error: Error) => void;
};
type Consumer = { emit: (event: AdapterEvent) => void; signal?: AbortSignal; sent: number };
type Leg = {
  input?: ClaudePersistentTurn;
  receipt: ClaudeTurnReceipt;
  state: StreamParseState;
  openCall?: ClaudeToolReceipt;
  nativeNames: Map<string, string>;
  mcpIds: Set<string>;
  recoveredCalls: Set<string>;
  buffered: boolean;
  finished: boolean;
  bytes: number;
  done: Promise<void>;
  settle: () => void;
  timer?: ReturnType<typeof setTimeout>;
  release: () => void;
};
type Session = {
  scope: string;
  snapshot: ClaudeSessionSnapshot;
  worker?: ClaudeSessionWorker;
  configuration?: string;
  catalog?: string;
  active?: Leg;
  consumer?: Consumer;
  pending: Map<string, Pending>;
  idle?: ReturnType<typeof setTimeout>;
  busy: boolean;
};

/** One owner per conversation; a disconnected HTTP response does not own the Claude process. */
export class ClaudeSessionManager {
  private readonly sessions = new Map<string, Session>();
  private readonly journal: ClaudeSessionJournal;
  private closed = false;

  constructor(private readonly deps: ClaudeSessionManagerDeps = {}) {
    this.journal = deps.journal ?? openClaudeSessionJournal();
  }

  async run(input: ClaudePersistentTurn, emit: (event: AdapterEvent) => void): Promise<void> {
    if (this.closed) throw new ClaudeSessionError("Claude sessions are shutting down.", "claude_session_shutdown", 503);
    if (input.incoming.abortSignal?.aborted) return;
    let session = this.sessions.get(input.scope);
    if (!session) {
      if (this.sessions.size >= 16) {
        const idle = [...this.sessions.values()].find(owner => !owner.busy && !owner.consumer && (!owner.active || owner.active.finished));
        if (!idle) throw new ClaudeSessionError("All Claude conversations are busy; retry when one finishes.", "claude_worker_capacity", 503, true);
        if (idle.idle) clearTimeout(idle.idle);
        this.stopWorker(idle);
        this.sessions.delete(idle.scope);
      }
      session = { scope: input.scope, snapshot: this.journal.load(input.scope) ?? { version: 1, history: [], turns: [] }, pending: new Map(), busy: false };
      this.sessions.set(input.scope, session);
    }
    if (session.busy || session.consumer) throw new ClaudeSessionError("Another request still owns this Claude conversation.", "claude_session_busy", 409, true);
    session.busy = true;
    if (session.idle) clearTimeout(session.idle);
    let leg: Leg | undefined;
    try {
      const history = claudeHistory(input.parsed.context.messages);
      const configuration = claudeDigest([input.parsed.modelId, input.binary, input.systemPrompt, input.effort, input.env.CLAUDE_CONFIG_DIR, claudeAccountIdentity()]);
      const catalog = claudeDigest(input.bridge.tools);
      const inputHash = claudeDigest([history, configuration, catalog, input.parsed.options]);
      const historyHash = claudeDigest(history);
      if (session.snapshot.turns.some(turn => turn.historyHash === historyHash && turn.inputHash !== inputHash && turn.deliveryStarted)) {
        throw new ClaudeSessionError("This input already delivered an external action. Changing model or tool settings cannot reissue it.", "claude_tool_delivery_uncertain");
      }
      const previous = session.snapshot.turns.find(turn => turn.inputHash === inputHash);
      if (previous?.phase === "expired") throw new ClaudeSessionError("This old response is no longer cached; it will not be executed again.", "claude_replay_expired");
      if (previous?.phase === "completed" || previous?.phase === "tools") {
        this.deliverReceipt(session, previous, emit);
        return;
      }
      if (session.active && !session.active.finished) {
        if (session.active.receipt.inputHash !== inputHash) throw new ClaudeSessionError("Claude is still generating the preceding response.", "claude_session_busy", 409, true);
        leg = session.active;
      } else {
        const pendingTurns = session.snapshot.turns.filter(turn => turn.phase === "tools");
        let received = false;
        for (const turn of session.snapshot.turns) for (const call of turn.calls) {
          const matches = input.parsed.context.messages.filter(message => message.role === "toolResult" && message.toolCallId === call.id);
          for (const message of matches) {
            const result = claudeResult(message);
            if (call.result && call.result.hash !== result.hash) throw new ClaudeSessionError("A tool result changed for an already recorded call ID.", "claude_tool_result_conflict");
            call.result = result;
            turn.deliveryStarted = true;
            received = true;
          }
        }
        if (received) this.save(session);
        if (pendingTurns.some(turn => turn.calls.some(call => !call.result))) {
          throw new ClaudeSessionError("Claude is waiting for the existing tool results. An uncertain tool action will not be issued again.", "claude_tool_results_pending");
        }
        const last = input.parsed.context.messages.at(-1);
        const extendsHistory = historyExtends(history, session.snapshot.history);
        const nativeContinuation = session.worker && session.configuration === configuration && extendsHistory
          && (pendingTurns.length > 0 ? last?.role === "toolResult" : history.length === session.snapshot.history.length + 1 && last?.role === "user");
        if (!nativeContinuation) this.stopWorker(session);
        if (session.snapshot.turns.length >= 4096 && !previous) throw new ClaudeSessionError("This conversation reached its receipt limit; start a new conversation.", "claude_session_history_limit", 413);
        if (previous) session.snapshot.turns.splice(session.snapshot.turns.indexOf(previous), 1);
        leg = this.newLeg(session, input, inputHash, historyHash);
        if (!nativeContinuation) {
          const resultIds = new Set<string>();
          for (const message of [...input.parsed.context.messages].reverse()) {
            if (message.role !== "toolResult") break;
            resultIds.add(message.toolCallId);
          }
          for (const turn of session.snapshot.turns) for (const call of turn.calls) {
            if (call.result && resultIds.has(call.id)) leg.recoveredCalls.add(claudeDigest([call.name, JSON.parse(call.arguments)]));
          }
        }
        session.snapshot.history = history;
        this.save(session);
        if (!session.worker) {
          const createWorker = this.deps.createWorker ?? (await import("./session-worker")).createClaudeSessionWorker;
          const owner = session;
          const worker = await createWorker({
            binary: input.binary, model: input.parsed.modelId, env: input.env, systemPrompt: input.systemPrompt,
            effort: input.effort, tools: input.bridge.tools,
            onMessage: message => this.onMessage(owner, message),
            onToolCall: (call, signal) => this.onToolCall(owner, call, signal),
            onError: error => this.fail(owner, error),
          });
          if (leg.finished || this.closed) worker.stop();
          else {
            session.worker = worker;
            session.configuration = configuration;
            session.catalog = catalog;
          }
        } else if (session.catalog !== catalog) {
          await session.worker.updateTools(input.bridge.tools);
          session.catalog = catalog;
        }
        if (input.incoming.abortSignal?.aborted) {
          this.fail(session, new ClaudeSessionError("The request was cancelled before inference.", "claude_request_cancelled", 499));
          return;
        }
        if (!leg.finished) {
          if (!session.worker) throw new ClaudeSessionError("The Claude worker did not start.", "claude_worker_unavailable", 502);
          this.admit(leg);
          for (const turn of pendingTurns) turn.phase = "completed";
          this.save(session);
          if (nativeContinuation && pendingTurns.length) {
            for (const turn of pendingTurns) for (const call of turn.calls) {
              const pending = session.pending.get(call.id);
              if (!pending || !call.result) throw new ClaudeSessionError("The pending MCP call was lost before continuation.", "claude_mcp_call_lost", 502);
              pending.resolve({ content: [{ type: "text", text: call.result.content }], isError: call.result.isError });
            }
          } else {
            const text = nativeContinuation && last ? claudeUserText(last) : this.bootstrap(input);
            session.worker.send(text);
          }
        }
      }
    } catch (error) {
      if (leg && !leg.finished) this.fail(session, error instanceof Error ? error : new Error(String(error)));
      throw error;
    } finally { session.busy = false; if (!leg) this.armIdle(session); }
    if (!leg) return;
    const consumer: Consumer = { emit, signal: input.incoming.abortSignal, sent: 0 };
    session.consumer = consumer;
    this.publish(session);
    let abort!: () => void;
    const cancelled = new Promise<void>(resolve => { abort = resolve; });
    const onAbort = (): void => {
      if (session.consumer === consumer) session.consumer = undefined;
      abort();
    };
    input.incoming.abortSignal?.addEventListener("abort", onAbort, { once: true });
    try {
      if (input.incoming.abortSignal?.aborted) onAbort();
      await Promise.race([leg.done, cancelled]);
    } finally {
      input.incoming.abortSignal?.removeEventListener("abort", onAbort);
      if (session.consumer === consumer) session.consumer = undefined;
      this.armIdle(session);
    }
  }

  private bootstrap(input: ClaudePersistentTurn): string {
    const maxHistoryChars = projectedHistoryCharLimit(modelRecordValue(input.provider.modelContextWindows, input.parsed.modelId) ?? input.provider.contextWindow);
    const frame = JSON.parse(buildStableClaudeConversationInput(input.parsed, { maxHistoryChars })[0]!);
    return frame.message.content.map((part: { text: string }) => part.text).join("\n\n");
  }

  private newLeg(session: Session, input: ClaudePersistentTurn, inputHash: string, historyHash: string): Leg {
    let settle!: () => void;
    const receipt: ClaudeTurnReceipt = { inputHash, historyHash, phase: "running", events: [], calls: [], updatedAt: Date.now() };
    const leg: Leg = {
      input, receipt, state: { sawPartialText: false, sawPartialThinking: false, sawTerminalResult: false,
        openToolBlocks: new Map(), partialToolCallIds: new Set(), strictToolBlockCapture: true, translatorBudget: createTranslatorBudget(),
        completeAssistantToolUse: true, toolBridgeNames: input.bridge.emittedNameMap, maxToolBlockStarts: CODEBUDDY_TOOL_LIMITS.maxTurnToolCalls },
      nativeNames: new Map(), mcpIds: new Set(), recoveredCalls: new Set(), buffered: false, finished: false, bytes: 0,
      done: new Promise(resolve => { settle = resolve; }), settle: () => settle(), release: () => {},
    };
    session.snapshot.turns.push(receipt);
    session.active = leg;
    leg.timer = setTimeout(() => this.fail(session, new ClaudeSessionError("Claude generation timed out.", "claude_session_timeout", 504, true)), this.deps.timeoutMs ?? 300_000);
    leg.timer.unref?.();
    return leg;
  }

  private admit(leg: Leg): void {
    const incoming = leg.input!.incoming;
    const producer = incoming.sendBudget?.beginSpendProducer?.();
    let permit: { release(): void; use(): boolean } | undefined;
    try {
      const decision = incoming.sendBudget?.reserveDispatch({ sendClass: "initial", targetKey: `claude-cli|${leg.input!.provider.baseUrl}|${leg.input!.parsed.modelId}` });
      if (decision && !decision.allowed) throw new SendBudgetExhaustedError();
      permit = decision?.allowed ? decision.permit : undefined;
      incoming.beforePhysicalSend?.();
      if (permit && !permit.use()) throw new SendBudgetExhaustedError();
      incoming.onPhysicalSend?.({ ordinal: 1 });
    } catch (error) { permit?.release(); producer?.close(); throw error; }
    let released = false;
    leg.release = () => { if (!released) { released = true; permit?.release(); producer?.close(); } };
  }

  private onMessage(session: Session, message: Record<string, unknown>): void {
    const leg = session.active;
    if (!leg || leg.finished) return;
    try {
      leg.bytes += Buffer.byteLength(JSON.stringify(message));
      if (leg.bytes > 64 * 1024 * 1024) throw new ClaudeSessionError("Claude output exceeded its buffer limit.", "claude_session_buffer_limit", 502);
      const raw = message.type === "stream_event" ? message.event as Record<string, unknown> : undefined;
      if (raw?.type === "message_start") leg.state.sawMessageStop = false;
      const events = mapStreamMessageToEvents(message, leg.state);
      if (leg.state.toolCallLimitExceeded || leg.state.uncapturedToolUse) throw new ClaudeSessionError("Claude returned an invalid tool batch.", "claude_tool_protocol_error", 502);
      for (const event of events) {
        if (event.type === "error") {
          if (message.type === "result") {
            try { leg.input!.onCliOutputError?.(event.message); } catch { /* Observers cannot replace the CLI failure. */ }
          }
          this.fail(session, new ClaudeSessionError(event.message, event.code ?? "claude_cli_error", event.status ?? 502)); return;
        }
        if (event.type === "done") {
          if (leg.receipt.calls.length) throw new ClaudeSessionError("Claude ended before its external tool batch was acknowledged.", "claude_tool_protocol_error", 502);
          if (leg.input!.bridge.requireToolCall) throw new ClaudeSessionError("Claude did not call the required external tool.", "tool_call_required", 502);
          this.complete(session, "completed", { ...event, ...(leg.state.partialUsage ? { usage: leg.state.partialUsage } : {}) });
          return;
        }
        if (event.type === "tool_call_start") {
          if (session.snapshot.turns.some(turn => turn !== leg.receipt && turn.calls.some(call => call.id === event.id))) {
            throw new ClaudeSessionError("Claude reused an earlier external tool call ID.", "claude_tool_id_conflict", 502);
          }
          const name = leg.input!.bridge.emittedNameMap.get(event.name);
          if (!name) throw new ClaudeSessionError("Claude called a tool outside the current catalog.", "undeclared_tool_call", 502);
          leg.buffered = true;
          leg.openCall = { id: event.id, name, arguments: "" };
          leg.receipt.calls.push(leg.openCall);
          leg.nativeNames.set(event.id, event.name);
          leg.receipt.events.push({ ...event, name });
        } else {
          if (event.type === "tool_call_delta" && leg.openCall) leg.openCall.arguments += event.arguments;
          if (event.type === "tool_call_end" && leg.openCall) {
            if (!leg.openCall.arguments.trim()) {
              leg.openCall.arguments = "{}";
              leg.receipt.events.push({ type: "tool_call_delta", arguments: "{}" });
            }
            leg.openCall = undefined;
          }
          leg.receipt.events.push(event);
        }
        if (!leg.buffered) this.publish(session);
      }
      this.finishTools(session);
    } catch (error) { this.fail(session, error instanceof Error ? error : new Error(String(error))); }
  }

  private onToolCall(session: Session, call: ClaudeWorkerToolCall, signal: AbortSignal): Promise<{ content: Array<{ type: "text"; text: string }>; isError?: boolean }> {
    const leg = session.active;
    const signature = claudeDigest([call.name, call.arguments]);
    const known = session.pending.get(call.id);
    if (known) {
      if (known.signature !== signature) return Promise.reject(new ClaudeSessionError("Claude reused a tool ID with different arguments.", "claude_tool_id_conflict", 502));
      return known.promise;
    }
    if (!leg || leg.finished || !leg.input!.bridge.emittedNameMap.has(`${CODEBUDDY_MCP_TOOL_PREFIX}${call.name}`)) {
      return Promise.reject(new ClaudeSessionError("An unexpected MCP tool call was refused.", "undeclared_tool_call", 502));
    }
    let resolve!: Pending["resolve"];
    let reject!: Pending["reject"];
    const promise = new Promise<Awaited<Pending["promise"]>>((yes, no) => { resolve = yes; reject = no; });
    const onAbort = (): void => reject(new Error("The Claude MCP request was cancelled."));
    signal.addEventListener("abort", onAbort, { once: true });
    const pending: Pending = { signature, promise, resolve, reject };
    leg.mcpIds.add(call.id);
    session.pending.set(call.id, pending);
    void promise.finally(() => signal.removeEventListener("abort", onAbort)).catch(() => {});
    if (signal.aborted) onAbort();
    this.finishTools(session);
    return promise;
  }

  private finishTools(session: Session): void {
    const leg = session.active;
    if (!leg || leg.finished || !leg.state.sawMessageStop || !leg.receipt.calls.length || leg.openCall) return;
    if (leg.receipt.calls.length !== leg.state.completedToolCalls) return;
    const capturedIds = new Set(leg.receipt.calls.map(call => call.id));
    if ([...leg.mcpIds].some(id => !capturedIds.has(id))) {
      this.fail(session, new ClaudeSessionError("The MCP batch contained an unrecognized tool call ID.", "claude_tool_id_conflict", 502));
      return;
    }
    for (const call of leg.receipt.calls) {
      if (leg.recoveredCalls.has(claudeDigest([call.name, JSON.parse(call.arguments)]))) {
        this.fail(session, new ClaudeSessionError("Claude tried to repeat an action whose result was already restored. Automatic execution was refused.", "claude_recovery_repeat_refused"));
        return;
      }
      const pending = session.pending.get(call.id);
      if (!pending) return;
      const name = leg.nativeNames.get(call.id)!.slice(CODEBUDDY_MCP_TOOL_PREFIX.length);
      if (pending.signature !== claudeDigest([name, JSON.parse(call.arguments)])) {
        this.fail(session, new ClaudeSessionError("The MCP request disagreed with the generated tool call.", "claude_tool_id_conflict", 502));
        return;
      }
    }
    this.complete(session, "tools", { type: "done", stopReason: "tool_use", endTurn: false,
      ...(leg.state.partialUsage ? { usage: leg.state.partialUsage } : {}) });
  }

  private complete(session: Session, phase: "tools" | "completed", terminal: AdapterEvent): void {
    const leg = session.active!;
    leg.receipt.events.push(terminal);
    leg.receipt.phase = phase;
    leg.receipt.updatedAt = Date.now();
    session.snapshot.history = claudeHistory([...leg.input!.parsed.context.messages, this.assistant(leg.receipt)]);
    try { this.save(session); }
    catch { leg.receipt.events.pop(); this.fail(session, new ClaudeSessionError("The Claude receipt could not be committed; no tool action was released.", "claude_journal_write_failed", 503)); return; }
    leg.finished = true;
    if (leg.timer) clearTimeout(leg.timer);
    releaseOpenToolBlocks(leg.state);
    leg.state.translatorBudget?.dispose();
    leg.release();
    leg.release = () => {};
    leg.input = undefined;
    this.publish(session);
    leg.settle();
  }

  private assistant(receipt: ClaudeTurnReceipt): OcxMessage {
    const content: Array<{ type: "text"; text: string } | { type: "toolCall"; id: string; name: string; arguments: Record<string, unknown> }> = [];
    for (const event of receipt.events) {
      if (event.type === "text_delta") {
        const last = content.at(-1);
        if (last?.type === "text") last.text += event.text;
        else content.push({ type: "text", text: event.text });
      } else if (event.type === "tool_call_start") {
        const call = receipt.calls.find(item => item.id === event.id)!;
        content.push({ type: "toolCall", id: call.id, name: call.name, arguments: JSON.parse(call.arguments) });
      }
    }
    return { role: "assistant", content, timestamp: Date.now() };
  }

  private publish(session: Session): void {
    const leg = session.active;
    const consumer = session.consumer;
    if (!leg || !consumer || consumer.signal?.aborted) return;
    if (leg.receipt.phase === "tools" && !leg.receipt.deliveryStarted) {
      leg.receipt.deliveryStarted = true;
      try { this.save(session); }
      catch {
        leg.receipt.deliveryStarted = false;
        leg.finished = false;
        this.fail(session, new ClaudeSessionError("Tool delivery could not be durably recorded.", "claude_journal_write_failed", 503)); return;
      }
    }
    while (consumer.sent < leg.receipt.events.length && session.consumer === consumer && !consumer.signal?.aborted) {
      const event = leg.receipt.events[consumer.sent]!;
      if (!leg.finished && leg.buffered && event.type === "tool_call_start") return;
      try { consumer.emit(event); consumer.sent++; }
      catch { session.consumer = undefined; return; }
    }
  }

  private deliverReceipt(session: Session, receipt: ClaudeTurnReceipt, emit: (event: AdapterEvent) => void): void {
    if (receipt.calls.length) {
      if (receipt.deliveryStarted) throw new ClaudeSessionError("The earlier tool batch may already have executed. Return its existing tool results; automatic replay was refused.", "claude_tool_delivery_uncertain");
      const previous = receipt.deliveryStarted;
      receipt.deliveryStarted = true;
      try { this.save(session); }
      catch (error) { receipt.deliveryStarted = previous; throw error; }
    }
    for (const event of receipt.events) emit(event.type === "done" && event.usage
      ? { ...event, usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 } } : event);
  }

  private save(session: Session): void {
    const retained = session.snapshot.turns.filter(turn => turn.phase === "completed" || turn.phase === "interrupted");
    const expire = (old: ClaudeTurnReceipt): void => {
      for (const call of old.calls) session.pending.delete(call.id);
      old.phase = "expired"; old.events = []; old.calls = [];
    };
    for (const old of retained.slice(0, -16)) expire(old);
    for (const old of retained) {
      if (Buffer.byteLength(JSON.stringify(session.snapshot)) <= 16 * 1024 * 1024) break;
      if (old !== session.active?.receipt && old.phase !== "expired") expire(old);
    }
    this.journal.save(session.scope, session.snapshot);
  }

  private fail(session: Session, error: Error): void {
    const leg = session.active;
    this.stopWorker(session);
    if (!leg || leg.finished) return;
    if (leg.timer) clearTimeout(leg.timer);
    leg.receipt.phase = "interrupted";
    const toolStart = leg.receipt.events.findIndex(event => event.type === "tool_call_start");
    leg.receipt.events = (toolStart < 0 ? leg.receipt.events : leg.receipt.events.slice(0, toolStart))
      .filter(event => !["done", "error"].includes(event.type));
    leg.receipt.calls = [];
    leg.receipt.events.push({ type: "error", message: error.message.slice(0, 8192),
      code: error instanceof ClaudeSessionError || error instanceof CodingAgentProtocolError || isTranslatorBudgetExceededError(error) ? error.code : "claude_session_disconnected",
      status: error instanceof ClaudeSessionError ? error.status : 502,
      retryable: error instanceof ClaudeSessionError ? error.retryable : !(error instanceof CodingAgentProtocolError || isTranslatorBudgetExceededError(error)) });
    leg.finished = true;
    releaseOpenToolBlocks(leg.state);
    leg.state.translatorBudget?.dispose();
    leg.release();
    leg.release = () => {};
    leg.input = undefined;
    try { this.save(session); } catch { /* The original error is still delivered; no tool batch was published. */ }
    this.publish(session);
    leg.settle();
  }

  private stopWorker(session: Session): void {
    const worker = session.worker;
    session.worker = undefined;
    worker?.stop();
    for (const pending of session.pending.values()) pending.reject(new Error("The Claude worker stopped."));
    session.pending.clear();
  }

  private armIdle(session: Session): void {
    if (session.idle) clearTimeout(session.idle);
    const delay = session.active && !session.active.finished ? this.deps.reconnectGraceMs ?? 30_000 : this.deps.idleMs ?? 30 * 60_000;
    session.idle = setTimeout(() => {
      if (session.consumer || session.busy) return;
      if (session.active && !session.active.finished) this.fail(session, new ClaudeSessionError("The reconnect grace period expired.", "claude_reconnect_expired", 499, true));
      else this.stopWorker(session);
      if (this.sessions.get(session.scope) === session) this.sessions.delete(session.scope);
    }, delay);
    session.idle.unref?.();
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    for (const session of this.sessions.values()) {
      if (session.idle) clearTimeout(session.idle);
      this.fail(session, new ClaudeSessionError("The Claude session manager is shutting down.", "claude_session_shutdown", 503, true));
    }
    this.sessions.clear();
    this.journal.close();
  }
}

const managers = new Map<string, ClaudeSessionManager>();
export function getClaudeSessionManager(): ClaudeSessionManager {
  if (didRunOptionalShutdownHooks()) throw new ClaudeSessionError("Claude sessions are shutting down.", "claude_session_shutdown", 503);
  const home = getConfigDir();
  let manager = managers.get(home);
  if (!manager) {
    manager = new ClaudeSessionManager();
    managers.set(home, manager);
    registerOptionalShutdownHook(`claude-cli-sessions:${home}`, () => { manager!.close(); managers.delete(home); });
  }
  return manager;
}
