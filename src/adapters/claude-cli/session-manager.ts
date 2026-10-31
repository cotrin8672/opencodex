import type { AdapterEvent, OcxMessage, OcxParsedRequest, OcxProviderConfig } from "../../types";
import type { SDKMessage } from "@anthropic-ai/claude-agent-sdk";
import { resolve } from "node:path";
import type { IncomingMeta } from "../base";
import type { CodeBuddyToolBridge } from "../codebuddy/tool-bridge";
import { CODEBUDDY_MCP_TOOL_PREFIX, CODEBUDDY_TOOL_LIMITS } from "../codebuddy/tool-bridge";
import { CodingAgentProtocolError, usageFromResult } from "../coding-agent/protocol";
import { SendBudgetExhaustedError } from "../../lib/upstream-retry";
import { getConfigDir } from "../../config/paths";
import { createTranslatorBudget, isTranslatorBudgetExceededError } from "../../lib/translator-budget";
import { registerOptionalShutdownHook, didRunOptionalShutdownHooks } from "../../lib/optional-shutdown-hooks";
import { buildClaudeSessionCatchUp, claudeAccountIdentity, claudeDigest, claudeHistory, claudeOperationIdentity, historyExtends } from "./session-context";
import { openClaudeSessionJournal, type ClaudeSessionJournal, type ClaudeSessionSnapshot, type ClaudeToolReceipt, type ClaudeTurnReceipt } from "./session-journal";
import { prepareClaudeToolResults } from "./session-recovery";
import { claudeHistoryBoundary, claudeInstructionContinuation, extractClaudeInstructions, restoreClaudeInstructions } from "./session-instructions";
import { planClaudeNativeSession, type ClaudeNativeSessionIO } from "./session-native";
import { createClaudeSdkStreamState, mapClaudeSdkMessage, releaseClaudeSdkBlocks, type ClaudeSdkStreamState } from "./session-events";
import { ClaudeSessionError } from "./session-errors";
import { ClaudeSessionSetup } from "./session-setup";
import type { ClaudeSessionWorker, ClaudeWorkerOptions, ClaudeWorkerToolCall } from "./session-worker";

export { ClaudeSessionError } from "./session-errors";

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
  nativeIO?: ClaudeNativeSessionIO;
  nativeDirectory?: string;
  timeoutMs?: number;
  setupTimeoutMs?: number;
  reconnectGraceMs?: number;
  idleMs?: number;
}

type Pending = {
  signature: string;
  settled: boolean;
  promise: Promise<{ content: Array<{ type: "text"; text: string }>; isError?: boolean }>;
  resolve: (value: { content: Array<{ type: "text"; text: string }>; isError?: boolean }) => void;
  reject: (error: Error) => void;
};
type Consumer = { emit: (event: AdapterEvent) => void; signal?: AbortSignal; sent: number; cancel: () => void };
type QueuedTool = { call: ClaudeToolReceipt; nativeName: string; events: AdapterEvent[] };
type Leg = {
  input?: ClaudePersistentTurn;
  receipt: ClaudeTurnReceipt;
  state: ClaudeSdkStreamState;
  openCall?: ClaudeToolReceipt;
  nativeNames: Map<string, string>;
  mcpIds: Set<string>;
  buffered: boolean;
  waitingTools?: boolean;
  waitingResults?: boolean;
  interrupting?: boolean;
  nativeId: string;
  assistantId?: string;
  instructionIds: string[];
  finished: boolean;
  bytes: number;
  done: Promise<void>;
  settle: () => void;
  timer?: ReturnType<typeof setTimeout>;
  setup?: ClaudeSessionSetup;
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
  busy: number;
  setup: Promise<void>;
  epoch: number;
  queued: QueuedTool[];
  nativeId?: string;
};

/** One owner per conversation; a disconnected HTTP response does not own the Claude process. */
export class ClaudeSessionManager {
  private readonly sessions = new Map<string, Session>();
  private readonly closingWorkers = new Map<string, Promise<void>>();
  private readonly journal: ClaudeSessionJournal;
  private readonly shutdown = new AbortController();
  private closed = false;
  private closing?: Promise<void>;

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
      session = { scope: input.scope, snapshot: this.journal.load(input.scope) ?? { version: 1, history: [], turns: [] },
        pending: new Map(), busy: 0, setup: Promise.resolve(), epoch: 0, queued: [] };
      this.sessions.set(input.scope, session);
    }
    const predecessor = session.setup;
    let unlock!: () => void;
    const gate = new Promise<void>(resolve => { unlock = resolve; });
    // A cancelled queued request must not let its successor overtake the current owner.
    session.setup = predecessor.then(() => gate);
    session.busy++;
    const setup = new ClaudeSessionSetup(this.deps.setupTimeoutMs ?? 30_000, input.incoming.abortSignal, this.shutdown.signal);
    let leg: Leg | undefined;
    let consumer: Consumer | undefined;
    let onAbort: (() => void) | undefined;
    let cancelled: Promise<void> | undefined;
    try {
      await setup.wait(predecessor, "waiting for the preceding request");
      if (session.idle) clearTimeout(session.idle);
      if (this.closed) throw new ClaudeSessionError("Claude sessions are shutting down.", "claude_session_shutdown", 503);
      if (input.incoming.abortSignal?.aborted) return;
      let history = claudeHistory(input.parsed.context.messages);
      const account = claudeAccountIdentity();
      const identity = claudeDigest([input.binary, input.env.CLAUDE_CONFIG_DIR, account]);
      const configuration = claudeDigest([input.parsed.modelId, input.binary, input.systemPrompt, input.effort, input.env.CLAUDE_CONFIG_DIR, account]);
      const catalog = claudeDigest(input.bridge.tools);
      const operationId = claudeOperationIdentity(input.parsed, input.incoming);
      const fingerprint = (): string => claudeDigest([history, configuration, catalog, input.parsed.options, operationId]);
      let inputHash = fingerprint();
      const requestHash = inputHash;
      const rawPrevious = this.receipt(session, requestHash);
      if ((session.active?.finished || rawPrevious !== session.active?.receipt) && this.replay(session, rawPrevious, operationId, emit)) return;
      if (session.active && !session.active.finished
        && this.matches(session.active.receipt, requestHash)) {
        leg = session.active;
      } else {
        // Never-delivered actions can be discarded; delivered actions still require their actual results.
        const obsolete = session.snapshot.turns.filter(turn => turn.phase === "tools" && !turn.deliveryStarted
          && turn.inputHash !== inputHash && turn.requestHash !== requestHash);
        if (obsolete.length) {
          const draft = structuredClone(session.snapshot);
          for (const turn of draft.turns) if (obsolete.some(old => old.inputHash === turn.inputHash)) {
            turn.phase = "expired"; turn.calls = []; turn.events = [];
          }
          this.save(session, draft);
          this.stopWorker(session);
        }
        const staged = prepareClaudeToolResults(session.snapshot, input.parsed.context.messages);
        if (staged.changed) this.save(session, staged.snapshot);
        else {
          session.snapshot = staged.snapshot;
          if (session.active) session.active.receipt = staged.snapshot.turns.find(turn => turn.inputHash === session.active!.receipt.inputHash) ?? session.active.receipt;
        }
        const restored = restoreClaudeInstructions(staged.messages, session.snapshot.instructions ?? []);
        input = { ...input, parsed: { ...input.parsed, context: { ...input.parsed.context, messages: restored } } };
        history = claudeHistory(restored);
        inputHash = fingerprint();
        const historyHash = claudeDigest(history);
        if (session.snapshot.turns.some(turn => turn.historyHash === historyHash && turn.inputHash !== inputHash
          && turn.deliveryStarted && turn.calls.length && (!operationId || !turn.operationId || operationId === turn.operationId
            || turn.calls.some(call => !call.result)))) {
          throw new ClaudeSessionError("This input already delivered an external action. Return its existing result before changing settings.", "claude_tool_delivery_uncertain");
        }
        const previous = this.receipt(session, inputHash) ?? this.receipt(session, requestHash);
        if ((session.active?.finished || previous !== session.active?.receipt) && this.replay(session, previous, operationId, emit)) return;
        if (session.active && !session.active.finished && this.matches(session.active.receipt, inputHash)) {
          leg = session.active;
        } else {
          let pendingTurns = session.snapshot.turns.filter(turn => turn.phase === "tools");
          const missingResults = pendingTurns.some(turn => turn.calls.some(call => !call.result));
          const current = session.active;
          const extending = extractClaudeInstructions(restored, session.snapshot.history);
          const queued = (session.snapshot.instructions ?? []).filter(entry => entry.phase === "queued");
          const queuedTexts = queued.flatMap(entry => entry.texts);
          const newTexts = (extending ?? []).filter(text => {
            if (queuedTexts[0] !== text) return true;
            queuedTexts.shift(); return false;
          });
          if (newTexts.length && (current && !current.finished || pendingTurns.length)) {
            this.enqueueInstructions(session, requestHash, operationId, newTexts);
            pendingTurns = session.snapshot.turns.filter(turn => turn.phase === "tools");
          }
          const accepted = (session.snapshot.instructions ?? []).filter(entry => entry.phase !== "completed");
          const sameWorker = session.worker && session.configuration === configuration && session.catalog === catalog;
          let steered = false;
          if (current && !current.finished && !current.waitingResults) {
            if (sameWorker && extending && claudeInstructionContinuation(restored, session.snapshot.history, this.assistant(current.receipt), new Set())) {
              await this.interruptGeneration(session, setup);
              steered = true;
            } else this.fail(session, new ClaudeSessionError("A newer input replaced this Claude generation.", "claude_turn_superseded", 499));
          } else if (current?.waitingResults && !current.finished) {
            this.finishInterrupted(session);
          }
          if (missingResults) {
            if (!accepted.length && newTexts.length === 0) throw new ClaudeSessionError("Claude is waiting for the existing tool results.", "claude_tool_results_pending");
            if (previous) session.snapshot.turns.splice(session.snapshot.turns.indexOf(previous), 1);
            leg = this.newLeg(session, input, inputHash, historyHash, operationId, requestHash, setup);
            leg.waitingResults = true;
            leg.nativeId = session.nativeId ?? leg.nativeId;
            if (leg.timer) clearTimeout(leg.timer);
            const waiting = leg;
            leg.timer = setTimeout(() => {
              if (session.active !== waiting || waiting.finished || !waiting.waitingResults) return;
              try {
                this.finishInterrupted(session, { type: "error", status: 409, code: "claude_tool_results_pending", retryable: false,
                  message: "The additional instructions were saved, but the existing external tool results have not arrived. Return the original tool results to continue; do not execute those calls again." });
              } catch { this.fail(session, new ClaudeSessionError("The waiting response could not be committed.", "claude_journal_write_failed", 503)); }
            }, this.deps.setupTimeoutMs ?? 30_000);
            leg.timer.unref?.();
            leg.receipt.requestHashes = [...new Set([requestHash, ...accepted.map(entry => entry.inputHash)])].slice(-64);
            leg.instructionIds = accepted.map(entry => entry.inputHash);
            leg.receipt.events.push({ type: "heartbeat", replayUnsafe: true });
            this.save(session);
          } else {
            const extendsHistory = historyExtends(history, session.snapshot.history);
            const nativeContinuation = session.worker && session.configuration === configuration && (extendsHistory || steered)
              && (!session.queued.length || session.catalog === catalog)
              && (pendingTurns.length > 0
                ? claudeInstructionContinuation(restored, session.snapshot.history, undefined, new Set(pendingTurns.flatMap(turn => turn.calls.map(call => call.id))))
                  && pendingTurns.every(turn => turn.calls.every(call => session.pending.has(call.id)))
                : steered || history.length > session.snapshot.history.length);
            let start = claudeHistoryBoundary(restored, session.snapshot.history) ?? 0;
            let resume: string | undefined;
            const directory = resolve(this.deps.nativeDirectory ?? getConfigDir());
            if (!nativeContinuation) {
              await setup.wait(this.stopWorker(session), "stopping the preceding worker");
              const plan = await setup.wait(planClaudeNativeSession({ record: session.snapshot.native, messages: restored, identity, directory }, this.deps.nativeIO, setup.signal), "restoring native history");
              session.snapshot.native = plan.native; start = plan.start; resume = plan.resume;
            }
            if (session.snapshot.turns.length >= 4096 && !previous) throw new ClaudeSessionError("This conversation reached its receipt limit; start a new conversation.", "claude_session_history_limit", 413);
            if (previous) session.snapshot.turns.splice(session.snapshot.turns.indexOf(previous), 1);
            leg = this.newLeg(session, input, inputHash, historyHash, operationId, requestHash, setup);
            leg.receipt.requestHashes = [...new Set([requestHash, ...accepted.map(entry => entry.inputHash)])].slice(-64);
            leg.instructionIds = accepted.map(entry => entry.inputHash);
            if (nativeContinuation && pendingTurns.length) leg.nativeId = session.nativeId ?? leg.nativeId;
            leg.waitingTools = !!nativeContinuation && pendingTurns.length > 0 && session.queued.length > 0;
            session.snapshot.history = history;
            if (session.snapshot.native) session.snapshot.native.dirty = true;
            this.save(session);
            if (!session.worker) {
              const createWorker = this.deps.createWorker ?? (await setup.wait(import("./session-worker"), "loading the Claude worker")).createClaudeSessionWorker;
              const owner = session;
              const epoch = session.epoch;
              const starting = createWorker({
                binary: input.binary, model: input.parsed.modelId, env: input.env, systemPrompt: input.systemPrompt,
                effort: input.effort, tools: input.bridge.tools, cwd: directory, persistSession: true,
                ...(resume ? { resume } : { sessionId: session.snapshot.native!.sessionId }),
                onMessage: message => { if (owner.epoch === epoch) this.onMessage(owner, message); },
                onToolCall: (call, signal) => owner.epoch === epoch ? this.onToolCall(owner, call, signal)
                  : Promise.reject(new ClaudeSessionError("The Claude worker was replaced.", "claude_turn_superseded", 499)),
                onError: error => { if (owner.epoch === epoch) this.fail(owner, error); },
              });
              let adopted = false;
              let decide!: () => void;
              const decision = new Promise<void>(resolve => { decide = resolve; });
              // The fence ends only after atomic adoption or verified orphan cleanup.
              this.trackClosing(session.scope, starting.then(async worker => {
                await decision;
                if (!adopted) await worker.stop();
              }, error => {
                if (error instanceof ClaudeSessionError && error.code === "claude_worker_shutdown_failed") throw error;
              }));
              try {
                const worker = await setup.wait(starting, "starting the Claude worker");
                setup.check();
                if (!leg.finished && !this.closed && owner.epoch === epoch) {
                  session.worker = worker;
                  session.configuration = configuration;
                  session.catalog = catalog;
                  adopted = true;
                }
              } finally { decide(); }
            } else if (session.catalog !== catalog) {
              await setup.wait(session.worker.updateTools(input.bridge.tools), "updating the tool catalog");
              session.catalog = catalog;
            }
            if (input.incoming.abortSignal?.aborted) {
              this.fail(session, new ClaudeSessionError("The request was cancelled before inference.", "claude_request_cancelled", 499));
              return;
            }
            if (!leg.finished) {
              if (!session.worker) throw new ClaudeSessionError("The Claude worker did not start.", "claude_worker_unavailable", 502);
              if (!leg.waitingTools) this.admit(leg);
              if (nativeContinuation && pendingTurns.length) {
                for (const instruction of accepted.filter(entry => entry.phase === "queued")) {
                  for (const text of instruction.texts) await setup.wait(session.worker.send(text, { shouldQuery: false, uuid: crypto.randomUUID() }), "writing a queued instruction");
                }
              }
              for (const instruction of accepted) instruction.phase = "sent";
              for (const turn of pendingTurns) turn.phase = "completed";
              try { this.save(session, session.snapshot, pendingTurns); }
              catch (error) { for (const turn of pendingTurns) turn.phase = "tools"; throw error; }
              if (nativeContinuation && pendingTurns.length) {
                for (const turn of pendingTurns) for (const call of turn.calls) {
                  const pending = session.pending.get(call.id);
                  if (!pending || !call.result) throw new ClaudeSessionError("The pending MCP call was lost before continuation.", "claude_mcp_call_lost", 502);
                  pending.resolve({ content: [{ type: "text", text: call.result.content }], isError: call.result.isError });
                }
                this.finishTools(session);
              } else {
                const text = steered && newTexts.length ? newTexts.join("\n\n")
                  : buildClaudeSessionCatchUp(input.parsed, input.provider, input.bridge, start);
                session.nativeId = leg.nativeId;
                await setup.wait(session.worker.send(text, { uuid: leg.nativeId }), "writing the Claude input");
              }
            }
          }
        }
      }
      if (leg.waitingResults) input.incoming.pacingSlot?.release();
      session.consumer?.cancel();
      let abort!: () => void;
      cancelled = new Promise<void>(resolve => { abort = resolve; });
      const owner = session;
      consumer = { emit, signal: input.incoming.abortSignal, sent: 0, cancel: () => {
        if (owner.consumer === consumer) owner.consumer = undefined;
        abort();
      } };
      session.consumer = consumer;
      onAbort = consumer.cancel;
      input.incoming.abortSignal?.addEventListener("abort", onAbort, { once: true });
      if (input.incoming.abortSignal?.aborted) onAbort();
      this.publish(session);
    } catch (error) {
      if (leg && !leg.finished) this.fail(session, error instanceof Error ? error : new Error(String(error)));
      if (input.incoming.abortSignal?.aborted && error instanceof ClaudeSessionError && error.code === "claude_request_cancelled") return;
      throw error;
    } finally {
      if (leg?.setup === setup) leg.setup = undefined;
      setup.dispose(); session.busy--; unlock(); this.armIdle(session);
    }
    if (!leg || !consumer) return;
    try {
      await Promise.race([leg.done, cancelled]);
    } finally {
      if (onAbort) input.incoming.abortSignal?.removeEventListener("abort", onAbort);
      if (session.consumer === consumer) session.consumer = undefined;
      this.armIdle(session);
    }
  }

  private matches(receipt: ClaudeTurnReceipt, hash: string): boolean {
    return receipt.inputHash === hash || receipt.requestHash === hash || receipt.requestHashes?.includes(hash) === true;
  }

  private receipt(session: Session, hash: string): ClaudeTurnReceipt | undefined {
    return session.snapshot.turns.findLast(receipt => this.matches(receipt, hash));
  }

  private replay(session: Session, receipt: ClaudeTurnReceipt | undefined, operationId: string | undefined, emit: Consumer["emit"]): boolean {
    if (receipt?.phase === "expired" && operationId) throw new ClaudeSessionError("This old response is no longer cached; it will not be executed again.", "claude_replay_expired");
    if (receipt?.phase === "tools" || (receipt?.phase === "completed" && operationId)
      || receipt?.events.some(event => event.type === "incomplete" && event.reason === "superseded")) {
      this.deliverReceipt(session, receipt!, emit); return true;
    }
    return false;
  }

  private enqueueInstructions(session: Session, inputHash: string, operationId: string | undefined, texts: string[]): void {
    if (session.snapshot.instructions?.some(entry => entry.inputHash === inputHash)) return;
    if ((session.snapshot.instructions ?? []).filter(entry => entry.phase !== "completed").length >= 64) {
      throw new ClaudeSessionError("The Claude instruction queue is full; return the pending tool results before adding more input.", "claude_instruction_queue_full", 429, true);
    }
    const draft = structuredClone(session.snapshot);
    // Interrupted assistant fragments may disappear from the caller's next request.
    (draft.instructions ??= []).push({ inputHash, ...(operationId ? { operationId } : {}), history: session.snapshot.history,
      texts, phase: "queued", updatedAt: Date.now() });
    this.save(session, draft);
  }

  private async interruptGeneration(session: Session, setup: ClaudeSessionSetup): Promise<void> {
    const leg = session.active!;
    const worker = session.worker!;
    leg.interrupting = true;
    let timer!: ReturnType<typeof setTimeout>;
    try {
      await setup.wait(Promise.race([Promise.all([worker.interrupt(), leg.done]), new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new ClaudeSessionError("Claude did not confirm the interrupted turn boundary.", "claude_interrupt_unconfirmed", 504, true)), 10_000);
      })]), "interrupting the preceding generation");
    } catch (error) {
      this.fail(session, error instanceof Error ? error : new Error(String(error))); throw error;
    } finally { clearTimeout(timer); }
  }

  private finishInterrupted(session: Session, terminal?: AdapterEvent): void {
    const leg = session.active!;
    if (leg.interrupting) {
      session.queued = [];
      for (const id of leg.mcpIds) {
        session.pending.get(id)?.reject(new Error("The native turn was interrupted before external delivery."));
        session.pending.delete(id);
      }
    }
    const toolStart = leg.receipt.events.findIndex(event => event.type === "tool_call_start");
    leg.receipt.events = (toolStart < 0 ? leg.receipt.events : leg.receipt.events.slice(0, toolStart))
      .filter(event => event.type !== "error" && event.type !== "done");
    leg.receipt.calls = [];
    leg.receipt.phase = "interrupted";
    leg.receipt.updatedAt = Date.now();
    leg.receipt.events.push(terminal ?? { type: "incomplete", reason: "superseded", endTurn: false, retryable: false,
      ...(leg.state.partialUsage ? { usage: leg.state.partialUsage } : {}) });
    this.save(session);
    this.finishLeg(session, leg);
  }

  private finishLeg(session: Session, leg: Leg): void {
    leg.finished = true;
    if (leg.timer) clearTimeout(leg.timer);
    releaseClaudeSdkBlocks(leg.state);
    leg.state.translatorBudget?.dispose();
    leg.release(); leg.release = () => {};
    leg.input = undefined;
    this.publish(session);
    leg.settle();
  }

  private newLeg(session: Session, input: ClaudePersistentTurn, inputHash: string, historyHash: string, operationId?: string, requestHash = inputHash, setup?: ClaudeSessionSetup): Leg {
    let settle!: () => void;
    const receipt: ClaudeTurnReceipt = { inputHash, historyHash, ...(operationId ? { operationId } : {}),
      ...(requestHash !== inputHash ? { requestHash } : {}), phase: "running", events: [], calls: [], updatedAt: Date.now() };
    const leg: Leg = {
      input, receipt, state: createClaudeSdkStreamState({ translatorBudget: createTranslatorBudget(),
        toolBridgeNames: input.bridge.emittedNameMap, maxToolBlockStarts: CODEBUDDY_TOOL_LIMITS.maxTurnToolCalls }),
      nativeNames: new Map(), mcpIds: new Set(), buffered: false, finished: false, bytes: 0,
      nativeId: crypto.randomUUID(), instructionIds: [], setup,
      done: new Promise(resolve => { settle = resolve; }), settle: () => settle(), release: () => {},
    };
    session.snapshot.turns.push(receipt);
    session.active = leg;
    this.refreshGenerationTimeout(session, leg);
    return leg;
  }

  private refreshGenerationTimeout(session: Session, leg: Leg): void {
    if (leg.finished || leg.waitingResults) return;
    if (leg.timer) clearTimeout(leg.timer);
    leg.timer = setTimeout(() => {
      if (session.active === leg && !leg.finished) this.fail(session, new ClaudeSessionError("Claude generation timed out.", "claude_session_timeout", 504, true));
    }, this.deps.timeoutMs ?? 300_000);
    leg.timer.unref?.();
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

  private onMessage(session: Session, message: SDKMessage): void {
    const leg = session.active;
    if (!leg || leg.finished) return;
    if ("user_message_uuid" in message && typeof message.user_message_uuid === "string" && message.user_message_uuid !== leg.nativeId
      && !("user_message_uuids" in message && message.user_message_uuids?.includes(leg.nativeId))) return;
    if (message.type === "assistant" || message.type === "result" || message.type === "stream_event"
      && ["message_start", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"].includes(message.event.type)) {
      this.refreshGenerationTimeout(session, leg);
    }
    if (leg.interrupting) {
      if (message.type === "result") {
        leg.state.partialUsage = usageFromResult(message) ?? leg.state.partialUsage;
        try { this.finishInterrupted(session); }
        catch (error) { this.fail(session, error instanceof Error ? error : new Error(String(error))); }
      }
      return;
    }
    if (leg.waitingResults) {
      if (message.type === "result") this.fail(session, new ClaudeSessionError("Claude ended its native turn before the external results arrived.", "claude_tool_continuation_lost", 502, true));
      return;
    }
    if (message.type === "assistant" && message.parent_tool_use_id == null) leg.assistantId = message.uuid;
    try {
      leg.bytes += Buffer.byteLength(JSON.stringify(message));
      if (leg.bytes > 64 * 1024 * 1024) throw new ClaudeSessionError("Claude output exceeded its buffer limit.", "claude_session_buffer_limit", 502);
      const raw = message.type === "stream_event" ? message.event : undefined;
      if (raw?.type === "message_start") leg.state.sawMessageStop = false;
      const events = mapClaudeSdkMessage(message, leg.state);
      if (leg.state.toolCallLimitExceeded || leg.state.uncapturedToolUse) throw new ClaudeSessionError("Claude returned an invalid tool batch.", "claude_tool_protocol_error", 502);
      for (const event of events) {
        if (event.type === "error") {
          if (message.type === "assistant" || message.type === "result") {
            try { leg.input!.onCliOutputError?.(event.message); } catch { /* Observers cannot replace the CLI failure. */ }
          }
          this.fail(session, new ClaudeSessionError(event.message, event.code ?? "claude_cli_error", event.status ?? 502,
            event.retryable, { errorType: event.errorType, usage: leg.state.partialUsage ?? event.usage })); return;
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
    const queued = session.queued.find(item => item.call.id === call.id);
    if (queued && (queued.nativeName !== `${CODEBUDDY_MCP_TOOL_PREFIX}${call.name}`
      || signature !== claudeDigest([call.name, JSON.parse(queued.call.arguments)]))) {
      const error = new ClaudeSessionError("The MCP request disagreed with its queued tool call.", "claude_tool_id_conflict", 502);
      this.fail(session, error);
      return Promise.reject(error);
    }
    if (!queued && (!leg || leg.finished || !leg.input!.bridge.emittedNameMap.has(`${CODEBUDDY_MCP_TOOL_PREFIX}${call.name}`))) {
      return Promise.reject(new ClaudeSessionError("An unexpected MCP tool call was refused.", "undeclared_tool_call", 502));
    }
    let resolve!: Pending["resolve"];
    let reject!: Pending["reject"];
    const promise = new Promise<Awaited<Pending["promise"]>>((yes, no) => { resolve = yes; reject = no; });
    const epoch = session.epoch;
    const onAbort = (): void => {
      const error = new ClaudeSessionError("The Claude MCP request was cancelled.", "claude_mcp_call_cancelled", 499, true);
      reject(error);
      if (!leg?.interrupting && session.epoch === epoch && session.worker) this.fail(session, error);
    };
    signal.addEventListener("abort", onAbort, { once: true });
    const pending: Pending = { signature, promise, settled: false,
      resolve: value => { pending.settled = true; resolve(value); },
      reject: error => { pending.settled = true; reject(error); } };
    if (!queued) leg!.mcpIds.add(call.id);
    session.pending.set(call.id, pending);
    void promise.finally(() => signal.removeEventListener("abort", onAbort)).catch(() => {});
    if (signal.aborted) onAbort();
    this.finishTools(session);
    return promise;
  }

  private finishTools(session: Session): void {
    const leg = session.active;
    if (leg?.interrupting || leg?.waitingResults) return;
    if (leg && !leg.finished && leg.waitingTools) {
      leg.waitingTools = false;
      leg.buffered = true;
      for (const item of session.queued.splice(0)) {
        leg.receipt.calls.push(item.call);
        leg.receipt.events.push(...item.events);
        leg.nativeNames.set(item.call.id, item.nativeName);
      }
      leg.state.sawMessageStop = true;
      leg.state.completedToolCalls = leg.receipt.calls.length;
    }
    if (!leg || leg.finished || !leg.state.sawMessageStop || !leg.receipt.calls.length || leg.openCall) return;
    if (leg.receipt.calls.length !== leg.state.completedToolCalls) return;
    const capturedIds = new Set(leg.receipt.calls.map(call => call.id));
    if ([...leg.mcpIds].some(id => !capturedIds.has(id))) {
      this.fail(session, new ClaudeSessionError("The MCP batch contained an unrecognized tool call ID.", "claude_tool_id_conflict", 502));
      return;
    }
    for (const call of leg.receipt.calls) {
      const pending = session.pending.get(call.id);
      if (!pending) continue;
      const name = leg.nativeNames.get(call.id)!.slice(CODEBUDDY_MCP_TOOL_PREFIX.length);
      if (pending.signature !== claudeDigest([name, JSON.parse(call.arguments)])) {
        this.fail(session, new ClaudeSessionError("The MCP request disagreed with the generated tool call.", "claude_tool_id_conflict", 502));
        return;
      }
    }
    let ready = 0;
    while (ready < leg.receipt.calls.length && session.pending.has(leg.receipt.calls[ready]!.id)) ready++;
    if (!ready) return;
    // The native CLI may invoke the next MCP tool only after the preceding result arrives.
    const deferred = leg.receipt.calls.slice(ready);
    if (deferred.length) {
      const events = new Map<string, AdapterEvent[]>();
      let id: string | undefined;
      for (const event of leg.receipt.events) {
        if (event.type === "tool_call_start") id = event.id;
        if (id) { const parts = events.get(id) ?? []; parts.push(event); events.set(id, parts); }
        if (event.type === "tool_call_end") id = undefined;
      }
      session.queued.push(...deferred.map(call => ({ call, nativeName: leg.nativeNames.get(call.id)!, events: events.get(call.id)! })));
      const hidden = new Set(deferred.map(call => call.id));
      leg.receipt.events = leg.receipt.events.filter(event => {
        if (event.type === "tool_call_start") id = event.id;
        const keep = !id || !hidden.has(id);
        if (event.type === "tool_call_end") id = undefined;
        return keep;
      });
      leg.receipt.calls = leg.receipt.calls.slice(0, ready);
    }
    this.complete(session, "tools", { type: "done", stopReason: "tool_use", endTurn: false,
      ...(leg.state.partialUsage ? { usage: leg.state.partialUsage } : {}) });
  }

  private complete(session: Session, phase: "tools" | "completed", terminal: AdapterEvent): void {
    const leg = session.active!;
    const draft = structuredClone(session.snapshot);
    const receipt = draft.turns.find(turn => turn.inputHash === leg.receipt.inputHash)!;
    receipt.events.push(terminal);
    receipt.phase = phase;
    receipt.updatedAt = Date.now();
    draft.history = claudeHistory([...leg.input!.parsed.context.messages, this.assistant(receipt)]);
    if (phase === "completed" && draft.native && leg.assistantId) {
      draft.native.history = draft.history;
      draft.native.checkpoints = [...draft.native.checkpoints.filter(point => point.length < draft.history.length),
        { length: draft.history.length, messageId: leg.assistantId }].slice(-64);
      draft.native.dirty = false;
    }
    if (phase === "completed") for (const entry of draft.instructions ?? []) {
      if (leg.instructionIds.includes(entry.inputHash)) entry.phase = "completed";
    }
    try { this.save(session, draft); }
    catch { this.fail(session, new ClaudeSessionError("The Claude receipt could not be committed; no tool action was released.", "claude_journal_write_failed", 503)); return; }
    this.finishLeg(session, leg);
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
    for (const event of receipt.events) emit((event.type === "done" || event.type === "incomplete") && event.usage
      ? { ...event, usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 } } : event);
  }

  private save(session: Session, snapshot = session.snapshot, pinned: ClaudeTurnReceipt[] = []): void {
    const draft = structuredClone(snapshot);
    const completedInstructions = draft.instructions?.filter(entry => entry.phase === "completed") ?? [];
    const expiredInstructions = new Set(completedInstructions.slice(0, -16).map(entry => entry.inputHash));
    if (expiredInstructions.size) draft.instructions = draft.instructions!.filter(entry => !expiredInstructions.has(entry.inputHash));
    const protectedHashes = new Set([session.active?.receipt.inputHash, ...pinned.map(turn => turn.inputHash)]);
    const retained = draft.turns.filter(turn => (turn.phase === "completed" || turn.phase === "interrupted")
      && !protectedHashes.has(turn.inputHash) && !turn.calls.some(call => session.pending.get(call.id)?.settled === false));
    const expired = new Set<string>();
    const expire = (old: ClaudeTurnReceipt): void => {
      expired.add(old.inputHash);
      old.phase = "expired"; old.events = []; old.calls = [];
    };
    const payloads = draft.turns.filter(turn => turn.phase === "completed" || turn.phase === "interrupted").length;
    const slots = Math.max(0, 16 - (payloads - retained.length));
    for (const old of retained.slice(0, Math.max(0, retained.length - slots))) expire(old);
    for (const old of retained) {
      if (Buffer.byteLength(JSON.stringify(draft)) <= 16 * 1024 * 1024) break;
      if (old.phase !== "expired") expire(old);
    }
    this.journal.save(session.scope, draft);
    if (expiredInstructions.size) snapshot.instructions = snapshot.instructions!.filter(entry => !expiredInstructions.has(entry.inputHash));
    for (const turn of snapshot.turns) if (expired.has(turn.inputHash)) {
      for (const call of turn.calls) session.pending.delete(call.id);
      turn.phase = "expired"; turn.calls = []; turn.events = [];
    }
    session.snapshot = snapshot;
    if (session.active) session.active.receipt = snapshot.turns.find(turn => turn.inputHash === session.active!.receipt.inputHash) ?? session.active.receipt;
  }

  private fail(session: Session, error: Error): void {
    const leg = session.active;
    if (leg && !leg.finished) leg.setup?.abort(error);
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
      retryable: error instanceof ClaudeSessionError ? error.retryable : !(error instanceof CodingAgentProtocolError || isTranslatorBudgetExceededError(error)),
      ...(error instanceof ClaudeSessionError ? error.details : {}) });
    leg.finished = true;
    releaseClaudeSdkBlocks(leg.state);
    leg.state.translatorBudget?.dispose();
    leg.release();
    leg.release = () => {};
    leg.input = undefined;
    try { this.save(session); } catch { /* The original error is still delivered; no tool batch was published. */ }
    this.publish(session);
    leg.settle();
  }

  private stopWorker(session: Session): Promise<void> {
    const worker = session.worker;
    session.worker = undefined;
    session.nativeId = undefined;
    session.epoch++;
    session.queued = [];
    if (worker) {
      let stopped: Promise<void>;
      try { stopped = Promise.resolve(worker.stop()); }
      catch (error) { stopped = Promise.reject(error); }
      this.trackClosing(session.scope, stopped);
    }
    for (const pending of session.pending.values()) pending.reject(new Error("The Claude worker stopped."));
    session.pending.clear();
    return this.closingWorkers.get(session.scope) ?? Promise.resolve();
  }

  private trackClosing(scope: string, stopped: Promise<void>): void {
    const closing = Promise.all([this.closingWorkers.get(scope), stopped]).then(() => {});
    this.closingWorkers.set(scope, closing);
    void closing.then(() => {
      if (this.closingWorkers.get(scope) === closing) this.closingWorkers.delete(scope);
    }, () => {});
  }

  private armIdle(session: Session): void {
    if (session.idle) clearTimeout(session.idle);
    if (session.consumer || session.busy) return;
    const delay = session.active && !session.active.finished ? this.deps.reconnectGraceMs ?? 30_000 : this.deps.idleMs ?? 30 * 60_000;
    session.idle = setTimeout(() => {
      if (session.consumer || session.busy) return;
      if (session.active && !session.active.finished) this.fail(session, new ClaudeSessionError("The reconnect grace period expired.", "claude_reconnect_expired", 499, true));
      else this.stopWorker(session);
      if (this.sessions.get(session.scope) === session) this.sessions.delete(session.scope);
    }, delay);
    session.idle.unref?.();
  }

  close(): Promise<void> {
    if (this.closing) return this.closing;
    this.closed = true;
    this.shutdown.abort();
    for (const session of this.sessions.values()) {
      if (session.idle) clearTimeout(session.idle);
      this.fail(session, new ClaudeSessionError("The Claude session manager is shutting down.", "claude_session_shutdown", 503, true));
    }
    this.closing = (async () => {
      const drain = new ClaudeSessionSetup(this.deps.setupTimeoutMs ?? 30_000);
      try {
        await drain.wait(Promise.all([...this.sessions.values()].map(session => session.setup)), "draining session setup");
        await drain.wait(Promise.all(this.closingWorkers.values()), "stopping Claude workers");
      } finally { drain.dispose(); this.sessions.clear(); this.journal.close(); }
    })();
    void this.closing.catch(() => {});
    return this.closing;
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
    registerOptionalShutdownHook(`claude-cli-sessions:${home}`, () => { const closing = manager!.close(); managers.delete(home); return closing; });
  }
  return manager;
}
