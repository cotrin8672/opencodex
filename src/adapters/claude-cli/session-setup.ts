import { ClaudeSessionError } from "./session-errors";

/** A request owns preparation only; an established generation survives HTTP disconnects. */
export class ClaudeSessionSetup {
  private readonly controller = new AbortController();
  private readonly timer: ReturnType<typeof setTimeout>;
  private stage = "waiting for the preceding request";
  private readonly onCancel = () => this.abort(new ClaudeSessionError("The request was cancelled before inference.", "claude_request_cancelled", 499));
  private readonly onShutdown = () => this.abort(new ClaudeSessionError("Claude sessions are shutting down.", "claude_session_shutdown", 503, true));

  constructor(timeoutMs: number, private readonly request?: AbortSignal, private readonly shutdown?: AbortSignal) {
    this.timer = setTimeout(() => this.abort(new ClaudeSessionError(
      `Claude session setup timed out while ${this.stage}.`, "claude_session_setup_timeout", 504, true,
    )), timeoutMs);
    request?.addEventListener("abort", this.onCancel, { once: true });
    shutdown?.addEventListener("abort", this.onShutdown, { once: true });
    if (request?.aborted) this.onCancel();
    if (shutdown?.aborted) this.onShutdown();
  }

  get signal(): AbortSignal { return this.controller.signal; }
  abort(error: Error): void { this.controller.abort(error); }
  check(): void { this.signal.throwIfAborted(); }

  async wait<T>(pending: Promise<T>, stage: string): Promise<T> {
    this.stage = stage;
    // Attach to pending even if already aborted, so late rejection remains observed.
    let onAbort!: () => void;
    const cancelled = new Promise<never>((_, reject) => {
      onAbort = () => reject(this.signal.reason);
      this.signal.addEventListener("abort", onAbort, { once: true });
      if (this.signal.aborted) onAbort();
    });
    try {
      const value = await Promise.race([pending, cancelled]);
      this.check();
      return value;
    } finally { this.signal.removeEventListener("abort", onAbort); }
  }

  dispose(): void {
    clearTimeout(this.timer);
    this.request?.removeEventListener("abort", this.onCancel);
    this.shutdown?.removeEventListener("abort", this.onShutdown);
  }
}
