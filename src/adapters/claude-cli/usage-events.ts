/** Only subscription windows with known units and scope may enter the admission cache. */
export type UsageWindow = { key: string; percent?: number; resetAt?: number; label?: string; scope?: "model" };
export const record = (value: unknown): Record<string, unknown> | undefined =>
  value !== null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : undefined;
const nonnegative = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value) && value >= 0;

export function usageEventWindows(message: Record<string, unknown>): UsageWindow[] {
  if (message.type !== "rate_limit_event") return [];
  const info = record(message.rate_limit_info);
  if (!info || !["allowed", "allowed_warning", "rejected"].includes(String(info.status))) return [];
  const windows = new Map<string, UsageWindow>();
  const add = (key: unknown, raw: Record<string, unknown> | undefined) => {
    if (!raw || typeof key !== "string" || !["five_hour", "seven_day", "seven_day_opus", "seven_day_sonnet"].includes(key)) return;
    const percent = nonnegative(raw.utilization) && Number.isFinite(raw.utilization * 100) ? raw.utilization * 100 : undefined;
    const resetAt = nonnegative(raw.resetsAt) && raw.resetsAt > 0 && Number.isFinite(new Date(raw.resetsAt * 1000).getTime()) ? raw.resetsAt * 1000 : undefined;
    if (percent === undefined && resetAt === undefined) return;
    const label = key === "seven_day_opus" ? "Opus" : key === "seven_day_sonnet" ? "Sonnet" : undefined;
    windows.set(key, { ...windows.get(key), key, ...(percent !== undefined ? { percent } : {}), ...(resetAt !== undefined ? { resetAt } : {}), ...(label ? { label, scope: "model" as const } : {}) });
  };
  // Private CLI payloads are optional; never depend on their presence or accept arbitrary labels.
  const unified = record(info.unifiedWindows);
  if (unified) for (const key of ["five_hour", "seven_day", "seven_day_opus", "seven_day_sonnet"]) add(key, record(unified[key]));
  add(info.rateLimitType, info);
  return [...windows.values()];
}

/** Native response IDs, rather than per-block SDK UUIDs, identify a model completion. */
export function createUsageActivity(onCompletion: () => void): { observe(message: Record<string, unknown>): void; finish(): void } {
  const responses = new Map<string, { id: string; completed: boolean }>();
  const completed = new Set<string>();
  let anonymous = 0;
  let activity = false;
  const remember = (channel: string, response: { id: string; completed: boolean }) => {
    responses.delete(channel);
    responses.set(channel, response);
    if (responses.size > 256) responses.delete(responses.keys().next().value!);
  };
  const complete = (response: { id: string; completed: boolean }) => {
    if (response.completed || completed.has(response.id)) return;
    response.completed = true;
    completed.add(response.id);
    // shortcut: retain the latest 256 native IDs; explicit replay flags cover older history.
    if (completed.size > 256) completed.delete(completed.values().next().value!);
    onCompletion();
  };
  return {
    observe(message) {
      if (message.isReplay === true || message.is_replay === true || message.isSynthetic === true || message.is_synthetic === true) return;
      const channel = typeof message.parent_tool_use_id === "string" ? message.parent_tool_use_id : "";
      const event = record(message.event);
      const body = record(message.message);
      if (message.type === "stream_event" && event && ["message_start", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"].includes(String(event.type))) {
        activity = true;
        const start = record(event.message);
        let response = responses.get(channel);
        if (event.type === "message_start") {
          const id = typeof start?.id === "string" ? start.id : `anonymous:${++anonymous}`;
          response = { id, completed: completed.has(id) };
          remember(channel, response);
        } else if (!response) {
          response = { id: `anonymous:${++anonymous}`, completed: false };
          remember(channel, response);
        }
        if (event.type === "message_stop") complete(response!);
      } else if (message.type === "assistant" && body && message.error === undefined) {
        activity = true;
        const id = typeof body.id === "string" ? body.id : responses.get(channel)?.id ?? `anonymous:${++anonymous}`;
        let response = responses.get(channel);
        if (response?.id !== id) { response = { id, completed: completed.has(id) }; remember(channel, response); }
        if (typeof body.stop_reason === "string" && body.stop_reason.length) complete(response);
      } else if (message.type === "result") {
        // A bare startup/error result is not evidence that a model was called.
        const usage = record(message.usage);
        const used = usage && ["input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"].some(key => nonnegative(usage[key]) && usage[key] > 0);
        if (!activity && !used) return;
        activity = true;
        const response = responses.get(channel);
        if (response) complete(response);
        else if (used) {
          const id = typeof message.uuid === "string" ? `result:${message.uuid}` : "result:anonymous";
          complete({ id, completed: false });
        }
      }
    },
    finish() {
      if (!activity) return;
      for (const response of responses.values()) complete(response);
    },
  };
}
