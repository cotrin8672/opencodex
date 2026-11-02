import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { getConfigDir } from "../../config/paths";
import { trackBackgroundWork } from "../../lib/background-work";
import { isTestHomeGuardArmed } from "../../lib/test-home-guard";
import { detectClaudeCodeTokenAsync } from "../../oauth/local-token-detect";
import { readAnthropicUsageQuota } from "../../providers/claude-cli-usage";
import type { ProviderQuota } from "../../providers/quota-types";
import { createUsageActivity, record, usageEventWindows, type UsageWindow } from "./usage-events";

export type ClaudeAdmission = {
  state: "available" | "exhausted" | "unknown";
  checkedAt: number;
  resetAt?: number;
  message?: string;
};
type Snapshot = { identity: string; checkedAt: number; quota?: ProviderQuota; refusalUntil?: number; fieldFreshness?: Record<string, number> };
type Identity = { key: string; access: string };
export type AdmissionDeps = {
  now?: () => number;
  /** Resolve the signed-in account. `fresh` bypasses the short in-memory identity cache. */
  identity?: (fresh?: boolean) => Identity | null | Promise<Identity | null>;
  probe?: (token: string) => Promise<ProviderQuota | null>;
  statePath?: string;
};
const CACHE_MS = 60_000;
type UsageProbe = { started: number; work: Promise<Snapshot | null> };
const inflight = new Map<string, UsageProbe>();
const memory = new Map<string, Snapshot>();
// Ordering also distinguishes a notification and GET started in the same millisecond.
let sequence = 0;
const fieldVersions = new Map<string, Map<string, number>>();
let identityCache: { scope: string; at: number; value: Promise<Identity | null> } | undefined;

const nonEmpty = (value: unknown): value is string => typeof value === "string" && value.trim().length > 0;

async function readLocalIdentity(): Promise<Identity | null> {
  // A test process never reads the operator's real sign-in or spends it on a usage probe;
  // tests that exercise admission inject `deps.identity`.
  if (isTestHomeGuardArmed()) return null;
  const credential = await detectClaudeCodeTokenAsync();
  if (!credential || !nonEmpty(credential.access)) return null;
  let account: unknown;
  try {
    const configDir = process.env.CLAUDE_CONFIG_DIR?.trim();
    const configFile = configDir ? join(configDir, ".claude.json") : join(homedir(), ".claude.json");
    account = JSON.parse(readFileSync(configFile, "utf8"))?.oauthAccount?.accountUuid;
  } catch { /* Token fingerprint is the fallback when CLI has no account metadata. */ }
  const seed = nonEmpty(account) ? account : nonEmpty(credential.refresh) ? credential.refresh : undefined;
  if (!seed) return null;
  return { key: createHash("sha256").update(seed).digest("hex"), access: credential.access };
}

/**
 * The signed-in Claude Code account, read through the CLI's own secure storage. Cached briefly so
 * a burst of turns costs one Keychain lookup; an account switch is noticed within the cache window.
 */
function localIdentity(fresh = false): Promise<Identity | null> {
  const scope = process.env.CLAUDE_CONFIG_DIR?.trim() ?? "";
  const now = Date.now();
  if (!fresh && identityCache && identityCache.scope === scope && now - identityCache.at < CACHE_MS) return identityCache.value;
  const value = readLocalIdentity().catch(() => null);
  identityCache = { scope, at: now, value };
  return value;
}
async function resolveIdentity(deps: AdmissionDeps, fresh = false): Promise<Identity | null> {
  try { return await (deps.identity ?? localIdentity)(fresh); } catch { return null; }
}
function statePath(deps: AdmissionDeps): string {
  return deps.statePath ?? join(getConfigDir(), "claude-usage-admission.json");
}
function load(path: string): Snapshot | undefined {
  try {
    const saved = JSON.parse(readFileSync(path, "utf8"));
    if (typeof saved.identity !== "string" || !Number.isFinite(saved.checkedAt)) return;
    if (saved.refusalUntil !== undefined && !finite(saved.refusalUntil)) return;
    if (saved.fieldFreshness !== undefined && (!record(saved.fieldFreshness) || !Object.values(saved.fieldFreshness).every(finite))) return;
    if (saved.quota !== undefined) {
      const q = saved.quota;
      if (!q || typeof q !== "object" || !finite(q.updatedAt)) return;
      for (const field of ["fiveHourPercent", "fiveHourResetAt", "weeklyPercent", "weeklyResetAt"])
        if (q[field] !== undefined && !finite(q[field])) return;
      if (q.customWindows !== undefined && (!Array.isArray(q.customWindows) || !q.customWindows.every((w: unknown) => {
        if (!w || typeof w !== "object") return false;
        const row = w as Record<string, unknown>;
        return typeof row.label === "string" && finite(row.percent) && (row.resetAt === undefined || finite(row.resetAt));
      }))) return;
    }
    return saved;
  } catch { /* Missing or malformed cache is unknown, never proof of capacity. */ }
  return undefined;
}
function save(path: string, snapshot: Snapshot): void {
  memory.set(path, snapshot);
  try {
    mkdirSync(dirname(path), { recursive: true, mode: 0o700 });
    const temp = `${path}.${process.pid}.tmp`;
    writeFileSync(temp, JSON.stringify(snapshot) + "\n", { mode: 0o600 });
    renameSync(temp, path);
  } catch { /* Memory still suppresses launches; no secrets or upstream text are persisted. */ }
}
function finite(value: unknown): value is number { return typeof value === "number" && Number.isFinite(value); }

function latest(path: string, identity?: string): Snapshot | undefined {
  return [memory.get(path), load(path)].filter((s): s is Snapshot => s !== undefined && (identity === undefined || s.identity === identity))
    .sort((a, b) => b.checkedAt - a.checkedAt || (b.quota?.updatedAt ?? 0) - (a.quota?.updatedAt ?? 0) || (b.refusalUntil ?? 0) - (a.refusalUntil ?? 0))[0];
}
function quotaWindows(quota?: ProviderQuota): UsageWindow[] {
  if (!quota) return [];
  return [
    { key: "five_hour", percent: quota.fiveHourPercent, resetAt: quota.fiveHourResetAt },
    { key: "seven_day", percent: quota.weeklyPercent, resetAt: quota.weeklyResetAt },
    ...(quota.customWindows ?? []).map(w => ({ ...w, key: `seven_day_${w.label.toLowerCase()}` })),
  ].filter(w => w.percent !== undefined || w.resetAt !== undefined);
}
function mergeWindows(path: string, identity: string, old: Snapshot | undefined, incoming: UsageWindow[], at: number, order: number, replace = false): Pick<Snapshot, "quota" | "fieldFreshness"> {
  const windows = new Map(quotaWindows(old?.quota).map(w => [w.key, { ...w }]));
  const freshness = { ...old?.fieldFreshness };
  const requestKey = `${path}:${identity}`;
  const versions = fieldVersions.get(requestKey) ?? new Map<string, number>();
  fieldVersions.set(requestKey, versions);
  for (const window of windows.values()) for (const field of ["percent", "resetAt"] as const) {
    if (window[field] !== undefined) freshness[`${window.key}.${field}`] ??= old!.quota!.updatedAt;
    const key = `${window.key}.${field}`;
    if (replace && (versions.get(key) ?? 0) <= order) {
      delete window[field]; delete freshness[key]; versions.set(key, order);
    }
  }
  if (replace) versions.set("$full", order);
  for (const patch of incoming) {
    const window = windows.get(patch.key) ?? { key: patch.key, ...(patch.label ? { label: patch.label, scope: patch.scope } : {}) };
    // A new reset without utilization describes a different window, not the old counter.
    if (!replace && patch.percent === undefined && patch.resetAt !== undefined && patch.resetAt !== window.resetAt
      && (versions.get(`${patch.key}.resetAt`) ?? versions.get("$full") ?? 0) <= order
      && (versions.get(`${patch.key}.percent`) ?? versions.get("$full") ?? 0) <= order) {
      delete window.percent; delete freshness[`${patch.key}.percent`]; versions.set(`${patch.key}.percent`, order);
    }
    for (const field of ["percent", "resetAt"] as const) {
      const key = `${patch.key}.${field}`;
      if (patch[field] === undefined || (versions.get(key) ?? versions.get("$full") ?? 0) > order) continue;
      window[field] = patch[field];
      freshness[key] = at;
      versions.set(key, order);
    }
    windows.set(patch.key, window);
  }
  if (!old?.quota && ![...windows.values()].some(w => w.percent !== undefined)) return {};
  const five = windows.get("five_hour"); const week = windows.get("seven_day");
  const customWindows = [...windows.values()].filter((w): w is UsageWindow & { label: string; percent: number } => w.label !== undefined && w.percent !== undefined)
    .map(({ label, percent, resetAt, scope }) => ({ label, percent, ...(resetAt !== undefined ? { resetAt } : {}), ...(scope ? { scope } : {}) }));
  // A partial notification cannot make an unobserved counter (or reset) look new.
  const observed = [...windows.values()].flatMap(w => (["percent", "resetAt"] as const).filter(field => w[field] !== undefined).map(field => freshness[`${w.key}.${field}`]!));
  return { quota: {
    ...(five?.percent !== undefined ? { fiveHourPercent: five.percent } : {}), ...(five?.resetAt !== undefined ? { fiveHourResetAt: five.resetAt } : {}),
    ...(week?.percent !== undefined ? { weeklyPercent: week.percent } : {}), ...(week?.resetAt !== undefined ? { weeklyResetAt: week.resetAt } : {}),
    ...(customWindows.length ? { customWindows } : {}), updatedAt: observed.length ? Math.min(...observed) : at,
  }, fieldFreshness: freshness };
}

function startProbe(identity: Identity, path: string, deps: AdmissionDeps): UsageProbe {
  const requestKey = `${path}:${identity.key}`;
  const existing = inflight.get(requestKey);
  if (existing) return existing;
  const started = ++sequence;
  const work = (async () => {
    const current = await resolveIdentity(deps, true);
    if (!current || current.key !== identity.key) return null;
    let quota: ProviderQuota | null = null;
    try { quota = await (deps.probe ?? readAnthropicUsageQuota)(current.access); } catch { /* Keep the last good measurement on transport failure. */ }
    if ((await resolveIdentity(deps, true))?.key !== identity.key) return null;
    const at = (deps.now ?? Date.now)();
    // Nothing awaits between the latest read, merge and write: notifications and refusals win races.
    const old = latest(path, identity.key);
    const snapshot: Snapshot = {
      ...old, identity: identity.key, checkedAt: quota || !old?.quota ? at : old.checkedAt,
      ...(quota ? mergeWindows(path, identity.key, old, quotaWindows(quota), at, started, true) : {}),
    };
    save(path, snapshot);
    return snapshot;
  })().finally(() => { if (inflight.get(requestKey)?.work === work) inflight.delete(requestKey); });
  const probe = { started, work };
  inflight.set(requestKey, probe);
  return probe;
}
/** Windows that gate this model: the plan windows plus custom windows in scope for its family. */
function applicableWindows(quota: ProviderQuota | undefined, model: string): { percent?: number; resetAt?: number }[] {
  const family = /opus|sonnet|fable|haiku/i.exec(model)?.[0]?.toLowerCase();
  return [
    { percent: quota?.fiveHourPercent, resetAt: quota?.fiveHourResetAt },
    { percent: quota?.weeklyPercent, resetAt: quota?.weeklyResetAt },
    ...(quota?.customWindows ?? []).filter(w => w.scope !== "model" || !family || w.label.toLowerCase() === family),
  ];
}
/** A reset time in the proxy host's own time zone (or the given one), with the zone named. */
export function formatClaudeReset(resetAt: number, timeZone?: string): string {
  return new Intl.DateTimeFormat("en-US", {
    ...(timeZone ? { timeZone } : {}), month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short",
  }).format(resetAt);
}
function full(w: { percent?: number }): boolean { return finite(w.percent) && w.percent >= 100; }
function admission(snapshot: Snapshot, model: string, now: number): ClaudeAdmission {
  const quota = snapshot.quota;
  const exhausted = applicableWindows(quota, model).filter(w => full(w) && (!finite(w.resetAt) || w.resetAt > now));
  if (finite(snapshot.refusalUntil) && snapshot.refusalUntil > now) exhausted.push({ percent: 100, resetAt: snapshot.refusalUntil });
  if (exhausted.length) {
    const resets = exhausted.map(w => w.resetAt).filter(finite);
    // Unknown reset windows require another read after the short cache, not an invented reset date.
    const resetAt = resets.length === exhausted.length ? Math.max(...resets) : Math.max(snapshot.checkedAt + CACHE_MS, ...resets);
    return { state: "exhausted", checkedAt: snapshot.checkedAt, resetAt, message: `Claude subscription limits are exhausted. Claude launches are paused until ${formatClaudeReset(resetAt)}, then usage will be checked again.` };
  }
  return { state: quota ? "available" : "unknown", checkedAt: snapshot.checkedAt };
}

/**
 * Read-only quota preflight for a `claude-cli` launch. Fails open: any local or upstream failure
 * reports `unknown`, which admits the turn and lets the CLI report its own limit.
 */
export async function checkClaudeUsageAdmission(model = "", deps: AdmissionDeps = {}): Promise<ClaudeAdmission> {
  const now = (deps.now ?? Date.now)();
  try {
    return await checkAdmission(model, deps, now);
  } catch {
    return { state: "unknown", checkedAt: now };
  }
}

async function checkAdmission(model: string, deps: AdmissionDeps, now: number): Promise<ClaudeAdmission> {
  const identity = await resolveIdentity(deps);
  if (!identity) return { state: "unknown", checkedAt: now };
  const path = statePath(deps);
  const cached = latest(path, identity.key);
  if (cached) {
    const status = admission(cached, model, now);
    if (status.state === "exhausted" && status.resetAt! > now) return status;
    // Same windows and threshold as admission(): only a full window in scope for this model forces a re-read.
    const expiredReset = [cached.refusalUntil, ...applicableWindows(cached.quota, model).filter(full).map(w => w.resetAt)]
      .some(reset => finite(reset) && reset <= now);
    if (!expiredReset && now >= cached.checkedAt && now - cached.checkedAt < CACHE_MS) return status;
  }
  const snapshot = await startProbe(identity, path, deps).work;
  if (!snapshot) return { state: "unknown", checkedAt: (deps.now ?? Date.now)() };
  return admission(snapshot, model, (deps.now ?? Date.now)());
}

let preflightOverride: ((model: string) => Promise<ClaudeAdmission>) | undefined;

/** The last usage the admission recorded, as `/api/quota-pool` reports it. */
export type ClaudeUsageObservation = Readonly<Snapshot>;

/**
 * The newest recorded usage snapshot, from memory or the admission file. Read-only: it resolves
 * no sign-in (no Keychain read), runs no probe and writes nothing. `identity` is the admission's
 * own SHA-256 account key, never the account uuid itself.
 */
export function peekClaudeUsageObservation(deps: Pick<AdmissionDeps, "statePath"> = {}): ClaudeUsageObservation | null {
  return latest(statePath(deps)) ?? null;
}
/** Test seam for the pre-dispatch check in the Responses executor; call with no argument to restore. */
export function setClaudeUsagePreflightForTests(fn?: (model: string) => Promise<ClaudeAdmission>): void { preflightOverride = fn; }
/** The pre-dispatch check the Responses executor runs before a claude-cli turn. */
export function claudeUsagePreflight(model: string): Promise<ClaudeAdmission> {
  return (preflightOverride ?? checkClaudeUsageAdmission)(model);
}

// Claude prints "resets 3pm (Europe/Paris)" or "resets 3:30pm (...)"; minutes are optional.
const CLAUDE_RESET_CLOCK = /resets?\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s*\(([^)]+)\)/i;

/**
 * Whether text carries the reset-clock wording the refusal parser honors. The check is lexical
 * only — no "limit" keyword and no clock resolution against the current time — because the phrase
 * matters wherever it appears: a caller-controlled model id can carry the clock while the CLI's
 * own error wording supplies the limit keyword around it.
 */
export function containsClaudeResetClock(text: string): boolean {
  return CLAUDE_RESET_CLOCK.test(text);
}

/** Parse Claude's explicit next-clock reset, including its IANA timezone. */
export function parseClaudeReset(message: string, now = Date.now()): number | undefined {
  if (!/hit.*limit|usage limit|rate limit/i.test(message)) return;
  const match = CLAUDE_RESET_CLOCK.exec(message);
  if (!match) return;
  const hour = Number(match[1]); const minute = Number(match[2] ?? 0);
  if (hour < 1 || hour > 12 || minute > 59) return;
  const hours = hour % 12 + (match[3]!.toLowerCase() === "pm" ? 12 : 0);
  try {
    const formatter = new Intl.DateTimeFormat("en-US", { timeZone: match[4], year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23" });
    const parts = (time: number) => Object.fromEntries(formatter.formatToParts(time).filter(p => p.type !== "literal").map(p => [p.type, Number(p.value)]));
    const date = parts(now);
    for (let day = 0; day < 2; day++) {
      const target = Date.UTC(date.year!, date.month! - 1, date.day! + day, hours, minute);
      let epoch = target;
      for (let i = 0; i < 3; i++) {
        const p = parts(epoch);
        epoch += target - Date.UTC(p.year!, p.month! - 1, p.day!, p.hour!, p.minute!, p.second!);
      }
      if (epoch > now && epoch - now <= 26 * 60 * 60_000) return epoch;
    }
  } catch { /* An invalid or unsupported timezone never establishes a cooldown. */ }
}

/**
 * Remember a limit the CLI itself reported ("You've hit your limit · resets 3pm (Europe/Paris)")
 * so later launches are refused locally until that reset. Only a parsed reset records anything.
 */
export async function recordClaudeUsageRefusal(message: string, deps: AdmissionDeps = {}): Promise<void> {
  const now = (deps.now ?? Date.now)();
  const until = parseClaudeReset(message, now);
  if (!until) return;
  const identity = await resolveIdentity(deps);
  if (!identity) return;
  const path = statePath(deps);
  const same = latest(path, identity.key);
  save(path, { ...same, identity: identity.key, checkedAt: same?.checkedAt ?? 0, refusalUntil: Math.max(until, same?.refusalUntil ?? 0) });
}

export interface ClaudeUsageObserver {
  ready?: Promise<void>;
  observe(message: unknown): void;
  finish(): void;
}
type ObserverDeps = AdmissionDeps & { minRefreshMs?: number };
type CompletionRefresh = { requested: number; served: number; lastStarted?: number; work?: Promise<void> };
const completionRefreshes = new Map<string, CompletionRefresh>();

function refreshAfterCompletion(identityKey: string, path: string, deps: ObserverDeps, completion: number): void {
  const key = `${path}:${identityKey}`;
  const refresh = completionRefreshes.get(key) ?? { requested: 0, served: 0 };
  completionRefreshes.set(key, refresh);
  refresh.requested = Math.max(refresh.requested, completion);
  if (refresh.work) return;
  const minRefreshMs = finite(deps.minRefreshMs) && deps.minRefreshMs >= 0 ? deps.minRefreshMs : 5_000;
  const work = Promise.resolve().then(async () => {
    while (refresh.served < refresh.requested) {
      const prior = inflight.get(key);
      if (prior) {
        await prior.work;
        refresh.served = Math.max(refresh.served, prior.started);
        if (refresh.served >= refresh.requested) break;
      }
      const delay = Math.max(0, (refresh.lastStarted ?? -Infinity) + minRefreshMs - (deps.now ?? Date.now)());
      if (delay > 0) await new Promise<void>(resolve => { const timer = setTimeout(resolve, delay); timer.unref?.(); });
      // Never spend a former worker's credential after the operator switches accounts.
      const current = await resolveIdentity(deps, true);
      if (!current || current.key !== identityKey) { refresh.served = refresh.requested; break; }
      const probe = startProbe(current, path, deps);
      refresh.lastStarted = (deps.now ?? Date.now)();
      await probe.work;
      refresh.served = Math.max(refresh.served, probe.started);
    }
  }).catch(() => { refresh.served = refresh.requested; }).finally(() => {
    refresh.work = undefined;
    if (refresh.served < refresh.requested) refreshAfterCompletion(identityKey, path, deps, refresh.requested);
  });
  refresh.work = work;
  void trackBackgroundWork("claude-usage-completion", work);
}

/** Bind once to the worker's account; telemetry never blocks or throws into its stream. */
export function createClaudeUsageObserver(deps: ObserverDeps = {}): ClaudeUsageObserver {
  const identity = trackBackgroundWork("claude-usage-identity", resolveIdentity(deps, true).then(value => value?.key ?? null));
  const schedule = (work: (identityKey: string, path: string) => void | Promise<void>) => {
    void trackBackgroundWork("claude-usage-observation", identity.then(async captured => {
      if (!captured) return;
      const path = statePath(deps);
      if ((await resolveIdentity(deps, true))?.key !== captured) return;
      await work(captured, path);
    }).catch(() => {}));
  };
  const activity = createUsageActivity(() => {
    const completion = ++sequence;
    schedule((captured, path) => refreshAfterCompletion(captured, path, deps, completion));
  });
  return {
    ready: identity.then(() => undefined),
    observe(message) {
      try {
        const raw = record(message);
        if (!raw || raw.isReplay === true || raw.is_replay === true) return;
        const windows = usageEventWindows(raw);
        if (windows.length) {
          const order = ++sequence;
          const at = (deps.now ?? Date.now)();
          schedule((captured, path) => {
            const old = latest(path, captured);
            const merged = mergeWindows(path, captured, old, windows, at, order);
            if (merged.quota) save(path, { ...old, identity: captured, checkedAt: old?.checkedAt ?? 0, ...merged });
          });
        }
        activity.observe(raw);
      } catch { /* Malformed optional telemetry cannot break generation. */ }
    },
    finish() { try { activity.finish(); } catch { /* Best-effort worker-stop fallback. */ } },
  };
}
