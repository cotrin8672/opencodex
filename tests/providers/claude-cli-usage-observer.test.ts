import { afterEach, describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  checkClaudeUsageAdmission, createClaudeUsageObserver, peekClaudeUsageObservation, recordClaudeUsageRefusal,
  type AdmissionDeps,
} from "../../src/adapters/claude-cli/usage-admission";
import { settleBackgroundWork } from "../../src/lib/background-work";
import type { ProviderQuota } from "../../src/providers/quota-types";

const dirs: string[] = [];
const gates: (() => void)[] = [];
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(done => { resolve = done; });
  return { promise, resolve };
}
function quota(at: number, five = 10, weekly = 20): ProviderQuota {
  return { fiveHourPercent: five, fiveHourResetAt: at + 3_600_000, weeklyPercent: weekly, weeklyResetAt: at + 86_400_000, updatedAt: at };
}
function harness(seed = true) {
  const dir = mkdtempSync(join(tmpdir(), "ocx-claude-observer-"));
  dirs.push(dir);
  const statePath = join(dir, "usage.json");
  let now = Date.UTC(2026, 10, 2, 12);
  const initial = now;
  const accountA = { key: createHash("sha256").update("fake-account-a").digest("hex"), access: "fake-access-a" };
  let account: typeof accountA | null = accountA;
  let probes = 0;
  let probe = async (_token: string): Promise<ProviderQuota | null> => quota(now, 30, 40);
  const deps: AdmissionDeps & { minRefreshMs: number } = {
    statePath, now: () => now, identity: () => account,
    probe: token => { probes++; return probe(token); }, minRefreshMs: 5,
  };
  if (seed) writeFileSync(statePath, JSON.stringify({ identity: accountA.key, checkedAt: now, quota: quota(now) }));
  return {
    deps, initial, accountA,
    advance(ms = 1000) { now += ms; },
    switchAccount() { account = { key: createHash("sha256").update("fake-account-b").digest("hex"), access: "fake-access-b" }; },
    rotateToken() { if (account) account = { ...account, access: "fake-access-refreshed" }; },
    logout() { account = null; },
    setProbe(fn: typeof probe) { probe = fn; },
    probes: () => probes,
    snapshot: () => peekClaudeUsageObservation(deps),
    saved: () => readFileSync(statePath, "utf8"),
  };
}
const event = (rateLimitType: unknown, utilization?: unknown, resetsAt?: unknown) => ({
  type: "rate_limit_event", rate_limit_info: { status: "allowed", rateLimitType, utilization, resetsAt },
});
const start = (id: string) => ({ type: "stream_event", event: { type: "message_start", message: { id } } });
const stop = () => ({ type: "stream_event", event: { type: "message_stop" } });
const assistant = (id: string, stop_reason: string | null = null) => ({ type: "assistant", uuid: `block-${id}`, message: { id, stop_reason, content: [{ type: "text", text: "PRIVATE RESPONSE" }] } });
const result = (id = "result") => ({ type: "result", subtype: "success", uuid: id, usage: { input_tokens: 2, output_tokens: 1 } });
async function settled() { expect(await settleBackgroundWork({ timeoutMs: 2000 })).toEqual({ settled: true, pending: [] }); }
async function until(check: () => boolean) {
  const deadline = Date.now() + 2000;
  while (!check()) {
    if (Date.now() > deadline) throw new Error("background work did not reach the expected state");
    await Bun.sleep(1);
  }
}
afterEach(async () => {
  for (const release of gates.splice(0)) release();
  await settled();
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

describe("Claude usage notification observations", () => {
  test("SDK fractions become percentages, Unix seconds become milliseconds, including >100%", async () => {
    const h = harness(); h.advance();
    const observer = createClaudeUsageObserver(h.deps);
    const seconds = (h.initial + 9_000_000) / 1000;
    observer.observe(event("five_hour", 0.37, seconds));
    observer.observe(event("seven_day", 1.24, seconds + 1));
    observer.observe(event("seven_day_opus", 0.5, seconds + 2));
    observer.observe(event("seven_day_sonnet", 1.1, seconds + 3));
    await settled();
    expect(h.snapshot()).toMatchObject({ checkedAt: h.initial, quota: {
      fiveHourPercent: 37, fiveHourResetAt: seconds * 1000, weeklyPercent: 124, weeklyResetAt: (seconds + 1) * 1000,
      updatedAt: h.initial + 1000,
      customWindows: [{ label: "Opus", scope: "model", percent: 50, resetAt: (seconds + 2) * 1000 }, { label: "Sonnet", scope: "model", percent: 110.00000000000001, resetAt: (seconds + 3) * 1000 }],
    } });
    expect(h.probes()).toBe(0);
  });

  test("partial values preserve other windows and their freshness; missing fields never become zero", async () => {
    const h = harness(); h.advance();
    const observer = createClaudeUsageObserver(h.deps);
    observer.observe(event("five_hour", 0.37));
    observer.observe(event("seven_day", undefined, (h.initial + 8_000_000) / 1000));
    observer.observe(event("seven_day_opus", undefined, 1_900_000_000));
    await settled();
    expect(h.snapshot()).toMatchObject({ checkedAt: h.initial, quota: {
      fiveHourPercent: 37, fiveHourResetAt: h.initial + 3_600_000,
      weeklyResetAt: h.initial + 8_000_000, updatedAt: h.initial,
    } });
    expect(h.snapshot()?.quota?.customWindows).toBeUndefined();
    expect(h.snapshot()?.quota?.weeklyPercent).toBeUndefined();
    expect(h.probes()).toBe(0);
  });

  test("rejects malformed, negative, infinite, string and unsupported fields without inventing quota", async () => {
    const h = harness(false);
    const observer = createClaudeUsageObserver(h.deps);
    for (const value of [null, [], {}, { type: "rate_limit_event" }, { type: "rate_limit_event", rate_limit_info: { utilization: 0.4, rateLimitType: "five_hour" } },
      event("five_hour"), event("five_hour", -1, -1), event("five_hour", "0.7", "123"), event("five_hour", Infinity, NaN),
      event("five_hour", Number.MAX_VALUE, Number.MAX_VALUE), event("overage", 0.7, 123), event("seven_day_overage_included", 1),
      event("five_hour", undefined, 0), event("five_hour", undefined, 8_640_000_000_001),
      event("seven_day_opus", null, null)]) observer.observe(value);
    observer.finish();
    await settled();
    expect(existsSync(h.deps.statePath!)).toBe(false);
    expect(h.probes()).toBe(0);
  });

  test("optional unifiedWindows validates each known window independently and accepts a real zero", async () => {
    const h = harness(); h.advance();
    const observer = createClaudeUsageObserver(h.deps);
    observer.observe({ type: "rate_limit_event", rate_limit_info: { status: "allowed_warning", unifiedWindows: {
      five_hour: { utilization: 0, resetsAt: 1_900_000_000 }, seven_day: { utilization: 0.45, resetsAt: "bad" },
      seven_day_opus: [], seven_day_sonnet: { utilization: -1 }, attacker: { utilization: 0.8, text: "PRIVATE" },
    } } });
    await settled();
    expect(h.snapshot()?.quota).toMatchObject({ fiveHourPercent: 0, fiveHourResetAt: 1_900_000_000_000, weeklyPercent: 45, weeklyResetAt: h.initial + 86_400_000, updatedAt: h.initial });
    expect(h.saved()).not.toContain("PRIVATE");
    expect(h.probes()).toBe(0);
  });

  test("fast updates from concurrent workers merge without losing unrelated windows", async () => {
    const h = harness();
    const a = createClaudeUsageObserver(h.deps); const b = createClaudeUsageObserver(h.deps);
    for (let i = 1; i <= 30; i++) { a.observe(event("five_hour", i / 100)); b.observe(event("seven_day", (i + 30) / 100)); }
    await settled();
    expect(h.snapshot()?.quota).toMatchObject({ fiveHourPercent: 30, weeklyPercent: 60 });
    expect(h.probes()).toBe(0);
  });

  test("reset-only rollover drops an exhausted counter instead of extending its block", async () => {
    const h = harness(false);
    writeFileSync(h.deps.statePath!, JSON.stringify({ identity: h.accountA.key, checkedAt: h.initial, quota: { fiveHourPercent: 100, fiveHourResetAt: h.initial + 1000, updatedAt: h.initial } }));
    const observer = createClaudeUsageObserver(h.deps); h.advance(2000);
    observer.observe(event("five_hour", undefined, (h.initial + 18_000_000) / 1000));
    await settled();
    expect(h.snapshot()?.quota?.fiveHourPercent).toBeUndefined();
    expect(h.snapshot()?.quota?.fiveHourResetAt).toBe(h.initial + 18_000_000);
    expect(h.snapshot()?.fieldFreshness?.["five_hour.percent"]).toBeUndefined();
    expect((await checkClaudeUsageAdmission("", h.deps)).state).toBe("available");
    expect(h.probes()).toBe(0);
  });

  test("same-reset notifications preserve a measured exhausted counter", async () => {
    const h = harness(false);
    writeFileSync(h.deps.statePath!, JSON.stringify({ identity: h.accountA.key, checkedAt: h.initial, quota: { fiveHourPercent: 100, fiveHourResetAt: h.initial + 3_600_000, updatedAt: h.initial } }));
    const observer = createClaudeUsageObserver(h.deps);
    observer.observe(event("five_hour", undefined, (h.initial + 3_600_000) / 1000)); await settled();
    expect(h.snapshot()?.quota?.fiveHourPercent).toBe(100);
    expect((await checkClaudeUsageAdmission("", h.deps)).state).toBe("exhausted");
  });
});

describe("Claude post-completion usage refresh", () => {
  test("a completion forces a fresh GET inside the 60-second preflight cache", async () => {
    const h = harness();
    expect((await checkClaudeUsageAdmission("", h.deps)).state).toBe("available");
    expect(h.probes()).toBe(0);
    h.advance();
    const observer = createClaudeUsageObserver(h.deps);
    observer.observe(start("response-a")); observer.observe(stop());
    await settled();
    expect(h.probes()).toBe(1);
    expect(h.snapshot()).toMatchObject({ checkedAt: h.initial + 1000, quota: { fiveHourPercent: 30, weeklyPercent: 40, updatedAt: h.initial + 1000 } });
    await checkClaudeUsageAdmission("", h.deps);
    expect(h.probes()).toBe(1);
  });

  test("completed assistants, block frames, message_stop, result and finish deduplicate by native response ID", async () => {
    const h = harness();
    const observer = createClaudeUsageObserver(h.deps);
    observer.observe(start("native-a"));
    observer.observe(assistant("native-a")); observer.observe(assistant("native-a"));
    observer.observe({ type: "stream_event", event: { type: "content_block_stop" } });
    await settled(); expect(h.probes()).toBe(0);
    observer.observe(assistant("native-a", "tool_use"));
    await settled(); expect(h.probes()).toBe(1);
    observer.observe(assistant("native-a", "tool_use")); observer.observe(stop()); observer.observe(result()); observer.finish();
    await settled(); expect(h.probes()).toBe(1);
  });

  test("result completes partial assistant blocks without waiting for worker shutdown", async () => {
    const h = harness();
    const observer = createClaudeUsageObserver(h.deps);
    observer.observe(assistant("native-a")); observer.observe(assistant("native-a"));
    await settled(); expect(h.probes()).toBe(0);
    observer.observe(result());
    await settled(); expect(h.probes()).toBe(1);
    observer.finish(); await settled(); expect(h.probes()).toBe(1);
  });

  test("standalone native success results are completion evidence; repeated results are deduplicated", async () => {
    const h = harness();
    const observer = createClaudeUsageObserver(h.deps);
    observer.observe(result("one")); await settled();
    observer.observe(result("one")); observer.finish(); await settled();
    expect(h.probes()).toBe(1);
  });

  test("a GET started before completion must finish before an additional fresh GET", async () => {
    const h = harness(false);
    const old = deferred<ProviderQuota | null>(); gates.push(() => old.resolve(null));
    h.setProbe(() => h.probes() === 1 ? old.promise : Promise.resolve(quota(h.initial + 1000, 55, 66)));
    const preflight = checkClaudeUsageAdmission("", h.deps);
    await until(() => h.probes() === 1);
    const observer = createClaudeUsageObserver(h.deps);
    observer.observe(start("response-a")); observer.observe(stop());
    await Bun.sleep(10); expect(h.probes()).toBe(1);
    h.advance(); old.resolve(quota(h.initial));
    await preflight; await settled();
    expect(h.probes()).toBe(2);
    expect(h.snapshot()?.quota).toMatchObject({ fiveHourPercent: 55, weeklyPercent: 66 });
  });

  test("a burst of simultaneous worker completions shares one account GET", async () => {
    const h = harness();
    const workers = Array.from({ length: 12 }, () => createClaudeUsageObserver(h.deps));
    for (const [i, worker] of workers.entries()) { worker.observe(start(`native-${i}`)); worker.observe(stop()); worker.finish(); }
    await settled(); expect(h.probes()).toBe(1);
  });

  test("new completions during a GET get one trailing refresh with a minimum interval", async () => {
    const h = harness(); h.deps.minRefreshMs = 25;
    const gate = deferred<ProviderQuota | null>(); gates.push(() => gate.resolve(null));
    const starts: number[] = [];
    h.setProbe(async () => { starts.push(Date.now()); return h.probes() === 1 ? gate.promise : quota(h.initial, 70, 80); });
    const a = createClaudeUsageObserver(h.deps); const b = createClaudeUsageObserver(h.deps);
    a.observe(start("native-a")); a.observe(stop());
    await until(() => h.probes() === 1);
    b.observe(start("native-b")); b.observe(stop());
    a.observe(start("native-c")); a.observe(stop());
    await Bun.sleep(3); gate.resolve(quota(h.initial));
    await settled();
    expect(h.probes()).toBe(2);
    expect(starts[1]! - starts[0]!).toBeGreaterThanOrEqual(20);
    expect(h.snapshot()?.quota).toMatchObject({ fiveHourPercent: 70, weeklyPercent: 80 });
    await Bun.sleep(35); expect(h.probes()).toBe(2);
  });

  test("new completions after a GET get one delayed trailing refresh, never idle polling", async () => {
    const h = harness(); h.deps.minRefreshMs = 20;
    const observer = createClaudeUsageObserver(h.deps);
    observer.observe(start("a")); observer.observe(stop()); await settled();
    observer.observe(start("b")); observer.observe(stop());
    observer.observe(start("c")); observer.observe(stop());
    await settled(); expect(h.probes()).toBe(2);
    await Bun.sleep(30); expect(h.probes()).toBe(2);
  });

  test("null and throwing background GETs retain last good quota and measurement clocks", async () => {
    for (const fail of [async () => null, async () => { throw new Error("fake-access-a PRIVATE RESPONSE"); }]) {
      const h = harness(); h.advance(); h.setProbe(fail);
      const observer = createClaudeUsageObserver(h.deps);
      observer.observe(result()); await settled();
      expect(h.snapshot()).toMatchObject({ checkedAt: h.initial, quota: quota(h.initial) });
      expect(h.saved()).not.toContain("fake-access-a"); expect(h.saved()).not.toContain("PRIVATE RESPONSE");
      expect(h.probes()).toBe(1);
    }
  });

  test("a late preflight GET preserves newer notification fields, including within one clock tick", async () => {
    const h = harness(false);
    const gate = deferred<ProviderQuota | null>(); gates.push(() => gate.resolve(null)); h.setProbe(() => gate.promise);
    const preflight = checkClaudeUsageAdmission("", h.deps); await until(() => h.probes() === 1);
    const observer = createClaudeUsageObserver(h.deps);
    observer.observe(event("five_hour", 0.73, (h.initial + 5_000_000) / 1000));
    await until(() => h.snapshot()?.quota?.fiveHourPercent === 73);
    gate.resolve(quota(h.initial)); await preflight; await settled();
    expect(h.snapshot()?.quota).toMatchObject({ fiveHourPercent: 73, fiveHourResetAt: h.initial + 5_000_000, weeklyPercent: 20 });
  });

  test("a late completion GET also preserves notifications while filling unobserved windows", async () => {
    const h = harness();
    const gate = deferred<ProviderQuota | null>(); gates.push(() => gate.resolve(null)); h.setProbe(() => gate.promise);
    const observer = createClaudeUsageObserver(h.deps);
    observer.observe(result()); await until(() => h.probes() === 1);
    h.advance(); observer.observe(event("seven_day", 0.91));
    await until(() => h.snapshot()?.quota?.weeklyPercent === 91);
    gate.resolve(quota(h.initial, 65, 50)); await settled();
    expect(h.snapshot()?.quota).toMatchObject({ fiveHourPercent: 65, weeklyPercent: 91 });
  });

  test("full GET replacement removes absent windows and resets instead of keeping retired caps", async () => {
    const h = harness();
    writeFileSync(h.deps.statePath!, JSON.stringify({ identity: h.accountA.key, checkedAt: h.initial, quota: {
      ...quota(h.initial), customWindows: [{ label: "Opus", scope: "model", percent: 100, resetAt: h.initial + 86_400_000 }],
    } }));
    h.setProbe(async () => ({ fiveHourPercent: 25, updatedAt: h.initial }));
    const observer = createClaudeUsageObserver(h.deps); observer.observe(result()); await settled();
    expect(h.snapshot()?.quota).toEqual({ fiveHourPercent: 25, updatedAt: h.initial });
    expect(h.snapshot()?.fieldFreshness).toEqual({ "five_hour.percent": h.initial });
    expect((await checkClaudeUsageAdmission("opus", h.deps)).state).toBe("available");
  });

  test("full replacement retains only newer passive model fields absent from the GET", async () => {
    const h = harness();
    writeFileSync(h.deps.statePath!, JSON.stringify({ identity: h.accountA.key, checkedAt: h.initial, quota: {
      ...quota(h.initial), customWindows: [{ label: "Opus", scope: "model", percent: 100 }, { label: "Sonnet", scope: "model", percent: 100 }],
    } }));
    const gate = deferred<ProviderQuota | null>(); gates.push(() => gate.resolve(null)); h.setProbe(() => gate.promise);
    const observer = createClaudeUsageObserver(h.deps); observer.observe(result()); await until(() => h.probes() === 1);
    h.advance(); observer.observe(event("seven_day_opus", 0.4, (h.initial + 9_000_000) / 1000));
    await until(() => h.snapshot()?.quota?.customWindows?.[0]?.percent === 40);
    gate.resolve(quota(h.initial)); await settled();
    expect(h.snapshot()?.quota?.customWindows).toEqual([{ label: "Opus", scope: "model", percent: 40, resetAt: h.initial + 9_000_000 }]);
    expect(h.snapshot()?.fieldFreshness?.["seven_day_sonnet.percent"]).toBeUndefined();
  });
});

describe("Claude observer account and lifecycle safety", () => {
  test("preflight re-reads fresh access credentials immediately before its GET", async () => {
    const h = harness(false);
    const tokens: string[] = [];
    h.deps.identity = fresh => fresh ? { ...h.accountA, access: "fake-access-refreshed" } : h.accountA;
    h.setProbe(async token => { tokens.push(token); return quota(h.initial); });
    expect((await checkClaudeUsageAdmission("", h.deps)).state).toBe("available");
    expect(tokens).toEqual(["fake-access-refreshed"]);
    expect(h.saved()).not.toContain("fake-access-refreshed");
  });

  test.each(["switched", "logged-out"])("preflight rejects a %s account before any GET or snapshot write", async state => {
    const h = harness(false);
    h.deps.identity = fresh => !fresh ? h.accountA : state === "logged-out" ? null : { key: "different-account", access: "fake-access-b" };
    expect((await checkClaudeUsageAdmission("", h.deps)).state).toBe("unknown");
    expect(h.probes()).toBe(0);
    expect(h.snapshot()).toBeNull();
    expect(existsSync(h.deps.statePath!)).toBe(false);
  });

  test("long-lived workers use refreshed access credentials for each GET of the same account", async () => {
    const h = harness();
    const tokens: string[] = [];
    h.setProbe(async token => { tokens.push(token); return quota(h.initial); });
    const observer = createClaudeUsageObserver(h.deps); await settled();
    h.rotateToken(); observer.observe(result("one")); await settled();
    expect(tokens).toEqual(["fake-access-refreshed"]);
    expect(h.saved()).not.toContain("fake-access-refreshed");
  });

  test("captures identity at creation and discards subsequent events/completions after account switch", async () => {
    const h = harness();
    const observer = createClaudeUsageObserver(h.deps); await settled();
    const before = h.saved(); h.switchAccount();
    observer.observe(event("five_hour", 1)); observer.observe(result()); observer.finish();
    await settled(); expect(h.saved()).toBe(before); expect(h.probes()).toBe(0);
  });

  test("account switches during GET discard its answer and preserve the current account snapshot", async () => {
    const h = harness();
    const gate = deferred<ProviderQuota | null>(); gates.push(() => gate.resolve(null)); h.setProbe(() => gate.promise);
    const old = createClaudeUsageObserver(h.deps); old.observe(result()); await until(() => h.probes() === 1);
    h.switchAccount();
    const current = createClaudeUsageObserver(h.deps); current.observe(event("five_hour", 0.61));
    await until(() => h.snapshot()?.quota?.fiveHourPercent === 61);
    gate.resolve(quota(h.initial, 100, 100)); await settled();
    expect(h.snapshot()?.identity).not.toBe(h.accountA.key);
    expect(h.snapshot()?.quota?.fiveHourPercent).toBe(61);
  });

  test("queued trailing GET does not use a credential after account switch", async () => {
    const h = harness(); h.deps.minRefreshMs = 20;
    const observer = createClaudeUsageObserver(h.deps);
    observer.observe(result("one")); await settled();
    observer.observe(result("two"));
    await Bun.sleep(3); h.switchAccount(); await settled();
    expect(h.probes()).toBe(1);
  });

  test("refusal racing a GET and event preserves refusalUntil without changing the full-probe clock", async () => {
    const h = harness();
    const gate = deferred<ProviderQuota | null>(); gates.push(() => gate.resolve(null)); h.setProbe(() => gate.promise);
    const observer = createClaudeUsageObserver(h.deps); observer.observe(result()); await until(() => h.probes() === 1);
    h.advance();
    await recordClaudeUsageRefusal("You've hit your limit · resets 3pm (UTC)", h.deps);
    observer.observe(event("five_hour", 0.48));
    await until(() => h.snapshot()?.quota?.fiveHourPercent === 48);
    expect(h.snapshot()?.checkedAt).toBe(h.initial);
    gate.resolve(quota(h.initial, 25, 35)); await settled();
    expect(h.snapshot()).toMatchObject({ refusalUntil: Date.UTC(2026, 10, 2, 15), quota: { fiveHourPercent: 48, weeklyPercent: 35 } });
    expect((await checkClaudeUsageAdmission("", h.deps)).state).toBe("exhausted");
  });

  test("idle, replay, startup errors and telemetry-only workers never refresh on finish", async () => {
    const h = harness();
    const observer = createClaudeUsageObserver(h.deps);
    observer.observe({ type: "system", subtype: "init" });
    observer.observe({ type: "result", subtype: "success", usage: { input_tokens: 0, output_tokens: 0 } });
    observer.observe({ ...assistant("replayed", "end_turn"), isReplay: true });
    observer.observe({ ...result(), is_replay: true });
    observer.observe({ ...assistant("synthetic", "end_turn"), isSynthetic: true });
    observer.observe({ type: "result", subtype: "error_during_execution", is_error: true, usage: { input_tokens: 0, output_tokens: 0 } });
    observer.observe(event("five_hour", 0.4)); observer.finish(); observer.finish();
    await settled(); expect(h.probes()).toBe(0); expect(h.snapshot()?.quota?.fiveHourPercent).toBe(40);
  });

  test("worker-stop fallback refreshes only unfinished model activity and is idempotent", async () => {
    const h = harness();
    const observer = createClaudeUsageObserver(h.deps);
    observer.observe(start("unfinished"));
    await settled(); expect(h.probes()).toBe(0);
    observer.finish(); observer.finish();
    await settled(); expect(h.probes()).toBe(1);
  });

  test("factory/observe/finish return synchronously while identity and GET are unresolved", async () => {
    const h = harness();
    const identity = deferred<ReturnType<NonNullable<AdmissionDeps["identity"]>>>();
    gates.push(() => identity.resolve(null));
    let resolving = 0;
    h.deps.identity = () => { resolving++; return identity.promise; };
    const probe = deferred<ProviderQuota | null>(); gates.push(() => probe.resolve(null)); h.setProbe(() => probe.promise);
    const observer = createClaudeUsageObserver(h.deps);
    expect(resolving).toBe(1);
    expect(observer).not.toBeInstanceOf(Promise);
    expect(observer.ready).toBeInstanceOf(Promise);
    let ready = false;
    const readiness = observer.ready!.then(() => { ready = true; });
    await Bun.sleep(0); expect(ready).toBe(false);
    expect(observer.observe(start("a"))).toBeUndefined();
    expect(observer.observe(stop())).toBeUndefined();
    expect(observer.finish()).toBeUndefined();
    expect(h.probes()).toBe(0);
    identity.resolve(h.accountA); await readiness; expect(ready).toBe(true);
    await until(() => h.probes() === 1);
    expect(observer.observe({ get type() { throw new Error("bad optional frame"); } })).toBeUndefined();
    probe.resolve(quota(h.initial)); await settled();
  });

  test("ready resolves without an identity and permanently disables that observer's writes", async () => {
    const h = harness(false);
    let calls = 0;
    h.deps.identity = () => calls++ === 0 ? null : h.accountA;
    const observer = createClaudeUsageObserver(h.deps);
    expect(observer.ready).toBeInstanceOf(Promise);
    await observer.ready;
    observer.observe(event("five_hour", 1)); observer.observe(result()); observer.finish();
    await settled();
    expect(h.probes()).toBe(0);
    expect(h.snapshot()).toBeNull();
    expect(existsSync(h.deps.statePath!)).toBe(false);
  });

  test("throwing identity resolution or absent login never leaks an unhandled rejection", async () => {
    const h = harness(false);
    h.deps.identity = async () => { throw new Error("private credential failure"); };
    const observer = createClaudeUsageObserver(h.deps);
    observer.observe(event("five_hour", 0.5)); observer.observe(result()); observer.finish();
    await settled(); expect(h.probes()).toBe(0); expect(h.snapshot()).toBeNull();
  });
});
