import clsx, { type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function fmtCost(usd: number | null | undefined): string {
  if (usd == null) return '—';
  if (usd === 0) return '$0';
  if (usd < 0.01) return '< $0.01';
  return `$${usd.toFixed(usd < 1 ? 2 : 1)}`;
}

export function fmtDuration(seconds: number | null | undefined): string {
  if (seconds == null) return '—';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) {
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return `${m}m ${s}s`;
  }
  // ≥1h: drop seconds; "5h 12m" is more scannable than "5h 12m 33s" and
  // matches the precision the user cares about for Deep / Deepest runs.
  const h = Math.floor(seconds / 3600);
  const m = Math.round((seconds % 3600) / 60);
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}

export function fmtNodes(count: number): string {
  return `${count.toLocaleString()} ${count === 1 ? 'node' : 'nodes'}`;
}

/** A `Date.now()` value that ticks every `intervalMs` ms.
 *
 * Returns `null` on the server and on the first client render (before the
 * mount effect fires). This matters for SSR hydration: if the hook returned
 * `Date.now()` at render time, the server-rendered HTML would carry a
 * different "now" than the client, causing a hydration mismatch warning.
 * Consumers should treat `null` as "no live clock yet" and render a
 * placeholder until the value populates.
 */
import { useEffect, useState } from 'react';
export function useNow(intervalMs = 1000): number | null {
  const [now, setNow] = useState<number | null>(null);
  useEffect(() => {
    setNow(Date.now());
    const id = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);
  return now;
}

/** Parse an ISO datetime string from our backend.
 *
 * The API serializes Python timezone-aware datetimes via Pydantic, which
 * strips tzinfo to a naive ISO string (e.g., "2026-05-08T09:20:59.842986").
 * `new Date(naive)` interprets that as *local* time, but the backend
 * actually stores UTC — that's how a UTC-8 machine ends up with elapsed
 * times of -28800s. Force the parser to read backend timestamps as UTC.
 */
export function parseBackendDate(iso: string | null | undefined): number | null {
  if (!iso) return null;
  // If a timezone suffix is already present, trust it.
  const suffixed = /(Z|[+-]\d\d:?\d\d)$/i.test(iso) ? iso : iso + 'Z';
  const t = new Date(suffixed).getTime();
  return Number.isFinite(t) ? t : null;
}

/** Seconds a run has been working, summed across restarts.
 *
 * `elapsed_prior_s` carries the time from earlier attempts, because a restart
 * resets `started_at` so the progress counters and the clock describe the same
 * attempt. Adding it back here is what keeps the displayed figure meaning "what
 * this run has consumed" instead of "since the latest retry".
 *
 * Single definition on purpose: this was inlined at four call sites, which is
 * exactly how one of them ends up right and the others wrong.
 */
export function runElapsedSec(
  run: { started_at: string | null; finished_at: string | null; elapsed_prior_s?: number },
  now: number | null,
): number | null {
  const prior = run.elapsed_prior_s ?? 0;
  const start = parseBackendDate(run.started_at);
  // Not started yet: earlier attempts are still the honest answer.
  if (start == null) return prior > 0 ? prior : null;
  const tip = parseBackendDate(run.finished_at) ?? now;
  if (tip == null) return null;
  return prior + Math.max(0, (tip - start) / 1000);
}

/** Format an ISO timestamp as "Xs / Xm / Xh ago".
 *
 * Pass `now` (from `useNow()`) when calling from a Client Component that's
 * SSR-rendered, so the server and the first client paint use the same wall
 * clock. Without `now`, this calls `Date.now()` directly — fine for code
 * paths that only ever run client-side, but a hydration mismatch waiting to
 * happen if SSR renders the result.
 */
export function fmtRelative(
  iso: string | null | undefined,
  now?: number | null,
): string {
  if (!iso) return '';
  const t = parseBackendDate(iso);
  if (t == null) return '';
  // If `now` is null (useNow before mount), suppress the relative string —
  // the consumer renders a placeholder until the clock populates.
  if (now === null) return '';
  const tip = now ?? Date.now();
  const delta = Math.max(0, (tip - t) / 1000);
  if (delta < 60) return `${Math.round(delta)}s ago`;
  if (delta < 3600) return `${Math.round(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.round(delta / 3600)}h ago`;
  return new Date(t).toLocaleDateString();
}
