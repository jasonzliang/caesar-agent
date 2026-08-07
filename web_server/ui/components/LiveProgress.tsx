'use client';

import { useMemo } from 'react';
import type { LiveEvent } from '@/lib/useEventSource';
import type { RunDetail } from '@/lib/api';
import { StatusBadge } from './StatusBadge';
import { FollowUpBadge } from './FollowUpBadge';
import { HelpTip } from './HelpTip';
import { ProgressBar } from './ProgressBar';
import { fmtCost, fmtDuration, parseBackendDate, runElapsedSec, useNow } from '@/lib/utils';
import {
  CircleDollarSign,
  Clock,
  Database,
  FileStack,
  Globe,
  Repeat2,
  WandSparkles,
} from 'lucide-react';

type Props = {
  run: RunDetail;
  events: LiveEvent[];
};

// Shape of the JSON payload Caesar's job_runner emits on every
// 'synthesis_progress' SSE event. Every field is optional / loosely typed
// because the server may add fields between client deployments — defensive
// coercions (Number / String) below convert what we actually consume.
type SynthesisProgressPayload = {
  total_drafts?: number | string;
  draft?: number | string;
  phase?: 'synthesize' | 'merge' | 'image_gen' | string;
  iter_n?: number | string;
  iter_total?: number | string;
};

export function LiveProgress({ run, events }: Props) {
  const isTerminal = run.status === 'completed' || run.status === 'failed';
  const summary = useMemo(() => {
    // Track per-phase progress separately. quick_explore and kb_ingest both
    // emit `iteration` events with their own counters, and a single global
    // monotonic `max` would let quick_explore's higher count swallow
    // kb_ingest's restart-from-1.
    const byPhase: Record<string, { n: number; total: number | null }> = {};
    // Dedup draft_complete by draft_n on the client side. Cross-process
    // replay (SQLite) can surface multiple events for the same draft from
    // older code that hadn't yet learned to dedup by draft_n, leaving the
    // count inflated past the preset target.
    const seenDrafts = new Set<string>();
    // Total drafts target — comes from either synthesis_progress (parsed
    // from console-log header) or draft_complete (preset YAML). Either
    // source is sufficient to cap the cross-restart double-count.
    let totalDrafts: number | null = null;
    let lastUrl: string | null = null;
    let phase: string | null = null;
    let liveCost: number | null = null;
    // True while a restart is waiting for the previous attempt's thread to let
    // go of the run directory. Every stat beside it is frozen on the old
    // attempt's numbers until then, which reads as a hung run unless we say so.
    let awaitingTakeover = false;
    let synth: {
      draft: number;
      total_drafts: number;
      merging?: boolean;
      imageGen?: boolean;
      iter_n: number;
      iter_total: number;
      tsMs: number | null;
    } | null = null;
    for (const e of events) {
      // Any progress event means the new attempt got the directory.
      if (e.event === 'takeover_wait') {
        awaitingTakeover = true;
      } else if (e.event !== 'ping' && e.event !== 'log') {
        awaitingTakeover = false;
      }
      if (e.event === 'iteration') {
        const n = Number(e.payload?.n);
        const t = Number(e.payload?.total);
        const ph = typeof e.payload?.phase === 'string' ? e.payload.phase : phase ?? 'unknown';
        if (typeof e.payload?.phase === 'string') phase = e.payload.phase;
        if (typeof e.payload?.url === 'string') lastUrl = e.payload.url;
        const slot = byPhase[ph] ?? (byPhase[ph] = { n: 0, total: null });
        if (Number.isFinite(n) && n > slot.n) slot.n = n;
        if (Number.isFinite(t) && t > 0) slot.total = t;
      } else if (e.event === 'graph_update') {
        const n = Number(e.payload?.iter);
        const slot = byPhase['quick_explore'] ?? (byPhase['quick_explore'] = { n: 0, total: null });
        if (Number.isFinite(n) && n > slot.n) slot.n = n;
      } else if (e.event === 'draft_complete') {
        const dn = String(e.payload?.draft_n ?? '');
        if (dn && !seenDrafts.has(dn)) seenDrafts.add(dn);
        const t = Number(e.payload?.total_drafts);
        if (Number.isFinite(t) && t > 0) totalDrafts = t;
      } else if (e.event === 'cost_update' && !isTerminal) {
        // Skip live cost ticks once the run is in a terminal state — the
        // watchdog's last cost_update fires before Caesar's synthesis/merge
        // phase, so it under-reports vs. the canonical final cost. The
        // `done` branch below is authoritative on completed runs.
        const c = Number(e.payload?.cost_usd);
        if (Number.isFinite(c)) liveCost = c;
      } else if (e.event === 'done') {
        const c = Number(e.payload?.total_cost_usd);
        if (Number.isFinite(c)) liveCost = c;
      } else if (e.event === 'resumed') {
        // Caesar's synthesizer has no checkpoint, so on resume it restarts
        // from draft 1. Drop all prior-cycle synth state — including the
        // draft_complete events from previous resume cycles, which would
        // otherwise inflate the displayed draft number past what Caesar's
        // current cycle has actually finished.
        synth = null;
        seenDrafts.clear();
      } else if (e.event === 'synthesis_progress') {
        const p = e.payload as SynthesisProgressPayload;
        const t = Number(p.total_drafts) || 1;
        const ph = (p.phase ?? '') as string;
        // image_gen events use `draft`/`total_drafts` to carry the image
        // count (e.g., 1 for fast preset), NOT the synthesis draft count.
        // Preserve the previously-seen draft total so the UI shows
        // "draft 3/3 · generating 1 image" rather than "draft 1/1".
        const isImageGen = ph === 'image_gen';
        const prevTotal: number = synth?.total_drafts ?? t;
        const totalForSynth = isImageGen ? prevTotal : t;
        const draftForSynth = isImageGen ? totalForSynth : (Number(p.draft) || 1);
        synth = {
          draft: draftForSynth,
          total_drafts: totalForSynth,
          merging: ph === 'merge',
          imageGen: isImageGen,
          iter_n: Number(p.iter_n) || 1,
          iter_total: Number(p.iter_total) || 1,
          tsMs: parseBackendDate(e.timestamp),
        };
        if (!isImageGen && t > 0) totalDrafts = t;
        phase = 'synthesis';
      }
    }
    // Display the bucket matching the latest phase. When the live phase is
    // 'synthesis', byPhase['synthesis'] is never populated — fall back to
    // the last exploration/ingest bucket so the counter stays meaningful.
    // 'explore' must come before 'quick_explore' in the fallback: deep-runs
    // populate byPhase['explore'] from the iteration events (with total)
    // AND byPhase['quick_explore'] from graph_update events (no total), so
    // without this order the deep-run counter flips from "240/240" during
    // exploration to "240" during synthesis.
    const iterPhase: string | null =
      (phase && byPhase[phase]) ? phase
        : byPhase['kb_ingest']    ? 'kb_ingest'
        : byPhase['explore']      ? 'explore'
        : byPhase['quick_explore'] ? 'quick_explore'
        : null;
    const active = (iterPhase && byPhase[iterPhase]) || { n: 0, total: null };
    return {
      lastIteration: active.n,
      total: active.total,
      iterPhase,
      pages: byPhase['quick_explore']?.n ?? 0,
      drafts: seenDrafts.size,
      totalDrafts,
      lastUrl,
      phase,
      liveCost,
      synth,
      awaitingTakeover,
    };
  }, [events]);

  // Per-second clock so the elapsed counter ticks live, even between events.
  // `useNow` returns null until the client mounts (avoids SSR hydration
  // mismatch); we fall back to "—" via the conditional below.
  const now = useNow(1000);
  const elapsedSec = useMemo(
    () => runElapsedSec(run, now),
    [run, now],
  );

  // Populated by the synthesis-bar IIFE below so the Draft stat shows the
  // exact same "N / M" the bar shows. Stays null when we have no total yet.
  let draftStatValue: string | null = null;
  const showInheritedProgress = run.mode === 'refine' && !summary.iterPhase;
  const displayedPhase = showInheritedProgress ? run.graph_progress_phase : summary.iterPhase;
  let iterationLabel = 'Iteration';
  let iterationIcon = <Repeat2 size={16} />;
  let iterationValue: string | number = '—';
  if (displayedPhase === 'quick_explore') {
    iterationLabel = 'Pages fetched';
    iterationIcon = <Globe size={16} />;
  } else if (displayedPhase === 'kb_ingest') {
    iterationLabel = 'Pages embedded';
    iterationIcon = <Database size={16} />;
  }
  if (showInheritedProgress) {
    const graphTotal = run.graph_progress_total?.toLocaleString();
    iterationValue = graphTotal ? `${graphTotal} / ${graphTotal}` : '—';
  }
  if (!showInheritedProgress && summary.lastIteration) {
    iterationValue = summary.total
      ? `${summary.lastIteration} / ${summary.total}`
      : `${summary.lastIteration}`;
  }

  return (
    <div className="rounded-2xl border border-gray-200 bg-white p-6 space-y-4">
      <div className="flex items-center justify-between gap-2 min-w-0">
        <div className="flex items-center gap-2 min-w-0">
          <StatusBadge status={run.status} />
          <FollowUpBadge mode={run.mode} parentRunId={run.parent_run_id} />
          {run.llm_model && (
            <span
              title="LLM model used for this run"
              className="pill pill-queued max-w-full"
            >
              <span className="text-gray-500">Model</span>
              <code className="font-mono text-gray-900 truncate">{run.llm_model}</code>
            </span>
          )}
        </div>
        <HelpTip
          className="shrink-0"
          label="Live status while the run works: pages explored, which answer draft it is on, elapsed time, and the running cost."
        />
      </div>

      <Stat
        icon={iterationIcon}
        label={iterationLabel}
        value={iterationValue}
      />
      {summary.total && summary.phase !== 'synthesis' ? (
        <div className="-mt-2">
          <ProgressBar value={(summary.lastIteration / summary.total) * 100} />
        </div>
      ) : null}

      {(() => {
        // Single source of (effectiveDraft, total) shared with the Draft stat
        // so the two displays never disagree. synth.draft from Caesar's
        // per-iter log is authoritative when present; otherwise fall back to
        // the filesystem signal (completed + 1).
        const total = summary.totalDrafts ?? summary.synth?.total_drafts ?? null;
        if (total == null) return null;
        // Only render once synthesis is actually happening — a completed
        // draft on disk (drafts > 0), or a recent synthesis_progress event
        // (logger actively emitting), or run terminal. Without this gate,
        // stale synthesis_progress events left in the DB from a different
        // run (singleton-logger bug, since fixed) would surface a bogus
        // synth bar while the current run is still in exploration phase.
        // Has to exceed one synthesis iteration, or the bar blinks out between
        // them and a healthy run reads as stopped: iterations run ~2 minutes on
        // the smaller presets and longer on the bigger ones, so 60s hid the bar
        // for most of every gap. Staleness is still bounded well under the
        // watchdog's 20-minute stall threshold.
        const FRESH_MS = 300_000;
        const synthActivelyRunning =
          summary.synth?.tsMs != null && now != null
            ? now - summary.synth.tsMs < FRESH_MS
            : false;
        if (!isTerminal && summary.drafts === 0 && !synthActivelyRunning) return null;
        const completed = Math.min(summary.drafts, total);
        // For terminal runs, show how many drafts ACTUALLY completed —
        // not the preset target. A cooperative shutdown can end a run
        // after only some of the planned drafts (e.g. server restart
        // mid-synthesis triggers the new shutdown_called check between
        // drafts), and the prior `total` shortcut overstated progress.
        const effectiveDraft = isTerminal
          ? completed
          : summary.synth
            ? Math.max(summary.synth.draft, Math.min(total, completed + 1))
            : Math.min(total, completed + 1);
        draftStatValue = `${effectiveDraft} / ${total}`;
        // Step counter is shown when synth has a value AND it matches the
        // effective draft (otherwise the step number is from an older draft);
        // suppressed on terminal runs (the run is done).
        const merging = !isTerminal && !!summary.synth && summary.synth.merging;
        const imageGen = !isTerminal && !!summary.synth && summary.synth.imageGen;
        const showStep =
          !isTerminal
          && !!summary.synth
          && !merging
          && !imageGen
          && summary.synth.draft === effectiveDraft;
        // While merging or rendering images, all per-draft work is done —
        // show the bar at 100% of the draft loop and let the phase label
        // carry the signal that an extra step is still in flight.
        const draftFraction = (merging || imageGen)
          ? 1
          : showStep
            ? (summary.synth!.draft - 1 + summary.synth!.iter_n / summary.synth!.iter_total) / total
            : completed / total;
        return (
          <div>
            <div className="flex items-center justify-between text-sm">
              <span className="flex items-center gap-2 text-gray-500">
                <WandSparkles size={16} />
                Synthesis
              </span>
              <span className="tabular-nums text-gray-900 font-medium">
                {imageGen ? (
                  <>
                    draft {total}/{total}
                    <span className="text-gray-500 font-normal"> · rendering</span>
                  </>
                ) : merging ? (
                  <>
                    draft {total}/{total}
                    <span className="text-gray-500 font-normal"> · merging</span>
                  </>
                ) : (
                  <>
                    draft {effectiveDraft}/{total}
                    {showStep ? (
                      <>{' · '}step {summary.synth!.iter_n}/{summary.synth!.iter_total}</>
                    ) : !isTerminal ? (
                      <span className="text-gray-500 font-normal"> · synthesizing</span>
                    ) : null}
                  </>
                )}
              </span>
            </div>
            <ProgressBar
              value={draftFraction * 100}
              pulsing={merging || imageGen}
              className="mt-1"
            />
          </div>
        );
      })()}
      <Stat
        icon={<FileStack size={16} />}
        label="Draft"
        value={draftStatValue ?? (summary.drafts ? `${summary.drafts}` : '—')}
      />
      <Stat
        icon={<Clock size={16} />}
        label="Elapsed"
        value={elapsedSec != null ? fmtDuration(elapsedSec) : '—'}
      />
      <Stat
        icon={<CircleDollarSign size={16} />}
        label="Cost"
        value={fmtCost(summary.liveCost ?? run.total_cost_usd)}
      />

      {summary.lastUrl && summary.phase === 'quick_explore' && run.status === 'running' && (
        <div className="text-xs text-gray-500 truncate" title={summary.lastUrl}>
          Currently exploring:{' '}
          <a className="text-brand-800 hover:underline" href={summary.lastUrl} target="_blank" rel="noopener">
            {summary.lastUrl}
          </a>
        </div>
      )}

      {summary.awaitingTakeover && !isTerminal && (
        <div className="rounded-lg bg-amber-50 border border-amber-200 px-3 py-2 text-sm text-amber-800">
          Stopping the previous attempt. The restart begins once it lets go, which
          takes up to about a minute; the figures above still describe the attempt
          being stopped.
        </div>
      )}

      {run.status === 'failed' && run.error_message && (
        <div className="rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-sm text-red-700">
          {run.error_message}
        </div>
      )}
    </div>
  );
}

function Stat({
  icon,
  label,
  value,
}: {
  icon?: React.ReactNode;
  label: string;
  value: string | number;
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="flex items-center gap-2 text-sm text-gray-500">
        {icon}
        {label}
      </span>
      <span className="text-base font-medium tabular-nums text-gray-900">{value}</span>
    </div>
  );
}
