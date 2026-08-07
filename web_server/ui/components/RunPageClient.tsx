'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { api, type RunDetail, type RunSummary, type SynthesisOut } from '@/lib/api';
import { useEventSource } from '@/lib/useEventSource';
import { LiveProgress } from './LiveProgress';
import { KnowledgeGraph, type GraphStats } from './KnowledgeGraph';
import { ArtifactView } from './ArtifactView';
import { SourcesPanel } from './SourcesPanel';
import { DeleteRunButton } from './DeleteRunButton';
import { RestartRunButton } from './RestartRunButton';
import { FollowUpDialog } from './FollowUpDialog';
import { HelpTip } from './HelpTip';
import { Download, MessageCirclePlus, WrapText } from 'lucide-react';

type Props = { initial: RunDetail };

const POLL_MS = 2500;

// Secondary action buttons in the answer-card toolbar (Wrap code / Download
// PDF). Extracted so the two stay visually identical without copy-paste drift.
const SECONDARY_BUTTON_CLASS =
  'inline-flex items-center gap-1.5 rounded-lg border border-gray-200 bg-white '
  + 'px-2.5 py-1 text-xs font-medium text-gray-700 hover:bg-gray-50 '
  + 'hover:border-gray-300 transition-colors';

export function RunPageClient({ initial }: Props) {
  const router = useRouter();
  const [run, setRun] = useState<RunDetail>(initial);
  const [synthesis, setSynthesis] = useState<SynthesisOut | null>(null);
  const [followupOpen, setFollowupOpen] = useState(false);
  // Stable callback so FollowUpDialog's focus-trap effect doesn't tear
  // down on every parent re-render (SSE ticks at ~2.5s).
  const closeFollowup = useCallback(() => setFollowupOpen(false), []);
  // Lifted from <KnowledgeGraph /> so <GraphStatsCard /> can display them
  // without duplicate-fetching /graph.
  const [graphStats, setGraphStats] = useState<GraphStats | null>(null);
  // Owned here so the "Wrap code" toggle button can live in the answer
  // toolbar (next to Download PDF). ArtifactView consumes it as a prop.
  const [wrapEnabled, setWrapEnabled] = useState(false);
  const handleGraphStats = useCallback((s: GraphStats) => setGraphStats(s), []);
  // `draft_complete` events from before a restart describe the previous
  // attempt. Without a baseline the effect below sees draftCount > 0, refetches
  // `draft=latest`, and re-displays the old answer while the restarted run is
  // still working. The `resumed` path has its own guard (hasFreshDraft), but a
  // restart with no checkpoint (refine runs never write one) emits no such
  // event, so record where the previous attempt's drafts ended.
  const [draftBaseline, setDraftBaseline] = useState(0);

  const isTerminal = run.status === 'completed' || run.status === 'failed';
  const isFollowupRun = (run.mode === 'explore' || run.mode === 'refine') && !!run.parent_run_id;
  const followupSourceRunId = isFollowupRun ? run.parent_run_id : null;
  const followupModeText = run.mode === 'refine' ? 'no exploration' : null;
  const displayQuery = run.merged_query || run.query;
  const queryLabel = isFollowupRun ? 'Follow-up query' : 'Query';
  // Follow-ups require a successful parent: a failed parent may have no
  // synthesis to seed and an incomplete KB, which would yield a degraded
  // follow-up answer.
  const canFollowUp = run.status === 'completed';
  // Open the SSE stream while the run is active. Closed runs use the events
  // already in `run.events` plus a one-shot REST refresh.
  const streamUrl = isTerminal ? null : `/api/runs/${run.id}/stream`;
  const { events: liveEvents, status: streamStatus } = useEventSource(streamUrl);

  // Combined event list (persisted + live, dedup by id where possible).
  const events = useMemo(() => {
    const persistedIds = new Set(run.events.map((e) => e.id));
    const merged = [...run.events];
    for (const e of liveEvents) {
      if (typeof e.id === 'number' && persistedIds.has(e.id)) continue;
      merged.push({
        id: typeof e.id === 'number' ? e.id : -merged.length,
        timestamp: e.timestamp ?? new Date().toISOString(),
        event: e.event,
        payload: e.payload,
      });
    }
    return merged;
  }, [run.events, liveEvents]);

  // Force the graph component to refresh whenever a new graph_update arrives.
  const graphRefreshKey = useMemo(() => {
    return events.filter((e) => e.event === 'graph_update' || e.event === 'done').length;
  }, [events]);

  // Pull the latest known phase + quick-explore progress from events so the
  // empty-state in <KnowledgeGraph /> can show "47 / 183 pages fetched"
  // instead of a generic "no graph yet" message.
  const { phase, progress } = useMemo(() => {
    // Per-phase tracking — quick_explore and kb_ingest each emit iteration
    // events with their own counters. The KnowledgeGraph empty state renders
    // only for the latest phase, so we keep a separate (n,total) per phase
    // and surface the one matching the current phase.
    const byPhase: Record<string, { n: number; total: number }> = {};
    let latestPhase: string | null = null;
    for (const e of events) {
      if (e.event !== 'iteration') continue;
      const ph = typeof e.payload?.phase === 'string' ? e.payload.phase : latestPhase;
      if (typeof e.payload?.phase === 'string') latestPhase = e.payload.phase;
      if (!ph) continue;
      const slot = byPhase[ph] ?? (byPhase[ph] = { n: 0, total: 0 });
      const n = Number(e.payload?.n);
      const t = Number(e.payload?.total);
      if (Number.isFinite(n) && n > slot.n) slot.n = n;
      if (Number.isFinite(t) && t > 0) slot.total = t;
    }
    const active = (latestPhase && byPhase[latestPhase]) || null;
    return {
      phase: latestPhase,
      progress: active && active.total > 0 ? { n: active.n, total: active.total } : null,
    };
  }, [events]);

  // Background poll while the run is non-terminal — kept on its own effect
  // so SSE frames don't tear down and recreate the interval.
  useEffect(() => {
    if (isTerminal) return;
    let cancelled = false;
    const refresh = async () => {
      try {
        const r = await api.getRun(run.id);
        if (!cancelled) setRun(r);
      } catch {
        // ignore transient
      }
    };
    const timer = setInterval(refresh, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [run.id, isTerminal]);

  // One-shot refresh when a terminal SSE event lands — keyed off the last
  // event id so we fire exactly once per terminal event, not on every render.
  const lastEvent = liveEvents[liveEvents.length - 1];
  const lastEventId = lastEvent?.id ?? null;
  useEffect(() => {
    if (!lastEvent) return;
    if (
      lastEvent.event !== 'done' &&
      lastEvent.event !== 'error' &&
      lastEvent.event !== 'draft_complete'
    ) {
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const r = await api.getRun(run.id);
        if (!cancelled) setRun(r);
      } catch {
        // ignore transient
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run.id, lastEventId, lastEvent?.event]);

  // Number of `draft_complete` events seen so far. Each new draft triggers
  // a synthesis refetch — the backend's /synthesis?draft=latest returns the
  // freshest file by mtime, so we get a rolling display of the current draft.
  const draftCount = useMemo(
    () => events.filter((e) => e.event === 'draft_complete').length,
    [events],
  );
  // Drafts produced by the attempt currently on screen.
  const freshDraftCount = draftCount - draftBaseline;

  // Restart returns the requeued summary; merging it flips this page out of its
  // terminal state, which re-arms the SSE stream and the background poll below.
  // Clearing the answer and baselining the draft count keeps the previous
  // attempt's output from being presented as this one's.
  const handleRestarted = useCallback(
    (s: RunSummary) => {
      setSynthesis(null);
      setDraftBaseline(draftCount);
      setRun((prev) => ({ ...prev, ...s }));
    },
    [draftCount],
  );

  // After a `resumed` event we suppress the displayed draft until a fresh
  // `draft_complete` lands. Caesar's synthesizer has no checkpoint and
  // restarts from draft 1 on resume; showing the pre-restart draft answer
  // misleads the user into thinking the resumed run is further along than
  // it is.
  const hasFreshDraft = useMemo(() => {
    if (run.status === 'completed') return true;
    let resumedAfterDraft = false;
    for (const e of events) {
      if (e.event === 'resumed') resumedAfterDraft = true;
      else if (e.event === 'draft_complete') resumedAfterDraft = false;
    }
    return !resumedAfterDraft;
  }, [events, run.status]);

  // On resume, drop any pre-restart synthesis so it doesn't linger while
  // the resumed Caesar is still re-generating draft 1.
  useEffect(() => {
    if (!hasFreshDraft) setSynthesis(null);
  }, [hasFreshDraft]);

  useEffect(() => {
    // Nothing to fetch yet — wait until at least one *post-resume* draft
    // has landed (or the run completed, in which case the merged file is
    // authoritative).
    if (!hasFreshDraft) return;
    if (freshDraftCount <= 0 && run.status !== 'completed') return;
    let cancelled = false;
    let attempt = 0;
    const tryFetch = async () => {
      while (!cancelled && attempt < 6) {
        try {
          const s = await api.getSynthesis(run.id, 'latest');
          if (!cancelled) setSynthesis(s);
          return;
        } catch (e) {
          attempt += 1;
          if (attempt >= 6) {
            if (!cancelled) console.warn('Could not fetch synthesis:', e);
            return;
          }
          // 0.5s, 1s, 2s, 4s, 8s — caps under 16s per draft.
          const delay = Math.min(8_000, 500 * 2 ** (attempt - 1));
          await new Promise((r) => setTimeout(r, delay));
        }
      }
    };
    void tryFetch();
    return () => {
      cancelled = true;
    };
  }, [run.id, run.status, freshDraftCount, hasFreshDraft]);

  return (
    <div className="run-page max-w-6xl mx-auto px-4 sm:px-6 py-6 sm:py-10 space-y-6 animate-fade-in">
      <header className="flex items-start justify-between gap-4">
        <div className="space-y-1 min-w-0 flex-1">
          <p className="text-xs uppercase tracking-wider text-gray-500">{queryLabel}</p>
          <h1 className="text-lg text-gray-900 leading-normal">{displayQuery}</h1>
          <p className="text-xs text-gray-400 min-w-0 truncate">
            Run <code className="font-mono">{run.id.slice(0, 8)}</code>
            {followupSourceRunId ? (
              <>
                {' '}·{' '}
                <Link
                  href={`/run/${followupSourceRunId}`}
                  aria-label={`Open parent run ${followupSourceRunId}`}
                  title="Open parent run"
                  className="hover:text-brand-800 transition-colors"
                >
                  from <code className="font-mono">{followupSourceRunId.slice(0, 8)}</code>
                </Link>
              </>
            ) : null}
            {' '}· <span>{run.preset}</span>
            {followupModeText ? <> · {followupModeText}</> : null}
            {' '}· started{' '}
            {run.started_at ? new Date(run.started_at).toLocaleString() : '—'}
          </p>
        </div>
        <div className="flex flex-col items-end gap-1 mt-1 shrink-0">
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => setFollowupOpen(true)}
              disabled={!canFollowUp}
              aria-label="Ask follow-up"
              title={
                canFollowUp
                  ? 'Ask a follow-up question'
                  : run.status === 'failed'
                    ? 'Follow-ups require a completed parent run'
                    : 'Follow-ups available once the run completes'
              }
              className="inline-flex items-center gap-1 px-2 h-7 rounded-md text-xs text-gray-600 hover:text-brand-800 hover:bg-brand-50 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:text-gray-600 disabled:hover:bg-transparent transition-colors"
            >
              <MessageCirclePlus size={14} />
              <span className="hidden sm:inline">Ask follow-up</span>
            </button>
            <RestartRunButton runId={run.id} onRestarted={handleRestarted} />
            <DeleteRunButton
              runId={run.id}
              query={run.query}
              onDeleted={() => router.push('/runs')}
            />
          </div>
          <div className="text-xs text-gray-400">
            <StreamStatusIndicator status={streamStatus} />
          </div>
        </div>
      </header>

      <FollowUpDialog
        open={followupOpen}
        onClose={closeFollowup}
        parentRunId={run.id}
        parentPreset={run.preset}
      />

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 lg:items-stretch">
        <div className="lg:col-span-3 min-w-0 min-h-[360px]">
          <KnowledgeGraph
            runId={run.graph_run_id ?? run.id}
            refreshKey={graphRefreshKey}
            phase={phase}
            progress={progress}
            onStats={handleGraphStats}
            mode={run.mode}
            parentRunId={run.parent_run_id}
            detailsHref={`/run/${run.id}/graph`}
          />
        </div>
        <div className="lg:col-span-2 space-y-6">
          <LiveProgress run={run} events={events} />
          <GraphStatsCard stats={graphStats} />
        </div>
      </div>

      {synthesis ? (
        <section
          className="answer-card mt-6 rounded-2xl border border-gray-200 bg-white p-6"
        >
          {/* Query block — hidden on screen (already shown in the page header),
              but rendered first in the printed PDF, above the Final Answer heading.
              For follow-ups, print the full query Caesar answered under the
              follow-up label instead of adding a second raw-query line. */}
          {run.query && (
            <div className="hidden print:block mb-6 pb-4 border-b border-gray-300">
              <h3 className="text-xs font-medium text-gray-600 uppercase tracking-wider mb-1 print:text-black">
                {isFollowupRun ? 'Follow-up query' : 'Query'}
              </h3>
              <p className="text-lg text-gray-900 print:text-black">{displayQuery}</p>
            </div>
          )}
          <div className="flex items-baseline justify-between mb-4 gap-3 flex-wrap">
            <h2 className="text-sm font-medium text-gray-500 uppercase tracking-wider">
              {synthesis.draft === 'merged' ? 'Final Answer' : `Draft ${synthesis.draft} answer`}
            </h2>
            <div className="flex items-center gap-3 print:hidden">
              {synthesis.draft !== 'merged' && run.status === 'running' && (
                <span className="text-xs text-brand-700 inline-flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-brand-600 animate-pulse-slow" />
                  Caesar is still refining. Newer drafts will replace this.
                </span>
              )}
              {(/```/.test(synthesis.artifact) || /```/.test(synthesis.abstract ?? '')) && (
                <button
                  type="button"
                  onClick={() => setWrapEnabled((v) => !v)}
                  aria-pressed={wrapEnabled}
                  title={wrapEnabled ? 'Disable code wrap' : 'Enable code wrap'}
                  className={SECONDARY_BUTTON_CLASS}
                >
                  <WrapText size={14} />
                  {wrapEnabled ? 'Unwrap code' : 'Wrap code'}
                </button>
              )}
              <button
                type="button"
                onClick={async () => {
                  // Wait for every artifact image to finish loading before
                  // firing the print dialog — Chrome/Safari can otherwise
                  // capture below-the-fold images mid-decode.
                  const imgs = Array.from(
                    document.querySelectorAll<HTMLImageElement>('.answer-prose img'),
                  );
                  await Promise.all(
                    imgs.map((i) =>
                      i.complete && i.naturalHeight > 0
                        ? Promise.resolve()
                        : new Promise<void>((resolve) => {
                            i.addEventListener('load', () => resolve(), { once: true });
                            i.addEventListener('error', () => resolve(), { once: true });
                          }),
                    ),
                  );
                  window.print();
                }}
                title="Save the answer as a PDF (opens your browser's print dialog)"
                className={SECONDARY_BUTTON_CLASS}
              >
                <Download size={14} />
                Download PDF
              </button>
              <HelpTip label="The synthesized answer. The summary up top is a quick overview; interim drafts show while the run works, replaced by the Final Answer when it completes." />
            </div>
          </div>
          <ArtifactView synthesis={synthesis} runId={run.id} wrapEnabled={wrapEnabled} />
          <SourcesPanel sources={synthesis.sources} />
        </section>
      ) : run.status === 'completed' || (freshDraftCount > 0 && hasFreshDraft) ? (
        <section className="mt-6 rounded-2xl border border-gray-200 bg-white p-6">
          <div className="skeleton h-32" />
          <div className="mt-3 text-xs text-gray-500">
            {run.status === 'completed' ? 'Loading the final answer…' : 'Loading the latest draft…'}
          </div>
        </section>
      ) : null}
    </div>
  );
}

function StreamStatusIndicator({
  status,
}: {
  status: 'connecting' | 'open' | 'closed' | 'error';
}) {
  if (status === 'closed') return null;
  const dot = 'w-1.5 h-1.5 rounded-full';
  if (status === 'connecting') {
    return (
      <span className="shrink-0 inline-flex items-center gap-1.5">
        <span className={`${dot} bg-gray-400 animate-pulse-slow`} />
        Connecting…
      </span>
    );
  }
  if (status === 'error') {
    return (
      <span className="shrink-0 inline-flex items-center gap-1.5 text-red-600">
        <span className={`${dot} bg-red-500`} />
        Stream disconnected. Retrying...
      </span>
    );
  }
  return (
    <span className="shrink-0 inline-flex items-center gap-1.5">
      <span className={`${dot} bg-green-500 animate-pulse-slow`} />
      Live
    </span>
  );
}

function GraphStatsCard({ stats }: { stats: GraphStats | null }) {
  const fmt = (n: number | undefined) =>
    n != null && n > 0 ? n.toLocaleString() : '—';
  return (
    <div className="relative rounded-2xl border border-gray-200 bg-white p-6">
      <HelpTip
        className="absolute right-4 top-4"
        label="Nodes = pages found. Edges = links between them. Max depth = furthest hop from the starting page."
      />
      <div className="grid grid-cols-3 text-center divide-x divide-gray-100">
        <div className="px-2">
          <div className="text-xs text-gray-500">Nodes</div>
          <div className="text-lg font-medium tabular-nums">{fmt(stats?.nodes)}</div>
        </div>
        <div className="px-2">
          <div className="text-xs text-gray-500">Edges</div>
          <div className="text-lg font-medium tabular-nums">{fmt(stats?.edges)}</div>
        </div>
        <div className="px-2">
          <div className="text-xs text-gray-500">Max depth</div>
          <div className="text-lg font-medium tabular-nums">{fmt(stats?.maxDepth)}</div>
        </div>
      </div>
    </div>
  );
}
