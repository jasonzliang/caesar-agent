'use client';

import { useState } from 'react';
import type { RunSummary } from '@/lib/api';
import { fmtCost, fmtDuration, fmtNodes, fmtRelative, runElapsedSec, useNow } from '@/lib/utils';
import { StatusBadge } from './StatusBadge';
import { FollowUpBadge } from './FollowUpBadge';
import { DeleteRunButton } from './DeleteRunButton';

export function RecentRunsClient({ initial }: { initial: RunSummary[] }) {
  // Optimistic-hide deleted rows immediately so the user sees instant feedback,
  // even before the router.refresh() round-trip completes.
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const visible = initial.filter((r) => !hidden.has(r.id));
  // Tick once a minute. Returns null on the server / first paint so the
  // initial HTML matches between SSR and CSR; relative-time labels populate
  // after mount.
  const now = useNow(60_000);

  if (initial.length === 0) {
    return (
      <div className="text-sm text-gray-500 italic">
        No runs yet. Submit a question to see Caesar in action.
      </div>
    );
  }
  if (visible.length === 0) {
    return (
      <div className="text-sm text-gray-500 italic">All visible runs deleted. Refresh for more.</div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
      {visible.map((r) => {
        // For running/queued runs finished_at is null — fall back to the
        // live wall clock so the elapsed pill ticks during the run instead
        // of going blank until completion. `now` is null on first paint
        // (pre-mount), which keeps SSR / CSR markup in sync.
        const elapsed = runElapsedSec(r, now);
        const displayQuery = r.merged_query || r.query;
        return (
        <a
          key={r.id}
          href={`/run/${r.id}`}
          className="group relative block rounded-xl border border-gray-200 bg-white p-4 hover:border-brand-200 hover:shadow-sm transition-all"
        >
          <div className="flex items-center justify-between gap-2 mb-1">
            <div className="flex items-center gap-2 min-w-0">
              <StatusBadge status={r.status} />
              <FollowUpBadge mode={r.mode} parentRunId={r.parent_run_id} />
            </div>
            <div className="flex items-center gap-1">
              <span className="text-xs text-gray-400" suppressHydrationWarning>
                {fmtRelative(r.created_at, now)}
              </span>
              <DeleteRunButton
                runId={r.id}
                query={displayQuery}
                stopPropagation
                onDeleted={() => setHidden((h) => new Set(h).add(r.id))}
                className="opacity-0 group-hover:opacity-100 focus:opacity-100"
              />
            </div>
          </div>
          <p className="text-sm text-gray-800 line-clamp-2 mb-1">{displayQuery}</p>
          <div className="text-xs text-gray-500 flex gap-3 mt-1.5">
            <span className="capitalize">{r.preset}</span>
            {/* Hide null pills entirely rather than render fmtCost(null)→"—".
                Caesar's agent warmup takes a few seconds before the watchdog
                populates live_cost_usd; without this guard a freshly launched
                run shows "deepest · — · 30s" with a stray dash. */}
            {r.total_cost_usd != null && <span>{fmtCost(r.total_cost_usd)}</span>}
            {r.graph_node_count != null && <span>{fmtNodes(r.graph_node_count)}</span>}
            {elapsed != null && <span>{fmtDuration(elapsed)}</span>}
          </div>
        </a>
        );
      })}
    </div>
  );
}
