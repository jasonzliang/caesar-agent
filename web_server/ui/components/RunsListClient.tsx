'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import type { RunSummary } from '@/lib/api';
import { fmtCost, fmtDuration, fmtNodes, fmtRelative, parseBackendDate, runElapsedSec, useNow } from '@/lib/utils';
import { StatusBadge } from './StatusBadge';
import { FollowUpBadge } from './FollowUpBadge';
import { DeleteRunButton } from './DeleteRunButton';
import { RestartRunButton } from './RestartRunButton';
import { WipeAllButton } from './WipeAllButton';

const SELECT_CLASS =
  'h-9 rounded-lg border border-gray-200 bg-white px-2.5 text-sm text-gray-700 ' +
  'focus:outline-none focus:ring-2 focus:ring-brand-200';

export function RunsListClient({ initial }: { initial: RunSummary[] }) {
  const router = useRouter();
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [q, setQ] = useState('');
  const [dur, setDur] = useState('any'); // any | 0-15 | 15-60 | 60-300 | over  (minutes)
  const [cost, setCost] = useState('any'); // any | lt1 | 1-5 | 5-30 | over  ($)
  const [age, setAge] = useState('any'); // any | hour | day | week | older
  // Admin (public-mode operator step-up) sees every user's runs; when admin we
  // label each card with its owner so the mixed list is navigable.
  const [isAdmin, setIsAdmin] = useState(false);
  const visible = initial.filter((r) => !hidden.has(r.id));
  const now = useNow(60_000);

  useEffect(() => {
    fetch('/api/whoami', { cache: 'no-store' })
      .then((r) => r.json())
      .then((d) => setIsAdmin(Boolean(d?.is_admin)))
      .catch(() => setIsAdmin(false));
  }, []);

  if (initial.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-gray-300 bg-white p-10 text-center">
        <p className="text-gray-500 mb-3">No runs yet.</p>
        <a
          href="/"
          className="inline-flex items-center rounded-lg bg-brand-800 hover:bg-brand-900 text-white text-sm font-medium px-4 py-2 transition-colors"
        >
          Run Caesar
        </a>
      </div>
    );
  }

  if (visible.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-gray-300 bg-white p-10 text-center text-sm text-gray-500">
        All runs deleted in this session. Refresh to fetch the latest.
      </div>
    );
  }

  // Elapsed wall-clock (seconds); running/queued runs tick against `now`.
  const elapsedSec = (r: RunSummary): number | null => runElapsedSec(r, now);

  const needle = q.trim().toLowerCase();
  const anyFilter = !!needle || dur !== 'any' || cost !== 'any' || age !== 'any';
  const H = 3_600_000;
  const D = 24 * H;

  const filtered = visible.filter((r) => {
    if (needle) {
      const dq = (r.merged_query || r.query || '').toLowerCase();
      if (
        !dq.includes(needle) &&
        !(r.preset || '').toLowerCase().includes(needle) &&
        !(r.preset_label || '').toLowerCase().includes(needle) &&
        !(r.status || '').toLowerCase().includes(needle)
      ) {
        return false;
      }
    }
    if (dur !== 'any') {
      const min = elapsedSec(r); // seconds; buckets are in minutes
      if (min == null) return false;
      if (dur === '0-15' && !(min < 900)) return false;
      if (dur === '15-60' && !(min >= 900 && min < 3600)) return false;
      if (dur === '60-300' && !(min >= 3600 && min < 18000)) return false;
      if (dur === 'over' && !(min >= 18000)) return false;
    }
    if (cost !== 'any') {
      const c = r.total_cost_usd;
      if (c == null) return false;
      if (cost === 'lt1' && !(c < 1)) return false;
      if (cost === '1-5' && !(c >= 1 && c < 5)) return false;
      if (cost === '5-30' && !(c >= 5 && c < 30)) return false;
      if (cost === 'over' && !(c >= 30)) return false;
    }
    if (age !== 'any' && now != null) {
      const created = parseBackendDate(r.created_at);
      if (created == null) return false;
      const ageMs = now - created;
      if (age === 'hour' && !(ageMs < H)) return false;
      if (age === 'day' && !(ageMs < D)) return false;
      if (age === 'week' && !(ageMs < 7 * D)) return false;
      if (age === 'older' && !(ageMs >= 7 * D)) return false;
    }
    return true;
  });

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <input
          type="text"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search query, status, or preset…"
          className="h-9 w-64 max-w-full rounded-lg border border-gray-200 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-brand-200"
        />
        <select value={cost} onChange={(e) => setCost(e.target.value)} className={SELECT_CLASS} title="Filter by cost">
          <option value="any">Any cost</option>
          <option value="lt1">Under $1</option>
          <option value="1-5">$1-$5</option>
          <option value="5-30">$5-$30</option>
          <option value="over">Over $30</option>
        </select>
        <select value={dur} onChange={(e) => setDur(e.target.value)} className={SELECT_CLASS} title="Filter by run duration">
          <option value="any">Any duration</option>
          <option value="0-15">Under 15 min</option>
          <option value="15-60">15-60 min</option>
          <option value="60-300">60-300 min</option>
          <option value="over">Over 300 min</option>
        </select>
        <select value={age} onChange={(e) => setAge(e.target.value)} className={SELECT_CLASS} title="Filter by how long ago it ran">
          <option value="any">Any time</option>
          <option value="hour">Past hour</option>
          <option value="day">Past 24 hours</option>
          <option value="week">Past 7 days</option>
          <option value="older">Older than 7 days</option>
        </select>
        {anyFilter && (
          <span className="text-xs text-gray-400 tabular-nums">
            showing {filtered.length} of {visible.length}
          </span>
        )}
        <div className="ml-auto">
          <WipeAllButton runCount={initial.length} />
        </div>
      </div>

      {filtered.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-gray-300 bg-white p-10 text-center text-sm text-gray-500">
          No runs match your search and filters.
        </div>
      ) : (
        <div className="rounded-2xl border border-gray-200 bg-white overflow-hidden divide-y divide-gray-100">
          {filtered.map((r) => {
            const elapsed = runElapsedSec(r, now);
            const displayQuery = r.merged_query || r.query;
            return (
              <a
                key={r.id}
                href={`/run/${r.id}`}
                className="group block px-5 py-4 hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center justify-between gap-3 mb-1">
                  <div className="flex items-center gap-2 min-w-0">
                    <StatusBadge status={r.status} />
                    <FollowUpBadge mode={r.mode} parentRunId={r.parent_run_id} />
                  </div>
                  <div className="flex items-center gap-1">
                    <span className="text-xs text-gray-400" suppressHydrationWarning>
                      {fmtRelative(r.created_at, now)}
                    </span>
                    {/* A failed row keeps it visible (it is the action that row
                        is asking for); on the rest it reveals on hover like
                        Delete, so a long list stays quiet. */}
                    <RestartRunButton
                      runId={r.id}
                      stopPropagation
                      onRestarted={() => router.refresh()}
                      className={
                        r.status === 'failed'
                          ? undefined
                          : 'opacity-0 group-hover:opacity-100 focus:opacity-100'
                      }
                    />
                    <DeleteRunButton
                      runId={r.id}
                      query={displayQuery}
                      stopPropagation
                      onDeleted={() => setHidden((h) => new Set(h).add(r.id))}
                      className="opacity-0 group-hover:opacity-100 focus:opacity-100"
                    />
                  </div>
                </div>
                <p className="text-sm text-gray-900 line-clamp-2">{displayQuery}</p>
                <div className="text-xs text-gray-500 mt-1.5 flex flex-wrap gap-4">
                  <span className="capitalize">{r.preset_label ?? r.preset}</span>
                  {/* Hide null pills rather than rendering "—": cost is null
                      until the watchdog populates it (~1.5s past agent warmup),
                      and graph size stays null until Caesar has more than its
                      seed node. */}
                  {r.total_cost_usd != null && <span>{fmtCost(r.total_cost_usd)}</span>}
                  {r.graph_node_count != null && <span>{fmtNodes(r.graph_node_count)}</span>}
                  {elapsed != null && <span>{fmtDuration(elapsed)}</span>}
                  {/* Run id (always) + caesar user id (admin only), pinned to
                      the right; preset/cost/nodes/duration stay on the left.
                      Full ids on hover. */}
                  <span
                    className="ml-auto font-mono font-semibold text-gray-700"
                    title={`run ${r.id}${isAdmin && r.owner_id ? ` · user ${r.owner_id}` : ''}`}
                  >
                    run {r.id.slice(0, 8)}
                    {isAdmin && <> · user {r.owner_id ? r.owner_id.slice(0, 8) : '—'}</>}
                  </span>
                </div>
              </a>
            );
          })}
        </div>
      )}
    </div>
  );
}
