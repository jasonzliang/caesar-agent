'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { api, type GraphOut } from '@/lib/api';
import { HelpTip } from './HelpTip';
import { Modal } from './Modal';
import { SearchResultsView } from './SearchResultsView';

// A read-only, sortable/searchable table view of a run's knowledge graph.
// Reuses the existing /graph endpoint (nodes carry url/depth/insights/iteration/
// visit_count, edges carry source/target/reason) and computes in/out adjacency
// client-side. No backend change and no new dependency — plain Tailwind.

type NeighborLink = { url: string; reason: string | null };
type Row = {
  url: string;
  depth: number;
  inDeg: number;
  outDeg: number;
  inLinks: NeighborLink[];
  outLinks: NeighborLink[];
  visits: number | null;
  iteration: number | null;
  insight: string;
  isRoot: boolean;
};
type SortKey = 'url' | 'depth' | 'inDeg' | 'outDeg' | 'visits' | 'iteration';
type ModalState = { type: 'insight' | 'in' | 'out' | 'search'; row: Row } | null;

// Mirror KnowledgeGraph's root pick: the reported starting_url if it matches a
// node, else the lowest-depth node (lexicographic tiebreak for stability).
function chooseRoot(nodes: { id: string; depth: number }[], startingUrl: string | null): string | null {
  if (startingUrl && nodes.some((n) => n.id === startingUrl)) return startingUrl;
  let best: string | null = null;
  let bestD = Infinity;
  for (const n of nodes) {
    if (n.depth < bestD || (n.depth === bestD && best !== null && n.id < best)) {
      bestD = n.depth;
      best = n.id;
    }
  }
  return best;
}

// Strip markdown to a clean one-line preview for the collapsed insight cell.
function toPreview(md: string): string {
  return md
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/!\[[^\]]*\]\([^)]*\)/g, '')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/^\s{0,3}#{1,6}\s+/gm, '')
    .replace(/[*_~>]/g, '')
    .replace(/^\s*[-+]\s+/gm, '')
    .replace(/^\s*\d+\.\s+/gm, '')
    .replace(/\s+/g, ' ')
    .trim();
}

// A URL rendered as a link, unless it's a server-local file:// seed page (which
// can't be opened in a browser) — those render as plain, unclickable text.
function UrlText({ url }: { url: string }) {
  if (url.startsWith('file://')) {
    // Local search-results seed page: a server-side file path, not browsable.
    // Show a friendly label (full path available on hover) instead of the path.
    return (
      <span title={url} className="text-xs text-gray-500">
        Starting Search Page
      </span>
    );
  }
  return (
    <a
      href={url}
      target="_blank"
      rel="noopener"
      className="font-mono text-xs leading-snug text-brand-800 hover:underline break-all"
    >
      {url}
    </a>
  );
}

export function GraphTableClient({
  runId,
  graphRunId,
  query,
}: {
  runId: string;
  graphRunId: string;
  query: string;
}) {
  const [graph, setGraph] = useState<GraphOut | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [q, setQ] = useState('');
  const [sort, setSort] = useState<{ key: SortKey; dir: 'asc' | 'desc' }>({
    key: 'depth',
    dir: 'asc',
  });
  const [modal, setModal] = useState<ModalState>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const g = await api.getGraph(graphRunId, 'latest');
        if (!cancelled) setGraph(g);
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [graphRunId]);

  const rows: Row[] = useMemo(() => {
    if (!graph) return [];
    const inMap = new Map<string, NeighborLink[]>();
    const outMap = new Map<string, NeighborLink[]>();
    const add = (m: Map<string, NeighborLink[]>, k: string, v: NeighborLink) => {
      const a = m.get(k);
      if (a) a.push(v);
      else m.set(k, [v]);
    };
    for (const e of graph.edges) {
      add(outMap, e.source, { url: e.target, reason: e.reason });
      add(inMap, e.target, { url: e.source, reason: e.reason });
    }
    const root = chooseRoot(graph.nodes, graph.starting_url);
    return graph.nodes.map((n) => {
      const inLinks = inMap.get(n.id) ?? [];
      const outLinks = outMap.get(n.id) ?? [];
      return {
        url: n.id,
        depth: n.depth,
        inDeg: inLinks.length,
        outDeg: outLinks.length,
        inLinks,
        outLinks,
        visits: n.visit_count,
        iteration: n.iteration,
        insight: (n.insights ?? '').trim(),
        isRoot: n.id === root,
      };
    });
  }, [graph]);

  const maxDepth = useMemo(() => rows.reduce((m, r) => Math.max(m, r.depth), 0), [rows]);

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const base = needle
      ? rows.filter(
          (r) => r.url.toLowerCase().includes(needle) || r.insight.toLowerCase().includes(needle),
        )
      : rows;
    const dir = sort.dir === 'asc' ? 1 : -1;
    const { key } = sort;
    return [...base].sort((a, b) => {
      const av: number | string = key === 'url' ? a.url : (a[key] ?? -1);
      const bv: number | string = key === 'url' ? b.url : (b[key] ?? -1);
      if (av < bv) return -1 * dir;
      if (av > bv) return 1 * dir;
      // Stable secondary sort by URL so equal-degree rows don't jitter.
      return a.url < b.url ? -1 : a.url > b.url ? 1 : 0;
    });
  }, [rows, q, sort]);

  const toggleSort = (key: SortKey) =>
    setSort((s) =>
      s.key === key
        ? { key, dir: s.dir === 'asc' ? 'desc' : 'asc' }
        : { key, dir: key === 'url' ? 'asc' : 'desc' },
    );

  const exportCsv = () => {
    const esc = (v: string | number) =>
      `"${String(v).replace(/"/g, '""').replace(/\r?\n/g, ' ')}"`;
    const header = ['url', 'depth', 'in_degree', 'out_degree', 'visits', 'iteration', 'is_root', 'insights'];
    const lines = [header.join(',')];
    for (const r of filtered) {
      lines.push(
        [r.url, r.depth, r.inDeg, r.outDeg, r.visits ?? 0, r.iteration ?? '', r.isRoot ? 'yes' : 'no', r.insight]
          .map(esc)
          .join(','),
      );
    }
    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `table_view_${runId.slice(0, 8)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const arrow = (key: SortKey) => (sort.key === key ? (sort.dir === 'asc' ? ' ↑' : ' ↓') : '');

  const numHeader = (key: SortKey, label: string, title: string) => (
    <th className="px-3 py-2 text-right font-medium">
      <button
        type="button"
        onClick={() => toggleSort(key)}
        title={title}
        className="inline-flex items-center gap-0.5 hover:text-gray-900 tabular-nums"
      >
        {label}
        <span className="text-gray-400">{arrow(key)}</span>
      </button>
    </th>
  );

  // A clickable degree cell that opens the neighbor-URL popup (plain 0 when none).
  const degreeCell = (r: Row, type: 'in' | 'out') => {
    const deg = type === 'in' ? r.inDeg : r.outDeg;
    return (
      <td className="px-3 py-2 text-right tabular-nums">
        {deg > 0 ? (
          <button
            type="button"
            onClick={() => setModal({ type, row: r })}
            title={type === 'in' ? 'Show pages that link to this node' : 'Show pages this node links to'}
            className="text-brand-800 hover:underline tabular-nums"
          >
            {deg}
          </button>
        ) : (
          <span className="text-gray-300">0</span>
        )}
      </td>
    );
  };

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-6 sm:py-10 space-y-6 animate-fade-in">
      <header className="space-y-1">
        <Link
          href={`/run/${runId}`}
          className="text-sm text-brand-800 hover:underline inline-flex items-center gap-1"
        >
          ← Back to run
        </Link>
        <div className="flex items-start justify-between gap-3">
          <h1 className="text-2xl font-semibold tracking-tight text-gray-900">
            Knowledge Graph Table View
          </h1>
          <HelpTip
            className="mt-1.5 shrink-0"
            label="Every page Caesar visited while building this run's knowledge graph, one row per node. Columns show each page's URL, depth from the starting page, in/out links, visit count, and the insight Caesar recorded there. Click a link count or an insight to expand it, sort by any column, search, or export the table as CSV."
          />
        </div>
        <p className="text-sm text-gray-500 whitespace-pre-wrap">
          <span className="font-medium text-gray-700">Query:</span> {query}
        </p>
      </header>

      {loading ? (
        <div className="skeleton h-64 rounded-2xl" />
      ) : error ? (
        <div className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          Could not load the graph: {error}
        </div>
      ) : rows.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-gray-300 bg-white p-10 text-center text-sm text-gray-500">
          This run has no graph nodes yet.
        </div>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-3">
            <div className="text-sm text-gray-500 tabular-nums">
              <span className="font-medium text-gray-900">{rows.length}</span> nodes ·{' '}
              <span className="font-medium text-gray-900">{graph?.edges.length ?? 0}</span> edges ·
              max depth <span className="font-medium text-gray-900">{maxDepth}</span>
            </div>
            <div className="flex-1" />
            <input
              type="text"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search URLs and insights…"
              className="w-64 max-w-full rounded-lg border border-gray-200 px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-brand-200"
            />
            <button
              type="button"
              onClick={exportCsv}
              className="inline-flex items-center gap-1.5 rounded-lg border border-gray-200 bg-white px-2.5 py-1.5 text-xs font-medium text-gray-700 hover:bg-gray-50 hover:border-gray-300 transition-colors"
            >
              Export CSV
            </button>
          </div>

          {q && (
            <p className="-mt-3 text-xs text-gray-400 tabular-nums">
              showing {filtered.length} of {rows.length}
            </p>
          )}

          <div className="rounded-2xl border border-gray-200 bg-white overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="text-xs text-gray-500 uppercase tracking-wider bg-gray-50 border-b border-gray-200">
                  <tr>
                    <th className="px-3 py-2 text-right font-medium w-10">#</th>
                    <th className="px-3 py-2 text-left font-medium">
                      <button
                        type="button"
                        onClick={() => toggleSort('url')}
                        className="inline-flex items-center gap-0.5 hover:text-gray-900"
                      >
                        URL<span className="text-gray-400">{arrow('url')}</span>
                      </button>
                    </th>
                    {numHeader('depth', 'Depth', 'BFS distance from the starting page')}
                    {numHeader('inDeg', 'In', 'In-degree: pages that link to this one (click a value for the list)')}
                    {numHeader('outDeg', 'Out', 'Out-degree: links from this page (click a value for the list)')}
                    {numHeader('visits', 'Visits', 'How many times this page was visited/revisited during exploration')}
                    <th className="px-3 py-2 text-left font-medium">Insight</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((r, i) => {
                    const hasInsight = r.insight.length > 0;
                    return (
                      <tr key={r.url} className="border-t border-gray-100 align-top hover:bg-gray-50">
                        <td className="px-3 py-2 text-right text-gray-400 tabular-nums">{i + 1}</td>
                        <td className="px-3 py-2 min-w-[24rem]">
                          {r.url.startsWith('file://') ? (
                            <button
                              type="button"
                              onClick={() => setModal({ type: 'search', row: r })}
                              title="View the search results this run started from"
                              className="text-xs text-brand-800 hover:underline"
                            >
                              Starting Search Page
                            </button>
                          ) : (
                            <UrlText url={r.url} />
                          )}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">{r.depth}</td>
                        {degreeCell(r, 'in')}
                        {degreeCell(r, 'out')}
                        <td className="px-3 py-2 text-right tabular-nums">{r.visits ?? 0}</td>
                        <td className="px-3 py-2 text-gray-700 min-w-[28rem]">
                          {hasInsight ? (
                            <button
                              type="button"
                              onClick={() => setModal({ type: 'insight', row: r })}
                              title="Show full insight"
                              className="text-left w-full line-clamp-2 hover:text-gray-900"
                            >
                              {toPreview(r.insight)}
                            </button>
                          ) : (
                            <span className="text-gray-400 italic">
                              No insights generated (page was not visited).
                            </span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {modal && (
        <Modal
          onClose={() => setModal(null)}
          title={
            modal.type === 'insight'
              ? 'Insight'
              : modal.type === 'search'
                ? 'Starting Search Page'
                : modal.type === 'in'
                  ? `${modal.row.inDeg} ${modal.row.inDeg === 1 ? 'page links' : 'pages link'} to this page`
                  : `This page links to ${modal.row.outDeg} ${modal.row.outDeg === 1 ? 'page' : 'pages'}`
          }
        >
          {modal.type === 'insight' ? (
            <div className="prose prose-sm max-w-none answer-prose text-gray-700">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{modal.row.insight}</ReactMarkdown>
            </div>
          ) : modal.type === 'search' ? (
            <SearchResultsView runId={graphRunId} />
          ) : (
            (() => {
              const links = modal.type === 'in' ? modal.row.inLinks : modal.row.outLinks;
              if (links.length === 0) return <p className="text-sm text-gray-400">None.</p>;
              return (
                <ul className="divide-y divide-gray-100">
                  {links.map((l, idx) => (
                    <li key={`${l.url}-${idx}`} className="py-2">
                      <UrlText url={l.url} />
                      {l.reason && (
                        <p className="mt-0.5 text-xs text-gray-500 leading-snug">{l.reason}</p>
                      )}
                    </li>
                  ))}
                </ul>
              );
            })()
          )}
        </Modal>
      )}
    </div>
  );
}
