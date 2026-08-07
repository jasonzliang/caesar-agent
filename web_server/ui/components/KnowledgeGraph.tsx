'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { HelpTip } from './HelpTip';
import { Modal } from './Modal';
import { SearchResultsView } from './SearchResultsView';
// d3-force-3d ships no types; re-uses the d3-force API surface so the
// minimal shim below is sufficient for what we use (force factories
// returning configurable force objects). Self-referential return types
// let us chain `.strength(n).distanceMax(m)` etc.
type ChargeForce = { strength: (n: number) => ChargeForce; distanceMax: (n: number) => ChargeForce };
type CollideForce = { radius: (n: number) => CollideForce };
// eslint-disable-next-line @typescript-eslint/no-require-imports
const { forceManyBody, forceCollide } = require('d3-force-3d') as {
  forceManyBody: () => ChargeForce;
  forceCollide: (radius: number) => CollideForce;
};
import { api, type GraphOut } from '@/lib/api';
import { ProgressBar } from './ProgressBar';

// Padding (px) around the auto-fit bounding box so nodes near the edge
// don't render flush against the canvas border.
const ZOOM_FIT_PADDING = 60;

// Node radius (px) — the visual circle drawn in nodeCanvasObject and the
// nodeRelSize hint to react-force-graph. Kept as one constant so the
// collide radius (NODE_R + breathing room) can derive from it.
const NODE_R = 5;

// react-force-graph-2d uses HTMLCanvasElement APIs that aren't available
// in Node, so dynamically import it client-only.
const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), { ssr: false });

// x/y are written by d3-force at runtime; we also seed them on the merge
// path so brand-new nodes don't spawn at origin (see merge logic below).
type GraphNodeData = {
  id: string;
  depth: number;
  iteration: number | null;
  insights: string | null;
  x?: number;
  y?: number;
};
type GraphState = {
  nodes: GraphNodeData[];
  links: { source: string; target: string; reason?: string | null }[];
  startingUrl: string | null;
  // BFS distance + 1 from startingUrl (root=1); viz-only override for older runs
  depthById: Map<string, number>;
  maxDepth: number;
};

export type GraphStats = { nodes: number; edges: number; maxDepth: number };

type Props = {
  runId: string;
  refreshKey: number; // bump to force a refetch
  phase?: string | null;             // 'quick_explore' | etc. — informs the empty-state copy
  progress?: { n: number; total: number } | null;
  onStats?: (s: GraphStats) => void; // lift graph stats up for the summary card
  // Run mode + parent linkage — used to produce a sensible empty-state
  // for follow-up runs (refine never writes a graph; explore writes only
  // its own deltas on top of the inherited KB).
  mode?: 'new' | 'explore' | 'refine';
  parentRunId?: string | null;
  detailsHref?: string; // link to the standalone node-table page, if available
};

// maxDepth=0 so the lifted-up stats card renders "—" for Max depth until
// real data lands. A non-zero default would surface a confusing "1" on
// the summary card before the first /graph fetch resolves.
const EMPTY: GraphState = { nodes: [], links: [], startingUrl: null, depthById: new Map(), maxDepth: 0 };

// Distinct red for the starting URL so it pops out regardless of where the depth gradient ends up
const ROOT_NODE_COLOR = '#ef4444';

// Canonical viridis stops; node outline (nodeCanvasObject) keeps the bright yellow readable on white
const PALETTE_STOPS = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'];

function hexToRgb(h: string): [number, number, number] {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function rgbToHex(r: number, g: number, b: number): string {
  const toHex = (v: number) => Math.round(v).toString(16).padStart(2, '0');
  return '#' + toHex(r) + toHex(g) + toHex(b);
}

// Map depth ∈ [1, maxDepth] linearly across PALETTE_STOPS so every depth gets a distinct color
function paletteFor(depth: number, maxDepth: number): string {
  const stops = PALETTE_STOPS;
  if (maxDepth <= 1 || depth <= 1) return stops[0];
  const t = Math.min(1, Math.max(0, (depth - 1) / (maxDepth - 1)));
  const segs = stops.length - 1;
  const segIdx = Math.min(segs - 1, Math.floor(t * segs));
  const segT = t * segs - segIdx;
  const a = hexToRgb(stops[segIdx]);
  const b = hexToRgb(stops[segIdx + 1]);
  return rgbToHex(a[0] + (b[0] - a[0]) * segT, a[1] + (b[1] - a[1]) * segT, a[2] + (b[2] - a[2]) * segT);
}

// Pick a root for BFS even when the reported `starting_url` doesn't match
// any node (Caesar has been seen to stamp a stale starting_url under some
// resume/multi-query scenarios). Fall back to the lowest-depth node in the
// graph — Caesar adds nodes in iteration order so depth=1 is the actual
// starting page.
function chooseRoot(
  nodes: { id: string; depth: number }[],
  startingUrl: string | null,
): string | null {
  if (startingUrl) {
    for (const n of nodes) if (n.id === startingUrl) return startingUrl;
  }
  // Deterministic tiebreak: lowest depth, then lexicographically smallest id.
  // Without the secondary key, two depth=1 siblings would flip-flop the root
  // highlight between polls (the dict iteration order is process-stable on
  // a given run but not across resume).
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

// Undirected BFS distance from root — viz wants path length regardless of edge direction
function bfsDepths(
  nodeIds: string[],
  links: { source: string; target: string }[],
  root: string | null,
): Map<string, number> {
  const out = new Map<string, number>();
  if (!root || !nodeIds.includes(root)) return out;
  const adj = new Map<string, string[]>();
  for (const id of nodeIds) adj.set(id, []);
  for (const l of links) {
    adj.get(l.source)?.push(l.target);
    adj.get(l.target)?.push(l.source);
  }
  out.set(root, 1);
  const queue: string[] = [root];
  while (queue.length) {
    const u = queue.shift()!;
    const du = out.get(u)!;
    for (const v of adj.get(u) ?? []) {
      if (!out.has(v)) {
        out.set(v, du + 1);
        queue.push(v);
      }
    }
  }
  return out;
}

export function KnowledgeGraph({ runId, refreshKey, phase, progress, onStats, mode, parentRunId, detailsHref }: Props) {
  const [graph, setGraph] = useState<GraphState>(EMPTY);
  const [error, setError] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<any>(null);
  const zoomedOnce = useRef(false);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [searchOpen, setSearchOpen] = useState(false);

  // Fetch the latest graph snapshot whenever refreshKey changes.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data: GraphOut = await api.getGraph(runId, 'latest');
        if (cancelled) return;
        // Preserve node-object identity across refreshes. react-force-graph
        // stores simulation state (x, y, vx, vy) on the node objects it
        // received; allocating fresh objects every refetch makes the
        // library treat them as "new" and restart the layout from random
        // positions. By reusing existing references (mutating only the
        // metadata fields), incremental updates land smoothly instead of
        // scrambling the whole graph — particularly visible at run
        // completion when the `done` refresh fires.
        setGraph((prev) => {
          const prevById = new Map(prev.nodes.map((n) => [n.id, n]));
          // Index incoming edges per node so brand-new nodes can be seeded
          // near an already-laid-out neighbor instead of spawning at (0,0).
          // Without this seed, every new node arrives at the origin and
          // forces a full alpha=1 reheat that scrambles settled positions.
          const newNodeAnchor = new Map<string, GraphNodeData>();
          for (const e of data.edges) {
            const newId = !prevById.has(e.source) ? e.source : !prevById.has(e.target) ? e.target : null;
            const oldId = newId === e.source ? e.target : newId === e.target ? e.source : null;
            if (newId == null || oldId == null) continue;
            const anchor = prevById.get(oldId);
            if (anchor && typeof anchor.x === 'number' && typeof anchor.y === 'number'
                && !newNodeAnchor.has(newId)) {
              newNodeAnchor.set(newId, anchor);
            }
          }
          const jitter = () => (Math.random() - 0.5) * 30;
          const nodes: GraphNodeData[] = data.nodes.map((n) => {
            const existing = prevById.get(n.id);
            if (existing) {
              existing.depth = n.depth;
              existing.iteration = n.iteration;
              existing.insights = n.insights;
              return existing;
            }
            const base: GraphNodeData = {
              id: n.id, depth: n.depth, iteration: n.iteration, insights: n.insights,
            };
            const anchor = newNodeAnchor.get(n.id);
            if (anchor) {
              base.x = (anchor.x ?? 0) + jitter();
              base.y = (anchor.y ?? 0) + jitter();
            }
            return base;
          });
          const validIds = new Set(nodes.map((n) => n.id));
          const links = data.edges
            .filter((e) => validIds.has(e.source) && validIds.has(e.target))
            .map((e) => ({ source: e.source, target: e.target, reason: e.reason }));
          const effectiveRoot = chooseRoot(nodes, data.starting_url ?? null);
          const depthById = bfsDepths(
            nodes.map((n) => n.id),
            links,
            effectiveRoot,
          );
          // Start at 0 so the "no root found / empty depthById" case
          // reports maxDepth=0 (consistent with EMPTY); real graphs always
          // have depthById populated → maxDepth ≥ 1.
          let maxDepth = 0;
          for (const d of depthById.values()) if (d > maxDepth) maxDepth = d;
          return { nodes, links, startingUrl: effectiveRoot, depthById, maxDepth };
        });
        setError(null);
      } catch (e) {
        if (!cancelled) {
          const msg = String((e as Error).message);
          // "No graph snapshots yet." (404) and similar are expected during
          // quick_explore phase 1 — don't surface them as errors so the
          // progress-bar empty-state can render.
          const isExpectedNoGraph =
            msg.startsWith('404') ||
            /no\s+graph|snapshots?\s+yet|no\s+snapshots/i.test(msg);
          setError(isExpectedNoGraph ? null : msg);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [runId, refreshKey]);

  // Bubble graph stats up so RunSummaryCard can render them without
  // duplicate-fetching /graph. Effect (not render-time) to avoid the
  // setState-during-render warning.
  useEffect(() => {
    if (!onStats) return;
    onStats({ nodes: graph.nodes.length, edges: graph.links.length, maxDepth: graph.maxDepth });
  }, [graph.nodes.length, graph.links.length, graph.maxDepth, onStats]);

  // Tune the force-directed layout the moment ForceGraph2D mounts.
  // Defaults (charge.strength=-30, no collision, link.distance=30) cluster
  // too tightly for graphs with 100-200+ nodes — Caesar deep runs routinely
  // produce that. Bump repulsion, add a collision force with breathing
  // room, and push link distance up so chains of nodes spread out. Using
  // a ref callback instead of useEffect avoids a timing race where the
  // dynamic ForceGraph2D import resolves on a render that doesn't change
  // useEffect deps, leaving the configuration unrun.
  const forcesConfigured = useRef(false);
  const fgRefCallback = useCallback((node: any) => {
    fgRef.current = node;
    // StrictMode (dev) and HMR fire `(node) → (null) → (node)`. Without
    // resetting on the unmount-ish branch, the second real mount finds
    // forcesConfigured=true and skips configuration — graph then runs
    // with default charge / no collision.
    if (!node) {
      forcesConfigured.current = false;
      return;
    }
    if (forcesConfigured.current) return;
    // distanceMax caps the O(N²) repulsion at 400px — well past the typical
    // canvas diagonal — so far-apart clusters don't keep pushing each other.
    node.d3Force?.('charge', forceManyBody().strength(-120).distanceMax(400));
    // Collide radius = NODE_R + 3px breathing room. Node visual radius is
    // NODE_R; the nodeCanvasObject stroke adds another ~1px at 1x zoom, so
    // (5 + 3) avoids visible overlap even when zoomed out.
    node.d3Force?.('collide', forceCollide(NODE_R + 3));
    // Stretch link distance so the BFS depth gradient is legible at large
    // N. Default ~30 packs chains into a blob; 45 spaces them out without
    // pushing the graph past the viewport for small graphs.
    node.d3Force?.('link')?.distance(45);
    forcesConfigured.current = true;
  }, []);

  // Auto-fit the graph to the canvas. The naive "fit once, then preserve
  // the user's pan/zoom" version had a pathological case on the live SSE
  // path: the very first /graph response often contains just the root
  // node (Caesar emits graph_iter1 with only the seed page), and fitting
  // a single node at 60px padding zooms the canvas extreme-close on that
  // one node. Subsequent incremental nodes then land far outside the
  // viewport — invisible and unclickable until the user manually zooms
  // out or refreshes. So we keep refitting until the graph crosses a
  // small threshold, after which we latch to respect any user pan/zoom.
  const FIT_LATCH_NODES = 3;
  useEffect(() => {
    if (graph.nodes.length === 0) {
      zoomedOnce.current = false;
      return;
    }
    if (zoomedOnce.current) return;
    const t = setTimeout(() => {
      try {
        fgRef.current?.zoomToFit?.(600, ZOOM_FIT_PADDING);
        // Only latch once there are enough nodes that the fit is
        // actually meaningful — otherwise the next SSE update grows
        // the graph and we should refit again.
        if (graph.nodes.length >= FIT_LATCH_NODES) {
          zoomedOnce.current = true;
        }
      } catch {
        // ignore — ref API can be missing during HMR
      }
    }, 500); // give the force layout a beat to settle before we fit
    return () => clearTimeout(t);
  }, [graph.nodes.length]);

  // Re-heat the d3 simulation whenever new nodes appear. Default behavior
  // bumps alpha when graphData changes, but once the simulation has cooled
  // (cooldownTicks=150 reached, alpha below alphaMin), a single small reheat
  // isn't always enough to push fresh nodes out of a dense existing cluster
  // — they spawn near (0,0) and get pinned by their own forceCollide
  // overlap. Explicit reheat-to-1 fixes that.
  const prevNodeCount = useRef(0);
  useEffect(() => {
    if (graph.nodes.length > prevNodeCount.current && fgRef.current?.d3ReheatSimulation) {
      try {
        fgRef.current.d3ReheatSimulation();
      } catch {
        // ignore — ref API can be missing during HMR
      }
    }
    prevNodeCount.current = graph.nodes.length;
  }, [graph.nodes.length]);

  // Track container size for canvas.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const update = () => setSize({ w: el.clientWidth, h: el.clientHeight });
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const empty = graph.nodes.length === 0;

  return (
    <div className="relative w-full h-full min-h-[360px] rounded-2xl border border-gray-200 bg-white overflow-hidden flex flex-col">
      <div className="px-4 py-4 border-b border-gray-100 text-xs font-medium text-gray-500 uppercase tracking-wider flex-shrink-0 flex items-center justify-between gap-2">
        <span>Knowledge graph</span>
        <div className="flex items-center gap-3">
          {detailsHref && (
            <Link
              href={detailsHref}
              className="normal-case tracking-normal font-medium text-brand-700 hover:text-brand-800 hover:underline"
            >
              Table View
            </Link>
          )}
          <HelpTip label="Each circle is a page explored; lines are references between them; color shows distance from the red starting page. Click a node to open it." />
        </div>
      </div>
      <div ref={containerRef} className="relative flex-1 min-h-0">
        {empty ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center px-6 text-gray-500">
            {error ? (
              <p className="text-sm">{error}</p>
            ) : mode === 'refine' ? (
              <>
                <p className="text-sm">
                  This answer was generated without new exploration.
                </p>
                <p className="text-xs text-gray-400 mt-1 max-w-md">
                  The synthesis drew on the parent run’s existing knowledge
                  base only; no new pages were fetched.
                </p>
                {parentRunId ? (
                  <a
                    href={`/run/${parentRunId}`}
                    className="text-xs text-brand-800 hover:underline mt-3"
                  >
                    View the parent run’s graph →
                  </a>
                ) : null}
              </>
            ) : phase === 'quick_explore' && progress && progress.total > 0 ? (
              <>
                <div className="w-3 h-3 rounded-full bg-brand-500/50 mb-3 animate-pulse-slow" />
                <p className="text-sm">
                  Caesar is fetching its first pages.
                </p>
                <p className="text-xs text-gray-400 mt-1 mb-4 max-w-md">
                  The graph will start growing here as nodes land. First
                  ones typically appear within ~30 seconds.
                </p>
                <div className="w-64 max-w-full">
                  <ProgressBar value={(progress.n / progress.total) * 100} />
                  <div className="text-xs text-gray-500 tabular-nums mt-2">
                    {progress.n} / {progress.total} pages fetched
                  </div>
                </div>
              </>
            ) : (
              <>
                <div className="w-3 h-3 rounded-full bg-brand-500/50 mb-2 animate-pulse-slow" />
                <p className="text-sm">
                  Waiting for Caesar to fetch its first pages…
                </p>
                <p className="text-xs text-gray-400 mt-1">
                  Nodes will appear here as the agent explores.
                </p>
              </>
            )}
          </div>
        ) : size.w > 0 && size.h > 0 ? (
          <ForceGraph2D
            // The library's ref typing accepts only MutableRefObject, not
            // a callback ref; React itself supports both, so `as any` is
            // the cleanest escape hatch. fgRefCallback fires correctly at
            // runtime — `as React.Ref<unknown>` (the obvious narrower
            // cast) doesn't satisfy the library's MutableRefObject<T>
            // requirement.
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            ref={fgRefCallback as any}
            width={size.w}
            height={size.h}
            graphData={graph}
            nodeLabel={(n: any) => {
              const d = graph.depthById.get(n.id) ?? Number(n.depth ?? 0);
              const iter = n.iteration ?? '—';
              // The root is a file:// local search-results seed page; show a
              // friendly label instead of the unbrowsable server path.
              const label = String(n.id ?? '').startsWith('file://') ? 'Starting Search Page' : n.id;
              return `${label} (Iteration: ${iter}, Depth: ${d})`;
            }}
            nodeRelSize={NODE_R}
            nodeColor={(n: any) => {
              if (graph.startingUrl && n.id === graph.startingUrl) return ROOT_NODE_COLOR;
              const d = graph.depthById.get(n.id) ?? Number(n.depth ?? 0);
              return paletteFor(d, graph.maxDepth);
            }}
            // Draw a thin dark border on top of every node so they're
            // readable on white at any zoom. lineWidth = 1/globalScale
            // keeps the stroke ~1 CSS pixel regardless of zoom level.
            nodeCanvasObjectMode={() => 'after'}
            nodeCanvasObject={(n: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
              ctx.beginPath();
              ctx.arc(n.x, n.y, NODE_R, 0, 2 * Math.PI);
              ctx.lineWidth = 1 / globalScale;
              ctx.strokeStyle = 'rgba(0,0,0,0.55)';
              ctx.stroke();
            }}
            linkColor={() => 'rgba(31,41,55,0.18)'}
            linkDirectionalArrowLength={3.5}
            linkDirectionalArrowRelPos={0.9}
            cooldownTicks={300}
            backgroundColor="#ffffff"
            // Click a node to open its source URL in a new tab. Only http(s)
            // — the root node is a `file:///` URL pointing at Caesar's local
            // search-results page, which the user's browser can't resolve
            // when the web server runs remotely (most browsers silently
            // refuse window.open('file://') from an http origin). Skipping
            // both the click and the pointer cursor avoids advertising a
            // dead affordance.
            onNodeClick={(n: any) => {
              const url = String(n.id ?? '');
              if (/^https?:\/\//i.test(url)) {
                window.open(url, '_blank', 'noopener,noreferrer');
              } else if (url.startsWith('file://')) {
                // The root is the local search-results seed; open a parsed,
                // safe view of it rather than the unbrowsable file:// path.
                setSearchOpen(true);
              }
            }}
            onNodeHover={(n: any) => {
              const el = containerRef.current;
              if (!el) return;
              const u = String(n?.id ?? '');
              const clickable = !!n && (/^https?:\/\//i.test(u) || u.startsWith('file://'));
              el.style.cursor = clickable ? 'pointer' : 'default';
            }}
          />
        ) : null}
      </div>
      <div className="absolute bottom-2 right-3 text-[11px] text-gray-400 select-none">
        {graph.nodes.length} nodes · {graph.links.length} edges
      </div>
      {searchOpen && (
        <Modal title="Starting Search Page" onClose={() => setSearchOpen(false)}>
          <SearchResultsView runId={runId} />
        </Modal>
      )}
    </div>
  );
}
