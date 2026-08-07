import { notFound } from 'next/navigation';
import { headers } from 'next/headers';
import { GraphTableClient } from '@/components/GraphTableClient';
import type { RunDetail } from '@/lib/api';

// Render at request time so `next start` doesn't pre-render (and 404-cache) a
// run that didn't exist at build time. Mirrors app/run/[id]/page.tsx.
export const dynamic = 'force-dynamic';

async function getRun(id: string): Promise<RunDetail | null> {
  const apiInternal = process.env.API_INTERNAL_URL ?? 'http://127.0.0.1:8090';
  // Owner-scoped: forward the browser cookie so public mode 404s a run owned
  // by a different browser instead of leaking it via SSR.
  const cookie = (await headers()).get('cookie') ?? '';
  const res = await fetch(`${apiInternal}/runs/${id}`, {
    cache: 'no-store',
    headers: { cookie },
  });
  if (res.status === 404 || res.status === 401) return null;
  if (!res.ok) throw new Error(`Backend error: ${res.status}`);
  return res.json();
}

export default async function GraphTablePage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const run = await getRun(id);
  if (!run) notFound();
  // Follow-up runs inherit their parent's graph; use the same run id the
  // KnowledgeGraph component uses for its data fetch.
  const graphRunId = run.graph_run_id ?? id;
  const query = run.merged_query || run.query;
  return <GraphTableClient key={id} runId={id} graphRunId={graphRunId} query={query} />;
}
