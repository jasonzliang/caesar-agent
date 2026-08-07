import { notFound } from 'next/navigation';
import { headers } from 'next/headers';
import { RunPageClient } from '@/components/RunPageClient';
import type { RunDetail } from '@/lib/api';

// Render at request time — `next start` would otherwise pre-render the
// page at build time (when the run didn't exist) and serve a 404 cache.
export const dynamic = 'force-dynamic';

async function getRun(id: string): Promise<RunDetail | null> {
  const apiInternal = process.env.API_INTERNAL_URL ?? 'http://127.0.0.1:8090';
  // Owner-scoped: forward the browser cookie so public mode 404s a run that
  // belongs to a different browser (instead of leaking it via SSR).
  const cookie = (await headers()).get('cookie') ?? '';
  const res = await fetch(`${apiInternal}/runs/${id}`, {
    cache: 'no-store',
    headers: { cookie },
  });
  // 404 = no such run (or owned by another browser); 401 = no identity cookie
  // yet (cold load before the middleware-minted cookie is sent). Both render
  // the not-found page rather than a 500.
  if (res.status === 404 || res.status === 401) return null;
  if (!res.ok) throw new Error(`Backend error: ${res.status}`);
  return res.json();
}

export default async function RunPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const run = await getRun(id);
  if (!run) notFound();
  // key={id} forces React to remount RunPageClient when navigating between
  // /run/A and /run/B. Without it the same component instance is reused and
  // useState retains run A's `run` / `synthesis` / `graphStats` until the
  // first poll overwrites them — flashing run A's numbers on run B's page.
  return <RunPageClient key={id} initial={run} />;
}
