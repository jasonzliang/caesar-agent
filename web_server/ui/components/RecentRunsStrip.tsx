import { headers } from 'next/headers';
import type { RunSummary } from '@/lib/api';
import { RecentRunsClient } from './RecentRunsClient';

async function fetchRunsServer(): Promise<RunSummary[]> {
  const apiInternal = process.env.API_INTERNAL_URL ?? 'http://127.0.0.1:8090';
  // Owner-scoped fetch: this server-component request hits FastAPI directly and
  // would not carry the browser's caesar_id cookie unless we forward it. Without
  // this, public mode returns an empty list and every user sees a blank strip.
  const cookie = (await headers()).get('cookie') ?? '';
  try {
    const res = await fetch(`${apiInternal}/runs?limit=8`, {
      cache: 'no-store',
      headers: { cookie },
    });
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

export async function RecentRunsStrip() {
  const runs = await fetchRunsServer();
  return <RecentRunsClient initial={runs} />;
}
