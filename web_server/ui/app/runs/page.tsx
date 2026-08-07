import { headers } from 'next/headers';
import type { RunSummary } from '@/lib/api';
import { RunsListClient } from '@/components/RunsListClient';

// Render at request time — `next start` would otherwise statically
// pre-render the empty list at build time and serve it indefinitely.
export const dynamic = 'force-dynamic';

async function getRuns(): Promise<RunSummary[]> {
  const apiInternal = process.env.API_INTERNAL_URL ?? 'http://127.0.0.1:8090';
  // Owner-scoped: forward the browser cookie so public mode returns only this
  // browser's runs (see RecentRunsStrip for the same pattern).
  const cookie = (await headers()).get('cookie') ?? '';
  try {
    const res = await fetch(`${apiInternal}/runs?limit=200`, {
      cache: 'no-store',
      headers: { cookie },
    });
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

export default async function RunsPage() {
  const runs = await getRuns();
  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-10 space-y-6">
      <header className="flex items-start justify-between gap-3">
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold tracking-tight text-gray-900">Past runs</h1>
          <p className="text-sm text-gray-500">
            Every Caesar exploration submitted to this server. Click into any row to see the live
            knowledge graph and final answer.
          </p>
        </div>
      </header>

      <RunsListClient initial={runs} />
    </div>
  );
}
