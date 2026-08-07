import { Suspense } from 'react';
import { headers } from 'next/headers';
import { QueryInput } from '@/components/QueryInput';
import { StatStrip } from '@/components/StatStrip';
import { RecentRunsStrip } from '@/components/RecentRunsStrip';
import type { Preset } from '@/lib/api';

// Fetch presets server-side. We hit the FastAPI backend directly (server-to-
// server, bypassing the rewrite proxy) so the request never leaves the host.
async function getPresetsServer(): Promise<Preset[]> {
  const apiInternal = process.env.API_INTERNAL_URL ?? 'http://127.0.0.1:8090';
  try {
    const res = await fetch(`${apiInternal}/presets`, { cache: 'no-store' });
    if (!res.ok) throw new Error(String(res.status));
    return res.json();
  } catch {
    // Sensible fallback so the page always renders even if the API is down.
    // Keep this in sync with PRESETS in web_server/api/app/config.py.
    return [
      {
        id: 'fast',
        label: 'Fast',
        description: '~$0.30, ~10 min',
        estimated_cost_usd: 0.3,
        estimated_time_min: 10,
      },
      {
        id: 'normal',
        label: 'Normal',
        description: '~$1, ~15 min',
        estimated_cost_usd: 1,
        estimated_time_min: 15,
      },
      {
        id: 'deeper',
        label: 'Deeper',
        description: '~$5, ~45-90 min',
        estimated_cost_usd: 5,
        estimated_time_min: 60,
      },
      {
        id: 'deepest',
        label: 'Deepest',
        description: '~$30, ~5 hours',
        estimated_cost_usd: 30,
        estimated_time_min: 300,
      },
    ];
  }
}

export default async function HomePage() {
  // Touch headers so Next renders this page dynamically per-request.
  await headers();
  const presets = await getPresetsServer();

  return (
    <div className="max-w-[50rem] mx-auto px-4 sm:px-6 py-10 sm:py-16 space-y-10">
      <section className="space-y-3">
        <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight text-gray-900">
          Watch an autonomous research agent{' '}
          <a
            href="https://www.youtube.com/watch?v=2i4P3g_rvmE"
            target="_blank"
            rel="noopener"
            className="text-brand-800 hover:underline"
          >
            think.
          </a>
        </h1>
        <p className="text-lg text-gray-600 leading-relaxed">
          Caesar explores the web as a knowledge graph and refines its answer through
          adversarial self-critique, writing deeper and more creative syntheses than
          today&apos;s deep-research tools.
          To learn more about how Caesar works, visit the{' '}
          <a
            href="https://jasonzliang.github.io/caesar-agent/"
            target="_blank"
            rel="noopener"
            className="text-brand-800 hover:underline"
          >
            project page
          </a>
          .
        </p>
        <StatStrip />
      </section>

      <QueryInput presets={presets} />

      <section className="space-y-3">
        <div className="flex items-baseline justify-between">
          <h2 className="text-sm font-medium text-gray-500 uppercase tracking-wider">
            Recent runs
          </h2>
          <a href="/runs" className="text-sm text-brand-800 hover:underline">
            See all →
          </a>
        </div>
        <Suspense
          fallback={
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {[0, 1, 2, 3].map((i) => (
                <div key={i} className="skeleton h-24 rounded-xl" />
              ))}
            </div>
          }
        >
          <RecentRunsStrip />
        </Suspense>
      </section>
    </div>
  );
}
