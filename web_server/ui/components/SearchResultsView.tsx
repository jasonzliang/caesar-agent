'use client';

import { useEffect, useState } from 'react';
import { api, type SearchResultItem } from '@/lib/api';

// Fetches a run's search-results seed (parsed to JSON server-side) and renders
// it as structured, auto-escaped React (never raw HTML), so it is XSS-safe.
export function SearchResultsView({ runId }: { runId: string }) {
  const [state, setState] = useState<{
    loading: boolean;
    error: string | null;
    results: SearchResultItem[];
  }>({ loading: true, error: null, results: [] });

  useEffect(() => {
    let cancelled = false;
    api
      .getSearchResults(runId)
      .then((r) => {
        if (!cancelled) setState({ loading: false, error: null, results: r.results });
      })
      .catch((e) => {
        if (!cancelled) setState({ loading: false, error: (e as Error).message, results: [] });
      });
    return () => {
      cancelled = true;
    };
  }, [runId]);

  if (state.loading) return <p className="text-sm text-gray-400">Loading search results…</p>;
  if (state.error) return <p className="text-sm text-red-600">{state.error}</p>;
  if (state.results.length === 0)
    return <p className="text-sm text-gray-400">No search results found.</p>;

  return (
    <ol className="divide-y divide-gray-100">
      {state.results.map((r, i) => {
        const safeHref = /^https?:\/\//i.test(r.url) ? r.url : undefined;
        return (
          <li key={i} className="py-2.5">
            {safeHref ? (
              <a
                href={safeHref}
                target="_blank"
                rel="noopener"
                className="text-sm font-medium text-brand-800 hover:underline break-words"
              >
                {r.title}
              </a>
            ) : (
              <span className="text-sm font-medium text-gray-800">{r.title}</span>
            )}
            <div className="font-mono text-xs text-gray-500 break-all">{r.url}</div>
            {r.description && (
              <p className="mt-0.5 text-xs text-gray-500 leading-snug">{r.description}</p>
            )}
          </li>
        );
      })}
    </ol>
  );
}
