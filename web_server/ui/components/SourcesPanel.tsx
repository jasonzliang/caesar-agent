import type { Citation } from '@/lib/api';
import { HelpTip } from './HelpTip';

export function SourcesPanel({ sources }: { sources: Citation[] }) {
  if (sources.length === 0) return null;
  return (
    <section className="mt-8">
      <div className="flex items-center justify-between gap-2 mb-3">
        <h2 className="text-sm font-medium text-gray-500 uppercase tracking-wider">
          Sources
        </h2>
        <HelpTip label="The URLs that powered the answer. The numbered [1] markers in the text link down to these." />
      </div>
      <ol className="space-y-2">
        {sources.map((c) => (
          <li
            key={c.index}
            id={`source-${c.index}`}
            className="source-row flex gap-3 text-sm leading-relaxed scroll-mt-20"
          >
            <span className="shrink-0 inline-flex items-center justify-center w-7 h-7 rounded-md bg-brand-50 text-brand-800 font-medium tabular-nums text-xs">
              {c.index}
            </span>
            <a
              href={c.url}
              target="_blank"
              rel="noopener"
              className="text-gray-700 hover:text-brand-800 break-all"
            >
              {c.url}
            </a>
          </li>
        ))}
      </ol>
    </section>
  );
}
