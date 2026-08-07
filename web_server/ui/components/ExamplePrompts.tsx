'use client';

import { Sparkles } from 'lucide-react';

const EXAMPLES = [
  {
    short: '100x patterns',
    full: 'What strategic patterns have repeatedly produced 100x outcomes across enterprise SaaS, marketplaces, and developer infra? Focus on the mechanism, not the names.',
  },
  {
    short: 'Deep-research fails',
    full: 'Where do current deep-research agents (Perplexity, ChatGPT Deep Research, Gemini) underperform a human analyst? Identify the failure modes most expensive at scale.',
  },
  {
    short: 'First-mover trap?',
    full: 'When does first-mover advantage compound into a durable lead, and when is it a trap that second-movers exploit? Identify the dividing factor.',
  },
  {
    short: 'Network effect myths',
    full: 'Which "network effects" actually defend against new entrants, and which are myths used to justify valuations? Compare proven moats with claimed ones.',
  },
  {
    short: 'Vertical AI moats',
    full: 'What defensible moats do vertical AI startups have in 2026, beyond proprietary data? Identify patterns that produce category winners vs. ones that fail.',
  },
];

export function ExamplePrompts({
  onPick,
  disabled,
}: {
  onPick: (q: string) => void;
  disabled?: boolean;
}) {
  return (
    <div className="mt-4 flex flex-wrap items-center gap-2">
      <span className="inline-flex items-center gap-1 text-xs text-gray-500">
        <Sparkles size={12} /> Try
      </span>
      {EXAMPLES.map((ex) => (
        <button
          key={ex.short}
          type="button"
          disabled={disabled}
          onClick={() => onPick(ex.full)}
          className="text-xs px-3 py-1.5 rounded-full border border-gray-200 text-gray-700 bg-white hover:bg-gray-50 hover:border-gray-300 transition-colors disabled:opacity-50"
        >
          {ex.short}
        </button>
      ))}
    </div>
  );
}
