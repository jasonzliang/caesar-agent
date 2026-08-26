'use client';

import { Fragment } from 'react';
import { cn } from '@/lib/utils';
import type { Preset } from '@/lib/api';

type Props = {
  presets: Preset[];
  value: string;
  onChange: (id: string) => void;
};

// Compact whole-number cost: $0.50 / $1 / $5 / $30.
function fmtCost(usd: number): string {
  if (usd >= 1 && Number.isInteger(usd)) return `$${usd}`;
  return `$${usd.toFixed(2)}`;
}

// Show durations >=60 min as hours so "90m" doesn't dominate.
function fmtMin(min: number): string {
  if (min >= 60) {
    const h = min / 60;
    return Number.isInteger(h) ? `${h}h` : `${h.toFixed(1)}h`;
  }
  return `${min}m`;
}

// Presets that are a different SEARCH SOURCE rather than a web-depth tier get a
// divider before them, so the row reads as "web depths | other sources" instead
// of implying arXiv is just an even-deeper speed setting. Extend as new sources
// (e.g. a future pubmed mode) are added.
const SOURCE_BREAK = new Set(['arxiv']);

export function PresetToggle({ presets, value, onChange }: Props) {
  // The per-pill "~$cost · time" was the row's space hog and pushed the 5th
  // pill off narrow viewports; pills now carry only the label and the selected
  // preset's cost/time drops to a single caption line below (full description
  // stays on hover). This de-crowds and scales to any number of presets.
  const selected = presets.find((p) => p.id === value) ?? presets[0];
  return (
    <div className="inline-flex flex-wrap items-center gap-2">
      <div className="inline-flex flex-wrap items-center rounded-xl border border-gray-200 bg-gray-50 p-1 gap-0.5">
        {presets.map((p, i) => {
          const active = p.id === value;
          const divide = SOURCE_BREAK.has(p.id) && i > 0;
          return (
            <Fragment key={p.id}>
              {divide && <span aria-hidden className="mx-1 self-stretch w-px bg-gray-300" />}
              <button
                type="button"
                onClick={() => onChange(p.id)}
                className={cn(
                  'px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
                  active
                    ? 'bg-white text-gray-900 shadow-sm border border-gray-200'
                    : 'text-gray-600 hover:text-gray-900',
                )}
                title={p.description}
                aria-pressed={active}
              >
                {p.label}
              </button>
            </Fragment>
          );
        })}
      </div>
      {/* Selected preset's cost/time sits to the RIGHT of the pill group on the
          same line. The preset name is not restated here -- the active pill
          already shows it. */}
      {selected && (
        <span className="text-sm text-gray-500 whitespace-nowrap" suppressHydrationWarning>
          ~{fmtCost(selected.estimated_cost_usd)} · {fmtMin(selected.estimated_time_min)}
        </span>
      )}
    </div>
  );
}
