'use client';

import { cn } from '@/lib/utils';
import type { Preset } from '@/lib/api';

type Props = {
  presets: Preset[];
  value: string;
  onChange: (id: string) => void;
};

// Compact whole-number cost: $0.30 / $1 / $5 / $30. The toggle's horizontal
// budget is tight at 4 presets — printing fixed two-decimal cents for whole
// dollars wastes pixels and pushes the rightmost pill off the row on narrow
// viewports.
function fmtCost(usd: number): string {
  if (usd >= 1 && Number.isInteger(usd)) return `$${usd}`;
  return `$${usd.toFixed(2)}`;
}

// Show durations ≥60 min as hours so "300m" doesn't dominate the pill.
function fmtMin(min: number): string {
  if (min >= 60) {
    const h = min / 60;
    return Number.isInteger(h) ? `${h}h` : `${h.toFixed(1)}h`;
  }
  return `${min}m`;
}

export function PresetToggle({ presets, value, onChange }: Props) {
  return (
    <div className="inline-flex flex-wrap rounded-xl border border-gray-200 bg-gray-50 p-1 gap-0.5">
      {presets.map((p) => {
        const active = p.id === value;
        return (
          <button
            key={p.id}
            type="button"
            onClick={() => onChange(p.id)}
            className={cn(
              'px-2.5 py-1.5 rounded-lg text-sm font-medium transition-colors',
              active
                ? 'bg-white text-gray-900 shadow-sm border border-gray-200'
                : 'text-gray-600 hover:text-gray-900',
            )}
            title={p.description}
            aria-pressed={active}
          >
            {p.label}
            <span className={cn('ml-1 text-xs', active ? 'text-brand-700' : 'text-gray-400')}>
              ~{fmtCost(p.estimated_cost_usd)} · {fmtMin(p.estimated_time_min)}
            </span>
          </button>
        );
      })}
    </div>
  );
}
