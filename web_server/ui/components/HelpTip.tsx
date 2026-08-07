'use client';

import { useId, useState } from 'react';
import { CircleHelp } from 'lucide-react';
import { cn } from '@/lib/utils';

// A small, muted "?" help marker for a box's upper-right corner. Shows a short
// tooltip on hover AND keyboard focus, and toggles on click for touch devices.
// Styling matches the app's light cards (white / gray-200 border / shadow) so
// it reads as part of the UI, not a foreign dark tooltip. Accessible: a real
// focusable button with aria-describedby to the tooltip copy; Esc / blur close
// the click-opened state; motion-safe fade (instant for reduced-motion).
export function HelpTip({ label, className }: { label: string; className?: string }) {
  const id = useId();
  const [open, setOpen] = useState(false);
  return (
    <span className={cn('group relative inline-flex leading-none', className)}>
      <button
        type="button"
        aria-label="What is this?"
        aria-describedby={id}
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        onKeyDown={(e) => {
          if (e.key === 'Escape') {
            setOpen(false);
            e.currentTarget.blur();
          }
        }}
        onBlur={() => setOpen(false)}
        className="rounded-full text-gray-400 transition-colors hover:text-gray-600 focus-visible:text-gray-600 focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500/50"
      >
        <CircleHelp size={14} aria-hidden />
      </button>
      <span
        id={id}
        role="tooltip"
        className={cn(
          'pointer-events-none absolute right-0 top-6 z-30 w-64 rounded-lg border border-gray-200 bg-white px-3 py-2',
          'text-left text-xs font-normal normal-case leading-relaxed tracking-normal text-gray-600 shadow-lg',
          'opacity-0 motion-safe:transition-opacity group-hover:opacity-100 group-focus-within:opacity-100',
          open && 'opacity-100',
        )}
      >
        {label}
      </span>
    </span>
  );
}
