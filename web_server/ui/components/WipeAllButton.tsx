'use client';

import { useState } from 'react';
import { Trash2 } from 'lucide-react';

export function WipeAllButton({ runCount }: { runCount: number }) {
  const [busy, setBusy] = useState(false);

  const onClick = async () => {
    if (busy) return;
    const ok = window.confirm(
      `Delete all ${runCount} run${runCount === 1 ? '' : 's'}?\n\n`
      + 'Every answer, knowledge graph and source list is removed, including any '
      + 'run still in progress. This cannot be undone.',
    );
    if (!ok) return;
    setBusy(true);
    try {
      const res = await fetch('/api/runs?confirm=yes', { method: 'DELETE' });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || `HTTP ${res.status}`);
      }
      // Hard reload so server-side rendered list refreshes from empty.
      window.location.reload();
    } catch (e) {
      alert(`Wipe failed: ${(e as Error).message}`);
      setBusy(false);
    }
  };

  if (runCount === 0) return null;
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={busy}
      className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-red-200 bg-white px-3 text-xs font-medium text-red-700 hover:bg-red-50 hover:border-red-300 disabled:opacity-50 transition-colors"
    >
      <Trash2 size={14} />
      {busy ? 'Wiping…' : `Wipe all (${runCount})`}
    </button>
  );
}
