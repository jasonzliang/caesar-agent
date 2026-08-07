'use client';

import { useState, type MouseEvent } from 'react';
import { useRouter } from 'next/navigation';
import { Trash2, Loader2 } from 'lucide-react';
import { api } from '@/lib/api';
import { cn } from '@/lib/utils';

type Props = {
  runId: string;
  query: string;
  /** Called *before* navigation refresh so a parent can hide the row optimistically. */
  onDeleted?: () => void;
  /** When the row is itself a link, set this so the click doesn't navigate. */
  stopPropagation?: boolean;
  className?: string;
};

export function DeleteRunButton({
  runId,
  query,
  onDeleted,
  stopPropagation,
  className,
}: Props) {
  const router = useRouter();
  const [busy, setBusy] = useState(false);

  const onClick = async (e: MouseEvent<HTMLButtonElement>) => {
    if (stopPropagation) {
      e.preventDefault();
      e.stopPropagation();
    }
    if (busy) return;
    const preview = query.length > 60 ? query.slice(0, 60) + '…' : query;
    // Says what is lost, not how it is stored: this dialog is the last thing a
    // visitor sees before an irreversible action, and "database row" tells them
    // nothing about the consequence.
    if (!window.confirm(
      `Delete this run?\n\n"${preview}"\n\n`
      + 'Its answer, knowledge graph and sources go with it. This cannot be undone.',
    )) {
      return;
    }
    setBusy(true);
    try {
      await api.deleteRun(runId);
      onDeleted?.();
      router.refresh();
    } catch (err) {
      window.alert(`Failed to delete: ${(err as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={busy}
      aria-label="Delete run"
      title="Delete run"
      className={cn(
        'inline-flex items-center justify-center w-7 h-7 rounded-md',
        'text-gray-400 hover:text-red-600 hover:bg-red-50',
        'disabled:opacity-50 disabled:cursor-not-allowed',
        'transition-colors',
        className,
      )}
    >
      {busy ? <Loader2 size={14} className="animate-spin" /> : <Trash2 size={14} />}
    </button>
  );
}
