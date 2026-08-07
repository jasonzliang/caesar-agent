'use client';

import { useState, type MouseEvent } from 'react';
import { RotateCcw, Loader2 } from 'lucide-react';
import { api, type RunSummary } from '@/lib/api';
import { usePublicMode, getStoredKey } from '@/lib/public-mode';
import { cn } from '@/lib/utils';

type Props = {
  runId: string;
  /** Receives the fresh (queued) summary so the parent can flip out of its
   *  terminal state and let its existing poll/SSE effects take over. */
  onRestarted?: (run: RunSummary) => void;
  /** When the row is itself a link, set this so the click doesn't navigate. */
  stopPropagation?: boolean;
  className?: string;
};

/** Restart a run in place, whatever its status. A live attempt is stopped
 *  server-side first; the only refusal is a worker that won't stop inside the
 *  grace period, which comes back as a 409 telling you to try again. */
export function RestartRunButton({ runId, onRestarted, stopPropagation, className }: Props) {
  const publicMode = usePublicMode();
  const [busy, setBusy] = useState(false);

  const onClick = async (e: MouseEvent<HTMLButtonElement>) => {
    if (stopPropagation) {
      e.preventDefault();
      e.stopPropagation();
    }
    if (busy) return;
    // Public mode reuses the key saved on the /config page (a cookie), same as
    // the follow-up dialog — there is no key input here.
    let apiKey = '';
    if (publicMode) {
      apiKey = getStoredKey().trim();
      if (!apiKey) {
        window.alert('Your OpenAI API key is missing. Add it on the Config page, then restart.');
        return;
      }
    }
    if (
      !window.confirm(
        'Restart this run?\n\n'
          + 'An attempt still in progress is stopped first. The checkpoint covers '
          + 'exploration only, so a run that has reached the synthesis stage '
          + 'restarts its drafts from the beginning and loses that work. With no '
          + 'checkpoint at all it re-runs the query. Either way this spends API '
          + 'credits.',
      )
    ) {
      return;
    }
    setBusy(true);
    try {
      const run = await api.retryRun(runId, publicMode ? { apiKey } : undefined);
      onRestarted?.(run);
    } catch (err) {
      window.alert(`Could not restart: ${(err as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={busy}
      aria-label="Restart run"
      title="Restart this run from its last checkpoint"
      className={cn(
        // Same square geometry as DeleteRunButton so the two sit together as a
        // pair of icon actions. The glyph carries it; aria-label and title keep
        // the wording for screen readers and hover.
        'inline-flex items-center justify-center w-7 h-7 rounded-md',
        'text-gray-400 hover:text-brand-800 hover:bg-brand-50',
        'disabled:opacity-50 disabled:cursor-not-allowed',
        'transition-colors',
        className,
      )}
    >
      {busy ? <Loader2 size={14} className="animate-spin" /> : <RotateCcw size={14} />}
    </button>
  );
}
