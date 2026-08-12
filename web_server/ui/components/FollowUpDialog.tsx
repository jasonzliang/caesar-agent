'use client';

import { useEffect, useRef, useState, type FormEvent } from 'react';
import { createPortal } from 'react-dom';
import { useRouter } from 'next/navigation';
import { ArrowRight, Loader2, X } from 'lucide-react';
import { api, type RunMode } from '@/lib/api';
import {
  usePublicMode,
  getStoredKey,
  getStoredModel,
  getStoredOutputLength,
} from '@/lib/public-mode';
import { cn } from '@/lib/utils';

type Props = {
  open: boolean;
  onClose: () => void;
  parentRunId: string;
  parentPreset: string;
};

type ModeChoice = Exclude<RunMode, 'new'>;

const MODE_ORDER: ModeChoice[] = ['refine', 'explore'];

const MODE_LABEL: Record<ModeChoice, string> = {
  explore: 'Explore further (slower)',
  refine: 'No exploration (faster)',
};

const MODE_SUBLABEL: Record<ModeChoice, string> = {
  explore: 'Improve existing knowledge with new exploration',
  refine: 'Synthesize answer from existing knowledge only',
};

const FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';

export function FollowUpDialog({ open, onClose, parentRunId, parentPreset }: Props) {
  const router = useRouter();
  const publicMode = usePublicMode();
  const [question, setQuestion] = useState('');
  const [mode, setMode] = useState<ModeChoice>('explore');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const panelRef = useRef<HTMLDivElement | null>(null);
  // Trigger that opened the dialog — restore focus here on close so
  // keyboard users don't get dumped onto <body> (WCAG 2.4.3).
  const triggerRef = useRef<HTMLElement | null>(null);

  // Reset transient state every time the dialog re-opens so a stale error,
  // half-typed question, or stale mode selection from a prior open doesn't
  // carry over. No exploration is the faster default for ordinary follow-ups;
  // users can opt into fresh exploration per question.
  useEffect(() => {
    if (open) {
      triggerRef.current = document.activeElement as HTMLElement | null;
      setQuestion('');
      setMode('refine');
      setError(null);
      setSubmitting(false);
      // Focus the textarea after the dialog mounts.
      requestAnimationFrame(() => textareaRef.current?.focus());
    } else if (triggerRef.current) {
      // Return focus to whatever opened the dialog.
      triggerRef.current.focus();
      triggerRef.current = null;
    }
  }, [open]);

  // Esc closes; Tab wraps inside the dialog (focus trap). Without the trap
  // Tab walks out of the modal into the still-rendered page underneath —
  // bad for keyboard users and screen readers.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !submitting) {
        onClose();
        return;
      }
      if (e.key !== 'Tab' || !panelRef.current) return;
      const focusable = panelRef.current.querySelectorAll<HTMLElement>(
        FOCUSABLE_SELECTOR,
      );
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const active = document.activeElement as HTMLElement | null;
      if (e.shiftKey && (active === first || !panelRef.current.contains(active))) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && active === last) {
        e.preventDefault();
        first.focus();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, submitting, onClose]);

  if (!open) return null;
  // SSR guard — createPortal needs a DOM target.
  if (typeof document === 'undefined') return null;

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (submitting) return;
    const q = question.trim();
    if (q.length < 4) {
      setError('Please enter at least a few words.');
      return;
    }
    // Public mode reuses the key saved on the /config page (a cookie). There is no
    // key input in this dialog, so block with a clear message if it is gone.
    let apiKey = '';
    if (publicMode) {
      apiKey = getStoredKey().trim();
      if (!apiKey) {
        setError('Your OpenAI API key is missing. Add it on the Config page.');
        return;
      }
    }
    setError(null);
    setSubmitting(true);
    try {
      const run = await api.createRun(q, parentPreset, {
        parent_run_id: parentRunId,
        mode,
        ...(publicMode ? { apiKey } : {}),
        // Reuse the synthesis-model override saved on the /config page.
        ...(publicMode
          ? { synthesisModel: getStoredModel(), outputLength: getStoredOutputLength() }
          : {}),
      });
      router.push(`/run/${run.id}`);
    } catch (e) {
      setError((e as Error).message);
      setSubmitting(false);
    }
  };

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/40 backdrop-blur-sm p-4 sm:p-10"
      onClick={() => {
        if (!submitting) onClose();
      }}
    >
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="followup-title"
        className="w-full max-w-lg rounded-2xl bg-white shadow-xl ring-1 ring-black/5 animate-fade-in"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 pt-4 pb-2">
          <h2 id="followup-title" className="text-base font-semibold text-gray-900">
            Follow-up query
          </h2>
          <button
            type="button"
            onClick={onClose}
            disabled={submitting}
            aria-label="Close"
            className="text-gray-400 hover:text-gray-700 disabled:opacity-50"
          >
            <X size={18} />
          </button>
        </div>

        <form onSubmit={onSubmit} className="px-5 pb-5 space-y-4">
          <textarea
            ref={textareaRef}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="What would you like to ask?"
            rows={5}
            disabled={submitting}
            className="w-full resize-none rounded-xl border border-gray-200 px-4 py-3 text-sm placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-brand-200 focus:border-brand-500"
          />

          <fieldset className="space-y-2">
            <legend className="text-xs uppercase tracking-wider text-gray-500 mb-1">
              How to answer
            </legend>
            {MODE_ORDER.map((m) => (
              <label
                key={m}
                className={cn(
                  'flex items-start gap-3 rounded-xl border px-3 py-2.5 cursor-pointer transition-colors',
                  mode === m
                    ? 'border-brand-500 bg-brand-50'
                    : 'border-gray-200 hover:bg-gray-50',
                )}
              >
                <input
                  type="radio"
                  name="followup-mode"
                  value={m}
                  checked={mode === m}
                  onChange={() => setMode(m)}
                  disabled={submitting}
                  className="mt-1 accent-brand-700"
                />
                <span className="flex-1 text-sm">
                  <span className="font-medium text-gray-900">
                    {MODE_LABEL[m]}
                  </span>
                  <span className="block text-xs text-gray-500 mt-0.5">
                    {MODE_SUBLABEL[m]}
                  </span>
                </span>
              </label>
            ))}
          </fieldset>

          {error && (
            <p className="text-sm text-red-600" role="alert">
              {error}
            </p>
          )}

          <div className="flex items-center justify-end gap-2 pt-1">
            <button
              type="button"
              onClick={onClose}
              disabled={submitting}
              className="rounded-md px-3 py-2 text-sm text-gray-600 hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-transparent transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting}
              className="inline-flex items-center gap-1.5 rounded-xl bg-brand-800 hover:bg-brand-900 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-brand-800 text-white text-sm font-medium px-4 py-2 transition-colors"
            >
              {submitting ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Submitting…
                </>
              ) : (
                <>
                  Submit <ArrowRight size={16} />
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>,
    document.body,
  );
}
