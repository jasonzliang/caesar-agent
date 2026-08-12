'use client';

import { useEffect, useState, type FormEvent } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { api, type Preset } from '@/lib/api';
import {
  usePublicMode,
  getStoredKey,
  getStoredModel,
  getStoredOutputLength,
} from '@/lib/public-mode';
import { PresetToggle } from './PresetToggle';
import { ExamplePrompts } from './ExamplePrompts';
import { ArrowRight, Loader2 } from 'lucide-react';

type Props = { presets: Preset[] };

export function QueryInput({ presets }: Props) {
  const router = useRouter();
  const publicMode = usePublicMode();
  const [query, setQuery] = useState('');
  const [preset, setPreset] = useState(presets[0]?.id ?? 'fast');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Bumped whenever an error is shown so the error line re-mounts and its
  // shake animation re-fires (CSS animations only play on mount).
  const [errorNonce, setErrorNonce] = useState(0);
  // Bumped when a submit is blocked for a missing key, so the persistent no-key
  // prompt shakes on that rejected click but stays calm on initial load.
  const [nokeyNonce, setNokeyNonce] = useState(0);
  // In public mode the OpenAI key (and the optional synthesis-model override)
  // are set on the dedicated /config page and saved to cookies. Track whether a
  // key is set so we can nudge the user.
  const [hasKey, setHasKey] = useState<boolean | null>(null);

  useEffect(() => {
    if (!publicMode) return;
    setHasKey(getStoredKey().length > 0);
  }, [publicMode]);

  // Transient errors only (validation / API failure). The persistent no-key
  // prompt does NOT go through this, so it never shakes on load.
  const showError = (msg: string) => {
    setError(msg);
    setErrorNonce((n) => n + 1);
  };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (submitting) return;
    // Clear any prior message first so a stale error can't linger and mask
    // the prompt we actually want to surface on this attempt.
    setError(null);
    // In public mode a missing key is the primary blocker — nothing runs
    // without it — so check it BEFORE the query-length hint. Otherwise a
    // short query would mask the persistent "Add your OpenAI API key" prompt.
    let apiKey = '';
    if (publicMode) {
      apiKey = getStoredKey().trim();
      if (!apiKey) {
        // Leave error cleared so the persistent "Add your OpenAI API key" line
        // shows; bump nokeyNonce so it shakes on this rejected click (nonce 0
        // on load means no shake until a click).
        setHasKey(false);
        setNokeyNonce((n) => n + 1);
        return;
      }
    }
    if (query.trim().length < 4) {
      showError('Please enter at least a few words.');
      return;
    }
    setSubmitting(true);
    try {
      const run = await api.createRun(query.trim(), preset, {
        ...(publicMode ? { apiKey } : {}),
        // Synthesis-model override set on the /config page (empty → preset
        // default; createRun omits it when falsy).
        ...(publicMode
          ? { synthesisModel: getStoredModel(), outputLength: getStoredOutputLength() }
          : {}),
      });
      router.push(`/run/${run.id}`);
    } catch (e) {
      showError((e as Error).message);
      setSubmitting(false);
    }
  };

  return (
    <form onSubmit={onSubmit} className="w-full" suppressHydrationWarning>
      <div className="rounded-2xl border border-gray-200 bg-white shadow-sm focus-within:ring-2 focus-within:ring-brand-200 transition-shadow">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask Caesar a research question, e.g. &quot;Apply the mathematical structure of calculus to cooking.&quot;"
          rows={3}
          className="w-full resize-none rounded-t-2xl px-5 pt-4 pb-2 text-base placeholder:text-gray-400 focus:outline-none"
          disabled={submitting}
          suppressHydrationWarning
        />
        <div className="flex flex-wrap items-center gap-3 px-3 pb-3 pt-1 border-t border-gray-100">
          <PresetToggle presets={presets} value={preset} onChange={setPreset} />
          <div className="flex-1" />
          <button
            type="submit"
            disabled={submitting}
            className="inline-flex items-center gap-1.5 rounded-xl bg-brand-800 hover:bg-brand-900 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium px-4 py-2 transition-colors"
          >
            {submitting ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                Starting…
              </>
            ) : (
              <>
                Run Caesar <ArrowRight size={16} />
              </>
            )}
          </button>
        </div>
      </div>
      {error ? (
        <p
          key={errorNonce}
          className="mt-2 text-sm text-red-600 motion-safe:animate-shake"
          role="alert"
        >
          {error}
        </p>
      ) : publicMode && hasKey === false ? (
        <p
          key={`nokey-${nokeyNonce}`}
          className={`mt-2 text-sm text-red-600${nokeyNonce > 0 ? ' motion-safe:animate-shake' : ''}`}
          suppressHydrationWarning
        >
          Add your{' '}
          <Link href="/config" className="underline hover:text-red-700">
            OpenAI API key
          </Link>{' '}
          to run a query.
        </p>
      ) : null}
      <ExamplePrompts onPick={setQuery} disabled={submitting} />
    </form>
  );
}
