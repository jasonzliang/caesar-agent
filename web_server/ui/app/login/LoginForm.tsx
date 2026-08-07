'use client';

import { useState } from 'react';
import { useSearchParams } from 'next/navigation';

// Constrain `next` to internal paths so a malicious /login?next=https://evil
// link can't redirect users off-site after they authenticate.
function safeNext(raw: string | null): string {
  if (!raw) return '/';
  if (!raw.startsWith('/') || raw.startsWith('//')) return '/';
  return raw;
}

export function LoginForm() {
  const params = useSearchParams();
  const next = safeNext(params.get('next'));
  const [password, setPassword] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ password }),
      });
      if (res.ok) {
        // Hard navigation so the middleware re-evaluates the cookie it
        // just received in the Set-Cookie header.
        window.location.href = next;
        return;
      }
      const body = await res.json().catch(() => ({}));
      setError(body?.detail || `Sign-in failed (${res.status})`);
    } catch {
      setError('Network error. Try again.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form
      onSubmit={onSubmit}
      className="rounded-2xl border border-gray-200 bg-white p-6 space-y-4 shadow-sm"
    >
      <label className="block">
        <span className="text-xs uppercase tracking-wider font-medium text-gray-500">
          Password
        </span>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          autoFocus
          required
          autoComplete="current-password"
          className="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:border-brand-600 focus:ring-2 focus:ring-brand-600/20"
          // Chrome autofill / password managers inject attributes
          // (`autofill-information`, `autofill-prediction`) on the client
          // after SSR but before hydration, which React flags as a
          // mismatch. Suppressing here is the standard workaround.
          suppressHydrationWarning
        />
      </label>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          {error}
        </div>
      )}

      <button
        type="submit"
        disabled={submitting || !password}
        className="w-full rounded-lg bg-brand-700 px-4 py-2 text-sm font-medium text-white hover:bg-brand-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {submitting ? 'Signing in…' : 'Sign in'}
      </button>
    </form>
  );
}
