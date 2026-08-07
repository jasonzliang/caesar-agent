'use client';

import { useEffect, useState } from 'react';

// Small footer link (public mode only) that steps the current browser up to
// admin. Clicking reveals an inline password field; a correct password issues
// the caesar_auth cookie via /api/auth/login, which the FastAPI layer then
// treats as admin (see + wipe every user's runs). Reuses the existing
// password-login route — no new auth primitive.
export function AdminLink() {
  const [isAdmin, setIsAdmin] = useState(false);
  const [open, setOpen] = useState(false);
  const [pw, setPw] = useState('');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetch('/api/whoami', { cache: 'no-store' })
      .then((r) => r.json())
      .then((d) => setIsAdmin(Boolean(d?.is_admin)))
      .catch(() => setIsAdmin(false));
  }, []);

  const login = async () => {
    if (busy || pw.trim().length === 0) return;
    setBusy(true);
    setErr(null);
    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password: pw }),
      });
      if (!res.ok) {
        // 400 = the server has no password configured, so nothing you type can
        // work. "Failed (400)" sent people hunting for a typo instead. Phrased
        // about the feature, not the field, so it can't read as "you left it
        // blank" next to the password you just typed.
        throw new Error(
          res.status === 401 ? 'Wrong password.'
            : res.status === 400 ? 'Admin not enabled.'
              : `Failed (${res.status}).`,
        );
      }
      // Full navigation so the server re-reads the new cookie as admin.
      window.location.href = '/runs';
    } catch (e) {
      setErr((e as Error).message);
      setBusy(false);
    }
  };

  const logout = async () => {
    await fetch('/api/auth/logout', { method: 'POST' }).catch(() => {});
    window.location.reload();
  };

  if (isAdmin) {
    // Toggle: clicking again exits admin mode.
    return (
      <button
        type="button"
        onClick={logout}
        title="Click to exit admin mode"
        className="text-green-700 hover:text-gray-700 hover:underline"
      >
        Admin ✓
      </button>
    );
  }

  if (!open) {
    return (
      <button type="button" onClick={() => setOpen(true)} className="hover:text-gray-700 hover:underline">
        Admin
      </button>
    );
  }

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        login();
      }}
      className="inline-flex items-center gap-2 align-middle"
    >
      {/* Don't let managers SAVE this admin password — otherwise Chrome
          cross-fills it into the /config OpenAI-key field on later visits. */}
      <input
        type="password"
        name="caesar-admin-pw"
        value={pw}
        onChange={(e) => setPw(e.target.value)}
        placeholder="Admin password"
        autoFocus
        autoComplete="off"
        data-1p-ignore="true"
        data-lpignore="true"
        data-bwignore="true"
        data-form-type="other"
        spellCheck={false}
        className="rounded border border-gray-200 px-2 py-1 text-xs focus:outline-none focus:ring-2 focus:ring-brand-200"
      />
      <button type="submit" disabled={busy} className="underline hover:text-gray-700 disabled:opacity-50">
        {busy ? '…' : 'Enter'}
      </button>
      {err && <span className="text-red-600">{err}</span>}
    </form>
  );
}
