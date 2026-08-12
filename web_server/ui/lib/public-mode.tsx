'use client';

// Runtime public-mode flag, sourced from the server's /version response (the
// single source of truth; see layout.tsx). No NEXT_PUBLIC_* build flag: the
// value is fetched server-side and handed down through this provider so client
// components (QueryInput, FollowUpDialog) can branch on it without a duplicate
// fetch or a build-time/runtime desync.

import { createContext, useContext, type ReactNode } from 'react';

const PublicModeContext = createContext<boolean>(false);

export function PublicModeProvider({
  value,
  children,
}: {
  value: boolean;
  children: ReactNode;
}) {
  return <PublicModeContext.Provider value={value}>{children}</PublicModeContext.Provider>;
}

export function usePublicMode(): boolean {
  return useContext(PublicModeContext);
}

// The user's OpenAI key in public mode is held in a browser cookie so it
// persists across tabs and reloads (set on the dedicated /config page). It is a
// CLIENT-readable cookie (not HttpOnly): the frontend reads it and sends the
// key in the POST /runs body. The server never reads the key from this cookie
// (only from the request body) and never persists it.
export const OPENAI_KEY_COOKIE = 'caesar_openai_key';
const OPENAI_KEY_MAX_AGE = 2592000; // 30 days in seconds.

export function getStoredKey(): string {
  if (typeof document === 'undefined') return '';
  const m = document.cookie.match(/(?:^|;\s*)caesar_openai_key=([^;]*)/);
  return m ? decodeURIComponent(m[1]) : '';
}

export function setStoredKey(value: string): void {
  if (typeof document === 'undefined') return;
  // Secure only over HTTPS so local plain-HTTP dev still works; over the
  // public funnel (HTTPS) the cookie is always Secure.
  const secure = location.protocol === 'https:' ? '; Secure' : '';
  document.cookie =
    `${OPENAI_KEY_COOKIE}=${encodeURIComponent(value)}; Path=/; ` +
    `Max-Age=${OPENAI_KEY_MAX_AGE}; SameSite=Lax${secure}`;
}

export function clearStoredKey(): void {
  if (typeof document === 'undefined') return;
  const secure = location.protocol === 'https:' ? '; Secure' : '';
  document.cookie = `${OPENAI_KEY_COOKIE}=; Path=/; Max-Age=0; SameSite=Lax${secure}`;
}

// The synthesis-model override (public mode) persists in a client cookie too,
// set on the /config page and read at submit time by QueryInput / FollowUpDialog.
// Unlike the key it is not a secret (just a model id), and the server ignores
// it outside public mode. Empty string = use the preset's default model.
export const SYNTH_MODEL_COOKIE = 'caesar_synthesis_model';

export function getStoredModel(): string {
  if (typeof document === 'undefined') return '';
  const m = document.cookie.match(/(?:^|;\s*)caesar_synthesis_model=([^;]*)/);
  return m ? decodeURIComponent(m[1]) : '';
}

export function setStoredModel(value: string): void {
  if (typeof document === 'undefined') return;
  const secure = location.protocol === 'https:' ? '; Secure' : '';
  if (!value) {
    // Empty selection ("Preset default") clears the cookie.
    document.cookie = `${SYNTH_MODEL_COOKIE}=; Path=/; Max-Age=0; SameSite=Lax${secure}`;
    return;
  }
  document.cookie =
    `${SYNTH_MODEL_COOKIE}=${encodeURIComponent(value)}; Path=/; ` +
    `Max-Age=${OPENAI_KEY_MAX_AGE}; SameSite=Lax${secure}`;
}

// Artifact word target, same cookie treatment as the model override: not a
// secret, set on /config, read at submit time, ignored by the server outside
// public mode. Empty string = use the preset's own length (all presets ship
// unconstrained). Stored as the raw number so the caller can send it as-is.
export const OUTPUT_LENGTH_COOKIE = 'caesar_output_length';

// The choices offered on /config. Values must sit inside the server's
// 500..20000 bound (RunCreate.output_length); '' means "preset default".
export const OUTPUT_LENGTH_CHOICES: { value: string; label: string }[] = [
  { value: '', label: 'Preset default (unconstrained)' },
  { value: '1500', label: 'Brief (~1,500 words)' },
  { value: '3000', label: 'Standard (~3,000 words)' },
  { value: '6000', label: 'Detailed (~6,000 words)' },
];

export function getStoredOutputLength(): string {
  if (typeof document === 'undefined') return '';
  const m = document.cookie.match(/(?:^|;\s*)caesar_output_length=([^;]*)/);
  if (!m) return '';
  // Accept only a value we actually offer. A digits-only check would let a
  // hand-edited or stale cookie (say 100, or a value from a future revision of
  // this list) through to the server, where it fails RunCreate's 500..20000
  // bound and 422s every submit with a pydantic message the user cannot act on
  // -- from a cookie they cannot see. Anything unrecognised falls back to the
  // preset default. Checking membership rather than re-stating the numeric
  // bound also keeps the server's range from being duplicated here.
  const v = decodeURIComponent(m[1]);
  return OUTPUT_LENGTH_CHOICES.some((c) => c.value === v) ? v : '';
}

export function setStoredOutputLength(value: string): void {
  if (typeof document === 'undefined') return;
  const secure = location.protocol === 'https:' ? '; Secure' : '';
  if (!value) {
    document.cookie = `${OUTPUT_LENGTH_COOKIE}=; Path=/; Max-Age=0; SameSite=Lax${secure}`;
    return;
  }
  document.cookie =
    `${OUTPUT_LENGTH_COOKIE}=${encodeURIComponent(value)}; Path=/; ` +
    `Max-Age=${OPENAI_KEY_MAX_AGE}; SameSite=Lax${secure}`;
}


