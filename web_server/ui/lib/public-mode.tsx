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
