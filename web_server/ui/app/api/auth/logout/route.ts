import { NextResponse } from 'next/server';

// POST → clear the auth cookie. Called by the logout button. No body, no
// password check — anyone holding the cookie can sign themselves out.

export const runtime = 'edge';

export async function POST() {
  const res = NextResponse.json({ ok: true });
  res.cookies.set('caesar_auth', '', {
    httpOnly: true,
    sameSite: 'lax',
    path: '/',
    maxAge: 0,
  });
  return res;
}
