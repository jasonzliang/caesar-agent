import { NextResponse, type NextRequest } from 'next/server';

// POST { password } → if it matches DEMO_PASSWORD, issue an HttpOnly cookie
// that the middleware accepts. Returns 200 on success, 401 on bad password,
// 400 if auth is disabled (i.e. DEMO_PASSWORD is unset).

export const runtime = 'edge';

export async function POST(req: NextRequest) {
  const password = process.env.DEMO_PASSWORD;
  if (!password) {
    return NextResponse.json(
      { detail: 'Password auth is not enabled on this server.' },
      { status: 400 },
    );
  }

  let body: { password?: unknown } = {};
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ detail: 'Invalid request body.' }, { status: 400 });
  }
  const submitted = typeof body.password === 'string' ? body.password : '';

  if (!constantTimeEqual(submitted, password)) {
    return NextResponse.json({ detail: 'Incorrect password.' }, { status: 401 });
  }

  const enc = new TextEncoder();
  const buf = await crypto.subtle.digest('SHA-256', enc.encode(`${password}:caesar:v1`));
  const token = Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');

  const res = NextResponse.json({ ok: true });
  res.cookies.set('caesar_auth', token, {
    httpOnly: true,
    sameSite: 'lax',
    path: '/',
    maxAge: 60 * 60 * 24 * 7, // 7 days
  });
  return res;
}

// Length-aware constant-time string compare so we don't leak the password
// length through response timing.
function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}
