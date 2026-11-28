import { NextResponse, type NextRequest } from 'next/server';

// Three exclusive modes, selected by env at boot (one launch.sh source):
//   1. PUBLIC_MODE === '1'  → mint an opaque `caesar_id` cookie per browser
//      (the tenant identity). Real ownership authz lives in FastAPI; the
//      middleware only ensures every browser carries an identity.
//      NOTE: strict '1' check, not truthiness. launch.sh exports the literal
//      string "0" when --public is off, and "0" is TRUTHY in JS, so a loose
//      check would skip the password gate below and serve every page unauthed.
//   2. else DEMO_PASSWORD set → password gate via the `caesar_auth` cookie.
//   3. else                  → no-op (launch.sh without --public/--password).

const PUBLIC_PATHS = ['/login', '/api/auth/login', '/api/auth/logout'];

// Identity cookie for public mode. Mirrors the LOCKED contract in deps.py
// (CAESAR_ID_COOKIE). HttpOnly + Secure + SameSite=Lax, 400-day lifetime.
const CAESAR_ID_COOKIE = 'caesar_id';
const CAESAR_ID_MAX_AGE = 34560000; // 400 days, Chrome's cookie cap.

export async function middleware(req: NextRequest) {
  // Branch 1: public mode. Mint the identity cookie when absent; on page
  // requests also re-issue it with the same value, so its lifetime counts from
  // the latest visit rather than the first. /api requests leave an existing
  // cookie alone: /api/restore sets a new one, and a second Set-Cookie here
  // could put the old value back. Never gate further here (FastAPI enforces
  // per-row ownership off this cookie).
  if (process.env.PUBLIC_MODE === '1') {
    const res = NextResponse.next();
    const existing = req.cookies.get(CAESAR_ID_COOKIE)?.value;
    if (!existing || !req.nextUrl.pathname.startsWith('/api/')) {
      res.cookies.set(CAESAR_ID_COOKIE, existing || crypto.randomUUID(), {
        httpOnly: true,
        sameSite: 'lax',
        secure: true,
        path: '/',
        maxAge: CAESAR_ID_MAX_AGE,
      });
    }
    return res;
  }

  // Branch 2: password gate. When DEMO_PASSWORD is set, every page + API
  // request requires a matching `caesar_auth` cookie (issued by
  // /api/auth/login).
  const password = process.env.DEMO_PASSWORD;
  if (!password) return NextResponse.next(); // Branch 3: no-op.

  const { pathname } = req.nextUrl;
  if (PUBLIC_PATHS.some((p) => pathname === p || pathname.startsWith(p + '/'))) {
    return NextResponse.next();
  }

  const expected = await sessionToken(password);
  const cookie = req.cookies.get('caesar_auth')?.value;
  if (cookie === expected) return NextResponse.next();

  // API request → return 401 JSON so the SPA can react gracefully.
  if (pathname.startsWith('/api/')) {
    return NextResponse.json({ detail: 'Unauthorized' }, { status: 401 });
  }

  // Page request → bounce to /login with a `next` redirect target.
  const loginUrl = new URL('/login', req.url);
  if (pathname !== '/') loginUrl.searchParams.set('next', pathname);
  return NextResponse.redirect(loginUrl);
}

// Cookie token: SHA-256 of the password plus a versioned salt so changing
// the password invalidates old sessions without storing the password itself
// in the cookie.
async function sessionToken(password: string): Promise<string> {
  const enc = new TextEncoder();
  const buf = await crypto.subtle.digest('SHA-256', enc.encode(`${password}:caesar:v1`));
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

// Run on every request except Next internals and the favicon.
export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico|robots.txt).*)'],
};
