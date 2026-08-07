"""Tenant-identity dependencies for owner-scoped run access.

In public (bring-your-own-key) mode every browser carries an opaque
`caesar_id` HttpOnly cookie minted by the Next.js middleware; its value is
the run's `owner_id`. `current_owner` reads that cookie (never a forgeable
request header, since the Next rewrites() proxy forwards client headers
verbatim) and `get_owned_run` enforces ownership on every run-scoped lookup.

When public mode is off, `current_owner` returns None and `owner_id == None`
emits SQL `IS NULL`, which matches all legacy rows: behavior is identical to
today's single-tenant deploy.
"""
from __future__ import annotations

import hashlib
import hmac

from fastapi import HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .models import Run

# Opaque per-browser tenant cookie. Must match web_server/ui/middleware.ts.
CAESAR_ID_COOKIE = "caesar_id"

# Admin step-up cookie, issued by the Next.js /api/auth/login route on a correct
# operator password. Its value is a token derived from the password; the
# derivation MUST mirror web_server/ui/middleware.ts `sessionToken()`.
ADMIN_COOKIE = "caesar_auth"

# Allowed cookie charset: uuid/hex characters (digits, a-f, dashes). A garbage
# value should not be able to mint or address owner rows.
_OWNER_ALLOWED = set("0123456789abcdefABCDEF-")
_OWNER_MIN_LEN = 16
_OWNER_MAX_LEN = 64

# Lifetime of the identity cookie when (re)set server-side; matches the value
# the Next.js middleware mints with (web_server/ui/middleware.ts).
CAESAR_ID_MAX_AGE = 2592000  # 30 days in seconds.


def is_valid_owner_token(value: str | None) -> bool:
    """True when `value` is a plausible caesar_id (charset + length sane)."""
    return (
        bool(value)
        and _OWNER_MIN_LEN <= len(value) <= _OWNER_MAX_LEN
        and all(ch in _OWNER_ALLOWED for ch in value)
    )


def current_owner(request: Request) -> str | None:
    """Resolve the calling browser's tenant identity.

    Returns None when public mode is off (single-tenant behavior). Otherwise
    reads the caesar_id cookie and raises 401 when it is absent or fails a
    charset/length sanity check; returns the validated value when it passes.
    """
    settings = get_settings()
    if not settings.public_mode:
        return None
    value = request.cookies.get(CAESAR_ID_COOKIE)
    if not value:
        raise HTTPException(status_code=401, detail="Missing session cookie.")
    if not is_valid_owner_token(value):
        raise HTTPException(status_code=401, detail="Invalid session cookie.")
    return value


def _admin_session_token(password: str) -> str:
    """Derive the admin cookie value from the operator password.

    MUST mirror web_server/ui/middleware.ts `sessionToken()`:
    sha256(f"{password}:caesar:v1"). The versioned salt means changing the
    password invalidates old admin sessions.
    """
    return hashlib.sha256(f"{password}:caesar:v1".encode()).hexdigest()


def is_admin(request: Request) -> bool:
    """True when the caller has stepped up to admin.

    Admin is a public-mode-only elevation on top of the anonymous caesar_id
    session: it requires public mode, a configured operator password, and a
    caesar_auth cookie matching that password. Admins bypass per-owner scoping
    (see every user's runs and wipe them all). Constant-time compare so the
    cookie can't be brute-forced by timing.
    """
    settings = get_settings()
    if not settings.public_mode or not settings.demo_password:
        return False
    cookie = request.cookies.get(ADMIN_COOKIE)
    if not cookie:
        return False
    return hmac.compare_digest(cookie, _admin_session_token(settings.demo_password))


async def get_owned_run(
    run_id: str, owner: str | None, session: AsyncSession, *, admin: bool = False
) -> Run:
    """Load a Run, enforcing ownership.

    Raises 404 when the run does not exist, and 404 (NOT 403, to avoid run-id
    enumeration) when it exists but belongs to a different owner. In
    single-tenant mode `owner` is None and the ownership check is skipped;
    `admin=True` skips it too (public-mode operator sees every run).
    """
    run = await session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    if owner is not None and not admin and run.owner_id != owner:
        raise HTTPException(status_code=404, detail="Run not found.")
    return run
