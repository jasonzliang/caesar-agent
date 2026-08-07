"""Public (bring-your-own-key) mode: privacy + tenant-isolation tests.

These assert the guarantees that make public mode safe to expose to anonymous
browsers:

  * per-browser ownership scoping (one opaque caesar_id cookie == one tenant);
  * cross-owner IDOR is blocked with 404 (not 403) on every run-scoped route;
  * a run-scoped request with no cookie is rejected (401);
  * a submission with no api_key is rejected (400);
  * the submitted api_key never lands in SQLite (error_message), in
    run_events payloads, or in any /runs or /runs/{id} response body;
  * with the flag OFF, existing single-tenant behavior is byte-for-byte intact.

All tests run with CAESAR_DRY_RUN=1 (forced by conftest), so no LLM calls
happen and a full run finishes in a few seconds. The leak test monkeypatches
_dry_run to raise a FatalLLMError whose message embeds the key, which is the
real-world sink (litellm AuthenticationError text) the scrubber must catch.
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from .conftest import make_query

pytestmark = pytest.mark.asyncio

# A syntactically valid OpenAI-style key (sk- prefix, >=20 chars) so the
# RunCreate shape validator accepts it. Long random tail so the scrubber's
# generic sk-[A-Za-z0-9_-]{20,} pattern has something to match.
FAKE_KEY = "sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6"


def _owner_cookie() -> str:
    """A caesar_id value that passes deps.current_owner's charset/length check
    (hex/uuid chars, length 16..64). uuid4().hex is 32 hex chars."""
    return uuid.uuid4().hex


def _cookie_header(value: str | None) -> dict[str, str]:
    """Build an explicit Cookie request header. We set the header directly
    rather than httpx's per-request cookies= kwarg (which is deprecated and has
    ambiguous jar-persistence semantics) so cookie A can never bleed into a
    cookie-B request through the shared client jar. This also mirrors how the
    Next.js rewrites() proxy forwards the browser Cookie header verbatim."""
    return {"cookie": f"caesar_id={value}"} if value is not None else {}


@pytest_asyncio.fixture
async def public_client(monkeypatch):
    """An httpx client mounted on a FastAPI app with PUBLIC_MODE forced on.

    The autouse isolated_data_dir fixture (conftest) has already set the
    per-test SQLite/runs isolation and cleared the settings cache. We layer
    PUBLIC_MODE=1 on top and re-clear the cache so current_owner sees public
    mode. The app/get_settings read settings lazily at request time, so this
    flips behavior for every request through this client.
    """
    monkeypatch.setenv("PUBLIC_MODE", "1")
    from app.config import get_settings  # noqa: WPS433
    get_settings.cache_clear()
    assert get_settings().public_mode is True

    # The app's public-mode lifespan strips operator LLM keys from the
    # process-global os.environ (fail-closed). Snapshot + restore around the
    # lifespan so that strip cannot bleed into other tests in this process.
    _saved = {
        k: os.environ.get(k)
        for k in ("OPENAI_API_KEY", "CHROMA_OPENAI_API_KEY",
                  "ANTHROPIC_API_KEY", "GOOGLE_API_KEY")
    }
    try:
        from app.main import app  # noqa: WPS433
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            async with app.router.lifespan_context(app):
                yield ac
    finally:
        for _k, _v in _saved.items():
            if _v is None:
                os.environ.pop(_k, None)
            else:
                os.environ[_k] = _v


async def _submit(client: AsyncClient, *, cookie: str | None, api_key: str | None):
    body: dict = {"query": make_query(), "preset": "fast"}
    if api_key is not None:
        body["api_key"] = api_key
    return await client.post("/runs", json=body, headers=_cookie_header(cookie))


async def _wait_terminal(client: AsyncClient, run_id: str, *, cookie: str, attempts: int = 40):
    final = None
    for _ in range(attempts):
        await asyncio.sleep(0.1)
        rg = await client.get(f"/runs/{run_id}", headers=_cookie_header(cookie))
        if rg.status_code == 200 and rg.json()["status"] in ("completed", "failed"):
            final = rg.json()
            break
    return final


# ---------------------------------------------------------------------------
# Ownership scoping + cross-owner IDOR
# ---------------------------------------------------------------------------

async def test_run_visible_only_to_owning_cookie(public_client):
    """Cookie A creates a run; A lists it, B does not."""
    owner_a = _owner_cookie()
    owner_b = _owner_cookie()

    r = await _submit(public_client, cookie=owner_a, api_key=FAKE_KEY)
    assert r.status_code == 201, r.text
    run_id = r.json()["id"]

    list_a = await public_client.get("/runs", headers=_cookie_header(owner_a))
    assert list_a.status_code == 200
    assert run_id in [row["id"] for row in list_a.json()]

    list_b = await public_client.get("/runs", headers=_cookie_header(owner_b))
    assert list_b.status_code == 200
    assert run_id not in [row["id"] for row in list_b.json()]


async def test_cross_owner_run_scoped_endpoints_404(public_client):
    """Every run-scoped route 404s when reached with a non-owning cookie:
    no existence/enumeration oracle, no cross-tenant data leak."""
    owner_a = _owner_cookie()
    owner_b = _owner_cookie()

    r = await _submit(public_client, cookie=owner_a, api_key=FAKE_KEY)
    assert r.status_code == 201, r.text
    run_id = r.json()["id"]
    cookie_b = _cookie_header(owner_b)

    # Owner A can read it (sanity: the run really exists).
    assert (await public_client.get(
        f"/runs/{run_id}", headers=_cookie_header(owner_a),
    )).status_code == 200

    # Owner B is blocked on every run-scoped surface with 404.
    assert (await public_client.get(f"/runs/{run_id}", headers=cookie_b)).status_code == 404
    assert (await public_client.get(f"/runs/{run_id}/stream", headers=cookie_b)).status_code == 404
    assert (await public_client.get(f"/runs/{run_id}/graph", headers=cookie_b)).status_code == 404
    assert (await public_client.get(
        f"/runs/{run_id}/synthesis", params={"draft": "latest"}, headers=cookie_b,
    )).status_code == 404
    assert (await public_client.get(
        f"/runs/{run_id}/file/images/x.png", headers=cookie_b,
    )).status_code == 404


async def test_missing_cookie_rejected_401_on_run_scoped(public_client):
    """A request with no caesar_id cookie is rejected (401) on a run-scoped
    endpoint in public mode."""
    owner_a = _owner_cookie()
    r = await _submit(public_client, cookie=owner_a, api_key=FAKE_KEY)
    assert r.status_code == 201, r.text
    run_id = r.json()["id"]

    # No cookie at all -> current_owner raises 401 before any DB lookup.
    no_cookie = await public_client.get(f"/runs/{run_id}")
    assert no_cookie.status_code == 401

    # The collection (list) endpoint is also run-scoped: 401 with no cookie.
    list_no_cookie = await public_client.get("/runs")
    assert list_no_cookie.status_code == 401

    # Submitting with no cookie is likewise rejected before the run is created.
    submit_no_cookie = await _submit(public_client, cookie=None, api_key=FAKE_KEY)
    assert submit_no_cookie.status_code == 401


# ---------------------------------------------------------------------------
# api_key required + never leaked
# ---------------------------------------------------------------------------

async def test_create_without_api_key_rejected_400(public_client):
    """In public mode a submission with no api_key in the body is a 400."""
    owner_a = _owner_cookie()
    r = await _submit(public_client, cookie=owner_a, api_key=None)
    assert r.status_code == 400, r.text


async def test_api_key_never_appears_in_any_sink(public_client, monkeypatch):
    """The submitted api_key must never surface in:
      - the persisted run row (error_message in SQLite),
      - any run_events payload,
      - any /runs or /runs/{id} response body.

    We force the worst case: a FatalLLMError whose message embeds the key
    (mirroring litellm's AuthenticationError text), which flows into
    error_message, the persisted run_events row, and the SSE frame. The
    scrubber must redact it before all three sinks.
    """
    from app.config import ensure_caesar_on_path
    ensure_caesar_on_path()
    from rome.llm_handler import FatalLLMError

    leaky = f"Incorrect API key provided: {FAKE_KEY}. You can find your key at ..."

    async def _explode(self, *args, **kwargs):
        raise FatalLLMError(leaky)

    monkeypatch.setattr("app.job_runner.JobPool._dry_run", _explode)

    owner_a = _owner_cookie()
    r = await _submit(public_client, cookie=owner_a, api_key=FAKE_KEY)
    assert r.status_code == 201, r.text
    run_id = r.json()["id"]

    # The 201 create response body itself must not echo the key.
    assert FAKE_KEY not in r.text

    final = await _wait_terminal(public_client, run_id, cookie=owner_a)
    assert final is not None, "run did not reach terminal status"
    assert final["status"] == "failed", final

    # 1. Detail response body (includes error_message + events) has no key.
    assert FAKE_KEY not in json.dumps(final)
    assert FAKE_KEY not in (final.get("error_message") or "")

    # 2. The listing response body has no key.
    listing = await public_client.get("/runs", headers=_cookie_header(owner_a))
    assert FAKE_KEY not in listing.text

    # 3. Directly inspect the persisted DB row + every run_events payload so we
    #    are not merely trusting the response-model exclude.
    from sqlalchemy import select  # noqa: WPS433

    from app.db import SessionLocal  # noqa: WPS433
    from app.models import Run, RunEvent  # noqa: WPS433

    async with SessionLocal() as session:
        row = await session.get(Run, run_id)
        assert row is not None
        assert FAKE_KEY not in (row.error_message or "")
        # owner_id was stamped from the cookie.
        assert row.owner_id == owner_a

        events = (
            await session.execute(select(RunEvent).where(RunEvent.run_id == run_id))
        ).scalars().all()
        assert events, "expected at least one persisted run event"
        for ev in events:
            assert FAKE_KEY not in (ev.payload or "")
        # The error event surfaced the failure but with the key redacted.
        error_payloads = [
            json.loads(ev.payload) for ev in events
            if ev.event == "error" and ev.payload
        ]
        assert error_payloads, "expected a persisted error event"
        assert any("[REDACTED]" in (p.get("message") or "") for p in error_payloads), (
            error_payloads
        )


# ---------------------------------------------------------------------------
# Flag OFF: existing single-tenant behavior unchanged (light regression)
# ---------------------------------------------------------------------------

async def test_private_mode_unchanged_no_cookie_no_key(client):
    """With public_mode OFF (the default `client` fixture), a run is creatable
    with neither a cookie nor an api_key, and it lists. owner_id stays NULL,
    so owner_id == None (IS NULL) keeps matching every row exactly as before."""
    r = await client.post("/runs", json={"query": make_query(), "preset": "fast"})
    assert r.status_code == 201, r.text
    run_id = r.json()["id"]

    listing = await client.get("/runs")
    assert listing.status_code == 200
    assert run_id in [row["id"] for row in listing.json()]

    detail = await client.get(f"/runs/{run_id}")
    assert detail.status_code == 200


# ---------------------------------------------------------------------------
# Recovery code: /whoami + /restore
# ---------------------------------------------------------------------------

async def test_whoami_returns_owner_id_in_public_mode(public_client):
    """/whoami echoes the caller's caesar_id (the recovery code)."""
    owner = _owner_cookie()
    r = await public_client.get("/whoami", headers=_cookie_header(owner))
    assert r.status_code == 200
    body = r.json()
    assert body["public_mode"] is True
    assert body["owner_id"] == owner


async def test_restore_sets_cookie_and_grants_runs(public_client):
    """/restore with a recovery code sets caesar_id (HttpOnly) and that browser
    then sees the owner's runs."""
    owner = _owner_cookie()
    r = await _submit(public_client, cookie=owner, api_key=FAKE_KEY)
    assert r.status_code == 201, r.text
    run_id = r.json()["id"]

    resp = await public_client.post("/restore", json={"code": owner})
    assert resp.status_code == 200, resp.text
    assert resp.json()["owner_id"] == owner
    set_cookie = resp.headers.get("set-cookie", "")
    assert f"caesar_id={owner}" in set_cookie
    assert "HttpOnly" in set_cookie

    listing = await public_client.get("/runs", headers=_cookie_header(owner))
    assert run_id in [row["id"] for row in listing.json()]


async def test_restore_invalid_code_400(public_client):
    """Garbage / wrong-length / wrong-charset codes are rejected, so a junk
    value cannot mint an owner identity."""
    for bad in ["", "short", "not hex!!", "z" * 100]:
        resp = await public_client.post("/restore", json={"code": bad})
        assert resp.status_code == 400, (bad, resp.text)


async def test_whoami_restore_off_when_not_public(client):
    """With public mode OFF, /whoami reports it and /restore is a 404."""
    who = await client.get("/whoami")
    assert who.status_code == 200
    assert who.json()["public_mode"] is False
    resp = await client.post("/restore", json={"code": _owner_cookie()})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Secret scrubbing: catch OpenAI's masked echo too
# ---------------------------------------------------------------------------

async def test_scrub_redacts_masked_openai_key():
    """OpenAI 401s echo the key masked (sk-proj-***...9f9f), embedding the real
    prefix + last 4. The scrubber must redact that form, not just the raw key."""
    from app.config import ensure_caesar_on_path
    ensure_caesar_on_path()
    from app.job_runner import _scrub_secrets

    masked = (
        "AuthenticationError: Incorrect API key provided: "
        "sk-proj-***************************************************9f9f."
    )
    out = _scrub_secrets(masked, api_key="sk-proj-realABCDEFGHIJ1234567890")
    assert "sk-proj-" not in out
    assert "9f9f" not in out
    assert "[REDACTED]" in out
