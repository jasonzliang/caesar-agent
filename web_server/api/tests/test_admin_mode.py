"""Public-mode admin step-up: password cookie sees + wipes all users' runs."""
from __future__ import annotations

import uuid

import pytest

from .conftest import make_query

VALID_KEY = "sk-" + "x" * 24
PW = "s3cret-admin-pw"


def _admin_cookies(owner: str) -> dict:
    """A browser cookie jar that is both a tenant (caesar_id) AND admin."""
    from app.deps import _admin_session_token

    return {"caesar_id": owner, "caesar_auth": _admin_session_token(PW)}


async def _submit(client, owner: str) -> str:
    r = await client.post(
        "/runs",
        cookies={"caesar_id": owner},
        json={"query": make_query(), "preset": "fast", "api_key": VALID_KEY},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


@pytest.mark.asyncio
async def test_admin_sees_and_opens_all_runs(client, monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "1")
    monkeypatch.setenv("DEMO_PASSWORD", PW)
    from app.config import get_settings

    get_settings.cache_clear()

    a, b = uuid.uuid4().hex, uuid.uuid4().hex
    run_a = await _submit(client, a)
    run_b = await _submit(client, b)

    # Normal tenant A sees only its own run and cannot open B's.
    la = await client.get("/runs", cookies={"caesar_id": a})
    ids_a = {r["id"] for r in la.json()}
    assert run_a in ids_a and run_b not in ids_a
    assert (await client.get(f"/runs/{run_b}", cookies={"caesar_id": a})).status_code == 404

    # Admin (A's browser stepped up) sees BOTH and can open B's run + owner shows.
    ladmin = await client.get("/runs", cookies=_admin_cookies(a))
    rows = {r["id"]: r for r in ladmin.json()}
    assert run_a in rows and run_b in rows
    assert rows[run_b]["owner_id"] == b
    assert (await client.get(f"/runs/{run_b}", cookies=_admin_cookies(a))).status_code == 200

    # /whoami reflects admin state.
    w = await client.get("/whoami", cookies=_admin_cookies(a))
    assert w.json()["is_admin"] is True
    assert (await client.get("/whoami", cookies={"caesar_id": a})).json()["is_admin"] is False


@pytest.mark.asyncio
async def test_wrong_password_is_not_admin(client, monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "1")
    monkeypatch.setenv("DEMO_PASSWORD", PW)
    from app.config import get_settings

    get_settings.cache_clear()

    a, b = uuid.uuid4().hex, uuid.uuid4().hex
    await _submit(client, a)
    run_b = await _submit(client, b)

    bad = {"caesar_id": a, "caesar_auth": "deadbeef" * 8}
    la = await client.get("/runs", cookies=bad)
    assert run_b not in {r["id"] for r in la.json()}
    assert (await client.get(f"/runs/{run_b}", cookies=bad)).status_code == 404


@pytest.mark.asyncio
async def test_admin_wipes_all_users_runs(client, monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "1")
    monkeypatch.setenv("DEMO_PASSWORD", PW)
    from app.config import get_settings

    get_settings.cache_clear()

    a, b = uuid.uuid4().hex, uuid.uuid4().hex
    await _submit(client, a)
    await _submit(client, b)

    wipe = await client.delete("/runs?confirm=yes", cookies=_admin_cookies(a))
    assert wipe.status_code == 200
    # Everyone's runs are gone.
    assert (await client.get("/runs", cookies=_admin_cookies(a))).json() == []


@pytest.mark.asyncio
async def test_admin_disabled_without_password(client, monkeypatch):
    """Public mode but no operator password → admin can never be granted."""
    monkeypatch.setenv("PUBLIC_MODE", "1")
    monkeypatch.delenv("DEMO_PASSWORD", raising=False)
    from app.config import get_settings

    get_settings.cache_clear()

    a, b = uuid.uuid4().hex, uuid.uuid4().hex
    await _submit(client, a)
    run_b = await _submit(client, b)

    # Even presenting a caesar_auth cookie grants nothing when no password is set.
    from app.deps import _admin_session_token

    cookies = {"caesar_id": a, "caesar_auth": _admin_session_token("")}
    assert run_b not in {r["id"] for r in (await client.get("/runs", cookies=cookies)).json()}
