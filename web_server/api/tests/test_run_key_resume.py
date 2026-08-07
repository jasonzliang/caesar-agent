"""Public-mode per-run key: persist on submit, delete on finish (resume support)."""
from __future__ import annotations

import asyncio
import sqlite3
import uuid

import pytest

from .conftest import make_query

VALID_KEY = "sk-" + "z" * 24


def _stored_key(run_id: str) -> str | None:
    from app.config import get_settings

    con = sqlite3.connect(get_settings().db_path)
    try:
        row = con.execute(
            "SELECT run_api_key FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
    finally:
        con.close()
    return row[0] if row else None


@pytest.mark.asyncio
async def test_public_key_persisted_then_cleared(client, monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "1")
    from app.config import get_settings

    get_settings.cache_clear()
    owner = uuid.uuid4().hex
    r = await client.post(
        "/runs",
        cookies={"caesar_id": owner},
        json={"query": make_query(), "preset": "fast", "api_key": VALID_KEY},
    )
    assert r.status_code == 201
    run_id = r.json()["id"]

    # After submit the key is stored so a restart can resume from checkpoint.
    assert _stored_key(run_id) == VALID_KEY

    # Once the (dry) run reaches a terminal state, the key is deleted.
    for _ in range(80):
        d = await client.get(f"/runs/{run_id}", cookies={"caesar_id": owner})
        if d.json()["status"] in ("completed", "failed"):
            break
        await asyncio.sleep(0.25)
    assert _stored_key(run_id) is None


@pytest.mark.asyncio
async def test_nonpublic_stores_no_key(client):
    r = await client.post("/runs", json={"query": make_query(), "preset": "fast"})
    assert r.status_code == 201
    assert _stored_key(r.json()["id"]) is None
