"""Public-mode artifact-length override: validation, persistence, gating, restart.

The value lands on ArtifactSynthesizer.synthesis_max_length, which caesar already
threads through both the per-draft prompt and the merge prompt. All four presets
ship it as null (unconstrained), which is why an unset run can produce ~8k words.
"""
from __future__ import annotations

import uuid

import pytest

from .conftest import make_query

VALID_KEY = "sk-" + "x" * 24
OWNER_COOKIE = {"caesar_id": uuid.uuid4().hex}


@pytest.fixture
def public(monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "1")
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def _submit(client, **extra):
    return await client.post(
        "/runs",
        cookies=OWNER_COOKIE,
        json={"query": make_query(), "preset": "fast", "api_key": VALID_KEY, **extra},
    )


@pytest.mark.asyncio
async def test_accepted_and_persisted(client, public):
    r = await _submit(client, output_length=3000)
    assert r.status_code == 201, r.text
    run_id = r.json()["id"]

    from sqlalchemy import select

    from app.db import SessionLocal
    from app.models import Run

    async with SessionLocal() as s:
        row = (await s.execute(select(Run).where(Run.id == run_id))).scalar_one()
        assert row.output_length == 3000


@pytest.mark.asyncio
async def test_omitted_stays_null(client, public):
    """No selection must leave the preset's own value untouched."""
    r = await _submit(client)
    assert r.status_code == 201
    from sqlalchemy import select

    from app.db import SessionLocal
    from app.models import Run

    async with SessionLocal() as s:
        row = (await s.execute(select(Run).where(Run.id == r.json()["id"]))).scalar_one()
        assert row.output_length is None


@pytest.mark.parametrize("bad", [499, 20001, 0, -100])
@pytest.mark.asyncio
async def test_out_of_range_rejected(client, public, bad):
    """Bounded by pydantic: a typo must not order a novel or starve the abstract."""
    r = await _submit(client, output_length=bad)
    assert r.status_code == 422, r.text


@pytest.mark.parametrize("edge", [500, 20000])
@pytest.mark.asyncio
async def test_bounds_are_inclusive(client, public, edge):
    r = await _submit(client, output_length=edge)
    assert r.status_code == 201, r.text


@pytest.mark.asyncio
async def test_ignored_outside_public_mode(client):
    """Like synthesis_model: accepted but not applied when the preset is boss."""
    r = await client.post(
        "/runs", json={"query": make_query(), "preset": "fast", "output_length": 3000}
    )
    assert r.status_code == 201
    from sqlalchemy import select

    from app.db import SessionLocal
    from app.models import Run

    async with SessionLocal() as s:
        row = (await s.execute(select(Run).where(Run.id == r.json()["id"]))).scalar_one()
        assert row.output_length is None


def test_override_targets_a_key_the_presets_actually_use():
    """The step that would silently no-op if the section name were wrong.

    Storing the value proves nothing on its own -- it has to land on the exact
    section+key the presets define, or caesar just ignores it and the answer
    stays unconstrained. Pinned against job_runner's real source and the real
    preset YAML, so renaming either side fails here instead of in production.
    """
    from pathlib import Path

    import yaml

    from app.job_runner import _RunState

    web_server = Path(__file__).resolve().parents[2]

    # 1. The preset defines the section and key we target.
    preset = yaml.safe_load((web_server / "config_preset" / "fast.yaml").read_text())
    assert "ArtifactSynthesizer" in preset
    assert "synthesis_max_length" in preset["ArtifactSynthesizer"]

    # 2. job_runner applies to that same section and key.
    src = (web_server / "api" / "app" / "job_runner.py").read_text()
    assert 'config.setdefault("ArtifactSynthesizer", {})["synthesis_max_length"]' in src
    assert "if state.output_length:" in src

    # 3. The state object carries it, defaulting to "leave the preset alone".
    assert _RunState("rid", preset_id="fast", output_length=3000).output_length == 3000
    assert _RunState("rid", preset_id="fast").output_length is None


@pytest.fixture
def spy_submit(monkeypatch):
    """Capture the kwargs handed to job_pool.submit.

    Asserting on the DB row is not enough: neither create nor retry rewrites
    output_length on restart, so a row keeps its value even if the value never
    reaches the pool -- which is exactly the code path that makes it take effect.
    """
    import app.routers.runs as runs_mod

    calls: list[dict] = []
    real = runs_mod.job_pool.submit

    async def recorder(*a, **kw):
        calls.append(kw)
        return await real(*a, **kw)

    monkeypatch.setattr(runs_mod.job_pool, "submit", recorder)
    return calls


@pytest.mark.asyncio
async def test_reaches_the_pool_on_create(client, public, spy_submit):
    r = await _submit(client, output_length=3000)
    assert r.status_code == 201, r.text
    assert spy_submit and spy_submit[-1].get("output_length") == 3000


@pytest.mark.asyncio
async def test_reaches_the_pool_on_restart(client, public, spy_submit):
    """A restart must reproduce the run, not silently fall back to the preset."""
    r = await _submit(client, output_length=1500)
    run_id = r.json()["id"]

    from sqlalchemy import select, update

    from app.db import SessionLocal
    from app.models import Run, RunStatus

    async with SessionLocal() as s:
        await s.execute(
            update(Run).where(Run.id == run_id).values(status=RunStatus.failed.value)
        )
        await s.commit()

    spy_submit.clear()
    retry = await client.post(
        f"/runs/{run_id}/retry", cookies=OWNER_COOKIE, json={"api_key": VALID_KEY}
    )
    assert retry.status_code == 200, retry.text
    assert spy_submit, "retry did not resubmit to the pool"
    assert spy_submit[-1].get("output_length") == 1500

    async with SessionLocal() as s:
        row = (await s.execute(select(Run).where(Run.id == run_id))).scalar_one()
        assert row.output_length == 1500
