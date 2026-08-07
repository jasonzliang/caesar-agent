"""POST /runs/{id}/retry — restart a failed run in place, resuming from its
checkpoint when one survived."""
from __future__ import annotations

import asyncio
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from .conftest import make_query

VALID_KEY = "sk-" + "y" * 24


async def _insert_failed_run(*, owner: str | None = None, preset: str = "fast") -> str:
    """A failed row, as the crash paths leave one: no key, error set."""
    from app.db import SessionLocal
    from app.models import Run, RunStatus

    run_id = uuid.uuid4().hex
    async with SessionLocal() as session:
        session.add(
            Run(
                id=run_id,
                query=make_query(),
                preset=preset,
                status=RunStatus.failed.value,
                mode="new",
                collection_name=f"web_{run_id}",
                owner_id=owner,
                created_at=datetime.now(timezone.utc),
                finished_at=datetime.now(timezone.utc),
                error_message="ConnectError: [Errno 111] Connection refused",
            )
        )
        await session.commit()
    return run_id


def _rome_dir(run_id: str) -> Path:
    from app.config import get_settings

    d = get_settings().runs_dir / run_id / "__rome__"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stored_key(run_id: str) -> str | None:
    from app.config import get_settings

    con = sqlite3.connect(get_settings().db_path)
    try:
        row = con.execute("SELECT run_api_key FROM runs WHERE id = ?", (run_id,)).fetchone()
    finally:
        con.close()
    return row[0] if row else None


async def _wait_terminal(client, run_id: str, **kw) -> dict:
    for _ in range(80):
        d = await client.get(f"/runs/{run_id}", **kw)
        body = d.json()
        if body["status"] in ("completed", "failed"):
            return body
        await asyncio.sleep(0.25)
    raise AssertionError(f"run {run_id} never reached a terminal status")


# --------------------------------------------------------------------------
# unarchive_checkpoint (pure)
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unarchive_restores_archived_checkpoint(client):
    from app.job_runner import unarchive_checkpoint

    run_id = uuid.uuid4().hex
    rome = _rome_dir(run_id)
    (rome / "agent_X.checkpoint.final.json").write_text('{"iteration": 7}')

    assert unarchive_checkpoint(rome.parent) is True
    assert (rome / "agent_X.checkpoint.json").read_text() == '{"iteration": 7}'
    assert not (rome / "agent_X.checkpoint.final.json").exists()


@pytest.mark.asyncio
async def test_unarchive_keeps_live_checkpoint_over_older_archive(client):
    """A live checkpoint is at least as fresh as the archive; never clobber it."""
    from app.job_runner import unarchive_checkpoint

    run_id = uuid.uuid4().hex
    rome = _rome_dir(run_id)
    (rome / "agent_X.checkpoint.json").write_text('{"iteration": 9}')
    (rome / "agent_X.checkpoint.final.json").write_text('{"iteration": 2}')

    assert unarchive_checkpoint(rome.parent) is True
    assert (rome / "agent_X.checkpoint.json").read_text() == '{"iteration": 9}'


@pytest.mark.asyncio
async def test_unarchive_reports_false_without_checkpoint(client):
    from app.job_runner import unarchive_checkpoint

    run_id = uuid.uuid4().hex
    rome = _rome_dir(run_id)
    # An empty checkpoint is not a usable one.
    (rome / "agent_X.checkpoint.final.json").write_text("")

    assert unarchive_checkpoint(rome.parent) is False
    assert unarchive_checkpoint(Path("/nonexistent/run/dir")) is False


# --------------------------------------------------------------------------
# endpoint
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_resumes_failed_run_from_checkpoint(client):
    run_id = await _insert_failed_run()
    (_rome_dir(run_id) / "agent_X.checkpoint.final.json").write_text('{"iteration": 7}')

    r = await client.post(f"/runs/{run_id}/retry", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == run_id
    assert body["status"] in ("queued", "running")
    assert body["error_message"] is None
    # Checkpoint is live again, so Caesar picks up instead of re-crawling.
    assert (_rome_dir(run_id) / "agent_X.checkpoint.json").exists()

    final = await _wait_terminal(client, run_id)
    assert final["status"] == "completed", final["error_message"]
    assert any(e["event"] == "resumed" for e in final["events"])


@pytest.mark.asyncio
async def test_retry_without_checkpoint_starts_over(client):
    run_id = await _insert_failed_run()

    r = await client.post(f"/runs/{run_id}/retry", json={})
    assert r.status_code == 200, r.text

    final = await _wait_terminal(client, run_id)
    assert final["status"] == "completed", final["error_message"]
    assert not any(e["event"] == "resumed" for e in final["events"])


@pytest.mark.asyncio
async def test_retry_regenerates_a_completed_run(client):
    """A completed run is restartable: its checkpoint has exploration already
    done, so this re-runs synthesis against the KB it already paid for."""
    r = await client.post("/runs", json={"query": make_query(), "preset": "fast"})
    run_id = r.json()["id"]
    first = await _wait_terminal(client, run_id)
    assert first["status"] == "completed"

    r = await client.post(f"/runs/{run_id}/retry", json={})
    assert r.status_code == 200, r.text
    assert r.json()["status"] in ("queued", "running")

    second = await _wait_terminal(client, run_id)
    assert second["status"] == "completed"
    # Same row, restarted in place rather than forked into a new one.
    assert second["id"] == run_id
    assert len(second["events"]) > len(first["events"])


@pytest.mark.asyncio
async def test_retry_takes_over_a_run_in_flight(client):
    """create_run awaits submit(), so the pool tracks the id by the time 201
    lands. Restarting then has to stop that attempt and wait for its worker to
    exit before starting another on the same directory."""
    from app.job_runner import job_pool

    r = await client.post("/runs", json={"query": make_query(), "preset": "fast"})
    assert r.status_code == 201
    run_id = r.json()["id"]
    assert job_pool.is_tracked(run_id)

    r = await client.post(f"/runs/{run_id}/retry", json={})
    assert r.status_code == 200, r.text
    assert r.json()["status"] in ("queued", "running")

    final = await _wait_terminal(client, run_id)
    assert final["status"] == "completed", final["error_message"]
    # One row, one surviving worker: the pool released the first attempt.
    assert not job_pool.is_tracked(run_id)


@pytest.mark.asyncio
async def test_request_stop_is_a_noop_for_unknown_runs(client):
    """Nothing to stop must not raise, so a restart always proceeds."""
    from app.job_runner import job_pool

    run_id = uuid.uuid4().hex
    await job_pool.request_stop(run_id)
    assert job_pool.has_live_worker(run_id) is False


@pytest.mark.asyncio
async def test_live_worker_is_tracked_outside_run_state(client):
    """The takeover gate must not be _states membership: a run whose takeover
    timed out is popped from _states while its Caesar thread works on, and
    trusting tracking there started a second agent on the same directory."""
    import threading

    from app.job_runner import job_pool

    run_id = uuid.uuid4().hex
    event = threading.Event()
    job_pool._worker_done[run_id] = event
    try:
        # Untracked, yet a worker is live: the takeover wait keys off this, not
        # off _states, so a restart still serialises behind the old thread.
        assert job_pool.is_tracked(run_id) is False
        assert job_pool.has_live_worker(run_id) is True
        event.set()
        assert job_pool.has_live_worker(run_id) is False
    finally:
        job_pool._worker_done.pop(run_id, None)


@pytest.mark.asyncio
async def test_retry_restarts_the_clock(client):
    """An explicit restart is a new attempt, so Elapsed counts from now rather
    than from the original start."""
    r = await client.post("/runs", json={"query": make_query(), "preset": "fast"})
    run_id = r.json()["id"]
    first = await _wait_terminal(client, run_id)
    assert first["started_at"] is not None

    r = await client.post(f"/runs/{run_id}/retry", json={})
    assert r.status_code == 200, r.text
    second = await _wait_terminal(client, run_id)
    assert second["started_at"] > first["started_at"]


@pytest.mark.asyncio
async def test_retry_missing_run_is_404(client):
    r = await client.post(f"/runs/{uuid.uuid4().hex}/retry", json={})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_retry_public_mode_requires_key_and_restores_it(client, monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "1")
    from app.config import get_settings

    get_settings.cache_clear()
    owner = uuid.uuid4().hex
    run_id = await _insert_failed_run(owner=owner)
    cookies = {"caesar_id": owner}

    # The stored key was purged when the run failed, so a restart must resend it.
    r = await client.post(f"/runs/{run_id}/retry", cookies=cookies, json={})
    assert r.status_code == 400
    assert "API key" in r.json()["detail"]

    # Another tenant cannot restart it, and learns nothing about its existence.
    r = await client.post(
        f"/runs/{run_id}/retry",
        cookies={"caesar_id": uuid.uuid4().hex},
        json={"api_key": VALID_KEY},
    )
    assert r.status_code == 404

    r = await client.post(f"/runs/{run_id}/retry", cookies=cookies, json={"api_key": VALID_KEY})
    assert r.status_code == 200, r.text
    assert _stored_key(run_id) == VALID_KEY

    final = await _wait_terminal(client, run_id, cookies=cookies)
    assert final["status"] == "completed", final["error_message"]
    # Delete-on-finish still applies to a restarted run.
    assert _stored_key(run_id) is None


@pytest.mark.asyncio
async def test_restarted_run_shows_live_cost_not_the_old_total(client):
    """A restarted run keeps the previous attempt's persisted cost until it
    finishes again. The listing must show the live figure meanwhile, or the page
    freezes at the old number for the whole new attempt."""
    from app.job_runner import job_pool
    from app.models import Run, RunStatus
    from app.routers.runs import _summary_from

    run = Run(
        id=uuid.uuid4().hex, query=make_query(), preset="fast",
        status=RunStatus.running.value, mode="new",
        created_at=datetime.now(timezone.utc), total_cost_usd=0.8417,
        graph_node_count=137,
    )
    state = type("S", (), {})()  # stand-in for _RunState's read surface
    state.live_cost_usd, state.live_graph_node_count = 0.8661, 140
    state.finished = __import__("asyncio").Event()
    job_pool._states[run.id] = state
    try:
        s = _summary_from(run)
        assert s.total_cost_usd == 0.8661, "stale persisted cost shadowed the live one"
        assert s.graph_node_count == 140
    finally:
        job_pool._states.pop(run.id, None)


@pytest.mark.asyncio
async def test_restart_banks_prior_elapsed(client):
    """A restart resets started_at so the clock matches the new attempt's
    progress counters. Without banking, every restart would silently discard the
    previous attempt's runtime from the displayed elapsed figure."""
    from app.db import SessionLocal
    from app.models import Run

    r = await client.post("/runs", json={"query": make_query(), "preset": "fast"})
    run_id = r.json()["id"]
    first = await _wait_terminal(client, run_id)
    assert first["elapsed_prior_s"] == 0.0

    r = await client.post(f"/runs/{run_id}/retry", json={})
    assert r.status_code == 200, r.text
    banked = r.json()["elapsed_prior_s"]
    assert banked > 0, "the finished attempt's runtime was thrown away"

    await _wait_terminal(client, run_id)
    # A second restart accumulates rather than replacing.
    r = await client.post(f"/runs/{run_id}/retry", json={})
    assert r.status_code == 200, r.text
    assert r.json()["elapsed_prior_s"] > banked
    await _wait_terminal(client, run_id)

    async with SessionLocal() as s:
        assert (await s.get(Run, run_id)).elapsed_prior_s > banked


@pytest.mark.asyncio
async def test_restart_is_not_refused_while_the_old_worker_winds_down(client):
    """The failure this replaced: the endpoint cancelled the attempt, waited 20s
    for a thread that only checks the flag every 45-90s, then returned 409. The
    cancel still landed, so the run died with nothing replacing it. A restart
    must always be accepted; the new attempt serialises behind the old thread."""
    import threading

    from app.job_runner import job_pool

    run_id = await _insert_failed_run()
    winding_down = threading.Event()  # deliberately not set: worker still alive
    job_pool._worker_done[run_id] = winding_down
    try:
        r = await client.post(f"/runs/{run_id}/retry", json={})
        assert r.status_code == 200, f"restart refused while a worker wound down: {r.text}"
    finally:
        winding_down.set()  # let the new attempt past its takeover wait
        job_pool._worker_done.pop(run_id, None)

    final = await _wait_terminal(client, run_id)
    assert final["status"] == "completed", final["error_message"]


@pytest.mark.asyncio
async def test_takeover_timeout_fails_cleanly_instead_of_ghosting(client, monkeypatch):
    """If the previous attempt never stops, the restart must end in a defined
    state. Raising outside the try left the task dead with the row still marked
    running and the concurrency slot held: a ghost, which is the bug this whole
    path exists to prevent."""
    import threading

    from app import job_runner
    from app.job_runner import job_pool

    monkeypatch.setattr(job_runner, "TAKEOVER_WAIT_S", 0.25)
    run_id = await _insert_failed_run()
    never_stops = threading.Event()
    job_pool._worker_done[run_id] = never_stops
    try:
        r = await client.post(f"/runs/{run_id}/retry", json={})
        assert r.status_code == 200

        final = await _wait_terminal(client, run_id)
        assert final["status"] == "failed"
        assert "still running" in (final["error_message"] or "")
        # Slot released, so the id is reusable rather than wedged forever.
        assert not job_pool.is_tracked(run_id)
    finally:
        never_stops.set()
        job_pool._worker_done.pop(run_id, None)


@pytest.mark.asyncio
async def test_takeover_wait_is_announced_as_its_own_event(client):
    """The UI renders this as a state, because every stat beside it stays frozen
    on the old attempt until the handover completes. A `log` line would leave the
    page looking hung for the ~minute a synthesis step takes to notice the flag."""
    import threading

    from app.job_runner import job_pool

    run_id = await _insert_failed_run()
    winding_down = threading.Event()
    job_pool._worker_done[run_id] = winding_down
    try:
        r = await client.post(f"/runs/{run_id}/retry", json={})
        assert r.status_code == 200
        for _ in range(40):
            d = await client.get(f"/runs/{run_id}")
            if any(e["event"] == "takeover_wait" for e in d.json()["events"]):
                break
            await asyncio.sleep(0.25)
        else:
            raise AssertionError("no takeover_wait event was emitted")
    finally:
        winding_down.set()
        job_pool._worker_done.pop(run_id, None)

    await _wait_terminal(client, run_id)
