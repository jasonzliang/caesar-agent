"""Completion must finish the real watchdog's final log pass before `done`.

Only worker execution and persistence are controlled. Lifecycle, log parsing,
secret scrubbing, deduplication, and cancellation cleanup remain real.
"""
from __future__ import annotations

import asyncio
import os
import socket
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def lifecycle(monkeypatch, tmp_path):
    # These API tests live outside test/conftest.py's credential/network guards.
    for name in tuple(os.environ):
        if name.upper().endswith(("_API_KEY", "_API_TOKEN")) or name in {
            "ANTHROPIC_AUTH_TOKEN", "GOOGLE_APPLICATION_CREDENTIALS",
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
        }:
            monkeypatch.delenv(name)

    def no_network(*args, **kwargs):
        raise AssertionError("Lifecycle tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)

    # _run lazily imports only this exception. Do not initialize provider SDKs.
    llm = ModuleType("rome.llm_handler")
    llm.FatalLLMError = type("FatalLLMError", (BaseException,), {})
    monkeypatch.setitem(sys.modules, "rome.llm_handler", llm)

    # The API's autouse isolated_data_dir fixture runs before this import.
    from app import job_runner as jobs

    settings = jobs.get_settings()
    assert settings.caesar_web_data_dir == tmp_path / "data"
    assert settings.caesar_dry_run is True
    pool = jobs.JobPool()
    state = jobs._RunState("completion-test", "fast", api_key="sk-test-ABCDEFGHIJKLMNOP")
    pool._states[state.run_id] = state
    context = SimpleNamespace(
        jobs=jobs, pool=pool, state=state, events=[], timeline=[], before_emit=None,
        seeded=asyncio.Event(),
        repo=settings.runs_dir / state.run_id,
    )
    context.log = context.repo / "__rome__" / "agent_Caesar.console.log"

    async def emit(state, event, **payload):
        if context.before_emit is not None:
            await context.before_emit(event, payload)
        context.events.append((event, payload))
        context.timeline.append(("event", event))
        state.last_activity_mono = jobs.time.monotonic()

    async def update_status(run_id, *, status, **fields):
        context.timeline.append(("status", status.value))

    original_seed = jobs._seed_watchdog_from_log

    def seed(repo):
        result = original_seed(repo)
        context.seeded.set()
        return result

    monkeypatch.setattr(pool, "_emit", emit)
    monkeypatch.setattr(pool, "_update_status", update_status)
    monkeypatch.setattr(jobs, "_seed_watchdog_from_log", seed)
    return context


def append_log(context, text):
    context.log.parent.mkdir(parents=True, exist_ok=True)
    with context.log.open("a", encoding="utf-8") as handle:
        handle.write(text)


def artifact():
    return {"artifact": "A successful text artifact", "num_drafts": 1,
            "metadata": {"total_cost_usd": 0.0}}


async def run(context):
    await context.pool._run(context.state.run_id, "controlled query", "fast", context.state)


@pytest.mark.asyncio
async def test_late_image_failure_precedes_done_without_replaying_history(lifecycle, monkeypatch):
    """A worker finishing between ticks must not lose its optional-image warning."""
    c = lifecycle
    append_log(c, "[IMAGE] Image generation failed: previous attempt failed\n")
    line = f"[IMAGE] Image generation failed: Incorrect API key provided: {c.state.api_key}\n"

    async def worker(run_id, query, preset, repo, state):
        await c.seeded.wait()
        append_log(c, line + line)
        return artifact()

    monkeypatch.setattr(c.pool, "_dry_run", worker)
    await asyncio.wait_for(run(c), timeout=5)

    assert [event for event, _ in c.events] == ["log", "image_failed", "done"]
    warning = c.events[1][1]["message"]
    assert warning == "Incorrect API key provided: [REDACTED]"
    assert c.state.api_key not in warning
    assert c.timeline.index(("event", "image_failed")) < c.timeline.index(("status", "completed"))
    assert c.state.finished.is_set()
    assert c.state.run_id not in c.pool._states


@pytest.mark.asyncio
async def test_completion_waits_for_inflight_emit_then_reads_later_log_bytes(lifecycle, monkeypatch):
    """Finishing during an emit needs another pass, not cancellation or replay."""
    c = lifecycle
    emitting = asyncio.Event()
    release = asyncio.Event()
    worker_returning = asyncio.Event()
    first = "[IMAGE] Image generation failed: first reason\n"

    async def delay_first_warning(event, payload):
        if event == "image_failed" and payload["message"] == "first reason":
            emitting.set()
            await release.wait()

    c.before_emit = delay_first_warning

    async def worker(run_id, query, preset, repo, state):
        await c.seeded.wait()
        append_log(c, first)
        await emitting.wait()
        append_log(c, first + "[IMAGE] Image embedding failed: second reason\n")
        worker_returning.set()
        return artifact()

    monkeypatch.setattr(c.pool, "_dry_run", worker)
    task = asyncio.create_task(run(c))
    try:
        await asyncio.wait_for(worker_returning.wait(), timeout=5)
        await asyncio.sleep(0)  # Let _run react to the returned artifact.
        assert ("status", "completed") not in c.timeline
        release.set()
        await asyncio.wait_for(task, timeout=5)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    warnings = [payload["message"] for event, payload in c.events if event == "image_failed"]
    assert warnings == ["first reason", "second reason"]
    assert c.events[-1][0] == "done"
    assert c.timeline.index(("event", "image_failed")) < c.timeline.index(("status", "completed"))


@pytest.mark.asyncio
async def test_cancellation_during_final_emit_does_not_complete_job(lifecycle, monkeypatch):
    c = lifecycle
    emitting = asyncio.Event()
    hold = asyncio.Event()

    async def delay_warning(event, payload):
        if event == "image_failed":
            emitting.set()
            await hold.wait()

    c.before_emit = delay_warning

    async def worker(run_id, query, preset, repo, state):
        await c.seeded.wait()
        append_log(c, "[IMAGE] Image generation failed: optional illustration refused\n")
        return artifact()

    monkeypatch.setattr(c.pool, "_dry_run", worker)
    task = asyncio.create_task(run(c))
    try:
        await asyncio.wait_for(emitting.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert ("status", "completed") not in c.timeline
    assert not any(event == "done" for event, _ in c.events)
    assert c.state.finished.is_set()
    assert c.state.run_id not in c.pool._states


@pytest.mark.asyncio
async def test_worker_failure_keeps_failure_cleanup_and_secret_scrubbing(lifecycle, monkeypatch):
    c = lifecycle
    checkpoint = c.repo / "__rome__" / "agent_Caesar.checkpoint.json"

    async def worker(run_id, query, preset, repo, state):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text("{}", encoding="utf-8")
        raise RuntimeError(f"Worker failed with key {state.api_key}")

    monkeypatch.setattr(c.pool, "_dry_run", worker)
    await asyncio.wait_for(run(c), timeout=5)

    assert [item for kind, item in c.timeline if kind == "status"] == ["running", "failed"]
    assert [event for event, _ in c.events] == ["log", "error"]
    assert c.state.api_key not in str(c.events)
    assert "[REDACTED]" in c.events[-1][1]["message"]
    assert not checkpoint.exists()
    assert checkpoint.with_name("agent_Caesar.checkpoint.final.json").exists()
    assert c.state.finished.is_set()
    assert c.state.run_id not in c.pool._states
