"""Backend smoke tests.

Drive these with: cd web_server/api && pytest -q
All tests run with CAESAR_DRY_RUN=1 (see conftest), so no LLM calls happen
and a full run finishes in ~5 seconds.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from .conftest import make_query

pytestmark = pytest.mark.asyncio


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


async def test_presets(client):
    r = await client.get("/presets")
    assert r.status_code == 200
    body = r.json()
    ids = [p["id"] for p in body]
    assert ids == ["fast", "normal", "deeper", "deepest", "deeper_arxiv", "deepest_arxiv"]
    required = {"id", "label", "description", "estimated_cost_usd", "estimated_time_min"}
    for p in body:
        assert required <= p.keys()


async def test_unknown_preset_rejected(client):
    r = await client.post("/runs", json={"query": make_query(), "preset": "ultra"})
    assert r.status_code == 400


async def test_init_db_migrates_legacy_deep_preset():
    import sqlite3

    from app.config import get_settings
    from app.db import init_db

    settings = get_settings()
    await init_db()

    conn = sqlite3.connect(settings.db_path)
    try:
        conn.execute(
            "INSERT INTO runs (id, query, preset, status, created_at, mode) "
            "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?)",
            ("legacy-deep", "query", "deep", "completed", "new"),
        )
        conn.commit()
    finally:
        conn.close()

    await init_db()

    conn = sqlite3.connect(settings.db_path)
    try:
        row = conn.execute(
            "SELECT preset FROM runs WHERE id = ?",
            ("legacy-deep",),
        ).fetchone()
    finally:
        conn.close()

    assert row == ("deeper",)


async def test_init_db_renames_legacy_total_iterations_column():
    import sqlite3

    from app.config import get_settings
    from app.db import init_db

    settings = get_settings()
    settings.caesar_web_data_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(settings.db_path)
    try:
        conn.execute(
            """
            CREATE TABLE runs (
                id VARCHAR(36) PRIMARY KEY,
                query TEXT NOT NULL,
                preset VARCHAR(32) NOT NULL,
                status VARCHAR(16) NOT NULL,
                parent_run_id VARCHAR(36),
                mode VARCHAR(16) NOT NULL DEFAULT 'new',
                collection_name VARCHAR(128),
                repository TEXT,
                agent_name VARCHAR(64),
                created_at DATETIME NOT NULL,
                started_at DATETIME,
                finished_at DATETIME,
                total_cost_usd FLOAT,
                total_iterations INTEGER,
                error_message TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO runs (id, query, preset, status, created_at, total_iterations) "
            "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?)",
            ("legacy-count", "query", "fast", "completed", 163),
        )
        conn.commit()
    finally:
        conn.close()

    await init_db()

    conn = sqlite3.connect(settings.db_path)
    try:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(runs)").fetchall()
        }
        value = conn.execute(
            "SELECT graph_node_count FROM runs WHERE id = ?",
            ("legacy-count",),
        ).fetchone()
    finally:
        conn.close()

    assert "total_iterations" not in columns
    assert "graph_node_count" in columns
    assert value == (163,)


async def test_dry_run_lifecycle_completes(client):
    """End-to-end happy path: submit, wait, fetch run, fetch synthesis."""
    r = await client.post("/runs", json={"query": make_query(), "preset": "fast"})
    assert r.status_code == 201, r.text
    run_id = r.json()["id"]

    # Poll until terminal AND the terminal event has landed. The status update
    # and the `done` event are separate commits (_run does _update_status then
    # _emit), so under load a run reads as completed a beat before its `done`
    # row is visible -- which failed this test roughly one run in three.
    #
    # Waiting for both is the honest assertion: nothing in the app requires
    # them to be atomic. The UI decides completion from run.status
    # (LiveProgress isTerminal), and SSE clients receive `done` from the
    # in-memory queue rather than this table. Asserting a particular
    # interleaving would be asserting a guarantee the design never made.
    final = None
    for _ in range(40):  # up to ~10s
        await asyncio.sleep(0.25)
        body = (await client.get(f"/runs/{run_id}")).json()
        # A failed run emits `error`, not `done`; break so the status assertion
        # below reports the real failure instead of timing out.
        if body["status"] == "failed" or (
            body["status"] == "completed"
            and any(e["event"] == "done" for e in body["events"])
        ):
            final = body
            break
    assert final is not None, "run did not finish (or emit `done`) in time"
    assert final["status"] == "completed", final
    assert final["finished_at"] is not None
    assert any(e["event"] == "iteration" for e in final["events"])

    rs = await client.get(f"/runs/{run_id}/synthesis", params={"draft": "latest"})
    assert rs.status_code == 200
    body = rs.json()
    assert body["abstract"]
    assert body["artifact"]
    assert len(body["sources"]) >= 1


async def test_concurrent_runs_are_isolated(client):
    """3 concurrent dry runs each get their own row, repository, and finish independently."""
    posts = await asyncio.gather(*[
        client.post("/runs", json={"query": make_query(), "preset": "fast"})
        for _ in range(3)
    ])
    ids = [r.json()["id"] for r in posts]
    assert len(set(ids)) == 3

    # Wait for all
    deadline = 12.0
    elapsed = 0.0
    while elapsed < deadline:
        await asyncio.sleep(0.4)
        elapsed += 0.4
        statuses = []
        for run_id in ids:
            rg = await client.get(f"/runs/{run_id}")
            statuses.append(rg.json()["status"])
        if all(s in ("completed", "failed") for s in statuses):
            break
    assert all(s == "completed" for s in statuses), statuses


async def test_unknown_run_404(client):
    r = await client.get("/runs/does-not-exist")
    assert r.status_code == 404


async def test_graph_source_for_refine_chains():
    from app.models import Run, RunMode
    from app.routers.runs import _graph_run_id_for

    runs = {
        "graph-root": Run(id="graph-root", mode=RunMode.new.value),
        "graph-explore": Run(
            id="graph-explore",
            mode=RunMode.explore.value,
            parent_run_id="graph-root",
        ),
        "graph-refine-1": Run(
            id="graph-refine-1",
            mode=RunMode.refine.value,
            parent_run_id="graph-explore",
        ),
        "graph-refine-2": Run(
            id="graph-refine-2",
            mode=RunMode.refine.value,
            parent_run_id="graph-refine-1",
        ),
    }

    class FakeSession:
        async def get(self, _model, run_id):
            return runs.get(run_id)

    session = FakeSession()
    assert await _graph_run_id_for(runs["graph-explore"], session) == "graph-explore"
    assert await _graph_run_id_for(runs["graph-refine-1"], session) == "graph-explore"
    assert await _graph_run_id_for(runs["graph-refine-2"], session) == "graph-explore"


async def test_refine_run_includes_graph_progress_total(client):
    from app.db import SessionLocal
    from app.models import Run, RunEvent, RunMode, RunStatus

    async with SessionLocal() as session:
        session.add_all([
            Run(
                id="graph-total-parent",
                query="parent",
                preset="fast",
                status=RunStatus.completed.value,
                mode=RunMode.new.value,
                graph_node_count=163,
            ),
            Run(
                id="graph-total-refine",
                query="child",
                preset="fast",
                status=RunStatus.running.value,
                parent_run_id="graph-total-parent",
                mode=RunMode.refine.value,
            ),
            RunEvent(
                run_id="graph-total-parent",
                event="iteration",
                payload=json.dumps({"phase": "explore", "n": 240, "total": 240}),
            ),
        ])
        await session.commit()

    response = await client.get("/runs/graph-total-refine")
    assert response.status_code == 200
    body = response.json()
    assert body["graph_node_count"] == 163
    assert body["graph_run_id"] == "graph-total-parent"
    assert body["graph_progress_total"] == 240
    assert body["graph_progress_phase"] == "explore"

    runs = (await client.get("/runs", params={"limit": 10})).json()
    child_summary = next(r for r in runs if r["id"] == "graph-total-refine")
    assert child_summary["graph_node_count"] == 163


async def test_refine_run_without_parent_progress_events_only_inherits_graph_size(client):
    from app.db import SessionLocal
    from app.models import Run, RunMode, RunStatus

    async with SessionLocal() as session:
        session.add_all([
            Run(
                id="graph-size-parent",
                query="parent",
                preset="fast",
                status=RunStatus.completed.value,
                mode=RunMode.new.value,
                graph_node_count=163,
            ),
            Run(
                id="graph-size-refine",
                query="child",
                preset="fast",
                status=RunStatus.running.value,
                parent_run_id="graph-size-parent",
                mode=RunMode.refine.value,
            ),
        ])
        await session.commit()

    response = await client.get("/runs/graph-size-refine")
    assert response.status_code == 200
    body = response.json()
    assert body["graph_node_count"] == 163
    assert body["graph_run_id"] == "graph-size-parent"
    assert body["graph_progress_total"] is None
    assert body["graph_progress_phase"] is None


async def test_run_summaries_include_merged_followup_query(client):
    from app.config import get_settings
    from app.db import SessionLocal
    from app.models import Run, RunMode, RunStatus

    settings = get_settings()
    repo = settings.runs_dir / "summary-merged"
    (repo / "__rome__").mkdir(parents=True, exist_ok=True)
    (repo / "__rome__" / "merged_query.txt").write_text(
        "Full merged follow-up query",
        encoding="utf-8",
    )

    async with SessionLocal() as session:
        session.add_all([
            Run(
                id="summary-parent",
                query="parent",
                preset="fast",
                status=RunStatus.completed.value,
                mode=RunMode.new.value,
            ),
            Run(
                id="summary-merged",
                query="terse follow-up",
                preset="fast",
                status=RunStatus.completed.value,
                parent_run_id="summary-parent",
                mode=RunMode.refine.value,
                repository=str(repo),
            ),
        ])
        await session.commit()

    runs = (await client.get("/runs", params={"limit": 10})).json()
    child_summary = next(r for r in runs if r["id"] == "summary-merged")
    assert child_summary["query"] == "terse follow-up"
    assert child_summary["merged_query"] == "Full merged follow-up query"

    detail = (await client.get("/runs/summary-merged")).json()
    assert detail["merged_query"] == "Full merged follow-up query"


async def test_delete_run_cleans_db_and_disk(client, tmp_path, monkeypatch):
    """DELETE removes the row, artifact directory, and owned Chroma collection."""
    # Submit + complete a dry run.
    r = await client.post("/runs", json={"query": make_query(), "preset": "fast"})
    run_id = r.json()["id"]
    for _ in range(40):
        await asyncio.sleep(0.25)
        rg = await client.get(f"/runs/{run_id}")
        if rg.json()["status"] in ("completed", "failed"):
            break
    detail = rg.json()
    assert detail["status"] == "completed"

    # We know runs_dir/<id>/ should exist on disk.
    from app.config import get_settings  # noqa: WPS433
    repo_path = get_settings().runs_dir / run_id
    assert repo_path.exists()

    from app.routers import runs as runs_router

    deleted_collections = []
    monkeypatch.setattr(
        runs_router,
        "_delete_chroma_collection",
        lambda collection_name: deleted_collections.append(collection_name) or True,
    )
    monkeypatch.setattr(
        runs_router,
        "_chroma_collection_names_with_prefix",
        lambda prefixes: {f"mem0_{run_id}_agent_TestAgent"},
    )

    # DELETE
    rd = await client.delete(f"/runs/{run_id}")
    assert rd.status_code == 204
    assert deleted_collections == [
        f"mem0_{run_id}_agent_TestAgent",
        f"web_{run_id}",
    ]

    # Row gone
    rg2 = await client.get(f"/runs/{run_id}")
    assert rg2.status_code == 404
    # Disk gone
    assert not repo_path.exists()

    # Idempotent: deleting again still returns 204
    rd2 = await client.delete(f"/runs/{run_id}")
    assert rd2.status_code == 204


async def test_delete_parent_with_followups_is_rejected(client, monkeypatch):
    from app.db import SessionLocal
    from app.models import Run, RunMode, RunStatus
    from app.routers import runs as runs_router

    async with SessionLocal() as session:
        session.add_all([
            Run(
                id="delete-parent",
                query="parent",
                preset="fast",
                status=RunStatus.completed.value,
                mode=RunMode.new.value,
                collection_name="web_delete-parent",
            ),
            Run(
                id="delete-child",
                query="child",
                preset="fast",
                status=RunStatus.completed.value,
                parent_run_id="delete-parent",
                mode=RunMode.refine.value,
                collection_name="web_delete-parent",
            ),
        ])
        await session.commit()

    deleted_collections = []
    monkeypatch.setattr(
        runs_router,
        "_delete_chroma_collection",
        lambda collection_name: deleted_collections.append(collection_name) or True,
    )
    monkeypatch.setattr(
        runs_router,
        "_chroma_collection_names_with_prefix",
        lambda prefixes: {
            f"{prefix}agent_TestAgent"
            for prefix in prefixes
        },
    )

    blocked = await client.delete("/runs/delete-parent")
    assert blocked.status_code == 409
    assert "follow-up" in blocked.json()["detail"]
    assert deleted_collections == []

    parent = await client.get("/runs/delete-parent")
    assert parent.status_code == 200

    child_deleted = await client.delete("/runs/delete-child")
    assert child_deleted.status_code == 204
    assert deleted_collections == ["mem0_delete-child_agent_TestAgent"]

    parent_deleted = await client.delete("/runs/delete-parent")
    assert parent_deleted.status_code == 204
    assert deleted_collections == [
        "mem0_delete-child_agent_TestAgent",
        "mem0_delete-parent_agent_TestAgent",
        "web_delete-parent",
    ]


async def test_wipe_all_still_deletes_parent_child_runs(client, monkeypatch):
    from app.db import SessionLocal
    from app.models import Run, RunMode, RunStatus
    from app.routers import runs as runs_router

    async with SessionLocal() as session:
        session.add_all([
            Run(
                id="wipe-parent",
                query="parent",
                preset="fast",
                status=RunStatus.completed.value,
                mode=RunMode.new.value,
                collection_name="web_wipe-parent",
            ),
            Run(
                id="wipe-child",
                query="child",
                preset="fast",
                status=RunStatus.completed.value,
                parent_run_id="wipe-parent",
                mode=RunMode.refine.value,
                collection_name="web_wipe-parent",
            ),
        ])
        await session.commit()

    deleted_collections = []
    monkeypatch.setattr(
        runs_router,
        "_delete_chroma_collection",
        lambda collection_name: deleted_collections.append(collection_name) or True,
    )
    monkeypatch.setattr(
        runs_router,
        "_chroma_collection_names_with_prefix",
        lambda prefixes: {
            f"{prefix}agent_TestAgent"
            for prefix in prefixes
        },
    )

    wiped = await client.delete("/runs", params={"confirm": "yes"})
    assert wiped.status_code == 200
    assert wiped.json()["deleted"] == 2
    assert wiped.json()["chroma_removed"] == 3
    assert wiped.json()["chroma_failed"] == 0
    assert deleted_collections == [
        "mem0_wipe-child_agent_TestAgent",
        "mem0_wipe-parent_agent_TestAgent",
        "web_wipe-parent",
    ]

    parent = await client.get("/runs/wipe-parent")
    child = await client.get("/runs/wipe-child")
    assert parent.status_code == 404
    assert child.status_code == 404


async def test_orphan_recovery_on_startup():
    """Pre-seed a 'running' row, restart the app, confirm it's marked failed."""
    # Don't use the `client` fixture here — we want full control of lifespan.
    from app.db import SessionLocal, init_db
    from app.main import app
    from app.models import Run, RunStatus

    # Boot 1: create a fake orphan row.
    async with app.router.lifespan_context(app):
        await init_db()
        async with SessionLocal() as session:
            session.add(Run(
                id="orphan-1", query="x", preset="fast",
                status=RunStatus.running.value,
            ))
            await session.commit()

    # Boot 2: lifespan should mark orphan as failed.
    async with app.router.lifespan_context(app):
        async with SessionLocal() as session:
            run = await session.get(Run, "orphan-1")
            assert run is not None
            assert run.status == RunStatus.failed.value
            assert run.error_message


async def test_archive_checkpoint_roundtrip(tmp_path):
    """archive_checkpoint renames the live file to *.checkpoint.final.json;
    has_checkpoint reports False afterwards (so it won't be re-resumed)."""
    from app.job_runner import archive_checkpoint, has_checkpoint

    rome = tmp_path / "__rome__"
    rome.mkdir()
    live = rome / "agent_CaesarExplorer.checkpoint.json"
    live.write_text("{}")

    assert has_checkpoint(tmp_path) is True
    archive_checkpoint(tmp_path)
    assert has_checkpoint(tmp_path) is False
    assert (rome / "agent_CaesarExplorer.checkpoint.final.json").exists()

    # Idempotent: archive again is a no-op (no live files left).
    archive_checkpoint(tmp_path)
    assert has_checkpoint(tmp_path) is False


async def test_parent_query_prefers_effective_query_cache():
    """Chained follow-ups should inherit what Caesar actually answered."""
    from app.config import get_settings
    from app.db import SessionLocal, init_db
    from app.job_runner import _fetch_parent_query_sync
    from app.models import Run, RunStatus

    settings = get_settings()
    repo = settings.runs_dir / "parent-effective"
    (repo / "__rome__").mkdir(parents=True, exist_ok=True)
    (repo / "__rome__" / "merged_query.txt").write_text(
        "Effective parent query",
        encoding="utf-8",
    )

    await init_db()
    async with SessionLocal() as session:
        session.add(Run(
            id="parent-effective",
            query="raw follow-up text",
            preset="fast",
            status=RunStatus.completed.value,
            repository=str(repo),
        ))
        await session.commit()

    assert _fetch_parent_query_sync("parent-effective") == "Effective parent query"
    (repo / "__rome__" / "merged_query.txt").write_text("", encoding="utf-8")
    assert _fetch_parent_query_sync("parent-effective") == "raw follow-up text"


async def test_merge_followup_query_cache_hit(tmp_path, monkeypatch):
    """A non-empty cache file short-circuits the LLM call."""
    from app.job_runner import _merge_followup_query

    cache = tmp_path / "merged_query.txt"
    cache.write_text("cached merged query", encoding="utf-8")

    def _explode(**_kw):
        raise AssertionError("litellm.completion must not run on cache hit")
    monkeypatch.setattr("litellm.completion", _explode)

    assert _merge_followup_query("parent", "follow-up", cache_path=cache) == (
        "cached merged query"
    )


async def test_merge_followup_query_llm_success_writes_cache(tmp_path, monkeypatch):
    """A successful LLM merge returns the stripped content and writes the
    cache file (creating the parent dir if missing)."""
    from app.job_runner import _merge_followup_query

    class _Msg:
        content = "  merged result\n"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    monkeypatch.setattr("litellm.completion", lambda **_kw: _Resp())

    cache = tmp_path / "nested" / "merged_query.txt"
    out = _merge_followup_query("parent", "follow-up", cache_path=cache)
    assert out == "merged result"
    assert cache.read_text(encoding="utf-8") == "merged result"


async def test_merge_followup_query_falls_back_on_llm_failure(monkeypatch):
    """An LLM exception falls back to the flat-sentence format that won't
    poison Caesar's downstream prompts."""
    from app.job_runner import _merge_followup_query

    def _raise(**_kw):
        raise RuntimeError("simulated LLM outage")
    monkeypatch.setattr("litellm.completion", _raise)

    assert _merge_followup_query("Parent: X vs Y", "go deeper") == (
        "go deeper (in the context of: Parent: X vs Y)"
    )


async def test_refine_agent_startup_skips_exploration_only_work(tmp_path):
    from app.job_runner import _configure_refine_agent_startup

    parent_artifact = tmp_path / "parent.synthesis.txt"
    parent_artifact.write_text("Parent answer", encoding="utf-8")
    config = {
        "CaesarAgent": {
            "starting_query": "Effective follow-up query",
            "max_iterations": 240,
            "additional_starting_queries": 9,
            "adapt_role": True,
        },
    }

    _configure_refine_agent_startup(config, tmp_path / "run", parent_artifact)

    agent_cfg = config["CaesarAgent"]
    assert agent_cfg["starting_query"] == "Effective follow-up query"
    assert agent_cfg["max_iterations"] == 0
    assert agent_cfg["additional_starting_queries"] == 0
    assert agent_cfg["adapt_role"] is False
    assert agent_cfg["starting_url"] == parent_artifact.resolve().as_uri()

    fallback_config = {"CaesarAgent": {"starting_query": "Effective follow-up query"}}
    fallback_repo = tmp_path / "fallback-run"
    _configure_refine_agent_startup(fallback_config, fallback_repo, None)

    fallback_path = fallback_repo / "__rome__" / "refine_start.html"
    assert fallback_path.exists()
    assert fallback_config["CaesarAgent"]["starting_url"] == fallback_path.resolve().as_uri()


async def test_stale_terminal_checkpoint_is_archived_on_startup():
    """A completed run with a stale live checkpoint should get archived on
    startup so it isn't accidentally resumed."""
    from app.config import get_settings
    from app.db import SessionLocal, init_db
    from app.job_runner import has_checkpoint
    from app.main import app
    from app.models import Run, RunStatus

    settings = get_settings()

    async with app.router.lifespan_context(app):
        await init_db()
        async with SessionLocal() as session:
            session.add(Run(
                id="stale-cp-1", query="x", preset="fast",
                status=RunStatus.completed.value,
            ))
            await session.commit()

    repo = settings.runs_dir / "stale-cp-1"
    (repo / "__rome__").mkdir(parents=True, exist_ok=True)
    (repo / "__rome__" / "agent_CaesarExplorer.checkpoint.json").write_text("{}")
    assert has_checkpoint(repo)  # sanity

    # Second boot's lifespan should archive the stale checkpoint.
    async with app.router.lifespan_context(app):
        assert not has_checkpoint(repo)
        archived = repo / "__rome__" / "agent_CaesarExplorer.checkpoint.final.json"
        assert archived.exists()


async def test_fatal_llm_error_marks_run_failed_with_message(client, monkeypatch):
    """Regression test for the BaseException + line-505 explicit-catch fix.

    FatalLLMError (BaseException-derived, raised by kb_client when openai
    returns insufficient_quota) must surface as the run's error_message, not
    be swallowed into a misleading "No synthesis artifacts created" or
    propagate as an unhandled asyncio task exception.

    Asserts that:
    1. The run is DB-marked status=failed (not completed, not stuck running).
    2. error_message contains the FatalLLMError text the user needs to see.
    3. An "error" SSE event with the same text is persisted.

    Without the line-505 `except (Exception, FatalLLMError)` carve-out, the
    BaseException-derived exception would bypass the run-failure handler
    entirely and the run would never reach a terminal status.
    """
    from app.config import ensure_caesar_on_path
    ensure_caesar_on_path()
    from rome.llm_handler import FatalLLMError

    async def _explode(self, *args, **kwargs):
        raise FatalLLMError("Insufficient quota / billing issue: simulated")

    monkeypatch.setattr("app.job_runner.JobPool._dry_run", _explode)

    r = await client.post("/runs", json={"query": make_query(), "preset": "fast"})
    assert r.status_code == 201, r.text
    run_id = r.json()["id"]

    # _dry_run is monkeypatched to raise immediately; the run should hit a
    # terminal status within ~100ms. Tight poll keeps the test under 0.5s.
    final = None
    for _ in range(40):  # up to ~2s
        await asyncio.sleep(0.05)
        rg = await client.get(f"/runs/{run_id}")
        if rg.json()["status"] in ("completed", "failed"):
            final = rg.json()
            break
    assert final is not None, "run did not reach terminal status"
    assert final["status"] == "failed", final
    assert "Insufficient quota" in (final.get("error_message") or ""), final
    error_events = [e for e in final.get("events", []) if e["event"] == "error"]
    assert error_events, final.get("events")
    assert "Insufficient quota" in (error_events[0]["payload"] or {}).get("message", "")
