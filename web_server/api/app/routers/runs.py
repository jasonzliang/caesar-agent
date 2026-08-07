"""POST /runs, GET /runs, GET /runs/{id} — submit a query and read run state."""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings, is_valid_synthesis_model, preset_by_id, preset_llm_model
from ..db import get_session
from ..deps import current_owner, get_owned_run, is_admin
from ..job_runner import job_pool, unarchive_checkpoint
from ..models import Run, RunEvent, RunMode, RunStatus
from ..schemas import RunCreate, RunDetail, RunEventOut, RunRetry, RunSummary

router = APIRouter(prefix="/runs", tags=["runs"])
logger = logging.getLogger("caesar.web.runs")


def _run_collection_name(run: Run) -> str | None:
    collection_name = run.collection_name
    if collection_name is None and run.mode == RunMode.new.value:
        collection_name = f"web_{run.id}"
    if collection_name and collection_name.startswith("web_"):
        return collection_name
    return None


def _run_memory_collection_prefix(run: Run) -> str:
    return f"mem0_{run.id}_"


async def _collection_is_referenced(
    session: AsyncSession,
    collection_name: str,
    *,
    excluding_run_id: str,
    owner: str | None,
) -> bool:
    ref_id = await session.scalar(
        select(Run.id)
        .where(Run.collection_name == collection_name, Run.id != excluding_run_id)
        .limit(1)
    )
    if ref_id is not None:
        return True

    # Older rows can pre-date the collection_name column. Fresh web runs have
    # always used web_<run_id>, so a legacy root row still references that
    # collection even if the explicit column is NULL. Legacy rows are NULL on
    # owner_id too, so this fallback can only apply in single-tenant mode
    # (owner is None); in public mode every row has an explicit owner_id and a
    # populated collection_name, so the strict reference query above is enough.
    if owner is not None:
        return False
    legacy_run_id = collection_name.removeprefix("web_")
    if not legacy_run_id or legacy_run_id == excluding_run_id:
        return False
    legacy_owner = await session.get(Run, legacy_run_id)
    return (
        legacy_owner is not None
        and legacy_owner.collection_name is None
        and legacy_owner.mode == RunMode.new.value
    )


def _delete_chroma_collection(collection_name: str) -> bool:
    try:
        settings = get_settings()
        chroma_dir = settings.caesar_web_data_dir.resolve() / "chroma"
        chroma_dir.mkdir(parents=True, exist_ok=True)

        from rome.kb_server import ChromaServerManager  # noqa: WPS433

        manager = ChromaServerManager.get_instance(config={
            "host": "localhost",
            "port": settings.chroma_port,
            "persist_path": str(chroma_dir),
        })
        if not manager.is_running() and not manager.start():
            logger.warning(
                "Failed to start ChromaDB server to delete collection %s",
                collection_name,
            )
            return False
        return manager.delete_collection(collection_name)
    except Exception as exc:
        logger.warning("Failed to delete Chroma collection %s: %s", collection_name, exc)
        return False


def _chroma_collection_names_with_prefix(prefixes: Iterable[str]) -> set[str]:
    prefixes = tuple(sorted(set(prefixes)))
    if not prefixes:
        return set()
    try:
        settings = get_settings()
        chroma_dir = settings.caesar_web_data_dir.resolve() / "chroma"
        chroma_dir.mkdir(parents=True, exist_ok=True)

        from rome.kb_server import ChromaServerManager  # noqa: WPS433

        manager = ChromaServerManager.get_instance(config={
            "host": "localhost",
            "port": settings.chroma_port,
            "persist_path": str(chroma_dir),
        })
        if not manager.is_running() and not manager.start():
            logger.warning("Failed to start ChromaDB server to list collections")
            return set()

        import chromadb  # noqa: WPS433

        client = chromadb.HttpClient(host=manager.host, port=manager.port)
        # chromadb.HttpClient hardcodes httpx Timeout(None) on its internal
        # session — see rome/kb_client.py:281 for the same override. Without
        # this, a wedged chroma (heartbeat OK, /collections hangs) locks the
        # FastAPI worker forever. Best-effort: private-API access, fine to
        # log-and-continue if the shape changes.
        try:
            import httpx as _httpx  # noqa: WPS433
            client._server._session.timeout = _httpx.Timeout(
                connect=5.0, read=30.0, write=30.0, pool=10.0
            )
        except (AttributeError, ImportError):
            logger.debug("Could not override chromadb httpx timeout (private API may have changed)")
        return {
            collection.name
            for collection in client.list_collections()
            if collection.name.startswith(prefixes)
        }
    except Exception as exc:
        logger.warning("Failed to list Chroma collections: %s", exc)
        return set()


async def _delete_chroma_collections(
    collection_names: Iterable[str],
    *,
    collection_prefixes: Iterable[str] = (),
) -> tuple[int, int]:
    collection_names = set(collection_names)
    prefixes = set(collection_prefixes)
    if prefixes:
        collection_names.update(
            await asyncio.to_thread(_chroma_collection_names_with_prefix, prefixes)
        )

    removed = 0
    failed = 0
    for collection_name in sorted(set(collection_names)):
        if await asyncio.to_thread(_delete_chroma_collection, collection_name):
            removed += 1
        else:
            failed += 1
    if failed:
        logger.warning("Failed to delete %d Chroma collection(s)", failed)
    return removed, failed


def _summary_from(run: Run) -> RunSummary:
    summary = RunSummary.model_validate(run, from_attributes=True)
    # In-flight runs haven't persisted total_cost_usd / graph_node_count
    # yet, so the cards on the home / past-runs pages render no $/nodes
    # pills until completion. Overlay the watchdog's live counters so the
    # user can see cost and graph metrics while the run is still going.
    # job_pool.live_metrics returns (None, None) once the run is terminal,
    # so completed rows always use the persisted DB values.
    # Live values win outright while a run is in flight, rather than only
    # filling a NULL. A restarted run still carries the previous attempt's
    # persisted cost and node count, so the "only if None" form skipped the
    # overlay entirely and the page froze at the old numbers for the whole new
    # attempt (observed: cost pinned at $0.8417 while cost_update events
    # reported $0.8661). live_metrics already returns (None, None) for terminal
    # runs, so completed rows still use the persisted values.
    if run.status in (RunStatus.queued.value, RunStatus.running.value):
        live_cost, live_graph_node_count = job_pool.live_metrics(run.id)
        if live_cost is not None:
            summary.total_cost_usd = live_cost
        if live_graph_node_count is not None:
            summary.graph_node_count = live_graph_node_count
    return summary


def _read_merged_query(run: Run) -> str | None:
    if not run.repository:
        return None
    merged_path = Path(run.repository) / "__rome__" / "merged_query.txt"
    try:
        if not merged_path.exists():
            return None
        text = merged_path.read_text(encoding="utf-8").strip()
    except OSError:
        logger.warning("Failed to read merged_query.txt for run %s", run.id)
        return None
    return text or None


def _event_to_out(event: RunEvent) -> RunEventOut:
    try:
        payload = json.loads(event.payload) if event.payload else {}
    except json.JSONDecodeError:
        payload = {"_raw": event.payload}
    return RunEventOut(
        id=event.id,
        timestamp=event.timestamp,
        event=event.event,
        payload=payload,
    )


def _payload_int(payload: dict, key: str) -> int:
    try:
        value = int(payload.get(key) or 0)
    except (TypeError, ValueError):
        return 0
    return value if value > 0 else 0


def _graph_progress_from_events(events: Iterable[RunEvent]) -> tuple[str | None, int | None]:
    by_phase: dict[str, tuple[int, int | None]] = {}
    for event in events:
        if event.event == "iteration":
            try:
                payload = json.loads(event.payload) if event.payload else {}
            except json.JSONDecodeError:
                payload = {}
            phase = payload.get("phase")
            if isinstance(phase, str):
                n = _payload_int(payload, "n")
                total = _payload_int(payload, "total") or None
                last_n, last_total = by_phase.get(phase, (0, None))
                by_phase[phase] = (
                    max(last_n, n),
                    max(last_total or 0, total or 0) or None,
                )
        elif event.event == "graph_update":
            try:
                payload = json.loads(event.payload) if event.payload else {}
            except json.JSONDecodeError:
                payload = {}
            n = _payload_int(payload, "iter")
            last_n, last_total = by_phase.get("quick_explore", (0, None))
            by_phase["quick_explore"] = (max(last_n, n), last_total)

    # Keep this in sync with LiveProgress' fallback priority.
    for phase in ("kb_ingest", "explore", "quick_explore"):
        if phase in by_phase:
            n, total = by_phase[phase]
            return phase, total or n or None
    return None, None


async def _graph_progress_for_run(
    run_id: str,
    session: AsyncSession,
) -> tuple[str | None, int | None]:
    result = await session.execute(
        select(RunEvent)
        .where(
            RunEvent.run_id == run_id,
            RunEvent.event.in_(("iteration", "graph_update")),
        )
    )
    return _graph_progress_from_events(result.scalars())


async def _graph_run_id_for(run: Run, session: AsyncSession) -> str | None:
    if run.mode != RunMode.refine.value:
        return run.id

    seen = {run.id}
    parent_id = run.parent_run_id
    while parent_id and parent_id not in seen:
        seen.add(parent_id)
        parent = await session.get(Run, parent_id)
        if parent is None:
            return None
        if parent.mode != RunMode.refine.value:
            return parent.id
        parent_id = parent.parent_run_id
    return None


async def _summary_for(run: Run, session: AsyncSession) -> RunSummary:
    summary = _summary_from(run)
    summary.merged_query = _read_merged_query(run)
    if run.mode != RunMode.refine.value:
        return summary

    graph_run_id = await _graph_run_id_for(run, session)
    if graph_run_id is None or graph_run_id == run.id:
        return summary

    graph_run = await session.get(Run, graph_run_id)
    if graph_run is not None and graph_run.graph_node_count is not None:
        summary.graph_node_count = graph_run.graph_node_count
    return summary


@router.post("", status_code=status.HTTP_201_CREATED, response_model=RunSummary)
async def create_run(
    body: RunCreate,
    session: AsyncSession = Depends(get_session),
    owner: str | None = Depends(current_owner),
) -> RunSummary:
    if preset_by_id(body.preset) is None:
        raise HTTPException(status_code=400, detail=f"Unknown preset '{body.preset}'.")

    settings = get_settings()
    # In public (bring-your-own-key) mode every submission must carry the
    # caller's own OpenAI key in the body; the server holds no key of its own.
    # The schema validator allows None (shared mode omits it), so the router
    # enforces the "required when public" rule here.
    if settings.public_mode and not (body.api_key and body.api_key.strip()):
        raise HTTPException(
            status_code=400,
            detail="An OpenAI API key is required to submit a query.",
        )

    # Public-mode only: honor an optional synthesis-model override. Validate it
    # against the supported-OpenAI list; ignore it entirely outside public mode
    # (where the preset YAML is authoritative).
    synthesis_model: str | None = None
    if settings.public_mode and body.synthesis_model:
        if not is_valid_synthesis_model(body.synthesis_model):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported synthesis model '{body.synthesis_model}'.",
            )
        synthesis_model = body.synthesis_model

    # Follow-up validation. Require a completed parent: a failed parent may
    # have no synthesis to seed from and only a partially-populated KB.
    parent: Run | None = None
    if body.mode != "new":
        if not body.parent_run_id:
            raise HTTPException(
                status_code=400,
                detail=f"mode={body.mode!r} requires parent_run_id.",
            )
        parent = await session.get(Run, body.parent_run_id)
        if parent is None:
            raise HTTPException(
                status_code=404, detail=f"Parent run {body.parent_run_id!r} not found."
            )
        # Enforce ownership BEFORE inheriting the parent's collection_name so a
        # caller cannot seed a new run from another tenant's KB. 404 (not 403)
        # to avoid confirming the parent run-id exists. Skipped in single-tenant
        # mode (owner is None).
        if owner is not None and parent.owner_id != owner:
            raise HTTPException(
                status_code=404, detail=f"Parent run {body.parent_run_id!r} not found."
            )
        if parent.status != RunStatus.completed.value:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Follow-ups require a completed parent run "
                    f"(current status: {parent.status})."
                ),
            )
        if not parent.repository:
            raise HTTPException(
                status_code=400,
                detail="Parent run has no artifact directory; cannot follow up.",
            )
        # Concurrent follow-ups against the same collection are safe:
        # Chroma serialises writes per-collection internally, the SHA256
        # dedup is idempotent (same content → same row), and refine is
        # read-only. The previous "one at a time" gate here was overly
        # conservative belt-and-suspenders and produced a real UX
        # papercut when users wanted to run a refine while an explore
        # was still finishing.

    if job_pool.active_count() >= settings.caesar_max_concurrent:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Server is at capacity ({settings.caesar_max_concurrent} concurrent runs). "
                "Try again in a few minutes."
            ),
        )

    run_id = uuid.uuid4().hex
    # Fresh runs own a brand-new collection; follow-ups inherit the parent's
    # so that chains of follow-ups all converge on the original ancestor's
    # KB instead of silently abandoning the inheritance at depth >1.
    if body.mode != "new" and parent is not None:
        collection_name = parent.collection_name or f"web_{parent.id}"
    else:
        collection_name = f"web_{run_id}"
    run = Run(
        id=run_id,
        query=body.query,
        preset=body.preset,
        synthesis_model=synthesis_model,
        # Public mode: stash the key so a restart can auto-resume from
        # checkpoint; cleared on terminal + startup purge + TTL. Non-public runs
        # use the operator env key, so nothing is stored.
        run_api_key=body.api_key if settings.public_mode else None,
        status=RunStatus.queued.value,
        parent_run_id=body.parent_run_id if body.mode != "new" else None,
        mode=body.mode,
        collection_name=collection_name,
        owner_id=owner,
        created_at=datetime.now(timezone.utc),
    )
    session.add(run)
    await session.flush()

    # Schedule the actual work. The pool is responsible for transitioning
    # status -> running -> completed/failed and writing RunEvent rows. The
    # per-run api_key (public mode) rides into in-memory _RunState only and is
    # never persisted; pass it as a keyword arg.
    await job_pool.submit(
        run_id=run_id,
        query=body.query,
        preset_id=body.preset,
        mode=body.mode,
        parent_run_id=body.parent_run_id if body.mode != "new" else None,
        collection_name=collection_name,
        api_key=body.api_key,
        synthesis_model=synthesis_model,
    )
    logger.info(
        "Submitted run %s preset=%s mode=%s parent=%s collection=%s synth_model=%s",
        run_id, body.preset, body.mode, body.parent_run_id, collection_name,
        synthesis_model or "(preset default)",
    )

    return await _summary_for(run, session)


@router.get("", response_model=list[RunSummary])
async def list_runs(
    limit: int = 50,
    session: AsyncSession = Depends(get_session),
    owner: str | None = Depends(current_owner),
    admin: bool = Depends(is_admin),
) -> list[RunSummary]:
    limit = max(1, min(limit, 200))
    # owner is None in single-tenant mode -> owner_id == None emits IS NULL,
    # which matches every legacy row (all NULL): identical to today's listing.
    # Admin (public-mode operator step-up) drops the owner filter to list every
    # user's runs.
    stmt = select(Run)
    if owner is not None and not admin:
        stmt = stmt.where(Run.owner_id == owner)
    result = await session.execute(
        stmt.order_by(desc(Run.created_at)).limit(limit)
    )
    return [await _summary_for(r, session) for r in result.scalars().all()]


@router.get("/{run_id}", response_model=RunDetail)
async def get_run(
    run_id: str,
    session: AsyncSession = Depends(get_session),
    owner: str | None = Depends(current_owner),
    admin: bool = Depends(is_admin),
) -> RunDetail:
    run = await get_owned_run(run_id, owner, session, admin=admin)
    graph_run_id = await _graph_run_id_for(run, session)
    # Force-load events for the response; inherited graph progress is computed
    # from the graph-owning ancestor below.
    await session.refresh(run, attribute_names=["events"])

    graph_progress_total: int | None = None
    graph_progress_phase: str | None = None
    if graph_run_id is not None and graph_run_id != run.id:
        graph_run = await session.get(Run, graph_run_id)
        if graph_run is not None:
            graph_progress_phase, graph_progress_total = await _graph_progress_for_run(
                graph_run.id,
                session,
            )
            if graph_progress_phase is not None:
                graph_progress_total = graph_progress_total or graph_run.graph_node_count

    return RunDetail(
        **(await _summary_for(run, session)).model_dump(),
        events=[_event_to_out(e) for e in run.events],
        llm_model=run.synthesis_model or preset_llm_model(run.preset),
        graph_run_id=graph_run_id,
        graph_progress_total=graph_progress_total,
        graph_progress_phase=graph_progress_phase,
    )


@router.post("/{run_id}/retry", response_model=RunSummary)
async def retry_run(
    run_id: str,
    body: RunRetry,
    session: AsyncSession = Depends(get_session),
    owner: str | None = Depends(current_owner),
    admin: bool = Depends(is_admin),
) -> RunSummary:
    """Re-submit a run in place: same row, same artifact dir, same KB
    collection, resuming from its checkpoint when one survived.

    The point of restarting rather than re-submitting is the checkpoint. A run
    that died late — say the KB server went away during synthesis — keeps its
    whole graph, so un-archiving the checkpoint turns a 4-hour re-crawl into a
    synthesis-only retry. With no checkpoint on disk this just starts the same
    query over, which is still better than a new row with a new empty KB.

    Every run is eligible, whatever its status. A completed run has a checkpoint
    whose exploration is already finished, so restarting it regenerates the
    answer from the KB it already paid for. A run that is still going is stopped
    first and picks up from the checkpoint its own shutdown just wrote, which
    makes this the way to unstick a run without losing its graph.

    The one hard rule is that two agents must never share a run directory, and
    that is a question about the worker thread rather than about the row's
    status, so `stop_run` waits for the thread to actually exit.
    """
    run = await get_owned_run(run_id, owner, session, admin=admin)
    settings = get_settings()

    # Ask any live attempt to stop and move on. The new attempt waits for that
    # worker to exit before touching the run directory, so a restart can't put
    # two agents on one run and can't be refused for arriving mid-step. Waiting
    # here instead used to cancel the attempt, time out, return 409, and leave
    # the row marked running with nothing running.
    await job_pool.request_stop(run_id)
    # The row outlives the preset config, so re-validate instead of requeueing
    # into a guaranteed failure.
    if preset_by_id(run.preset) is None:
        raise HTTPException(
            status_code=400,
            detail=f"Preset '{run.preset}' no longer exists; restart is unavailable.",
        )
    # Same bring-your-own-key rule as submit: the stored key was deleted the
    # moment this run went terminal, so a restart must carry a fresh one.
    if settings.public_mode and not (body.api_key and body.api_key.strip()):
        raise HTTPException(
            status_code=400,
            detail="An OpenAI API key is required to restart a run.",
        )
    if job_pool.active_count() >= settings.caesar_max_concurrent:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Server is at capacity ({settings.caesar_max_concurrent} concurrent runs). "
                "Try again in a few minutes."
            ),
        )

    # `_run` always works out of runs_dir/<run_id>, so that is the only place a
    # restored checkpoint can be found — don't chase run.repository here.
    resuming = unarchive_checkpoint(settings.runs_dir / run.id)

    # An explicit restart starts a new attempt, so the clock restarts with it:
    # the progress counters beside it (draft, iteration) reset too, and leaving
    # started_at alone made Elapsed span the idle gap, reading "31h 45m" on a run
    # restarted a day later. Bank the finished attempt's span first, so the
    # displayed figure stays "what this run has consumed" rather than dropping
    # every earlier attempt. Wall-clock of the attempt, so a server outage inside
    # it counts as elapsed; tracking true process-active time would need the
    # worker to report and isn't worth the machinery.
    if run.started_at is not None:
        started = run.started_at
        if started.tzinfo is None:  # SQLite hands back naive UTC
            started = started.replace(tzinfo=timezone.utc)
        spent = (datetime.now(timezone.utc) - started).total_seconds()
        run.elapsed_prior_s = (run.elapsed_prior_s or 0.0) + max(0.0, spent)
    run.started_at = None
    run.status = RunStatus.queued.value
    run.error_message = None
    run.finished_at = None
    if settings.public_mode:
        run.run_api_key = body.api_key
    # Commit before submitting: the worker flips this row to `running` from its
    # own session, and this request's end-of-scope commit would otherwise stomp
    # that back to `queued` with nothing left to correct it.
    await session.commit()

    try:
        await job_pool.submit(
            run_id=run.id,
            query=run.query,
            preset_id=run.preset,
            resuming=resuming,
            mode=run.mode or "new",
            parent_run_id=run.parent_run_id,
            collection_name=run.collection_name,
            api_key=body.api_key,
            synthesis_model=run.synthesis_model,
        )
    except Exception:
        # create_run gets rollback-on-error for free because its commit happens
        # after submit; ours already landed, so put the row back by hand rather
        # than leave a queued ghost for the next restart to clean up.
        logger.exception("Restart of run %s failed to submit; reverting row.", run.id)
        run.status = RunStatus.failed.value
        run.error_message = "Restart could not be scheduled. Please try again."
        run.finished_at = datetime.now(timezone.utc)
        run.run_api_key = None
        await session.commit()
        raise HTTPException(status_code=500, detail="Could not restart this run.") from None

    logger.info(
        "Restarted run %s preset=%s mode=%s resuming=%s collection=%s",
        run.id, run.preset, run.mode, resuming, run.collection_name,
    )
    return await _summary_for(run, session)


@router.delete("", status_code=status.HTTP_200_OK)
async def wipe_all_runs(
    confirm: str = "",
    session: AsyncSession = Depends(get_session),
    owner: str | None = Depends(current_owner),
    admin: bool = Depends(is_admin),
) -> dict:
    """Cancel every in-flight run, drop all DB rows, and delete every
    artifact directory under runs_dir. Requires `?confirm=yes` so an
    accidental DELETE doesn't nuke the demo. Returns counts.

    Auth is enforced upstream by the Next.js cookie middleware. We
    can't check `request.client.host` here because uvicorn defaults to
    `proxy_headers=True`, which makes `client.host` the original
    end-user IP (via X-Forwarded-For), not the Next.js peer. The real
    defense against direct-port end-runs is launch.sh binding the API
    to 127.0.0.1 when --password is set."""
    if confirm != "yes":
        raise HTTPException(
            status_code=400,
            detail="Add ?confirm=yes to wipe all runs (this is irreversible).",
        )

    # Cancel every in-flight task and await its finally block so the
    # background worker doesn't try to commit a row we're about to drop.
    # Scope the snapshot to this owner: everything downstream (cancel,
    # repo/collection removal, DB delete) derives from this list, so an owner
    # can only ever wipe their own runs. In single-tenant mode owner is None ->
    # owner_id == None matches all NULL rows = nuke-everything as before. Admin
    # (public-mode operator step-up) drops the filter to wipe every user's runs.
    stmt = select(Run)
    if owner is not None and not admin:
        stmt = stmt.where(Run.owner_id == owner)
    result = await session.execute(stmt)
    runs = list(result.scalars().all())
    cancelled = 0
    for run in runs:
        if run.status in (RunStatus.queued.value, RunStatus.running.value):
            await job_pool.cancel_run(run.id)
            cancelled += 1

    # Snapshot repo paths BEFORE the DB delete — once the row is gone
    # we lose the artifact location. Drop DB rows first so a mid-loop
    # rmtree failure doesn't leave half-consistent state (DB rows
    # claiming an artifact dir that's already gone).
    settings = get_settings()
    runs_root = settings.runs_dir.resolve()
    repos_to_remove: list[Path] = []
    collections_to_remove = {
        collection_name
        for collection_name in (_run_collection_name(run) for run in runs)
        if collection_name is not None
    }
    collection_prefixes_to_remove = {_run_memory_collection_prefix(run) for run in runs}
    for run in runs:
        if run.repository:
            repo_path = Path(run.repository).resolve()
            try:
                repo_path.relative_to(runs_root)
                repos_to_remove.append(repo_path)
            except ValueError:
                logger.warning(
                    "Refusing to delete repo outside runs_dir: %s", repo_path)

    rows = 0
    for run in runs:
        await session.delete(run)
        rows += 1
    # Commit the DB transaction before touching the filesystem so the
    # ORM session releases its lock and we can't end up with rows
    # pointing at deleted directories.
    await session.commit()

    repos_removed = 0
    for repo_path in repos_to_remove:
        try:
            shutil.rmtree(repo_path, ignore_errors=True)
            repos_removed += 1
        except Exception as e:
            logger.warning("Failed to rmtree %s: %s", repo_path, e)

    chroma_removed, chroma_failed = await _delete_chroma_collections(
        collections_to_remove,
        collection_prefixes=collection_prefixes_to_remove,
    )

    logger.info(
        "Wiped %d run(s); cancelled %d in-flight; removed %d repo dirs; "
        "removed %d Chroma collection(s)",
        rows,
        cancelled,
        repos_removed,
        chroma_removed,
    )
    return {
        "deleted": rows,
        "cancelled": cancelled,
        "repos_removed": repos_removed,
        "chroma_removed": chroma_removed,
        "chroma_failed": chroma_failed,
    }


@router.delete("/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_run(
    run_id: str,
    session: AsyncSession = Depends(get_session),
    owner: str | None = Depends(current_owner),
    admin: bool = Depends(is_admin),
) -> Response:
    """Delete a run: cancel if in-flight, then drop the DB rows and the
    artifact directory/owned Chroma collection. Idempotent: deleting a
    missing run returns 204."""
    run = await session.get(Run, run_id)
    # Idempotent for a missing run, and treat a cross-owner run the same way
    # (no-op 204) so this route is neither a deletion oracle nor a run-id
    # enumeration oracle. owner is None in single-tenant mode, so this never
    # blocks the existing behavior. Admin may delete any run.
    if run is None or (owner is not None and not admin and run.owner_id != owner):
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # Block deletion when a follow-up child exists. Admin considers children
    # across all owners; a normal owner only their own.
    child_stmt = select(Run.id).where(Run.parent_run_id == run_id)
    if owner is not None and not admin:
        child_stmt = child_stmt.where(Run.owner_id == owner)
    child_id = await session.scalar(child_stmt.limit(1))
    if child_id is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Cannot delete this run because it has follow-up runs. "
                "Delete the follow-ups first."
            ),
        )

    # If the run is still running, cancel and wait for the task's finally
    # block to settle so we don't race with a concurrent DB write.
    if run.status in (RunStatus.queued.value, RunStatus.running.value):
        await job_pool.cancel_run(run_id)
        # The cancel path writes a 'failed' row; refresh our local handle.
        await session.refresh(run)

    collection_name = _run_collection_name(run)
    remove_collection = (
        collection_name is not None
        and not await _collection_is_referenced(
            session,
            collection_name,
            excluding_run_id=run_id,
            owner=owner,
        )
    )

    settings = get_settings()
    runs_root = settings.runs_dir.resolve()
    repo_to_remove: Path | None = None
    if run.repository:
        repo_path = Path(run.repository).resolve()
        try:
            repo_path.relative_to(runs_root)  # raises if not under runs_root
            repo_to_remove = repo_path
        except ValueError:
            logger.warning(
                "Refusing to delete repo outside runs_dir: %s (runs_dir=%s)",
                repo_path,
                runs_root,
            )

    await session.delete(run)
    await session.commit()

    if repo_to_remove is not None:
        try:
            shutil.rmtree(repo_to_remove, ignore_errors=True)
        except Exception as exc:
            logger.warning("Failed to rmtree %s: %s", repo_to_remove, exc)
    collections_to_remove = [collection_name] if remove_collection and collection_name else []
    await _delete_chroma_collections(
        collections_to_remove,
        collection_prefixes=[_run_memory_collection_prefix(run)],
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
