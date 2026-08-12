"""FastAPI entry point for the Caesar web server."""
from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, update

from . import deps
from .config import get_settings
from .db import SessionLocal, engine, init_db
from .job_runner import archive_checkpoint, has_checkpoint, job_pool
from .models import Run, RunStatus
from .routers import artifacts, models, presets, runs, stream

logger = logging.getLogger("caesar.web")


def _read_caesar_version() -> str:
    # web_server runs only from source checkouts (it's excluded from the
    # PyPI wheel), so the top-level pyproject.toml is always at parents[3].
    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return "0.0.0+unknown"
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else "0.0.0+unknown"


CAESAR_VERSION = _read_caesar_version()


def _read_commit_sha() -> str | None:
    # Prefer a build-time stamp over asking git. Two reasons:
    #
    #  1. Containers have no .git to ask. .dockerignore excludes it from the
    #     build context, so `git rev-parse` inside the image exits 128 and the
    #     footer silently dropped its commit link on every k8s deploy (git IS
    #     installed there, so the binary was never the problem).
    #  2. It cannot go stale against the code that is running. The git read
    #     below reports whatever HEAD is at import; for an image that is fixed
    #     and correct, but a source checkout can be pulled forward underneath a
    #     long-lived process, leaving the footer pointing at a commit that no
    #     longer describes the running build (or, after a history rewrite, at
    #     one that is no longer in any branch).
    #
    # CI passes github.sha as a --build-arg that deploy/Dockerfile turns into
    # this env var. Require hex so a mangled value falls back to git instead of
    # rendering a dead GitHub link.
    stamped = os.environ.get("CAESAR_COMMIT_SHA", "").strip()
    if re.fullmatch(r"[0-9a-fA-F]{7,40}", stamped):
        return stamped[:8].lower()

    repo_root = Path(__file__).resolve().parents[3]
    try:
        out = subprocess.check_output(  # noqa: S603
            ["git", "-C", str(repo_root), "rev-parse", "--short=8", "HEAD"],  # noqa: S607
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode("utf-8", "replace").strip() or None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


CAESAR_COMMIT_SHA = _read_commit_sha()
SERVER_START_TIME = time.time()

# Synthetic HTML touching every common element + attribute. Parsing this
# once at startup populates lxml's name-interning dict serially, eliminating
# the most common code path of the 3.14 thread race.
# ruff: noqa: E501
_LXML_PRIMING_HTML = """\
<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="description" content="x"><meta property="og:title" content="x">
<title>warmup</title><link rel="canonical" href="x"><base href="x"><style>a{color:red}</style>
<script type="application/ld+json">{}</script>
</head><body>
<header><nav><a href="#" rel="noopener" target="_blank">a</a></nav></header>
<main><article id="x" class="x" data-x="x">
<h1>h1</h1><h2>h2</h2><h3>h3</h3><h4>h4</h4><h5>h5</h5><h6>h6</h6>
<p>p <strong>strong</strong> <em>em</em> <code>code</code> <span>span</span> <br> <small>small</small>
<sub>sub</sub> <sup>sup</sup> <abbr title="x">abbr</abbr> <cite>cite</cite> <q>q</q></p>
<ul><li>li1</li><li>li2</li></ul><ol><li>li</li></ol><dl><dt>dt</dt><dd>dd</dd></dl>
<blockquote cite="x"><p>quote</p></blockquote><pre><code>pre</code></pre>
<table><thead><tr><th scope="col">th</th></tr></thead><tbody><tr><td>td</td></tr></tbody></table>
<figure><img src="x" alt="x" width="1" height="1" loading="lazy" decoding="async">
<figcaption>cap</figcaption></figure>
<picture><source srcset="x" type="image/png"><img src="x" alt="x"></picture>
<video controls preload="none" poster="x"><source src="x" type="video/mp4"></video>
<audio controls preload="none"><source src="x" type="audio/mp3"></audio>
<form action="x" method="post" enctype="multipart/form-data">
<fieldset><legend>l</legend>
<label for="x">l</label><input type="text" name="x" id="x" placeholder="x" required>
<input type="email" name="e"><input type="password"><input type="checkbox"><input type="radio">
<textarea rows="2" cols="2"></textarea>
<select name="s"><option value="a">a</option><optgroup label="g"><option>b</option></optgroup></select>
<button type="submit">b</button>
</fieldset></form>
<details open><summary>s</summary><p>x</p></details>
<section><aside><div><b>b</b><i>i</i><u>u</u></div></aside></section>
<iframe src="x" sandbox=""></iframe><noscript>x</noscript>
</article></main><footer><address>x</address></footer>
</body></html>"""


async def _archive_stale_terminal_checkpoints(runs_dir: Path) -> None:
    """One-time-style backfill that survives every startup.

    Any run whose DB status is already terminal (completed/failed) but
    whose live checkpoint file is still on disk gets the checkpoint
    archived. This bridges the transition where Caesar's
    ``agent.shutdown()`` saves a final checkpoint at the end of every
    run, including successful ones — pre-resume code never archived
    them, so the runs_dir has stale live checkpoints we'd otherwise
    re-run on next boot.

    Idempotent and cheap: O(N) on number of run directories, single
    DB roundtrip, no-op once the backfill has run on a clean state.
    """
    if not runs_dir.exists():
        return
    stale: list[str] = []
    candidates: dict[str, Path] = {}
    for repo in runs_dir.iterdir():
        if not repo.is_dir():
            continue
        if has_checkpoint(repo):
            candidates[repo.name] = repo
    if not candidates:
        return
    async with SessionLocal() as session:
        rows = await session.execute(
            select(Run.id, Run.status).where(Run.id.in_(list(candidates.keys())))
        )
        for run_id, status in rows.all():
            if status in (RunStatus.completed.value, RunStatus.failed.value):
                stale.append(run_id)
    for run_id in stale:
        archive_checkpoint(candidates[run_id])
        logger.info("Archived stale checkpoint for terminal run %s.", run_id)


async def _resume_runs_with_live_checkpoints(runs_dir: Path) -> None:
    """For every run dir with a live checkpoint, fetch its DB row and
    resubmit. Runs with no matching DB row are logged and skipped (orphan
    artifact directories). Runs whose status was already terminal will
    have had their checkpoint archived by the backfill above, so they
    won't show up here.
    """
    if not runs_dir.exists():
        return
    live_dirs: list[Path] = []
    for repo in runs_dir.iterdir():
        if repo.is_dir() and has_checkpoint(repo):
            live_dirs.append(repo)
    if not live_dirs:
        return
    async with SessionLocal() as session:
        rows = await session.execute(
            select(Run).where(Run.id.in_([d.name for d in live_dirs]))
        )
        runs_by_id = {r.id: r for r in rows.scalars().all()}
    terminal = (RunStatus.completed.value, RunStatus.failed.value)
    submit_failures: list[str] = []
    for repo in live_dirs:
        run = runs_by_id.get(repo.name)
        if run is None:
            logger.warning("Orphan checkpoint at %s (no matching DB row); skipping.", repo)
            continue
        # Belt-and-suspenders: pass 1 (_archive_stale_terminal_checkpoints)
        # should have already archived any terminal-status run's checkpoint.
        # If it failed for any reason (partial archive, OSError), we still
        # don't want to resubmit a finished/failed run.
        if run.status in terminal:
            logger.warning(
                "Live checkpoint at %s but DB status=%s; skipping resume.",
                repo, run.status,
            )
            continue
        try:
            # Carry the follow-up linkage through the resume call: without
            # these the worker re-derives a fresh `web_<run_id>` collection
            # instead of the inherited one, silently abandoning the parent's
            # KB and reference draft on every server restart.
            await job_pool.submit(
                run_id=run.id,
                query=run.query,
                preset_id=run.preset,
                resuming=True,
                mode=run.mode or "new",
                parent_run_id=run.parent_run_id,
                collection_name=run.collection_name,
            )
            logger.info(
                "Resuming run %s from checkpoint (preset=%s, mode=%s, parent=%s).",
                run.id, run.preset, run.mode, run.parent_run_id,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to resubmit run %s on startup.", run.id)
            submit_failures.append(run.id)

    # If submit() failed, the row is stuck at its old status (running/queued)
    # with no worker. Mark these failed so the UI doesn't show ghost-live runs;
    # checkpoint stays live on disk for manual recovery (rename to retry).
    if submit_failures:
        async with SessionLocal() as session:
            await session.execute(
                update(Run)
                .where(Run.id.in_(submit_failures))
                .values(
                    status=RunStatus.failed.value,
                    error_message="Resubmit failed on server startup; rename checkpoint to retry.",
                )
                .execution_options(synchronize_session=False)
            )
            await session.commit()


async def _cleanup_ghost_runs(runs_dir: Path) -> None:
    """Mark rows still showing ``running``/``queued`` as ``failed`` if they
    have no live checkpoint. Runs with a live checkpoint were resubmitted
    by ``_resume_runs_with_live_checkpoints`` and their workers will update
    status themselves. The remaining rows are ghosts left by a crash before
    the first checkpoint write — without this they'd persist in the UI as
    fake live runs.
    """
    async with SessionLocal() as session:
        result = await session.execute(
            select(Run.id, Run.repository).where(
                Run.status.in_([RunStatus.queued.value, RunStatus.running.value])
            )
        )
        ghosts: list[str] = []
        for run_id, repository in result.all():
            repo = runs_dir / run_id
            if not has_checkpoint(repo) and (
                not repository or not has_checkpoint(Path(repository))
            ):
                ghosts.append(run_id)
        if ghosts:
            await session.execute(
                update(Run)
                .where(Run.id.in_(ghosts))
                .values(
                    status=RunStatus.failed.value,
                    error_message="Server restarted while run was in flight (no checkpoint).",
                    run_api_key=None,
                )
                .execution_options(synchronize_session=False)
            )
            await session.commit()
            logger.warning(
                "Marked %d ghost run(s) as failed on startup (no checkpoint).",
                len(ghosts),
            )


async def _resubmit_interrupted_runs() -> None:
    """Resubmit any runs whose synthesis exited mid-flight last boot.

    Detection has two paths:

    1. **Explicit:** row at ``status=interrupted``. The worker had time
       to write it before the process exited (cooperative shutdown
       reached the per-draft check between LLM calls).

    2. **Implicit:** row at ``status=running``/``queued``, mode is
       ``refine`` or ``explore``, and no live checkpoint on disk. A
       hard restart that arrives while the LLM call is in flight blows
       past the 15s grace and force-cancels before the worker can
       update status. Refine never writes a checkpoint at all, so its
       in-flight runs are always in this bucket. We promote them to
       ``interrupted`` here so the resubmit query catches them and
       ``_cleanup_ghost_runs`` (next pass) leaves them alone.

    ``new``-mode runs without a checkpoint are LEFT alone — they
    haven't done meaningful work yet, and auto-restarting a genuinely
    broken fresh run could loop forever. Those keep the existing
    ghost-fail behavior.

    Refine resubmits redo synthesis from scratch. Explore resubmits
    re-init the agent; the existing ``_resume_runs_with_live_checkpoints``
    pass would have already handled any with a checkpoint, and
    ``job_pool.submit`` dedupes by ``run_id`` so a double-submit is
    a no-op.
    """
    settings = get_settings()
    runs_dir = settings.runs_dir
    async with SessionLocal() as session:
        # Promote implicit orphans to status=interrupted so they're
        # picked up by the resubmit query below.
        candidates = await session.execute(
            select(Run).where(
                Run.status.in_(
                    [RunStatus.queued.value, RunStatus.running.value]
                ),
                Run.mode.in_(["refine", "explore"]),
            )
        )
        promote: list[str] = []
        for run in candidates.scalars().all():
            repo = runs_dir / run.id
            if not has_checkpoint(repo) and (
                not run.repository or not has_checkpoint(Path(run.repository))
            ):
                promote.append(run.id)
        if promote:
            await session.execute(
                update(Run)
                .where(Run.id.in_(promote))
                .values(status=RunStatus.interrupted.value)
                .execution_options(synchronize_session=False)
            )
            await session.commit()
            logger.info(
                "Promoted %d orphan synthesis run(s) to 'interrupted' for restart.",
                len(promote),
            )

        # Clear started_at on all interrupted rows so the restarted run's
        # elapsed timer measures THIS attempt only — not the failed first
        # attempt plus the downtime gap. submit(resuming=True) below will
        # populate started_at via _update_status's started_at_if_null path
        # when _run starts executing.
        await session.execute(
            update(Run)
            .where(Run.status == RunStatus.interrupted.value)
            .values(started_at=None)
            .execution_options(synchronize_session=False)
        )
        await session.commit()

        result = await session.execute(
            select(Run).where(Run.status == RunStatus.interrupted.value)
        )
        rows = list(result.scalars().all())
    submit_failures: list[str] = []
    for run in rows:
        try:
            await job_pool.submit(
                run_id=run.id,
                query=run.query,
                preset_id=run.preset,
                # resuming=True emits a "resumed" event instead of "log",
                # which the UI's LiveProgress + hasFreshDraft logic uses
                # to wipe pre-restart draft_complete events from the
                # progress display. Synthesis still has no checkpoint to
                # load — `resuming` here is purely about event signalling.
                resuming=True,
                mode=run.mode or "new",
                parent_run_id=run.parent_run_id,
                collection_name=run.collection_name,
            )
            logger.info(
                "Restarting interrupted run %s (preset=%s, mode=%s).",
                run.id, run.preset, run.mode,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to restart interrupted run %s.", run.id)
            submit_failures.append(run.id)

    if submit_failures:
        async with SessionLocal() as session:
            await session.execute(
                update(Run)
                .where(Run.id.in_(submit_failures))
                .values(
                    status=RunStatus.failed.value,
                    error_message=(
                        "Restart failed on server boot after interrupted "
                        "synthesis; resubmit manually if needed."
                    ),
                )
                .execution_options(synchronize_session=False)
            )
            await session.commit()


# Public-mode resume TTL: a run whose owner never comes back keeps its key at
# rest only up to this cutoff; past it the run is failed and the key purged, so
# abandoned keys don't linger.
PUBLIC_RESUME_TTL = timedelta(hours=72)


async def _recover_public_runs(runs_dir: Path) -> None:
    """Public-mode restart recovery: auto-resume where it is safe to.

    A public run's key rides into the DB at submit and is deleted the instant
    the run goes terminal. On boot we (1) purge any key left on an
    already-terminal run (crash-safety for delete-on-finish), then (2) for each
    non-terminal run, auto-resume it from its checkpoint IF it still has a live
    checkpoint, a stored key, and is within PUBLIC_RESUME_TTL; otherwise fail it
    with a re-submit message and purge its key. The key is never logged or
    serialized.
    """
    now = datetime.now(timezone.utc)
    cap = get_settings().caesar_max_concurrent
    async with SessionLocal() as session:
        # (1) Startup purge: a terminal run must never retain a key.
        await session.execute(
            update(Run)
            .where(
                Run.status.in_([RunStatus.completed.value, RunStatus.failed.value]),
                Run.run_api_key.is_not(None),
            )
            .values(run_api_key=None)
            .execution_options(synchronize_session=False)
        )
        await session.commit()
        rows = await session.execute(
            select(Run).where(
                Run.status.in_(
                    [
                        RunStatus.queued.value,
                        RunStatus.running.value,
                        RunStatus.interrupted.value,
                    ]
                )
            )
        )
        runs = list(rows.scalars().all())
    if not runs:
        return

    resumed: list[str] = []
    failed: list[str] = []
    for run in runs:
        repo = runs_dir / run.id
        repo_alt = Path(run.repository) if run.repository else None
        has_ckpt = has_checkpoint(repo) or (repo_alt is not None and has_checkpoint(repo_alt))
        key = run.run_api_key
        created = run.created_at
        if created is not None and created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        within_ttl = created is not None and (now - created) <= PUBLIC_RESUME_TTL
        can_resume = has_ckpt and key and within_ttl
        # Respect the concurrency cap: resume up to the limit, fail the overflow
        # (owner can re-submit) so a restart can't stampede past CAESAR_MAX_CONCURRENT.
        if can_resume and job_pool.active_count() >= cap:
            logger.warning(
                "Public resume at capacity (%d); failing run %s (owner can re-submit).",
                cap, run.id,
            )
            can_resume = False
        if can_resume:
            try:
                await job_pool.submit(
                    run_id=run.id,
                    query=run.query,
                    preset_id=run.preset,
                    resuming=True,
                    mode=run.mode or "new",
                    parent_run_id=run.parent_run_id,
                    collection_name=run.collection_name,
                    api_key=key,
                    synthesis_model=run.synthesis_model,
                    output_length=run.output_length,
                )
                resumed.append(run.id)
                continue
            except Exception:  # noqa: BLE001
                logger.exception("Public resume failed for run %s.", run.id)
        # Not resumable: fail + purge key + archive any checkpoint.
        failed.append(run.id)
        if has_checkpoint(repo):
            archive_checkpoint(repo)
        elif repo_alt is not None and has_checkpoint(repo_alt):
            archive_checkpoint(repo_alt)

    if failed:
        async with SessionLocal() as session:
            await session.execute(
                update(Run)
                .where(Run.id.in_(failed))
                .values(
                    status=RunStatus.failed.value,
                    error_message="The server restarted and this run couldn't be resumed automatically. Please re-submit your query.",
                    run_api_key=None,
                )
                .execution_options(synchronize_session=False)
            )
            await session.commit()
    logger.info("Public recovery: resumed %d run(s), failed %d.", len(resumed), len(failed))


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("rome_root=%s data_dir=%s dry_run=%s",
                settings.rome_root, settings.caesar_web_data_dir, settings.caesar_dry_run)

    # Public mode is bring-your-own-key: every run must supply its own OpenAI
    # key. Strip any operator LLM keys from the process environment at startup
    # so the server is FAIL-CLOSED however it was launched (systemd units source
    # the operator's shell rc, which re-exports these). The per-run key reaches
    # the agent via config + the per-run env window; with no ambient key, a
    # missed seam errors loudly instead of silently billing the operator. This
    # is the authoritative guard; launch.sh unsets them too as a pre-process
    # backstop. BRAVE_API_KEY (web search) is server-funded and kept.
    if settings.public_mode:
        for _k in ("OPENAI_API_KEY", "CHROMA_OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                   "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
            if os.environ.pop(_k, None) is not None:
                logger.warning("public_mode: stripped operator %s from env (fail-closed)", _k)

    # Set up the SQLite schema before serving the first request.
    await init_db()

    # Pre-warm lxml's HTML names dict to mitigate a Python-3.14 race in
    # __pyx_f_4lxml_5etree__fixHtmlDictNames (parallel inserts into the
    # global hash-cons dict can double-free). Parsing a single doc that
    # touches every common HTML tag/attribute populates the dict serially
    # before any thread pool spawns, so subsequent parses in worker
    # threads only do read-only lookups.
    try:
        import lxml.html as _lxml_html  # noqa: WPS433
        _lxml_html.fromstring(_LXML_PRIMING_HTML)
        logger.info("Pre-warmed lxml HTML names dict.")
    except Exception:  # noqa: BLE001
        logger.exception("lxml pre-warm failed (non-fatal).")

    # Resume recovery: a live `*.checkpoint.json` means "incomplete work."
    await _archive_stale_terminal_checkpoints(settings.runs_dir)
    if settings.public_mode:
        # Public-mode runs stash their key encrypted at submit, so we can
        # auto-resume from checkpoint where safe (and fail + purge the rest).
        await _recover_public_runs(settings.runs_dir)
    else:
        await _resume_runs_with_live_checkpoints(settings.runs_dir)
        await _resubmit_interrupted_runs()
    await _cleanup_ghost_runs(settings.runs_dir)

    yield

    # On shutdown, cancel any in-flight tasks so they don't keep writing to a
    # dead event loop.
    await job_pool.shutdown()
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Caesar Web Server",
        description=(
            "Backend API for the Caesar autonomous AI research agent. "
            "Submits queries, streams live progress via SSE, serves run artifacts."
        ),
        version=CAESAR_VERSION,
        lifespan=lifespan,
    )

    # The browser only ever talks to the Next.js process on :3000, which
    # rewrites /api/* to here server-side. So in production we don't need
    # any cross-origin allowance. In dev, allow localhost so a developer can
    # `curl localhost:8090/...` directly without proxy.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1|\d+\.\d+\.\d+\.\d+)(:\d+)?",
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["meta"])
    async def health():
        return {"ok": True}

    @app.get("/version", tags=["meta"])
    async def version():
        settings = get_settings()
        return {
            "version": CAESAR_VERSION,
            "commit": CAESAR_COMMIT_SHA,
            "uptime_seconds": time.time() - SERVER_START_TIME,
            "public_mode": settings.public_mode,
        }

    @app.get("/whoami", tags=["meta"])
    async def whoami(request: Request):
        """Public mode: return the caller's recovery code (their caesar_id) so
        they can save it and restore their run history after a cookie reset."""
        settings = get_settings()
        if not settings.public_mode:
            return {"public_mode": False, "owner_id": None}
        value = request.cookies.get(deps.CAESAR_ID_COOKIE)
        return {
            "public_mode": True,
            "owner_id": value if deps.is_valid_owner_token(value) else None,
            # Whether this browser has stepped up to admin (sees all runs).
            "is_admin": deps.is_admin(request),
        }

    @app.post("/restore", tags=["meta"])
    async def restore(request: Request, response: Response):
        """Public mode: adopt a previous identity from a recovery code by
        setting the caesar_id cookie to it, so the browser regains that
        owner's runs. The code IS the identity, so a valid one always works
        (it simply scopes which runs are visible)."""
        settings = get_settings()
        if not settings.public_mode:
            raise HTTPException(status_code=404, detail="Not found.")
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = None
        code = (body or {}).get("code") if isinstance(body, dict) else None
        code = code.strip() if isinstance(code, str) else ""
        if not deps.is_valid_owner_token(code):
            raise HTTPException(status_code=400, detail="Invalid recovery code.")
        response.set_cookie(
            deps.CAESAR_ID_COOKIE,
            code,
            max_age=deps.CAESAR_ID_MAX_AGE,
            httponly=True,
            samesite="lax",
            secure=True,
            path="/",
        )
        return {"ok": True, "owner_id": code}

    app.include_router(presets.router)
    app.include_router(models.router)
    app.include_router(runs.router)
    app.include_router(stream.router)
    app.include_router(artifacts.router)

    return app


app = create_app()
