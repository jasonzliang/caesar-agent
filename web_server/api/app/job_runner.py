"""Async task pool that runs Caesar in the background.

Lifecycle: queued -> running -> (completed | failed). `agent.explore()` is
blocking, so we run it in `asyncio.to_thread` and tail the run's artifact
dir + console log for derived SSE events.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import traceback
from collections.abc import AsyncIterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import update

from .config import (
    ensure_caesar_on_path,
    get_settings,
    preset_total_drafts,
    resolve_preset_yaml,
)
from .db import SessionLocal
from .models import Run, RunEvent, RunStatus

logger = logging.getLogger("caesar.web.jobs")

# All event types we know how to emit. Keep in sync with the UI's useEventSource.
EVENT_TYPES = (
    "log",
    "iteration",
    "synthesis_progress",
    "graph_update",
    "draft_complete",
    "cost_update",          # emitted only when accumulated_cost changes by ≥ $0.001
    "resumed",
    "done",
    "error",
)

CAESAR_SHUTDOWN_GRACE_S = 15.0  # grace per agent to checkpoint before force-cancel
# How long a restarted attempt waits for the previous attempt's thread to exit
# before refusing to share the run directory. Generous because it costs nothing
# to wait (the request already returned) and Caesar only checks the shutdown
# flag between steps: a synthesis step runs 45-90s and an LLM call can run far
# longer.
TAKEOVER_WAIT_S = 300.0

# Watchdog stall threshold. If no non-ping event fires for this long, the
# worker is considered stuck (thread wedged in untimed I/O, or died silently
# without state.finished.set() firing) and the run is marked failed. Chosen
# to be well above any legitimate long single-call — deep-explore quick_explore
# workers can take minutes; LLMHandler's own timeout is 900s.
WATCHDOG_STALL_S = 1200.0  # 20 minutes

# Public-mode bring-your-own-key handling. Chat/synthesis LLM calls get the
# per-run key threaded through config["LLMHandler"]["api_key"], but chromadb's
# OpenAIEmbeddingFunction, mem0's embedder, and LlamaIndex read os.environ at
# construction and ignore the config dict. Those clients are all built inside
# CaesarAgent.__init__, so we serialize a brief env window around just that
# construction. The lock prevents the 8-wide worker pool from racing on the
# shared os.environ dict.
_ENV_KEY_LOCK = threading.Lock()

# Both vars must be set: chromadb's OpenAIEmbeddingFunction prefers
# OPENAI_API_KEY, and kb_client copies CHROMA_OPENAI_API_KEY at import time and
# hard-raises if it is absent.
_OPENAI_ENV_KEYS = ("OPENAI_API_KEY", "CHROMA_OPENAI_API_KEY")

# Matches an OpenAI-style secret key so it can be redacted before any error
# text reaches SQLite / run_events / the SSE frame. The `*` in the class also
# catches OpenAI's own masked echo on auth errors (e.g. "sk-proj-***...9f9f",
# which still embeds the real key's prefix and last 4 chars).
_SK_RE = re.compile(r"sk-[A-Za-z0-9_*-]{8,}")


@contextmanager
def _openai_env_window(api_key: str | None):
    """Temporarily set OPENAI_API_KEY + CHROMA_OPENAI_API_KEY to the per-run
    key, restoring prior values on exit. No-op (and lock-free) when api_key is
    falsy so shared/password mode is unchanged. Lock-guarded so concurrent
    workers don't clobber each other's env during agent construction."""
    if not api_key:
        yield
        return
    with _ENV_KEY_LOCK:
        prev = {k: os.environ.get(k) for k in _OPENAI_ENV_KEYS}
        for k in _OPENAI_ENV_KEYS:
            os.environ[k] = api_key
        try:
            yield
        finally:
            for k in _OPENAI_ENV_KEYS:
                v = prev[k]
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def _scrub_secrets(text: str | None, api_key: str | None = None) -> str | None:
    """Redact OpenAI-style keys from error text before it is persisted/emitted.
    Removes anything matching sk-... plus the exact per-run key (which may not
    match the generic pattern). Returns the input unchanged when it is None."""
    if text is None:
        return None
    scrubbed = _SK_RE.sub("[REDACTED]", text)
    if api_key:
        scrubbed = scrubbed.replace(api_key, "[REDACTED]")
    return scrubbed

# Caesar writes synthesis output as .txt always and .json only when
# SYNTHESIS_SAVE_JSON is true (default false). Match either.
GRAPH_ITER_RE = re.compile(r"\.graph_iter(?P<n>\d+)\.json(?:\.gz)?$")
SYNTHESIS_RE = re.compile(r"\.synthesis-(?P<draft>\d+)\.[\d_-]+\.(?:json|txt)$")
MERGED_RE = re.compile(r"\.merged-(?P<n>\d+)\.[\d_-]+\.(?:json|txt)$")


def _find_parent_artifact(parent_repo: Path) -> Path | None:
    """Resolve the parent run's "best" synthesis file for a follow-up to
    consume as synthesis_reference_draft. Prefers a merged-N file; falls
    back to the latest synthesis-N draft. Returns None if no synthesis
    files exist (parent failed mid-exploration).

    Mirrors the resolution that routers/artifacts.py uses for
    /runs/{id}/synthesis?draft=latest, scoped down to the file lookup only.
    """
    if not parent_repo.exists():
        return None
    candidates: list[Path] = []
    for pattern in ("*.txt", "*.json"):
        for p in parent_repo.rglob(pattern):
            if MERGED_RE.search(p.name) or SYNTHESIS_RE.search(p.name):
                candidates.append(p)
    if not candidates:
        return None
    merged = [p for p in candidates if MERGED_RE.search(p.name)]
    pool = merged or candidates
    pool.sort(key=lambda p: p.stat().st_mtime)
    return pool[-1]


def _fetch_parent_query_sync(parent_run_id: str) -> str | None:
    """Return the parent's effective query when available, else raw query."""
    import sqlite3
    try:
        settings = get_settings()
        conn = sqlite3.connect(str(settings.db_path))
        try:
            cur = conn.execute(
                "SELECT query, repository FROM runs WHERE id = ?",
                (parent_run_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            raw_query, repository = row
            if repository:
                merged_path = Path(repository) / "__rome__" / "merged_query.txt"
                try:
                    merged = merged_path.read_text(encoding="utf-8").strip()
                    if merged:
                        return merged
                except OSError:
                    pass
            return raw_query
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to fetch parent query for %s", parent_run_id)
        return None


# Cheap model for the one-shot query-merge preprocessing — using the
# run's main preset model here would be ~50× more expensive for no lift.
_MERGE_QUERY_MODEL = "gpt-5.4-mini"

# Lazy-instantiated LLMHandler so we gain its auth/quota error classification,
# reasoning-model temperature stripping, cost tracking, and timeout/retry
# defaults rather than calling litellm.completion directly. One handler is
# reused across all runs (the merge call is stateless).
_merge_handler = None


def _get_merge_handler(api_key: str | None = None):
    # Tests can hit this without going through _invoke_caesar's
    # ensure_caesar_on_path() call, so guarantee `rome.*` is importable
    # before the LLMHandler import.
    ensure_caesar_on_path()
    from rome.config import DEFAULT_CONFIG  # noqa: WPS433
    from rome.llm_handler import LLMHandler  # noqa: WPS433
    # LLMHandler.set_attributes_from_config validates that every key in
    # DEFAULT_CONFIG['LLMHandler'] is present, so start from defaults and
    # override only what differs.
    cfg = {**DEFAULT_CONFIG["LLMHandler"], **{
        "model": _MERGE_QUERY_MODEL,
        "temperature": 0.0,
        # Headroom for reasoning tokens: the merge runs gpt-5.x at
        # reasoning_effort=high, and on a reasoning model max_completion_tokens
        # caps reasoning + visible output combined. The merged query is short,
        # but high-effort reasoning can burn most of the budget first; 1000
        # risked truncating mid-reasoning to empty output (-> mechanical
        # fallback). Only tokens actually used are billed.
        "max_completion_tokens": 8000,
        "max_retries": 0,
    }}
    if api_key:
        # Public mode: the server holds no operator key in its env, so the
        # merge must use the caller's per-run key. Build a fresh handler and
        # never cache it as the shared singleton (keys differ per user).
        cfg["api_key"] = api_key
        return LLMHandler(cfg)
    global _merge_handler
    if _merge_handler is None:
        _merge_handler = LLMHandler(cfg)
    return _merge_handler


def _merge_followup_query(
    parent_query: str,
    follow_up_query: str,
    cache_path: Path | None = None,
    api_key: str | None = None,
) -> str:
    """Rewrite (parent_query, follow_up_query) into one self-contained query
    so Caesar's synthesizer + search-keyword paths see a properly-scoped
    prompt. Cached on first success so an auto-restart reuses the same
    merge (Caesar's checkpoint validator would otherwise flag a mismatch).
    Falls back to a mechanical concatenation on LLM failure."""
    if cache_path is not None and cache_path.exists():
        try:
            cached = cache_path.read_text(encoding="utf-8").strip()
            if cached:
                logger.info("[FOLLOWUP] Using cached merged query (%s)", cache_path.name)
                return cached
        except OSError:
            logger.warning("Failed to read merged-query cache at %s; re-merging", cache_path)

    try:
        handler = _get_merge_handler(api_key)
        # A wrong merge poisons starting_query for the whole follow-up
        # (Caesar reuses it for keyword extraction + synthesis context).
        # 2-5s extra latency is invisible against a multi-minute run.
        resp = handler.completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You merge an original research query and a "
                        "follow-up into a single self-contained query, "
                        "what the user wants answered now.\n\n"
                        "CORE RULE: the follow-up determines the "
                        "question. The original supplies context "
                        "(framing, scope, criteria, comparison sets, "
                        "named examples) that makes the follow-up "
                        "answerable. Do not let the original take over "
                        "the question.\n\n"
                        "Special case: if the follow-up is a pure depth "
                        "or elaboration request ('explain more', 'go "
                        "deeper', 'in more detail'), the original is "
                        "the question and the follow-up just modifies "
                        "it.\n\n"
                        "KEEP from the original whatever the follow-up "
                        "needs to be answerable: comparison sets ('X vs "
                        "Y'), criteria lists, scope qualifiers (domain, "
                        "time, audience), methodological constraints, "
                        "named examples that anchor the topic.\n\n"
                        "DROP whatever the follow-up replaces or makes "
                        "redundant, plus filler phrases that don't carry "
                        "meaning.\n\n"
                        "LENGTH: match the inputs. A short parent + "
                        "short follow-up should yield a short merge. A "
                        "detailed parent should preserve its load-"
                        "bearing details. Self-contained means readable "
                        "without the original, not verbose.\n\n"
                        "Correct any spelling or grammar errors in the "
                        "follow-up while preserving the writer's intent.\n\n"
                        "Output ONLY the merged query. No quotes, labels, "
                        "explanation, or markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original query:\n{parent_query}\n\n"
                        f"Follow-up:\n{follow_up_query}\n\n"
                        f"Combined query:"
                    ),
                },
            ],
            reasoning_effort="high",
        )
        merged = (resp.choices[0].message.content or "").strip()
        if merged:
            logger.info("[FOLLOWUP] Merged query (LLM): %s",
                        merged[:200] + ("…" if len(merged) > 200 else ""))
            if cache_path is not None:
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(merged, encoding="utf-8")
                except OSError:
                    logger.warning("Failed to cache merged query at %s", cache_path)
            return merged
    except Exception:  # noqa: BLE001
        logger.exception("[FOLLOWUP] LLM query merge failed; using mechanical fallback")
    # Flat sentence rather than a labelled multi-line block so Caesar's
    # downstream prompts (search-keyword extraction, synthesis query_context)
    # don't see "Original question:" / "Follow-up:" labels embedded in the
    # starting_query — they'd be treated as part of the question text.
    fallback = f"{follow_up_query} (in the context of: {parent_query})"
    logger.info("[FOLLOWUP] Mechanical merge: %s",
                fallback[:200] + ("…" if len(fallback) > 200 else ""))
    return fallback


def _configure_refine_agent_startup(
    config: dict[str, Any],
    repo_dir: Path,
    parent_artifact_path: Path | None,
) -> None:
    """Disable exploration for synthesis-only follow-ups. CaesarAgent
    requires a starting_url even when we skip exploration; the parent's
    synthesis file works as a stable local URL, with a tiny placeholder
    as fallback so validation never touches the network."""
    agent_cfg = config.setdefault("CaesarAgent", {})
    agent_cfg["max_iterations"] = 0
    agent_cfg["additional_starting_queries"] = 0
    agent_cfg["adapt_role"] = False

    if agent_cfg.get("starting_url"):
        return

    start_path = parent_artifact_path
    if start_path is None or not start_path.exists():
        start_path = repo_dir / "__rome__" / "refine_start.html"
        start_path.parent.mkdir(parents=True, exist_ok=True)
        start_path.write_text(
            "<!doctype html><title>Refine follow-up</title>"
            "<p>Synthesis-only follow-up.</p>",
            encoding="utf-8",
        )
    agent_cfg["starting_url"] = start_path.resolve().as_uri()


class _RunState:
    """Holds the asyncio task, an event queue for SSE, and a 'finished' flag."""

    def __init__(
        self,
        run_id: str,
        preset_id: str,
        api_key: str | None = None,
        synthesis_model: str | None = None,
        output_length: int | None = None,
    ) -> None:
        self.run_id = run_id
        self.preset_id = preset_id
        # Public-mode bring-your-own-key. Held in memory only for the run's
        # lifetime; never persisted. None in shared/password mode.
        self.api_key = api_key
        # Public-mode synthesis-model override (LLMHandler.model). None keeps
        # the preset's model. In-memory only, like api_key.
        self.synthesis_model = synthesis_model
        # Public-mode artifact word target (ArtifactSynthesizer.synthesis_max_length).
        # None keeps the preset's value. In-memory only, like the two above.
        self.output_length = output_length
        self.task: asyncio.Task | None = None
        # Bounded so a stuck SSE client can't grow memory; oldest event drops.
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=512)
        self.finished = asyncio.Event()
        # Worker thread writes, watchdog reads. GIL-atomic.
        self.agent: Any | None = None
        # Set by shutdown() to bridge the window where shutdown fires before
        # state.agent is written — _invoke_caesar checks this after construct.
        self.shutdown_requested = False
        # Clear only while the Caesar thread is inside asyncio.to_thread; the
        # thread sets it from its own finally, after agent.shutdown() has written
        # the last checkpoint. Cancelling the task does NOT stop that thread, so
        # this is the only honest answer to "is the worker gone?", which a restart
        # must know before putting a second agent on the same run directory.
        # threading.Event (not asyncio.Event): the setter is the worker thread,
        # where asyncio.Event.set() would be unsafe. Starts set, so the dry-run
        # path (which never enters a thread) needs no wait.
        self.worker_done = threading.Event()
        self.worker_done.set()
        # Overlay metrics for in-flight runs (the /runs listing reads these
        # before total_cost_usd / graph_node_count land in the DB).
        self.live_cost_usd: float | None = None
        self.live_graph_node_count: int | None = None
        # Refreshed on every non-ping _emit. The watchdog compares this
        # against monotonic time to detect a worker that stopped producing
        # events (thread wedged inside an untimed I/O call, or died without
        # firing the _invoke_caesar finally). Prevents runs from sitting at
        # status='running' forever when the worker is gone.
        self.last_activity_mono: float = time.monotonic()


class JobPool:
    def __init__(self) -> None:
        self._states: dict[str, _RunState] = {}
        # run_id -> the worker thread's completion Event, deliberately kept
        # OUTSIDE _states. A cancelled run is popped from _states as soon as its
        # task unwinds, but its Caesar thread keeps running (asyncio cannot kill
        # it), so state membership answers the wrong question. Whoever wants to
        # know "may I take this run directory over?" has to ask this.
        self._worker_done: dict[str, threading.Event] = {}
        self._lock = asyncio.Lock()

    # -------- public API --------

    def active_count(self) -> int:
        return sum(1 for s in self._states.values() if not s.finished.is_set())

    def has_live_worker(self, run_id: str) -> bool:
        """True while a Caesar thread for this run is still running.

        Independent of _states on purpose: a run whose takeover timed out is
        already untracked while its thread works on. Trusting is_tracked() there
        let a second restart start a second agent on the same directory, with
        both writing the same checkpoint and graph files.
        """
        event = self._worker_done.get(run_id)
        return event is not None and not event.is_set()

    def is_tracked(self, run_id: str) -> bool:
        """True while the pool still holds an entry for this run.

        A row can read terminal while the entry lives on (the recovery paths in
        main.py mark rows failed without going through the worker, and the
        worker pops its state in a `finally` a beat after writing the status).
        `submit()` silently ignores a duplicate id, so a retry that raced one
        would leave the row at `queued` with nothing running.
        """
        return run_id in self._states

    def live_metrics(self, run_id: str) -> tuple[float | None, int | None]:
        """Return (live_cost_usd, live_graph_node_count) for an in-flight run, or
        (None, None) if the run isn't tracked or has already finished.
        Used by /runs and /runs/{id} to overlay metadata onto the DB row
        before the worker persists the final totals on completion."""
        state = self._states.get(run_id)
        if state is None or state.finished.is_set():
            return (None, None)
        return (state.live_cost_usd, state.live_graph_node_count)

    async def submit(
        self,
        *,
        run_id: str,
        query: str,
        preset_id: str,
        resuming: bool = False,
        mode: str = "new",
        parent_run_id: str | None = None,
        collection_name: str | None = None,
        api_key: str | None = None,
        synthesis_model: str | None = None,
        output_length: int | None = None,
    ) -> None:
        async with self._lock:
            if run_id in self._states:
                logger.warning("submit(%s) but already tracked; ignoring duplicate", run_id)
                return
            state = _RunState(
                run_id, preset_id=preset_id, api_key=api_key,
                synthesis_model=synthesis_model, output_length=output_length,
            )
            self._states[run_id] = state
            state.task = asyncio.create_task(
                self._run(
                    run_id, query, preset_id, state,
                    resuming=resuming, mode=mode, parent_run_id=parent_run_id,
                    collection_name=collection_name,
                ),
                name=f"caesar-run-{run_id}",
            )

    async def shutdown(self) -> None:
        """Set shutdown_called on each agent, wait up to CAESAR_SHUTDOWN_GRACE_S
        for workers to checkpoint, then force-cancel anything still live."""
        async with self._lock:
            states = list(self._states.values())

        for s in states:
            s.shutdown_requested = True
            if s.agent is not None:
                s.agent.shutdown_called = True

        tasks = [s.task for s in states if s.task and not s.task.done()]
        if tasks:
            await asyncio.wait(tasks, timeout=CAESAR_SHUTDOWN_GRACE_S)

        for t in tasks:
            if not t.done():
                t.cancel()
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: S110
                pass

    async def cancel_run(self, run_id: str) -> bool:
        """Cancel an in-flight run; returns True if it was active. The
        task's CancelledError handler leaves status=running, so callers
        that need a terminal row must update + delete themselves.

        asyncio.cancel only unblocks the awaiter; the cooperative shutdown
        flag is what lets the sync Caesar worker (inside asyncio.to_thread)
        actually exit between iterations."""
        state = self._states.get(run_id)
        if state is None or state.task is None or state.task.done():
            return False
        state.shutdown_requested = True
        if state.agent is not None:
            state.agent.shutdown_called = True
        state.task.cancel()
        try:
            await state.task
        except (asyncio.CancelledError, Exception):  # noqa: S110
            pass
        return True

    async def request_stop(self, run_id: str) -> None:
        """Ask an in-flight run to stop, without waiting for it.

        Deliberately does not block. Caesar only notices the cooperative flag
        between steps, and a synthesis step runs 45-90s, so waiting here meant a
        restart request routinely timed out *after* the cancel had already taken
        effect: the attempt died a minute later and nothing replaced it, leaving
        the row marked running with no worker.

        The new attempt does the waiting instead (see `_run`), which is where the
        real constraint lives: two agents must never share a run directory.
        """
        state = self._states.get(run_id)
        if state is not None:
            await self.cancel_run(run_id)
            # A task cancelled before the loop ran its first step executes no
            # body, so _run's finally never fires and the entry would linger with
            # `finished` unset: submit() would reject the id as a duplicate and a
            # concurrency slot would stay held. `is state` so a newer attempt is
            # never evicted.
            if self._states.get(run_id) is state:
                state.finished.set()
                self._states.pop(run_id, None)

    async def stream(self, run_id: str) -> AsyncIterator[dict[str, Any]]:
        """Yield events for a live run. Finished runs yield nothing; the caller
        replays persisted events from SQLite instead."""
        state = self._states.get(run_id)
        if state is None:
            return
        while True:
            try:
                event = await asyncio.wait_for(state.queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                if state.finished.is_set():
                    return
                # Application-level "ping" event every 5s, distinct from
                # sse-starlette's transport keepalive (stream.py ping=2,
                # which emits raw `: ping\n\n` SSE comments invisible to
                # the JS event API). This explicit ping event is what
                # useEventSource listens to as a liveness signal.
                yield {"event": "ping", "payload": {"ts": time.time()}}
                continue
            yield event
            if event.get("event") in ("done", "error"):
                return

    # -------- internals --------

    async def _run(
        self,
        run_id: str,
        query: str,
        preset_id: str,
        state: _RunState,
        resuming: bool = False,
        mode: str = "new",
        parent_run_id: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        settings = get_settings()
        repo_dir = settings.runs_dir / run_id
        repo_dir.mkdir(parents=True, exist_ok=True)

        # Resume preserves the original `started_at`. `started_at_if_null`
        # handles the rare queued-row case where it was never set.
        now_utc = datetime.now(timezone.utc)
        extra: dict[str, Any] = (
            {"error_message": None, "started_at_if_null": now_utc} if resuming
            else {"started_at": now_utc}
        )
        await self._update_status(
            run_id, status=RunStatus.running, repository=str(repo_dir), **extra,
        )

        if resuming:
            await self._emit(state, "resumed", message="Resuming from saved checkpoint.")
        else:
            await self._emit(state, "log", message=f"Run started for preset {preset_id!r}.")

        # A restart signals the previous attempt and submits immediately, so this
        # attempt may have inherited a run directory whose old Caesar thread is
        # still finishing its step. Captured before this attempt registers its
        # own event; awaited inside the try below so a timeout fails the run
        # cleanly instead of killing the task with the row still marked running.
        previous_worker = self._worker_done.get(run_id)

        # FatalLLMError is BaseException-derived (so generic `except Exception`
        # in intermediate retry/recovery layers can't swallow it), which means
        # this top-level handler must catch it explicitly. Import lazily so
        # FastAPI startup doesn't pay litellm's load cost.
        ensure_caesar_on_path()
        from rome.llm_handler import FatalLLMError  # noqa: WPS433

        watchdog_task: asyncio.Task | None = None
        try:
            # asyncio can't kill the previous attempt's thread; it exits when it
            # next checks the cooperative flag. Wait for that here rather than in
            # the HTTP request: two agents must never share a run directory, but
            # the caller shouldn't be held open (or refused) for a constraint
            # this attempt can honour itself. On timeout the raise lands in the
            # handler below, so the run ends `failed` with a message that says
            # what to do, and the finally releases the pool slot.
            if previous_worker is not None and not previous_worker.is_set():
                # Its own event type, not a `log` line: the UI has to render this
                # as a state (the stats beside it are frozen on the old attempt's
                # numbers for the duration), and matching on message text would
                # break the moment the wording changes.
                await self._emit(
                    state,
                    "takeover_wait",
                    message="Waiting for the previous attempt to stop…",
                )
                if not await asyncio.to_thread(previous_worker.wait, TAKEOVER_WAIT_S):
                    raise RuntimeError(
                        f"The previous attempt was still running {TAKEOVER_WAIT_S:.0f}s "
                        "after being asked to stop, so this restart was abandoned "
                        "rather than run two agents on the same directory. "
                        "Restart again once it has finished."
                    )

            watchdog_task = asyncio.create_task(
                self._watchdog(run_id, repo_dir, state),
                name=f"caesar-watchdog-{run_id}",
            )

            if settings.caesar_dry_run:
                artifact = await self._dry_run(run_id, query, preset_id, repo_dir, state)
            else:
                # Claim it before handing off: the thread sets it in its own
                # finally, and a restart waits on that. Registered pool-side too,
                # so the answer survives this run being popped from _states.
                state.worker_done.clear()
                self._worker_done[run_id] = state.worker_done
                artifact = await asyncio.to_thread(
                    self._invoke_caesar,
                    run_id, query, preset_id, repo_dir, state,
                    mode, parent_run_id, collection_name,
                )

            # Some Caesar paths put the count in metadata.synthesis_drafts
            # instead of top-level num_drafts; check both before rejecting.
            art = artifact or {}
            meta = art.get("metadata") or {}
            num_drafts = int(art.get("num_drafts") or meta.get("synthesis_drafts") or 0)
            artifact_text = (art.get("artifact") or "").strip()
            # num_drafts == 0 only happens via the empty-KB fast-path in
            # artifact_synthesis.py, which returns a placeholder "No insights
            # collected..." string that bypasses an `artifact_text`-only check.
            # Reject on either signal so empty-exploration runs fail loudly.
            if not artifact_text or num_drafts == 0:
                raise RuntimeError(
                    f"Caesar produced no synthesis output "
                    f"(num_drafts={num_drafts}, artifact_chars={len(artifact_text)})."
                )

            # Prefer the graph node count over meta's pages_visited so the
            # listing-card label matches the user-visible graph; the two
            # diverge whenever the agent discovered but didn't visit links.
            kg_nodes = state.live_graph_node_count
            if mode == "refine":
                # Refines don't build their own graph — inherit the parent's
                # exploration scope so the listing reflects the actual work
                # (routers/runs.py snapshots parent.graph_node_count onto the
                # child row at submission; preserve that value here rather
                # than overwriting to None, which reads as "0 nodes" in the UI
                # and hides the parent's exploration entirely).
                async with SessionLocal() as session:
                    existing = await session.get(Run, run_id)
                    graph_node_count = (
                        existing.graph_node_count if existing else None
                    )
            else:
                graph_node_count = kg_nodes if kg_nodes else meta.get("pages_visited")

            # If the stall watchdog already flipped this run to `failed`,
            # don't silently overwrite the terminal status with `completed`
            # (a truly-stuck thread that later returns would clobber the
            # correct failed verdict). See _watchdog stall-detection block.
            if state.finished.is_set():
                logger.warning(
                    "Run %s completed after watchdog marked it failed; "
                    "keeping failed status.",
                    run_id,
                )
                return

            await self._update_status(
                run_id,
                status=RunStatus.completed,
                finished_at=datetime.now(timezone.utc),
                total_cost_usd=meta.get("total_cost_usd"),
                graph_node_count=graph_node_count,
            )
            archive_checkpoint(repo_dir)
            # Set finished before `done` so the watchdog can't enqueue a stale
            # event between here and client disconnect.
            state.finished.set()
            await self._emit(
                state,
                "done",
                total_cost_usd=meta.get("total_cost_usd"),
                graph_node_count=graph_node_count,
                num_drafts=art.get("num_drafts"),
            )
        except asyncio.CancelledError:
            # Shutdown: leave status=running + live checkpoint for next boot.
            raise
        except (Exception, FatalLLMError) as e:
            # FatalLLMError (BaseException-derived) must be explicit here:
            # bare `except Exception` won't catch it, and we want it to mark
            # the run failed with the actual quota/auth message rather than
            # propagate as an unhandled asyncio-task exception.
            #
            # If shutdown was requested mid-flight, the exception is almost
            # certainly a side effect of cooperative shutdown firing (e.g.
            # synthesize_artifact's "No synthesis artifacts created" when
            # draft 1 is killed). Mark interrupted so auto-restart picks it
            # up on next boot instead of fail-ghosting a recoverable run.
            if state.shutdown_requested:
                logger.info(
                    "Run %s caught %s during shutdown; marking interrupted "
                    "for restart: %s", run_id, type(e).__name__, e,
                )
                try:
                    await self._update_status(
                        run_id, status=RunStatus.interrupted,
                    )
                    await self._emit(
                        state,
                        "log",
                        message=(
                            "Run interrupted during server shutdown; will "
                            "restart on next boot."
                        ),
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "Failed to mark interrupted for %s", run_id,
                    )
            else:
                # Log the failure class only; the full (scrubbed) message and
                # traceback are persisted to the run via _mark_failed below. A
                # raw exc_info traceback here would write the provider's masked
                # key echo to the operator log.
                logger.error("Run %s failed: %s", run_id, type(e).__name__)
                # Archive *before* _mark_failed: a DB-write failure here mustn't
                # leave a live checkpoint that the next boot auto-retries.
                archive_checkpoint(repo_dir)
                # Scrub before persist/emit: a litellm AuthenticationError
                # routinely embeds the offending key in both the message and
                # the traceback, which flow into runs.error_message (SQLite),
                # run_events.payload (via _emit), and the live SSE frame.
                await self._mark_failed(
                    run_id, state,
                    _scrub_secrets(f"{type(e).__name__}: {e}", state.api_key),
                    traceback_text=_scrub_secrets(
                        traceback.format_exc(limit=5), state.api_key,
                    ),
                )
        finally:
            state.finished.set()
            if watchdog_task is not None:
                watchdog_task.cancel()
                try:
                    await watchdog_task
                except (asyncio.CancelledError, Exception):  # noqa: S110
                    pass
            # Drop per-run state once the watchdog is done; live_metrics
            # already returns (None, None) for finished runs.
            #
            # Deliberately unlocked. `async with self._lock` is an await, and on
            # a cancelled task that await re-raises CancelledError, so the pop
            # was skipped and the entry leaked: the id then stayed "tracked"
            # forever and submit() refused to ever reuse it. dict.pop is
            # GIL-atomic, and the lock only guards submit's check-then-insert,
            # which a pop can't corrupt (losing the race there just means the
            # new attempt wins, which is what a takeover wants anyway).
            self._states.pop(run_id, None)

    # ---- Caesar invocation (real & dry) ----

    def _invoke_caesar(
        self,
        run_id: str,
        query: str,
        preset_id: str,
        repo_dir: Path,
        state: _RunState,
        mode: str = "new",
        parent_run_id: str | None = None,
        collection_name: str | None = None,
    ) -> dict[str, Any]:
        """Synchronous Caesar invocation. Runs in a worker thread.

        Modes:
          * "new"     — fresh exploration + synthesis (default).
          * "explore" — follow-up: inherit parent's KB collection, seed draft
                        1 with the parent's final answer, then explore() the
                        new query. Parent embeddings are SHA256-deduped.
          * "refine"  — follow-up, synthesis-only: inherit KB + reference
                        draft, skip exploration, call the synthesizer.
        """
        settings = get_settings()
        ensure_caesar_on_path()

        from caesar.caesar_agent import CaesarAgent  # noqa: WPS433
        from rome.config import load_config  # noqa: WPS433  (delayed import)

        yaml_path = resolve_preset_yaml(preset_id, settings)
        if yaml_path is None or not yaml_path.exists():
            raise FileNotFoundError(f"Preset YAML not found for {preset_id!r}: {yaml_path}")
        config = load_config(str(yaml_path))

        # Public-mode bring-your-own-key: thread the per-run key into the
        # LLMHandler config so chat/synthesis calls use it (rome's
        # llm_handler prefers config["api_key"] over the env var). The
        # embedders pick it up via the env window around construction below.
        if state.api_key:
            config.setdefault("LLMHandler", {})["api_key"] = state.api_key

        # Public-mode synthesis-model override: point LLMHandler.model at the
        # user's chosen model. Exploration + KB (ChromaClientManager) configs
        # keep the preset's model, so only the synthesis/default path changes.
        if state.synthesis_model:
            config.setdefault("LLMHandler", {})["model"] = state.synthesis_model

        # Public-mode artifact-length override. synthesis_max_length is already
        # threaded through both the per-draft prompt and the merge prompt, so
        # setting it here is the whole feature -- caesar needs no change. The
        # presets all ship null (unconstrained), which is why an unset run can
        # run to ~8k words.
        if state.output_length:
            config.setdefault("ArtifactSynthesizer", {})["synthesis_max_length"] = (
                state.output_length
            )

        # For follow-up modes, fold parent + follow-up via a small LLM call
        # so Caesar's synthesizer + search-keyword paths see a properly-scoped
        # question instead of a terse follow-up with no conceptual anchor.
        agent_cfg = config.setdefault("CaesarAgent", {})
        is_followup = mode in ("explore", "refine") and bool(parent_run_id)
        parent_artifact_path: Path | None = None
        if is_followup:
            parent_query = _fetch_parent_query_sync(parent_run_id)
            parent_artifact_path = _find_parent_artifact(
                settings.runs_dir / parent_run_id,
            )
            if parent_query:
                # Cache the merge so an auto-restart of an interrupted run
                # reuses the same starting_query (Caesar's checkpoint
                # validator would otherwise log a mismatch).
                cache_path = repo_dir / "__rome__" / "merged_query.txt"
                agent_cfg["starting_query"] = _merge_followup_query(
                    parent_query,
                    query,
                    cache_path=cache_path,
                    api_key=state.api_key,
                )
            else:
                agent_cfg["starting_query"] = query
        else:
            agent_cfg["starting_query"] = query

        # ChromaServerManager is a first-caller-wins singleton; AgentMemory
        # constructs it before our config can land, so pre-warm it here with
        # the web-server-local persist path. Without this AgentMemory's
        # defaults win (port 8000, global ~/.rome/) and our config is silently
        # ignored.
        chroma_dir = settings.caesar_web_data_dir.resolve() / "chroma"
        chroma_dir.mkdir(parents=True, exist_ok=True)
        from rome.kb_server import ChromaServerManager  # noqa: WPS433
        ChromaServerManager.get_instance(config={
            "host": "localhost",
            "port": settings.chroma_port,
            "persist_path": str(chroma_dir),
        })

        # The router resolves `collection_name` already (web_<run_id> for a
        # fresh run, the parent's for a follow-up — transitive, so chains
        # converge on the ancestor's KB).
        client_cfg = config.setdefault("ChromaClientManager", {})
        client_cfg["collection_name"] = collection_name or f"web_{run_id}"

        # Follow-up: seed draft 1 with the parent's final answer.
        if is_followup:
            if parent_artifact_path is not None:
                syn_cfg = config.setdefault("ArtifactSynthesizer", {})
                syn_cfg["synthesis_reference_draft"] = str(parent_artifact_path)
                if parent_query:
                    syn_cfg["synthesis_reference_query"] = parent_query
                logger.info(
                    "Follow-up run %s (mode=%s): inheriting KB %s, "
                    "reference draft %s",
                    run_id, mode, client_cfg["collection_name"],
                    parent_artifact_path.name,
                )
            else:
                logger.warning(
                    "Follow-up run %s (mode=%s): parent %s has no synthesis "
                    "file; proceeding without reference draft",
                    run_id, mode, parent_run_id,
                )

        if mode == "refine":
            _configure_refine_agent_startup(config, repo_dir, parent_artifact_path)

        agent_name = config.get("Agent", {}).get("name", "CaesarAgent")
        # Fail closed: in public mode the server process has no OPENAI_API_KEY
        # in env, so a missing per-run key must raise here rather than silently
        # bill the operator's key (or crash deep in a litellm call).
        if settings.public_mode:
            assert state.api_key is not None, (
                "public_mode requires a per-run api_key before constructing the agent"
            )
        # Only the construction is env-window-scoped: chromadb's embedder,
        # mem0's embedder, and LlamaIndex capture os.environ at __init__.
        # explore()/synthesize_artifact() below stay outside the window.
        with _openai_env_window(state.api_key):
            agent = CaesarAgent(name=agent_name, repository=str(repo_dir), config=config)
        # Expose the agent to the watchdog coroutine so it can read live
        # cost / graph snapshots while explore() runs in this worker thread.
        state.agent = agent
        # If JobPool.shutdown() ran during the heavy CaesarAgent init above,
        # the per-state flag was set but we couldn't propagate to the agent
        # (it didn't exist yet). Refuse to start explore(): the catch-up
        # path used to set agent.shutdown_called=True and let explore() no-op,
        # but quick_explore's submit-time filter silently produces a 0-result
        # "completed" run with a placeholder synthesis string. Raise instead
        # and let the outer handler mark this interrupted for auto-restart.
        if state.shutdown_requested:
            raise RuntimeError(
                "shutdown requested during agent init; refusing to start "
                "explore() to avoid silent no-op completion."
            )
        try:
            if mode == "refine":
                # Synthesis-only follow-up: the synthesizer is constructed in
                # CaesarAgent.__init__ and ready to query the inherited
                # collection immediately. No exploration runs.
                artifact = agent.synthesizer.synthesize_artifact()
            else:
                artifact = agent.explore()
                # quick_explore never writes a graph_iter file (it skips
                # Caesar's iterative checkpointer); force one so the UI has
                # a graph by the time phase 2 starts.
                self._save_final_graph(agent, repo_dir)
            # artifact_synthesis omits cost from metadata; inject it here.
            artifact = artifact or {}
            handler = getattr(agent, "llm_handler", None)
            artifact.setdefault("metadata", {})["total_cost_usd"] = (
                float(getattr(handler, "accumulated_cost", 0.0)) if handler else 0.0
            )
            return artifact
        finally:
            # Snapshot graph size before nulling state.agent: meta's
            # pages_visited (= len(visited_urls)) understates by every node
            # the agent discovered but didn't visit.
            try:
                state.live_graph_node_count = int(agent.graph.number_of_nodes())
            except Exception:  # noqa: BLE001, S110
                pass
            state.agent = None
            try:
                agent.shutdown()
            except Exception:  # noqa: BLE001
                logger.exception("agent.shutdown() failed for run %s", run_id)
            # Last statements in the thread: from here on nothing else touches
            # the run directory, so a restart may safely take it over. Deregister
            # only our own event, so a newer attempt's registration survives.
            state.worker_done.set()
            if self._worker_done.get(run_id) is state.worker_done:
                self._worker_done.pop(run_id, None)

    def _save_final_graph(self, agent: Any, repo_dir: Path) -> None:
        """Save a graph_iter snapshot matching Caesar's filename so the watchdog
        picks it up. quick_explore mode would otherwise never write one (Caesar's
        iterative checkpointer is what produces them, and quick_explore skips it)."""
        try:
            import networkx as nx  # noqa: WPS433

            graph = getattr(agent, "graph", None)
            if graph is None:
                return

            data = nx.node_link_data(graph, edges="edges")
            # Caesar's attr is `current_iteration`; `iteration` was a typo
            # that always read 0 and stamped graph_iter0.json on every run.
            agent_iter = int(getattr(agent, "current_iteration", 0) or 0)
            # Mirror checkpoint.py: top-level + nested both carry these so
            # callers don't have to know which producer wrote the file.
            data["iteration"] = agent_iter
            data["starting_url"] = getattr(agent, "starting_url", None)
            data["graph"] = data.get("graph", {}) or {}
            data["graph"]["iteration"] = agent_iter
            data["graph"].setdefault("starting_url", getattr(agent, "starting_url", None))

            agent_id = getattr(agent, "get_id", lambda: "agent")()
            out = repo_dir / f"{agent_id}.graph_iter{agent_iter}.json"
            # Atomic write: tmp + rename so a watchdog tick can't observe a
            # half-written file mid-flush.
            tmp = out.with_suffix(out.suffix + ".tmp")
            tmp.write_text(json.dumps(data, default=str), encoding="utf-8")
            tmp.replace(out)
            logger.info("Wrote final graph snapshot: %s (%d nodes, iter=%d)",
                        out.name, len(data.get("nodes", [])), agent_iter)
        except Exception:
            logger.exception("Final graph save failed (non-fatal).")

    async def _dry_run(
        self,
        run_id: str,
        query: str,
        preset_id: str,
        repo_dir: Path,
        state: _RunState,
    ) -> dict[str, Any]:
        """Synthetic run for UI development without burning LLM credits."""
        await self._emit(state, "log", message="Running in dry-run mode.")
        # Mirror the real path's live_metrics so /runs listing cards aren't
        # blank during a dry run — the whole point of dry mode is UI dev.
        state.live_cost_usd = 0.0
        state.live_graph_node_count = 0

        for i in range(1, 6):
            await asyncio.sleep(0.6)
            state.live_graph_node_count = i
            # Match the real watchdog's `iteration` payload shape so the UI's
            # phase chip + progress meter aren't half-empty in dry mode.
            await self._emit(state, "iteration", n=i, total=5, phase="quick_explore",
                             url=f"https://example.org/page/{i}", depth=1)

        graph_path = repo_dir / "dryrun.graph_iter5.json"
        nodes = [
            {"id": f"https://example.org/page/{i}", "depth": i,
             "insights": f"Synthetic insight {i}"} for i in range(1, 6)
        ]
        edges = [
            {"source": nodes[i]["id"], "target": nodes[i + 1]["id"], "reason": "next"}
            for i in range(len(nodes) - 1)
        ]
        graph_path.write_text(json.dumps({
            "directed": True, "multigraph": False,
            "graph": {"iteration": 5, "starting_url": nodes[0]["id"]},
            "nodes": nodes, "links": edges,
        }))

        await asyncio.sleep(0.4)
        await self._emit(state, "draft_complete", draft_n=1, abstract="Dry-run draft 1.",
                         artifact_chars=120)
        await asyncio.sleep(0.4)

        artifact = {
            "abstract": f"Synthetic dry-run answer for: {query[:80]}",
            "artifact": (
                "This is a synthetic Caesar dry-run output produced because "
                "CAESAR_DRY_RUN=1 was set. It demonstrates the end-to-end UI "
                "flow without consuming LLM credits. The cited [1] and [2] "
                "markers point at the example sources below."
            ),
            "sources": {"https://example.org/page/1": 1, "https://example.org/page/2": 2},
            "metadata": {
                "pages_visited": 5,
                "insights_collected": 5,
                "sources_cited": 2,
                "synthesis_drafts": 1,
                "total_cost_usd": 0.0,
                "synthesis_mode": "dry-run",
            },
            "artifact_dir": str(repo_dir),
            "artifact_files": [str(graph_path)],
            "num_drafts": 1,
        }
        # Persist a minimal synthesis JSON so /artifacts can read it back.
        synth_path = repo_dir / f"dryrun.synthesis-1.{int(time.time())}.json"
        synth_path.write_text(json.dumps(artifact))
        return artifact

    # ---- Watchdog: tails the artifact dir + console log for SSE events ----
    # Regexes match Caesar's log lines: each gives the UI progress during a
    # phase that's otherwise silent (quick-explore fetch, KB ingest, synthesis).
    QUICK_EXPLORE_RE = re.compile(
        r"\[QUICK_EXPLORE\]\s+Completed\s+(?P<n>\d+)/(?P<total>\d+):\s+(?P<url>\S+)"
    )
    ITERATION_RE = re.compile(r"Iteration\s+(?P<n>\d+)/(?P<total>\d+)")
    KB_INGEST_START_RE = re.compile(
        r"\[QUICK_EXPLORE\]\s+KB ingest started\s+\((?P<total>\d+) results\)"
    )
    KB_INGEST_ADDED_RE = re.compile(
        r"\[QUICK_EXPLORE\]\s+Added insights\s+\(\d+ length\) for (?P<url>\S+)"
    )
    SYNTHESIS_PROGRESS_RE = re.compile(
        r"\[SYNTHESIS\s+DRAFT\s+(?P<draft>\d+)/(?P<total_drafts>\d+)"
        r"\s+ITERATION\s+(?P<iter_n>\d+)/(?P<iter_total>\d+)\]"
    )
    # Phase-3 merge banner Caesar logs at the start of _merge_artifacts.
    # Matches "[MERGING 3 ARTIFACTS]" inside the surrounding ==== rule.
    MERGE_START_RE = re.compile(
        r"\[MERGING\s+(?P<n>\d+)\s+ARTIFACTS\]"
    )
    # Phase-4 image-gen banner: "[IMAGE] Generating N image(s) across M
    # artifact section(s)" — emitted by embed_images_in_artifact at the
    # start of section-aware image gen.
    IMAGE_GEN_START_RE = re.compile(
        r"\[IMAGE\]\s+Generating\s+(?P<n>\d+)\s+image"
    )

    async def _watchdog(self, run_id: str, repo_dir: Path, state: _RunState) -> None:
        # Seed de-dup sets from existing files so a resumed watchdog doesn't
        # replay history. `seen_drafts` is keyed by full path so each new
        # synthesis file (incl. rewrites of the same draft N in a new subdir)
        # fires an event — the frontend dedups by draft_n for the counter.
        seen_graph_iter: set[int] = set()
        seen_drafts: set[str] = set()
        try:
            for f in repo_dir.rglob("*"):
                if not f.is_file():
                    continue
                m = GRAPH_ITER_RE.search(f.name)
                if m:
                    seen_graph_iter.add(int(m.group("n")))
                    continue
                if SYNTHESIS_RE.search(f.name) or MERGED_RE.search(f.name):
                    seen_drafts.add(str(f))
        except OSError:
            pass

        # Per-mode counters so quick_explore and iterative emits don't share
        # state (a quick_explore counter at 146 would otherwise suppress an
        # iterative emit at iter 5/240).
        last_qx_n = 0
        last_iter_n = 0
        last_kb_n = 0
        # Tuple compare so (1,10) > (1,9); string compare on "1:10" vs "1:9"
        # would order them lexically.
        last_synth_key: tuple[int, int] = (0, 0)
        last_cost_emitted = -1.0
        # On resume: skip past existing log + seed kb_total from the historic
        # "KB ingest started (N)" marker so post-resume "Added insights"
        # lines aren't dropped by the kb_total>0 gate.
        log_offset, kb_total = _seed_watchdog_from_log(repo_dir)
        try:
            while not state.finished.is_set():
                await asyncio.sleep(1.5)

                if state.agent is not None:
                    try:
                        cost = float(getattr(state.agent.llm_handler, "accumulated_cost", 0.0))
                    except Exception:  # noqa: BLE001
                        cost = -1.0
                    if cost >= 0:
                        # live_cost_usd always tracks (the /runs listing reads
                        # it for in-flight runs); the SSE event only fires on
                        # ≥ $0.001 delta to avoid spam.
                        state.live_cost_usd = round(cost, 4)
                        if abs(cost - last_cost_emitted) >= 0.001:
                            last_cost_emitted = cost
                            await self._emit(state, "cost_update", cost_usd=round(cost, 4))

                # Filesystem scan — recursive because synthesis lands inside
                # `<id>.synthesis.<timestamp>/`, not the repo root.
                try:
                    files = [p for p in repo_dir.rglob("*") if p.is_file()]
                except FileNotFoundError:
                    files = []

                for f in files:
                    name = f.name

                    m = GRAPH_ITER_RE.search(name)
                    if m:
                        n = int(m.group("n"))
                        if n not in seen_graph_iter:
                            seen_graph_iter.add(n)
                            await self._emit(
                                state,
                                "graph_update",
                                iter=n,
                                file=name,
                            )
                        continue

                    m = SYNTHESIS_RE.search(name) or MERGED_RE.search(name)
                    if m:
                        key = str(f)
                        if key not in seen_drafts:
                            seen_drafts.add(key)
                            draft = m.groupdict().get("draft") or "merged"
                            await self._emit(
                                state,
                                "draft_complete",
                                draft_n=draft,
                                file=name,
                                total_drafts=preset_total_drafts(state.preset_id),
                            )

                # Tail the console log for quick-explore / synthesis progress.
                (
                    last_qx_n,
                    last_iter_n,
                    kb_total,
                    last_kb_n,
                    last_synth_key,
                    log_offset,
                ) = await self._tail_console_log(
                    state,
                    repo_dir,
                    last_qx_n,
                    last_iter_n,
                    kb_total,
                    last_kb_n,
                    last_synth_key,
                    log_offset,
                )

                # Surface the graph node count (not the iteration counter,
                # which advances on revisits) to /runs listing cards.
                if state.agent is not None:
                    try:
                        n_nodes = int(state.agent.graph.number_of_nodes())
                    except Exception:  # noqa: BLE001
                        n_nodes = 0
                    if n_nodes > 0:
                        state.live_graph_node_count = n_nodes

                # Stall detection. `last_activity_mono` is refreshed on every
                # non-ping _emit; if it's stale, the worker has stopped
                # producing anything the watchdog can see (thread wedged in
                # untimed I/O, or died without firing _invoke_caesar's
                # finally). Mark the run failed so the DB stops claiming
                # it's running, and exit — the watchdog owns the flip.
                if (time.monotonic() - state.last_activity_mono) > WATCHDOG_STALL_S:
                    stale_s = int(time.monotonic() - state.last_activity_mono)
                    logger.error(
                        "Run %s stalled (%ds since last event); marking failed.",
                        run_id, stale_s,
                    )
                    await self._mark_failed(
                        run_id, state,
                        f"Worker stalled: no activity for {stale_s}s "
                        f"(threshold {WATCHDOG_STALL_S}s). The subprocess "
                        f"may have died or is stuck in untimed I/O.",
                    )
                    state.finished.set()
                    return
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001
            # Without this log a crashed watchdog would be silently masked
            # by _run's finally swallowing `await watchdog_task`.
            logger.exception("Watchdog crashed for run %s", run_id)

    async def _tail_console_log(
        self,
        state: _RunState,
        repo_dir: Path,
        last_qx_n: int,
        last_iter_n: int,
        kb_total: int,
        last_kb_n: int,
        last_synth_key: tuple[int, int],
        log_offset: int,
    ) -> tuple[int, int, int, int, tuple[int, int], int]:
        """Tail the console log and emit derived events. Returns the updated
        (last_qx_n, last_iter_n, kb_total, last_kb_n, last_synth_key, log_offset)."""
        unchanged = (last_qx_n, last_iter_n, kb_total, last_kb_n, last_synth_key, log_offset)
        try:
            log_files = list((repo_dir / "__rome__").glob("agent_*.console.log"))
        except OSError:
            return unchanged
        if not log_files:
            return unchanged

        log_path = log_files[0]
        try:
            # Rotation: file shrank below our offset → reset all counters
            # (the running last_kb_n / kb_total would otherwise double-count
            # post-rotation "Added insights" against the pre-rotation total).
            if log_path.stat().st_size < log_offset:
                log_offset = 0
                last_kb_n = 0
                kb_total = 0
                last_qx_n = 0
                last_iter_n = 0
                last_synth_key = (0, 0)
            with log_path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(log_offset)
                chunk = f.read()
                new_offset = f.tell()
        except OSError:
            return unchanged

        if not chunk:
            return unchanged

        # ---- quick_explore mode (Fast/Normal presets) ----
        latest_qx_n = last_qx_n
        latest_qx_total: int | None = None
        latest_qx_url: str | None = None
        for m in self.QUICK_EXPLORE_RE.finditer(chunk):
            n = int(m.group("n"))
            if n > latest_qx_n:
                latest_qx_n, latest_qx_total, latest_qx_url = (
                    n, int(m.group("total")), m.group("url")
                )
        if latest_qx_n > last_qx_n and latest_qx_total is not None:
            await self._emit(
                state, "iteration",
                n=latest_qx_n, total=latest_qx_total, url=latest_qx_url, depth=1,
                phase="quick_explore",
            )
            last_qx_n = latest_qx_n

        # ---- iterative mode (Deep run preset) — independent counter ----
        latest_iter_n = last_iter_n
        latest_iter_total: int | None = None
        for m in self.ITERATION_RE.finditer(chunk):
            n = int(m.group("n"))
            if n > latest_iter_n:
                latest_iter_n, latest_iter_total = n, int(m.group("total"))
        if latest_iter_n > last_iter_n and latest_iter_total is not None:
            await self._emit(
                state, "iteration",
                n=latest_iter_n, total=latest_iter_total, depth=1, phase="explore",
            )
            last_iter_n = latest_iter_n

        # ---- KB ingest (between fetch loop and synthesis) ----
        m_start = self.KB_INGEST_START_RE.search(chunk)
        if m_start:
            kb_total = int(m_start.group("total"))
        new_kb_n = last_kb_n + sum(1 for _ in self.KB_INGEST_ADDED_RE.finditer(chunk))
        if new_kb_n > last_kb_n and kb_total > 0:
            await self._emit(state, "iteration", n=new_kb_n, total=kb_total, phase="kb_ingest")

        # ---- Synthesis progress (Phase 2) ----
        new_synth_key = last_synth_key
        synth_payload: dict[str, Any] | None = None
        for m in self.SYNTHESIS_PROGRESS_RE.finditer(chunk):
            g = m.groupdict()
            key: tuple[int, int] = (int(g["draft"]), int(g["iter_n"]))
            if key > new_synth_key:
                new_synth_key = key
                synth_payload = {
                    "phase": "synthesis",
                    "draft": int(g["draft"]),
                    "total_drafts": int(g["total_drafts"]),
                    "iter_n": int(g["iter_n"]),
                    "iter_total": int(g["iter_total"]),
                }
        if synth_payload is not None:
            await self._emit(state, "synthesis_progress", **synth_payload)

        # ---- Merge phase (Phase 3) ----
        # The merge is a single high-reasoning LLM call (plus optional
        # clarify pass) with no per-iteration logging. Emit one event so
        # the UI can show "Merging drafts…" instead of looking stuck on
        # the last draft's step counter for 30-60s. Encoded as a
        # synthesis_progress payload with phase="merge" so the existing
        # consumer machinery handles it; the UI branches on the phase.
        m_merge = self.MERGE_START_RE.search(chunk)
        if m_merge:
            merge_n = int(m_merge.group("n"))
            # Bump new_synth_key past the highest draft we've seen so a
            # later iteration scan doesn't accidentally suppress this.
            new_synth_key = (max(new_synth_key[0], merge_n) + 1, 0)
            await self._emit(
                state,
                "synthesis_progress",
                phase="merge",
                draft=merge_n,
                total_drafts=merge_n,
                iter_n=1,
                iter_total=1,
            )

        # ---- Image-gen phase (Phase 4) ----
        # Emit phase="image_gen" so the UI shows "Generating images…"
        # instead of staying on "merging". The N from the banner becomes
        # the total (1 per fast/normal, 3 deeper, 5 deepest).
        m_image = self.IMAGE_GEN_START_RE.search(chunk)
        if m_image:
            img_n = int(m_image.group("n"))
            new_synth_key = (max(new_synth_key[0], 99), 0)
            await self._emit(
                state,
                "synthesis_progress",
                phase="image_gen",
                draft=img_n,
                total_drafts=img_n,
                iter_n=1,
                iter_total=1,
            )

        return last_qx_n, last_iter_n, kb_total, new_kb_n, new_synth_key, new_offset

    # ---- DB writes & event fan-out ----

    async def _emit(self, state: _RunState, event: str, **payload: Any) -> None:
        """Persist an event to SQLite (for replay) then fan it out to the SSE
        queue with the autoincrement id attached so clients can dedup the
        replay-then-tail seam."""
        if event not in EVENT_TYPES and event != "ping":
            logger.warning("Unknown event type %r — emitting anyway.", event)

        ts = datetime.now(timezone.utc)
        event_id: int | None = None

        if event != "ping":
            state.last_activity_mono = time.monotonic()
            try:
                async with SessionLocal() as session:
                    row = RunEvent(
                        run_id=state.run_id,
                        timestamp=ts,
                        event=event,
                        payload=json.dumps(payload, default=str),
                    )
                    session.add(row)
                    await session.commit()
                    event_id = row.id
            except Exception:
                logger.exception("Failed to persist event for run %s", state.run_id)

        body = {
            "id": event_id,
            "event": event,
            "payload": payload,
            "timestamp": ts.isoformat(),
        }

        try:
            state.queue.put_nowait(body)
        except asyncio.QueueFull:
            # Stuck consumer — drop the oldest and retry once.
            try:
                state.queue.get_nowait()
                state.queue.put_nowait(body)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    async def _update_status(
        self,
        run_id: str,
        *,
        status: RunStatus | None = None,
        **fields: Any,
    ) -> None:
        # `started_at_if_null` writes only if the column is currently NULL.
        started_at_if_null = fields.pop("started_at_if_null", None)
        values: dict[str, Any] = dict(fields)
        if status is not None:
            values["status"] = status.value
            # Delete-on-finish: a terminal run can't be resumed, so drop its
            # stored key immediately. Interrupted keeps it so the next boot can
            # resume. Belt-and-suspenders startup purge in main.py covers a
            # crash that skipped this.
            if status in (RunStatus.completed, RunStatus.failed):
                values["run_api_key"] = None
        async with SessionLocal() as session:
            await session.execute(
                update(Run).where(Run.id == run_id).values(**values)
            )
            if started_at_if_null is not None:
                await session.execute(
                    update(Run)
                    .where(Run.id == run_id, Run.started_at.is_(None))
                    .values(started_at=started_at_if_null)
                )
            await session.commit()

    async def _mark_failed(
        self,
        run_id: str,
        state: _RunState,
        message: str,
        traceback_text: str | None = None,
    ) -> None:
        await self._update_status(
            run_id,
            status=RunStatus.failed,
            finished_at=datetime.now(timezone.utc),
            error_message=message,
        )
        await self._emit(state, "error", message=message, traceback=traceback_text)


# Module-level singleton: the routers import and use this directly.
job_pool = JobPool()


# Live checkpoint = `{repo}/__rome__/{agent_id}.checkpoint.json`. On terminal
# status the file is renamed to `*.checkpoint.final.json` so the lifespan
# scanner ignores it; rename back to retry.
LIVE_CHECKPOINT_GLOB = "*.checkpoint.json"
ARCHIVED_CHECKPOINT_SUFFIX = ".checkpoint.final.json"
ARCHIVED_CHECKPOINT_GLOB = f"*{ARCHIVED_CHECKPOINT_SUFFIX}"


def has_checkpoint(repo_dir: Path) -> bool:
    """True iff a non-empty live (un-archived) checkpoint exists in repo_dir."""
    try:
        return any(
            p.is_file() and p.stat().st_size > 0
            for p in (repo_dir / "__rome__").glob(LIVE_CHECKPOINT_GLOB)
        )
    except OSError:
        return False


def _seed_watchdog_from_log(repo_dir: Path) -> tuple[int, int]:
    """Return (log_offset, kb_total) for seeding the watchdog on resume.

    `log_offset` is the file's byte size (from stat) so it matches the
    `f.tell()` offsets used in `_tail_console_log`. Using `len(content)`
    or `len(content.encode())` after a text-mode read can drift from the
    real byte offset because of universal-newlines translation and
    error="replace" substitutions.

    `kb_total` is extracted from the most recent "[QUICK_EXPLORE]
    KB ingest started (N results)" line so post-resume "Added insights"
    lines pass the `kb_total > 0` emit gate.
    """
    try:
        log_files = list((repo_dir / "__rome__").glob("agent_*.console.log"))
        if not log_files:
            return 0, 0
        path = log_files[0]
        size = path.stat().st_size
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0, 0
    kb_total = 0
    for m in JobPool.KB_INGEST_START_RE.finditer(content):
        kb_total = int(m.group("total"))
    return size, kb_total


def archive_checkpoint(repo_dir: Path) -> None:
    """Rename live checkpoint files to the archived form. Idempotent; overwrites
    any existing archived target."""
    try:
        for live in (repo_dir / "__rome__").glob(LIVE_CHECKPOINT_GLOB):
            agent_id = live.name[: -len(".checkpoint.json")]
            try:
                live.replace(repo_dir / "__rome__" / f"{agent_id}{ARCHIVED_CHECKPOINT_SUFFIX}")
            except OSError:
                logger.exception("Failed to archive checkpoint: %s", live)
    except OSError:
        pass


def unarchive_checkpoint(repo_dir: Path) -> bool:
    """Inverse of :func:`archive_checkpoint`: restore the live checkpoint name.

    Returns whether a usable live checkpoint exists afterwards, which is exactly
    the ``resuming=`` flag a retry should hand to :meth:`JobPool.submit` — True
    means Caesar picks up where it stopped, False means it starts over.

    A live checkpoint already present wins: it is at least as fresh as the
    archived copy, so we never let an older archive clobber it.
    """
    if has_checkpoint(repo_dir):
        return True
    rome_dir = repo_dir / "__rome__"
    try:
        for archived in rome_dir.glob(ARCHIVED_CHECKPOINT_GLOB):
            agent_id = archived.name[: -len(ARCHIVED_CHECKPOINT_SUFFIX)]
            try:
                archived.replace(rome_dir / f"{agent_id}.checkpoint.json")
            except OSError:
                logger.exception("Failed to un-archive checkpoint: %s", archived)
    except OSError:
        pass
    return has_checkpoint(repo_dir)
