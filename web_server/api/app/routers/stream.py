"""SSE endpoint for live run progress.

Behavior:
  1. Replay every persisted event from SQLite first (so a client opening the
     page mid-run / reconnecting after a blip immediately catches up).
  2. Then attach to the in-memory queue and forward live events.

If the run has already finished by the time the client connects, the live
attach yields nothing and the connection cleanly closes after the replay.
"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from ..db import get_session
from ..deps import current_owner, get_owned_run, is_admin
from ..job_runner import job_pool
from ..models import RunEvent

logger = logging.getLogger("caesar.web.stream")
router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("/{run_id}/stream")
async def stream_run(
    run_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
    owner: str | None = Depends(current_owner),
    admin: bool = Depends(is_admin),
) -> EventSourceResponse:
    # Owner-checked lookup (404 on miss or cross-owner) before any replay so a
    # caller cannot tail another tenant's run events. Admin bypasses the check.
    await get_owned_run(run_id, owner, session, admin=admin)

    # Snapshot persisted events for replay.
    result = await session.execute(
        select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.id)
    )
    persisted = list(result.scalars().all())

    last_replayed_id = persisted[-1].id if persisted else 0

    def _frame(envelope: dict) -> dict:
        # Build a sse-starlette frame. `id` enables Last-Event-ID reconnects.
        return {
            "id": str(envelope["id"]) if envelope.get("id") is not None else None,
            "event": envelope["event"],
            "data": json.dumps(envelope, default=str),
        }

    async def event_gen() -> AsyncIterator[dict]:
        # 1. Replay every persisted event so a client connecting mid-run
        #    catches up to the current state immediately.
        for ev in persisted:
            if await request.is_disconnected():
                return
            try:
                payload = json.loads(ev.payload) if ev.payload else {}
            except json.JSONDecodeError:
                payload = {"_raw": ev.payload}
            yield _frame(
                {
                    "id": ev.id,
                    "event": ev.event,
                    "payload": payload,
                    "timestamp": ev.timestamp.isoformat(),
                }
            )

        # 2. Live tail. The job pool's queue may still hold events that we
        #    just replayed (race: emit() persists then queues, but our
        #    replay snapshot was already past those rows). De-duplicate
        #    by event id — replay and live share the same id sequence.
        async for live in job_pool.stream(run_id):
            if await request.is_disconnected():
                return
            ev_id = live.get("id")
            if isinstance(ev_id, int) and ev_id <= last_replayed_id:
                continue
            yield _frame(live)

        # If the run already finished and there's no live tail, just exit.

    # Defaults to text/event-stream with proper headers for proxies.
    return EventSourceResponse(
        event_gen(),
        # 2s ping (was 5s) — Tailscale Funnel's reverse-proxy layer
        # silently drops streaming connections at 10-40s irrespective of
        # actual byte flow (tailscale/tailscale#18827). 5s heartbeat
        # still left occasional gaps where two consecutive heartbeats
        # could miss the proxy's idle window. 2s gives 5-20 heartbeats
        # per drop window — much safer.
        ping=2,
        # send_timeout=15 (was 60) — sse-starlette's per-frame send
        # timeout wraps every `await send(...)` in anyio.move_on_after(N).
        # 60s is way longer than the proxy's actual idle drop window
        # (10-40s), so when the proxy half-closed the socket we'd hold
        # the response coroutine open up to a minute before finally
        # detecting the dead pipe — meanwhile the browser had already
        # reconnected and the "Stream disconnected" badge flashed.
        # 15s surfaces dead sockets fast, before they accumulate.
        send_timeout=15,
        # Disable response buffering for nginx-like proxies.
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
