"""SQLAlchemy ORM models for runs and run events."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class RunStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    # Transient state for a run whose synthesis exited via cooperative
    # shutdown with fewer drafts than the preset planned. The startup
    # lifespan picks these up and resubmits them on the next boot, so
    # users normally only see this status during the restart gap.
    interrupted = "interrupted"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RunMode(str, Enum):
    new = "new"          # ordinary fresh exploration (default)
    explore = "explore"  # follow-up: new exploration, inherit parent's KB + answer
    refine = "refine"    # follow-up: synthesis-only over parent's KB + answer


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    preset: Mapped[str] = mapped_column(String(32), nullable=False)
    # Public-mode per-run synthesis (LLMHandler) model override; NULL = the
    # preset's default model. Not a secret, so unlike the api_key it is
    # persisted — the run page reads it for the model badge.
    synthesis_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Public-mode per-run OpenAI key, stored so a restart can auto-resume the
    # run from its checkpoint. Cleared the instant the run reaches a terminal
    # state (see job_runner._update_status), plus a startup purge + TTL — so it
    # is at rest only while the run is in flight. Never serialized: it is absent
    # from every response schema.
    run_api_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=RunStatus.queued.value)

    # Follow-up linkage. parent_run_id points at the run we forked from;
    # mode controls how the follow-up uses the parent (KB inherit, synthesis-
    # only, etc.). Both are null/"new" for a normal homepage submission.
    parent_run_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("runs.id", ondelete="SET NULL"), nullable=True
    )
    mode: Mapped[str] = mapped_column(String(16), nullable=False, default=RunMode.new.value)
    # Chroma collection actually written/read by this run. For a fresh run
    # this is `web_<id>`; a follow-up inherits the parent's value so a chain
    # of follow-ups all converge on the original ancestor's collection.
    # Persisted because the worker thread needs to resolve it without
    # walking the parent chain at submit time.
    collection_name: Mapped[str | None] = mapped_column(String(128), nullable=True)

    repository: Mapped[str | None] = mapped_column(Text, nullable=True)
    agent_name: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Tenant identity in public mode: the opaque caesar_id cookie value.
    # NULL for legacy/shared/password rows; an equality filter against NULL
    # degenerates to today's single-tenant behavior.
    owner_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Lifecycle timestamps
    created_at: Mapped[datetime] = mapped_column(default=_utcnow, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
    # Seconds this run burned in *earlier* attempts. A restart resets started_at
    # (the new attempt is what the progress counters describe), so without this
    # the elapsed figure would silently drop every restart's work: 3h of real
    # compute reading as "4m". Banked by the retry endpoint, never by a resume,
    # whose attempt is genuinely still the same one.
    elapsed_prior_s: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0, server_default="0"
    )

    # Results / metrics
    total_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    graph_node_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    events: Mapped[list[RunEvent]] = relationship(
        back_populates="run",
        order_by="RunEvent.id",
        cascade="all, delete-orphan",
    )


class RunEvent(Base):
    __tablename__ = "run_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(default=_utcnow, nullable=False)
    # event type: iteration | graph_update | draft_complete | done | error | log
    event: Mapped[str] = mapped_column(String(32), nullable=False)
    # JSON-encoded payload
    payload: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    run: Mapped[Run] = relationship(back_populates="events")


# Index frequently-filtered columns
Index("ix_runs_status", Run.status)
Index("ix_runs_created_at", Run.created_at)
Index("ix_runs_parent_run_id", Run.parent_run_id)
Index("ix_runs_owner_id", Run.owner_id)
Index("ix_run_events_run_id", RunEvent.run_id)
