"""SQLAlchemy engine and session for the Caesar web server."""
from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .config import get_settings
from .models import Base

logger = logging.getLogger("caesar.web.db")


def _engine_url() -> str:
    return f"sqlite+aiosqlite:///{get_settings().db_path}"


def _restrict_db_permissions() -> None:
    """Make the SQLite files owner-only (0600).

    In public mode `runs.run_api_key` holds a user's OpenAI key in plaintext for
    the life of the run (it is deleted on finish so a restart can resume). The
    default 0644 from SQLite's create would let any local account read a live
    key straight out of the file, so tighten the DB and its WAL/SHM sidecars,
    which hold recently-written pages. Best-effort: a filesystem that rejects
    chmod should not stop the server from booting.
    """
    db_path = Path(get_settings().db_path)
    for path in (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")):
        try:
            if path.exists():
                os.chmod(path, 0o600)
        except OSError as exc:
            logger.warning("Could not restrict permissions on %s: %s", path, exc)


# A single engine for the lifetime of the process. SQLite handles concurrent
# readers fine; writes are serialised by the bus.
engine = create_async_engine(
    _engine_url(),
    echo=False,
    future=True,
    pool_pre_ping=True,
    # SQLite-specific: allow reuse across the asyncio event loop's threads.
    connect_args={"check_same_thread": False},
)


# Per-connection PRAGMA setup. Runs on every new connection (pool prime
# + pool refresh).
#   foreign_keys=ON  — SQLite parses FK clauses but doesn't enforce them by
#     default; without this, runs.parent_run_id ON DELETE SET NULL and
#     run_events.run_id ON DELETE CASCADE are dead metadata.
#   journal_mode=WAL — readers don't block writers. Several watchdogs (one
#     per concurrent run) and the SSE replay paths read while _emit /
#     _update_status write; without WAL they serialise behind each write's
#     rollback-journal lock and tail latency climbs sharply.
#   synchronous=NORMAL — pairs with WAL: fsync only on checkpoint, not per
#     commit. Crash-safe for SQLite's WAL.
@event.listens_for(engine.sync_engine, "connect")
def _sqlite_connection_setup(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
    finally:
        cursor.close()

SessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


# Idempotent additive column migrations for tables that pre-date a model
# change. SQLAlchemy's create_all() adds new tables but not new columns, so
# each tuple here ALTERs the existing table only when the column is absent.
_ADDITIVE_COLUMNS: list[tuple[str, str, str]] = [
    # (table, column, "<col_def>" appended after "ADD COLUMN <name> ")
    ("runs", "parent_run_id", "VARCHAR(36) REFERENCES runs(id) ON DELETE SET NULL"),
    ("runs", "mode", "VARCHAR(16) NOT NULL DEFAULT 'new'"),
    ("runs", "collection_name", "VARCHAR(128)"),
    ("runs", "owner_id", "VARCHAR(64)"),
    ("runs", "synthesis_model", "VARCHAR(64)"),
    ("runs", "run_api_key", "TEXT"),
    ("runs", "elapsed_prior_s", "FLOAT NOT NULL DEFAULT 0"),
]


_COLUMN_RENAMES: list[tuple[str, str, str]] = [
    # (table, old_column, new_column)
    ("runs", "total_iterations", "graph_node_count"),
    # Dropped Fernet encryption; the per-run key is now stored plaintext and
    # deleted on finish. Rename the old (encrypted) column in place.
    ("runs", "run_api_key_enc", "run_api_key"),
]


# Additive indexes for columns added via _ADDITIVE_COLUMNS (create_all only
# creates indexes for newly-created tables; ALTER doesn't auto-index).
_ADDITIVE_INDEXES: list[tuple[str, str, str]] = [
    # (index_name, table, column)
    ("ix_runs_parent_run_id", "runs", "parent_run_id"),
    ("ix_runs_owner_id", "runs", "owner_id"),
]

# Idempotent value migrations for persisted identifiers. These run on startup
# only when old values exist, avoiding permanent compatibility aliases.
_VALUE_RENAMES: list[tuple[str, str, str, str]] = [
    # (table, column, old_value, new_value)
    ("runs", "preset", "deep", "deeper"),
]


async def init_db() -> None:
    """Create tables if they don't yet exist, then run additive column +
    index migrations for any model fields added since the table was first
    created. Persisted value renames run only when old values exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        for table, old_col, new_col in _COLUMN_RENAMES:
            rows = (await conn.execute(text(f"PRAGMA table_info({table})"))).all()
            existing = {r[1] for r in rows}  # PRAGMA: cid, name, type, ...
            if old_col in existing and new_col not in existing:
                await conn.execute(
                    text(f"ALTER TABLE {table} RENAME COLUMN {old_col} TO {new_col}")
                )
        for table, col, definition in _ADDITIVE_COLUMNS:
            rows = (await conn.execute(text(f"PRAGMA table_info({table})"))).all()
            existing = {r[1] for r in rows}  # PRAGMA: cid, name, type, ...
            if col not in existing:
                await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {definition}"))
        for index_name, table, col in _ADDITIVE_INDEXES:
            await conn.execute(
                text(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table}({col})")
            )
        for table, col, old_value, new_value in _VALUE_RENAMES:
            needed = (await conn.execute(
                text(f"SELECT 1 FROM {table} WHERE {col} = :old LIMIT 1"),
                {"old": old_value},
            )).first()
            if not needed:
                continue
            await conn.execute(
                text(f"UPDATE {table} SET {col} = :new WHERE {col} = :old"),
                {"new": new_value, "old": old_value},
            )
    # After create_all, so a first-boot database is tightened too.
    _restrict_db_permissions()


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Async context manager that opens a session, commits on success, rolls
    back on error, and always closes."""
    session = SessionLocal()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency that yields a per-request session."""
    async with session_scope() as session:
        yield session
