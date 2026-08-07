"""Pytest fixtures: an isolated FastAPI app per test with dry-run forced on."""
from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


def _drop_app_modules() -> None:
    """Forget every imported `app.*` module so the next import rebuilds them
    against the current environment."""
    for mod in list(sys.modules):
        if mod == "app" or mod.startswith("app."):
            sys.modules.pop(mod)


@pytest.fixture(autouse=True)
def isolated_data_dir(monkeypatch, tmp_path: Path):
    """Each test gets its own SQLite + runs/ subtree."""
    data_dir = tmp_path / "data"
    monkeypatch.setenv("CAESAR_WEB_DATA_DIR", str(data_dir))
    monkeypatch.setenv("CAESAR_DRY_RUN", "1")
    monkeypatch.setenv("CAESAR_MAX_CONCURRENT", "5")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    # Also drop them on the way IN, not just on the way out. A test module with a
    # module-level `from app... import x` is imported during collection, before
    # any fixture runs -- and app.db builds its engine at import time from
    # get_settings().db_path. Such an import therefore binds the engine to the
    # DEFAULT data dir (api/data), which is a real deployment's database: tests
    # then read and write live data, and test_admin_mode's wipe-all deletes every
    # row in it. Dropping first guarantees the engine is built after the env
    # above is in place, whatever collection did.
    _drop_app_modules()

    # Clear lru_cache so settings rebuild from env.
    from app.config import get_settings  # noqa: WPS433
    get_settings.cache_clear()

    yield

    # Drop the imported app modules so the next test gets a fresh DB engine.
    _drop_app_modules()
    if data_dir.exists():
        shutil.rmtree(data_dir, ignore_errors=True)


@pytest_asyncio.fixture
async def client(tmp_path: Path):
    """An httpx AsyncClient mounted on the FastAPI app."""
    from app.db import engine  # noqa: WPS433
    from app.main import app  # noqa: WPS433

    # Refuse to run against anything but this test's sandbox. Without this, an
    # engine bound to the default api/data dir just silently works -- and the
    # suite reads, writes, and (via the admin wipe-all test) DELETES a real
    # deployment's runs. Cheap assertion, catastrophic failure mode.
    assert str(tmp_path) in str(engine.url), (
        f"test DB engine escaped its sandbox: {engine.url} (expected under {tmp_path}). "
        "Usually caused by a test module importing `app.*` at module level, which "
        "runs during collection, before the isolated_data_dir fixture."
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Trigger the lifespan so init_db runs.
        async with app.router.lifespan_context(app):
            yield ac


def make_query() -> str:
    return f"Synthetic test query {uuid.uuid4().hex[:6]}"
