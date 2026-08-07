"""The /version commit sha: build-time stamp wins, git is the fallback.

Regression cover for a footer that silently lost its commit link on every
container deploy. .dockerignore keeps .git out of the build context, so the git
fallback exits 128 inside the image and the UI's `{commit && ...}` rendered
nothing -- while the version, read from pyproject.toml, kept showing. Nothing
locally reproduced that, because a source checkout always has a .git to read.
"""
from __future__ import annotations

import pytest

# NB: `app` is imported inside each test, never at module level. Module-level
# imports run at collection, before conftest's isolated_data_dir fixture, and
# app.db builds its engine at import time from get_settings().db_path -- so a
# top-level `from app.main import ...` binds the engine to the DEFAULT data dir,
# i.e. the real deployment database. conftest now defends against this too, but
# don't reintroduce it here.


def test_env_stamp_is_preferred_and_shortened(monkeypatch):
    from app.main import _read_commit_sha

    monkeypatch.setenv("CAESAR_COMMIT_SHA", "f7d4cebce88aa8b24c35115d21169c894f1cee26")
    # Shortened to 8 for display, matching the git fallback's --short=8.
    assert _read_commit_sha() == "f7d4cebc"


def test_env_stamp_normalizes_case_and_whitespace(monkeypatch):
    from app.main import _read_commit_sha

    monkeypatch.setenv("CAESAR_COMMIT_SHA", "  F7D4CEBCE88A  ")
    assert _read_commit_sha() == "f7d4cebc"


@pytest.mark.parametrize(
    "value",
    [
        "",  # unstamped local `docker build` -- ARG defaults to empty
        "   ",
        "not-a-sha",
        "zzzzzzzz",
        "abc",  # too short to be a usable prefix
        "$CAESAR_COMMIT_SHA",  # an un-substituted build-arg
        "${{ github.sha }}",  # a workflow templating mistake
    ],
)
def test_junk_stamp_falls_back_to_git(monkeypatch, value):
    """A mangled stamp must not become a dead GitHub link in the footer."""
    from app.main import _read_commit_sha

    monkeypatch.setenv("CAESAR_COMMIT_SHA", value)
    result = _read_commit_sha()
    # Tests run from a source checkout, so the git fallback answers. Either way
    # the junk itself must never be returned.
    assert result != value.strip()
    if result is not None:
        assert len(result) == 8 and all(c in "0123456789abcdef" for c in result)


def test_git_fallback_used_when_unstamped(monkeypatch):
    from app.main import _read_commit_sha

    monkeypatch.delenv("CAESAR_COMMIT_SHA", raising=False)
    result = _read_commit_sha()
    assert result is not None, "source checkout should resolve a sha via git"
    assert len(result) == 8


@pytest.mark.asyncio
async def test_version_endpoint_exposes_commit(client):
    r = await client.get("/version")
    assert r.status_code == 200
    body = r.json()
    assert "commit" in body
    # Captured at import, so this asserts the shape the UI keys off rather than
    # a specific sha: layout.tsx renders the hash only when it is truthy.
    assert body["commit"] is None or len(body["commit"]) == 8
