"""Public-mode synthesis-model override: /models endpoint, validation, gating."""
from __future__ import annotations

import uuid

import pytest

from .conftest import make_query

VALID_KEY = "sk-" + "x" * 24
# Public mode requires the per-browser tenant cookie (normally minted by the
# Next.js middleware); mint a valid-charset one for the ASGI test client.
OWNER_COOKIE = {"caesar_id": uuid.uuid4().hex}


@pytest.mark.asyncio
async def test_models_endpoint_lists_openai_only(client):
    r = await client.get("/models")
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()]
    # Newly-added GPT-5.6 family is present as concrete tiers...
    assert "gpt-5.6-sol" in ids
    # ...but the bare alias is gone entirely: it routed to -sol, so carrying it
    # offered one model under two names and hid which tier was being bought.
    assert "gpt-5.6" not in ids
    assert "gpt-5.6-terra" in ids
    assert "gpt-5.6-luna" in ids
    # ...alongside an existing default and older models.
    assert "gpt-5.4-mini" in ids
    # Non-OpenAI providers, realtime (audio), and the older GPT-4.x line
    # (gpt-4o / gpt-4.1) are excluded.
    assert not any(i.startswith(("claude-", "gemini-", "gpt-4")) for i in ids)
    assert not any("realtime" in i for i in ids)
    # o-series models are offered alongside GPT-5.x.
    assert "o3" in ids
    # Single source of truth: the endpoint exactly mirrors (order included)
    # LLMHandler.synthesis_models() — no separate web-side list.
    from app.config import ensure_caesar_on_path

    ensure_caesar_on_path()
    from rome.llm_handler import LLMHandler

    assert ids == LLMHandler.synthesis_models()
    # Pricing rides along for the dropdown's cost hint.
    sol = next(m for m in r.json() if m["id"] == "gpt-5.6-sol")
    assert sol["input_per_mtok"] == 5.0 and sol["output_per_mtok"] == 30.0


@pytest.mark.asyncio
async def test_new_models_priced_and_sized(client):
    """The GPT-5.6 family is fully described in rome's LLMHandler tables."""
    from app.config import ensure_caesar_on_path

    ensure_caesar_on_path()
    from rome.llm_handler import LLMHandler

    # Per developers.openai.com/api/docs/pricing, re-verified 2026-08-03 and
    # matching litellm's live map. luna/terra were previously pinned here at
    # $1.00/$6.00 and $2.50/$15.00, which is 5x and 1.25x over the real prices,
    # so this test was holding the error in place: correcting the table failed
    # it. Do not "restore" these without checking the vendor page.
    expected = {
        "gpt-5.6-sol": (5.0, 30.0),
        "gpt-5.6-terra": (2.0, 12.0),
        "gpt-5.6-luna": (0.2, 1.2),
    }
    # The alias is absent from every table, not merely unlisted. No run ever
    # recorded it, and litellm's live map prices it if one somehow arrives.
    assert "gpt-5.6" not in LLMHandler.MODEL_PRICING
    assert "gpt-5.6" not in LLMHandler.MODEL_CONTEXT_SIZE
    for mid, (inp, out) in expected.items():
        assert LLMHandler.MODEL_PRICING[mid] == {"input": inp, "output": out}
        assert LLMHandler.MODEL_CONTEXT_SIZE[mid] == 1050000
        # gpt-5* prefix already makes these reasoning models; assert explicitly.
        assert LLMHandler()._is_reasoning_model(mid) is True


@pytest.mark.asyncio
async def test_public_submit_rejects_unknown_model(client, monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "1")
    from app.config import get_settings

    get_settings.cache_clear()
    r = await client.post(
        "/runs",
        cookies=OWNER_COOKIE,
        json={
            "query": make_query(),
            "preset": "fast",
            "api_key": VALID_KEY,
            "synthesis_model": "definitely-not-a-model",
        },
    )
    assert r.status_code == 400
    assert "synthesis model" in r.text.lower()


@pytest.mark.asyncio
async def test_public_submit_accepts_known_model(client, monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "1")
    from app.config import get_settings

    get_settings.cache_clear()
    r = await client.post(
        "/runs",
        cookies=OWNER_COOKIE,
        json={
            "query": make_query(),
            "preset": "fast",
            "api_key": VALID_KEY,
            "synthesis_model": "gpt-5.6-sol",
        },
    )
    assert r.status_code == 201
    run_id = r.json()["id"]
    # The override is persisted and surfaces on the run detail's model badge
    # (which otherwise reads the preset's default model).
    detail = await client.get(f"/runs/{run_id}", cookies=OWNER_COOKIE)
    assert detail.status_code == 200
    assert detail.json()["llm_model"] == "gpt-5.6-sol"


@pytest.mark.asyncio
async def test_nonpublic_ignores_synthesis_model(client):
    """Outside public mode the override is ignored, not validated (no 400)."""
    r = await client.post(
        "/runs",
        json={"query": make_query(), "preset": "fast", "synthesis_model": "bogus"},
    )
    assert r.status_code == 201
