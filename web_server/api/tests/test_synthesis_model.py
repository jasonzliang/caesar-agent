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
    # The newest model of each OpenAI cost tier, cheapest first. Older tier
    # members (gpt-5.6-sol, gpt-6-sol...), non-OpenAI providers, realtime and
    # GPT-4.x are not offered.
    assert ids == ["gpt-6-luna", "gpt-6.1-terra", "gpt-6.2-sol", "gpt-6.2-astra"]
    # Single source of truth: the endpoint exactly mirrors (order included)
    # LLMHandler.synthesis_models() — no separate web-side list.
    from app.config import ensure_caesar_on_path

    ensure_caesar_on_path()
    from rome.llm_handler import LLMHandler

    assert ids == LLMHandler.synthesis_models()
    # Pricing rides along for the dropdown's cost hint.
    sol = next(m for m in r.json() if m["id"] == "gpt-6.2-sol")
    assert sol["input_per_mtok"] == 2.0 and sol["output_per_mtok"] == 10.0


def test_synthesis_models_track_the_newest_of_each_tier(monkeypatch):
    from app.config import ensure_caesar_on_path

    ensure_caesar_on_path()
    from rome.llm_handler import LLMHandler

    pricing = {**LLMHandler.MODEL_PRICING, "gpt-6.10-sol": {"input": 1.0, "output": 1.0}}
    monkeypatch.setattr(LLMHandler, "MODEL_PRICING", pricing)
    assert LLMHandler.synthesis_models() == [
        "gpt-6-luna", "gpt-6.1-terra", "gpt-6.10-sol", "gpt-6.2-astra"]


@pytest.mark.asyncio
async def test_new_models_priced_and_sized(client):
    """The GPT-5.6 and GPT-6 tiers are fully described in rome's LLMHandler tables."""
    from app.config import ensure_caesar_on_path

    ensure_caesar_on_path()
    from rome.llm_handler import LLMHandler

    # Per developers.openai.com/api/docs/pricing, re-verified 2026-08-03 and
    # matching litellm's live map. luna/terra were previously pinned here at
    # $1.00/$6.00 and $2.50/$15.00, which is 5x and 1.25x over the real prices,
    # so this test was holding the error in place: correcting the table failed
    # it. Do not "restore" these without checking the vendor page. GPT-6 and
    # gpt-5.6-sol ($5/$30 before) re-verified 2026-11-23.
    expected = {
        "gpt-6-luna": (0.1, 0.5),
        "gpt-6.1-terra": (1.0, 5.0),
        "gpt-6.2-sol": (2.0, 10.0),
        "gpt-6.2-astra": (8.0, 40.0),
        "gpt-5.6-sol": (4.0, 20.0),
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
        # gpt-5*/gpt-6* prefixes already make these reasoning models; assert explicitly.
        assert LLMHandler.is_reasoning_model(mid) is True


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
            "synthesis_model": "gpt-6.2-sol",
        },
    )
    assert r.status_code == 201
    run_id = r.json()["id"]
    # The override is persisted and surfaces on the run detail's model badge
    # (which otherwise reads the preset's default model).
    detail = await client.get(f"/runs/{run_id}", cookies=OWNER_COOKIE)
    assert detail.status_code == 200
    assert detail.json()["llm_model"] == "gpt-6.2-sol"


@pytest.mark.asyncio
async def test_public_submit_accepts_a_retired_dropdown_choice(client, monkeypatch):
    """A model saved in a browser cookie before the dropdown moved on still runs."""
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


@pytest.mark.asyncio
async def test_nonpublic_ignores_synthesis_model(client):
    """Outside public mode the override is ignored, not validated (no 400)."""
    r = await client.post(
        "/runs",
        json={"query": make_query(), "preset": "fast", "synthesis_model": "bogus"},
    )
    assert r.status_code == 201
