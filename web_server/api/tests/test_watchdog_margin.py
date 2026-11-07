"""Guards for the stall-watchdog fixes:
  1. WATCHDOG_STALL_S must stay strictly above the largest per-call LLM timeout
     (a single full-length synthesis call must not by itself trip the watchdog).
  2. The synthesizer's per-attempt heartbeat must match the watchdog regex, so a
     stacked retry ladder resets the stall clock between attempts.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def _preset_llm_timeouts() -> list[float]:
    preset_dir = Path(__file__).resolve().parents[2] / "config_preset"
    out: list[float] = []
    for y in sorted(preset_dir.glob("*.yaml")):
        data = yaml.safe_load(y.read_text(encoding="utf-8")) or {}
        t = (data.get("LLMHandler") or {}).get("timeout")
        if isinstance(t, (int, float)):
            out.append(float(t))
    return out


def test_watchdog_threshold_exceeds_max_preset_timeout():
    from app.job_runner import WATCHDOG_STALL_S

    timeouts = _preset_llm_timeouts()
    mx = max(timeouts) if timeouts else 0.0
    # The exact regression this fixes: threshold == max call timeout (1200==1200)
    # made a single legitimate synthesis call trip the "1201s" stall.
    assert WATCHDOG_STALL_S > mx, (
        f"WATCHDOG_STALL_S ({WATCHDOG_STALL_S}) must exceed the largest preset "
        f"LLMHandler.timeout ({mx})"
    )
    assert WATCHDOG_STALL_S - mx >= 300, (
        f"want >=300s margin over the max single-call timeout; "
        f"got {WATCHDOG_STALL_S - mx}s"
    )


def test_llm_attempt_heartbeat_matches_watchdog_regex():
    from app.job_runner import JobPool

    # Must match the line artifact_synthesis._llm_call logs per attempt:
    #   "[LABEL] attempt k/N (reasoning_effort=...)"
    for label in ("SYNTHESIS", "MERGE", "CLARIFY", "POST-PROCESS"):
        line = f"[{label}] attempt 1/5 (reasoning_effort=high)"
        assert JobPool.LLM_ATTEMPT_RE.search(line), f"{label} heartbeat not matched"
    # A non-heartbeat synthesis line must NOT match (avoid false resets).
    assert not JobPool.LLM_ATTEMPT_RE.search("[SYNTHESIS] Generating synthesis artifact")


def test_image_failure_lines_match_watchdog_regex():
    from app.job_runner import EVENT_TYPES, JobPool

    assert "image_failed" in EVENT_TYPES
    reason = ("the OpenAI organization behind this API key is not verified for image "
              "models (BadRequestError: OpenAIException - Your organization must be verified)")
    # The three producers: image_generator (generation) and artifact_synthesis
    # (embedding, import). Each line is the plain console-log format.
    for prefix in ("[IMAGE] Image generation failed: ",
                   "[IMAGE] Image embedding failed: ",
                   "[IMAGE] Cannot import image_generator: "):
        line = f"05:53:15 - Agent.t1 - ERROR - {prefix}{reason}"
        m = JobPool.IMAGE_FAIL_RE.search(line)
        assert m, line
        assert m.group("reason") == reason
    # The start banner and a successful write must not read as failures.
    for line in ("[IMAGE] Generating 1 image(s) across 1 artifact section(s)",
                 "Image 1/1 written to /x/y.webp (via images.generate, refs_used=0)",
                 "[IMAGE] Embedded image(s) and re-saved synthesis"):
        assert not JobPool.IMAGE_FAIL_RE.search(line), line


@pytest.mark.asyncio
async def test_tail_emits_one_scrubbed_image_failed_event(tmp_path):
    from app.job_runner import JobPool, _RunState

    pool = JobPool()
    emitted: list[tuple[str, dict]] = []

    async def _record(state, event, **payload):
        emitted.append((event, payload))

    pool._emit = _record  # type: ignore[method-assign]
    key = "sk-live-ABCDEFGHIJKLMNOP"
    state = _RunState("run-1", "fast", api_key=key)
    rome_dir = tmp_path / "__rome__"
    rome_dir.mkdir()
    log = rome_dir / "agent_CaesarExplorer.console.log"
    line = ("05:53:15 - Agent.t1 - ERROR - [IMAGE] Image generation failed: "
            f"OpenAI rejected the API key (AuthenticationError: Incorrect API key "
            f"provided: {key})\n")
    log.write_text(line + line, encoding="utf-8")  # duplicated on purpose

    first = await pool._tail_console_log(state, tmp_path, 0, 0, 0, 0, (0, 0), 0)
    # Second tick from the returned offset: nothing new, nothing re-emitted.
    await pool._tail_console_log(state, tmp_path, *first)

    failures = [p for ev, p in emitted if ev == "image_failed"]
    assert len(failures) == 1, emitted
    message = failures[0]["message"]
    assert key not in message and "sk-live" not in message
    assert "[REDACTED]" in message
    assert message.startswith("OpenAI rejected the API key")
