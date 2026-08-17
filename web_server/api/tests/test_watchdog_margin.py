"""Guards for the stall-watchdog fixes:
  1. WATCHDOG_STALL_S must stay strictly above the largest per-call LLM timeout
     (a single full-length synthesis call must not by itself trip the watchdog).
  2. The synthesizer's per-attempt heartbeat must match the watchdog regex, so a
     stacked retry ladder resets the stall clock between attempts.
"""
from __future__ import annotations

from pathlib import Path

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
