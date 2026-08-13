"""Settings, paths, and preset definitions for the Caesar web server."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------

API_DIR = Path(__file__).resolve().parent.parent  # web_server/api/
WEB_SERVER_DIR = API_DIR.parent                   # web_server/
ROME_ROOT_DEFAULT = WEB_SERVER_DIR.parent          # rome/
CAESAR_DIR_DEFAULT = ROME_ROOT_DEFAULT / "caesar"


class Settings(BaseSettings):
    """Environment-driven configuration. Loaded from .env at the repo root."""

    model_config = SettingsConfigDict(
        env_file=[
            str(WEB_SERVER_DIR / ".env"),
            str(WEB_SERVER_DIR / ".env.local"),
        ],
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8090, alias="API_PORT")
    # ChromaDB subprocess port. Needs to differ between two simultaneous
    # caesar-web instances on the same machine.
    chroma_port: int = Field(default=8091, alias="CAESAR_CHROMA_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Storage
    caesar_web_data_dir: Path = Field(
        default=API_DIR / "data",
        alias="CAESAR_WEB_DATA_DIR",
    )

    # Caesar integration
    caesar_rome_root: Path | None = Field(default=None, alias="CAESAR_ROME_ROOT")
    caesar_dry_run: bool = Field(default=False, alias="CAESAR_DRY_RUN")
    caesar_max_concurrent: int = Field(default=8, alias="CAESAR_MAX_CONCURRENT")
    # Public (bring-your-own-key) mode. When on, every browser gets an opaque
    # caesar_id cookie as its tenant identity and supplies its own OpenAI key
    # per run. Mutually exclusive with password mode at the launch layer.
    public_mode: bool = Field(default=False, alias="PUBLIC_MODE")
    # Operator password. In non-public mode it's the full login gate (enforced
    # by the Next.js middleware). In public mode it's an OPTIONAL admin step-up:
    # entering it at /login elevates that browser to admin, which can see and
    # wipe every user's runs. Empty = admin disabled.
    demo_password: str = Field(default="", alias="DEMO_PASSWORD")

    @property
    def rome_root(self) -> Path:
        """Resolve the rome repo root, defaulting to the parent of web_server/."""
        return (self.caesar_rome_root or ROME_ROOT_DEFAULT).resolve()

    @property
    def caesar_dir(self) -> Path:
        return self.rome_root / "caesar"

    @property
    def db_path(self) -> Path:
        d = self.caesar_web_data_dir.resolve()
        d.mkdir(parents=True, exist_ok=True)
        return d / "caesar_web.sqlite"

    @property
    def runs_dir(self) -> Path:
        """Where each run's artifact directory is created."""
        d = self.caesar_web_data_dir.resolve() / "runs"
        d.mkdir(parents=True, exist_ok=True)
        return d


@lru_cache
def get_settings() -> Settings:
    return Settings()


# ---------------------------------------------------------------------------
# Preset definitions exposed to the UI
# ---------------------------------------------------------------------------

# UI preset name -> Caesar preset name. The web server has its own copy of
# the preset YAMLs at web_server/config_preset/ — independent of caesar's
# standalone presets in caesar/config/config_preset/. Edit the web copy
# without affecting the standalone CLI workflow.
# The figures below are the operator's. Measured runs are recorded underneath as
# evidence, not as the advertised numbers, and the two deliberately differ in both
# directions: fast is advertised under what it measured (10 min vs 48), deeper over
# ($5 vs $1.86). They are guides for someone choosing a preset, and erring generous
# on cost while brisk on time is the safer way to be wrong.
#
# 64e8fc3 raised every estimate on
# the assumption that GPT-5.6 Luna cost more than gpt-5.4-mini; it is in fact
# 3.75x cheaper ($0.20/$1.20 vs $0.75/$4.50), the same mistake that put Luna in
# MODEL_PRICING at 5x its real price. Luna is also markedly slower per call
# (~60s vs ~16s synthesis steps), so the time estimates were all short.
#
# Sources, post-migration completed runs:
#   fast    $0.83 in 48 min  (43ca06cc, first attempt); advertised at $0.50 by
#           preference -- one run is thin evidence, and the figure is a guide not a cap
#   deeper  $1.86 in 119 min (0a60e102, full 240/240) and $1.10 in 33 min (840b3da0)
#   deepest $29.58, whose first attempt alone spent 435 min on exploration (ed613d7d)
# `normal` has no post-migration run yet; $1 is the pre-migration figure, a better
# prior now that Luna is known to be cheaper rather than dearer.
PRESETS: list[dict] = [
    {
        "id": "fast",
        "label": "Fast",
        "caesar_preset": "fast",
        "description": "Quick parallel exploration with GPT-5.6 Luna. Roughly $0.50 and ~10 min.",
        "estimated_cost_usd": 0.50,
        "estimated_time_min": 10,
    },
    {
        "id": "normal",
        "label": "Normal",
        "caesar_preset": "normal",
        "description": "Mid-depth exploration with GPT-5.6 Luna. Roughly $1 and ~10–20 min.",
        "estimated_cost_usd": 1.00,
        "estimated_time_min": 20,
    },
    {
        "id": "deeper",
        "label": "Deeper",
        "caesar_preset": "deeper",
        "description": (
            "Iterative depth-first walk; richer graph shapes (trees, branches). "
            "GPT-5.6 Luna synthesis. ~$5 and ~45–90 min."
        ),
        "estimated_cost_usd": 5.00,
        "estimated_time_min": 90,
    },
    {
        "id": "deepest",
        "label": "Deepest",
        "caesar_preset": "deepest",
        "description": (
            "Extended iterative walk with the full GPT-5.6 Sol model. "
            "Roughly $30 and ~6 hours."
        ),
        "estimated_cost_usd": 30.00,
        "estimated_time_min": 360,
    },
]


def preset_by_id(preset_id: str) -> dict | None:
    return next((p for p in PRESETS if p["id"] == preset_id), None)


WEB_PRESET_DIR = WEB_SERVER_DIR / "config_preset"


def resolve_preset_yaml(preset_id: str, settings: Settings) -> Path | None:
    """Map a UI preset id to the corresponding Caesar config YAML path."""
    p = preset_by_id(preset_id)
    if p is None:
        return None
    return WEB_PRESET_DIR / f"{p['caesar_preset']}.yaml"


@lru_cache
def _load_preset_yaml(preset_id: str) -> dict | None:
    """Cached read of the preset YAML as a dict; None on any error."""
    yaml_path = resolve_preset_yaml(preset_id, get_settings())
    if yaml_path is None or not yaml_path.exists():
        return None
    try:
        import yaml  # noqa: WPS433
        return yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except (OSError, ImportError, Exception):  # noqa: BLE001
        return None


def preset_total_drafts(preset_id: str) -> int | None:
    """Read `ArtifactSynthesizer.synthesis_drafts` from the preset YAML."""
    data = _load_preset_yaml(preset_id)
    if not data:
        return None
    n = (data.get("ArtifactSynthesizer") or {}).get("synthesis_drafts")
    return int(n) if isinstance(n, int) else None


def preset_llm_model(preset_id: str) -> str | None:
    """Read `LLMHandler.model` (or exploration fallback) from the preset YAML."""
    data = _load_preset_yaml(preset_id)
    if not data:
        return None
    model = (data.get("LLMHandler") or {}).get("model")
    if not model:
        model = ((data.get("CaesarAgent") or {}).get("exploration_llm_config") or {}).get("model")
    return model if isinstance(model, str) else None


@lru_cache
def synthesis_model_choices() -> list[dict]:
    """Synthesis-model choices for the public-mode override dropdown. The
    eligible set and its ordering are defined once, in
    LLMHandler.synthesis_models(); here we only attach each model's pricing for
    the UI's cost hint. Cached — the import pulls in litellm, so we pay it once,
    lazily, on first request rather than at app startup."""
    ensure_caesar_on_path()
    try:
        from rome.llm_handler import LLMHandler  # noqa: WPS433
    except Exception:  # noqa: BLE001
        return []
    pricing = LLMHandler.MODEL_PRICING
    return [
        {
            "id": mid,
            "input_per_mtok": pricing[mid].get("input"),
            "output_per_mtok": pricing[mid].get("output"),
        }
        for mid in LLMHandler.synthesis_models()
    ]


def is_valid_synthesis_model(model_id: str) -> bool:
    """True if model_id is one of the OpenAI models we allow as a synthesis override."""
    return any(c["id"] == model_id for c in synthesis_model_choices())


# ---------------------------------------------------------------------------
# Importing Caesar — done lazily because importing it has heavy side effects
# (loads ChromaDB, NetworkX, llama-index). We don't want to pay that on cold
# import of the FastAPI app; only when a job actually runs.
# ---------------------------------------------------------------------------

def ensure_caesar_on_path() -> None:
    """Add rome/ to sys.path so `from caesar.caesar_agent import CaesarAgent` works."""
    import sys

    settings = get_settings()
    root = str(settings.rome_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    # Also expose it via env var for any subprocess fall-back.
    os.environ.setdefault("PYTHONPATH", root)
