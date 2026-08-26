"""Pydantic request/response schemas. All wire formats live here."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

class PresetOut(BaseModel):
    id: str
    label: str
    description: str
    estimated_cost_usd: float
    estimated_time_min: int


class ModelOut(BaseModel):
    """An OpenAI model offered as a synthesis-model override (public mode)."""
    id: str
    input_per_mtok: float | None = None
    output_per_mtok: float | None = None


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

RunModeLiteral = Literal["new", "explore", "refine"]
GraphProgressPhase = Literal["quick_explore", "explore", "kb_ingest"]


class _ApiKeyBody(BaseModel):
    """Shared per-run-key field for every request body that starts work.

    Per-run OpenAI key in public (bring-your-own-key) mode. Never echoed:
    exclude=True / repr=False are belt-and-suspenders on top of the real
    guarantees (it is absent from every response model and the job runner
    scrubs secrets before any persist/emit sink). None is always allowed;
    shared/password mode omits it. The router enforces "required when
    public_mode". Submit and restart share this so the shape check can't drift.
    """

    api_key: str | None = Field(default=None, exclude=True, repr=False)

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str | None) -> str | None:
        # Allow None (shared/password mode omits the key). Reject an obviously
        # malformed non-null value; the real check is a clean FatalLLMError on
        # the first LLM call, so keep this to a cheap shape check.
        if v is None:
            return v
        v = v.strip()
        if not v.startswith("sk-") or len(v) < 20:
            raise ValueError("api_key must start with 'sk-' and be at least 20 characters.")
        return v


class RunCreate(_ApiKeyBody):
    query: str = Field(min_length=4, max_length=4000)
    preset: str = Field(default="fast")
    # Follow-up linkage. Both default to a normal homepage submission.
    parent_run_id: str | None = Field(default=None)
    mode: RunModeLiteral = Field(default="new")
    # Public-mode only: override the synthesis (LLMHandler) model. The router
    # validates membership against the supported-OpenAI list and ignores it
    # outside public mode; here it is just trimmed to None-or-nonempty.
    synthesis_model: str | None = Field(default=None)
    # Public-mode only: target word count for the synthesized artifact, applied
    # to ArtifactSynthesizer.synthesis_max_length. None = the preset's own value
    # (all four ship null, i.e. unconstrained). Bounded rather than enumerated so
    # there is no choice list to drift out of sync with the UI's dropdown; the
    # floor keeps a target from starving the abstract, the ceiling keeps a typo
    # from ordering a novel. Ignored outside public mode.
    output_length: int | None = Field(default=None, ge=500, le=20000)

    @field_validator("query")
    @classmethod
    def trim(cls, v: str) -> str:
        return v.strip()

    @field_validator("synthesis_model")
    @classmethod
    def trim_model(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip()
        return v or None


class RunRetry(_ApiKeyBody):
    """Body for POST /runs/{id}/retry. The key is the only thing the caller can
    supply: query, preset, mode, parent linkage and KB collection all come from
    the stored row, so a restart cannot quietly become a different run."""


class RunSummary(BaseModel):
    id: str
    query: str
    preset: str
    # Human-facing preset name (e.g. "ArXiv") resolved from the preset registry;
    # null if the stored id has no current entry. Clients fall back to `preset`.
    preset_label: str | None = None
    status: str

    parent_run_id: str | None = None
    mode: str = "new"
    collection_name: str | None = None
    # Public-mode tenant that owns the run. Surfaced so the admin view can label
    # whose run each is; null in single-tenant mode.
    owner_id: str | None = None

    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    # Time already spent in earlier attempts; clients add it to the current
    # attempt's span so a restarted run reports what it actually consumed.
    # Coerced from None because a Run built in memory has no column default
    # until it is flushed, and "never restarted" is the same thing as zero.
    elapsed_prior_s: Annotated[float, BeforeValidator(lambda v: v or 0.0)] = 0.0

    total_cost_usd: float | None = None
    graph_node_count: int | None = None
    error_message: str | None = None
    # For follow-up runs: the LLM-rewritten question Caesar actually used,
    # read from the run's merged_query.txt cache file when present.
    merged_query: str | None = None

    model_config = ConfigDict(from_attributes=True)


class RunEventOut(BaseModel):
    id: int
    timestamp: datetime
    event: str
    payload: dict[str, Any]

    model_config = ConfigDict(from_attributes=True)


class RunDetail(RunSummary):
    events: list[RunEventOut] = Field(default_factory=list)
    llm_model: str | None = None
    # Run whose graph should be visualized. Explore/new runs own their graph;
    # refine runs inherit the nearest non-refine ancestor's graph.
    graph_run_id: str | None = None
    graph_progress_total: int | None = None
    graph_progress_phase: GraphProgressPhase | None = None


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

class GraphNode(BaseModel):
    id: str
    depth: int = 0
    insights: str | None = None
    iteration: int | None = None
    visit_count: int | None = None


class GraphEdge(BaseModel):
    source: str
    target: str
    reason: str | None = None


class GraphOut(BaseModel):
    iteration: int
    starting_url: str | None = None
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class SearchResultItem(BaseModel):
    title: str
    url: str
    description: str = ""


class SearchResultsOut(BaseModel):
    results: list[SearchResultItem]


class CitationOut(BaseModel):
    index: int
    url: str


class SynthesisOut(BaseModel):
    draft: str           # "1", "2", ..., or "merged"
    abstract: str
    artifact: str        # raw text with [N] citation markers
    sources: list[CitationOut]
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Path of the artifact's parent directory relative to run.repository.
    # Empty when the artifact lives at repo root. Used by the UI to resolve
    # markdown image refs that the synthesizer writes relative to the
    # artifact file: `images/<file>` → /api/runs/{id}/file/<artifact_dir>/images/<file>.
    artifact_dir: str = ""
