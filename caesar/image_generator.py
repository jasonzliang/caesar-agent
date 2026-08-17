#!/usr/bin/env python3
"""Image generator post-processor for Caesar artifacts.

Pipeline:
  1. Load a Caesar run's final artifact text + cited URLs
  2. Scrape <img> tags from each cited page (parallel, with size/boilerplate filters)
  3. VLM-score each candidate image for relevance to the artifact abstract
  4. Pick top-K by relevance with per-domain cap for diversity
  5. VLM-caption each picked reference (dense, vision-oriented)
  6. LLM-synthesize an image-gen prompt from artifact + reference captions
  7. Render via OpenAI's images API (gpt-image-2 default; --model overrides)
  8. Write image + metadata JSON next to the run dir

The references inform the prompt (via captions); they are passed as
reference-image inputs to the API only when the top-scoring ref meets the
use_refs_top_score threshold, which keeps the pipeline portable across
image-gen backends and avoids weak refs dragging the output.

User-tunable parameters live in CAESAR_CONFIG["ImageGenerator"]. Static
heuristics (boilerplate tokens, regexes, API caps) are module-level constants.

CLI:
  python -m caesar.image_generator <run_dir> [-o OUT] [-r N] [-m MODEL]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import re
import sys
import traceback
from concurrent.futures import (ThreadPoolExecutor,
                                TimeoutError as FuturesTimeoutError,
                                as_completed)
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
from urllib.parse import urljoin, urlparse

import litellm
from bs4 import BeautifulSoup
from curl_cffi import requests
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rome.config import set_attributes_from_config
from rome.llm_handler import LLMHandler
from rome.logger import get_logger
from caesar.caesar_config import (CAESAR_CONFIG, REQUESTS_HEADERS,
                                  REQUESTS_TIMEOUT)


# Static heuristics, regexes, API limits. User config in CAESAR_CONFIG["ImageGenerator"].

MAX_REF_IMAGE_BYTES = 10 * 1024 * 1024   # per-ref download cap
MAX_HTML_BYTES = 5 * 1024 * 1024         # cap before BS4 parse — guards against
                                         # html.parser's worst-case exponential
                                         # cost on malformed/deep documents
SCRAPE_POOL_TIMEOUT_S = 120              # ceiling for the whole scrape pool;
                                         # bail with partial results past this
                                         # so one stuck worker can't gate the
                                         # entire pipeline.
SCORE_POOL_TIMEOUT_S = 300               # VLM scoring pool ceiling
CAPTION_POOL_TIMEOUT_S = 180             # VLM captioning pool ceiling
DOWNLOAD_POOL_TIMEOUT_S = 60             # image download pool ceiling
IMAGE_API_TIMEOUT_S = 240.0              # per-call OpenAI images.edit /
                                         # litellm.image_generation timeout;
                                         # SDK default is ~10 min × 2 retries.
                                         # gpt-image-1 has a ~180s server
                                         # timeout; 240s + max_retries=1
                                         # gives headroom over the tightest
                                         # observed window.
MAX_GPT_IMAGE_REFS = 16                  # OpenAI gpt-image-* multi-image cap
ABSTRACT_CHARS = 2000                    # chars of artifact_text fed to the VLM
                                         # as the "topic" during candidate scoring.
                                         # VLM output is a 0-10 integer; the
                                         # opening already conveys the topic.
                                         # Per-image section-vs-caption cosine
                                         # downstream handles section-aware
                                         # discrimination — no need to push that
                                         # job upstream to coarse VLM scoring.

# Quality-filter heuristics — defensible defaults, not typical user knobs.
MIN_PER_REF_SCORE = 4.0    # drop individual refs scoring below this (logos, UI chrome)
PER_DOMAIN_CAP = 2         # max references kept from the same domain (diversity)

# I/O parallelism — platform-dependent rather than use-case-dependent.
SCRAPE_WORKERS = 10
SCORE_WORKERS = 6
CAPTION_WORKERS = 4
DOWNLOAD_WORKERS = 6
# Parallel cap for per-image synth_prompt + render. OpenAI image tiers
# throttle around 5-8 concurrent, so 5 covers the common --num-images 5
# case without queueing.
IMAGE_GEN_WORKERS = 5

# Per-image USD pricing for gpt-image-* (landscape sizes ≈ 1536x1024 /
# 1792x1024). Reported into LLMHandler since image_generation responses
# don't carry token usage. Estimates from OpenAI's published per-image
# table — fine for the cost-limit gate; exact billing applies upstream.
IMAGE_GEN_PRICING_USD = {
    "gpt-image-2": {"low": 0.005, "medium": 0.041, "high": 0.165},
    "gpt-image-1": {"low": 0.005, "medium": 0.041, "high": 0.165},
}

# Saved-file extension per configured output_format.
_FORMAT_TO_EXT = {
    "png": ".png", "jpg": ".jpg", "jpeg": ".jpg", "webp": ".webp",
}
# What we ask the OpenAI API to render in.
_FORMAT_TO_API = {
    "png": "png", "jpg": "jpeg", "jpeg": "jpeg", "webp": "webp",
}

# For multi-image runs, each image's images.edit call gets a chunk-specific
# top-refs_per_image subset ranked by caption-vs-chunk cosine similarity.
EMBED_MODEL = "text-embedding-3-small"
EMBED_USD_PER_TOKEN = 0.02 / 1_000_000  # OpenAI list price; reported via report_external_cost

BOILERPLATE_TOKENS = {
    "logo", "logos", "icon", "icons", "favicon", "avatar", "avatars", "badge",
    "ad", "ads", "tracking", "pixel", "sprite", "sprites", "emoji",
    "social", "share", "button", "buttons", "spinner", "loader",
}
URL_PATTERN = re.compile(r"https?://[^\s\]\"\'<>]+")  # allow '(' and ')' for Wikipedia
_SYNTH_NUM_RE = re.compile(r"synthesis-(\d+)")
_SCORE_RE = re.compile(r"\d+(?:\.\d+)?")

# Heuristic prefilter keyword sets (used in _heuristic_candidate_score). Kept
# at module scope so they're compiled once, not per-candidate. Positive set:
# tokens that hint a real figure/diagram. Negative set: tokens that hint UI
# chrome / branding. Logo penalty is heavier than the figure reward: being
# a logo is a stronger signal than being a figure, since "diagram" can
# appear in many false-positive contexts.
_HEURISTIC_POSITIVE_TOKENS = (
    "figure", "diagram", "chart", "schematic", "plot", "graph",
    "flowchart", "architecture", "pipeline", "illustration",
)
_HEURISTIC_NEGATIVE_TOKENS = (
    "logo", "icon", "banner", "header", "footer", "avatar", "thumbnail",
    "sprite", "bg", "background", "social", "share",
)
_DIMENSION_RE = re.compile(r"(\d{3,4})x(\d{3,4})")


class PickedRef(NamedTuple):
    """A reference image that survived VLM scoring + per-domain cap.
    Positional-tuple-compatible so existing unpack patterns still work."""
    url: str
    alt: str
    score: float


# Step-4 reference-caption directive — varies by variant role. Injected as a
# bullet inside Step 4 PROMPT requirements; refers to the REFERENCE IMAGE
# CAPTIONS (not Step 3 CAPTION). Anchor form (n=1 and variant-0) prescribes
# style-anchoring with no opposing pressure. Diverge form (variants 2..n)
# softens to avoid contradicting the variant directive's "do NOT inherit
# references' default palette" instruction.
CAPTION_DIRECTIVE_ANCHOR = (
    "- Use the REFERENCE IMAGE CAPTIONS above as the primary style anchor: "
    "borrow palette, materials, and lighting from them when vivid so the "
    "image reads as visually continuous with the cited sources."
)
CAPTION_DIRECTIVE_DIVERGE = (
    "- The REFERENCE IMAGE CAPTIONS above may give useful style anchors "
    "(palette, materials, lighting) when vivid; borrow only what serves the "
    "insight, never let the references dictate the aesthetic."
)
# Diagram mode rejects the palette/materials/lighting framing that the metaphor
# directives use — diagrams must stay flat, evenly lit, sans-serif regardless
# of reference aesthetic. References are still useful for label density and
# annotation layout conventions; they just must not bleed style into the diagram.
CAPTION_DIRECTIVE_DIAGRAM = (
    "- Use the REFERENCE IMAGE CAPTIONS above only as layout/annotation hints "
    "(label density, arrow conventions, panel structure). Ignore reference "
    "palette, materials, lighting, and texture entirely. Diagrams stay flat, "
    "evenly lit, and sans-serif regardless of reference aesthetic."
)


PROMPT_SYNTH_TEMPLATE = """You are inventing a single striking image whose creative concept is grounded in a specific research artifact section. Concept must come from THIS section's particulars, not generic "research" imagery. The reference captions describe images from the artifact's cited sources; they provide visual grounding (palette, materials, lighting) only.

RESEARCH ARTIFACT SECTION (excerpt):
{artifact_excerpt}

REFERENCE IMAGE CAPTIONS:
{captions_block}

Step 1. INSIGHT: State one specific, surprising, or load-bearing claim from the section as a direct declarative sentence in the artifact's own concrete terms. Name the specific concept, construct, or mechanism (e.g., the theory's name, a defined operator, a key term) rather than paraphrasing it into generic equivalents. No meta-framing ("The section argues that...", "This passage shows...").

Step 2. METAPHOR: Invent a concrete scene, object, or material that embodies the INSIGHT claim. Where possible, draw the metaphor's materials, setting, era, or conceptual primitives (signals, gates, paths, traces) from the artifact's own world. A reader of the artifact should say "yes, this is about *this* finding, not a generic picture." Map the claim via a familiar action (filtering, sealing, weaving, dissolving, or similar). Avoid stock metaphors (lightbulbs, puzzle pieces, icebergs, lighthouses, scales, gears). Show a SINGLE STATE in ONE frame: the cut, not the cutting; the seal cracked, not the cracking. If the INSIGHT references multiple example types (a list of cases, a taxonomy), commit the scene to ONE. Encoding multiple sub-types in a single frame defeats the figure.

Step 3. CAPTION: One sentence (under 50 words). State the INSIGHT claim plainly first, then anchor with one concrete visual element from METAPHOR (see Examples below). Follow the Examples: claim, then ':', then a single concrete visual anchor as one clause. Do not use parenthetical metaphor-mappings inside the caption; if the claim cannot be read without visual-element decoding, simplify the visual instead. If the claim contains opaque technical jargon, paraphrase the obscure parts in plain everyday language while keeping the artifact's central named concept. A reader who has not seen the artifact should grasp the claim from one read of the caption alone. Active voice, present tense. No "this image shows/depicts..." framing.

Example 1: "Trust depends on sealing, not on what's inside: a wax seal across a folded letter proves only that no one opened it first."

Example 2: "One small change can decide the entire outcome: a single snapped warp thread on a loom determines which pattern the cloth can form."

Step 4. PROMPT: Write the image-generation prompt that renders the METAPHOR, encodes the INSIGHT, and matches the CAPTION. Requirements:
- Make ONE bold compositional or material choice that encodes the claim's texture, scale, mechanism, or stakes, not decoration. The choice should feel inevitable for THIS artifact.
- Specify mood and atmosphere with concrete sensory phrases, never bare adjectives. Write "late-afternoon raking light through dust" not "dramatic lighting"; "graphite haze over warm grey paper" not "atmospheric".
- Name concrete visual elements: subject, materials, lighting type, palette, shot scale, composition principle.
- Name a specific visual idiom or medium: medieval illumination, blueprint, cyanotype, etching, lithograph, woodblock, specimen plate, oil portrait, risograph, travel poster, retro postcard, vintage tourism brochure. Match the idiom to the subject's NATIVE register: don't dignify consumer/leisure content with prestige-art idioms (Persian miniature) unless the thematic content earns it. Don't default to editorial photograph or studio still life.
- The scene's dominant visible state must be the state the CAPTION's claim depends on. If the claim hinges on a failure, contradiction, or counterfactual (e.g., "every pair fails", "brim-full leaves no room"), render that state, not its picturesque opposite. A viewer of the rendered scene should be able to verify the CAPTION's claim directly from what is depicted.
{caption_directive}
- Stay under 200 words. Vision-oriented language only, no abstract jargon.

AVOID these clichés regardless of topic: glowing brains, neural-network meshes, scientists at whiteboards, hands on keyboards, gradient data visualizations with axes, isometric infographics, exploded-diagram science illustrations, "futuristic" blue-glow palettes, photorealistic stock photography of people working, and the "a16z editorial infographic" envelope (the COMBINATION of single hero object + cream backdrop + raking sidelight + cream/navy/burgundy serif palette + faint paper texture; any single element alone is fine).

Format your response exactly as, in this exact order:
INSIGHT: <one sentence>
METAPHOR: <one sentence>
CAPTION: <one sentence>
PROMPT: <the image-generation prompt>
(PROMPT MUST be the LAST section.)"""


PROMPT_SYNTH_TEMPLATE_DIAGRAM = """You are designing a single labeled diagram whose informational content is grounded in a specific research artifact section. The diagram must communicate the section's structure (stages, equations, components, comparisons) directly, NOT through metaphor. The reference captions describe images from the artifact's cited sources; they provide visual grounding (layout conventions, label density) only.

RESEARCH ARTIFACT SECTION (excerpt):
{artifact_excerpt}

REFERENCE IMAGE CAPTIONS:
{captions_block}

Step 1. INSIGHT: State one specific, load-bearing structural claim from the section as a direct declarative sentence in the artifact's own concrete terms. Name the specific algorithm, system, equation, or taxonomy (e.g., the procedure's name, the defined symbols, the named stages) rather than paraphrasing into generic equivalents. No meta-framing.

Step 2. FIGURE_KIND: Pick exactly ONE of: flowchart (sequential stages with arrows), algorithm_panel (numbered pseudocode lines), equation_diagram (rendered formula with annotated variables), system_schematic (named blocks connected by labeled edges), comparison_matrix (rows of named things, columns of attributes). Justify in one clause why this kind matches the INSIGHT's structure. If the section enumerates 3 or more named stages, prefer flowchart or system_schematic; if it contains a defining equation, prefer equation_diagram; if it contrasts 2 or more named approaches, prefer comparison_matrix.

Step 3. CAPTION: One sentence (under 50 words). State the INSIGHT claim plainly, then name what the diagram shows. Active voice, present tense. A reader who has not seen the artifact should grasp the claim from one read of the caption alone. Paraphrase opaque jargon in plain everyday language while keeping the artifact's central named concept. No "this image shows" framing.

Example 1: "The RAG pipeline retrieves before it generates: a four-stage flowchart from query to embedding to top-k retrieval to LLM response."

Example 2: "Loss combines reconstruction and KL terms: the VAE objective annotated with each variable's role."

Step 4. PROMPT: Write the image-generation prompt that renders the FIGURE_KIND, encodes the INSIGHT, and matches the CAPTION. Requirements:
- Specify the visual idiom explicitly: clean whitepaper figure, textbook diagram, architectural blueprint with callouts, lab notebook page with handwritten annotations, technical schematic. Pick ONE. NEVER vintage lithograph, oil portrait, woodblock, etching, or any decorative-art idiom.
- List every text label that must appear, verbatim and in quotes, with each box/stage/variable named exactly as the artifact names it. Spell out arrows: arrow from 'Embed' to 'Retrieve' labeled 'top-k'. For equations, give the LaTeX-rendered form as a quoted string.
- Specify layout: left-to-right flowchart, top-down pipeline, two-column comparison, centered equation with radial annotations. State spacing convention (generous whitespace, gridded backdrop, none).
- Palette: limit to 3 colors maximum (e.g., black text on white, one accent for highlights). Monochrome blueprint OK. NO atmospheric lighting, NO dramatic mood, NO raking light. Flat, evenly lit, sans-serif typography.
- Every label must be legible at thumbnail size. Prefer fewer, larger labels over many small ones.
{caption_directive}
- Stay under 200 words. Vision-oriented language only.

AVOID: decorative borders, vintage paper texture, sepia tones, illustration metaphors, allegorical scenes, hand-painted aesthetics, neural-network mesh visualizations, glowing brains, isometric infographics, "futuristic" blue-glow palettes, the "a16z editorial infographic" envelope.

Format your response exactly as, in this exact order:
INSIGHT: <one sentence>
FIGURE_KIND: <one of the five kinds, plus one-clause justification>
CAPTION: <one sentence>
PROMPT: <the image-generation prompt>
(PROMPT MUST be the LAST section.)"""


# ── Module-level pure helpers ──────────────────────────────────────────────

def _extract_section(text: str, label: str,
                     single_line: bool = False) -> str:
    """Extract one section of the synth chain's INSIGHT/METAPHOR/CAPTION/PROMPT output.

    Anchored to start-of-line (MULTILINE) so a literal "insight:" inside body
    text doesn't match. Tolerates markdown bold and em-dash/hyphen separators.
    For single_line=True (INSIGHT, METAPHOR, CAPTION), returns just the rest
    of the matching line, stripped of asterisks; for single_line=False,
    returns everything after the marker to end-of-text."""
    pattern = (r'^\s*\*{0,2}\s*' + re.escape(label) +
               r'\s*\*{0,2}\s*[:\-—]\s*\*{0,2}')
    m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    if not m:
        return ""
    start = m.end()
    if single_line:
        end = text.find("\n", start)
        body = text[start:end if end >= 0 else len(text)]
    else:
        body = text[start:]
    return body.strip().strip("*").strip()


_DIAGRAM_HEADER_RE = re.compile(
    r"(?im)^\s*(?:#{1,4}\s*)?[\d.]*\s*"
    r"(algorithm|theorem|lemma|proposition|pseudocode|procedure|"
    r"pipeline|architecture|framework|protocol|workflow|schema)\b"
)
_DIAGRAM_MATH_RE = re.compile(
    r"(\$\$.+?\$\$)|(\\\([^)]+\\\))|(\\\[[^\]]+\\\])|"
    r"\b(alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|phi|psi|omega)"
    r"\s*(?:=|<-|:=)"
)
_DIAGRAM_STAGES_RE = re.compile(
    r"(?im)(?:^\s*(?:\d+\.|[-*])\s+.*?\b"
    r"(stage|step|phase|layer|agent|component|module|block|node)\b.*$\s*){3,}"
)
# Bullet marker required: section sub-headings ("Methods:", "Note:", speaker
# names) otherwise false-trigger as compare rows. Real comparison tables use
# bullets.
_DIAGRAM_COMPARE_RE = re.compile(
    r"(?im)^\s*[-*]\s+\*{0,2}([A-Z][A-Za-z0-9_-]{2,30})\*{0,2}\s*:\s+\S",
)
# Strong-signal override: an explicit "Algorithm N", "Theorem N", "Lemma N",
# "Proposition N", "Pseudocode N", or "Procedure N" block at line start (with
# optional markdown header marker) is on its own enough to warrant diagram
# mode, regardless of the broader score-2-AND-structural rule. Per-chunk text
# is often too small to contain multiple signals; this catches formal-math
# chunks reliably. Requires a numeric suffix so prose mentions of "the
# algorithm" don't false-trigger.
_DIAGRAM_FORMAL_BLOCK_RE = re.compile(
    r"(?im)^\s*(?:#{1,4}\s*)?"
    r"(Algorithm|Theorem|Lemma|Proposition|Pseudocode|Procedure)\s+\d+\b"
)


def _classify_image_mode(artifact_text: str) -> str:
    """Classify whether an artifact section warrants a labeled diagram, a metaphor scene, or a hybrid.

    Pure regex; zero LLM cost.

    Strong-signal overrides fire first (diagram regardless of overall score):
      - Explicit "Algorithm N" / "Theorem N" / "Lemma N" / etc. block
      - Three or more display-math expressions

    Otherwise scores 4 signals across two families:
      Semantic: header_hit, math_hit (the section is *about* something diagrammatic)
      Structural: stages_hit, compare_hit (the section *enumerates* something)

    Diagram mode requires at least one STRUCTURAL signal — narrative prose with
    a header keyword and an inline greek-letter assignment ('## Architecture ...
    we set sigma = 0.3') would otherwise tip to diagram falsely. Returns:
      'metaphor' — default, classic Caesar metaphor scene (zero signals)
      'diagram'  — labeled flowchart / algorithm panel / equation diagram
      'hybrid'   — metaphor template with a callout-label rider
    """
    # Strong-signal overrides — these alone justify diagram mode.
    if _DIAGRAM_FORMAL_BLOCK_RE.search(artifact_text):
        return "diagram"
    if len(_DIAGRAM_MATH_RE.findall(artifact_text)) >= 3:
        return "diagram"
    header_hit = bool(_DIAGRAM_HEADER_RE.search(artifact_text))
    math_hit = bool(_DIAGRAM_MATH_RE.search(artifact_text))
    stages_hit = bool(_DIAGRAM_STAGES_RE.search(artifact_text))
    compare_rows = _DIAGRAM_COMPARE_RE.findall(artifact_text)
    compare_hit = len(set(compare_rows)) >= 3
    score = sum([header_hit, math_hit, stages_hit, compare_hit])
    structural = stages_hit or compare_hit
    if score >= 2 and structural:
        return "diagram"
    if score == 0:
        return "metaphor"
    return "hybrid"


def _strip_trailing_punct(url: str) -> str:
    """URL regex over-captures trailing punctuation; trim it.

    Also strips trailing ')' only when *unbalanced* — preserves Wikipedia-style
    disambiguation URLs like .../Foo_(bar) but trims '...com/x)' that came from
    enclosing parens in prose."""
    url = url.rstrip(".,;:!?")
    while url.endswith(')') and url.count('(') < url.count(')'):
        url = url[:-1]
    return url


def _is_boilerplate(src: str) -> bool:
    """True if any URL path segment is EXACTLY a known boilerplate token
    (after stripping extension). Segment-exact matching avoids the substring
    false-positives of plain `in` matching (e.g. '/buttonwood-tree.jpg' no
    longer matches 'button'; '/loader-theory.jpg' no longer matches 'loader').
    Edge cases like '/header-logo.png' slip through but are caught downstream
    by the VLM scoring step, which scores them near zero."""
    path = urlparse(src).path.lower()
    for seg in path.split("/"):
        if seg and seg.split(".", 1)[0] in BOILERPLATE_TOKENS:
            return True
    return False


def _heuristic_candidate_score(img_url: str, alt_text: str) -> float:
    """Cheap text-only relevance score used to prefilter candidates before
    VLM scoring. Higher = more likely to be a real figure/diagram. Signals:

      - URL path keywords: +1.0 per positive token (figure, diagram, ...),
        -1.5 per negative token (logo, icon, banner, ...). Negative weight
        is heavier because logo/icon hits are a stronger signal than figure
        hits ("diagram" appears in many false-positive contexts).
      - Alt text length: +1.0 if >=20 chars, +0.5 if >=10. Real figures
        tend to carry descriptive alt text.
      - Alt text content keywords: +0.5 per positive token match.
      - URL-encoded dimensions like 1024x768: +0.5 when both dims >=400.
        Tiny dims hint at icons or thumbnails.
      - File extension: -0.5 for .svg (typically logos/icons); png/jpg/
        webp neutral.

    Returns a float capped at +5.0 (no lower bound; negative scores are
    fine: they sort to the bottom).
    """
    path = urlparse(img_url).path.lower()
    score = 0.0

    # URL path keyword matches. Substring search on the path is intentional:
    # a path like "/wp-content/figure-1-architecture.png" should match both
    # "figure" and "architecture". _is_boilerplate has already segment-exact
    # filtered the worst offenders (favicon, sprite), so substring matches
    # here are a softer secondary signal.
    for tok in _HEURISTIC_POSITIVE_TOKENS:
        if tok in path:
            score += 1.0
    for tok in _HEURISTIC_NEGATIVE_TOKENS:
        if tok in path:
            score -= 1.5

    alt_stripped = (alt_text or "").strip()
    alt_len = len(alt_stripped)
    if alt_len >= 20:
        score += 1.0
    elif alt_len >= 10:
        score += 0.5

    if alt_stripped:
        alt_lower = alt_stripped.lower()
        for tok in _HEURISTIC_POSITIVE_TOKENS:
            if tok in alt_lower:
                score += 0.5

    dim_match = _DIMENSION_RE.search(path)
    if dim_match:
        try:
            w, h = int(dim_match.group(1)), int(dim_match.group(2))
            if w >= 400 and h >= 400:
                score += 0.5
        except ValueError:
            pass

    if path.endswith(".svg"):
        score -= 0.5

    # Small positive baseline for valid bitmap image extensions. Without this,
    # CDN-served hash-named WebPs (e.g. /p/abc123.webp, common on Substack,
    # Ghost, Medium) score 0 — indistinguishable from a vague stock photo —
    # because no keywords match. Python's stable sort then makes the merit
    # pool a function of scrape order rather than the heuristic's intent.
    if path.endswith((".png", ".jpg", ".jpeg", ".webp", ".avif")):
        score += 0.25

    return min(score, 5.0)


def _parse_score(txt: str) -> float:
    m = _SCORE_RE.search(txt or "")
    if not m:
        return 0.0
    try:
        v = float(m.group(0))
    except ValueError:
        return 0.0
    return max(0.0, min(10.0, v))


# ── ImageGenerator ────────────────────────────────────────────────────────

class ImageGenerator:
    """End-to-end pipeline: Caesar artifact + cited URLs → generated image.

    All user-tunable parameters come from CAESAR_CONFIG["ImageGenerator"], either
    by default or via the `config` dict passed at construction. Pipeline stages
    are private methods (`_scrape_*`, `_score_*`, `_caption_*`, `_synth_prompt`,
    `_render`); public entry points are `run()` (raw artifact + URLs) and
    `run_from_dir()` (load a Caesar run directory).
    """

    def __init__(self, config: Optional[Dict] = None,
                 llm_handler: Optional[LLMHandler] = None):
        self.logger = get_logger()
        # Layer user config over CAESAR_CONFIG defaults. ImageGenerator is often
        # used standalone (CLI, post-processor) so it can't rely on the agent's
        # load_config() to pre-merge defaults the way other components can.
        self.config = {**CAESAR_CONFIG["ImageGenerator"], **(config or {})}
        set_attributes_from_config(
            self, self.config, CAESAR_CONFIG["ImageGenerator"].keys())
        # Synthesizer passes its agent's handler so VLM/synth costs share the
        # exploration cost bucket; CLI gets a fresh isolated handler.
        self.llm_handler = llm_handler or LLMHandler()

    # ── Step 1: Load Caesar run ───────────────────────────────────────

    def _load_run(self, run_dir: Path) -> Dict:
        """Load final artifact text + cited URL list from a Caesar run directory.

        Searches recursively for synthesis*.txt and the exp summary JSON so it
        handles both root-level and __rome__/ subdir layouts.
        """
        run_dir = Path(run_dir).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Not a directory: {run_dir}")

        # Prefer merged synthesis; fall back to the highest-numbered draft.
        # Filter merged out of drafts list and numeric-sort drafts so synthesis-10
        # is treated higher than synthesis-2 (lexicographic sort would reverse them).
        merged = sorted(run_dir.rglob("*synthesis-merged*.txt"))
        all_synth = list(run_dir.rglob("*synthesis-*.txt"))
        drafts = [p for p in all_synth if "synthesis-merged" not in p.name]
        drafts.sort(key=lambda p: (int(m.group(1)) if (m := _SYNTH_NUM_RE.search(p.name)) else -1, p.name))
        artifact_path = merged[-1] if merged else (drafts[-1] if drafts else None)
        if artifact_path is None:
            raise FileNotFoundError(f"No synthesis*.txt artifact in {run_dir}")
        artifact_text = artifact_path.read_text(encoding="utf-8", errors="replace")
        if not artifact_text.strip():
            raise ValueError(f"Artifact {artifact_path} is empty; nothing to generate from")

        # The exp summary may carry structured sources/visited_urls. run_agent.py
        # writes "<exp_id>.exp_summary.json"; the older "experiment_summary"
        # spelling is matched too so pre-0.4 run dirs still resolve.
        summary: Dict = {}
        for p in (*run_dir.rglob("*exp_summary*.json"), *run_dir.rglob("*experiment_summary*.json")):
            try:
                summary = json.loads(p.read_text(encoding="utf-8"))
                break
            except json.JSONDecodeError:
                continue

        urls: List[str] = []
        for k in ("sources", "visited_urls", "urls"):
            v = summary.get(k)
            if isinstance(v, list):
                urls.extend(str(u.get("url") if isinstance(u, dict) else u) for u in v)
        # Also pull inline URLs from the artifact text itself.
        urls.extend(_strip_trailing_punct(u) for u in URL_PATTERN.findall(artifact_text))
        # Dedupe, preserve order.
        seen, out = set(), []
        for u in urls:
            if u and u not in seen and u.startswith("http"):
                seen.add(u)
                out.append(u)

        return {"artifact": artifact_text, "urls": out,
                "summary": summary, "artifact_path": str(artifact_path)}

    # ── Step 2-3: Scrape candidates ───────────────────────────────────

    def _scrape_url(self, url: str) -> List[Tuple[str, str]]:
        """Return list of (img_url, alt_text) for a page. Best-effort, swallows errors."""
        try:
            resp = requests.get(url, impersonate="chrome", timeout=REQUESTS_TIMEOUT,
                                headers=REQUESTS_HEADERS, allow_redirects=True)
            resp.raise_for_status()
            # See MAX_HTML_BYTES — guards BS4 worst-case parsing cost.
            if len(resp.content) > MAX_HTML_BYTES:
                self.logger.debug(
                    f"skip oversized page {url}: "
                    f"{len(resp.content)} bytes > {MAX_HTML_BYTES}")
                return []
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            self.logger.debug(f"scrape failed for {url}: {e}")
            return []

        out: List[Tuple[str, str]] = []
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
            if not src:
                continue
            src = urljoin(url, src)
            if not src.startswith("http") or _is_boilerplate(src):
                continue
            alt = (img.get("alt") or "").strip()[:200]
            out.append((src, alt))
        return out

    def _scrape_all(self, urls: List[str]) -> List[Tuple[str, str]]:
        """Scrape candidate images from many URLs in parallel; dedupe by image URL.

        We manage the executor lifecycle manually (no `with`) so that on pool
        timeout we can shutdown(wait=False) instead of blocking on stuck
        workers. Python can't kill threads, so a worker stuck in a BS4 parse
        will keep running until the process exits — but it no longer holds
        up the rest of the pipeline.
        """
        all_candidates: List[Tuple[str, str]] = []
        ex = ThreadPoolExecutor(max_workers=SCRAPE_WORKERS)
        try:
            futures = {ex.submit(self._scrape_url, u): u for u in urls}
            try:
                for fut in as_completed(futures, timeout=SCRAPE_POOL_TIMEOUT_S):
                    try:
                        all_candidates.extend(fut.result())
                    except Exception as e:
                        self.logger.error(
                            f"scrape worker failed for "
                            f"{futures[fut][:80]}: {e.__class__.__name__}: {e}")
            except FuturesTimeoutError:
                # concurrent.futures.TimeoutError is a separate class from the
                # builtin TimeoutError on Python 3.10 — they're only aliased in
                # 3.11+. Catch the futures one explicitly for 3.10 support.
                n_done = sum(1 for f in futures if f.done())
                self.logger.error(
                    f"scrape pool exceeded {SCRAPE_POOL_TIMEOUT_S}s; proceeding "
                    f"with {n_done}/{len(futures)} completed URLs")
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
        seen, deduped = set(), []
        for src, alt in all_candidates:
            if src in seen:
                continue
            seen.add(src)
            deduped.append((src, alt))
        return deduped

    # ── Step 4: VLM scoring ───────────────────────────────────────────

    def _score_image(self, image_url: str, alt: str, abstract: str) -> float:
        """VLM scores 0-10: how visually relevant is this image to the abstract?

        Score-only call (10 output tokens). Captioning is deferred to a second pass
        that runs only on the picked top-K, to avoid wasting ~150 captioning calls
        on candidates we'll never use. The alt text is included so the VLM can
        downscore UI screenshots / dashboard chrome that look topical but won't
        serve as visual references for illustrative outputs."""
        try:
            resp = self.llm_handler.completion(
                num_retries=0,  # VLM scoring/captioning fails soft; don't stack timeouts
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (
                            "Score 0-10 how useful this image is as a VISUAL "
                            "REFERENCE for an illustrative figure about the "
                            "topic below. Judge from the IMAGE PIXELS; alt "
                            "text is metadata.\n"
                            "0 = unrelated; UI captures, app screenshots, "
                            "dashboard panels, admin consoles, settings pages, "
                            "code editors, browser windows showing software "
                            "interfaces (regardless of how topical the alt text "
                            "sounds); also corporate logos, app icons, brand "
                            "marks.\n"
                            "5 = on-topic but generic.\n"
                            "10 = distinctive imagery that directly illustrates "
                            "the topic, including period photographs, "
                            "primary-source portraits, and archival imagery of "
                            "the topic's protagonist on biographical/historical "
                            "topics.\n"
                            "Reply with ONLY a number. No commentary.\n\n"
                            f"Topic: {abstract}\n"
                            f"Image alt text (may be empty or noisy): "
                            f"{alt or '(none)'}"
                        )},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }],
                model=self.vlm_model,
                max_completion_tokens=10,
                temperature=0,
            )
            return _parse_score(resp.choices[0].message.content)
        except Exception as e:
            # Log exceptions at debug so transient API failures are diagnosable
            # without spamming INFO; the 0.0 return still collides with legit zero
            # scores, but at least operators can grep for these on suspicious runs.
            self.logger.debug(f"VLM score failed for {image_url[:80]}: {e}")
            return 0.0

    def _select_references(self, candidates: List[Tuple[str, str]],
                           abstract: str,
                           pool_size: int) -> List[PickedRef]:
        """Score candidates in parallel, return top-pool_size with per-domain cap."""
        if not candidates:
            return []
        # Heuristic-rank ALL candidates (cheap regex over strings). Merit pool
        # = top by heuristic. Exploration pool = random sample from the FULL
        # heuristic tail (not the truncated top), so the control reaches
        # candidates whose URL keywords miss but whose alt text or content is
        # strong — exactly the case the heuristic systematically fails on.
        # Total VLM scoring cost is bounded by vlm_score_pool_size +
        # vlm_score_explore_size regardless of input size. Per-domain diversity
        # is enforced after VLM scoring via PER_DOMAIN_CAP.
        heur_scored = sorted(
            candidates,
            key=lambda c: -_heuristic_candidate_score(c[0], c[1]),
        )
        merit_pool = heur_scored[:self.vlm_score_pool_size]
        remainder = heur_scored[self.vlm_score_pool_size:]
        explore_n = min(self.vlm_score_explore_size, len(remainder))
        exploration_pool = (random.sample(remainder, explore_n)
                            if explore_n else [])
        candidates = merit_pool + exploration_pool
        self.logger.info(
            f"Prefilter: {len(candidates)} candidates "
            f"(merit={len(merit_pool)}, explore={len(exploration_pool)}, "
            f"raw={len(heur_scored)})")

        self.logger.info(f"Scoring {len(candidates)} candidate images via VLM...")

        scored: List[PickedRef] = []
        # Manual executor lifecycle — a `with ThreadPoolExecutor(...) as ex`
        # block calls shutdown(wait=True) on exit, which blocks on stuck
        # futures. See the scrape pool in _scrape_all for the same pattern.
        ex = ThreadPoolExecutor(max_workers=SCORE_WORKERS)
        try:
            futures = {
                ex.submit(self._score_image, src, alt, abstract): (src, alt)
                for src, alt in candidates
            }
            try:
                for fut in as_completed(futures, timeout=SCORE_POOL_TIMEOUT_S):
                    src, alt = futures[fut]
                    # _score_image swallows its own exceptions and returns 0.0.
                    s = fut.result()
                    # MIN_PER_REF_SCORE drops weak individual refs (logos, UI chrome
                    # that score 0-3) before they pollute the per-domain cap.
                    if s >= MIN_PER_REF_SCORE:
                        scored.append(PickedRef(src, alt, s))
            except FuturesTimeoutError:
                n_done = sum(1 for f in futures if f.done())
                self.logger.error(
                    f"VLM score pool exceeded {SCORE_POOL_TIMEOUT_S}s; "
                    f"proceeding with {n_done}/{len(futures)} scored refs")
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

        scored.sort(key=lambda r: -r.score)
        picked: List[PickedRef] = []
        domain_counts: Dict[str, int] = {}
        for ref in scored:
            dom = urlparse(ref.url).netloc
            if domain_counts.get(dom, 0) >= PER_DOMAIN_CAP:
                continue
            picked.append(ref)
            domain_counts[dom] = domain_counts.get(dom, 0) + 1
            if len(picked) >= pool_size:
                break
        return picked

    # ── Step 5: Caption picked refs ───────────────────────────────────

    def _caption_image(self, image_url: str) -> str:
        """Produce a 2-3 sentence visual caption focused on concrete attributes.
        Returns "" on any failure so the orchestrator can move on."""
        try:
            resp = self.llm_handler.completion(
                num_retries=0,  # VLM scoring/captioning fails soft; don't stack timeouts
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (
                            "Describe this image in 2-3 sentences focusing on concrete visual elements: "
                            "subject, composition, lighting, palette, style, materials, mood. "
                            "No generic commentary, no interpretation; just what's visible."
                        )},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }],
                model=self.vlm_model,
                max_completion_tokens=220,
                temperature=0.2,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            self.logger.debug(f"VLM caption failed for {image_url[:80]}: {e}")
            return ""

    def _caption_all(self, picked: List[PickedRef]) -> List[str]:
        """Caption picked references in parallel (preserves input order).
        _caption_image swallows exceptions and returns "" on failure."""
        captions = [""] * len(picked)
        # Manual executor lifecycle so the pool timeout actually bounds
        # wall-clock (see _scrape_all / _pick_and_score_refs).
        ex = ThreadPoolExecutor(max_workers=CAPTION_WORKERS)
        try:
            futures = {ex.submit(self._caption_image, ref.url): i
                       for i, ref in enumerate(picked)}
            try:
                for fut in as_completed(futures, timeout=CAPTION_POOL_TIMEOUT_S):
                    captions[futures[fut]] = fut.result()
            except FuturesTimeoutError:
                n_done = sum(1 for f in futures if f.done())
                self.logger.error(
                    f"VLM caption pool exceeded {CAPTION_POOL_TIMEOUT_S}s; "
                    f"proceeding with {n_done}/{len(futures)} captions")
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
        return captions

    # ── Step 6: Synthesize the image-gen prompt ───────────────────────

    def _synth_prompt(self, artifact_text: str, captions: List[str],
                      variant: Optional[Tuple[int, int]] = None,
                      mode: str = "metaphor",
                      is_anchor: Optional[bool] = None,
                      ) -> Tuple[str, str]:
        """LLM-synthesize the image-gen prompt from artifact + reference captions.

        Forces a 4-step chain (INSIGHT → METAPHOR → CAPTION → PROMPT) so the
        creative concept is grounded in a specific claim from the artifact.
        Returns (prompt, caption): the PROMPT block sent to images.edit, and
        the CAPTION (a one-sentence figure caption that explicitly binds the
        rendered visual from METAPHOR to the claim from INSIGHT). INSIGHT
        and METAPHOR are logged with the full chain for debug but not
        threaded — nothing downstream consumes them directly.

        `variant=(idx, n)` for multi-image runs appends a per-variant directive
        asking the LLM to explore an angle other variants would miss — a real
        instruction the model can act on, since gpt-5.x reasoning models often
        ignore temperature.
        """
        usable = [c for c in captions if c]
        captions_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(usable)) \
            if usable else "(no usable references)"
        # Anchor (n=1 or variant-0 in multi-image runs): prescriptive caption
        # anchoring — no opposing variant directive will follow. Variants 2..n:
        # softer directive so the prescriptive line doesn't contradict the
        # "do NOT inherit references' default palette" instruction appended
        # below.
        # Anchor = single image OR variant-0 of a multi-image fan-out. If the
        # caller (_gen_one) didn't pass an explicit flag, derive from variant.
        if is_anchor is None:
            is_anchor = (variant is None or variant[1] == 1
                         or variant[0] == 0)
        if mode == "diagram":
            template = PROMPT_SYNTH_TEMPLATE_DIAGRAM
            caption_directive = CAPTION_DIRECTIVE_DIAGRAM
        else:
            template = PROMPT_SYNTH_TEMPLATE
            caption_directive = (CAPTION_DIRECTIVE_ANCHOR if is_anchor
                                 else CAPTION_DIRECTIVE_DIVERGE)
        prompt = template.format(
            artifact_excerpt=artifact_text, captions_block=captions_block,
            caption_directive=caption_directive)
        if mode == "hybrid":
            # "include 2-3 callouts" (the old phrasing) primed the model to
            # add unsolicited decoration on top of the specified labels.
            # Discretionary + capped + a hard "no other text" guard.
            prompt += (
                "\n\nADDITIONAL: you may add at most two short sans-serif "
                "labels naming the artifact's key terms, only if the visual "
                "alone would leave them ambiguous. No other text on the image.")
        if variant is not None and variant[1] > 1:
            idx, total = variant
            if idx == 0:
                # Anchor variant: take the canonical reading the others diverge
                # from. Without this, n=2 collapses into two contrarians with
                # no baseline.
                prompt += (f"\n\nThis is variant 1 of {total}. Lead with the "
                           f"artifact's most central, load-bearing claim: the "
                           f"canonical reading the other variants will diverge "
                           f"from.")
            elif mode == "diagram":
                # Diagram-mode diverge keeps the flat sans-serif idiom (the
                # template requires it) but picks a different structural facet
                # to render. Aesthetic divergence would directly contradict the
                # diagram template's "NO atmospheric lighting" / "flat, evenly
                # lit, sans-serif" rules.
                prompt += (
                    f"\n\nThis is variant {idx + 1} of {total}. Variant 1 "
                    f"covers the canonical structural claim; this variant must "
                    f"diverge by picking a secondary structural facet or a "
                    f"different FIGURE_KIND on the same content. Keep the "
                    f"flat, sans-serif, evenly-lit diagram idiom required by "
                    f"the template."
                )
            else:
                prompt += (
                    f"\n\nThis is variant {idx + 1} of {total}. Variant 1 "
                    f"covers the canonical reading; this variant must diverge. "
                    f"Pick a secondary finding (not the headline claim) AND a "
                    f"different aesthetic from the obvious default for "
                    f"research-artifact images (forensic-archive interiors, "
                    f"lab benches, cold fluorescent + corroded metal + "
                    f"oxidized teal). Do NOT inherit the references' default "
                    f"palette."
                )
        # The 4-step INSIGHT→METAPHOR→CAPTION→PROMPT chain is the most
        # cognitively dense LLM call in the pipeline, but on most calls medium
        # reasoning produces a complete, well-formed chain. Try medium first
        # (cheaper, faster); fall back to high+more tokens only when the
        # response is truncated or missing one of the required sections.
        # 10000 -> 15000 on retry: high reasoning consumes more internal tokens,
        # so headroom for the visible output shrinks. Cost impact of the retry
        # is bounded (<=$0.05/call worst case) and only paid on the minority of
        # calls that need it.
        def _call(reasoning: str, max_tokens: int):
            return self.llm_handler.completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.prompt_model,
                temperature=0.9,
                max_completion_tokens=max_tokens,
                reasoning_effort=reasoning,
                # Image gen fails soft (returns None), so it owns its retry:
                # num_retries=0 keeps a hung call from stacking 3x the timeout.
                num_retries=0,
            )

        def _is_incomplete(resp_text: str) -> bool:
            """True if any required section (INSIGHT / METAPHOR-or-FIGURE_KIND /
            CAPTION / PROMPT) is missing. METAPHOR and FIGURE_KIND are
            interchangeable so this stays correct if/when diagram mode adds
            FIGURE_KIND in place of METAPHOR."""
            for label in ("INSIGHT", "CAPTION", "PROMPT"):
                if not _extract_section(resp_text, label,
                                        single_line=(label != "PROMPT")):
                    return True
            # METAPHOR or FIGURE_KIND (either satisfies the middle step)
            if not (_extract_section(resp_text, "METAPHOR", single_line=True)
                    or _extract_section(resp_text, "FIGURE_KIND",
                                        single_line=True)):
                return True
            return False

        resp = _call("medium", 10000)
        truncated = getattr(resp.choices[0], "finish_reason", None) == "length"
        text = (resp.choices[0].message.content or "").strip()
        if truncated or _is_incomplete(text):
            self.logger.info(
                "synth: medium truncated/incomplete, retrying with high")
            resp = _call("high", 15000)
            if getattr(resp.choices[0], "finish_reason", None) == "length":
                self.logger.error(
                    "Synth output truncated at max_completion_tokens even on "
                    "high-reasoning retry. Visible PROMPT may be cut; bump "
                    "max_completion_tokens if this recurs.")
            text = (resp.choices[0].message.content or "").strip()
        else:
            self.logger.info("synth: medium reasoning OK")
        # Log the full INSIGHT / METAPHOR / CAPTION / PROMPT chain — most
        # useful signal when a generated image misses the mark.
        self.logger.info(f"Prompt synthesis reasoning:\n{text}")
        # Caption source: CAPTION (Step 3) explicitly binds the rendered
        # visual (from METAPHOR) to the artifact's claim (from INSIGHT).
        # METAPHOR alone is image-faithful but doesn't convey the claim;
        # INSIGHT alone states the claim but doesn't match the picture —
        # CAPTION synthesizes both. INSIGHT and METAPHOR are logged with
        # the full chain above but not extracted — no other consumer.
        caption = _extract_section(text, "CAPTION", single_line=True)
        # Anchor to start-of-line (MULTILINE) so a literal "prompt:" inside
        # INSIGHT/METAPHOR/CAPTION text doesn't match. Tolerate markdown bold
        # and em-dash/hyphen separators (consume them so the body extract is
        # clean). Prefer the LAST match if multiple (the actual Step-4 marker
        # is always last in the 4-step chain).
        markers = list(re.finditer(
            r'^\s*\*{0,2}\s*PROMPT\s*\*{0,2}\s*[:\-—]\s*\*{0,2}',
            text, re.IGNORECASE | re.MULTILINE))
        if not markers:
            self.logger.error(
                "Synth output missing PROMPT marker — full LLM text "
                "(INSIGHT+METAPHOR+CAPTION+PROMPT) will be sent to images.edit.")
            return text, caption
        prompt_start = markers[-1].end()
        # Bound at the next start-of-line CAPTION/INSIGHT/METAPHOR marker so a
        # misordered emission (PROMPT before CAPTION) can't leak a trailing
        # stanza into the image-gen prompt body. No-op when ordering is obeyed.
        tail = re.search(
            r'^\s*\*{0,2}\s*(?:CAPTION|INSIGHT|METAPHOR|FIGURE_KIND)\s*\*{0,2}\s*[:\-—]',
            text[prompt_start:], re.IGNORECASE | re.MULTILINE)
        prompt_end = prompt_start + tail.start() if tail else len(text)
        prompt_body = text[prompt_start:prompt_end].strip().strip('*').strip()
        return prompt_body, caption

    # ── Step 7: Image generation ─────────────────────────────────────

    @staticmethod
    def _save_image_data(data_obj, output_path: Path) -> None:
        """OpenAI/litellm image responses carry b64_json or url; current litellm
        always returns pydantic objects (never raw dicts), so plain getattr is
        sufficient."""
        b64 = getattr(data_obj, "b64_json", None)
        url = getattr(data_obj, "url", None)
        if b64:
            output_path.write_bytes(base64.b64decode(b64))
            return
        if url:
            resp = requests.get(url, impersonate="chrome", timeout=REQUESTS_TIMEOUT,
                                headers=REQUESTS_HEADERS)
            resp.raise_for_status()
            output_path.write_bytes(resp.content)
            return
        raise RuntimeError("Image API returned neither b64_json nor url")

    def _download_image(self, url: str) -> Optional[io.BytesIO]:
        """Download a URL to a named BytesIO for the OpenAI SDK.

        Returns None on any failure (network error, non-image MIME, oversized) so
        parallel callers can drop failed refs silently. The .name attribute is what
        the SDK uses to infer the MIME type when uploading."""
        try:
            resp = requests.get(url, impersonate="chrome", timeout=REQUESTS_TIMEOUT,
                                headers=REQUESTS_HEADERS, allow_redirects=True)
            resp.raise_for_status()
        except Exception:
            return None
        ct = resp.headers.get("content-type", "").lower()
        body = resp.content
        if not ct.startswith("image/") or len(body) > MAX_REF_IMAGE_BYTES:
            return None
        if "jpeg" in ct or "jpg" in ct: ext = "jpg"
        elif "webp" in ct: ext = "webp"
        elif "png" in ct: ext = "png"
        else: return None  # skip gif/svg/bmp — sketchy for the edits endpoint
        buf = io.BytesIO(body)
        buf.name = f"ref.{ext}"
        return buf

    def _download_refs_parallel(self, urls: List[str]) -> List[io.BytesIO]:
        """Download a list of reference URLs in parallel; drops failures, preserves
        input order so the highest-scored ref is first (image-gen models often
        weight earlier reference images more heavily)."""
        if not urls:
            return []
        # _download_image swallows exceptions and returns None on failure.
        results: List[Optional[io.BytesIO]] = [None] * len(urls)
        # Manual executor lifecycle so the pool timeout actually bounds
        # wall-clock (see _scrape_all / _pick_and_score_refs).
        ex = ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)
        try:
            futures = {ex.submit(self._download_image, u): i for i, u in enumerate(urls)}
            try:
                for fut in as_completed(futures, timeout=DOWNLOAD_POOL_TIMEOUT_S):
                    results[futures[fut]] = fut.result()
            except FuturesTimeoutError:
                n_done = sum(1 for f in futures if f.done())
                self.logger.error(
                    f"image download pool exceeded {DOWNLOAD_POOL_TIMEOUT_S}s; "
                    f"proceeding with {n_done}/{len(futures)} downloads")
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
        out = [b for b in results if b is not None]
        if len(out) < len(urls):
            self.logger.info(f"  downloaded {len(out)}/{len(urls)} reference images "
                             f"({len(urls) - len(out)} dropped: non-image MIME, oversized, or unreachable)")
        return out

    def _render(self, prompt: str, output_path: Path,
                reference_urls: Optional[List[str]] = None) -> Dict:
        """Render the image and return {via, refs_used}.

        Two paths, picked once — no silent fallback between them:
          - reference_urls non-empty: download bytes, call OpenAI's images.edit so
            gpt-image-* conditions directly on the reference pixels.
          - reference_urls empty: call litellm.image_generation, prompt-only.
        Exceptions propagate — silently retrying as prompt-only would mask real
        API errors (model rejection, auth, etc.) behind a downgrade."""
        refs: List[io.BytesIO] = []
        if reference_urls:
            refs = self._download_refs_parallel(reference_urls[:MAX_GPT_IMAGE_REFS])

        api_format = _FORMAT_TO_API.get(self.output_format, "webp")
        if refs:
            # Use the per-run key from the LLM handler (not the ambient env): in
            # public mode the server strips OPENAI_API_KEY, and image gen runs
            # during synthesis, outside the per-run env window.
            # timeout + max_retries=1 bound the call — OpenAI SDK defaults to
            # ~10 min × 2 retries, which stacks into ~30 min per image when
            # the API is slow. That's an ugly failure mode for an
            # otherwise-completed run; the watchdog would kill the whole run.
            client = OpenAI(
                api_key=self.llm_handler.api_key,
                timeout=IMAGE_API_TIMEOUT_S,
                max_retries=1,
            )
            kwargs = dict(model=self.image_model, image=refs, prompt=prompt,
                          size=self.size, n=1, output_format=api_format)
            if self.quality:
                kwargs["quality"] = self.quality
            result = client.images.edit(**kwargs)
            data = getattr(result, "data", None)
            if not data:
                raise RuntimeError("images.edit returned empty data")
            self._save_image_data(data[0], output_path)
            self._report_image_cost()
            return {"via": "images.edit", "refs_used": len(refs)}

        kwargs = dict(model=self.image_model, prompt=prompt, size=self.size,
                      n=1, output_format=api_format,
                      api_key=self.llm_handler.api_key,
                      timeout=IMAGE_API_TIMEOUT_S)
        if self.quality:
            kwargs["quality"] = self.quality
        result = litellm.image_generation(**kwargs)
        data = getattr(result, "data", None)
        if not data:
            raise RuntimeError("Image API returned empty data")
        self._save_image_data(data[0], output_path)
        self._report_image_cost()
        return {"via": "images.generate", "refs_used": 0}

    def _per_image_ref_urls(self, picked: List[PickedRef],
                            captions: List[str],
                            chunk_texts: List[str]
                            ) -> List[Optional[List[str]]]:
        """Pick top-refs_per_image URLs per chunk by caption-vs-chunk cosine.
        Returns one entry per chunk: None=use shared order (fallback on any
        failure); [urls...]=ranked ref URLs."""
        valid = [i for i, c in enumerate(captions) if c]
        chunk_pairs = [(i, t) for i, t in enumerate(chunk_texts) if t and t.strip()]
        if len(valid) <= 1 or not chunk_pairs:
            return [None] * len(chunk_texts)

        n_caps = len(valid)
        # Per-run key, like the render calls above. An unkeyed client falls back
        # to OPENAI_API_KEY, which public mode strips from the environment on
        # purpose, so this never worked there: it raised, was swallowed by the
        # except below, and every image quietly used the shared ref order.
        resp = OpenAI(api_key=getattr(self.llm_handler, "api_key", None)).embeddings.create(
            model=EMBED_MODEL,
            input=[captions[i] for i in valid] + [t for _, t in chunk_pairs],
        )
        # Route embedding spend through llm_handler so accumulated_cost / the
        # web UI's live_cost_usd include it (same pattern as _report_image_cost).
        tokens = getattr(getattr(resp, "usage", None), "prompt_tokens", 0)
        if tokens and self.llm_handler:
            self.llm_handler.report_external_cost(tokens * EMBED_USD_PER_TOKEN)
        if len(resp.data) != n_caps + len(chunk_pairs):
            self.logger.error(
                f"Embeddings returned {len(resp.data)} vectors, "
                f"expected {n_caps + len(chunk_pairs)} — falling back")
            return [None] * len(chunk_texts)
        cap_embs = [d.embedding for d in resp.data[:n_caps]]
        chunk_embs = [d.embedding for d in resp.data[n_caps:]]

        # text-embedding-3-small returns unit-normalized vectors → cosine = dot
        def _cos(a, b): return sum(x * y for x, y in zip(a, b))

        all_urls = [ref.url for ref in picked]
        top_n = max(1, min(self.refs_per_image, len(picked), n_caps))

        results: List[Optional[List[str]]] = [None] * len(chunk_texts)
        for j, (orig_idx, _) in enumerate(chunk_pairs):
            sims = [_cos(chunk_embs[j], e) for e in cap_embs]
            ranked = sorted(range(n_caps), key=lambda i: -sims[i])
            picked_caps = [valid[ranked[k]] for k in range(top_n)]
            results[orig_idx] = [all_urls[i] for i in picked_caps]
            self.logger.info(
                f"  [ref-pick] image {orig_idx + 1}: top sim {sims[ranked[0]]:.2f}, "
                f"selected {top_n}/{len(picked)} refs")
        return results

    def _report_image_cost(self) -> None:
        """Track this render's cost in LLMHandler so accumulated_cost
        includes image-gen spend, not just chat completions."""
        cost = IMAGE_GEN_PRICING_USD.get(self.image_model, {}).get(
            self.quality or "high", 0.0)
        self.llm_handler.report_external_cost(
            cost, model=f"{self.image_model}/{self.quality or 'high'}")

    # ── Orchestrator ─────────────────────────────────────────────────

    @staticmethod
    def _numbered_output_paths(base: Path, n: int) -> List[Path]:
        """Return [base] for n=1; numbered <stem>_1<ext>..<stem>_n<ext> for n>1."""
        if n <= 1:
            return [base]
        stem, suffix, parent = base.stem, base.suffix, base.parent
        return [parent / f"{stem}_{i}{suffix}" for i in range(1, n + 1)]

    def run(self, artifact_text: str, urls: List[str],
            output_path: Path, n: int = 1,
            per_image_texts: Optional[List[str]] = None
            ) -> Union[Dict, List[Dict]]:
        """Run the full pipeline. n=1 returns Dict; n>1 returns List[Dict].
        Shared work (scrape/score/caption) runs once; per-image synth+render
        fans out across IMAGE_GEN_WORKERS threads.

        `per_image_texts`: optional list of length n. When provided, image i's
        synth_prompt uses per_image_texts[i] instead of artifact_text — lets
        callers ground each image in a specific section. Ref scoring/captioning
        still use the full artifact_text.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if per_image_texts is not None and len(per_image_texts) != n:
            raise ValueError(
                f"per_image_texts length {len(per_image_texts)} != n {n}")
        output_path = Path(output_path).resolve()
        # Normalize the extension to match the configured output_format so
        # callers don't have to track it (e.g. CLI default + synthesizer pass
        # `.png`, but the file will be saved as .webp/.avif if configured).
        ext = _FORMAT_TO_EXT.get(self.output_format, ".webp")
        if output_path.suffix.lower() != ext:
            output_path = output_path.with_suffix(ext)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Scrape references.
        candidates: List[Tuple[str, str]] = []
        if urls:
            candidates = self._scrape_all(urls[:self.max_cited_urls])
        self.logger.info(f"Scraped {len(candidates)} candidate images from "
                         f"{min(len(urls), self.max_cited_urls)} cited URLs")

        # Auto-size the captioned pool: refs_per_image + (n-1) extra slots so
        # the per-image subset selection can pick distinct refs across images.
        # Capped at OpenAI's 16-ref images.edit limit.
        pool_size = min(MAX_GPT_IMAGE_REFS, self.refs_per_image + max(0, n - 1))

        # Score-only across the wide candidate pool; captioning is deferred so we
        # don't pay to caption ~150 candidates we won't use.
        abstract = artifact_text[:ABSTRACT_CHARS].strip()
        picked: List[PickedRef] = []
        if candidates and abstract:
            picked = self._select_references(candidates, abstract, pool_size)
            # Pool-stats log: gives the score distribution of survivors in one
            # line so a fallback-to-text-only run is diagnosable without
            # opening per-image metadata sidecars. "3/15 [6.0,5.0,4.0]" reads
            # as "only 3 of 15 scored ≥MIN_PER_REF_SCORE; best was 6.0".
            self.logger.info(
                f"Ref pool: {len(picked)}/{len(candidates)} survived "
                f"MIN_PER_REF_SCORE={MIN_PER_REF_SCORE} "
                f"(scores={[round(r.score, 1) for r in picked]})")

        # Single rule: use refs iff the best one is genuinely topical (top score
        # ≥ use_refs_top_score). Below that, a mid-quality reference biases
        # gpt-image-* more than it helps — the prompt is already strongly
        # grounded in the artifact text.
        use_references = picked and picked[0].score >= self.use_refs_top_score
        captions: List[str] = []
        if use_references:
            captions = self._caption_all(picked)
            for ref, cap in zip(picked, captions):
                self.logger.info(f"  ref(score={ref.score:.1f}) {ref.url[:80]} :: {cap[:120]}")
        elif picked:
            self.logger.info(f"Top ref score {picked[0].score:.1f} < {self.use_refs_top_score} "
                             "— falling back to text-only image gen")

        # When refs are used, pass their URLs so _render conditions gpt-image-*
        # directly via images.edit. Captions still go into the prompt — they
        # carry the conceptual framing that raw pixels alone don't. Sliced to
        # refs_per_image so the shared/fallback path matches the per-image
        # subset semantics (single API-cap applies uniformly across n).
        ref_urls_for_api = (
            [ref.url for ref in picked[:self.refs_per_image]]
            if use_references else []
        )
        # URL → caption lookup so each image's synth prompt sees captions for
        # the refs ACTUALLY passed to its images.edit call (not the full pool).
        url_to_caption = dict(zip([ref.url for ref in picked], captions))
        metadata_refs = [
            {"url": ref.url, "alt": ref.alt, "score": ref.score,
             "caption": (captions[i] if i < len(captions) else "")}
            for i, ref in enumerate(picked)
        ]
        output_paths = self._numbered_output_paths(output_path, n)

        # Per-image ref subset by chunk-vs-caption cosine; fall back to
        # shared order on any failure.
        per_image_refs: List[Optional[List[str]]] = [None] * n
        if (use_references and per_image_texts and n > 1 and len(picked) > 1):
            try:
                per_image_refs = self._per_image_ref_urls(
                    picked, captions, per_image_texts)
            except Exception as e:
                self.logger.error(
                    f"Per-image ref selection failed ({e}); keeping shared refs")

        def _gen_one(idx: int, out: Path) -> Dict:
            """Synthesize a prompt and render one image (one task in the n>1 fan-out)."""
            focus_text = per_image_texts[idx] if per_image_texts else artifact_text
            image_refs = (per_image_refs[idx]
                          if per_image_refs[idx] is not None
                          else ref_urls_for_api)
            # Captions aligned 1:1 with image_refs so the prompt's caption
            # block describes the refs actually passed to images.edit.
            image_captions = [url_to_caption.get(u, "") for u in image_refs]
            # Anchor = single image OR the first of a multi-image fan-out.
            # For multi-image, the anchor stays metaphor so variants 2..n can
            # diverge aesthetically without contradicting the canonical reading.
            # Pass is_anchor explicitly to _synth_prompt so the directive choice
            # has one source of truth.
            is_anchor = (n == 1) or (idx == 0)
            if n > 1 and is_anchor:
                mode = "metaphor"
            else:
                mode = _classify_image_mode(focus_text)
            self.logger.info(f"Image {idx + 1}/{n} synth mode: {mode}")
            prompt, caption = self._synth_prompt(
                focus_text, image_captions, variant=(idx, n), mode=mode,
                is_anchor=is_anchor)
            self.logger.info(f"Image-gen prompt {idx + 1}/{n} ({len(prompt)} chars):\n{prompt}")
            # Diagram mode goes text-only: reference pixels strongly condition
            # images.edit toward the references' visual texture, which fights
            # the diagram template's "flat, sans-serif, evenly lit" rules.
            # Captions stay in the prompt as layout hints; only the pixels drop.
            render_refs = [] if mode == "diagram" else image_refs
            try:
                gen_info = self._render(prompt, out, reference_urls=render_refs)
            except Exception as gen_err:
                # On render failure, write a metadata JSON capturing what was
                # attempted so the failure is debuggable (picked refs, prompt,
                # model params) before bubbling the exception up.
                meta_path = out.with_suffix(out.suffix + ".json")
                meta_path.write_text(json.dumps({
                    "image_path": str(out),
                    "image_model": self.image_model,
                    "prompt": prompt,
                    "caption": caption,
                    "mode": mode,
                    "picked_pool": metadata_refs,
                    "refs_sent_to_api": list(render_refs),
                    "error_type": type(gen_err).__name__,
                    "candidate_count": len(candidates),
                    "cited_urls_scraped": min(len(urls), self.max_cited_urls),
                    "size": self.size, "quality": self.quality,
                }, indent=2))
                self.logger.error(f"Generation {idx + 1}/{n} failed; partial metadata at {meta_path}")
                raise
            self.logger.info(f"Image {idx + 1}/{n} written to {out} (via {gen_info['via']}, "
                             f"refs_used={gen_info['refs_used']})")
            metadata = {
                "image_path": str(out),
                "image_model": self.image_model,
                "prompt": prompt,
                "caption": caption,
                "mode": mode,
                "picked_pool": metadata_refs,
                "refs_sent_to_api": list(render_refs),
                "generation_via": gen_info["via"],
                "refs_passed_to_api": gen_info["refs_used"],
                "candidate_count": len(candidates),
                "cited_urls_scraped": min(len(urls), self.max_cited_urls),
                "size": self.size, "quality": self.quality,
            }
            meta_path = out.with_suffix(out.suffix + ".json")
            meta_path.write_text(json.dumps(metadata, indent=2))
            self.logger.info(f"Metadata: {meta_path}")
            return metadata

        if n == 1:
            return _gen_one(0, output_paths[0])

        # Parallel fan-out for n > 1. Per-image failures don't abort the batch:
        # a failed image gets a stub metadata entry with error fields and the
        # rest still complete. If ALL fail, we re-raise the last exception so
        # callers don't silently end up with no images.
        results: List[Dict] = [None] * n
        errors: List[Exception] = []
        with ThreadPoolExecutor(max_workers=min(IMAGE_GEN_WORKERS, n)) as ex:
            futures = {ex.submit(_gen_one, i, output_paths[i]): i for i in range(n)}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    self.logger.error(f"Image {i + 1}/{n} failed: "
                                      f"{e.__class__.__name__}: {e}")
                    errors.append(e)
                    results[i] = {"image_path": str(output_paths[i]),
                                  "error": str(e),
                                  "error_type": type(e).__name__}
        success_count = sum(1 for r in results if r and "error" not in r)
        if success_count == 0:
            raise RuntimeError(
                f"All {n} image generations failed; last error: {errors[-1]}")
        self.logger.info(f"Generated {success_count}/{n} images")
        return results

    def run_from_dir(self, run_dir: Path,
                     output_path: Optional[Path] = None,
                     n: int = 1) -> Union[Dict, List[Dict]]:
        """Convenience entry point: load a Caesar run and run the pipeline.

        Default output is `<artifact_dir>/images/<artifact_stem>.{ext}` —
        matching where ArtifactSynthesizer places its inline-embedded images
        so the CLI and synthesis paths share one `images/` subdir per run.
        Extension follows the configured output_format (webp by default).
        For n>1, the numeric suffix `_1`...`_n` is appended before the ext.
        """
        run_dir = Path(run_dir).resolve()
        bundle = self._load_run(run_dir)
        if output_path is None:
            artifact = Path(bundle["artifact_path"])
            ext = _FORMAT_TO_EXT.get(self.output_format, ".webp")
            output_path = artifact.parent / "images" / f"{artifact.stem}{ext}"
        return self.run(bundle["artifact"], bundle["urls"], output_path, n=n)


# ── Section-aware embedding helpers ───────────────────────────────────────

_ARTIFACT_HEADER_RE = re.compile(r"^ARTIFACT:[ \t]*\n", re.MULTILINE)
_POST_ARTIFACT_RE = re.compile(r"\n(?:SOURCES|REFERENCES):[ \t]*\n", re.MULTILINE)


def _extract_artifact_body(full_text: str) -> str:
    """Extract the ARTIFACT: section body; returns "" if the marker isn't found."""
    m = _ARTIFACT_HEADER_RE.search(full_text)
    if not m:
        return ""
    start = m.end()
    nxt = _POST_ARTIFACT_RE.search(full_text, start)
    return full_text[start:nxt.start() if nxt else None].strip()


def _splice_artifact_body(full_text: str, new_body: str) -> str:
    """Replace the ARTIFACT body, preserving surrounding ABSTRACT/SOURCES sections.
    When the envelope is missing, fall back to returning new_body — the caller
    has already spliced image markdown into it, and silently returning the
    unmodified original would lose the work."""
    m = _ARTIFACT_HEADER_RE.search(full_text)
    if not m:
        return new_body
    start = m.end()
    nxt = _POST_ARTIFACT_RE.search(full_text, start)
    end = nxt.start() if nxt else len(full_text)
    return (full_text[:start] + new_body.strip() + "\n\n"
            + full_text[end:].lstrip("\n"))


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _balanced_split(items: List[str], k: int, joiner: str
                    ) -> Tuple[List[str], List[int]]:
    """Split items into k contiguous chunks of roughly equal size. Returns
    (chunk_texts, heads) where heads[i] is the 1-based index of chunk i's
    first item in `items`. Caller controls the joiner ("\\n\\n" for
    paragraph chunking, " " for sentence chunking)."""
    if k < 1 or not items:
        return [], []
    sizes = [len(items) // k] * k
    for i in range(len(items) % k):
        sizes[i] += 1
    chunks, heads, start = [], [], 0
    for size in sizes:
        chunks.append(joiner.join(items[start:start + size]))
        heads.append(start + 1)
        start += size
    return chunks, heads


def _chunk_paragraphs(paragraphs: List[str], n: int
                      ) -> Tuple[List[str], List[int]]:
    """Split into n contiguous chunks for per-image grounding.

    Common path (paragraphs >= n): chunk by paragraphs. heads[i] is the
    1-based paragraph index where image i is placed (chapter-illustration
    style). heads[0] is always 1 → image 1 sits at the top of the artifact
    when n=1 (hero image).

    Short-artifact fallback (paragraphs < n): split into sentences so each
    image still gets a distinct slice of the artifact (otherwise variance
    falls back to LLM temperature, which gpt-5.x reasoning models often
    ignore). heads stay paragraph-indexed (length=len(paragraphs)); images
    beyond len(heads) tail-append via _splice_image_markdown.
    """
    total = len(paragraphs)
    if total == 0 or n < 1:
        return [], []
    if total >= n:
        return _balanced_split(paragraphs, n, "\n\n")

    sentences = [s.strip() for p in paragraphs
                 for s in _SENTENCE_RE.split(p) if s.strip()]
    if len(sentences) >= n:
        chunk_texts, _ = _balanced_split(sentences, n, " ")
        # Heads stay paragraph-indexed (length=total<n); extras tail-append
        # via _splice_image_markdown. Sentence-chunk i's content may not
        # topically correspond to paragraph i — known tradeoff of the
        # short-artifact path; fixing it would require duplicate-head
        # handling in the markdown splicer.
        return chunk_texts, list(range(1, total + 1))

    # Too few sentences too — return paragraph chunks; embed_images_in_artifact
    # pads remaining slots with the full artifact text.
    return _balanced_split(paragraphs, total, "\n\n")


def _figure_md(j: int, filename: str, caption: str) -> str:
    """Render one image as markdown with chunk-specific caption when
    available. Falls back to bare 'Figure N' alt + no caption line when
    caption is empty (preserves prior behavior).

    Sanitizes the caption: ']' would close the alt text early, '*' would
    open stray italic in the caption line. Diagram-mode captions are more
    likely to contain bracket-like notation, so the sanitization matters
    more after the diagram template ships."""
    if caption:
        # Sanitize each context separately: ']' would close markdown alt text
        # early, '*' would break the italic span. Doing both replacements on
        # the visible line silently dropped legitimate '[Smith 2020]'-style
        # citations and emphasis from captions; doing both on alt text was
        # over-conservative for the italic line.
        label_alt = f"Figure {j + 1}: {caption.replace(']', ')')}"
        label_vis = f"Figure {j + 1}: {caption.replace('*', '')}"
        return f"![{label_alt}](images/{filename})\n\n*{label_vis}*"
    return f"![Figure {j + 1}](images/{filename})"


def _splice_image_markdown(paragraphs: List[str],
                           image_filenames: List[Optional[str]],
                           heads: List[int],
                           captions: Optional[List[str]] = None) -> str:
    """Place each image right before its source chunk's first paragraph (per
    `heads`). Images beyond len(heads) — extras when n exceeds the paragraph
    count — tail-append at the end. `image_filenames[i]` may be None for
    slots where the i-th image generation failed; those slots are skipped so
    remaining successes stay aligned with their original chunks (otherwise
    filtering would shift image 3 into image 2's slot). `captions[i]` (the
    chunk-specific CAPTION sentence extracted from synth output, binding
    visual to claim) renders as both alt text and a visible italic caption
    line under each image; empty entries fall back to bare 'Figure N' alt
    with no caption."""
    if not paragraphs:
        return ""
    if not image_filenames or not any(image_filenames):
        return "\n\n".join(paragraphs)
    if captions is None:
        captions = [""] * len(image_filenames)
    head_to_img = {h: j for j, h in enumerate(heads)
                   if j < len(image_filenames) and image_filenames[j] is not None}
    out: List[str] = []
    for i, para in enumerate(paragraphs):
        j = head_to_img.get(i + 1)
        if j is not None:
            cap = captions[j] if j < len(captions) else ""
            out.append(_figure_md(j, image_filenames[j], cap))
        out.append(para)
    for j in range(len(heads), len(image_filenames)):
        if image_filenames[j] is not None:
            cap = captions[j] if j < len(captions) else ""
            out.append(_figure_md(j, image_filenames[j], cap))
    return "\n\n".join(out)


def embed_images_in_artifact(artifact_text: str, urls: List[str],
                             output_dir: Path,
                             agent_id: str,
                             n: int,
                             llm_handler: Optional[LLMHandler] = None,
                             **image_gen_kwargs) -> Optional[str]:
    """Generate N section-aware images for `artifact_text` and return the
    artifact with markdown image refs spliced in. Returns None on any failure
    (no artifact text, no paragraphs, image-gen error, all images failed) so
    callers can leave the original artifact intact.

    Images are written under `output_dir` with names of the form
    `<agent_id>.image.<ts>.png` (n=1) or `<...>.image.<ts>_<i>.png` (n>1).
    The artifact is partitioned into N chunks; each image's prompt synthesis
    is grounded in its own chunk so the INSIGHT/METAPHOR matches the
    surrounding text. `image_gen_kwargs` are forwarded to `ImageGenerator`'s
    config (e.g., `image_model`, `refs_per_image`)."""
    logger = get_logger()
    if n < 1 or not artifact_text.strip():
        return None
    paragraphs = [p for p in re.split(r'\n\s*\n', artifact_text) if p.strip()]
    if not paragraphs:
        logger.error("[IMAGE] No paragraphs to anchor images; skipping")
        return None

    chunk_texts, boundaries = _chunk_paragraphs(paragraphs, n)
    # When n > number of paragraphs, _chunk_paragraphs caps chunks at total;
    # pad per_image_texts with the full artifact so the extras (which will
    # tail-append in the doc) are at least grounded in the whole text.
    per_image_texts = list(chunk_texts) + [artifact_text] * (n - len(chunk_texts))

    ts = datetime.now().strftime("%m%d%H%M")
    fmt = image_gen_kwargs.get("output_format",
                               CAESAR_CONFIG["ImageGenerator"].get("output_format", "webp"))
    ext = _FORMAT_TO_EXT.get(fmt, ".webp")
    out_path = Path(output_dir) / f"{agent_id}.image.{ts}{ext}"

    logger.info(f"[IMAGE] Generating {n} image(s) across "
                f"{len(chunk_texts)} artifact section(s)")
    try:
        results = ImageGenerator(config=image_gen_kwargs,
                                 llm_handler=llm_handler).run(
            artifact_text=artifact_text, urls=urls, output_path=out_path,
            n=n, per_image_texts=per_image_texts)
    except Exception as e:
        logger.error(f"[IMAGE] Image generation failed: {type(e).__name__}")
        return None

    # n=1 returns Dict, n>1 returns List[Dict]. Preserve slot alignment with
    # the original chunk indices by keeping None for failed entries — the
    # splice helper then leaves those chunk boundaries empty rather than
    # shifting later images into earlier chunks' slots.
    metadata_list = results if isinstance(results, list) else [results]
    slot_filenames: List[Optional[str]] = [
        Path(m["image_path"]).name
        if (m and "error" not in m and m.get("image_path"))
        else None
        for m in metadata_list
    ]
    # Captions: CAPTION (Step 3) extracted per-image; binds the rendered
    # visual to the artifact's claim. Empty string for failed slots or
    # extraction misses.
    captions: List[str] = [
        (m.get("caption") or "") if isinstance(m, dict) else ""
        for m in metadata_list
    ]
    if not any(slot_filenames):
        logger.error("[IMAGE] No images succeeded; artifact unchanged")
        return None
    return _splice_image_markdown(paragraphs, slot_filenames, boundaries,
                                  captions=captions)


def generate_image_from_run(run_dir: Path,
                            output_path: Optional[Path] = None,
                            n: int = 1,
                            llm_handler: Optional[LLMHandler] = None,
                            **kwargs) -> Union[Dict, List[Dict]]:
    """Wrapper around ImageGenerator.run_from_dir(); kwargs become config."""
    return ImageGenerator(config=kwargs, llm_handler=llm_handler).run_from_dir(
        run_dir, output_path, n=n)


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> int:
    """CLI: run image generation on a Caesar run dir, embed markdown refs into the artifact."""
    defaults = CAESAR_CONFIG["ImageGenerator"]
    parser = argparse.ArgumentParser(
        description="Generate an image from a Caesar run's artifact + cited URLs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python -m caesar.image_generator caesar/result/<run> "
            "--output out.png --references 4 --model gpt-image-2\n"
        ),
    )
    parser.add_argument("run_dir", type=str, help="Path to a Caesar run directory")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output image path (default: "
                             "<artifact_dir>/images/<artifact_stem>.png)")
    parser.add_argument("-n", "--num-images", type=int, default=1, metavar="N",
                        help="Number of images to generate (default: 1). N>1 fans "
                             "out the prompt-synth + render steps in parallel; "
                             "scrape/score/caption are still shared work.")
    parser.add_argument("-r", "--references", type=int, default=defaults["refs_per_image"],
                        help=f"Refs per image (default: {defaults['refs_per_image']})")
    parser.add_argument("-m", "--model", type=str, default=defaults["image_model"],
                        help=f"Image-gen model (default: {defaults['image_model']})")
    parser.add_argument("--vlm-model", type=str, default=defaults["vlm_model"],
                        help=f"Vision model for scoring/captioning (default: {defaults['vlm_model']})")
    parser.add_argument("--prompt-model", type=str, default=defaults["prompt_model"],
                        help=f"LLM for prompt synthesis (default: {defaults['prompt_model']})")
    parser.add_argument("-s", "--size", type=str, default=defaults["size"],
                        help=f"Image size (default: {defaults['size']})")
    parser.add_argument("-q", "--quality", type=str, default=defaults["quality"],
                        help=f"Image quality (default: {defaults['quality']})")
    parser.add_argument("--max-urls", type=int, default=defaults["max_cited_urls"],
                        help=f"Max cited URLs to scrape (default: {defaults['max_cited_urls']})")
    args = parser.parse_args()

    # Validate run_dir up-front so we can mirror logs into its __rome__/.
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"ERROR: Not a directory: {run_dir}", file=sys.stderr)
        return 1
    rome_dir = run_dir / "__rome__"
    rome_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger()
    logger.configure({
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "console": True,
        # Mirror logs into the run's __rome__/ alongside the agent's console.log
        # so the image-gen pass is auditable from the same place as the run.
        "base_dir": str(rome_dir),
        "filename": "image_generator.log",
    })

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        return 1

    try:
        config = {
            "refs_per_image": args.references,
            "max_cited_urls": args.max_urls,
            "image_model": args.model,
            "vlm_model": args.vlm_model,
            "prompt_model": args.prompt_model,
            "size": args.size,
            "quality": args.quality,
        }
        if args.num_images < 1:
            print(f"ERROR: -n/--num-images must be >= 1 (got {args.num_images})",
                  file=sys.stderr)
            return 1

        # Load the run, extract just the ARTIFACT body for section-aware
        # embedding, then write a `<stem>.with-images.txt` sibling preserving
        # the surrounding ABSTRACT/SOURCES sections. End-to-end on a finished
        # experiment: scrape → score → caption → per-chunk synth + render →
        # markdown splice → save.
        gen = ImageGenerator(config=config)
        bundle = gen._load_run(run_dir)
        artifact_path = Path(bundle["artifact_path"])
        body = _extract_artifact_body(bundle["artifact"]) or bundle["artifact"]
        images_dir = (Path(args.output).parent if args.output
                      else artifact_path.parent / "images")
        new_body = embed_images_in_artifact(
            artifact_text=body, urls=bundle["urls"],
            output_dir=images_dir, agent_id=artifact_path.stem,
            n=args.num_images, **config)
        if not new_body:
            print("ERROR: image embedding failed (see logs)", file=sys.stderr)
            return 1
        out_path = (Path(args.output) if args.output
                    else artifact_path.with_name(
                        f"{artifact_path.stem}.with-images.txt"))
        out_path.write_text(
            _splice_artifact_body(bundle["artifact"], new_body),
            encoding="utf-8")
        n_imgs = new_body.count("![Figure")
        print(f"\nEmbedded {n_imgs} image(s); wrote {out_path}")
        print(f"Images saved under: {images_dir}")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
