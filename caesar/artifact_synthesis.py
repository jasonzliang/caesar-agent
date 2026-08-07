"""Caesar Synthesizer - Logic for synthesizing exploration artifacts"""
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator

from rome.config import set_attributes_from_config
from rome.llm_handler import FatalLLMError
from rome.logger import get_logger
from .caesar_config import (CAESAR_CONFIG, MAX_SYNTHESIS_QUERY_SOURCES,
    MAX_SYNTHESIS_QA_CONTEXT, NUM_SYNTHESIS_RETRIES, SYNTHESIS_SAVE_JSON)


class ArtifactSynthesizer:
    """Handles the synthesis of insights into final artifacts"""

    def __init__(self, agent, config: Dict = None):
        self.agent = agent
        self.logger = get_logger()

        self.config = config or {}
        set_attributes_from_config(self, self.config, CAESAR_CONFIG['ArtifactSynthesizer'].keys())

        self.kb_manager = self.agent.kb_manager; self.filters = None

        # Files written by _save_synthesis in the current synthesize_artifact() call.
        self.saved_artifact_files = []

        # Monotonic reasoning-effort floor for SYNTHESIS draft calls: a "high"
        # timeout steps down and persists so later drafts don't re-burn it.
        # Other labels skip the floor via use_floor in _llm_call.
        self._reasoning_floor: Dict[str, str] = {}

        # KB query filter: exclude insights from iterations >= synthesis_iteration_filter.
        if self.synthesis_iteration_filter:
            self.logger.assert_true(
                isinstance(self.synthesis_iteration_filter, int) and self.synthesis_iteration_filter > 0,
                "synthesis_iteration_filter must be an non-negative integer")
            self.filters = MetadataFilters(
                filters=[MetadataFilter(
                    key="iteration",
                    value=self.synthesis_iteration_filter,
                    operator=FilterOperator.LT)]
                )

    # ── Shared helpers ────────────────────────────────────────────────────

    def _llm_call(self, prompt: str, required_keys: List[str],
                  retries: int = NUM_SYNTHESIS_RETRIES,
                  reasoning: str = None, label: str = "LLM") -> Optional[Dict]:
        """LLM call with JSON parsing, key validation, and retries.

        Reasoning step-down ladder: each retry drops one rung, floor at "low".
        For an original "high" call the ladder runs high → medium → low → low …
        Quality drops with each step, but a lower-effort artifact in 60s is
        far better than burning the full retry budget on repeated 900s
        timeouts. The first attempt still gets the caller's preferred effort
        so we don't sacrifice quality on the happy path.

        Step-downs persist across calls via self._reasoning_floor. Once a
        timeout forces a step-down to medium (or lower) for a given original
        effort, subsequent calls passing the same `reasoning=` argument start
        at that floor instead of re-burning a full 900s timeout discovering
        high doesn't work for this synthesis run. Monotonic: never restored
        upward within a synthesizer's lifetime.
        """
        # Ensure prompt mentions JSON (required by OpenAI for json_object response format)
        if "json" not in prompt.lower():
            prompt = f"{prompt}\n\nRespond in JSON format."
        REASONING_LADDER = ["high", "medium", "low"]
        # Floor is SYNTHESIS-only. MERGE/CLARIFY/REFINE/POST-PROCESS/
        # NEXT_QUERY are one-shot calls that shouldn't inherit a draft-
        # phase timeout's downgrade.
        use_floor = (label == "SYNTHESIS")
        original_reasoning = reasoning
        start_reasoning = (
            self._reasoning_floor.get(reasoning, reasoning)
            if (reasoning and use_floor) else reasoning
        )
        if start_reasoning != original_reasoning:
            self.logger.info(
                f"[{label}] Starting at reasoning_effort={start_reasoning} "
                f"(floor from prior timeout; original requested {original_reasoning})"
            )
        for attempt in range(1, retries + 1):
            if start_reasoning in REASONING_LADDER:
                idx = min(REASONING_LADDER.index(start_reasoning) + (attempt - 1),
                          len(REASONING_LADDER) - 1)
                effective_reasoning = REASONING_LADDER[idx]
            else:
                effective_reasoning = start_reasoning
            if attempt > 1 and effective_reasoning != start_reasoning:
                self.logger.info(
                    f"[{label}] Retry {attempt}/{retries}: "
                    f"stepping reasoning_effort → {effective_reasoning} "
                    f"(orig {original_reasoning})"
                )
            override = {"reasoning_effort": effective_reasoning} if effective_reasoning else {}
            try:
                # num_retries=0: this wrapper owns the retry strategy (with the
                # reasoning_effort step-down above), so the lower-tier litellm/
                # openai-SDK retries must not silently amplify a hung call —
                # which they do up to N×timeout when num_retries > 0.
                response = self.agent.chat_completion(prompt,
                    override_config=override,
                    response_format={"type": "json_object"},
                    num_retries=0)
                result = self.agent.parse_json_response(response)
                if not result:
                    raise ValueError("parse_json_response returned None/empty")
                missing = [k for k in required_keys if k not in result]
                if missing:
                    raise ValueError(f"Missing required keys: {missing}")
                # Persist on step-down (SYNTHESIS-only — see use_floor above).
                if (use_floor
                        and original_reasoning in REASONING_LADDER
                        and effective_reasoning != original_reasoning
                        and effective_reasoning in REASONING_LADDER):
                    prior = self._reasoning_floor.get(original_reasoning, original_reasoning)
                    # Only update if this floor is lower than what we'd already remember.
                    if (REASONING_LADDER.index(effective_reasoning)
                            > REASONING_LADDER.index(prior)):
                        self._reasoning_floor[original_reasoning] = effective_reasoning
                        self.logger.info(
                            f"[{label}] Persisted reasoning_effort floor: "
                            f"{original_reasoning} → {effective_reasoning} "
                            f"(future synthesis calls will start here)"
                        )
                return result
            except FatalLLMError:
                raise  # don't waste retries on auth/quota errors
            except Exception as e:
                self.logger.error(f"[{label}] Attempt {attempt}/{retries} failed: {e}")
        self.logger.error(f"[{label}] All attempts exhausted")
        return None

    @staticmethod
    def _format_source_list(sources: Dict[str, int]) -> str:
        """Format a {url: idx} dict as a sorted '[idx] url' string."""
        return "\n".join(f"[{idx}] {url}"
            for url, idx in sorted(sources.items(), key=lambda x: x[1]))

    def _load_reference_draft(self, path: str) -> Optional[str]:
        """Load text content from a reference answer file.

        Accepts either Caesar's saved synthesis JSON (extracts the "artifact"
        field) or a plain text file (returned as-is). Citation markers are
        stripped — when reused cross-run, their indices don't match the new
        run's source_map and would become fabricated attributions. Returns
        None on failure.
        """
        if not os.path.exists(path):
            self.logger.error(f"[SYNTHESIS] Reference draft path not found: {path}")
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError as e:
            self.logger.error(f"[SYNTHESIS] Failed to read reference draft {path}: {e}")
            return None
        try:
            data = json.loads(content)
            if isinstance(data, dict) and isinstance(data.get("artifact"), str) and data["artifact"].strip():
                return re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", "", data["artifact"])
        except (json.JSONDecodeError, ValueError):
            pass
        content = content.strip()
        return re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", "", content) if content else None

    def _get_artifact_text(self, result: Dict) -> Optional[str]:
        """Extract artifact text with abstract prepended, or None if empty."""
        artifact_text = result.get("artifact", "")
        if not artifact_text:
            self.logger.error("[POST-PROCESS] No artifact text to process")
            return None
        abstract_text = result.get("abstract", "")
        if abstract_text:
            artifact_text = f"Artifact Abstract:\n{abstract_text}\n\nArtifact Text:\n{artifact_text}"
        return artifact_text

    def _get_query_context(self) -> Tuple[str, str]:
        """Return (query_context, query_role) strings used in prompts."""
        q = self.agent.starting_query
        query_context = f" that creatively answers this query: {q}" if q else ""
        query_role = f" to the query creatively!" if q else "!"
        return query_context, query_role

    def _post_process(self, result: Dict, prompt: str, response_key: str,
                      label: str, base_dir: str = None, suffix: str = "post",
                      timestamp: str = None) -> Optional[Dict]:
        """Shared post-process: LLM call → wrap as artifact → save."""
        llm_result = self._llm_call(prompt, [response_key],
            reasoning="high", label=label)
        if not llm_result:
            return None
        processed = {
            "artifact": llm_result[response_key],
            "metadata": result.get("metadata", {}),
        }
        self._save_synthesis(processed, base_dir=base_dir,
            suffix=suffix, timestamp=timestamp)
        return processed

    # ── Main synthesis entry point ────────────────────────────────────────

    def _shutdown_requested(self) -> bool:
        """Whether the agent has been asked to stop.

        The exploration loop has honored this since it existed, but synthesis
        never checked it, so an agent asked to stop during synthesis kept working
        for its full multi-draft run: measured at 21 minutes past the request on a
        live run. That made the web server unable to hand a run directory over
        (its takeover has to wait for the thread) and left a graceful server
        shutdown with nothing to do but abandon the thread.
        """
        return bool(getattr(self.agent, 'shutdown_called', False))

    def synthesize_artifact(self, num_drafts: int = None) -> None:
        """Generate final synthesis with optional multi-draft refinement.

        Returns the final result dict. In addition to the synthesis fields
        (abstract/artifact/sources/metadata), the dict includes:
          - artifact_dir: directory where synthesis files were written
          - artifact_files: list of full paths written during this call
          - num_drafts: how many drafts were produced
        """
        if not num_drafts: num_drafts = self.synthesis_drafts
        num_drafts = max(num_drafts, 1)

        # Reset per-call tracking of files written by _save_synthesis
        self.saved_artifact_files = []

        mode = f"iterative (n={self.synthesis_iterations})" if not self.synthesis_classic_mode else "classic"
        self.logger.info(f"[SYNTHESIS] Using {mode} mode with {num_drafts} draft(s)")

        if self.kb_manager.size() == 0:
            return {"abstract": "", "artifact": "No insights collected during exploration.",
                    "artifact_dir": None, "artifact_files": [], "num_drafts": 0}

        # Multi-draft synthesis loop
        current_query = self.agent.starting_query
        all_drafts = []; previous_result = None; base_dir = None
        if num_drafts > 1:
            base_dir = os.path.join(self.agent.get_repo(),
                f"{self.agent.get_id()}.synthesis.{datetime.now().strftime("%m%d%H%M")}")
        artifact_dir = base_dir if base_dir else self.agent.get_repo()

        # Optional: seed draft 1 with an external reference answer.
        if self.synthesis_reference_draft:
            ref_text = self._load_reference_draft(self.synthesis_reference_draft)
            if ref_text:
                self.logger.info(
                    f"[SYNTHESIS] Conditioning draft 1 on reference: "
                    f"{self.synthesis_reference_draft} ({len(ref_text)} chars)")
                previous_result = {"artifact": ref_text, "is_external_ref": True,
                    **({"query": self.synthesis_reference_query}
                       if self.synthesis_reference_query else {})}

        self._num_drafts = num_drafts
        for draft_num in range(1, num_drafts + 1):
            if self._shutdown_requested():
                self.logger.info(
                    f"[SYNTHESIS] Shutdown requested; stopping before draft {draft_num}")
                break
            self._current_draft = draft_num
            self.logger.info(f"\n{'='*80}\n[SYNTHESIS DRAFT {draft_num}/{num_drafts}]\n{'='*80}")
            if current_query: self.logger.info(f"Current query: {current_query}")

            # Generate synthesis for current draft (with previous artifact context)
            result = self._synthesize_single_draft(mode, current_query, previous_result)
            if not result: break
            self._save_synthesis(result, base_dir=base_dir, suffix=f'synthesis-{draft_num}')
            self._post_process_eli5(result, base_dir=base_dir, suffix=f'synth-eli5-{draft_num}')
            self._post_process_human_eval(
                result, base_dir=base_dir, suffix=f'synth-human-eval-{draft_num}')
            all_drafts.append(result); previous_result = result

            # Refine query for next draft (if not last draft)
            if draft_num < num_drafts:
                current_query = self._refine_query(result)
                if not current_query: break

        if len(all_drafts) < 1:
            raise ValueError("No synthesis artifacts created!")
        elif len(all_drafts) < num_drafts:
            self.logger.error(f"Artifact synthesis ended early at draft: {len(all_drafts)}/{num_drafts}")

        # Merge artifacts if requested and multiple drafts exist. Skipped on
        # shutdown: merging is another long LLM call, and whatever resumes will
        # redo synthesis from draft 1 anyway, so it would only delay the stop.
        final = None
        final_suffix = f'synthesis-{len(all_drafts)}'
        if (self.synthesis_merge_artifacts and len(all_drafts) > 1
                and not self._shutdown_requested()):
            self.logger.info(f"\n{'='*80}\n[MERGING {len(all_drafts)} ARTIFACTS]\n{'='*80}")
            merged_result = self._merge_artifacts(all_drafts, base_dir=base_dir)
            if merged_result:
                final_suffix = f'merged-{len(all_drafts)}'
                self._save_synthesis(merged_result, base_dir=base_dir, suffix=final_suffix)
                self._post_process_eli5(merged_result, base_dir, suffix=f'merged-eli5-{len(all_drafts)}')
                self._post_process_human_eval(
                    merged_result, base_dir, suffix=f'merged-human-eval-{len(all_drafts)}')
                final = merged_result
        if final is None:
            final = all_drafts[-1]

        # Best-effort: generate N images and re-save the artifact under the
        # same suffix (new mtime → web server's latest-by-mtime selection
        # picks the image-embedded version).
        if self.synthesis_generate_images > 0 and not self._shutdown_requested():
            self._post_process_generate_images(
                final, base_dir=base_dir, n=self.synthesis_generate_images,
                suffix=final_suffix)

        final["artifact_dir"] = artifact_dir
        final["artifact_files"] = list(self.saved_artifact_files)
        final["num_drafts"] = len(all_drafts)
        return final

    # ── Single draft synthesis ────────────────────────────────────────────

    def _synthesize_single_draft(self, mode: str, current_query: Optional[str] = None,
                                 previous_artifact: Optional[Dict] = None) -> Dict[str, str]:
        """Execute a single synthesis draft with optional query and previous artifact"""
        self.logger.info("[SYNTHESIS] Generating synthesis artifact")

        qa_pairs = self._generate_qa_pairs(mode, current_query)
        if not qa_pairs:
            self.logger.error("[SYNTHESIS] Unable to generate Q/A pairs for artifact")
            return None
        qa_list, source_list, source_map = self._build_answers_with_citations(qa_pairs)

        query_context, query_role = self._get_query_context()
        length_context = f" ({self.synthesis_max_length} words)" if self.synthesis_max_length else ""


        # Build context from previous artifact if available. is_external_ref
        # distinguishes a cross-run seed (parent artifact in a follow-up run)
        # from an in-run prior draft — the former must NOT be "built upon" or
        # the new artifact paraphrases the parent instead of answering the
        # new query.
        previous_context = ""
        is_external_ref = bool(
            previous_artifact and previous_artifact.get("is_external_ref")
        )
        query_block = ""
        if is_external_ref and previous_artifact.get("query"):
            query_block = (
                f"\nPREVIOUS QUERY (what PREVIOUS ARTIFACT answered):\n"
                f"{previous_artifact['query']}\n"
            )
        if previous_artifact and previous_artifact.get("artifact"):
            previous_context = f"""{query_block}
PREVIOUS ARTIFACT:
{previous_artifact["artifact"]}
--- END OF ARTIFACT ---
"""

        prompt = f"""You explored {len(self.agent.visited_urls)} sources and gathered {self.kb_manager.size()} insights.

KEY INSIGHTS (with source citations):
{qa_list}
--- END OF INSIGHTS ---

SOURCES:
{source_list}
--- END OF SOURCES ---
{previous_context}
YOUR TASK:
Drawing heavily upon the patterns that emerged from the key insights{', and building upon the previous artifact,' if (previous_artifact and not is_external_ref) else ''} create a novel, exciting, and thought provoking artifact{query_context}

1. **Artifact Abstract** (80-120 words):
    - Summary of the artifact's core discovery and its significance

2. **Artifact Main Text**{length_context}:
    - IMPORTANT: Carefully analyze every relevant key insight to generate a comprehensive and detailed response
    - General guidelines for response:
        a. Emergent patterns not visible in individual insights
        b. Novel discoveries, connections, or applications
        c. Surprising new directions or perspectives
        d. Interesting tensions, contradictions, or open questions
    - Cite from "SOURCES" using [n] notation to support relevant claims and statements (e.g., "This pattern emerged... [1,3]") with up to {MAX_SYNTHESIS_QUERY_SOURCES} citations per claim, but do NOT create a "SOURCES" or "References" section
    {('- The previous artifact is the prior answer to an earlier question. Assume the reader has internalized its framework; do not restate its structure or definitions. Start directly from what the current query asks beyond that prior answer. Reuse established terminology where accurate; do not repeat, paraphrase, or extend prior exposition.' if is_external_ref else '- Build upon the previous artifact by analyzing it for weaknesses in organization, arguments, or content, and then use key insights to deepen, improve, and extend the previous artifact') if previous_artifact else ''}

{'IMPORTANT: do NOT mention or reference the previous artifact, the new artifact should make sense by itself as a standalone text' if previous_artifact else ''}
IMPORTANT: AVOID excessive jargon, ensure artifact text is well-organized (logical, clear, focused), and convincing to a skeptical reader
IMPORTANT: Use your role as a guide on how to respond {query_role}

Respond with valid JSON only:
{{
    "abstract": "<abstract text>",
    "artifact": "<artifact text>"
}}"""
        result = self._llm_call(prompt, ["abstract", "artifact"],
            reasoning="high", label="SYNTHESIS")
        if not result: return None

        result["sources"] = dict(sorted(source_map.items(), key=lambda x: x[1]))
        result["metadata"] = {
            "insights_collected": self.kb_manager.size(),
            "iterations_run": self.agent.current_iteration,
            "pages_visited": len(self.agent.visited_urls),
            "sources_cited": len(source_map),
            "starting_url": self.agent.starting_url,
            "starting_query": self.agent.starting_query,
            "synthesis_drafts": self.synthesis_drafts,
            "synthesis_eli5_length": self.synthesis_eli5_length,
            "synthesis_iteration_filter": self.synthesis_iteration_filter,
            "synthesis_max_length": self.synthesis_max_length,
            "synthesis_mode": mode,
            "synthesis_queries": len(qa_pairs),
        }

        return result

    # ── Query refinement ──────────────────────────────────────────────────

    def _refine_query(self, artifact_result: Dict, current_query: Optional[str] = None) -> Optional[str]:
        """Refine the synthesis query based on previous artifact"""

        artifact_text = artifact_result.get("artifact", "")
        if not artifact_text:
            self.logger.error("[REFINE] Cannot refine query: no artifact text")
            return None

        if not current_query: current_query = self.agent.starting_query
        query_context = f"PREVIOUS QUERY: {current_query}\n\n" if current_query else ""

        prompt = f"""{query_context}PREVIOUS ARTIFACT:
{artifact_text}
--- END OF ARTIFACT ---

YOUR TASK:
Based on the previous query and artifact above, identify the most promising direction for deeper exploration. What NEW question or angle would:
    - Build on the insights already discovered
    - Explore gaps, contradictions, or unexplored connections
    - Lead to novel perspectives or applications
    - Go deeper rather than broader

The refined query should be concise (1-2 sentences), straightforward, clear, and understandable.

IMPORTANT: Use your role as a guide on how to respond!

Respond with JSON:
{{
    "refined_query": "<your refined exploration query, posed as a question>",
    "reason": "<brief explanation of why the refined query improves upon the previous query>"
}}"""

        result = self._llm_call(prompt, ["refined_query"], retries=1, label="REFINE")
        if result:
            self.logger.info(f"[REFINE] Query: {result['refined_query']}\nReason: {result.get('reason', 'No reason provided')}")
            return result["refined_query"]
        return None

    # ── Artifact merging ──────────────────────────────────────────────────
    # TODO: Ring merge of drafts (feed random draft sequence in connected ring of unique agents)
    # TODO: Debate merge of drafts (multiple of rounds of unique agents critiquing drafts)

    _CITE_RE = re.compile(r'\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]')

    @classmethod
    def _remap_citations(cls, text: str, mapping: Dict[int, int],
                         drop_unmapped: bool = True) -> str:
        """Rewrite inline [n] / [n,m,...] markers via an int→int mapping.
        With drop_unmapped=True (default) indices missing from the mapping
        are dropped; with drop_unmapped=False they pass through unchanged
        so non-citation bracketed numbers (e.g. years like [2024]) survive."""
        def _sub(m):
            parts = []
            for tok in m.group(1).split(','):
                try:
                    n = int(tok.strip())
                except ValueError:
                    return m.group(0)
                g = mapping.get(n)
                if g is not None:
                    parts.append(str(g))
                elif not drop_unmapped:
                    parts.append(str(n))
            return f"[{','.join(parts)}]" if parts else ""
        return cls._CITE_RE.sub(_sub, text)

    @classmethod
    def _extract_cited_indices(cls, *texts: str) -> set:
        cited: set = set()
        for text in texts:
            if not text:
                continue
            for m in cls._CITE_RE.finditer(text):
                for tok in m.group(1).split(','):
                    try:
                        cited.add(int(tok.strip()))
                    except ValueError:
                        pass
        return cited

    def _merge_artifacts(self, all_drafts: List[Dict],
                         base_dir: str = None) -> Optional[Dict[str, str]]:
        """Merge artifacts from all drafts into a single comprehensive artifact.

        Citations are pre-numbered globally across drafts before the LLM sees
        them, so the LLM only has to preserve [N] markers verbatim. The final
        sources dict is reconstructed deterministically from the markers that
        survive merge (and clarify). This replaced an earlier design where the
        LLM renumbered/remapped sources itself and would silently drop URLs
        whenever it consolidated citations.
        """
        self.logger.info(f"[MERGE] Generating merged artifact")
        if len(all_drafts) == 0:
            self.logger.error("[MERGE] No artifacts to merge"); return None

        # Build deterministic global URL→index map (union across drafts,
        # ordered by first appearance).
        global_sources: Dict[str, int] = {}
        next_idx = 1
        for r in all_drafts:
            coerced = []
            for url, idx in r.get('sources', {}).items():
                if not url:
                    continue
                try:
                    coerced.append((url, int(idx)))
                except (TypeError, ValueError):
                    continue
            for url, _ in sorted(coerced, key=lambda x: x[1]):
                if url not in global_sources:
                    global_sources[url] = next_idx
                    next_idx += 1
        idx_to_url = {idx: url for url, idx in global_sources.items()}

        # Rewrite each draft's inline citations into the global numbering
        # before showing it to the LLM.
        artifacts_context = []
        for i, r in enumerate(all_drafts, 1):
            local_to_global = {}
            for url, idx in r.get('sources', {}).items():
                if url not in global_sources:
                    continue
                try:
                    local_to_global[int(idx)] = global_sources[url]
                except (TypeError, ValueError):
                    continue
            rewritten = self._remap_citations(r['artifact'], local_to_global,
                drop_unmapped=False)
            artifacts_context.append(
                f"--- DRAFT {i} ---\n\n"
                f"ARTIFACT:\n{rewritten}\n\n"
                f"--- END OF DRAFT {i} ---\n\n"
            )
        artifacts_text = "\n\n".join(artifacts_context)
        global_source_list = self._format_source_list(global_sources)

        query_context, query_role = self._get_query_context()
        length_context1 = f"{self.synthesis_max_length} words" if self.synthesis_max_length else ">= average draft artifact length"
        length_context2 = f" ({self.synthesis_max_length} words)" if self.synthesis_max_length else ""

        prompt = f"""You are merging {len(all_drafts)} drafts of research artifacts into one unified artifact{query_context}

=== GLOBAL SOURCES (shared numbering used by every draft below) ===
{global_source_list}
=== END OF GLOBAL SOURCES ===

=== RESEARCH ARTIFACTS ===
{artifacts_text}
=== END OF RESEARCH ARTIFACTS ===

YOUR TASK:
Create a comprehensive merged artifact that:
    - Fuse the draft artifacts into one standalone work with a single clear narrative spine (not a stitched summary).
    - Curate and reinterpret the highest‑leverage insights across all drafts; cut redundancy while preserving essential caveats.
    - Derive new patterns/tensions across all drafts and at least one unifying framework that doesn't appear in any single artifact.
    - Strengthen and extend the result: tighten structure, resolve/reconcile contradictions, and push into implications, applications, and open questions (flag speculation explicitly).

MERGED ARTIFACT CITATIONS:
    - Citations have ALREADY been remapped to the GLOBAL numbering shown in GLOBAL SOURCES above; the same [N] refers to the same URL in every draft.
    - Preserve [N] citation markers EXACTLY as they appear in the drafts (e.g., [1], [2,5,7]). Do NOT renumber, dedupe, consolidate, or invent new numbers.
    - Carry over citations from claims you keep; you may use up to {MAX_SYNTHESIS_QUERY_SOURCES} citations per claim. Do NOT create a "Sources" or "References" section.

MERGED ARTIFACT TEXT:
    - IMPORTANT: Avoid excessive jargon, ensure artifact text is well-organized (logical, clear, focused), and convincing to a skeptical reader
    - Merged artifact length: {length_context1}
    - Do NOT mention "Draft 1", "Draft 2", etc, in text

RESPONSE INSTRUCTIONS:
    - IMPORTANT: Use your role as a guide on how to respond{query_role}
    - Your response must ONLY be valid JSON starting with {{ and ending with }}
    - Following 2 fields required: abstract, artifact

EXAMPLE OUTPUT:
{{
    "abstract": "A 80-120 word summary of the artifact's core discovery and its significance",
    "artifact": "Full merged text{length_context2} with citations [1,2]..."
}}
"""

        result = self._llm_call(prompt, ["abstract", "artifact"],
            reasoning="high", label="MERGE")
        if not result: return None

        result["metadata"] = all_drafts[-1].get("metadata", {})

        # Clarity post-process (preserves [N] markers per its prompt contract).
        # _post_process_clarify is gated internally by
        # self.synthesis_merge_clarify; it returns None when disabled.
        clarified = self._post_process_clarify(result)
        if clarified and clarified.get("artifact"):
            result["artifact"] = clarified["artifact"]
            if clarified.get("abstract"):
                result["abstract"] = clarified["abstract"]

        # Reconstruct sources from [N] markers that survived merge + clarify,
        # then renumber to sequential 1..N for backward compat with draft
        # output shape (which is always sequential from 1).
        all_cited = self._extract_cited_indices(
            result.get("artifact", ""), result.get("abstract", ""))
        cited_globals = {i for i in all_cited if i in idx_to_url}
        orphan_markers = all_cited - cited_globals
        if orphan_markers:
            # Markers the LLM emitted that don't correspond to a known source
            # (e.g. a bracketed year, or an index hallucinated outside the
            # global range). We preserve them in text (drop_unmapped=False)
            # to match the previous code's tolerance for non-citation
            # brackets, but they cannot appear in the sources dict.
            self.logger.error(
                f"[MERGE] {len(orphan_markers)} marker(s) in merged text "
                f"don't map to a global source: {sorted(orphan_markers)[:10]}"
            )
        global_to_seq = {g: f for f, g in enumerate(sorted(cited_globals), 1)}
        result["artifact"] = self._remap_citations(
            result.get("artifact", ""), global_to_seq, drop_unmapped=False)
        result["abstract"] = self._remap_citations(
            result.get("abstract", ""), global_to_seq, drop_unmapped=False)
        result["sources"] = {
            idx_to_url[g]: seq for g, seq in sorted(global_to_seq.items(), key=lambda x: x[1])
        }
        if isinstance(result.get("metadata"), dict):
            result["metadata"]["sources_cited"] = len(result["sources"])

        # Normalize formatting on the final merged text so the saved artifact
        # has consistent Markdown whether clarify ran, was gated off, or
        # returned None. _normalize_formatting is idempotent.
        result["artifact"] = self._normalize_formatting(result.get("artifact") or "")
        result["abstract"] = self._normalize_formatting(result.get("abstract") or "")

        self.logger.info(
            f"[MERGE] Sources reconstructed: {len(result['sources'])} cited "
            f"/ {len(global_sources)} in draft union"
        )

        return result

    # ── Q&A pair generation ───────────────────────────────────────────────

    def _generate_qa_pairs(self, mode: str, starting_query: str = None) -> List[Tuple[str, str, List[Dict]]]:
        """Generate Q&A pairs with sources"""
        queries = [
            "What are the central themes and patterns discovered?",
            "What unexpected connections or insights emerged?",
            "What contradictions or tensions were revealed?",
            "What questions remain open or were raised?",
            "What novel perspective emerged from this exploration?"
        ]
        if starting_query: queries = [starting_query] + queries
        if mode == "classic": return self._generate_qa_pairs_classic(queries)

        # Validate iterations
        # if not isinstance(self.synthesis_iterations, int) or self.synthesis_iterations < 1:
        #     self.logger.error(f"[SYNTHESIS] Invalid synthesis_iterations: {self.synthesis_iterations}, using 1")
        #     self.synthesis_iterations = 1

        queries, answers, sources_list = [queries[0]], [], []

        for i in range(self.synthesis_iterations):
            # Finest-grained exit point in synthesis: one iteration is a KB query
            # plus two LLM calls, so this bounds how long a stop request waits.
            if self._shutdown_requested():
                self.logger.info(
                    f"[SYNTHESIS] Shutdown requested; stopping at iteration {i+1}")
                break
            answer, sources = self.kb_manager.query(
                queries[-1],
                top_k=self.synthesis_top_k,
                top_n=self.synthesis_top_n,
                return_sources=True,
                filters=self.filters)

            # Check if iteration filter is working properly
            if self.filters and sources:
                iters = [s.get('iteration') for s in sources]
                violations = [it for it in iters if isinstance(it, int) and it >= self.synthesis_iteration_filter]
                self.logger.assert_true(not violations,
                    f"Iteration filter failed: {violations} >= {self.synthesis_iteration_filter}")

            iter_str = f"[SYNTHESIS DRAFT {self._current_draft}/{self._num_drafts} ITERATION {i+1}/{self.synthesis_iterations}]"
            if not answer:
                # Skip iter on transient KB failure rather than abort draft.
                self.logger.error(f"{iter_str} KB query failed; skipping iter")
                continue
            if not isinstance(sources, list): sources = []

            answers.append(answer)
            sources_list.append(sources)
            self.logger.info(f"{iter_str}\nQ: {queries[-1]}\nA: {answers[-1]}")

            if i >= self.synthesis_iterations - 1: break

            next_query = self._generate_next_query(queries, answers)
            if not next_query:
                self.logger.error(f"{iter_str} Next query failed"); break
            queries.append(next_query)

        return list(zip(queries, answers, sources_list))

    def _generate_qa_pairs_classic(self, queries: List[str]) -> List[Tuple[str, str, List[Dict]]]:
        """Execute queries against KB with sources"""
        qa_pairs = []
        for query in queries:
            try:
                answer, sources = self.kb_manager.query(
                    query,
                    top_k=self.synthesis_top_k,
                    top_n=self.synthesis_top_n,
                    return_sources=True,
                    filters=self.filters)
                if answer:
                    qa_pairs.append((query, answer, sources))
            except Exception as e:
                self.logger.error(f"KB query failed for '{query}': {e}")
        return qa_pairs

    def _generate_next_query(self, previous_queries: List[str],
                             previous_responses: List[str]) -> Optional[str]:
        """Generate next synthesis query"""
        recent_queries = previous_queries[-MAX_SYNTHESIS_QA_CONTEXT:]
        recent_responses = previous_responses[-MAX_SYNTHESIS_QA_CONTEXT:]

        init_query = self.agent.starting_query if self.agent.starting_query else previous_queries[0]
        prev_insights = "\n\n".join([f"Q: {q}\nA: {r}"
            for q, r in zip(recent_queries, recent_responses)])

        prompt = f"""INITIAL QUERY:
{init_query}

PREVIOUS INSIGHTS:
{prev_insights}

YOUR TASK:
Based on the initial query and previous insights gathered so far, what is the next most important question to ask to deepen understanding and reveal emergent patterns? The question should:
- Build on past insights rather than repeat them
- Seek connections between different themes
- Identify gaps or contradictions to explore
- Move toward synthesis and creation rather than enumeration

IMPORTANT: Use your role as a guide on how to respond!

Respond with JSON:
{{
    "query": "<your next question>",
    "reason": "<brief explanation of why this question deepens understanding>"
}}"""

        result = self._llm_call(prompt, ["query"], retries=1, label="NEXT_QUERY")
        return result["query"] if result else None

    # ── Citation building ─────────────────────────────────────────────────

    def _build_answers_with_citations(self, qa_pairs):
        """Build formatted answers with citations and source index from Q&A pairs."""
        source_map = {}; answers = []

        for i, (q, a, sources) in enumerate(qa_pairs):
            # Add new sources to index (exclude file:// URLs)
            for src in sources:
                if (url := src['url']) and not url.startswith('file://') and url not in source_map:
                    source_map[url] = len(source_map) + 1

            # Format with citations (only for URLs in source_map)
            refs = ", ".join([f"[{source_map[s['url']]}]"
                for s in sources[:MAX_SYNTHESIS_QUERY_SOURCES] if s.get('url') and s['url'] in source_map])
            answers.append(f"({i+1}) Question: {q}\n\nAnswer: {a} {refs}")

        qa_list = "\n\n\n".join(answers)
        source_list = self._format_source_list(source_map)

        return qa_list, source_list, source_map

    # ── Saving ────────────────────────────────────────────────────────────

    def _save_synthesis(self, result: Dict,
            base_dir: str = None,
            suffix: str = "synthesis",
            timestamp: str = None) -> None:
        """Save synthesis with sources in JSON and text formats"""
        if not base_dir: base_dir = self.agent.get_repo()
        if not timestamp: timestamp = datetime.now().strftime("%m%d%H%M")
        os.makedirs(base_dir, exist_ok=True)
        base_path = os.path.join(base_dir, f"{self.agent.get_id()}.{suffix}.{timestamp}")
        meta_path = os.path.join(base_dir, "metadata.txt")

        try:
            if SYNTHESIS_SAVE_JSON:
                with open(f"{base_path}.json", 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=4, sort_keys=True)
                self.saved_artifact_files.append(f"{base_path}.json")

            with open(f"{base_path}.txt", 'w', encoding='utf-8') as f:
                if abstract := result.get('abstract'):
                    f.write(f"ABSTRACT:\n{abstract}\n\n")

                f.write(f"ARTIFACT:\n{result['artifact']}\n\n")

                if sources := result.get('sources'):
                    f.write(f"SOURCES:\n{self._format_source_list(sources)}\n\n")
            self.saved_artifact_files.append(f"{base_path}.txt")

            if metadata := result.get('metadata'):
                with open(meta_path, 'a', encoding='utf-8') as f:
                    f.write(f"{base_path}\n{json.dumps(metadata, indent=4, sort_keys=True)}\n\n")

        except Exception as e:
            self.logger.error(f"Failed to save synthesis: {e}")

    # ── Post-processing ───────────────────────────────────────────────────

    def _post_process_clarify(self, result: Dict) -> Optional[Dict]:
        """Polish abstract + artifact for plain-language clarity while preserving
        all claims, numbers, and citations. Returns {"abstract", "artifact"} or
        None when synthesis_merge_clarify is False / no artifact text. Reads
        result["abstract"] and result["artifact"] directly (NOT _get_artifact_text,
        which would concatenate them and cause duplication on save)."""
        if not self.synthesis_merge_clarify: return None

        artifact_text = result.get("artifact", "")
        if not artifact_text:
            self.logger.error("[CLARIFY] No artifact text to clarify")
            return None
        abstract_text = result.get("abstract", "")

        self.logger.info("[CLARIFY] Polishing merged abstract and artifact for clarity")

        prompt = f"""--- ABSTRACT ---
{abstract_text}
--- END OF ABSTRACT ---

--- ARTIFACT ---
{artifact_text}
--- END OF ARTIFACT ---

YOUR TASK:
Rewrite BOTH the abstract and the artifact above to be clearer and easier
to read for a non-expert but college-educated reader, WITHOUT removing
information or changing meaning. Use the SAME plain-language style across
both so the summary and the body feel consistent.

CONTENT - preserve, do not strip:
 - MUST KEEP every claim, fact, number, recommendation, and named concept.
   Every number appears verbatim; do not round, summarize, or paraphrase
   (no "approximately X").
 - MUST KEEP every citation marker EXACTLY as written (e.g., [1], [2,5,7]).
 - Remove only empty connective filler; never drop a claim, number,
   citation, recommendation, or named concept to shorten the text.
 - Tighten redundant phrasing. Introduce each concept once; refer back
   briefly rather than re-explaining it across sections. Prefer the
   shortest formulation that preserves the meaning.
 - The abstract should remain a concise summary paragraph.

LANGUAGE - translate jargon, do not strip:
 - Replace consultant/business/technical jargon with plain English. If a
   term is load-bearing, keep it and add a brief parenthetical gloss on
   first use. Examples:
     "rate compression" → "pressure to lower hourly prices"
     "control plane" → "central control system"
     "blast radius" → "scope of damage if it goes wrong"
     "race to zero" → "competitive price collapse"
 - Re-gloss specialized terms when they reappear in distant sections;
   better to gloss twice than strand the reader.
 - Spell out acronyms on first use; use the acronym alone after.
 - Remove filler phrases that add nothing ("at the end of the day",
   "in essence").

FORMATTING - clean Markdown, do not decorate:
 - Use **bold** for emphasis; do NOT use *italics* or _underscore italics_.
 - Use `-` for every bullet (not `*` or `+`); indent sub-bullets by 2 spaces.
 - Preserve heading levels exactly; do not demote, promote, or merge.
 - Separate paragraphs, lists, and headings with one blank line.
 - Use ordered lists (`1.` `2.` `3.`) only when sequence or ranking matters.
 - Fence multi-line code with ```triple-backticks```.
 - Do NOT add a "Sources" or "References" section; citations stay inline as [N].
 - Do NOT introduce horizontal rules, emojis, or decorative headings.

Respond with valid JSON only:
{{
    "clarified_abstract": "<the rewritten abstract text>",
    "clarified_artifact": "<the rewritten artifact text in markdown>"
}}"""

        llm_result = self._llm_call(prompt,
            ["clarified_abstract", "clarified_artifact"],
            reasoning="high", label="CLARIFY")
        if not llm_result: return None

        new_artifact = llm_result.get("clarified_artifact") or ""
        if not new_artifact:
            self.logger.error("[CLARIFY] LLM returned empty clarified artifact")
            return None

        # Abstract is best-effort: if the LLM returns an empty string, fall
        # back to the original abstract rather than blanking it out.
        new_abstract = llm_result.get("clarified_abstract") or abstract_text

        return {
            "abstract": self._normalize_formatting(new_abstract),
            "artifact": self._normalize_formatting(new_artifact),
        }

    @staticmethod
    def _normalize_formatting(text: str) -> str:
        """Deterministic Markdown cleanup: fix bold/italic nesting, strip trailing whitespace, collapse blank-line runs."""
        # 1. Collapse bold-wrapping-italic (**Foo *Bar***) -> **Foo Bar**
        text = re.sub(r'\*\*([^*\n]*?)\*([^*\n]+?)\*([^*\n]*?)\*\*',
                      r'**\1\2\3**', text)
        # 2. Convert standalone italic (*foo*) to bold (**foo**)
        text = re.sub(r'(?<!\*)\*([^*\n]+?)\*(?!\*)', r'**\1**', text)
        # 3. Strip trailing whitespace on every line
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
        # 4. Collapse runs of >2 blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    def _post_process_eli5(self, result: Dict,
        base_dir: str = None,
        suffix: str = "eli5",
        timestamp: str = None) -> Optional[Dict]:
        """Generate ELI5 explanation(s) of artifact and save them"""
        if not self.synthesis_eli5: return None

        artifact_text = self._get_artifact_text(result)
        if not artifact_text: return None

        # Support single int, list of ints, or None
        word_lengths = ([self.synthesis_eli5_length] if isinstance(self.synthesis_eli5_length, int)
                        else self.synthesis_eli5_length if isinstance(self.synthesis_eli5_length, list)
                        else [None])

        results = []
        for length in word_lengths:
            self.logger.info(f"[POST-PROCESS] Generating {length or 'unconstrained'} word ELI5")

            # Sanitize length suffix string and setup prompt context
            # re.sub(r'[<>:\"/\\|?*\s]', '-', str(length))
            current_suffix = f"{suffix}.{re.sub(r'\D', '', str(length))}w" if length else suffix
            length_context = f"\nIMPORTANT: Your explanation MUST be {length} words, double check to make sure" if length else ""

            prompt = f"""--- ARTIFACT ---
{artifact_text}
--- END OF ARTIFACT ---

YOUR TASK:
"Explain Like I'm 5" (ELI5) the artifact above:
 - Do NOT mention or reference the artifact, your explanation should be a standalone text
 - Your target audience is a non-expert but college educated reader
 - Capture the main ideas without oversimplifying
 - Clarify any confusing or convoluted parts of the artifact
 - Your explanation should start with a short but interesting title
{length_context}
IMPORTANT: Format your explanation so humans can easily read it (use more than one big paragraph)

Respond with valid JSON only:
{{
    "eli5": "<your ELI5 explanation>"
}}"""

            processed = self._post_process(result, prompt, "eli5",
                label="POST-PROCESS", base_dir=base_dir,
                suffix=current_suffix, timestamp=timestamp)
            if processed: results.append(processed)
            else: self.logger.error(f"[POST-PROCESS] {length or 'unconstrained'} ELI5 generation failed")

        return results[-1] if results else None

    def _post_process_human_eval(self, result: Dict,
        base_dir: str = None,
        suffix: str = "human-eval",
        timestamp: str = None) -> Optional[Dict]:
        """Generate a two-paragraph human eval summary: core idea + creativity argument"""
        if not self.synthesis_human_eval: return None

        artifact_text = self._get_artifact_text(result)
        if not artifact_text: return None

        prompt = f"""--- ARTIFACT ---
{artifact_text}
--- END OF ARTIFACT ---

YOUR TASK:
Write exactly two short paragraphs about the artifact above:

PARAGRAPH 1 - CORE IDEA SUMMARY (2-3 sentences):
 - Identify and summarize the most important creative key idea in the artifact
 - Make sure the summary is clear and understandable to a non-expert
 - The reader should immediately understand what the artifact is about
 - Do not overuse jargon or too many technical terms

PARAGRAPH 2 - CREATIVITY ARGUMENT (3-4 sentences):
 - Make a clear, convincing argument for why the key idea in the artifact is creative
 - Explain what makes it novel, surprising, or useful
 - Ground your argument in specific aspects of the idea, not generic praise
 - Write to persuade a skeptical reader

IMPORTANT: Do NOT mention or reference "the artifact"; write as if describing the idea itself
IMPORTANT: Start paragraph 1 with "SUMMARY: " and paragraph 2 with "WHY ITS CREATIVE: "

Respond with valid JSON only:
{{
    "human_eval": "SUMMARY: <paragraph 1>\\n\\nWHY ITS CREATIVE: <paragraph 2>"
}}"""

        return self._post_process(result, prompt, "human_eval",
            label="POST-PROCESS", base_dir=base_dir,
            suffix=suffix, timestamp=timestamp)

    def _post_process_generate_images(self, result: Dict,
                                      base_dir: Optional[str],
                                      n: int,
                                      suffix: str) -> None:
        """Thin wire: generate N section-aware images and re-save the artifact with markdown refs spliced in.
        Best-effort — failures leave the saved synthesis canonical."""
        # Lazy import keeps synthesis working without curl_cffi/bs4/openai
        # when synthesis_generate_images=0.
        try:
            from caesar.image_generator import embed_images_in_artifact
        except Exception as e:
            self.logger.error(f"[IMAGE] Cannot import image_generator: {e}")
            return

        artifact_dir = base_dir or self.agent.get_repo()
        sources = result.get("sources") or {}
        urls = list(sources.keys()) if isinstance(sources, dict) else []
        # Forward the agent's merged ImageGenerator config (so preset YAMLs'
        # quality / max_candidates_to_score overrides reach ImageGenerator,
        # which otherwise reconstructs from module-level CAESAR_CONFIG only).
        agent_config = getattr(self.agent, "config", {}) or {}
        image_gen_kwargs = dict(agent_config.get("ImageGenerator", {}) or {})
        try:
            new_artifact = embed_images_in_artifact(
                artifact_text=result.get("artifact", ""), urls=urls,
                output_dir=os.path.join(artifact_dir, "images"),
                agent_id=self.agent.get_id(), n=n,
                llm_handler=getattr(self.agent, "llm_handler", None),
                **image_gen_kwargs)
        except Exception as e:
            self.logger.error(f"[IMAGE] Image embedding failed: {e}")
            return
        if not new_artifact:
            return
        result["artifact"] = new_artifact
        self._save_synthesis(result, base_dir=base_dir, suffix=suffix)
        self.logger.info("[IMAGE] Embedded image(s) and re-saved synthesis")
