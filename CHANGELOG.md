# Changelog

All notable changes to Caesar are documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.12] — 2026-08-07

### Fixed

- README claims corrected against the paper. The headline asserted statistical
  significance (`p < 0.001`, Mann-Whitney U) that arXiv:2604.20855v3 does not
  report -- Appendix B.5 states it uses magnitude-of-difference framing rather
  than null-hypothesis testing. The ablation summary also collapsed several
  distinct results into one effect size; it now reports each separately.
- Documented config defaults corrected. `caesar/README.md` listed five
  LLMHandler values from rome's `DEFAULT_CONFIG`, but Caesar deep-merges its own
  over them, so none were the effective values for a Caesar run.
  `rome/README.md` documented a `max_tokens` key and an
  `EditCodeAction.max_iterations` that nothing reads.
- `config_test/single_agent_test.yaml` pointed both role files at
  `config/role/`; the directory is `config/custom_role/`, so the config could
  not load.
- `--image-model` help advertised `gpt-image-1`; the default is `gpt-image-2`.
- Model IDs naming `claude-opus-4-7`, which exists nowhere in the code, and a
  rubric scale given as 0-10 where every shipped rubric specifies 1-10.

### Changed

- `release.sh` mirrors the tagged source to a public repo given by `--repo`
  (or `PUBLIC_REPO`), rewriting repo URLs to the mirror target and refusing to
  publish if internal identifiers, personal addresses or credential-shaped
  strings survive the scan.

## [0.4.11] — 2026-08-06

### Changed

- Release artifacts no longer carry deployment internals. `.gitattributes`
  marks `.github/`, `deploy/`, `test/` and `benchmark/` `export-ignore`, so
  the source archives GitHub generates from a tag stop shipping the ECR/OIDC
  workflow and the deploy manifests. `monitor/` is still included, and its
  two hardcoded hosts are gone: `download_exp.py` reads `ROME_REMOTE_HOST`
  and `cleanup_chroma.py` reads `ROME_ARTIFACT_ROOTS`, both overridable by
  flag as before. This affects archives cut from here on, not ones already
  attached to existing releases.
- `.dockerignore` excludes the run-output trees. `.gitignore` does not apply
  to `docker build`, so `COPY . /app` was baking roughly 6.8 GB of
  transcripts and evaluation records into every image.
- Contact addresses in `pyproject.toml` and `CITATION.cff` are now
  `jasonzliang@utexas.edu`; the maintainer address reaches PyPI in the
  wheel's metadata. The header's Feedback link is a mailto rather than a
  link into an internal wiki.

### Fixed

- `graph_password` no longer defaults to a literal. `config.py` shipped
  `neo4jneo4j` inside the published wheel, and because the value was always
  truthy, `agent_memory.py`'s "graph enabled but no password" guard could
  never fire. The password is now resolved in `AgentMemory.__init__` from
  `NEO4J_PASSWORD`, which keeps it out of `self.config` entirely — so no
  exported or logged config can carry it — and reads at construction rather
  than at import. An explicit config value still wins; unset fails closed
  with the vector store unaffected.

## [0.4.10] — 2026-07-31

### Added
- Restart button on the run page and on Past Runs rows, resuming from the run's checkpoint.

### Fixed
- The crawler refuses any address that is not the public internet: loopback, private, link-local and shared ranges are all rejected.
- The login password no longer appears in the process command line, where any local account could read it.
- Two long-standing job-pool bugs: a cancelled run leaked its pool entry forever, and a refused restart could leave two agents writing one run directory.

## [0.4.9] — 2026-07-30

### Added
- Admin step-up: entering the operator password elevates that browser session.
- Public runs survive a server restart — the per-run key is persisted so the run auto-resumes.
- Migration to the GPT-5.6 family across synthesis, exploration and KB.

## [0.4.8] — 2026-07-27

### Added
- Past Runs search bar with duration, cost and age filters.
- Knowledge Graph table view: sortable, searchable node table with neighbor popups.

### Fixed
- A Logger crash that masked real LLM errors — percent-style logging calls in the LLM error path raised `TypeError`, hiding the underlying failure.

## [0.4.7] — 2026-06-30

### Added
- Public, bring-your-own-key mode for the web server (`launch.sh --public`), with per-browser private histories keyed by an opaque `caesar_id`.

### Fixed
- A password-mode auth-gate bypass, and the multi-instance `/api` proxy routing.

## [0.4.6] — 2026-06-23

### Changed
- Image generator overhaul driven by an N=11 A/B audit; new diagram mode for procedural and mathematical sections.
- Multi-instance web server support, with `launch.sh` cross-checking `SYSTEMD_UNIT_NAME` against `CAESAR_INSTANCE_ID` at boot.

### Fixed
- SQLite-pool deadlock resolved by a WAL-mode preset in `kb_server.py`; `chromadb` unpinned from 1.5.2.

## [0.4.5] — 2026-06-04

### Changed
- Per-image first-reference selection via `text-embedding-3-small`, routed through `llm_handler`.
- Output format configurable (`output_format`, default `webp`); AVIF support dropped.

### Fixed
- `FatalLLMError` is now `BaseException`-derived, so quota and auth errors surface as run failures instead of an empty synthesis.

## [0.4.4] — 2026-06-02

### Changed
- `n=1` image generation defaults to 3 references (was 5); `n=1` and `n>1` share one `refs_per_image` budget.
- `-r` / `--references` now means references per image, not pool size.

## [0.4.1] — 2026-05-27

### Changed

- `ArtifactSynthesizer` clarify-pass prompt hardened for markdown +
  content preservation: numbers must appear verbatim (no rounding or
  paraphrasing), citation markers cannot be dropped or invented,
  bullets pinned to `-`, underscore italics banned, heading levels
  preserved exactly, fenced code blocks required for multi-line code,
  decorative emojis banned. Dropped the contradictory "SAME LENGTH OR
  LONGER" rule in favor of "remove only filler, never drop content".
- Synthesis prompt now branches on `is_external_ref`: cross-run
  reference seeds (`synthesis_reference_draft`) get "reuse vocabulary
  for continuity, do NOT repeat, paraphrase, or extend" framing, so
  follow-up runs stop drifting into paraphrases of the parent draft.

### Fixed

- `_normalize_formatting` now runs unconditionally at the end of
  `_merge_artifacts`. Previously only the clarified branch was
  normalized; the gated-off and clarify-failed branches shipped raw
  merged output.

## [0.4.0] — 2026-05-25

### Added
- Caesar Web Server: FastAPI + Next.js GUI for submitting runs, streaming progress and rendering the live knowledge graph.
- Follow-up exploration mode — chain queries onto previous runs without re-exploring.

### Changed
- README aligned to paper v3 numbers (Caesar 26.96 vs Gemini 3 Deep Research 23.78).

## [0.3.15] — 2026-05-13

### Added

- `caesar/image_generator.py`: post-processor that turns a Caesar run's
  artifact + cited URLs into a generated image. Scrapes candidate images
  from cited pages, scores them via VLM, captions the top-K, synthesizes
  a creative image-gen prompt, and renders via OpenAI's images API.
  CLI entry: `python -m caesar.image_generator <run_dir>`. `run_agent.py`
  exposes the pipeline via `--generate-image`.

### Changed

- `PROMPT_SYNTH_TEMPLATE` reworked into a forced 3-step chain
  (INSIGHT → METAPHOR → PROMPT) so the creative concept is grounded in a
  specific claim from the artifact rather than the broad topic. Added a
  cliché ban list and a one-bold-choice requirement; shrank the artifact
  excerpt window from 100k → 12k chars and bumped synth temperature
  0.7 → 0.9 to favour invention over summarization.
- README Quickstart's from-source clone URL and the `cd` on the next
  line now name the same directory, so the snippet works when pasted.

## [0.3.5] — 2026-05-08

### Added

- New `web_server/` directory: a single-shareable-URL FastAPI + Next.js
  demo server that lets a visitor type a research question, watch the
  knowledge graph grow live (SSE), and read the rendered final answer
  with citations. Run with `./web_server/launch.sh`.
- Progressive draft display on the run page: the synthesis section
  refetches on every `draft_complete` event, rolling through
  "Draft 1 answer" → ... → "Final Answer" as Caesar refines.
- `quick_explore` now writes graph snapshots and adds nodes/edges
  incrementally as each future completes, so live viewers can render
  the graph during phase 1 instead of waiting for the bulk write.
- `checkpoint.save()` accepts an optional `save_graph_interval` override
  so callers can force a snapshot regardless of the modulo gate.
- DuckDuckGo (`ddgs`) on by default in nano/mini presets.

### Changed

- `quick_explore` cleaned up: dropped a discarded `self.think()` call on
  the search-results page and an unused `text` field carrying hundreds of
  KB through the worker result dict; `current_iteration` is now
  monotonic; snapshot frequency capped at one per ~5–20 completions
  independent of the config default.
- `executor.shutdown(cancel_futures=True)` on shutdown so in-flight
  workers don't keep computing results that get discarded.

### Fixed

- `ArtifactSynthesizer` now adds `total_cost_usd` to artifact metadata
  (was always 0/null).
- `failed_urls.add()` on the empty-text return path in `quick_explore`.

## [0.3.0] — 2026-04-19

Major release coinciding with the Caesar paper publication ([arXiv: 2604.20855](https://arxiv.org/abs/2604.20855)).

### Added

- Caesar paper published on ResearchGate with DOI.
- Multi-provider LLM support via litellm: OpenAI, Anthropic, Gemini, and any OpenAI-compatible endpoint.
- `experiment_summary.json` emitted per run: wall-time, tokens, cost, iterations, pages visited, artifact paths, config snapshot.
- `multi_query` method in `kb_client` for parallel RAG-Fusion-style query rewriting with answer fusion (diverse retrieval on reasoning-model backends).
- Dynamic `Referer` and `Sec-Fetch-Site` headers per navigation (match real browser behavior, reduce bot detection).
- `iterations_elapsed` field in experiment summary for distinguishing full runs from early exits.
- Parallel `quick_explore` workers with thread-safe referer handling.
- Knowledge graph checkpoints saved as compressed `.json.gz`.

### Changed

- Repo renamed `rome` → `caesar-agent` (old URLs redirect).
- README rewritten around Caesar with concrete benchmark numbers (25.29 vs 22.27 runner-up), comparison table, use cases, and example outputs.
- `REQUESTS_TIMEOUT` raised from 10s to 25s for slow academic sites.
- `ArtifactSynthesizer.synthesize_artifact()` now returns enriched dict with `artifact_dir`, `artifact_files`, `num_drafts`.
- `pyproject.toml` overhauled: package name, authors, URLs, correct script entry (`caesar = "caesar.run_agent:main"`), and `caesar` added to installable packages.

### Fixed

- Agent repository clobbering between concurrent experiments.
- `override_config` duplicate `timeout` kwarg crash.
- `max_completion_tokens` calculation for thinking-enabled providers (Claude, o-series).
- JSON response format requirement placement.
- Broken script entries in `pyproject.toml` (`rome.cli:main` did not exist).
- All 159/159 tests passing.

### Removed

- `OpenAIHandler` (replaced by `LLMHandler`).
- Gemini 2.5 family (use Gemini 3 Pro family instead).

## [0.2.0] — 2026-02-22

Initial Caesar release.

[0.4.12]: https://github.com/jasonzliang/caesar-agent/releases/tag/0.4.12
[0.4.11]: https://github.com/jasonzliang/caesar-agent/releases/tag/0.4.11
[0.4.10]: https://github.com/jasonzliang/caesar-agent/releases/tag/0.4.10
[0.4.9]: https://github.com/jasonzliang/caesar-agent/releases/tag/0.4.9
[0.4.8]: https://github.com/jasonzliang/caesar-agent/releases/tag/0.4.8
[0.4.7]: https://github.com/jasonzliang/caesar-agent/releases/tag/0.4.7
[0.4.6]: https://github.com/jasonzliang/caesar-agent/releases/tag/0.4.6
[0.4.5]: https://github.com/jasonzliang/caesar-agent/releases/tag/0.4.5
[0.4.4]: https://github.com/jasonzliang/caesar-agent/releases/tag/0.4.4
[0.4.0]: https://github.com/jasonzliang/caesar-agent/releases/tag/0.4.0
[0.3.0]: https://github.com/jasonzliang/caesar-agent/releases/tag/0.3.0
[0.2.0]: https://github.com/jasonzliang/caesar-agent/releases/tag/0.2.0
