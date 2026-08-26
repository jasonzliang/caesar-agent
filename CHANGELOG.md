# Changelog

All notable changes to Caesar are documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.25] - 2026-08-26

### Changed

- **Web GUI shows the friendly preset name instead of the raw preset id.** Runs
  on the `deeper_arxiv` / `deepest_arxiv` presets rendered the underscored id in
  the run header and run lists; they now show "ArXiv" / "ArXiv Deep". The label
  is resolved once from the preset registry in `_summary_from` and exposed as
  `RunSummary.preset_label`, with the UI falling back to the raw id if a stored
  id has no current entry.

## [0.4.24] - 2026-08-26

### Changed

- **Migrate existing web runs off the retired `arxiv` preset id.** The 0.4.23
  split left runs submitted under the old single `arxiv` preset pointing at a
  preset id that no longer exists (breaking their badge, restart, and
  resume-time config load). A startup DB value migration now rewrites
  `runs.preset` `arxiv` -> `deeper_arxiv`, reusing the same idempotent mechanism
  as the earlier `deep` -> `deeper` rename. Also refreshes the `/presets` smoke
  test to the current six-preset set.

## [0.4.23] - 2026-08-26

### Changed

- **Split the web GUI arXiv preset into two tiers**, "ArXiv" and "ArXiv Deep",
  replacing the single "arXiv" preset. "ArXiv" (`deeper_arxiv`) runs 250
  exploration iterations and synthesizes with GPT-5.6 Terra; "ArXiv Deep"
  (`deepest_arxiv`) runs 1000 iterations and synthesizes with GPT-5.6 Sol,
  matching the "Deepest" web preset. Both explore the Semantic Scholar citation
  graph identically (mode `arxiv`, GPT-5.6 Luna); only depth and synthesis model
  differ. The two pills share one divider in the preset selector so they read as
  a single arXiv group.

## [0.4.22] - 2026-08-26

### Added

- **arXiv citation-graph exploration mode** (`mode: arxiv`; bundled `arxiv`
  preset, plus an `arXiv` preset in the web GUI). Instead of Brave/DDGS web
  search and open-web crawling, Caesar searches Semantic Scholar and traverses
  the arXiv citation graph: papers are graph nodes (keyed by their
  `arxiv.org/abs/` URL), reference and citation links are edges, and the
  synthesized artifact's citations become real arXiv links. `ArxivExplorer`
  subclasses `WebExplorer` and overrides only node fetch/extract, so the
  knowledge graph, vector KB, adversarial synthesis, and checkpointing are
  unchanged. A new `SemanticScholarClient` (stdlib `requests`) adds a shared
  process-wide rate throttle with 429/5xx backoff, and reads an optional
  `SEMANTIC_SCHOLAR_API_KEY` (works anonymously otherwise). Each visited paper's
  PDF is fetched and parsed for full-text content, falling back to the
  Semantic Scholar abstract.

## [0.4.21] — 2026-08-17

### Fixed

- False `Worker stalled` watchdog failures. A root-cause analysis traced
  `no activity for 1201s (threshold 1200.0s)` to several causes that all reduce
  to more than 1200s of no watchdog-observable progress, and fixes them at four
  points:
  - `WATCHDOG_STALL_S` goes from 1200 to 1800. It had been *equal* to the
    largest per-call `LLMHandler.timeout` (`deepest` and `regular` both set
    1200s), so a single legitimate full-length synthesis call went
    watchdog-invisible for ~1200s and tripped the stall a second later. 1800 is
    that 1200s ceiling plus 600s of margin.
  - mem0's Chroma `HttpClient` inherits chromadb's hardcoded
    `httpx.Timeout(timeout=None)`, an infinite read, so a wedged or half-open
    Chroma subprocess could hang `remember()` or `recall()` indefinitely — which
    surfaced under the web server as a silent stall. It now gets the finite
    budget `kb_client` already applies to its own client (connect 5s, read and
    write 300s, pool 10s). The override reaches through a private attribute
    path, so it is guarded and warns rather than failing if that path changes.
  - `num_retries=0` on the exploration, startup, image-generation and
    Brave-search query-shortening LLM calls, mirroring the synthesizer, so
    litellm cannot stack up to three times the request timeout on a hung call.
    All four paths already fail soft or self-correct.
  - The synthesizer now logs a per-attempt heartbeat, `[LABEL] attempt k/N
    (reasoning_effort=…)`, before each otherwise watchdog-invisible blocking
    call; a new `LLM_ATTEMPT_RE` in the job pool matches it and resets the stall
    clock between attempts, so a stacked retry ladder stays visible. Guarded by
    `test_watchdog_margin.py`, which asserts the threshold stays above the
    highest preset timeout with at least 300s of margin, and that the heartbeat
    matches the regex while an ordinary synthesis log line does not.

### Changed

- The SSE reconnect debounce widens from 10s to 30s. Tailscale Funnel's reverse
  proxy drops streaming connections every 10–40s regardless of byte flow; the
  browser reconnects in 3–8s but funnel-side handshakes can spike to ~10s, so a
  10s debounce sat exactly at that ceiling and normal reconnects still flashed
  the `disconnected` badge. A genuinely dead stream still surfaces the badge,
  after the longer window.
- The takeover notice is clearer about what it is waiting for: "Stopping the
  previous attempt. The restart begins once it has fully stopped, which can take
  up to a minute."
- README: a live-demo and project-home-page link row above the badges, and the
  tagline cut to one rendered line.

## [0.4.20] — 2026-08-13

### Changed

- The Luna revert in 0.4.19 is narrowed to the knowledge base. That release
  swapped every `gpt-5.6-luna` reference across the presets, which was wider
  than the evidence: only the KB swap was measured, at 32.1s against 18.9s per
  query in a controlled same-collection A/B, about 6.5 minutes per 30-query run,
  with retrieval quality unchanged — 240 blind-judged passages over 6 queries
  scored luna 1.85, mini 1.87 and no-rerank 1.90, all inside a ±0.3 noise floor.
  Synthesis and exploration go back to Luna. Synthesis writes the artifact, and
  swapping the model that produces the product on latency grounds with no
  measurement of artifact quality is not a defensible trade for the ~2 minutes
  it bought in a ~33-minute run; exploration is 1–2 minutes either way and
  equally unmeasured. The net effect is one line per preset. `deepest` keeps
  sol, `mini` and `regular` keep terra, and the UI descriptions name Luna again
  because synthesis is Luna again.
- For the record, the underlying cause is that `reasoning_effort` barely moves
  Luna: reasoning tokens per call measured 618 unset, 558 minimal, 318 low, 560
  medium, 632 high — a floor it will not come off, with no monotonic trend.

## [0.4.19] — 2026-08-12

### Changed

- Presets are back on `gpt-5.4-mini` for the KB, exploration and synthesis
  models, reverting the move to `gpt-5.6-luna`. Runs had gone from ~17 minutes
  to 30–56. The measured cause is reasoning tokens, not speed: on a capped
  100-word answer luna spent 948 output tokens (808 reasoning, 140 visible) in
  8.5s against mini's 332 (201 reasoning, 132 visible) in 2.8s. Throughput is
  effectively identical at 111 against 118 tok/s, so luna is not slower per
  token — it emits three to four times as many, nearly all invisible. That also
  reconciles two measurements that had looked contradictory: tokens/sec put luna
  at 0.93x, while wall-clock per call was 1.7–3x worse. Both were right. Preset
  descriptions and the cost and duration estimates in `web_server` follow.

## [0.4.18] — 2026-08-12

### Added

- Public mode can set a target length for the synthesized artifact. `/config`
  offers preset default (unconstrained), brief, standard and detailed; the
  choice rides a `caesar_output_length` cookie and lands on
  `ArtifactSynthesizer.synthesis_max_length`. The server bounds it at
  500..20000 rather than enumerating the four options, so the API has no choice
  list to drift out of sync with the dropdown, while the client accepts only a
  value it actually offers — a hand-edited or stale cookie falls back to the
  preset default instead of 422-ing every submit with a pydantic message the
  user cannot act on, from a cookie they cannot see. Ignored outside public
  mode. Covered by `test_output_length.py`.

### Fixed

- The web UI footer's commit link pointed at a single SHA, which 404s once that
  commit is no longer reachable in the published mirror; it now points at the
  history.

### Changed

- README: setup notes rewritten, badges matched to the landing page, the Python
  badge swapped for a live-demo badge, and the docs pointer folded into the
  bullets. The NOTICE link is dropped from the License lines in both READMEs.
- `web_server/ui/next-env.d.ts` is no longer tracked; Next.js generates it.

## [0.4.17] — 2026-08-07

### Fixed

- `reasoning_effort` is no longer silently dropped for KB models. `kb_client`
  carried its own hand-maintained copy of `REASONING_MODELS`, and it had
  drifted: the copy was missing the entire GPT-5.6 family, so the `mini`
  preset's KB model — `gpt-5.6-luna`, configured with `reasoning_effort: low` —
  failed the membership test and had its effort discarded when the LlamaIndex
  LLM was built. It then ran at the model's default effort against a 10k
  `max_tokens` cap that covers reasoning *and* output. Exhausting that budget on
  reasoning returns an empty completion — which is exactly the input that made
  `LLMRerank` parse nothing and return zero nodes in 0.4.16. That release fixed
  the symptom; this removes the cause.

  The duplicate list is deleted rather than refreshed. `LLMHandler` gains
  `is_reasoning_model()` as a classmethod so callers can ask without
  constructing a handler, and `_is_reasoning_model` delegates to it, so the two
  forms cannot disagree. The lists are merged, keeping `kb_client`'s extras —
  notably `o3-pro`, which the `gpt-5*` prefix rule does not cover.

## [0.4.16] — 2026-08-07

### Fixed

- An unparseable rerank no longer discards the entire retrieval. `LLMRerank`
  reads `Doc: N, Relevance: M` lines, so anything that makes a batch
  unparseable — an off-format reply, or empty content because a reasoning model
  spent its completion budget on reasoning tokens — yielded zero nodes. The
  synthesizer was then handed no context and answered `Empty Response` with no
  sources, which from the outside is indistinguishable from a genuinely empty
  knowledge base: a 232-document KB could answer an on-topic question with
  nothing. When the rerank returns empty but retrieval did not, the top `top_n`
  in embedding order are used instead and a warning is logged, so a broken
  reranker stays visible rather than being papered over. Note this is distinct
  from an unreachable Chroma, which `query()`'s broad exception handler still
  reports as an empty KB.

### Changed

- The bare `gpt-5.6` alias is removed from the model tables. It routed to
  `-sol`, so it listed one model twice under two names and hid which tier — and
  which price — was being selected. Safe to remove rather than merely unlist: no
  run in either database ever recorded it, pricing consults litellm's live map
  before this table, and reasoning-model detection resolves any `gpt-5*` by
  prefix. Name the tier explicitly instead.

## [0.4.15] — 2026-08-07

### Fixed

- Agent shutdown no longer kills unrelated subprocesses. `shutdown_processes`
  walked `psutil.Process(parent_pid).children(recursive=True)`, so every
  per-run `agent.shutdown()` swept the host's entire descendant tree and
  SIGTERMed sibling processes — including the shared Chroma, which
  cascade-failed every concurrent run's KB calls with `ConnectError`. Recorded
  in `chroma.log` on 2026-07-30 (×2), 08-01, 08-02, 08-06 and 08-07 (×2), each
  immediately after a run's shutdown line. `ProcessManager` now reaps only
  subprocesses an agent explicitly tracked; `BaseAgent` tracks none, so
  embedded agents shut down without touching anything they do not own.
  Process-lifetime infrastructure is still cleaned up at interpreter exit via
  the existing `atexit` hook, so nothing leaks.
- Chroma startup is serialized behind a lock. Two concurrent observers of
  `is_running=False` could both spawn a server and race for :8091, which is how
  two instances ended up in the tree and why the old reaper fired paired
  SIGTERMs 65–180ms apart.
- Follow-up runs accept any terminal parent, not only successful ones. Refine
  and explore need the parent's knowledge base and artifact directory, not a
  successful synthesis; requiring `completed` forced operators to edit run rows
  in SQL to recover work. Queued and running parents are still rejected, and
  `interrupted` joins them since the lifespan boot auto-resumes those and a
  follow-up would race the resume.
- Refine runs show the parent's exploration scope instead of `0`. They build no
  graph of their own, so `graph_node_count` is snapshotted from the parent at
  submission and preserved at completion, and the live-count overlay is skipped
  for refine mode rather than stomping the inherited value back to a
  placeholder.
- Grammar slip in an LLM-facing exploration-strategy prompt.

## [0.4.14] — 2026-08-07

### Fixed

- The comparison table asserted six absences in ChatGPT Deep Research and
  Perplexity. The paper declines to make those claims -- section 5.1 notes those
  systems' "internal behavior, token budgets, and retrieval strategies are not
  publicly documented" -- and one was wrong as a product claim, since Perplexity
  offers model selection. Replaced with the landing page's wording, which marks
  undocumented mechanisms as "not published", says why that differs from
  absence, and includes the rows where Caesar loses.
- Phase 2 is named Adversarial Artifact Synthesis and its loop adversarial
  refinement, matching the paper. "Generator-Verifier loop" appeared in neither
  the paper nor the source. The prose also described "an independent adversarial
  module"; Appendix K.2 and Figure 1 have the same model critiquing its own
  draft, with drafts recurrent rather than independent. The `generator-verifier`
  package keyword is renamed to `adversarial-refinement`.

Documentation only -- no code changes.

## [0.4.13] — 2026-08-07

### Changed

- Install footprint cut by roughly 730MB. `llmlingua` pulled torch,
  transformers, sympy, accelerate, tokenizers, safetensors and nltk for a
  feature reached through one lazy import that already fails gracefully without
  it; it is now the `compress` extra. `evalplus` moves to the `benchmark` extra
  -- only `benchmark/` imports it, and that directory is export-ignored, so
  nothing in the wheel ever needed it. `tabulate` and `colorama` are dropped
  outright: neither is imported anywhere in `rome/` or `caesar/`.
- Licence declared as a PEP 639 SPDX expression rather than the deprecated TOML
  table, with `license-files` stated explicitly instead of relying on
  setuptools' default glob. Metadata now carries `License-Expression`, and the
  build emits no deprecation warnings. Build floor raised to `setuptools>=77`.
- `regular` preset synthesises with `gpt-5.6-sol` at medium reasoning effort,
  with the timeout raised to 1200s to match `deepest`. Exploration stays on
  `gpt-5.6-terra` at low effort.

### Fixed

- Preset cost figures in 0.4.12's published description were wrong. `regular` is
  ~$5-$10 and `mini` ~$2, as before; 0.4.12 shipped inflated estimates that
  cannot be corrected in place, since a PyPI description is immutable.
- `release.sh` tags HEAD from the pyproject version instead of requiring a tag to
  exist. The old behaviour published whatever an existing tag pointed at, so
  commits made after tagging were dropped from the release without warning. It
  now refuses a dirty tree, refuses to move a tag whose version is already on
  PyPI, and fails rather than silently skipping when `--pypi` targets a
  published version.

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

[0.4.17]: https://github.com/jasonzliang/caesar-agent/releases/tag/0.4.17
[0.4.16]: https://github.com/jasonzliang/caesar-agent/releases/tag/0.4.16
[0.4.15]: https://github.com/jasonzliang/caesar-agent/releases/tag/0.4.15
[0.4.14]: https://github.com/jasonzliang/caesar-agent/releases/tag/0.4.14
[0.4.13]: https://github.com/jasonzliang/caesar-agent/releases/tag/0.4.13
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
