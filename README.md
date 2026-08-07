<p align="center">
  <img src="https://jasonzliang.github.io/caesar-agent/caesar.webp" alt="Caesar autonomous research agent architecture: Perceive-Think-Act exploration loop and adversarial artifact synthesis" width="720"/>
</p>

<h1 align="center">Caesar: Autonomous AI Research Agent</h1>

<p align="center">
  <strong>Deep web exploration and creative answer synthesis. The open-source alternative to ChatGPT Deep Research and Perplexity.</strong>
</p>

<p align="center">
  <a href="https://caesar.evolution.ml"><img alt="Live demo" src="https://img.shields.io/badge/Live%20Demo-Try%20Caesar-2ea44f?logo=googlechrome&logoColor=white"></a>
  <a href="https://arxiv.org/abs/2604.20855"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2604.20855-b31b1b?logo=arxiv&logoColor=white"></a>
  <a href="https://www.researchgate.net/publication/402554537_Caesar_Deep_Agentic_Web_Exploration_for_Creative_Answer_Synthesis"><img alt="ResearchGate" src="https://img.shields.io/badge/ResearchGate-Caesar-00ccbb?logo=researchgate&logoColor=white"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue?logo=apache&logoColor=white"></a>
  <a href="https://github.com/jasonzliang/caesar-agent/commits/main"><img alt="Last commit" src="https://img.shields.io/github/last-commit/jasonzliang/caesar-agent?logo=github&logoColor=white&label=Last%20Commit&color=181717"></a>
</p>

**Caesar** is an autonomous AI research agent that navigates the web, reasons over a dynamic knowledge graph, and synthesizes novel, grounded answers. In blinded LLM-as-a-Judge creativity evaluations, Caesar scored **26.96 / 30** on the headline full-answers configuration, beating the runner-up (Gemini 3 Deep Research, 23.78) by **3.18 points** and outscoring GPT-5.2 Deep Research (15.74) by over **11 points**. Cliff's δ effect sizes are uniformly large (**≥ 0.76**, well above the 0.47 large-effect threshold), and the result is corroborated by an independent 23-rater human study.

If you're looking for an **agentic RAG system that goes beyond retrieval** (graph-based exploration, adversarial verification, and multi-draft synthesis), this is it.

> 📄 **Read the paper:** [*Caesar: Deep Agentic Web Exploration for Creative Answer Synthesis*](https://arxiv.org/abs/2604.20855) (Liang, Meyerson, Miikkulainen, 2026 — **v3**, 8 May 2026) · [DOI: 10.48550/arXiv.2604.20855](https://doi.org/10.48550/arXiv.2604.20855) · [PDF](https://arxiv.org/pdf/2604.20855v3)

## Quickstart

**From PyPI:**

```bash
pip install caesar-agent
export OPENAI_API_KEY=your_key
caesar regular -q "your research question"
```

**From source:**

```bash
git clone https://github.com/jasonzliang/caesar-agent.git
cd caesar-agent && pip install -e .
export OPENAI_API_KEY=your_key
caesar regular -q "your research question"
```

**In a browser:**

```bash
cd web_server && ./launch.sh
# then open http://localhost:3000
```

The **[Caesar Web Server](web_server/README.md)** is a FastAPI + Next.js GUI that submits runs, streams progress, and renders the live knowledge graph and final artifact.

## Setup Notes

Both install paths give you the same `caesar` console script, so every command in
this README behaves identically whether you came from PyPI or from a checkout.
From a checkout you can also skip the script entirely and run
`python caesar/run_agent.py regular -q "..."` — that still needs the dependencies
(`pip install -r requirements.txt`), just not the package itself.

Everything else worth knowing before your first run:

- **Python** — 3.10 through 3.13
- **Presets** — `nano` (fast, ~$0.80), `mini` (balanced, ~$2), `regular` (deep, ~$5–$10). Cost scales with synthesis output tokens, so treat these as order-of-magnitude rather than quotes.
- **API keys** — `OPENAI_API_KEY` alone is enough to run. Add `ANTHROPIC_API_KEY` or `GOOGLE_API_KEY` to reach Claude and Gemini models, and `BRAVE_API_KEY` for better web search — without it, search falls back to DuckDuckGo.
- **Results** — `caesar/result/` from a source checkout, `~/.caesar/result/` from a PyPI install. `CAESAR_RESULT_DIR` overrides both.
- **Custom configs** — drop a YAML in `~/.caesar/configs/` and call it by name (`caesar my_preset -q "..."`). Bundled preset names take precedence, so avoid naming yours `nano`, `mini`, or `regular`.
- **Going deeper** — the full env-var list, exploration modes, and synthesis options are in the **[Caesar module docs](caesar/README.md)**.

## What It's Good For

Caesar shines on **open-ended, creative, cross-disciplinary** research, where retrieval alone won't work:

- **Hypothesis generation**: novel cross-domain connections (e.g., bridging materials science and biology)
- **Literature synthesis**: graph-grounded review that spots tensions and gaps between papers
- **Competitive intelligence**: deep mapping of a technical or market landscape
- **Counterfactual & meta-creative reasoning**: "what if X was different?" style inquiry
- **Novel solution ideation**: e.g., ARC-AGI–style problem exploration

It's **not** the right tool for quick factual lookups or latency-sensitive apps. Caesar is designed for depth, not speed.

## Why Caesar?

Current deep-research agents (ChatGPT Deep Research, Perplexity, GPT Researcher, Gemini Deep Research) optimize for **retrieval precision over a flat sequence of documents**. They produce competent summaries but suffer from **navigational amnesia**, fall into local minima, and generate derivative, consensus-driven outputs.

Caesar is different:

| Design choice | Caesar | ChatGPT / Gemini Deep Research | Perplexity Research | GPT Researcher |
|---|:-:|:-:|:-:|:-:|
| Persistent knowledge graph built during exploration | ✅ | not published | not published | ❌ |
| Adversarial refinement over its own drafts | ✅ | not published | 🟡 | 🟡 |
| Serialized graph and JSON run log you keep | ✅ | 🟡 | 🟡 | 🟡 |
| Runs on your own keys and hardware | ✅ | ❌ | ❌ | ✅ |
| Works with no setup and no API keys | ❌ | ✅ | ✅ | ❌ |
| Typical time to an answer | 10 min – 1.5 hrs | 5–30 min | under 3 min | 2–5 min |
| Browser and mobile access | ❌ | ✅ | ✅ | 🟡 |
| Maturity | research prototype | GA product | GA product | established open source |

<sub>Compared as of July 2026 against each product's then-current tier. **"Not published"** means the vendor has not documented the mechanism, which is not the same as its absence: these are closed systems and we can report only what they disclose. ChatGPT and Deep Research are trademarks of OpenAI; Perplexity of Perplexity AI; Gemini of Google.</sub>

## How It Works

Caesar operates in two cognitive phases:

### 1. Deep Web Exploration: stateful graph traversal

A recursive **Perceive–Think–Act** loop performs topological traversal of information spaces. Rather than isolating summaries, Caesar generates context-aware insights conditioned on the **local structure of the exploration graph**, analyzing how new content builds upon or contradicts neighboring nodes. A dynamic policy, informed by a vector knowledge base and episodic memory, autonomously switches between depth-first expansion, strategic backtracking, and targeted web search.

### 2. Adversarial Artifact Synthesis

Rather than a single-pass summary, Caesar runs as a recursive self-correction environment. Between drafts the agent re-reads its own artifact and formulates a refined, **orthogonal challenge** targeting narrative weaknesses, gaps and contradictions in the current belief state. Drafts are produced recurrently, each conditioned on the last, then merged — forcing the agent out of the consensus basin that traps single-pass LLMs.

## Architectural Innovations

- **Domain-Specific Role Adaptation**: the agent rewrites its own system prompt per task, overriding the safety-biased generic responses typical of RLHF models.
- **Graph-Augmented Insight Generation**: insights are conditioned on the exploration graph neighborhood, enabling online associative reasoning.
- **Knowledge-Guided Exploration Policy**: detects navigational stagnation via episodic memory and forces backtracking.
- **Adversarial Query Refinement**: orthogonal queries push the agent out of generic LLM consensus toward novel, grounded facts.

## Benchmark Results

Evaluated with a blinded **3-model LLM-as-a-Judge panel** (Claude Sonnet 4.5, GPT-5.2, Gemini 3 Pro) across three creativity dimensions (**New**, **Useful**, **Surprising**), scored 1–10 each. Headline configuration: full answers, unconstrained length.

| Agent | New | Useful | Surprising | **Total** | Cliff's δ |
|---|:-:|:-:|:-:|:-:|:-:|
| **Caesar** | **9.11** | **8.87** | **8.98** | **26.96** | — |
| Gemini 3 Deep Research | 8.09 | 7.60 | 8.09 | 23.78 | 0.84 |
| Sonnet 4.5 Deep Research | 6.73 | 7.49 | 6.42 | 20.64 | 1.00 |
| GPT-5.2 Deep Research | 5.07 | 6.31 | 4.36 | 15.74 | 1.00 |

A **13–23% improvement over state-of-the-art deep research agents**, with Caesar leading across all three output formats (Full, ELI5, ELI5-450W). Cliff's δ ≥ 0.76 in every comparison (well above the 0.47 large-effect threshold); δ = 1.00 vs. GPT-5.2 Deep Research denotes strict dominance in every output format; vs. Sonnet 4.5 Deep Research it is 1.00 on full answers and 0.76–0.96 on the ELI5 formats. The paper frames these results by magnitude of difference rather than null-hypothesis testing: with n = 5 challenges per group, δ estimates stochastic dominance and no p-values are reported.

**Compute-controlled comparison.** At a matched ~$5/challenge budget (T=250 with GPT-5-mini), Caesar still wins: 26.16 vs. Gemini 3 Deep 24.37, Sonnet 4.5 Deep 21.00, GPT-5.2 Deep 16.16. The lead is not an artifact of larger compute.

**Human evaluation.** 23 raters preferred Caesar in 63 of 112 pairwise A/B matchups vs. Gemini 3 Deep Research (56.25%, odds ratio 1.29), independently corroborating the LLM-judge findings.

Ablations isolate what each component contributes. Removing the knowledge graph costs δ = 0.52 (large), and restricting traversal to a single hop costs the same; shallow exploration costs δ = 0.92 on total score. Adversarial refinement is a trade rather than a uniform gain — it lifts Surprising (+1.76) and New at the expense of Useful, and the generative merge recovers utility (7.04 → 8.28) while keeping the discovered insights. See the [paper](https://arxiv.org/abs/2604.20855) for full methodology, per-output-format tables, exploration-budget ablation, and judge bias analysis.

## Example Output

After a run, Caesar writes a full artifact (abstract + body with citations) plus a structured summary:

```json
{
  "wall_time": 591.31,
  "tokens_used": 109873,
  "token_cost": 0.29,
  "api_calls": 20,
  "webpages_visited": 4,
  "iterations_elapsed": 5,
  "artifact_dir": "result/.../agent_CaesarExplorer.synthesis.04161850",
  "num_drafts": 2,
  "config_summary": { "..." : "..." }
}
```

The `artifact_dir` contains one `.txt` per synthesis draft, a final merged artifact, and a metadata file tracking sources cited in each draft. Knowledge graphs are saved as compressed JSON checkpoints for reproducibility or post-hoc analysis.

## Built on Rome

Caesar is built on **Rome**, a Finite State Machine framework for stateful AI agents that provides the agent runtime (FSM lifecycles, action selection, memory, LLM handling) that Caesar builds on, episodic memory, dynamic policy routing, and verifiable code execution. See the [Rome framework docs](rome/README.md) if you want to build your own agent on top.

## Project Layout

```
caesar-agent/
├── caesar/          # Caesar agent (see caesar/README.md for full usage)
│   ├── caesar_agent.py
│   ├── artifact_synthesis.py
│   ├── run_agent.py
│   ├── config/      # YAML configs and creativity benchmarks
│   └── paper/       # Caesar paper (PDF)
├── rome/            # Rome framework (see rome/README.md): FSM, memory, LLM handlers, KB client
├── web_server/      # FastAPI + Next.js web GUI (see web_server/README.md)
└── web_app/         # Streamlit operator tools: human eval + graph explorer (see web_app/README.md)
```

## FAQ

**How is this different from LangGraph / CrewAI / AutoGen?**
Those are orchestration frameworks: they help you wire up agents. Rome is an opinionated runtime for *how* agents should reason (graph-structured exploration, adversarial verification, episodic memory). Caesar is a concrete research agent built on it.

**Do I need GPUs?**
No. Caesar uses hosted LLM APIs (OpenAI, Anthropic). A local ChromaDB instance handles the vector store. Runs on a laptop.

**Which models are supported?**
OpenAI (GPT-5 family, o-series reasoning models), Anthropic (Claude 4.5 / 4.6), Google (Gemini 3.x), and any OpenAI-compatible endpoint. Model selection is per-subsystem (exploration, synthesis, judging) via YAML config.

**How much does a typical run cost?**
A 5-iteration exploration with GPT-5.4-mini (`caesar/config/config_test/single_agent_test.yaml`) runs at roughly $0.30 and 10 minutes. A 250-iteration deep run with the `regular` preset (`gpt-5.6-terra` exploration, `gpt-5.6-sol` synthesis) is typically $5–$10.

**Can I reproduce the benchmarks?**
Yes. Configs, judge rubrics, and evaluation scripts are in `caesar/config/` and `caesar/analysis/`.

## Contributing & Community

- ⭐ **Star the repo** if Caesar is useful for your research
- 💬 **[Open a Discussion](https://github.com/jasonzliang/caesar-agent/discussions)** for ideas, questions, or use cases
- 🐛 **[File an Issue](https://github.com/jasonzliang/caesar-agent/issues)** for bugs or feature requests
- 🔧 **PRs welcome**, especially new exploration policies, synthesis strategies, and benchmark domains

### Good places to start a fork

| Goal | Effort |
|---|:---:|
| Adapt Caesar for your research domain (legal, biology, finance, …) | 1–2 hrs |
| Add a new web-search backend (Tavily, Exa, Serper, …) | 2–3 hrs |
| Experiment with multi-agent synthesis (ring or debate merge) | 4–6 hrs |

If you fork Caesar for your own work, [open a Discussion](https://github.com/jasonzliang/caesar-agent/discussions) — we'd love to see what you build.

## Citation

If you use Caesar in your research, please cite:

```bibtex
@misc{liang26caesar,
  title={Caesar: Deep Agentic Web Exploration for Creative Answer Synthesis}, 
  author={Jason Liang and Elliot Meyerson and Risto Miikkulainen},
  year={2026},
  eprint={2604.20855},
  archivePrefix={arXiv},
  primaryClass={cs.IR},
  url={https://arxiv.org/abs/2604.20855}, 
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
