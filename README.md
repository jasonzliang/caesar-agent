# Caesar — Autonomous AI Research Agent

Caesar is an autonomous AI research agent. Instead of summarizing a flat list of search results, it treats the web as a graph — building a dynamic knowledge graph as it explores, backtracking when it stagnates, and refining its answer through an adversarial Generator–Verifier loop. The result is deeper, more novel synthesis on the open-ended, cross-disciplinary questions retrieval alone cannot answer.

**Live site:** <https://jasonzliang.github.io/caesar-agent/>

This repository hosts the public landing page only. It is built and served via GitHub Pages.

## What Caesar does

Today's deep-research agents — ChatGPT Deep Research, Perplexity, Gemini Deep Research, GPT Researcher — optimize retrieval precision over a flat sequence of documents. They produce competent summaries but fall into local minima, suffer from *navigational amnesia*, and converge on derivative, consensus-driven outputs.

Caesar is built differently:

- **Builds a knowledge graph as it explores** — each new page is analyzed against insights already attached to predecessor and neighbor nodes.
- **Adversarial self-critique on its own draft** — an independent verifier formulates orthogonal queries that target weaknesses in the current draft, escaping the consensus basin that traps single-pass LLMs.
- **Multiple drafts, then merged into one** — each draft chains off the previous one until a final merge.
- **Backtracks when an exploration path stalls** — depth-first drill-down with a stack to pop back and explore orthogonal branches.
- **Multi-provider** — OpenAI, Anthropic, Google Gemini, or any OpenAI-compatible endpoint.
- **Reproducible run logs (JSON)** — tokens, cost, wall-time, pages visited, per-draft provenance.

## Benchmark results

Blinded 3-model LLM-as-a-Judge panel (Claude Sonnet 4.5, GPT-5.2, Gemini 3 Pro) scored 0–10 across three creativity dimensions: **New**, **Useful**, **Surprising**.

| Agent | New | Useful | Surprising | Total |
| --- | --- | --- | --- | --- |
| **Caesar** | **9.11** | **8.87** | **8.98** | **26.96** |
| Gemini 3 Deep Research | 8.09 | 7.60 | 8.09 | 23.78 |
| Sonnet 4.5 Deep Research | 6.73 | 7.49 | 6.42 | 20.64 |
| GPT-5.2 Deep Research | 5.07 | 6.31 | 4.36 | 15.74 |

Cliff's Delta effect sizes are uniformly large (**δ ≥ 0.76**, well above the 0.47 large-effect threshold) across all baselines and output formats; **δ = 1.00** against five of six baselines indicates strict dominance. The advantage holds in a compute-controlled run (Caesar at $5/challenge with GPT-5-mini still tops Gemini 3 Deep Research) and in a 21-rater blinded human A/B study, where raters preferred Caesar's Cross-Domain Synthesis answer 90.5% of the time (Holm-corrected p = 0.001). Ablations confirm both graph exploration and the adversarial verifier loop are independently necessary. See the [paper](https://arxiv.org/abs/2604.20855) for full methodology, exploration-budget ablation, and judge bias analysis.

## Read more

- **Landing page (full details, figures, FAQ):** <https://jasonzliang.github.io/caesar-agent/>
- **Paper (arXiv):** <https://arxiv.org/abs/2604.20855>
- **DOI:** <https://doi.org/10.48550/arXiv.2604.20855>
- **ResearchGate:** [Caesar publication](https://www.researchgate.net/publication/402554537_Caesar_Deep_Agentic_Web_Exploration_for_Creative_Answer_Synthesis)

## Citation

```bibtex
@misc{liang26caesar,
  title={Caesar: Deep Agentic Web Exploration for Creative Answer Synthesis},
  author={Jason Liang and Elliot Meyerson and Risto Miikkulainen},
  year={2026},
  eprint={2604.20855},
  archivePrefix={arXiv},
  primaryClass={cs.IR},
  url={https://arxiv.org/abs/2604.20855}
}
```

## Authors

By [Jason Liang](https://jasonzliang.github.io/), Elliot Meyerson, and Risto Miikkulainen — Cognizant AI Lab and The University of Texas at Austin.

Apache License 2.0.
