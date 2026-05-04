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

Blinded 3-model LLM-as-a-Judge panel (Claude Sonnet 4.5, GPT-5.2, Gemini 3 Pro) scoring across creativity dimensions (New, Useful, Surprising):

| Agent | Total |
| --- | --- |
| **Caesar** | **25.29 / 30** |
| Gemini 3 Deep Research | 22.27 |
| Sonnet 4.5 Deep Research | 20.89 |
| GPT-5.2 Deep Research | 15.40 |

Mann–Whitney U across all settings: **p < 0.001**. See the [paper](https://arxiv.org/abs/2604.20855) for full methodology, ablations, and judge bias analysis.

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
