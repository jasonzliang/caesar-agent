# Caesar web apps (internal tooling)

Streamlit-based local utilities for power users and researchers.

## Apps

- **`caesar_human_eval_app.py`** — A/B human evaluator for comparing Caesar
  runs. Loads a directory of completed experiments and walks the evaluator
  through blinded pairwise comparisons.
- **`caesar_agent_web_app.py`** — Interactive knowledge-graph explorer with
  t-SNE / Word2Vec analysis over insight embeddings.
- **`agent_web_app.py`** — Generic agent run dashboard.
- **`streamlit_run.py`** — Entry-point wrapper.

## Run

Prerequisite: `pip install streamlit` (not included in the Caesar
`requirements.txt` / `pyproject.toml`).

```bash
streamlit run web_app/caesar_human_eval_app.py
```

## Security model

These apps are **local-only operator tools**. They run on the operator's own
machine, bind to `localhost`, and operate under the operator's filesystem
privileges. The operator types directory paths and file inputs directly into
Streamlit text fields by design — that is the intended UX.

There is **no remote attack surface, no privilege boundary, and no untrusted
input source** in this threat model. CodeQL `py/path-injection` findings on
`caesar_human_eval_app.py` (alerts #3–#10) are accepted as not-applicable for
this reason and have been dismissed with rationale.

If you ever expose any of these apps over a network (e.g. via
`--server.address 0.0.0.0` or a reverse proxy), this threat model no longer
holds and the path-input controls must be replaced with an explicit allowlist.
