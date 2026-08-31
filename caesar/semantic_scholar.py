"""Semantic Scholar Graph API client for arxiv citation-graph exploration.

A thin, dependency-light wrapper (stdlib `requests` only, already a Caesar
dependency) around the Semantic Scholar Academic Graph API. Purpose-built for a
citation-graph crawler:

  * a single process-wide inter-request delay (S2 grants ~1 rps to an individual
    API key, shared across ALL endpoints, so the throttle is class-level and
    survives quick_explore fanning this client across a thread pool),
  * retry on 429/5xx with capped exponential backoff + jitter (S2 does not send
    a reliable Retry-After header),
  * exactly the four calls the crawler needs: search, single-paper lookup,
    references (backward edges), citations (forward edges).

Unlike BraveSearch there is no HTML and no file on disk: ArxivExplorer consumes
the returned paper dicts directly. See caesar/arxiv_explorer.py.
"""
import os
import random
import threading
import time
from typing import Dict, List, Optional

import requests

from rome.config import set_attributes_from_config
from rome.logger import get_logger
from .caesar_config import CAESAR_CONFIG, REQUESTS_TIMEOUT, MAX_BACKOFF_DELAY


# graph/v1 is the live host; partner.semanticscholar.org was retired in 2024.
BASE_URL = "https://api.semanticscholar.org/graph/v1"

# Env var for an optional S2 API key (fixed convention, not a per-run knob).
# Anonymous access works but shares a heavily-throttled pool; a key is advised.
S2_API_KEY_ENV = "SEMANTIC_SCHOLAR_API_KEY"

# Fields for a fully-hydrated paper node -- exactly what ArxivExplorer reads:
# title/authors/venue/year/abstract/tldr for the LLM text, externalIds +
# citationCount for the node id and ranking, openAccessPdf for _pdf_url's
# non-arxiv fallback. Nothing extra, to keep responses under S2's 10 MB cap.
PAPER_FIELDS = ("paperId,externalIds,title,abstract,tldr,year,authors,venue,"
                "citationCount,openAccessPdf")
# Nested fields for a neighbour paper inside a citations/references edge.
# IMPORTANT: /citations and /references REJECT `tldr` here with a 400
# ("Unrecognized or unsupported fields: [tldr]") -- unlike /paper/{id} -- so
# requesting it makes every edge fetch fail and the graph stalls at depth 2
# (tldr is a FullPaper-only property; abstract/authors/venue/openAccessPdf are
# accepted, verified live). Must be set here, not just in PAPER_FIELDS: an
# edge-discovered node is served from ArxivExplorer._paper_cache and never hits
# get_paper(), so anything missing here is missing for good past the seed.
NEIGHBOR_FIELDS = ["paperId", "externalIds", "title", "year", "abstract",
                    "citationCount", "authors", "venue", "openAccessPdf"]
# Forward edges (papers citing this one): isInfluential is the per-edge flag
# ArxivExplorer ranks on; the nested citingPaper carries the neighbour's fields.
CITATIONS_FIELDS = "isInfluential," + ",".join(
    "citingPaper." + f for f in NEIGHBOR_FIELDS)
# Backward edges (papers this one cites): same shape via citedPaper.
REFERENCES_FIELDS = "isInfluential," + ",".join(
    "citedPaper." + f for f in NEIGHBOR_FIELDS)

# HTTP retry budget. Deliberately NOT a config knob: a bounded retry budget plus
# the shared capped backoff (caesar_config.MAX_BACKOFF_DELAY) is a robustness
# invariant, not a raisable stall risk. Timeout reuses REQUESTS_TIMEOUT; the
# per-user rate setting (API-key tier) is the min_request_interval knob.
MAX_RETRIES = 5
# Max results per S2 /search page (API ceiling).
MAX_SEARCH_LIMIT = 100
# Max neighbours per S2 /citations or /references page (API ceiling).
MAX_EDGE_LIMIT = 1000


class SemanticScholarClient:
    """Rate-limited Semantic Scholar Graph API client (stdlib requests)."""

    # One throttle shared across ALL instances/threads in a process: S2's rate
    # limit is per-key and global, not per-object, and quick_explore fans this
    # client out across a worker pool. A class-level lock + timestamp serialises
    # every outbound call regardless of how many explorer threads exist.
    _throttle_lock = threading.Lock()
    _last_request_ts = 0.0

    def __init__(self, agent, config: Dict = None):
        self.agent = agent
        self.logger = get_logger()
        self.config = config if config is not None else {}
        set_attributes_from_config(
            self, self.config, CAESAR_CONFIG['SemanticScholar'].keys())

        self.api_key = os.getenv(S2_API_KEY_ENV)
        if self.api_key:
            self.logger.info(
                f"Semantic Scholar: using API key from ${S2_API_KEY_ENV}")
        else:
            self.logger.info(
                f"Semantic Scholar: no API key set; using the shared anonymous "
                f"pool (heavily rate-limited). Set ${S2_API_KEY_ENV} for "
                f"reliable throughput.")
        self._session = requests.Session()

    # ── rate-limited transport ────────────────────────────────────────────
    def _throttle(self) -> None:
        """Block until min_request_interval has elapsed since the last call
        anywhere in the process. Cheap when calls are already spaced out."""
        with type(self)._throttle_lock:
            wait = self.min_request_interval - (
                time.monotonic() - type(self)._last_request_ts)
            if wait > 0:
                time.sleep(wait)
            type(self)._last_request_ts = time.monotonic()

    def _request(self, method: str, path: str, *, params=None) -> Optional[dict]:
        """Issue one API call with the shared throttle + backoff retry.

        Returns parsed JSON, or None on unrecoverable failure. Callers degrade
        gracefully -- a dead node simply yields no text / no neighbours rather
        than aborting the whole exploration.
        """
        url = f"{BASE_URL}{path}"
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        delay = 1.0
        for attempt in range(1, MAX_RETRIES + 1):
            self._throttle()
            try:
                resp = self._session.request(
                    method, url, params=params,
                    headers=headers, timeout=REQUESTS_TIMEOUT)
            except requests.RequestException as e:
                self.logger.error(
                    f"Semantic Scholar request error on {path}: {e}; "
                    f"retry {attempt}/{MAX_RETRIES} after ~{delay:.1f}s")
            else:
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError as e:  # includes JSONDecodeError
                        self.logger.error(
                            f"Semantic Scholar 200 with invalid JSON on {path}: "
                            f"{e}; retry {attempt}/{MAX_RETRIES}")
                elif resp.status_code == 429 or resp.status_code >= 500:
                    self.logger.error(
                        f"Semantic Scholar {resp.status_code} on {path}; "
                        f"retry {attempt}/{MAX_RETRIES} after ~{delay:.1f}s")
                elif resp.status_code == 404:
                    self.logger.error(f"Semantic Scholar 404 (not found): {path}")
                    return None
                else:
                    self.logger.error(
                        f"Semantic Scholar {resp.status_code} on {path}: "
                        f"{resp.text[:200]}")
                    return None
            # Reached on a request exception, a 429/5xx, or a 200 with bad JSON.
            # Jittered exponential backoff (plain runtime, so random is fine).
            # Guarded: a sleep is the gap BETWEEN attempts, so skip it after the
            # last one. Unguarded, the final (largest, ~18s) sleep fired and then
            # the loop immediately gave up -- 53% of the total backoff spent
            # waiting for a retry that never happened.
            if attempt < MAX_RETRIES:
                time.sleep(delay + random.uniform(0, delay * 0.25))
                delay = min(delay * 2, MAX_BACKOFF_DELAY)
        self.logger.error(
            f"Semantic Scholar giving up on {path} after {MAX_RETRIES} tries")
        return None

    # ── public API ────────────────────────────────────────────────────────
    def search(self, query: str, limit: int = None) -> List[dict]:
        """Relevance search. Returns a list of paper dicts (possibly empty)."""
        limit = min(limit or self.num_results, MAX_SEARCH_LIMIT)
        data = self._request("GET", "/paper/search", params={
            "query": query, "limit": limit, "fields": PAPER_FIELDS})
        return (data or {}).get("data") or []

    def get_paper(self, paper_id: str) -> Optional[dict]:
        """Fetch one paper by S2 paperId / ARXIV:<id> / DOI:<doi> / etc."""
        return self._request(
            "GET", f"/paper/{paper_id}", params={"fields": PAPER_FIELDS})

    def references(self, paper_id: str, limit: int = None) -> List[dict]:
        """Papers this one cites (backward edges), as citedPaper dicts tagged
        with _isInfluential / _edge."""
        limit = min(limit or self.refs_limit, MAX_EDGE_LIMIT)
        data = self._request("GET", f"/paper/{paper_id}/references", params={
            "limit": limit, "fields": REFERENCES_FIELDS})
        return self._unwrap_edges(data, "citedPaper", "reference")

    def citations(self, paper_id: str, limit: int = None) -> List[dict]:
        """Papers that cite this one (forward edges), as citingPaper dicts
        tagged with _isInfluential / _edge."""
        limit = min(limit or self.citations_limit, MAX_EDGE_LIMIT)
        data = self._request("GET", f"/paper/{paper_id}/citations", params={
            "limit": limit, "fields": CITATIONS_FIELDS})
        return self._unwrap_edges(data, "citingPaper", "citation")

    @staticmethod
    def _unwrap_edges(data, key: str, edge_kind: str) -> List[dict]:
        """Flatten an edge-list response into neighbour paper dicts, copying
        the per-edge isInfluential flag onto each paper as _isInfluential."""
        out = []
        for item in (data or {}).get("data") or []:
            paper = item.get(key)
            if not paper:
                continue
            paper = dict(paper)
            paper["_isInfluential"] = bool(item.get("isInfluential"))
            paper["_edge"] = edge_kind
            out.append(paper)
        return out
