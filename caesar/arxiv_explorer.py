"""ArxivExplorer - Semantic Scholar citation-graph traversal for CaesarAgent.

A drop-in sibling of WebExplorer. It SUBCLASSES WebExplorer so it inherits the
LLM-driven navigation unchanged (select_next_link, format_link_options,
determine_exploration_strategy, recall_navigation_history). Only the three "how
do I read a node" operations are overridden, which is all it takes to make the
agent's Perceive-Think-Act loop walk an arxiv citation graph instead of the
open web:

    fetch_html(url)          -> a payload dict {kind, text, ...} (NOT HTML)
                                built from Semantic Scholar API calls
    extract_text_from_html   -> title / authors / abstract / tldr text for think
    extract_links            -> reference + citation neighbours as (node_id,label)

Plus get_web_search_links (the mid-run WEB_SEARCH strategy action) is redirected
to an S2 search.

Node identity is a stable, human-clickable URL string:
  * arxiv paper -> https://arxiv.org/abs/<arxiv_id>
  * other paper -> https://www.semanticscholar.org/paper/<paperId>
  * the seed    -> https://www.semanticscholar.org/search?q=<query>
so the whole downstream (NetworkX graph, KB `url` metadata, synthesis citations,
checkpoint (de)serialisation) keeps working with zero changes -- it only ever
sees opaque URL strings. node_id <-> S2-id conversion is stateless (parsed from
the URL), so it survives checkpoint resume without any in-memory map.
"""
import threading
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote, unquote

from rome.config import LONGER_SUMMARY_LEN

from .semantic_scholar import SemanticScholarClient
from .web_explorer import WebExplorer


# Node-id prefix for an arXiv paper
ARXIV_ABS_PREFIX = "https://arxiv.org/abs/"
# Node-id prefix for a non-arXiv S2 paper
S2_PAPER_PREFIX = "https://www.semanticscholar.org/paper/"
# Node-id prefix for the seed (S2 search)
S2_SEARCH_PREFIX = "https://www.semanticscholar.org/search?q="

# Max chars of paper text fed to the LLM. Bigger than the web MAX_TEXT_LENGTH:
# an arXiv PDF is a whole paper, and the web cap truncated the longest ~18% of
# papers (measured). 2x captures nearly all in full; only long papers pay it.
ARXIV_MAX_TEXT_LENGTH = 200000

# Max neighbour edges (references + citations) surfaced per paper after
# influential-first ranking -- the frontier/menu cap. Distinct from
# MAX_EDGE_LIMIT, the S2 per-page *fetch* cap.
ARXIV_MAX_NUM_LINKS = 1000


def paper_node_id(paper: dict) -> Optional[str]:
    """Canonical node URL for a paper dict, preferring the arxiv abs URL.
    Returns None when the paper has neither an arxiv id nor a paperId."""
    ext = paper.get("externalIds") or {}
    arxiv = ext.get("ArXiv")
    if arxiv:
        return f"{ARXIV_ABS_PREFIX}{arxiv}"
    pid = paper.get("paperId")
    return f"{S2_PAPER_PREFIX}{pid}" if pid else None


def node_id_to_s2_id(node_id: str) -> Optional[str]:
    """Reverse of paper_node_id: the id string to pass to the S2 API. Stateless
    (parsed straight from the URL), so it works after a checkpoint resume."""
    if node_id.startswith(ARXIV_ABS_PREFIX):
        return "ARXIV:" + node_id[len(ARXIV_ABS_PREFIX):]
    if node_id.startswith(S2_PAPER_PREFIX):
        return node_id[len(S2_PAPER_PREFIX):]
    return None


def make_seed_url(query: str) -> str:
    """Deterministic seed node URL for a query."""
    return f"{S2_SEARCH_PREFIX}{quote(query, safe='')}"


def seed_query(node_id: str) -> Optional[str]:
    """The query encoded in a seed URL, or None if node_id is not a seed."""
    if node_id.startswith(S2_SEARCH_PREFIX):
        return unquote(node_id[len(S2_SEARCH_PREFIX):])
    return None


class ArxivExplorer(WebExplorer):
    """Citation-graph explorer over Semantic Scholar. See module docstring."""

    def __init__(self, agent, config: Dict = None):
        super().__init__(agent, config)
        self.client = SemanticScholarClient(
            agent=agent, config=agent.config.get("SemanticScholar", {}))
        # node_id -> paper dict, populated by search + edge hydration so that
        # following an edge (or re-perceiving a node) needs no extra API call.
        self._paper_cache: Dict[str, dict] = {}
        self._cache_lock = threading.Lock()
        # The seed search is run once and memoised (perceive may revisit it).
        self._seed_payload: Optional[dict] = None

    # ── seed setup ─────────────────────────────────────────────────────────
    def setup_seed(self) -> str:
        """Return the deterministic seed node URL for the agent's query. The
        actual S2 search runs lazily on first perceive (and is cached)."""
        if not self.agent.starting_query:
            raise ValueError("arxiv mode requires a starting_query to search")
        return make_seed_url(self.agent.starting_query)

    def _cache_papers(self, papers: List[dict]) -> None:
        with self._cache_lock:
            for p in papers:
                nid = paper_node_id(p)
                # First write wins. Note which write is usually first: a node is
                # normally seen as an edge neighbour (extract_links caches the
                # whole edge page) before it is ever visited, so the edge dict is
                # the one that sticks and _fetch_paper's get_paper() never runs
                # for it. That is only safe because NEIGHBOR_FIELDS requests the
                # same fields as PAPER_FIELDS except tldr -- keep the two in step
                # or non-seed nodes silently lose whatever diverges.
                if nid and nid not in self._paper_cache:
                    self._paper_cache[nid] = p

    # ── node reading (WebExplorer overrides) ────────────────────────────────
    def fetch_html(self, url: str, referer_url: Optional[str] = None) -> Optional[dict]:
        """Return a payload dict for a node (NOT HTML). None marks a dead node.

        The two extract_* overrides below read this payload. referer_url is
        accepted for signature-compatibility with the parallel quick_explore
        caller and ignored (there is no referer for an API lookup)."""
        q = seed_query(url)
        if q is not None:
            return self._fetch_seed(q)
        return self._fetch_paper(url)

    def _fetch_seed(self, query: str) -> dict:
        if self._seed_payload is not None:
            return self._seed_payload
        # Honour additional_starting_queries: seed from the starting query plus
        # its LLM-generated variants (shared with the web path), one S2 search
        # each, merged + de-duped. Breadth then comes from following citations.
        queries = self.agent._build_starting_queries()
        papers = self._multi_search(queries)
        self._cache_papers(papers)
        payload = {
            "kind": "seed",
            "papers": papers,
            "text": self._format_seed_text(query, papers),
        }
        self._seed_payload = payload
        self.logger.info(
            f"[ARXIV] Seed search ({len(queries)} "
            f"{'query' if len(queries) == 1 else 'queries'}): {len(papers)} "
            f"papers ({len(self._papers_to_links(papers))} usable graph nodes)")
        return payload

    def _multi_search(self, queries: List[str]) -> List[dict]:
        """One S2 search per query, merged and de-duped by node id. First
        occurrence wins, so the primary query's ranking leads the frontier.

        Each search is logged: the seed phase can be several rate-limited calls
        (one per query) and would otherwise be silent while it churns, which is
        worst on the throttled anonymous S2 pool."""
        seen = set()
        merged = []
        for i, q in enumerate(queries, 1):
            hits = self.client.search(q)
            self.logger.info(
                f"[ARXIV] Seed search {i}/{len(queries)} '{q}': {len(hits)} hits")
            for p in hits:
                nid = paper_node_id(p)
                if nid and nid not in seen:
                    seen.add(nid)
                    merged.append(p)
        return merged

    def _fetch_paper(self, url: str) -> Optional[dict]:
        s2_id = node_id_to_s2_id(url)
        if not s2_id:
            self.logger.error(f"[ARXIV] Unrecognized node id: {url}")
            return None
        paper = self._paper_cache.get(url)
        if paper is None:
            paper = self.client.get_paper(s2_id)
            if not paper:
                return None
            self._cache_papers([paper])
        # Content = the paper's full PDF text when available (parsed via the web
        # fetcher's pypdf path), falling back to the S2 abstract when PDF fetch
        # is disabled or the download/parse yields nothing.
        text = self._format_paper_text(paper)
        if getattr(self.client, "arxiv_fetch_pdf", True):
            pdf = self._pdf_url(paper)
            if pdf:
                body = self._fetch_pdf_text(pdf, referer=url)
                if body:
                    text = f"{self._paper_header(paper)}\n\nFull text (from {pdf}):\n{body}"
                    self.logger.info(f"[ARXIV] Parsed PDF ({len(body)} chars) for {url}")
                else:
                    self.logger.info(f"[ARXIV] No PDF text for {url}; using abstract")
        return {
            "kind": "paper",
            "paper": paper,
            "s2_id": s2_id,
            "text": text,
        }

    @staticmethod
    def _pdf_url(p: dict) -> Optional[str]:
        """Best PDF URL: arxiv.org/pdf for arxiv papers (open + reliable), else
        S2's openAccessPdf link when present."""
        ax = (p.get("externalIds") or {}).get("ArXiv")
        if ax:
            return f"https://arxiv.org/pdf/{ax}"
        return ((p.get("openAccessPdf") or {}).get("url")) or None

    def _fetch_pdf_text(self, pdf_url: str, referer: str) -> str:
        """Fetch + parse a PDF's full text via WebExplorer's fetcher (curl_cffi +
        pypdf, including its SSRF guard). Returns '' on any failure. A non-None
        referer keeps this off the shared _last_fetched_url so it stays
        thread-safe under quick_explore."""
        try:
            html = super().fetch_html(pdf_url, referer_url=referer)
            return super().extract_text_from_html(html, max_length=ARXIV_MAX_TEXT_LENGTH) if html else ""
        except Exception as e:
            self.logger.error(f"[ARXIV] PDF fetch failed for {pdf_url}: {e}")
            return ""

    def extract_text_from_html(self, payload, max_length: int = ARXIV_MAX_TEXT_LENGTH) -> str:
        """Return the node's text (bounded by max_length). Named for
        WebExplorer signature-compatibility; `payload` is our fetch dict."""
        if not payload:
            return ""
        return (payload.get("text") or "")[:max_length]

    def extract_links(self, payload, base_url: str) -> List[Tuple[str, str]]:
        """Return the node's neighbours as (node_id, label) tuples.

        Seed neighbours are the search hits. Paper neighbours are its references
        + citations, fetched lazily here (so quick_explore, which never calls
        this, pays nothing for edges)."""
        if not payload:
            return []
        if payload.get("kind") == "seed":
            return self._papers_to_links(payload.get("papers") or [], exclude=base_url)
        # Paper node: neighbours = its references + citations, fetched lazily
        # here. perceive() calls extract_links once per node, and quick_explore
        # never calls it, so no edge fetches happen on the quick path.
        s2_id = payload.get("s2_id")
        if not s2_id:
            return []
        edges = self.client.references(s2_id) + self.client.citations(s2_id)
        self._cache_papers(edges)
        return self._papers_to_links(edges, exclude=base_url)

    def get_web_search_links(self, query: str) -> List[Tuple[str, str]]:
        """Mid-run WEB_SEARCH strategy action, redirected to an S2 search."""
        if self.agent.web_searches_used >= self.agent.max_web_searches:
            self.logger.error(
                f"[ARXIV] Search limit reached during exploration: "
                f"{self.agent.max_web_searches}")
            return []
        try:
            papers = self.client.search(query)
            self.agent.web_searches_used += 1
            self._cache_papers(papers)
            links = self._papers_to_links(papers, exclude=self.agent.current_url)
            self.logger.info(f"[ARXIV] Search '{query}' returned {len(links)} nodes")
            return links
        except Exception as e:
            self.logger.error(
                f"[ARXIV] Search during exploration failed for '{query}': {e}")
            return []

    # ── helpers ─────────────────────────────────────────────────────────────
    def _papers_to_links(self, papers: List[dict],
                         exclude: Optional[str] = None) -> List[Tuple[str, str]]:
        """Rank + filter neighbour papers into (node_id, label) tuples.

        Drops non-arxiv papers when arxiv_only, the excluded/base node, dupes,
        known-failed nodes, and over-revisited nodes (mirrors WebExplorer's
        extract_links gates). Ranks influential citations first, then by
        citation count, so the frontier surfaces load-bearing papers."""
        arxiv_only = getattr(self.client, "arxiv_only", True)
        max_revisits = getattr(self.agent, "max_allowed_revisits", 20)
        failed = getattr(self.agent, "failed_urls", set()) or set()
        visited = getattr(self.agent, "visited_urls", {}) or {}

        seen = set()
        kept = []
        for p in papers:
            ext = p.get("externalIds") or {}
            if arxiv_only and not ext.get("ArXiv"):
                continue
            nid = paper_node_id(p)
            if not nid or nid == exclude or nid in seen:
                continue
            if nid in failed or visited.get(nid, 0) > max_revisits:
                continue
            seen.add(nid)
            kept.append(p)

        # Influential citations first, then by citation count, so the frontier
        # surfaces load-bearing papers before the long tail -- but an influential
        # paper must have at least one citation of its own to earn that top
        # tier. S2 sets isInfluential on brand-new papers too: measured on
        # ARXIV:1706.03762, 40 of the 43 influential citers in a 1000-edge page
        # had ZERO citations, so without the floor they all outranked
        # 100k-citation references and the frontier led with noise. The floor is
        # `> 0`, not a tuned threshold: a paper nobody has cited yet carries no
        # corroborating signal, while a single citation still beats raw
        # popularity (see test_frontier_ranks_influential_first).
        kept.sort(key=lambda p: (
            not (p.get("_isInfluential", False) and (p.get("citationCount") or 0) > 0),
            -(p.get("citationCount") or 0)))

        return [(paper_node_id(p), self._label(p)) for p in kept[:ARXIV_MAX_NUM_LINKS]]

    @staticmethod
    def _label(p: dict) -> str:
        """Human-readable link label: 'Title (year, N cites, influential, citation)'."""
        title = (p.get("title") or "[untitled]").strip()
        bits = []
        if p.get("year"):
            bits.append(str(p["year"]))
        if p.get("citationCount") is not None:
            bits.append(f"{p['citationCount']} cites")
        if p.get("_isInfluential"):
            bits.append("influential")
        if p.get("_edge"):
            bits.append(p["_edge"])
        return f"{title} ({', '.join(bits)})" if bits else title

    @staticmethod
    def _tldr_text(p: dict) -> Optional[str]:
        tldr = p.get("tldr")
        return tldr.get("text") if isinstance(tldr, dict) else None

    @classmethod
    def _paper_header(cls, p: dict) -> str:
        """Metadata header: title / authors / year / venue / arxiv id /
        citation count + TL;DR. Prefixes both the abstract-only and the
        full-text node content."""
        lines = []
        if p.get("title"):
            lines.append(f"Title: {p['title']}")
        authors = [a.get("name") for a in (p.get("authors") or []) if a.get("name")]
        if authors:
            lines.append(f"Authors: {', '.join(authors[:12])}")
        if p.get("year"):
            lines.append(f"Year: {p['year']}")
        if p.get("venue"):
            lines.append(f"Venue: {p['venue']}")
        ext = p.get("externalIds") or {}
        if ext.get("ArXiv"):
            lines.append(f"arXiv: {ext['ArXiv']}")
        if p.get("citationCount") is not None:
            lines.append(f"Citations: {p['citationCount']}")
        tldr = cls._tldr_text(p)
        if tldr:
            lines.append(f"\nTL;DR: {tldr}")
        return "\n".join(lines).strip()

    @classmethod
    def _format_paper_text(cls, p: dict) -> str:
        """Abstract-only node content (header + abstract). Used when PDF fetch
        is off or the PDF can't be retrieved."""
        header = cls._paper_header(p)
        abstract = p.get("abstract")
        return f"{header}\n\nAbstract:\n{abstract}" if abstract else header

    @staticmethod
    def _format_seed_text(query: str, papers: List[dict]) -> str:
        """Compose the root node's overview text (a ranked list of the search
        hits with abstract snippets) so the loop has non-empty content to
        summarise and proceed to the first Act. On an empty result set it still
        returns a non-empty line so the Perceive-Think-Act loop reaches the
        clean 'exploration exhausted' exit instead of spinning on empty content
        at the root."""
        if not papers:
            return f"Semantic Scholar search for '{query}' returned no results."
        lines = [f"Semantic Scholar search results for: {query}", ""]
        for i, p in enumerate(papers, 1):
            title = (p.get("title") or "[untitled]").strip()
            year = p.get("year") or "n.d."
            head = f"{i}. {title} ({year}"
            if p.get("citationCount") is not None:
                head += f", {p['citationCount']} cites"
            head += ")"
            lines.append(head)
            ab = p.get("abstract")
            if ab:
                lines.append(f"   {ab[:LONGER_SUMMARY_LEN].strip()}...")
        return "\n".join(lines)
