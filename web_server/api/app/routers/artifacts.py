"""GET /runs/{id}/graph and /runs/{id}/synthesis — read run artifacts off disk."""
from __future__ import annotations

import gzip
import json
import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session
from ..deps import current_owner, get_owned_run, is_admin
from ..job_runner import GRAPH_ITER_RE, MERGED_RE, SYNTHESIS_RE
from ..schemas import (
    CitationOut,
    GraphEdge,
    GraphNode,
    GraphOut,
    SearchResultItem,
    SearchResultsOut,
    SynthesisOut,
)

logger = logging.getLogger("caesar.web.artifacts")
router = APIRouter(prefix="/runs", tags=["runs"])


def _read_json(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_synthesis_txt(text: str) -> dict:
    """Parse the plain-text synthesis format Caesar writes by default.

    Format:
        ABSTRACT:
        <text>

        ARTIFACT:
        <text with [N] citations>

        SOURCES:
        [1] https://...
        [2] https://...
    """
    sections: dict[str, str] = {"abstract": "", "artifact": "", "sources": ""}
    current: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper in ("ABSTRACT:", "ARTIFACT:", "SOURCES:"):
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = upper.rstrip(":").lower()
            buf = []
            continue
        if current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()

    sources: dict[str, int] = {}
    src_re = re.compile(r"^\s*\[(\d+)\]\s+(\S.*)$")
    for line in sections["sources"].splitlines():
        m = src_re.match(line)
        if m:
            sources[m.group(2).strip()] = int(m.group(1))

    return {
        "abstract": sections["abstract"],
        "artifact": sections["artifact"],
        "sources": sources,
        "metadata": {},
    }


def _list_graph_files(repo: Path) -> list[tuple[int, Path]]:
    """Return (iteration, path) tuples sorted by iteration then mtime.

    The mtime tiebreak resolves the case where two writers produce a file at
    the same iteration N (e.g. Caesar's `.json.gz` + the web-server's
    uncompressed `.json` final-graph save). Without it, `latest` selection
    becomes filesystem-iteration-order dependent and can return the older
    file.
    """
    out: list[tuple[int, float, Path]] = []
    for f in repo.iterdir():
        if not f.is_file():
            continue
        m = GRAPH_ITER_RE.search(f.name)
        if m:
            try:
                mtime = f.stat().st_mtime
            except OSError:
                mtime = 0.0
            out.append((int(m.group("n")), mtime, f))
    out.sort(key=lambda t: (t[0], t[1]))
    return [(n, p) for n, _m, p in out]


def _node_link_to_out(data: dict) -> GraphOut:
    """Convert a NetworkX node-link JSON blob to our typed GraphOut."""
    # caesar/checkpoint.py writes starting_url and iteration at the top
    # level of the saved JSON (not inside the node-link graph attrs
    # dict), so check there first and fall back to graph_meta for
    # forward compatibility if that ever moves.
    graph_meta = data.get("graph", {}) or {}
    starting_url = data.get("starting_url") or graph_meta.get("starting_url")
    iteration = data.get("iteration") or graph_meta.get("iteration", 0)

    nodes_in = data.get("nodes", []) or []
    nodes = [
        GraphNode(
            id=str(n.get("id")),
            depth=int(n.get("depth", 0) or 0),
            insights=n.get("insights"),
            iteration=n.get("iteration"),
            visit_count=n.get("visit_count"),
        )
        for n in nodes_in
        if n.get("id") is not None
    ]

    edges_in = data.get("links") or data.get("edges") or []
    edges = [
        GraphEdge(
            source=str(e.get("source")),
            target=str(e.get("target")),
            reason=e.get("reason"),
        )
        for e in edges_in
        if e.get("source") is not None and e.get("target") is not None
    ]

    return GraphOut(
        iteration=int(iteration or 0),
        starting_url=starting_url,
        nodes=nodes,
        edges=edges,
    )


@router.get("/{run_id}/graph", response_model=GraphOut)
async def get_graph(
    run_id: str,
    iter: str = Query("latest", description="'latest' or an integer iteration number"),
    session: AsyncSession = Depends(get_session),
    owner: str | None = Depends(current_owner),
    admin: bool = Depends(is_admin),
) -> GraphOut:
    # Owner check (404 on miss or cross-owner) BEFORE any filesystem access so
    # this route is not a cross-tenant existence oracle.
    run = await get_owned_run(run_id, owner, session, admin=admin)
    if not run.repository:
        raise HTTPException(status_code=404, detail="Run has no repository on disk.")

    repo = Path(run.repository)
    if not repo.exists():
        raise HTTPException(status_code=404, detail="Run repository directory missing.")

    files = _list_graph_files(repo)
    if not files:
        raise HTTPException(status_code=404, detail="No graph snapshots yet.")

    if iter.lower() == "latest":
        # Newest-first iteration so a partial write on the freshest snapshot
        # (EOFError / JSONDecodeError) falls back to the prior stable one.
        # Caesar's writer is atomic (tmp + os.replace), but we keep the
        # fallback as defense in depth in case of an unrelated I/O blip.
        candidates = [f for _, f in reversed(files)]
    else:
        try:
            wanted = int(iter)
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail="iter must be an integer or 'latest'."
            ) from e
        # _list_graph_files sorts (iter_n, mtime); among duplicates at the
        # same iter, the newer file is at the end of the matching run, so
        # take the LAST match rather than the first to honor the mtime
        # tiebreak documented on _list_graph_files.
        matches = [f for n, f in files if n == wanted]
        if not matches:
            raise HTTPException(status_code=404, detail=f"No snapshot for iteration {wanted}.")
        candidates = [matches[-1]]

    last_err: Exception | None = None
    for chosen in candidates:
        try:
            data = _read_json(chosen)
            return _node_link_to_out(data)
        except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError, gzip.BadGzipFile) as e:
            last_err = e
            logger.warning(
                "Partial/unreadable graph file %s (%s); trying older snapshot",
                chosen,
                e,
            )
            continue

    # last_err's str representation can include absolute filesystem paths
    # (FileNotFoundError, OSError); log the detail server-side and surface
    # only a generic message to the client.
    logger.error("All graph snapshots unreadable for run %s: %s", run_id, last_err)
    raise HTTPException(
        status_code=500, detail="Could not read any graph snapshot."
    )


@router.get("/{run_id}/synthesis", response_model=SynthesisOut)
async def get_synthesis(
    run_id: str,
    draft: str = Query("latest", description="'latest', 'merged', or a draft number"),
    session: AsyncSession = Depends(get_session),
    owner: str | None = Depends(current_owner),
    admin: bool = Depends(is_admin),
) -> SynthesisOut:
    # Owner check (404 on miss or cross-owner) BEFORE any filesystem access.
    run = await get_owned_run(run_id, owner, session, admin=admin)
    if not run.repository:
        raise HTTPException(status_code=404, detail="Run has no repository on disk.")

    repo = Path(run.repository)

    # Walk both the repo and any nested .synthesis.* sub-directory.
    # Caesar writes .txt by default (SYNTHESIS_SAVE_JSON=False); .json is opt-in.
    candidates: list[Path] = []
    for pattern in ("*.json", "*.txt"):
        for p in repo.rglob(pattern):
            if SYNTHESIS_RE.search(p.name) or MERGED_RE.search(p.name):
                candidates.append(p)
    if not candidates:
        raise HTTPException(status_code=404, detail="No synthesis files yet.")

    candidates.sort(key=lambda p: p.stat().st_mtime)

    chosen: Path
    chosen_label: str
    if draft == "latest":
        chosen = candidates[-1]
        m = MERGED_RE.search(chosen.name) or SYNTHESIS_RE.search(chosen.name)
        chosen_label = (m.groupdict().get("draft") if m else None) or "merged"
    elif draft == "merged":
        merged = [p for p in candidates if MERGED_RE.search(p.name)]
        if not merged:
            raise HTTPException(status_code=404, detail="No merged synthesis available.")
        chosen = merged[-1]
        chosen_label = "merged"
    else:
        try:
            wanted = int(draft)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail="draft must be 'latest', 'merged', or a draft number.",
            ) from e
        match: Path | None = None
        for p in candidates:
            m = SYNTHESIS_RE.search(p.name)
            if m and int(m.group("draft")) == wanted:
                match = p
                break
        if match is None:
            raise HTTPException(status_code=404, detail=f"Draft {wanted} not found.")
        chosen = match
        chosen_label = str(wanted)

    try:
        if chosen.suffix == ".json" or chosen.suffixes[-2:] == [".json", ".gz"]:
            data = _read_json(chosen)
        else:
            # errors="replace" handles the rare non-UTF-8 byte without 500ing
            # the client (which would leave the UI's loading state stuck).
            data = _parse_synthesis_txt(chosen.read_text(encoding="utf-8", errors="replace"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Could not read synthesis: {e}") from e

    sources_in = data.get("sources") or {}
    if isinstance(sources_in, list):
        # tolerate older formats {url: idx} as a list of pairs
        sources = [
            CitationOut(index=int(idx), url=str(url))
            for url, idx in sources_in
        ]
    else:
        sources = [
            CitationOut(index=int(idx), url=str(url))
            for url, idx in sources_in.items()
        ]
    sources.sort(key=lambda c: c.index)

    # Parent dir of the artifact relative to repo (empty when at repo root).
    # The UI prepends this when rewriting markdown image refs through the
    # file-serving route. .as_posix() so the URL uses forward-slashes even
    # on non-POSIX hosts.
    try:
        artifact_dir = chosen.parent.relative_to(repo).as_posix()
    except ValueError:
        artifact_dir = ""
    if artifact_dir == ".":
        artifact_dir = ""

    return SynthesisOut(
        draft=chosen_label,
        abstract=data.get("abstract", ""),
        artifact=data.get("artifact", ""),
        sources=sources,
        metadata=data.get("metadata", {}) or {},
        artifact_dir=artifact_dir,
    )


# Allowed extensions for the run-file route. Restricted to image MIME types
# the markdown renderer is meant to embed; anything else (e.g. .py, .log) is
# not exposed even though it lives in the run dir.
def _find_seed_html(repo: Path) -> Path | None:
    """Locate the run's search-results seed page (the file:// root)."""
    gfiles = _list_graph_files(repo)
    if gfiles:
        try:
            su = _read_json(gfiles[-1][1]).get("starting_url")
        except Exception:  # noqa: BLE001
            su = None
        if su and su.startswith("file://"):
            p = Path(su[len("file://"):]).resolve()
            try:
                p.relative_to(repo)
                if p.suffix.lower() == ".html" and p.is_file():
                    return p
            except ValueError:
                pass
    # Fallback: newest search-results HTML on disk (prefer the merged seed).
    sr = repo / "__rome__" / "search_result"
    cands = list(sr.glob("*.html")) if sr.is_dir() else list(repo.rglob("search_result/*.html"))
    merged = [p for p in cands if p.name.startswith("multi-query-")]
    pool = merged or cands
    return max(pool, key=lambda p: p.stat().st_mtime) if pool else None


def _parse_search_results(html: str) -> list[SearchResultItem]:
    """Extract {title, url, description} from Caesar's generated search-results
    HTML. We return structured data so the UI can render it safely (React
    escapes it) instead of serving the raw, unescaped HTML."""
    soup = BeautifulSoup(html, "html.parser")
    out: list[SearchResultItem] = []
    for div in soup.select("div.result"):
        a = div.select_one("h3 a")
        url = str((a.get("href") if a else "") or "")
        # Skip tracking-redirect / relative junk that metasearch (DDGS) sometimes
        # emits, e.g. "/clev?event=StartpageResultClick&...". Real results are
        # absolute http(s) URLs.
        if not url.lower().startswith(("http://", "https://")):
            continue
        title = (a.get_text(strip=True) if a else "") or "(no title)"
        desc_el = div.select_one(".description")
        desc = desc_el.get_text(strip=True) if desc_el else ""
        out.append(SearchResultItem(title=title, url=url, description=desc))
    return out


@router.get("/{run_id}/search-results", response_model=SearchResultsOut)
async def get_search_results(
    run_id: str,
    session: AsyncSession = Depends(get_session),
    owner: str | None = Depends(current_owner),
    admin: bool = Depends(is_admin),
) -> SearchResultsOut:
    """Parse the run's search-results seed page into structured JSON.

    The seed HTML is generated with unescaped content, so we never serve it
    raw; we parse it server-side and return data the UI renders safely."""
    run = await get_owned_run(run_id, owner, session, admin=admin)
    if not run.repository:
        raise HTTPException(status_code=404, detail="Run has no repository on disk.")
    repo = Path(run.repository).resolve()
    seed = _find_seed_html(repo)
    if seed is None:
        raise HTTPException(status_code=404, detail="No search-results page for this run.")
    html = seed.read_text(encoding="utf-8", errors="replace")
    return SearchResultsOut(results=_parse_search_results(html))


_ALLOWED_FILE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}


@router.get("/{run_id}/file/{file_path:path}")
async def get_run_file(
    run_id: str,
    file_path: str,
    session: AsyncSession = Depends(get_session),
    owner: str | None = Depends(current_owner),
    admin: bool = Depends(is_admin),
) -> FileResponse:
    """Serve a file from the run's repository (images/* for embedded
    artifact images). Path-traversal guard rejects `..` segments and ensures
    the resolved path stays under run.repository."""
    # Owner check (404 on miss or cross-owner) BEFORE any filesystem access.
    run = await get_owned_run(run_id, owner, session, admin=admin)
    if not run.repository:
        raise HTTPException(status_code=404, detail="Run has no repository on disk.")
    repo = Path(run.repository).resolve()
    rel = Path(file_path)
    if rel.is_absolute() or any(part == ".." for part in rel.parts):
        raise HTTPException(status_code=403, detail="Invalid path.")
    if rel.suffix.lower() not in _ALLOWED_FILE_EXTS:
        raise HTTPException(status_code=403, detail="File type not served.")
    full = (repo / rel).resolve()
    # Resolved path must remain under repo (defense in depth against symlink
    # escapes / weird normalizations the parts-check above could miss).
    try:
        full.relative_to(repo)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid path.") from None
    if not full.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(full)
