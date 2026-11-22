"""Citation and Markdown text processing for synthesized artifacts.

Stateless helpers used by ArtifactSynthesizer: find numbered [n] citations in
prose (skipping code, math, links, and URLs), remap them between draft and
global source namespaces, normalize emphasis, and parse saved synthesis text.
"""
import re
from typing import Dict, Optional


def format_source_list(sources: Dict[str, int]) -> str:
    """Format a {url: idx} dict as a sorted '[idx] url' string."""
    return "\n".join(f"[{idx}] {url}"
        for url, idx in sorted(sources.items(), key=lambda x: x[1]))


# The .txt layout written by ArtifactSynthesizer._save_synthesis: optional
# ABSTRACT, ARTIFACT, and an optional trailing SOURCES list of "[n] url" lines
# (older files also end with a METADATA JSON block). Strict on purpose, to keep
# source identity; the web API's lenient display parser is
# web_server/api/app/routers/artifacts.py:_parse_synthesis_txt.
_SYNTHESIS_TXT_RE = re.compile(
    r"\A(?:ABSTRACT:\n.*?\n\n)?ARTIFACT:\n(?P<artifact>.*?)"
    r"(?:\n+SOURCES:\n(?P<sources>(?:\[\d+\] [^\n]*(?:\n|\Z))+))?"
    r"(?:\n+METADATA:\n\{.*\})?\Z",
    re.DOTALL)


def parse_synthesis_txt(content: str) -> Optional[Dict]:
    """Parse Caesar's saved synthesis .txt into {"artifact", "sources"}, or None."""
    match = _SYNTHESIS_TXT_RE.match(content.replace("\r\n", "\n").rstrip())
    if not match or not match.group("artifact").strip():
        return None
    sources = {url: int(idx) for idx, url in re.findall(
        r"^\[(\d+)\] (\S+)[ \t]*$", match.group("sources") or "", re.MULTILINE)}
    return {"artifact": match.group("artifact"), "sources": sources}


# Explicit [n](url) links carry their own URL, not a source-map ID. Source
# IDs start at 1, so groups containing 0 (e.g. the interval [0,1]) are
# left alone rather than half-remapped.
CITE_RE = re.compile(r'\[\s*([1-9]\d*(?:\s*,\s*[1-9]\d*)*)\s*\](?!\()')


# Spans where numeric brackets and '*' are literal syntax rather than prose:
# fenced/indented code, single-paragraph inline code, LaTeX math, links/URLs,
# and backslash-escaped characters. Link destinations stay on one line and
# use disjoint alternatives so an unclosed "[label](" cannot backtrack
# exponentially.
PROTECTED_SPAN_RE = re.compile(
    # Fence opener only; _fence_end finds where the block closes.
    r"(?P<fence>^(?P<fence_indent>[ \t]*)(?P<fence_delim>`{3,}|~{3,})[^\n]*\n)"
    r"|(?P<indented>(?:\A|(?<=\n\n)|(?<=\r\n\r\n))(?:(?: {4}|\t)[^\n]*(?:\r?\n|$)+)+)"
    r"|(?P<inline_code>(?P<ticks>`+)(?!`)(?:(?<!\n)\n(?![ \t]*(?:[-*+]|\d+[.)]|>|#)[ \t]|\r?\n)|[^\n])*?(?<!`)(?P=ticks)(?!`))"
    r"|(?P<math>\\\((?:\\[\s\S]|[^\\])*?\\\)|\\\[(?:\\[\s\S]|[^\\])*?\\\])"
    r"|(?P<link_def>^[ \t]{0,3}\[[^\]\n]+\]:[ \t]*\S+[^\n]*)"
    r"|(?P<link>!?\[(?:[^\[\]\n]|\[[^\]\n]*\])*\]\((?:<[^>\n]*>[^)\n]*|[^)<\n][^)\n]*)?\))"
    r"|(?P<ref_link>!?\[(?P<ref_text>(?:[^\[\]\n]|\[[^\]\n]*\])*)\]\[(?P<ref_label>[^\]\n]*)\])"
    r"|(?P<autolink><https?://[^>\s]+>)"
    r"|(?P<bare_url>https?://[^\s<>()]+(?<![.,;:!?]))"
    r"|(?P<escape>\\[\s\S])",
    re.MULTILINE,
)


_LINK_DEF_LABEL_RE = re.compile(r"^[ \t]{0,3}\[([^\]\n]+)\]:", re.MULTILINE)


_LIST_LINE_RE = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+")


def _link_label(label: str) -> str:
    return " ".join(label.split()).casefold()


def _continues_list_item(text: str, start: int) -> bool:
    """Whether an indented block at `start` continues an earlier list item.

    Walks back over blank and indented lines to the paragraph that owns the
    block; a list marker there (or on a nested indented line) makes the
    block list prose rather than an indented code block.
    """
    lines = text[:start].rstrip("\r\n").splitlines()
    i = len(lines) - 1
    while i >= 0 and (not lines[i].strip() or lines[i][:1] in (" ", "\t")):
        if _LIST_LINE_RE.match(lines[i]):
            return True
        i -= 1
    while i >= 0 and lines[i].strip():
        if lines[i].lstrip().startswith(("```", "~~~", "#")):
            return False  # a fence or heading ends any list above it
        if _LIST_LINE_RE.match(lines[i]):
            return True
        i -= 1
    return False


def _fence_end(text: str, match) -> int:
    """End of a fenced block: the first same-delimiter line indented at most
    3 columns past the opener; an indented (nested) fence also ends where a
    less-indented line ends its container; otherwise the end of the text."""
    indent = len(match.group("fence_indent").expandtabs(4))
    delim = match.group("fence_delim")
    pos = match.end()
    while pos < len(text):
        line_end = text.find("\n", pos)
        line_end = len(text) if line_end < 0 else line_end
        line = text[pos:line_end]
        body = line.lstrip(" \t").rstrip()
        lead = len(line[:len(line) - len(line.lstrip(" \t"))].expandtabs(4))
        if (len(body) >= len(delim) and body == delim[0] * len(body)
                and lead <= indent + 3):
            return line_end
        if body and lead < indent:
            return pos
        pos = line_end + 1
    return len(text)


def protected_spans(text: str):
    """Yield (start, end) spans of code, math, links, URLs, and escapes.

    A candidate that turns out to be prose (an indented list continuation,
    or "[a][b]" without a "[b]:" definition, e.g. adjacent citations
    "[1][2]") is rescanned from its next character, so protected spans
    nested inside it (inline code, URLs) are still found. A fence at the
    same position would have matched first, as fence is the first
    alternative.
    """
    defined_labels = None
    pos = 0
    while (match := PROTECTED_SPAN_RE.search(text, pos)):
        start, end = match.span()
        kind = match.lastgroup
        if kind == "fence":
            end = _fence_end(text, match)
        elif (kind == "indented" and start > 0
                and not match.group(0).startswith(("        ", "\t\t"))
                and _continues_list_item(text, start)):
            pos = start + 1
            continue
        elif kind == "ref_link":
            if defined_labels is None:
                defined_labels = {_link_label(m.group(1))
                                  for m in _LINK_DEF_LABEL_RE.finditer(text)}
            label = match.group("ref_label") or match.group("ref_text")
            if _link_label(label) not in defined_labels:
                pos = start + 1
                continue
        yield start, end
        pos = end


def prose_ranges(text: str):
    """Yield (start, end) slices of text outside code, math, links, URLs, and escapes."""
    cursor = 0
    for start, end in protected_spans(text):
        if start > cursor:
            yield cursor, start
        cursor = end
    if cursor < len(text):
        yield cursor, len(text)


def citation_matches(text: str):
    """Find numeric markers only in prose source ranges."""
    for start, end in prose_ranges(text):
        yield from CITE_RE.finditer(text, start, end)


def source_urls_by_index(sources) -> Dict[int, str]:
    """Accept only unambiguous positive integer IDs in an eligible URL→ID map."""
    if not isinstance(sources, dict):
        return {}
    by_index, ambiguous = {}, set()
    for url, idx in sources.items():
        if not eligible_source_url(url):
            continue
        if isinstance(idx, bool) or not isinstance(idx, (int, str)):
            continue
        if isinstance(idx, str) and not re.fullmatch(r"\d+", idx.strip()):
            continue
        try:
            idx = int(idx)
        except ValueError:
            continue
        if idx < 1:
            continue
        if idx in by_index and by_index[idx] != url:
            ambiguous.add(idx)
        by_index[idx] = url
    return {idx: url for idx, url in by_index.items() if idx not in ambiguous}


def remap_citations(text: str, mapping: Dict[int, int],
                     drop_unmapped: bool = True) -> str:
    """Rewrite [n] / [n,m,...] markers via an int→int mapping.

    With drop_unmapped=True (default), indices not in `mapping` are dropped
    so unmapped markers cannot collide across draft namespaces. That also
    drops other bracketed positive numbers in prose (e.g. a year written as
    [2024]); code, math, links, and URLs are untouched. With
    drop_unmapped=False unmapped indices pass through unchanged, as in the
    final renumbering after merge.
    """
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
    fragments = []
    cursor = 0
    for match in citation_matches(text):
        start, end = match.span()
        replacement = _sub(match)
        fragments.append(text[cursor:start])
        # Drop the space before a dropped citation that ends a clause
        # ("Claim [1]." -> "Claim.", "Claim [5][6]." -> "Claim.").
        following = text[end:end + 1]
        if not replacement and (not following or following.isspace()
                                or following in ".,;:!?)"):
            while fragments and not fragments[-1]:
                fragments.pop()
            if fragments and fragments[-1].endswith(" "):
                fragments[-1] = fragments[-1][:-1]
        fragments.append(replacement)
        cursor = end
    fragments.append(text[cursor:])
    return "".join(fragments)


def extract_cited_indices(*texts: str) -> set:
    cited: set = set()
    for text in texts:
        if not isinstance(text, str):
            continue
        for m in citation_matches(text):
            for tok in m.group(1).split(','):
                try:
                    cited.add(int(tok.strip()))
                except ValueError:
                    pass
    return cited


def eligible_source_url(url):
    return (isinstance(url, str) and bool(url.strip())
            and not url.startswith('file://')
            and 'semanticscholar.org/search' not in url)


# Private-use placeholders for protected spans while normalizing prose.
_MASK_OPEN, _MASK_CLOSE = "\ue000", "\ue001"


def normalize_formatting(text: str) -> str:
    """Normalize prose emphasis, trailing spaces, and blank-line runs without touching code, math, or links.

    Protected spans are masked with placeholders rather than processed per
    prose fragment, so emphasis that wraps inline code or a link
    (*the `kv` trick*) still pairs its own asterisks.
    """
    if _MASK_OPEN in text or _MASK_CLOSE in text:
        return text  # placeholder characters already present; leave as-is
    spans, parts, cursor = [], [], 0
    for start, end in protected_spans(text):
        parts.append(text[cursor:start])
        parts.append(f"{_MASK_OPEN}{len(spans)}{_MASK_CLOSE}")
        spans.append(text[start:end])
        cursor = end
    parts.append(text[cursor:])
    prose = "".join(parts)
    # 1. Collapse bold-wrapping-italic (**Foo *Bar***) -> **Foo Bar**
    prose = re.sub(r'\*\*([^*\n]*?)\*([^*\n]+?)\*([^*\n]*?)\*\*',
                   r'**\1\2\3**', prose)
    # 2. Convert standalone italic (*foo*) to bold (**foo**). A single '*'
    # opens only before non-space and closes only after non-space, so
    # "* item with *emphasis*" bullets and "a * b * c" stay intact.
    prose = re.sub(r'(?<!\*)\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\*)', r'**\1**', prose)
    # 3. Strip trailing horizontal whitespace at line ends in prose
    prose = re.sub(r'[ \t]+(?=\r?\n)', '', prose)
    prose = re.sub(r'[ \t]+\Z', '', prose)
    # 4. Collapse runs of >2 blank lines in prose
    prose = re.sub(r'\n{3,}', '\n\n', prose)
    return re.sub(f"{_MASK_OPEN}(\\d+){_MASK_CLOSE}", lambda m: spans[int(m.group(1))], prose)
