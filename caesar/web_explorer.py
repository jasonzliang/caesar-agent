"""Web Explorer - Web fetching, content extraction, and navigation strategy"""
from collections import Counter
import io
import ipaddress
from pathlib import Path
import socket
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, quote

from bs4 import BeautifulSoup
from curl_cffi import requests
import numpy as np
from pypdf import PdfReader
import trafilatura

from rome.config import LONGER_SUMMARY_LEN
from rome.logger import get_logger
from .brave_search import BraveSearch
from .caesar_config import (MAX_NUM_LINKS, MAX_NUM_VISITED_LINKS, MAX_TEXT_LENGTH,
    REQUESTS_TIMEOUT, REQUESTS_HEADERS, BASELINE_BLOCKED_DOMAINS)


# Schemes the crawler may fetch from the open web. Everything else is refused,
# which notably rules out file:// (outside the agent's own directory), data:,
# ftp:// and gopher://.
FETCHABLE_SCHEMES = ('http', 'https')


class WebExplorer:
    """Web fetching, content extraction, and navigation strategy for CaesarAgent"""

    def __init__(self, agent, config: Dict = None):
        self.agent = agent
        self.logger = get_logger()
        self.kb_manager = agent.kb_manager
        self._last_fetched_url = None

    # ── Web Fetching ─────────────────────────────────────────────────────

    def _is_own_repository_file(self, file_path: str) -> bool:
        """True when a file:// target resolves inside the agent's own directory.

        Exploration seeds itself from a search-results page written under the
        repository, so file:// has to keep working for that one case. resolve()
        collapses `..` and follows symlinks before the containment check, so
        neither can walk out.
        """
        repo = getattr(self.agent, 'repository', None)
        if not repo:
            return False
        try:
            Path(file_path).resolve().relative_to(Path(repo).resolve())
            return True
        except (ValueError, OSError):
            return False

    @staticmethod
    def _is_public_ip(ip) -> bool:
        """is_global is already False for loopback, private, link-local, shared
        (100.64/10), reserved and unspecified ranges, so it carries that whole
        list; multicast needs saying separately. test_url_guard.py pins the
        cases that matter rather than trusting the summary."""
        return ip.is_global and not ip.is_multicast

    def is_fetchable_url(self, url: str, resolve: bool = True) -> bool:
        """Reject URLs that point anywhere except the public internet.

        The crawl frontier is built from links an LLM picks out of page content,
        and in public mode anyone can start a run, so a page can propose any URL
        it likes. Unguarded that reaches loopback services, cloud metadata, and
        the rest of the private address space.

        `resolve=False` skips DNS and judges literal addresses only. Link
        extraction uses it because it runs over up to MAX_NUM_LINKS candidates,
        and a page listing that many hostnames on a resolver that never answers
        would stall the crawl. Nothing is lost: the fetch path resolves and
        re-checks, which also normalises obfuscated literals (2130706433,
        0x7f.0.0.1) for free. Not a defence against an attacker who controls DNS
        and re-answers between check and fetch, which needs pinned addresses.
        """
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        if scheme == 'file':
            # url[7:] matches what fetch_html opens; parsed.path covers `file:/x`.
            path = url[7:] if url.startswith('file://') else parsed.path
            return self._is_own_repository_file(path)
        if scheme not in FETCHABLE_SCHEMES:
            return False
        host = parsed.hostname
        if not host:
            return False
        try:
            # hostname already strips IPv6 brackets, so literals parse directly.
            return self._is_public_ip(ipaddress.ip_address(host))
        except ValueError:
            pass  # not a literal address, so it needs resolving
        if not resolve:
            return True
        try:
            infos = socket.getaddrinfo(
                host, parsed.port or (443 if scheme == 'https' else 80),
                proto=socket.IPPROTO_TCP,
            )
        except (socket.gaierror, UnicodeError, ValueError):
            # Unresolvable or malformed: nothing to fetch either way.
            return False
        for info in infos:
            try:
                ip = ipaddress.ip_address(info[4][0])
            except ValueError:
                return False
            if not self._is_public_ip(ip):
                return False
        return True

    def is_allowed_url(self, url: str) -> bool:
        """Allow-list + blocked_domains (netloc) + blocked_urls (full-URL) substring checks."""
        # No DNS here: fetch_html re-checks with resolution before fetching.
        if not self.is_fetchable_url(url, resolve=False):
            return False
        parsed = urlparse(url)
        all_blocked = BASELINE_BLOCKED_DOMAINS + (self.agent.blocked_domains or [])
        if any(d in parsed.netloc for d in all_blocked):
            return False
        # str()-coerce so float YAML entries (e.g. arxiv 2602.05192) don't crash on .lower().
        url_substrings = getattr(self.agent, 'blocked_urls', None) or []
        if url_substrings:
            url_lower = url.lower()
            if any(s and str(s).lower() in url_lower for s in url_substrings):
                return False
        if self.agent.allow_all_domains:
            return True
        return any(domain in parsed.netloc for domain in self.agent.allowed_domains)

    def compute_nav_headers(self, url: str, referer_url: Optional[str] = None) -> Dict[str, str]:
        """Compute Referer + Sec-Fetch-Site from prev URL.

        First fetch fakes Google. Same-site labeled cross-site to avoid wrong same-origin labels.
        """
        prev = referer_url if referer_url is not None else self._last_fetched_url
        if not prev:
            return {'Referer': 'https://www.google.com/', 'Sec-Fetch-Site': 'cross-site'}
        prev_host = urlparse(prev).netloc
        curr_host = urlparse(url).netloc
        site = 'same-origin' if prev_host and prev_host == curr_host else 'cross-site'
        # Percent-encode non-ASCII so latin-1 header encoding (RFC 7230) never raises
        return {'Referer': quote(prev, safe=":/?#[]@!$&'()*+,;=~-._%"), 'Sec-Fetch-Site': site}

    def fetch_html(self, url: str, referer_url: Optional[str] = None) -> Optional[str]:
        """Fetch HTML/PDF from URL or file://.

        Explicit referer_url skips _last_fetched_url update so parallel callers don't race.
        """
        try:
            # Local file handling
            if url.startswith('file://'):
                file_path = url[7:]
                # Same string the open() below uses, so the check and the read
                # can't disagree about which file this is.
                if not self._is_own_repository_file(file_path):
                    self.logger.warning(
                        f"Refusing file:// fetch outside the agent directory: {url}")
                    return None
                if file_path.lower().endswith('.pdf'):
                    with open(file_path, 'rb') as f:
                        text = '\n\n'.join(page.extract_text() for page in PdfReader(f).pages)
                    return f"<html><body><div>{text}</div></body></html>"
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()

            # Remote URL handling. Every fetch funnels through here, including
            # the three CaesarAgent call sites that never consult
            # is_allowed_url, so this is where the address guard has to live.
            if not self.is_fetchable_url(url):
                self.logger.warning(f"Refusing fetch of non-public URL: {url}")
                return None

            headers = {**REQUESTS_HEADERS, **self.compute_nav_headers(url, referer_url)}
            response = requests.get(
                url,
                impersonate="chrome",  # Matches a modern Chrome browser
                timeout=REQUESTS_TIMEOUT,
                headers=headers,
                allow_redirects=True
            )
            # Redirects are followed, so a public URL can still land on an
            # internal one. The body comes from wherever the chain ended, so
            # re-checking that one URL is what keeps internal content from being
            # returned. The requests along the way were still issued: this
            # protects the response, not reachability.
            final_url = getattr(response, 'url', None)
            if final_url and final_url != url and not self.is_fetchable_url(final_url):
                self.logger.warning(
                    f"Refusing content: {url} redirected to non-public {final_url}")
                return None
            response.raise_for_status()
            if referer_url is None:
                self._last_fetched_url = url

            # PDF detection and extraction
            is_pdf = ('application/pdf' in response.headers.get('content-type', '').lower() or
                      response.content[:4] == b'%PDF')
            if is_pdf:
                text = '\n\n'.join(page.extract_text()
                                  for page in PdfReader(io.BytesIO(response.content)).pages)
                return f"<html><body><div>{text}</div></body></html>"

            return response.text

        except Exception as e:
            self.logger.error(f"Fetch failed for {url}: {e}")
            return None

    def extract_links(self, html: str, base_url: str) -> List[Tuple[str, str]]:
        """Extract links with anchor text"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
        except Exception:
            return []

        links = []
        seen = set()
        base_parsed = urlparse(base_url)
        base_path = f"{base_parsed.scheme}://{base_parsed.netloc}{base_parsed.path}"

        for a in soup.find_all('a', href=True):
            try:
                url = urljoin(base_url, a['href'])
            except Exception:
                continue

            if not self.agent.same_page_links:
                parsed = urlparse(url)
                url_without_fragment = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if url_without_fragment == base_path:
                    continue

            text = a.get_text(strip=True)[:LONGER_SUMMARY_LEN] or "[no text]"

            if url not in seen and \
                self.is_allowed_url(url) and \
                url not in self.agent.failed_urls and \
                self.agent.visited_urls.get(url, 0) <= self.agent.max_allowed_revisits and \
                url.startswith('http'):
                links.append((url, text))
                seen.add(url)

        return links

    # Pruned before trafilatura — sidebar-heavy sites fool its length-based heuristics.
    _PRUNE_XPATH = [
        "//nav", "//footer", "//aside", "//header",
        "//*[@role='navigation']", "//*[@role='banner']", "//*[@role='contentinfo']",
        "//*[contains(@class, 'sidebar')]", "//*[contains(@id, 'sidebar')]",
        "//*[contains(@class, 'comment')]",
        "//*[contains(@class, 'related')]",
        "//*[contains(@class, 'share')]",
        "//script", "//style", "//noscript",
    ]

    def extract_text_from_html(self, html: str, max_length: int = MAX_TEXT_LENGTH) -> str:
        """Extract clean text content from HTML using trafilatura for robust boilerplate removal"""
        try:
            common = dict(
                include_comments=False,
                include_tables=True,
                deduplicate=True,
                prune_xpath=self._PRUNE_XPATH,
            )
            # Precision first (drops boilerplate); fall back to recall for link-list
            # pages (HN, sitemaps) where precision returns empty.
            text = trafilatura.extract(html, favor_precision=True, **common)
            if not text:
                text = trafilatura.extract(html, favor_recall=True, **common)
            if not text:
                soup = BeautifulSoup(html, 'html.parser')
                for tag in soup(["script", "style", "noscript", "nav", "footer",
                                 "aside", "header", "form", "iframe"]):
                    tag.decompose()
                text = soup.get_text(separator=' ', strip=True)
            return ' '.join(text.split())[:max_length]
        except Exception as e:
            self.logger.error(f"HTML text extraction failed: {e}")
            return ""

    # ── Navigation Strategy ──────────────────────────────────────────────

    def determine_exploration_strategy(self, kb_context, memory_context) -> Dict:
        """Chooses best exploration strategy based on current knowledge and memory"""
        if self.agent.web_searches_used < self.agent.max_web_searches:
            web_search_option = f"\n4. **WEB_SEARCH** relevant topics to address current exploration insights (remaining uses: {self.agent.max_web_searches - self.agent.web_searches_used}/{self.agent.max_web_searches})"
        else:
            web_search_option = f"\n4. **WEB_SEARCH** (do NOT pick, no uses remaining)"

        # Expose the path so BACKTRACK can target a specific ancestor.
        stack_lines = "\n".join(
            f"  [{i}]{' (root)' if i == 0 else ' (current)' if i == len(self.agent.url_stack) - 1 else ''} {url}"
            for i, url in enumerate(self.agent.url_stack)
        )

        # Empty sections drop their entire block (each leads with \n\n).
        # Strategy-level anchor fires under query_influence "strategy" or "full".
        anchor_strategy = self.agent.starting_query and self.agent.query_influence in ("strategy", "full")
        query_section = f"\n\nSTARTING QUERY:\n{self.agent.starting_query}" if anchor_strategy else ""
        kb_section = f"\n\nCURRENT EXPLORATION INSIGHTS:\n{kb_context}" if kb_context else ""
        memory_section = f"\n\nHISTORICAL NAVIGATION PATTERNS:\n{memory_context}" if memory_context else ""

        goal_suffix = " that best advances the starting query" if anchor_strategy else ""
        prompt = f"""Based on accumulated knowledge and navigation patterns, determine the optimal exploration strategy{goal_suffix}.{query_section}

CURRENT EXPLORATION CONTEXT:
- Current iteration: {self.agent.current_iteration}/{self.agent.max_iterations}
- Current depth: {self.agent.current_depth}/{self.agent.max_depth}
- Current page: {self.agent.current_url}
- Web pages visited: {len(self.agent.visited_urls)}
- Average visits per page: {float(np.mean(list(self.agent.visited_urls.values())))}

CURRENT NAVIGATION PATH (oldest → current):
{stack_lines}{kb_section}{memory_section}

Analyze whether the agent should:
1. **REVISIT** previously visited pages outside the current path to deepen understanding of relevant known areas
2. **EXPLORE** new un-visited pages to discover novel information or knowledge
3. **BACKTRACK** to any ancestor or the root on the current navigation path to abandon a stagnant branch{web_search_option}

Consider:
- Knowledge gaps vs areas of saturation
- Depth, diversity, and novelty of current exploration branch
- Success patterns from previous decisions
- Risk/reward of new exploration vs consolidation
- Trade-off between current branch and revisiting earlier paths (e.g. returning to root)

IMPORTANT: Only select WEB_SEARCH if exploration has stagnated and there are still uses remaining. The less uses remaining, the most sparingly WEB_SEARCH should be used.
IMPORTANT: BACKTRACK is for dead, stagnant, or saturated branches; commit to EXPLORE/REVISIT when merely uncertain.
IMPORTANT: Use your role as a guide on how to respond!

Respond with a JSON object in this exact format:
{{
    "action": "choose one action from EXPLORE, REVISIT, BACKTRACK, or WEB_SEARCH",
    "reasoning": "strategic recommendation of exploration strategy in 2-3 sentences",
    "search_query": "short query to find relevant topics or N/A if not WEB_SEARCH",
    "backtrack_target": "URL of an ancestor from CURRENT NAVIGATION PATH to retreat to; null = default single-level pop to immediate parent, or specify any deeper ancestor URL to rewind multiple levels in one step. Use null when action is not BACKTRACK."
}}"""
        # Optional 5th Consider: bullet — re-insert inside the prompt above to enable:
        # - Trade-off between continuing on this branch and returning to root for initial search results

        try:
            response = self.agent.chat_completion(prompt,
                override_config=self.agent.exploration_llm_config,
                response_format={"type": "json_object"})
            data = self.agent.parse_json_response(response)
            if not data or "action" not in data or "reasoning" not in data or \
                "search_query" not in data:
                raise ValueError("Missing required keys in response")
            # Normalize optional backtrack_target so downstream skips defensive .get().
            data.setdefault("backtrack_target", None)
            return data
        except Exception as e:
            self.logger.error(f"Exploration strategy determination failed: {e}")
            return ""

    def get_web_search_links(self, query: str) -> List[Tuple[str, str]]:
        """Execute web search and return links"""
        if self.agent.web_searches_used >= self.agent.max_web_searches:
            self.logger.error(f"Web search limit reached during exploration: {self.agent.max_web_searches}")
            return []

        try:
            search_engine = BraveSearch(agent=self.agent, config=self.agent.config.get("BraveSearch", {}))
            search_results_url = search_engine.search_and_save([query])
            self.agent.web_searches_used += 1

            html = self.fetch_html(search_results_url)
            if not html: return []

            links = self.extract_links(html, search_results_url)
            self.logger.info(f"Web search '{query}' returned {len(links)} links")
            return links

        except Exception as e:
            self.logger.error(f"Web search during exploration failed for '{query}': {e}")
            return []

    def format_link_options(self, links: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
        """Format links into display options, returns (link_options, url_map)"""
        link_options, url_map = [], []

        if self.agent.fancy_link_display:
            parent_url = self.agent._get_parent_url()
            visited_links = [(url, text, self.agent.visited_urls.get(url, 0))
                             for url, text in links
                             if url in self.agent.visited_urls and url != self.agent.current_url]
            new_links = [(url, text)
                         for url, text in links
                         if url not in self.agent.visited_urls and url != self.agent.current_url]

            link_options.extend([
                "- INITIAL STARTING LINK -",
                f"1. [Back to starting LINK] ({self.agent.visited_urls.get(self.agent.starting_url, 0)} visits) {self.agent.starting_url}"
            ])
            url_map.append(self.agent.starting_url)

            if parent_url:
                link_options.extend([
                    "\n- IMMEDIATE PREVIOUS LINK -",
                    f"2. [Back to previous link] ({self.agent.visited_urls.get(parent_url, 0)} visits) {parent_url}"
                ])
                url_map.append(parent_url)

            if visited_links:
                # Split ancestors vs others: in-stack truncates (backtrack), out-of-stack
                # pushes (revisit). Without distinct labels the LLM picks an ancestor
                # intending revisit and gets silently backtracked.
                already_shown = {self.agent.starting_url}
                if parent_url:
                    already_shown.add(parent_url)
                stack_set = set(self.agent.url_stack[:-1]) - already_shown
                ancestors = [(u, t, c) for (u, t, c) in visited_links
                             if u in stack_set]
                others = [(u, t, c) for (u, t, c) in visited_links
                          if u not in stack_set and u not in already_shown]
                if ancestors:
                    link_options.append(
                        "\n- DEEPER ANCESTORS ON CURRENT PATH (picking = backtrack) -")
                    for url, text, visit_count in ancestors[:MAX_NUM_VISITED_LINKS]:
                        link_options.append(f"{len(url_map)+1}. [{text}] ({visit_count} visits) {url}")
                        url_map.append(url)
                if others:
                    link_options.append(
                        "\n- PREVIOUSLY VISITED ELSEWHERE (picking = revisit) -")
                    for url, text, visit_count in others[:MAX_NUM_VISITED_LINKS]:
                        link_options.append(f"{len(url_map)+1}. [{text}] ({visit_count} visits) {url}")
                        url_map.append(url)

            if new_links:
                link_options.append("\n- NEW UNEXPLORED LINKS -")
                for url, text in new_links[:MAX_NUM_LINKS]:
                    link_options.append(f"{len(url_map)+1}. [{text}] {url}")
                    url_map.append(url)
        else:
            for url, text in links[:MAX_NUM_LINKS]:
                if url == self.agent.current_url:
                    continue
                visit_count = self.agent.visited_urls.get(url, 0)
                visit_info = f" ({visit_count} visits so far) " if visit_count else " "
                link_options.append(f"{len(url_map)+1}. [{text}]{visit_info}{url}")
                url_map.append(url)

        return link_options, url_map

    def recall_navigation_history(self, kb_context: str = None, num_kb_terms: int = 5) -> str:
        """Recall navigation history with optional context awareness"""
        # Base navigation query
        base_query = "visited domains pages backtracked repeated"

        # If kb_context exists and is substantial, enhance query
        if kb_context:
            # Extract and count key terms
            words = kb_context.lower().split()
            stop_words = {'the', 'a', 'an', 'and', 'or', 'is', 'are', 'was', 'were',
                          'this', 'that', 'with', 'from', 'about', 'has', 'have'}

            # Filter and clean words
            filtered_words = [w.strip('.,!?:;') for w in words
                             if len(w) > 5 and w not in stop_words]

            # Count frequency and get top 5
            word_counts = Counter(filtered_words)
            key_terms = [term for term, count in word_counts.most_common(num_kb_terms)]

            if key_terms:
                # Append up to 5 key terms to base query
                enhanced_query = f"{base_query} {' '.join(key_terms)}"
                return self.agent.recall(enhanced_query)

        # TODO: Use agent memory's summarize_past instead of recall
        # Default is just navigation query
        return self.agent.recall(base_query)

    def select_next_link(self, links: List[Tuple[str, str]]) -> Tuple[Optional[str], str]:
        """Use LLM to select best link, returns (url, reason)"""
        kb_context = memory_context = explore_strategy = ""
        # Per-hop anchor only under query_influence="full"; "strategy"/"none"
        # use the neutral prompt to preserve drift.
        anchor_link_select = self.agent.starting_query and self.agent.query_influence == "full"
        try:
            # TODO: Replace with Qdrant or SummaryIndex?
            if anchor_link_select:
                # Hybrid framing: anchor to the starting query but ask for gap analysis,
                # not retrieval. LLMs are better at "what's missing from this sample"
                # than "find me more of this."
                kb_context = self.kb_manager.query(
                    f"In the context of investigating '{self.agent.starting_query}', "
                    "what patterns, gaps, or questions have emerged from our knowledge? "
                    "What should we explore next?"
                )
            else:
                # Content-shaped multi-aspect prompt: longer, declaration-style, multiple
                # keyword zones — embeds closer to actual insight chunks than a short
                # question does, which improves single-call retrieval relevance for free.
                kb_context = self.kb_manager.query(
                    "Recent insights, themes, and findings from our exploration. "
                    "Key open questions, contradictions, and underexplored aspects. "
                    "Patterns across sources, gaps in current knowledge, "
                    "and promising directions worth investigating next."
                )
            memory_context = self.recall_navigation_history(kb_context)

            if self.agent.use_explore_strategy:
                strat = self.determine_exploration_strategy(kb_context, memory_context)
                if strat['action'] == "WEB_SEARCH":
                    explore_strategy = ""
                    web_search_links = self.get_web_search_links(strat['search_query'])
                    if web_search_links: links = web_search_links
                elif strat['action'] == "BACKTRACK":
                    # Strategy decider already chose; skip the second LLM call.
                    # Use explicit target if in-stack, else fall back to immediate parent.
                    target = strat.get('backtrack_target')
                    if target and target in self.agent.url_stack[:-1]:
                        return target, strat['reasoning']
                    parent = self.agent._get_parent_url()
                    if parent:
                        return parent, strat['reasoning']
                    # Already at root — fall through to normal selection.
                    explore_strategy = f"{strat['reasoning']}"
                else:
                    explore_strategy = f"{strat['reasoning']}"

        except Exception as e:
            self.logger.error(f"KB/memory/explore for link selection failed: {e}")

        link_options, url_map = self.format_link_options(links)

        # Empty sections drop their entire block (each leads with \n\n).
        query_section = f"\n\nSTARTING QUERY:\n{self.agent.starting_query}" if anchor_link_select else ""
        kb_section = f"\n\nCURRENT EXPLORATION INSIGHTS:\n{kb_context}" if kb_context else ""
        memory_section = f"\n\nHISTORICAL EXPLORATION PATTERNS:\n{memory_context}" if memory_context else ""
        strategy_section = f"\n\nEXPLORATION STRATEGY:\n{explore_strategy}" if explore_strategy else ""
        link_block = "\n".join(link_options)

        # TASK line lists only the sections actually included above.
        task_inputs = ", ".join(filter(None, [
            "current insights" if kb_context else "",
            "historical patterns" if memory_context else "",
            "exploration strategy" if explore_strategy else "",
        ]))
        if not task_inputs: task_inputs = "the paths available"

        query_suffix = " for the starting query" if anchor_link_select else ""
        goal_suffix = f"deepens understanding and is an promising/novel direction to visit{query_suffix}"

        prompt = f"""You are selecting the next webpage link to explore{query_section}

CURRENT EXPLORATION CONTEXT:
- Current iteration: {self.agent.current_iteration}/{self.agent.max_iterations}
- Current depth: {self.agent.current_depth}/{self.agent.max_depth}
- Web pages visited: {len(self.agent.visited_urls)}
- Average visits per page: {float(np.mean(list(self.agent.visited_urls.values())))}
- Current URL: {self.agent.current_url}{kb_section}{memory_section}{strategy_section}

AVAILABLE PATHS FORWARD:
{link_block}

TASK: Based on {task_inputs}, which page link {goal_suffix}?

IMPORTANT: Use your role as a guide on how to respond!

Respond with a JSON object in this exact format:
{{
    "choice": <number from 1 to {len(url_map)}>,
    "reason": "<brief explanation of why this path is promising>"
}}

Your response must be valid JSON only, nothing else."""

        try:
            response = self.agent.chat_completion(
                prompt,
                override_config=self.agent.exploration_llm_config,
                response_format={"type": "json_object"}
            )
            decision = self.agent.parse_json_response(response)
            if decision and 'choice' in decision:
                choice_num = max(0, min(int(decision['choice']) - 1, len(url_map) - 1))
                url = url_map[choice_num]
                reason = decision.get('reason', 'No reason provided')
                return url, reason
        except Exception as e:
            self.logger.error(f"LLM decision failed: {e}")

        return url_map[0] if url_map else None, "Fallback due to error"
