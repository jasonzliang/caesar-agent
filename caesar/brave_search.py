"""BraveSearch with hardcoded constants and automatic file naming"""
import time
import re
import os
import requests
import sys
from typing import Dict, Optional, List, Union, Tuple
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rome.config import (DEFAULT_CONFIG, set_attributes_from_config,
                     SHORT_SUMMARY_LEN, SUMMARY_LENGTH, LONG_SUMMARY_LEN,
                     LONGER_SUMMARY_LEN, LONGEST_SUMMARY_LEN)
from rome.logger import get_logger
from rome.parsing import hash_string
from .caesar_config import CAESAR_CONFIG


# ── DDGS Wikipedia-engine monkeypatch ────────────────────────────────────
# DDGS's Wikipedia engine (ddgs/engines/wikipedia.py) naively does
# `_, lang = region.lower().split("-")` and uses lang as a Wikipedia
# language subdomain. For any region whose second half isn't a real
# Wikipedia language (e.g. "wt-wt" worldwide -> wt.wikipedia.org, DNS-fails),
# the engine raises ConnectError and breaks the whole metasearch.
#
# We patch the engine's build_payload to fall back to "en" whenever the
# parsed language wouldn't yield a valid Wikipedia subdomain. This is more
# robust than excluding the engine from the backend list because it works
# regardless of how DDGS is called downstream — if anyone changes our
# `backend=` argument or DDGS rotates its default engine set, the patch
# still applies. Confirmed needed in ddgs 9.14.2 and 9.14.4.
def _install_ddgs_wikipedia_patch() -> None:
    try:
        from ddgs.engines.wikipedia import Wikipedia as _Wikipedia
    except ImportError:
        return  # DDGS not installed; only the Brave path is used
    if getattr(_Wikipedia, "_caesar_lang_patched", False):
        return  # already patched (multi-import safe)
    _orig_build_payload = _Wikipedia.build_payload
    # Known-bad region halves: DDGS conventions that aren't real Wikipedia
    # languages. "wt" is DuckDuckGo's "worldwide" sentinel, not Wolof (wo).
    _BAD_LANGS = frozenset({"wt", "xx", "ww", "all"})
    def _safe_build_payload(self, query, region, safesearch, timelimit,
                            page=1, **kwargs):
        try:
            _country, lang = region.lower().split("-")
        except ValueError:
            lang = ""
        # Wikipedia language subdomains are alphabetic, typically 2-3 chars
        # (some go longer, e.g. "simple", "zh-yue"). Reject only obvious
        # garbage and the known-bad set; let the engine try anything else
        # because there are 300+ valid Wikipedia languages.
        if not lang.isalpha() or len(lang) > 12 or lang in _BAD_LANGS:
            region = "us-en"
        return _orig_build_payload(self, query, region, safesearch,
                                   timelimit, page, **kwargs)
    _Wikipedia.build_payload = _safe_build_payload
    _Wikipedia._caesar_lang_patched = True


_install_ddgs_wikipedia_patch()

# Search endpoint
ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
# Result directory in which to store results
SEARCH_RESULT_DIR = "search_result"
# Max number of search results per API request
MAX_NUM_RESULTS_PER_PAGE = 20
# Max total results supported via pagination (offset max is 9, so 10 pages x 20)
MAX_NUM_RESULTS = 200
# Multiplier for backoff during search
BACKOFF_MULTIPLIER = 2
# Cap on per-retry sleep. Without this the exponential backoff would reach
# 2^N seconds at retry N (e.g. retry 30 = 34 years), turning a transient
# endpoint outage into an unkillable hang. 30s is generous for any real
# transient and lets max_retries actually bound wall-clock duration.
MAX_BACKOFF_DELAY = 30
# Delay in sec between each search to avoid hitting rate limit
SEARCH_DELAY = 1
# Brave API query limits
MAX_QUERY_CHARS = 400; MAX_QUERY_WORDS = 50


class BraveSearchError(Exception):
    """Base exception for BraveSearch errors"""
    pass
class RateLimitError(BraveSearchError):
    """Rate limit exceeded"""
    pass
class APIKeyError(BraveSearchError):
    """Invalid API key"""
    pass


class BraveSearch:
    """Convert Brave Search API results to local HTML file with retry logic"""

    def __init__(self, agent, config: Dict = None):
        self.agent = agent
        self.logger = get_logger()

        # Load settings from config first so use_ddgs is set before we decide
        # which backend to run with.
        self.config = config
        set_attributes_from_config(self, self.config, CAESAR_CONFIG['BraveSearch'].keys())

        # Pick a backend: explicit config override, or auto-fallback to DDGS
        # when BRAVE_API_KEY is not in the environment. self.api_key is None
        # in DDGS mode and is only consulted by the Brave path.
        self.api_key = os.getenv("BRAVE_API_KEY")
        if not self.use_ddgs and not self.api_key:
            self.logger.info(
                "BRAVE_API_KEY not set; falling back to DDGS metasearch backend")
            self.use_ddgs = True
        elif self.use_ddgs:
            self.logger.info("DDGS backend forced via use_ddgs config")

        if self.num_results > MAX_NUM_RESULTS:
            self.logger.error(f"num_results={self.num_results} exceeds max ({MAX_NUM_RESULTS}), capping")
            self.num_results = MAX_NUM_RESULTS

        if self.shorten_query not in ("truncation", "summary"):
            raise ValueError(
                f"shorten_query must be 'truncation' or 'summary', "
                f"got {self.shorten_query!r}")

        self.output_dir = os.path.join(self.agent.get_log_dir(), SEARCH_RESULT_DIR)
        os.makedirs(self.output_dir, exist_ok=True)

    def _query_exceeds_limits(self, query: str) -> bool:
        """Check if a query exceeds Brave API limits."""
        return len(query) > MAX_QUERY_CHARS or len(query.split()) > MAX_QUERY_WORDS

    def _truncate_query(self, query: str) -> str:
        """Truncate query to fit within Brave API limits."""
        words = query.split()
        if len(words) > MAX_QUERY_WORDS:
            words = words[:MAX_QUERY_WORDS]
        truncated = ' '.join(words)
        if len(truncated) > MAX_QUERY_CHARS:
            truncated = truncated[:MAX_QUERY_CHARS].rsplit(' ', 1)[0]
        return truncated

    def _summarize_query(self, query: str) -> str:
        """Use the agent's LLM to summarize a query to fit Brave API limits."""
        prompt = (
            f"Condense the following search query into a shorter search query "
            f"that preserves the key intent and important keywords. "
            f"The result MUST be under {MAX_QUERY_CHARS} characters and "
            f"under {MAX_QUERY_WORDS} words. Return ONLY the shortened query, "
            f"nothing else.\n\nOriginal query:\n{query}"
        )
        # num_retries=0: fails soft to _truncate_query below, so don't let
        # litellm stack up to 3x the request timeout on a hung call (would
        # otherwise reproduce the "Worker stalled" watchdog failure).
        shortened = self.agent.chat_completion(prompt, num_retries=0).strip().strip('"\'')
        if not shortened:
            self.logger.error("LLM returned empty summary, falling back to truncation")
            shortened = self._truncate_query(query)
        elif self._query_exceeds_limits(shortened):
            self.logger.error("LLM summary still exceeds limits, falling back to truncation")
            shortened = self._truncate_query(shortened)
        return shortened

    def _shorten_query(self, query: str) -> str:
        """Shorten a query if it exceeds Brave API limits. self.shorten_query
        is validated at init to be 'truncation' or 'summary', so this method
        never has to handle a disabled or unknown value."""
        if not self._query_exceeds_limits(query):
            return query

        self.logger.info(
            f"Shortening query ({len(query)} chars, {len(query.split())} words) "
            f"using method: {self.shorten_query}")

        if self.shorten_query == "truncation":
            shortened = self._truncate_query(query)
        else:  # "summary" — guaranteed by init validation
            shortened = self._summarize_query(query)

        self.logger.info(
            f"Shortened query: {len(shortened)} chars, {len(shortened.split())} words")
        return shortened

    def _generate_filename(self, query) -> str:
        """Generate filename from query and timestamp"""
        # Sanitize query for filename (remove special chars, limit length)
        if type(query) is list and len(query) > 1:
            query = f"multi-query-{len(query)}_{query[0]}"
        elif type(query) is list and len(query) == 1:
            query = query[0]
        self.logger.assert_true(type(query) is str, "Query must be string or list")

        safe_query = re.sub(r'[^\w\s-]', '', query)
        safe_query = re.sub(r'[-\s]+', '-', safe_query)
        safe_query = safe_query[:SHORT_SUMMARY_LEN]  # Limit length

        # Add timestamp/hash for uniqueness
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_hash = hash_string(query)[:8]

        return f"{safe_query}_{query_hash}.html"

    def is_cached(self, queries: Union[str, List[str]]) -> Optional[str]:
        """Return file:// URL of the cached HTML if it exists, else None.
        Lightweight probe — does not execute a search."""
        filename = self._generate_filename(queries)
        html_file = os.path.abspath(os.path.join(self.output_dir, filename))
        return f"file://{html_file}" if os.path.exists(html_file) else None

    def search_and_save(self, queries: Union[str, List[str]], use_cache: bool = True) -> str:
        """Execute search with retries and return local file URL.
        When use_cache=False, bypass the cached-html short-circuit so the
        search is re-run and the file is overwritten."""
        # Normalize to list
        query_list = [queries] if isinstance(queries, str) else queries
        self.logger.debug(f"Brave search queries: {query_list}")

        # Setup file name and check cached file
        filename = self._generate_filename(queries)
        html_file = os.path.abspath(os.path.join(self.output_dir, filename))
        if use_cache and os.path.exists(html_file):
            self.logger.debug(f"Using cached search results html file: {html_file}")
            return f"file://{html_file}"

        # Shorten queries that exceed Brave API limits
        query_list = [self._shorten_query(q) for q in query_list]

        # Execute all searches
        all_results = []
        for q in query_list:
            all_results.append((q, self._search_with_retry(q)))
            time.sleep(SEARCH_DELAY)

        # Generate HTML (merged or single)
        html = (self._json_to_html_merged(all_results) if len(query_list) > 1
                else self._json_to_html(all_results[0][1], all_results[0][0]))

        for _, results in all_results:
            self.logger.debug(f"Brave search results: {results}")

        # Write html to file in log dir
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html)

        self.logger.debug(f"Saved search results to html file: {html_file}")
        return f"file://{html_file}"

    def _search_with_retry(self, query: str) -> Dict:
        """Run the active backend with retry. DDGS path uses simple
        exception-based retry; Brave path keeps the detailed status-code
        handling that knows about 401/429/5xx semantics."""
        if self.use_ddgs:
            return self._search_ddgs_with_retry(query)
        return self._search_brave_with_retry(query)

    def _search_ddgs_with_retry(self, query: str) -> Dict:
        """Query DDGS (DuckDuckGo metasearch) and reshape the results to
        match the Brave dict layout so downstream HTML generation works
        unchanged. Retries any exception with the same backoff config as
        the Brave path; DDGS exception types are not stable across
        versions so we catch broadly here."""
        try:
            from ddgs import DDGS
        except ImportError as e:
            raise BraveSearchError(
                "DDGS backend requested but the 'ddgs' package is not "
                "installed. Install it with: pip install ddgs"
            ) from e

        # 10s per-engine timeout (default 5 is too tight for ddgs's metasearch
        # fan-out across multiple providers). region=wt-wt avoids US-en bias
        # for research queries; safesearch=off so technical/medical content
        # isn't filtered. The Wikipedia engine's region-as-subdomain bug is
        # neutralized by _install_ddgs_wikipedia_patch() at module import.
        client = DDGS(timeout=10)
        delay = self.retry_delay
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                raw = client.text(
                    query,
                    region="wt-wt",
                    safesearch="off",
                    max_results=self.num_results,
                ) or []
                return {
                    "web": {"results": [
                        {"title": r.get("title", ""),
                         "url": r.get("href", ""),
                         "description": r.get("body", "")}
                        for r in raw
                    ]},
                    "query": {"original": query},
                }
            except Exception as e:
                last_exception = e
                self.logger.error(
                    f"DDGS error: {e}. Retry {attempt + 1}/{self.max_retries} "
                    f"after {delay}s...")
                time.sleep(delay)
                delay = min(delay * BACKOFF_MULTIPLIER, MAX_BACKOFF_DELAY)

        raise BraveSearchError(
            f"DDGS failed after {self.max_retries} retries"
        ) from last_exception

    def _search_brave_with_retry(self, query: str) -> Dict:
        """Execute Brave search with exponential backoff retry logic"""
        last_exception = None
        delay = self.retry_delay

        for attempt in range(self.max_retries):
            try:
                return self._search_brave(query)

            except requests.exceptions.HTTPError as e:
                last_exception = e

                if e.response.status_code == 401:
                    raise APIKeyError("Invalid or expired API key") from e

                elif e.response.status_code == 429:
                    # Rate limit - check headers for retry time
                    retry_after = e.response.headers.get('Retry-After')
                    if retry_after:
                        wait_time = int(retry_after)
                        self.logger.error(f"Rate limit hit. Waiting {wait_time}s (from Retry-After header)...")
                        time.sleep(wait_time)
                    else:
                        self.logger.error(f"Rate limit hit. Retry {attempt + 1}/{self.max_retries} after {delay}s...")
                        time.sleep(delay)
                        delay = min(delay * BACKOFF_MULTIPLIER, MAX_BACKOFF_DELAY)

                    if attempt == self.max_retries - 1:
                        raise RateLimitError(f"Rate limit exceeded after {self.max_retries} retries") from e

                elif e.response.status_code >= 500:
                    # Server error - retry
                    self.logger.error(f"Server error {e.response.status_code}. Retry {attempt + 1}/{self.max_retries} after {delay}s...")
                    time.sleep(delay)
                    delay = min(delay * BACKOFF_MULTIPLIER, MAX_BACKOFF_DELAY)

                    if attempt == self.max_retries - 1:
                        raise BraveSearchError(f"Server error after {self.max_retries} retries") from e
                else:
                    # Other HTTP error - don't retry
                    raise BraveSearchError(f"HTTP {e.response.status_code}: {e}") from e

            except requests.exceptions.Timeout as e:
                last_exception = e
                self.logger.error(f"Request timeout. Retry {attempt + 1}/{self.max_retries} after {delay}s...")
                time.sleep(delay)
                delay = min(delay * BACKOFF_MULTIPLIER, MAX_BACKOFF_DELAY)

                if attempt == self.max_retries - 1:
                    raise BraveSearchError(f"Request timeout after {self.max_retries} retries") from e

            except requests.exceptions.RequestException as e:
                last_exception = e
                self.logger.error(f"Network error. Retry {attempt + 1}/{self.max_retries} after {delay}s...")
                time.sleep(delay)
                delay = min(delay * BACKOFF_MULTIPLIER, MAX_BACKOFF_DELAY)

                if attempt == self.max_retries - 1:
                    raise BraveSearchError(f"Network error after {self.max_retries} retries") from e

        # Should never reach here, but just in case
        raise BraveSearchError(f"Failed after {self.max_retries} retries") from last_exception

    def _search_brave(self, query: str) -> Dict:
        """Query Brave Search API with pagination if num_results > 20"""
        headers = {
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'X-Subscription-Token': self.api_key
        }

        if self.num_results <= MAX_NUM_RESULTS_PER_PAGE:
            # Single request
            params = {'q': query, 'count': self.num_results}
            response = requests.get(
                ENDPOINT, headers=headers, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()

        # Paginated requests
        all_web_results = []
        remaining = self.num_results
        offset = 0
        first_response = None

        while remaining > 0:
            count = min(remaining, MAX_NUM_RESULTS_PER_PAGE)
            params = {'q': query, 'count': count, 'offset': offset}
            response = requests.get(
                ENDPOINT, headers=headers, params=params, timeout=self.timeout)

            if response.status_code == 422 and offset > 0:
                # API plan doesn't support this offset; return what we have
                self.logger.error(
                    f"Brave API 422 at offset={offset}, pagination not supported; "
                    f"returning {len(all_web_results)} results")
                break

            response.raise_for_status()
            data = response.json()

            if first_response is None:
                first_response = data

            page_results = data.get('web', {}).get('results', [])
            if not page_results:
                break

            all_web_results.extend(page_results)
            remaining -= len(page_results)
            offset += count

            if len(page_results) < count:
                break  # No more results available

            if remaining > 0:
                time.sleep(SEARCH_DELAY)

        # Merge paginated results into first response
        if first_response is None:
            return {'web': {'results': []}}
        if 'web' not in first_response:
            first_response['web'] = {}
        first_response['web']['results'] = all_web_results
        return first_response

    def _json_to_html(self, search_results: Dict, query: str) -> str:
        """Convert JSON to detailed HTML"""
        results = search_results.get('web', {}).get('results', [])
        api_query = search_results.get('query', {}).get('original', query)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{query}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 20px auto; padding: 0 20px; }}
        h1 {{ color: #1a0dab; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
        .query-info {{ background: #f0f0f0; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .query-info strong {{ color: #333; }}
        .result {{ margin: 25px 0; padding: 15px; border-left: 3px solid #4285f4; background: #f9f9f9; }}
        .result h3 {{ margin: 0 0 8px 0; }}
        .result a {{ color: #1a0dab; text-decoration: none; font-size: 18px; }}
        .result a:hover {{ text-decoration: underline; }}
        .url {{ color: #006621; font-size: 14px; margin: 5px 0; }}
        .description {{ color: #545454; margin: 10px 0; line-height: 1.5; }}
        .meta {{ color: #70757a; font-size: 13px; margin-top: 8px; }}
    </style>
</head>
<body>
    <h1>Search Results: {api_query}</h1>
    <div class="query-info">
        <strong>Query:</strong> {query}<br>
        <strong>Search Results:</strong> {len(results)}
    </div>
"""

        for r in results:
            title = r.get('title', 'No Title')
            url = r.get('url', '')
            description = r.get('description', 'No description available')
            language = r.get('language', 'N/A')
            page_age = r.get('page_age', 'N/A').split('T')[0] if r.get('page_age') else 'N/A'

            html += f"""
    <div class="result">
        <h3><a href="{url}">{title}</a></h3>
        <div class="url">{url}</div>
        <div class="description">{description}</div>
        <div class="meta">Language: {language} | Published: {page_age}</div>
    </div>
"""

        html += """
</body>
</html>
"""
        return html

    def _json_to_html_merged(self, query_results: List[Tuple[str, Dict]]) -> str:
        """Convert multiple query results to merged HTML"""
        total_results = sum(len(results.get('web', {}).get('results', []))
                          for _, results in query_results)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Merged Search Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 20px auto; padding: 0 20px; }}
        h1 {{ color: #1a0dab; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
        h2 {{ color: #1a0dab; margin-top: 30px; font-size: 20px; }}
        .summary {{ background: #f0f0f0; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .result {{ margin: 15px 0; padding: 12px; border-left: 3px solid #4285f4; background: #f9f9f9; }}
        .result h3 {{ margin: 0 0 6px 0; }}
        .result a {{ color: #1a0dab; text-decoration: none; font-size: 16px; }}
        .result a:hover {{ text-decoration: underline; }}
        .url {{ color: #006621; font-size: 13px; margin: 4px 0; }}
        .description {{ color: #545454; margin: 8px 0; line-height: 1.4; }}
    </style>
</head>
<body>
    <h1>Merged Search Results</h1>
    <div class="summary">
        <strong>Queries:</strong> {len(query_results)} | <strong>Total Search Results:</strong> {total_results}
    </div>
"""

        seen_urls = set()
        for query, search_results in query_results:
            results = search_results.get('web', {}).get('results', [])
            api_query = search_results.get('query', {}).get('original', query)

            html += f"""
    <h2>Query: {api_query}</h2>
"""

            for r in results:
                title = r.get('title', 'No Title')
                url = r.get('url', '')
                description = r.get('description', 'No description available')

                if self.filter_duplicates and url in seen_urls:
                    continue
                seen_urls.add(url)

                html += f"""
    <div class="result">
        <h3><a href="{url}">{title}</a></h3>
        <div class="url">{url}</div>
        <div class="description">{description}</div>
    </div>
"""

        html += """
</body>
</html>
"""
        return html