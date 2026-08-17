"""Caesar Agent - Web exploration agent with graph-based navigation"""
import concurrent.futures
import copy
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rome.base_agent import BaseAgent
from rome.config import set_attributes_from_config, DEFAULT_CONFIG, SHORT_SUMMARY_LEN, SUMMARY_LENGTH
from rome.logger import get_logger
from rome.kb_client import ChromaClientManager
from rome.llm_handler import FatalLLMError

from .artifact_synthesis import ArtifactSynthesizer
from .brave_search import BraveSearch
from .checkpoint import CheckpointManager
from .web_explorer import WebExplorer
from .caesar_config import NEIGHBORHOOD_DISTANCE, MAX_NUM_NEIGHBORS, MAX_NUM_ANCESTORS, MAX_RESEEDS, CAESAR_CONFIG


class CaesarAgent(BaseAgent):
    """Veni, Vidi, Vici - Web exploration agent with checkpointing support"""
    def __init__(self, name: str = None, role: str = None,
             repository: str = None, config: Dict = None,
             starting_url: str = None, allowed_domains: List[str] = None):

        # Prepare merged config BEFORE calling super().__init__()
        merged_config = self._prepare_caesar_config(config, starting_url, allowed_domains)
        # Pass the merged config to BaseAgent (this configures the logger)
        if not role: role = self._get_default_role()
        super().__init__(name, role, repository, merged_config)

        # NOW setup Caesar-specific attributes (logger is configured)
        self.caesar_config = self.config.get('CaesarAgent', {})
        set_attributes_from_config(self, self.caesar_config, CAESAR_CONFIG['CaesarAgent'].keys())

        self._setup_allowed_domains()
        self._setup_knowledge_base()
        self._setup_brave_search()
        self._setup_synthesizer()
        self.web_explorer = WebExplorer(agent=self)
        self.checkpoint_manager = CheckpointManager(agent=self)

        self._setup_exploration_state()
        if self.checkpoint_manager.load():
            self.logger.info("Resumed from checkpoint")
        else:
            self._update_role()
            self.logger.info("Starting fresh exploration")

        self._validate_caesar_config()
        self._validate_starting_url()
        self._log_initialization()

    def _prepare_caesar_config(self, config: Dict = None,
                               starting_url: str = None,
                               allowed_domains: List[str] = None) -> Dict:
        """Prepare Caesar config before parent init"""
        def deep_merge(base: Dict, overlay: Dict) -> None:
            """Deep merge overlay into base (modifies base in place)"""
            for key, value in overlay.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value

        merged_config = copy.deepcopy(DEFAULT_CONFIG)

        # Apply CAESAR_CONFIG (Caesar defaults). The deep-copy is mandatory:
        # `deep_merge` only recurses into dicts that already exist on `base`,
        # otherwise it assigns the reference. DEFAULT_CONFIG has no
        # 'CaesarAgent' key, so without this copy merged_config['CaesarAgent']
        # would BE the shared CAESAR_CONFIG['CaesarAgent'] — and the subsequent
        # `if config: deep_merge` would mutate the module-level dict in place.
        # Two concurrent agents would race to mutate the same dict, and each
        # could end up with the other's starting_query/starting_url.
        deep_merge(merged_config, copy.deepcopy(CAESAR_CONFIG))

        # Apply custom config if provided (overrides Caesar defaults)
        if config: deep_merge(merged_config, config)

        # Apply constructor parameters (highest priority)
        if starting_url:
            merged_config.setdefault('CaesarAgent', {})['starting_url'] = starting_url
        if allowed_domains:
            merged_config.setdefault('CaesarAgent', {})['allowed_domains'] = allowed_domains
        return merged_config

    def _get_default_role(self) -> str:
        """Return default Caesar role"""
        return """Your role: You are an explorer seeking novel patterns and connections in information.

Your approach:
- Identify limitations in current understanding as opportunities for deeper exploration
- Seek non-obvious connections between seemingly unrelated concepts
- Question assumptions and explore alternative interpretations
- Synthesize insights from diverse sources into novel perspectives

You navigate through information space systematically yet creatively, always within defined boundaries, building a web of understanding that reveals emergent patterns."""

    def _setup_allowed_domains(self) -> None:
        """Configure allowed domains"""
        if not self.allowed_domains:
            if self.starting_query:
                self.allowed_domains = ["*"]
                self.logger.info("Starting query detected - allowing ALL domains")
            elif self.starting_url:
                netloc = urlparse(self.starting_url).netloc
                self.allowed_domains = [netloc] if netloc else ["*"]
                self.logger.info(f"Auto-extracted domain: {netloc}" if netloc else "Cannot extract domain - allowing ALL domains")
            else:
                raise ValueError(
                    "Must provide starting_url, starting_query, and/or allowed_domains")

        self.allow_all_domains = "*" in self.allowed_domains
        if self.allow_all_domains:
            self.logger.info("Wildcard '*' detected - allowing ALL domains")

    def _setup_brave_search(self, use_cache: bool = True) -> None:
        """Generate (or regenerate) the starting_url via a fresh Brave search
        over LLM-generated queries. Safe to call multiple times. Pass
        use_cache=False to bypass the cached-html short-circuit (used by
        the reseed path to force fresh results)."""
        if not hasattr(self, 'web_searches_used'):
            self.web_searches_used = 0
        if not self.starting_query or self.max_iterations == 0:
            return

        old_starting_url = self.starting_url
        search_engine = BraveSearch(agent=self, config=self.config.get("BraveSearch", {}))

        # Cache filename depends only on starting_query and query count (via
        # the "multi-query-N_" prefix), not on the additional query *contents*,
        # so probe the cache with the expected count and skip chat_completion
        # on a hit. On miss we fall through to the full search path.
        if use_cache:
            expected_queries = [self.starting_query] + [''] * self.additional_starting_queries
            cached = search_engine.is_cached(expected_queries)
            if cached:
                self.starting_url = cached
                # Count the original (cached) queries against the budget, as
                # the pre-probe code path did on cache hit inside search_and_save.
                self.web_searches_used += len(expected_queries)
                self.logger.info(f"Overwriting existing starting_url ({old_starting_url}) with cached search results: {self.starting_url}")
                return

        # Generate additional queries if requested
        queries = [self.starting_query]
        if self.additional_starting_queries > 0:
            try:
                prompt = f"""Given this query: "{self.starting_query}"

Generate anywhere from 0 to {self.additional_starting_queries} additional search queries that would help comprehensively answer the original query. These queries should:
- Explore different aspects or angles of the original query
- Cover related concepts that provide essential context
- Include specific technical or domain-specific variations
- Be concise (1-6 words each for optimal search results)

IMPORTANT: Use your role as a guide on how to respond!

Respond with valid JSON only:
{{
"queries": ["query1", "query2", ...]
}}"""
# IMPORTANT: If no additional queries are generated, return an empty list

                response = self.chat_completion(
                    prompt,
                    response_format={"type": "json_object"},
                    # Own retry strategy: on failure this falls back to the
                    # single starting query. num_retries=0 stops litellm from
                    # stacking up to 3x the request timeout on a hung call.
                    num_retries=0,
                )
                result = self.parse_json_response(response)
                additional = result["queries"][:self.additional_starting_queries]
                queries.extend(additional)
                self.logger.info(f"Generated {len(additional)} additional queries: {additional}")

            except Exception as e:
                self.logger.error(f"Failed to generate additional queries: {e}")

        # Execute search with all queries
        self.starting_url = search_engine.search_and_save(queries, use_cache=use_cache)
        self.web_searches_used += len(queries)
        self.logger.info(f"Overwriting existing starting_url ({old_starting_url}) with query search results: {self.starting_url}")

    def _setup_knowledge_base(self) -> None:
        """Setup knowledge base"""
        self.kb_manager = ChromaClientManager(agent=self)
        self.logger.info(f"Knowledge base initialized: {self.kb_manager.info()}")

    def _setup_synthesizer(self) -> None:
        """Setup artifact synthesizer"""
        self.synthesizer = ArtifactSynthesizer(self, self.config.get("ArtifactSynthesizer", {}))
        self.logger.info(f"Artifact synthesizer initialized")

    def _update_role(self):
        """Adapt agent role based on starting URL content and optional insights"""
        try:
            # Check overwrite (highest priority)
            if self.overwrite_role_file and os.path.exists(self.overwrite_role_file):
                with open(self.overwrite_role_file, 'r', encoding='utf-8') as f:
                    if role := f.read().strip():
                        self.role = role
                        self.logger.info(f"[OVERWRITE ROLE] Using overwritten role:\n{self.role}")

            # Early return if adaptation disabled
            if not self.adapt_role: return

            # Fetch and extract content
            self.logger.info(f"[ADAPT ROLE] Analyzing {self.starting_url}")
            html = self.web_explorer.fetch_html(self.starting_url)
            content = self.web_explorer.extract_text_from_html(html) if html else ""
            if not content: return

            # Load insights if available
            insights = ""
            if self.adapt_role_file and os.path.exists(self.adapt_role_file):
                with open(self.adapt_role_file, 'r', encoding='utf-8') as f:
                    insights = f.read().strip()

            # Generate adapted role
            starting_query = f"\nSTARTING QUERY:\n{self.starting_query}\n" if self.starting_query else ""
            starting_query_task = " based on the starting query" if self.starting_query else ""

            insights_section = f"\nPRIOR INSIGHTS:\n{insights}\n" if insights else ""
            insights_info = " and prior insights" if insights else ""
            insights_task = "\n - Builds upon themes and gaps identified in prior insights" if insights else ""

            prompt = f"""You are adapting your current role based on the following starting content{insights_info}.
{starting_query}
STARTING URL:
{self.starting_url}

STARTING CONTENT:
{content}
{insights_section}
CURRENT ROLE:
{self.role}

YOUR TASK:
Using your current role as basis, analyze the page content{insights_info} to create a specialized role that:
 - Improves upon core exploration philosophy
 - Creates an overall goal for the agent to strive for{starting_query_task}
 - Focuses exploration toward most promising areas revealed by the page content{insights_task}

Provide an adapted role description (~{self.role_max_length} words) that is creative, innovative, and original!

IMPORATNT: Your response must start with "Your role:" followed by the adapted role description."""

            if not (adapted_role := self.chat_completion(prompt, num_retries=0).strip()) or len(adapted_role) < 50:
                self.logger.error("[ADAPT ROLE] Invalid LLM response, keeping default role")
                return

            # Save and apply
            self.role = adapted_role
            self.logger.info(f"[ADAPT ROLE] Using newly adapted role:\n{self.role}")

        except Exception as e:
            self.logger.error(f"[ADAPT ROLE] Role adaptation failed: {e}, keeping default role")

    def _setup_exploration_state(self) -> None:
        """Initialize exploration state"""
        self.graph = nx.DiGraph()
        # Seed root with depth=1 so it has the attribute even if think() never runs
        self.graph.add_node(self.starting_url, depth=1)
        self.visited_urls = {}
        self.failed_urls = set()
        self.url_stack = [self.starting_url]
        self.current_url = self.starting_url
        self.current_depth = len(self.url_stack)
        self.current_iteration = 0
        self.traversal_history = []
        self.session_costs = []
        self.session_start_cost = 0.0
        self.session_start_calls = 0
        self.reseeds_used = 0

    def _validate_caesar_config(self) -> None:
        """Validate Caesar-specific configuration"""
        self.logger.assert_true(self.starting_url, "starting_url required")
        self.logger.assert_true(self.allowed_domains, "allowed_domains required")
        self.logger.assert_true(self.max_iterations >= 0, "max_iterations must be >= 0")
        self.logger.assert_true(self.max_depth > 0, "max_depth must be > 0")

    def _validate_starting_url(self):
        """Validate starting URL (http/https/file)"""
        parsed = urlparse(self.starting_url)

        if parsed.scheme == 'file':
            file_path = parsed.path
            if not os.path.isfile(file_path):
                raise ValueError(f"File does not exist or not readable: {file_path}")
            if not os.access(file_path, os.R_OK):
                raise ValueError(f"File is not readable: {file_path}")
        elif parsed.scheme in ['http', 'https']:
            if not parsed.netloc:
                raise ValueError(f"Invalid URL - missing domain: {self.starting_url}")
        else:
            raise ValueError(f"URL must use http/https/file scheme: {self.starting_url}")

    def _log_initialization(self) -> None:
        """Log initialization summary"""
        self.logger.info(f"CaesarAgent '{self.name}' initialized")
        self.logger.info(f"Log dir: {self.get_log_dir()}")
        self.logger.info(f"Starting: {self.starting_url}")
        self.logger.info(f"Domains: {self.allowed_domains}")
        self.logger.info(f"Iterations: {self.max_iterations}, Depth: {self.max_depth}")

    def _log_iteration(self, iteration: int, explore_start_time: float, explore_start_iter: int) -> None:
        """Log iteration header with elapsed time and ETA"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Iteration {iteration}/{self.max_iterations}")
        self.logger.info(f"Depth: {self.current_depth}/{self.max_depth}")
        self.logger.info(f"URL: {self.current_url}")

        completed = iteration - explore_start_iter
        if completed > 0:
            elapsed = time.time() - explore_start_time
            avg_per_iter = elapsed / completed
            remaining = avg_per_iter * (self.max_iterations - iteration + 1)
            mins, secs = divmod(int(remaining), 60)
            hrs, mins = divmod(mins, 60)
            eta = f"{hrs}h {mins}m {secs}s" if hrs else f"{mins}m {secs}s"
            e_mins, e_secs = divmod(int(elapsed), 60)
            e_hrs, e_mins = divmod(e_mins, 60)
            elapsed_str = f"{e_hrs}h {e_mins}m {e_secs}s" if e_hrs else f"{e_mins}m {e_secs}s"
            self.logger.info(f"Elapsed: {elapsed_str} | ETA: {eta} remaining ({avg_per_iter:.1f}s/iter)")

        self.logger.info(f"{'='*80}")

    def perceive(self) -> Tuple[str, List[Tuple[str, str]]]:
        """Phase 1: Extract content and links from current page"""
        self.logger.info(f"[PERCEIVE] {self.current_url}")

        try:
            html = self.web_explorer.fetch_html(self.current_url)
            if not html:
                self.failed_urls.add(self.current_url)
                return "", []

            text = self.web_explorer.extract_text_from_html(html)
            links = self.web_explorer.extract_links(html, self.current_url)
            return text, links

        except Exception as e:
            self.logger.error(f"Perceive phase failed: {e}")
            self.failed_urls.add(self.current_url)
            return "", []

    def think(self, content: str) -> str:
        """Phase 2: analyze content with traversal history and extended-
        neighborhood context. Set CaesarAgent.use_orig_think: True in
        config to fall back to _orig_think (1-hop neighbors only, no
        history) for ablation reproducibility."""
        if self.use_orig_think:
            return self._orig_think(content)

        self.logger.info("[THINK] Analyzing content")
        if not content: return ""

        # Collect past insights from traversal history (unique URLs, preserve order)
        past_urls = []
        seen = set()
        for entry in self.traversal_history:
            url = entry['from_url']
            if url not in seen and url != self.current_url:
                seen.add(url)
                past_urls.append(url)
        past_insights_list = [
            (url, self.graph.nodes[url].get('insights', ''))
            for url in reversed(past_urls)
            if url in self.graph.nodes and self.graph.nodes[url].get('insights')
        ][:MAX_NUM_ANCESTORS]
        past_insights = "\n\n\n".join(
            f"[{i+1}] Source: {url}\n{insight}"
            for i, (url, insight) in enumerate(past_insights_list)
        )
        past_urls_set = {url for url, _ in past_insights_list}

        # Previous insights for current URL
        curr_insights = ''
        if self.current_url in self.graph.nodes:
            curr_insights = self.graph.nodes[self.current_url].get('insights', '')

        # Extended neighborhood: all nodes within NEIGHBORHOOD_DISTANCE hops
        related_insights_list = []
        if self.current_url in self.graph:
            undirected = self.graph.to_undirected(as_view=True)
            nearby = nx.single_source_shortest_path_length(
                undirected, self.current_url, cutoff=NEIGHBORHOOD_DISTANCE)
            for node, dist in nearby.items():
                if node == self.current_url or dist == 0:
                    continue
                if node in past_urls_set:
                    continue
                insight = self.graph.nodes[node].get('insights', '')
                if insight:
                    related_insights_list.append((node, dist, insight))
            related_insights_list.sort(key=lambda x: x[1])

        related_insights = "\n\n\n".join(
            f"[{i+1}] (dist={dist}) Source: {url}\n{insight}"
            for i, (url, dist, insight) in enumerate(related_insights_list[:MAX_NUM_NEIGHBORS])
        )

        query_task = "- How to answer the query\n" if self.starting_query else ""
        context_task = "- How this builds upon or challenges current/past/related insights" if (
            curr_insights or related_insights or past_insights) else ""

        prompt = f"""CONTENT:
{content}

QUERY:
{self.starting_query if self.starting_query else 'No query is available'}

EXISTING INSIGHTS FOR CURRENT PAGE:
{curr_insights if curr_insights else 'No current insights available'}

PAST INSIGHTS FROM EXPLORATION HISTORY:
{past_insights if past_insights else 'No past insights from exploration history'}

RELATED INSIGHTS OF NEIGHBORING PAGES:
{related_insights if related_insights else 'No related insights available'}

YOUR TASK:
Analyze CONTENT and extract key insights focusing on:
- Novel patterns or unexpected connections
- Assumptions being made and alternative perspectives
- Interesting questions raised by the content
{query_task}{context_task}

IMPORTANT: Extract insights from CONTENT while using other insights as reference
IMPORTANT: Use your role as a guide on how to respond!

Depending on the complexity of the content, provide anywhere from 1 to 6 concise but substantive insights, but do not exceed ~600 words in total length:"""

        try:
            insights = self.chat_completion(
                prompt,
                override_config=self.exploration_llm_config,
                # The explore loop is the retry strategy (a failed think is
                # caught and the iteration backtracks/continues). num_retries=0
                # stops litellm stacking up to 3x the timeout on a hung call.
                num_retries=0,
            )
        except FatalLLMError:
            raise  # propagate to explore() → job_runner so the run is marked failed
        except Exception as e:
            self.logger.error(f"LLM call failed in think phase: {e}")
            return ""

        try:
            self.kb_manager.add_text(insights, metadata={
                'url': self.current_url,
                'depth': self.current_depth,
                'iteration': self.current_iteration
            })
        except Exception as e:
            self.logger.error(f"KB add_text failed: {e}")

        self.visited_urls[self.current_url] = self.visited_urls.get(self.current_url, 0) + 1
        # depth set once at first discovery (setup/act); revisits keep it
        self.graph.add_node(self.current_url, insights=insights,
            iteration=self.current_iteration,
            visit_count=self.visited_urls[self.current_url])
        return insights

    def _orig_think(self, content: str) -> str:
        """Original think baseline kept for `use_orig_think: True` ablation
        reproducibility — the default think supersedes this with traversal
        history and extended-neighborhood context."""
        self.logger.info("[ORIG_THINK] Analyzing content")
        if not content: return ""

        # TODO: More advanced context with larger neighborhood and long-term traversal history
        prev_insights = ''; related_insights = ''
        if self.current_url in self.graph.nodes: # Removed self.graph_augmented_insights
            prev_insights = self.graph.nodes[self.current_url].get('insights', '')

            # Get neighbor URLs (not node dicts)
            neighbors = (set(self.graph.successors(self.current_url)) |
                         set(self.graph.predecessors(self.current_url))) - {self.current_url}

            # Get insights from neighbor nodes
            related_insights = [
                (n, self.graph.nodes[n].get('insights', ''))
                for n in neighbors if n in self.graph.nodes
                and self.graph.nodes[n].get('insights')
            ]
            related_insights = "\n\n".join(
                f"[{i+1}] Source: {url}\n{insight}"
                for i, (url, insight) in enumerate(related_insights[:MAX_NUM_NEIGHBORS])
            )

        query_task = "- How to answer the query\n" if self.starting_query else ""
        prev_insight_task = "- How this builds upon or challenges previous/related insights" if (prev_insights or related_insights) else ""

        prompt = f"""CONTENT:
{content}

QUERY:
{self.starting_query if self.starting_query else 'No query is available'}

PREVIOUS INSIGHTS:
{prev_insights if prev_insights else 'No previous insights available'}

RELATED INSIGHTS:
{related_insights if related_insights else 'No related insights available'}

YOUR TASK:
Analyze CONTENT and extract key insights focusing on:
- Novel patterns or unexpected connections
- Assumptions being made and alternative perspectives
- Interesting questions raised by the content
{query_task}{prev_insight_task}

IMPORTANT: Extract insights from CONTENT while using other insights as reference
IMPORTANT: Use your role as a guide on how to respond!

Depending on the complexity of the content, provide anywhere from 1 to 6 concise but substantive insights, but do not exceed ~600 words in total length:"""

        try:
            insights = self.chat_completion(
                prompt,
                override_config=self.exploration_llm_config,
                # The explore loop is the retry strategy (a failed think is
                # caught and the iteration backtracks/continues). num_retries=0
                # stops litellm stacking up to 3x the timeout on a hung call.
                num_retries=0,
            )
        except FatalLLMError:
            raise  # propagate to explore() → job_runner so the run is marked failed
        except Exception as e:
            self.logger.error(f"LLM call failed in think phase: {e}")
            return ""

        try:
            self.kb_manager.add_text(insights, metadata={
                'url': self.current_url,
                'depth': self.current_depth,
                'iteration': self.current_iteration
            })
        except Exception as e:
            self.logger.error(f"KB add_text failed: {e}")

        self.visited_urls[self.current_url] = self.visited_urls.get(self.current_url, 0) + 1
        # depth set once at first discovery (setup/act); revisits keep it
        self.graph.add_node(self.current_url, insights=insights,
            iteration=self.current_iteration,
            visit_count=self.visited_urls[self.current_url])
        return insights


    def _advance_to_url(self, url: str) -> None:
        """Advance exploration to new URL"""
        self.url_stack.append(url)
        self.current_url = url
        self.current_depth = len(self.url_stack)  # Derive from stack

    def _backtrack(self, target: Optional[str] = None) -> bool:
        """Backtrack to a URL in the current path.

        target=None: pop one level (used by auto-backtrack: depth cap,
            no-links, perceive failure).
        target=URL: pop until url_stack[-1] == target. Falls back to a
            single pop if target isn't in the stack.
        """
        if len(self.url_stack) <= 1:
            self.logger.error(
                f"Cannot backtrack, no parent link for {self.current_url} (url_stack size <= 1)")
            return False

        if target is None:
            self.url_stack.pop()
        elif target == self.url_stack[-1]:
            return True  # already there
        elif target not in self.url_stack:
            self.logger.error(
                f"Backtrack target {target!r} not in url_stack — falling back to single pop")
            self.url_stack.pop()
        else:
            while len(self.url_stack) > 1 and self.url_stack[-1] != target:
                self.url_stack.pop()

        self.current_url = self.url_stack[-1]
        self.current_depth = len(self.url_stack)  # Derive from stack
        self.logger.debug(
            f"Backtracked to: {self.current_url} (stack depth {self.current_depth})")
        return True

    def _get_parent_url(self):
        """Get url to parent page"""
        return self.url_stack[-2] if len(self.url_stack) > 1 else None

    def act(self, links: List[Tuple[str, str]]) -> Optional[str]:
        """Phase 3: Choose next URL based on accumulated knowledge"""
        if not links and len(self.url_stack) == 1:
            new_url = self._reseed_starting_page()
            if new_url:
                return new_url
            self.logger.error("[ACT] No links at starting_url - exploration exhausted")
            return ""
        if self.current_depth > self.max_depth or not links:
            self._backtrack(); return self.current_url

        next_url, reason = self.web_explorer.select_next_link(links)
        self.logger.info(f"[ACT] Selected link: {next_url}")
        self.logger.info(f"Reason: {reason}")

        # select_next_link can return None / "" on JSON-parse failure — treat as no-link, backtrack
        if not next_url:
            self.logger.error("[ACT] select_next_link returned empty URL — backtracking")
            self._backtrack()
            return self.current_url

        # Picking any in-path URL = retreat to that ancestor. starting_url
        # at url_stack[0] is just the deepest possible target — _backtrack
        # pops the stack down to it. url_stack[:-1] excludes current_url
        # so picking the current page falls through to the normal advance.
        if next_url in self.url_stack[:-1]:
            self._backtrack(target=next_url)
            return self.current_url

        # Pre-set depth so add_edge doesn't leave next_url attribute-less if think() never runs
        if next_url not in self.graph:
            self.graph.add_node(next_url, depth=self.current_depth + 1)
        self.graph.add_edge(self.current_url, next_url, reason=reason)

        traversal_metadata = {
            'iteration': self.current_iteration,
            'depth': self.current_depth,
            'from_url': self.current_url,
            'to_url': next_url,
            'reason': reason,
            'alternatives': len(links),
        }
        self.traversal_history.append(traversal_metadata)

        current_domain = urlparse(self.current_url).netloc
        next_domain = urlparse(next_url).netloc
        self.remember(
            f"Agent navigated from {self.current_url} to visit {next_url} "
            f"(from domain {current_domain} to domain {next_domain}). "
            f"Navigation performed on iteration {self.current_iteration} at depth {self.current_depth}. "
            f"Agent selected new webpage from {len(links)} options because: {reason}",
            context="navigation",
            metadata=traversal_metadata
        )

        self._advance_to_url(next_url)
        return next_url

    def _draw_graph_visualization(self, iteration: int) -> None:
        """Create Graphviz visualization"""
        if not self.draw_graph:
            return

        try:
            import pygraphviz as pgv
        except ImportError:
            return

        try:
            viz = pgv.AGraph(directed=True, strict=False)
            viz.graph_attr.update(rankdir='TB', size='16,12', dpi='150')
            viz.node_attr.update(shape='box', style='rounded,filled',
                               fillcolor='lightblue', fontsize='8')
            viz.edge_attr.update(fontsize='8')

            for node in self.graph.nodes():
                parsed = urlparse(node)
                path_parts = parsed.path.strip('/').split('/')
                label = path_parts[-1] if path_parts and path_parts[-1] else parsed.netloc
                label = label[:SHORT_SUMMARY_LEN] + '...' if len(label) > SHORT_SUMMARY_LEN else label

                insights_preview = self.graph.nodes[node].get('insights', '')[:SUMMARY_LENGTH]
                if insights_preview:
                    label += f"\n{insights_preview}..."

                depth = self.graph.nodes[node].get('depth', 0)
                color = f"0.{min(9, depth)} 0.3 1.0"
                viz.add_node(node, label=label, fillcolor=color)

            for u, v in self.graph.edges():
                reason = self.graph.edges[u, v].get('reason', '')[:SUMMARY_LENGTH] + "..."
                viz.add_edge(u, v, label=reason)

            filepath = os.path.join(self.get_repo(),
                f"{self.get_id()}.graph_iter{iteration}.png")
            viz.draw(filepath, prog='dot')

        except Exception as e:
            self.logger.error(f"Failed to create graph visualization: {e}")

    def _reseed_starting_page(self) -> Optional[str]:
        """When the starting page's link pool is exhausted, regenerate it by
        re-running the same Brave-search setup. Bounded by MAX_RESEEDS
        (independent of max_web_searches). Returns the new starting_url, or
        None if the reseed budget is spent or the search fails."""
        if not self.starting_query:
            return None
        if self.reseeds_used >= MAX_RESEEDS:
            self.logger.error(
                f"[RESEED] Budget exhausted ({self.reseeds_used}/{MAX_RESEEDS})")
            return None

        old_url = self.starting_url
        try:
            self._setup_brave_search(use_cache=False)
        except Exception as e:
            self.logger.error(f"[RESEED] Brave search failed: {e}")
            self.starting_url = old_url
            return None

        if not self.starting_url:
            self.starting_url = old_url
            return None

        self.reseeds_used += 1
        self.url_stack = [self.starting_url]
        self.current_url = self.starting_url
        self.current_depth = len(self.url_stack)
        # New starting_url: seed it the same way _setup_exploration_state seeds the original root
        if self.starting_url not in self.graph:
            self.graph.add_node(self.starting_url, depth=1)
        self.logger.info(
            f"[RESEED] Regenerated starting_url "
            f"({self.reseeds_used}/{MAX_RESEEDS}): {self.starting_url}")
        return self.starting_url

    def _quick_perceive_and_think(self, url: str, iteration: int) -> Optional[Dict[str, Any]]:
        """Fetch content from URL and generate insights. Thread-safe for the
        IO-bound portions (fetch + LLM call). Returns a dict
        {url, insights, iteration} or None on failure."""
        try:
            # Pass the referer explicitly — workers run in parallel and
            # would otherwise race on self._last_fetched_url.
            html = self.web_explorer.fetch_html(url, referer_url=self.starting_url)
            if not html:
                self.failed_urls.add(url)
                return None

            text = self.web_explorer.extract_text_from_html(html)
            if not text:
                self.failed_urls.add(url)
                return None
            if not self.quick_explore_insights:
                return {"url": url, "insights": text, "iteration": iteration}

            # Per-page think prompt — quick_explore workers see only their
            # own page, so there's no graph context to feed in.
            query_task = "- How to answer the query\n" if self.starting_query else ""
            prompt = f"""CONTENT:
{text}

QUERY:
{self.starting_query if self.starting_query else 'No query is available'}

YOUR TASK:
Analyze this content and extract key insights focusing on:
- Novel patterns or unexpected connections
- Assumptions being made and alternative perspectives
- Interesting questions raised by the content
{query_task}
IMPORTANT: Use your role as a guide on how to respond!

Depending on the complexity of the content, provide anywhere from 1 to 6 concise but substantive insights, but do not exceed ~600 words in total length:"""

            insights = self.chat_completion(
                prompt, override_config=self.exploration_llm_config,
                num_retries=0)
            return {"url": url, "insights": insights, "iteration": iteration}

        except FatalLLMError:
            raise  # propagate to the as_completed consumer → quick_explore → job_runner
        except Exception as e:
            self.logger.error(f"[QUICK_EXPLORE] Failed for {url}: {e}")
            self.failed_urls.add(url)
            return None

    def quick_explore(self) -> str:
        """Fast parallel exploration: perceive+think all search result links, skip the act phase."""
        if self.current_iteration > 0 and self.visited_urls:
            self.logger.info("[QUICK_EXPLORE] Checkpoint found — skipping exploration, going straight to synthesis")
            return self.synthesizer.synthesize_artifact()

        self.logger.info("[QUICK_EXPLORE] Starting parallel exploration of search result links")

        # Search-results page is only used for its outbound links —
        # per-page insights come from the worker pool below.
        content, links = self.perceive()
        if not content or not links:
            self.logger.error("[QUICK_EXPLORE] No content or links from starting page")
            return self.synthesizer.synthesize_artifact()

        links_to_explore = [url for url, _ in links
                            if url not in self.visited_urls
                            and url not in self.failed_urls][:self.max_iterations]
        n_links = len(links_to_explore)
        self.logger.info(
            f"[QUICK_EXPLORE] Found {len(links)} links, exploring {n_links} in parallel")

        # Throttle snapshots — save_graph_interval=1 (the config default)
        # would otherwise rewrite the full checkpoint on every completion.
        graph_save_every = max(5, min(20, n_links // 20))
        results: List[Dict[str, Any]] = []
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.quick_explore_workers)
        try:
            futures = {
                executor.submit(self._quick_perceive_and_think, url, i): url
                for i, url in enumerate(links_to_explore, start=1)
                if not self.shutdown_called
            }
            for f in concurrent.futures.as_completed(futures):
                if self.shutdown_called:
                    break
                try:
                    result = f.result()
                except FatalLLMError:
                    raise  # one fatal worker error aborts the whole run
                except Exception as e:
                    self.logger.error(f"[QUICK_EXPLORE] Worker exception for {futures[f]}: {e}")
                    continue
                if not result:
                    continue
                results.append(result)
                url, iteration = result["url"], result["iteration"]
                self.visited_urls[url] = self.visited_urls.get(url, 0) + 1
                # depth=2 since these are one-hop children of the root.
                self.graph.add_node(url, insights=result["insights"], depth=2,
                    iteration=iteration, visit_count=self.visited_urls[url])
                self.graph.add_edge(self.starting_url, url)
                self.current_iteration = max(self.current_iteration, iteration)

                self.logger.info(f"[QUICK_EXPLORE] Completed {len(results)}/{n_links}: {url}")

                # Use the completion count, not `iteration` (the per-URL
                # submission id) — completions land out of order, so
                # filename-monotonic snapshots need a sequential id.
                if len(results) % graph_save_every == 0:
                    self.checkpoint_manager.save(len(results), save_graph_interval=1)
        finally:
            # Cancel pending futures on shutdown rather than draining them.
            executor.shutdown(
                wait=not self.shutdown_called,
                cancel_futures=self.shutdown_called)

        # KB inserts in iteration order — synthesis expects iteration
        # metadata to be monotonic relative to insertion order.
        # Marker lets the web watchdog show progress through this otherwise
        # silent ~minutes-long phase.
        ordered = sorted(results, key=lambda r: r["iteration"])
        self.logger.info(
            f"[QUICK_EXPLORE] KB ingest started ({len(ordered)} results)")
        # Single batched upsert — Chroma's embedding function round-trips
        # the LLM provider once for the whole list instead of per-item, which
        # turns a ~50s phase into ~2s for ~120 pages.
        self.kb_manager.add_texts(
            [(r["insights"],
              {"url": r["url"], "depth": 2, "iteration": r["iteration"]})
             for r in ordered]
        )
        # Emit per-item lines after the batch returns so the watchdog regex
        # (`Added insights (N length) for URL`) keeps incrementing the
        # kb_ingest progress counter the UI is bound to.
        for r in ordered:
            self.logger.info(
                f"[QUICK_EXPLORE] Added insights ({len(r['insights'])} length) for {r['url']} to knowledge base")

        # Terminal snapshot. save_graph_interval=1 bypasses save()'s modulo
        # gate so the final write always lands.
        self.checkpoint_manager.save(self.current_iteration, save_graph_interval=1)
        self.logger.info(
            f"[QUICK_EXPLORE] Complete: explored {len(results)}/{n_links} links")
        return self.synthesizer.synthesize_artifact()

    def explore(self) -> str:
        """Execute main exploration loop"""
        if self.use_quick_explore:
            return self.quick_explore()

        start_time = time.time()
        start_iter = self.current_iteration + 1; end_iter = self.max_iterations + 1
        if start_iter < end_iter:
            self.logger.info(f"[EXPLORE] Beginning exploration: iterations {start_iter} to {self.max_iterations}")

        # TODO: Replace Perceive-Think-Act with LangGraph nodes
        for iteration in range(start_iter, end_iter):
            if self.shutdown_called: break
            self.current_iteration = iteration

            self._log_iteration(iteration, start_time, start_iter)

            content, links = self.perceive()
            if not content:
                self._backtrack()
                if iteration % self.checkpoint_interval == 0:
                    self.checkpoint_manager.save(iteration)
                continue

            insights = self.think(content)

            if iteration < self.max_iterations:
                next_url = self.act(links)
                if not next_url:
                    self.logger.error("[EXPLORE] No links to explore, exiting loop early")
                    break

            if iteration % self.save_graph_interval == 0 or iteration == self.max_iterations:
                self._draw_graph_visualization(iteration)

            if iteration % self.checkpoint_interval == 0 or iteration == self.max_iterations:
                self.checkpoint_manager.save(iteration)

        self._draw_graph_visualization(self.current_iteration)
        self.checkpoint_manager.save(self.current_iteration)

        self.logger.info(f"\n[EXPLORE] Exploration complete: visited {len(self.visited_urls)} pages")
        return self.synthesizer.synthesize_artifact()

    def shutdown(self) -> None:
        """Clean up CaesarAgent resources with immediate flag setting"""
        if self.shutdown_called:
            return

        # Set flags IMMEDIATELY to stop loops
        self.shutdown_called = True

        # Save final checkpoint with session cost
        if hasattr(self, 'current_iteration') and hasattr(self, 'llm_handler') and self.url_stack:
            self.checkpoint_manager.save(self.current_iteration)
            self.logger.info("Final checkpoint saved on shutdown")

        # Cleanup in reverse order of initialization
        if hasattr(self, 'kb_manager'):
            self.kb_manager.shutdown()

        # Call parent shutdown
        super().shutdown()

        self.logger.info("CaesarAgent shutdown completed")