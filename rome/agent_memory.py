"""Agent Memory - Intelligent memory layer using Mem0 with ChromaDB + Neo4j backend"""
import os
import time
import traceback
from typing import Dict, Optional

# mem0 enables telemetry by default and creates a Chroma collection named
# "mem0migrations". Keep Caesar private-by-default unless the caller opts in.
os.environ.setdefault("MEM0_TELEMETRY", "false")

from mem0 import Memory

from .config import set_attributes_from_config, DEFAULT_CONFIG, LONGER_SUMMARY_LEN
from .logger import get_logger
from .kb_server import ChromaServerManager, CHROMA_BASE_DIR
from .kb_client import EMBEDDING_MODELS

OPENAI_CONFIG =  {
    # Reasoning effort not supported in mem0ai v1.0.4
    # "model": "gpt-5-mini",
    # "reasoning_effort": "low",
    "model": "gpt-4o",
    "temperature": 0.1,
    "max_tokens": DEFAULT_CONFIG['LLMHandler']['max_completion_tokens'],
    # api_key is read lazily in _initialize_mem0 so it picks up the
    # per-run OPENAI_API_KEY value rather than whatever was set at import
    # time (which may be unset on a bring-your-own-key server).
    "openai_base_url": DEFAULT_CONFIG['LLMHandler']['base_url'],
}


class AgentMemory:
    """Intelligent memory layer for agents using Mem0 with optional graph storage"""

    def __init__(self, agent, config: Dict = None):
        """Initialize agent memory with vector +/- graph storage"""
        self.agent = agent
        self.mem_id = f"{os.path.basename(agent.get_repo())}_{agent.get_id()}"
        self.logger = get_logger()

        # Apply config (defaults come from DEFAULT_CONFIG)
        set_attributes_from_config(
            self, config, DEFAULT_CONFIG['AgentMemory'].keys(), ['openai_config']
        )

        # Resolved here, not in DEFAULT_CONFIG, so the password never enters self.config
        # and cannot reach an exported or logged config at all -- rather than relying on
        # base_agent.export_config's redaction to catch it on the way out. Reading it at
        # construction (not import) also means a later os.environ change is picked up.
        # An explicit config value still wins.
        self.graph_password = self.graph_password or os.environ.get("NEO4J_PASSWORD")

        self.memory = None
        if self.enabled:
            self._initialize_mem0()

            # Handle interactive memory clearing
            if self.clear_memory: self._prompt_clear_memory()

    def _prompt_clear_memory(self) -> None:
        """Interactive prompt to ask user if they want to clear memory"""
        results = self.memory.get_all(filters={"user_id": self.mem_id}, top_k=10000)
        memory_count = len(results.get('results', [])) if results else 0
        if self.use_graph:
            result = self.memory.graph.graph.query(
                "MATCH (n {user_id: $user_id}) RETURN count(n) as total",
                {"user_id": self.mem_id}
            )
            memory_count += result[0]['total']
            # self.logger.info(f"Found {result[0]['total']} graph nodes/relations")
        if memory_count == 0:
            self.logger.info("No existing memories, skipping clear"); return

        # Display status
        sep = '=' * 80
        self.logger.info(f"\n{sep}\nAgent Memory Status for: {self.mem_id}\n{sep}")
        self.logger.info(f"Found {memory_count} existing memories\n{sep}")

        # Prompt and clear
        response = input(f"Clear all existing memories? [y/N]: ").strip().lower()
        if response in ('y', 'yes'):
            self.logger.info("Clearing memories...")
            self.logger.info("✓ Memories cleared successfully\n" if self.clear() else "✗ Failed to clear memories\n")
        else:
            self.logger.info("Keeping existing memories")

    # def _test_neo4j_connection(self) -> bool:
    #     """Test if Neo4j is accessible"""
    #     if not self.use_graph or not self.memory:
    #         return False

    #     try:
    #         # Try to access the graph store directly
    #         if hasattr(self.memory, 'graph'):
    #             # Run a simple test query
    #             test_query = "MATCH (n) RETURN count(n) as total LIMIT 1"
    #             result = self.memory.graph.graph.query(test_query)
    #             self.logger.info(f"Neo4j connection test successful: {result}")
    #             return True
    #     except Exception as e:
    #         self.logger.error(f"Neo4j connection test failed: {e}")
    #         self.logger.error(traceback.format_exc())
    #         return False

    def _initialize_mem0(self) -> None:
        """Initialize Mem0 with ChromaDB vector store and optional Neo4j graph store"""
        try:
            # Validate embedding model
            if self.embedding_model not in EMBEDDING_MODELS:
                raise RuntimeError(f"Invalid embedding model {self.embedding_model}")

            if "OPENROUTER_API_KEY" in os.environ:
                self.logger.debug("Removing OPENROUTER_API_KEY to prevent auto-routing")
                del os.environ["OPENROUTER_API_KEY"]

            # Build base config. Read the api_key lazily here (rather than at
            # module import) so it reflects the OPENAI_API_KEY active for this
            # run (e.g. a per-run bring-your-own-key env window).
            base_openai_config = {**OPENAI_CONFIG, "api_key": os.environ.get("OPENAI_API_KEY")}
            openai_config = base_openai_config | self.openai_config if self.openai_config else base_openai_config
            llm_config = {
                "provider": "openai",
                "config": openai_config
            }
            mem0_config = {
                "version": "v1.1",
                "llm": llm_config
            }

            # Embedder configuration with explicit dimensions
            embedder_config = {
                "provider": "openai",
                "config": {
                    "model": self.embedding_model,
                    "embedding_dims": EMBEDDING_MODELS[self.embedding_model]
                }
            }
            mem0_config["embedder"] = embedder_config

            # ALWAYS configure vector store (required for both modes)
            server_config = {
                'host': self.vector_host,
                'port': self.vector_port,
                'persist_path': None
            }

            server_manager = ChromaServerManager.get_instance(config=server_config)
            if not server_manager.is_running() and not server_manager.start():
                raise RuntimeError("Failed to start ChromaDB server")

            collection_name = f"mem0_{self.mem_id}"
            mem0_config["vector_store"] = {
                "provider": "chroma",
                "config": {
                    "collection_name": collection_name,
                    "host": server_manager.host,
                    "port": server_manager.port
                }
            }

            # Optionally ADD graph store (works alongside vector store)
            if self.use_graph:
                if not self.graph_password:
                    self.logger.error("Graph enabled but no password provided")
                    return

                mem0_config["graph_store"] = {
                    "provider": "neo4j",
                    "config": {
                        "url": self.graph_url,
                        "username": self.graph_username,
                        "password": self.graph_password,
                        "llm": llm_config
                    }
                }
                self.logger.info(f"Mem0 initialized: ChromaDB @ {server_manager.server_url} + Neo4j @ {self.graph_url}")
            else:
                self.logger.info(f"Mem0 initialized: ChromaDB @ {server_manager.server_url} (No Neo4j)")

            # Initialize memory with the complete configuration
            self.memory = Memory.from_config(config_dict=mem0_config)

        except Exception as e:
            self.logger.error(f"Memory initialization failed: {e}")
            self.logger.error(traceback.format_exc())
            self.enabled = False; self.memory = None

    def is_enabled(self) -> bool:
        """Check if memory is enabled and initialized"""
        return self.enabled and self.memory is not None

    def remember(self, content: str, context: str = None, metadata: Dict = None) -> bool:
        """Store content in memory (vector + graph if enabled)"""
        if not self.is_enabled():
            self.logger.error("Memory not enabled, skipping remember")
            return False

        try:
            # Sanitize content to avoid hyphens in potential relationship names
            # content = content.replace('-', '_')

            # Make content more "memorable" by adding personal/experiential context
            message = f"{context}: {content}" if context else content

            meta = metadata or {}
            if context: meta['context'] = context
            meta['timestamp'] = time.time()

            # Structure as conversation for entity extraction
            messages = [{"role": "user", "content": message}]
            result = self.memory.add(
                messages,
                user_id=self.mem_id,
                metadata=meta,
                infer=self.infer,
            )

            if not result:
                self.logger.error("No result returned from memory.add()")
                return False

            results = result.get('results', [])
            relations = result.get('relations', [])

            if results:
                g_str = f" | Graph DB: {len(relations)} new relations" if self.use_graph else ""
                self.logger.info(f"Vector DB: {len(results)} new memories{g_str}")
                self.logger.debug(f"Remembered memories/relations: {result}")
                return True
            else:
                self.logger.error("No memories or graph relations extracted")
                return False

        except Exception as e:
            self.logger.error(f"Failed to remember: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def recall(self, query: str, context: str = None) -> str:
        """
        Query memory and return most relevant results
        Uses vector similarity + graph traversal (if enabled)
        """
        if not self.is_enabled():
            self.logger.error("Memory not enabled, skipping recall")
            return ""

        try:
            # mem0ai 2.0+ removed top-level entity kwargs (user_id, agent_id,
            # run_id) on search/get_all — they must go inside `filters=`. The
            # `limit` kwarg was likewise renamed to `top_k`. Same applies to
            # the get_all calls in _prompt_clear_memory, summarize_memory, and
            # clear() below.
            search_filters = {"user_id": self.mem_id}
            if context:
                search_filters["context"] = context
            results = self.memory.search(
                query=query,
                top_k=self.recall_limit,
                filters=search_filters,
            )

            if results and results.get('results'):
                recalled_mem =  '\n\n'.join(r['memory'] for r in results['results'])
                self.logger.debug(f"Recalled: {recalled_mem}...")
                return recalled_mem
            else:
                self.logger.error(f"No memories recalled, results empty")
                return ""

        except Exception as e:
            self.logger.error(f"Failed to recall: {e}")
            self.logger.error(traceback.format_exc())
            return ""

    def summarize_past(self, total_items: int, recent_ratio: float) -> str:
        """
        Combines a dense buffer of recent memories with equidistant samples from distant history
        Args:
            total_items: Total number of memories to retrieve for the context window
            recent_ratio: Fraction defining the fraction of memories to the dense buffer
        """
        if not self.is_enabled():
            return "Memory is disabled."

        try:
            # 1. Fetch a large pool to ensure we have deep history to sample from
            # (Fetching 10x the needed amount ensures we catch the "beginning of time")
            fetch_pool_size = max(total_items * 10, 2000)
            results = self.memory.get_all(
                filters={"user_id": self.mem_id}, top_k=fetch_pool_size)

            if not results or not results.get('results'):
                return "No memories available to summarize."

            all_memories = results['results']

            # 2. Sort all by timestamp (Newest -> Oldest)
            # We use 0 as fallback if timestamp is missing
            all_memories.sort(
                key=lambda x: x.get('metadata', {}).get('timestamp', 0),
                reverse=True
            )

            # 3. Calculate Slice Sizes
            num_recent = int(total_items * recent_ratio)
            num_distant = total_items - num_recent

            # 4. Dense Sampling: Take the most recent block as-is
            selected_memories = all_memories[:num_recent]

            # 5. Sparse Sampling: Stride through the remaining history
            history_pool = all_memories[num_recent:]

            if history_pool and num_distant > 0:
                if len(history_pool) <= num_distant:
                    # If history is small, just take it all
                    selected_memories.extend(history_pool)
                else:
                    # Calculate stride (step size) to sample evenly
                    step = len(history_pool) / num_distant
                    for i in range(num_distant):
                        idx = int(i * step)
                        selected_memories.append(history_pool[idx])

            # 6. Re-sort the final selection for the LLM
            # (Newest -> Oldest is maintained to keep the prompt consistent)
            selected_memories.sort(
                key=lambda x: x.get('metadata', {}).get('timestamp', 0),
                reverse=True
            )

            # 7. Format Context
            context_items = []
            for m in selected_memories:
                text = m.get('memory', '').strip()
                ctx_type = m.get('metadata', {}).get('context', 'general')
                # Optional: You could add " (Old)" or " (Recent)" tags here if helpful
                context_items.append(f"- [{ctx_type}] {text}")

            context_block = "\n".join(context_items)

            # 8. Generate Summary
            prompt = f"""Analyze the following sparse memory log (containing both dense recent events and sparse historical samples) and generate a "State of the Exploration" summary.

MEMORY LOG (Sorted Newest to Oldest):
{context_block}

INSTRUCTIONS:
- Construct a cohesive narrative of the agent's journey.
- Connect recent actions (top of list) to long-term patterns, original goals, or old failures (sparse items from bottom).
- Explicitly check if the agent is repeating a strategy attempted long ago.

SUMMARY:"""

            return self.agent.chat_completion(prompt=prompt)
                # system_message="You are an analytical observer summarizing an agent's memory stream.",
                # override_config={"reasoning_effort": "medium"}


        except Exception as e:
            self.logger.error(f"Failed to summarize memory: {e}")
            self.logger.error(traceback.format_exc())
            return "Error generating memory summary."

    def should_remember(self, prompt: str, response: str) -> bool:
        """Heuristic check if interaction should be remembered"""
        if len(prompt) + len(response) < self.auto_remember_len:
            return False

        memorable_patterns = [
            'decided', 'chose', 'learned', 'discovered', 'found', 'realized',
            'prefer', 'strategy', 'failed', 'succeeded', 'worked', "didn't work",
            'important', 'critical', 'key', 'pattern', 'should', 'avoid',
            'remember', 'note', 'always', 'never', 'visited', 'explored',
            'analyzed', 'edited', 'file', 'directory'
        ]

        combined = (prompt + ' ' + response).lower()
        return any(pattern in combined for pattern in memorable_patterns)

    def chat_completion(self, prompt: str, system_message: str = None,
                       override_config: Dict = None, response_format: Dict = None,
                       num_retries: Optional[int] = None) -> str:
        """Enhanced chat completion with automatic memory integration"""
        system_message = system_message or self.agent.role
        # Get memory context if enabled
        if self.is_enabled() and self.auto_recall:
            # Use prompt as query for recall
            memory_context = self.recall(prompt[:LONGEST_SUMMARY_LEN])

            if memory_context:
                # Inject memory into system message
                system_message = f"{system_message}\n\n[Relevant Memory Context]\n{memory_context}"
                self.logger.debug(f"Injected memory: {len(memory_context)} chars")

        # Call original chat completion
        response = self.agent.llm_handler.chat_completion(
            prompt=prompt,
            system_message=system_message,
            override_config=override_config,
            response_format=response_format,
            num_retries=num_retries,
        )

        # Auto-remember after getting response
        if self.is_enabled() and self.auto_remember:
            if self.should_remember(prompt, response):
                summary = \
                    f"Q: {prompt[:LONGEST_SUMMARY_LEN]}... A: {response[:LONGEST_SUMMARY_LEN]}..."
                self.remember(summary, context="interaction")

        return response

    def clear(self) -> bool:
        """Clear all memories from vector store and Neo4j graph"""
        if not self.is_enabled():
            return False

        try:
            # Delete from both vector and graph stores
            self.memory.delete_all(user_id=self.mem_id)

            # Verify vector store cleanup
            results = self.memory.get_all(
                filters={"user_id": self.mem_id}, top_k=10000)
            vector_count = len(results.get('results', [])) if results else 0

            # Verify Neo4j cleanup if graph enabled
            graph_count = 0
            if self.use_graph and hasattr(self.memory, 'graph'):
                query = "MATCH (n {user_id: $user_id}) RETURN count(n) as total"
                result = self.memory.graph.graph.query(query, {"user_id": self.mem_id})
                graph_count = result[0]['total'] if result else 0

            if vector_count == 0 and graph_count == 0:
                self.logger.info(f"Cleared all memories for {self.mem_id} for both vector/graph")
                return True
            else:
                self.logger.error(f"Clear incomplete: vector={vector_count}, graph={graph_count} memories remain")
                return False

        except Exception as e:
            self.logger.error(f"Failed to clear memories: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def shutdown(self) -> None:
        """Clean up memory resources"""
        if self.memory:
            # self.clear()
            self.logger.debug(f"Memory resources released for {self.mem_id}")
