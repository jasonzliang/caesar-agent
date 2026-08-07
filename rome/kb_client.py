# knowledge_base.py
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from typing import Optional, List, Dict
import os
from pathlib import Path
import re
import traceback
import time
import sys
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore', message='.*validate_default.*', category=UserWarning)

from .llm_handler import FatalLLMError

try:
    from chromadb.utils.embedding_functions import (
       SentenceTransformerEmbeddingFunction, OpenAIEmbeddingFunction)
    from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.postprocessor.llm_rerank import LLMRerank
    from llama_index.core.response_synthesizers import get_response_synthesizer
    from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI
    from llama_index.vector_stores.chroma import ChromaVectorStore
    # Needed for OpenAIEmbeddingFunction. Guarded so importing this module
    # without OPENAI_API_KEY set doesn't crash before the CLI can emit its
    # own friendly error.
    if 'OPENAI_API_KEY' in os.environ:
        os.environ['CHROMA_OPENAI_API_KEY'] = os.environ['OPENAI_API_KEY']
except ImportError as e:
    # Raise (don't exit(1)): this module is imported lazily by the web server's
    # run worker, so a hard process exit would crash the whole server on a
    # missing optional dep. Raising fails just the calling run instead. `e`
    # already names the missing module; point at requirements.txt for the fix.
    raise ImportError(
        f"Caesar knowledge-base dependencies are not installed ({e}). "
        "Install them with: pip install -r requirements.txt"
    ) from e

import openai

from .config import set_attributes_from_config, DEFAULT_CONFIG
from .logger import get_logger
from .kb_server import ChromaServerManager
from .parsing import parse_json_response

# Embedding model configurations
EMBEDDING_MODELS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "text-embedding-ada-001": 1024,
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "paraphrase-MiniLM-L6-v2": 384
}

# OpenAI's `text-embedding-3-*` endpoint caps each request at 300k tokens
# aggregated across all `input` items, AND each individual item at 8192
# tokens. ChromaDB's OpenAIEmbeddingFunction forwards the whole `input`
# list as one request and chromadb's `create_batches` only splits by item
# count (SQLite param limit) — neither layer enforces either OpenAI cap.
# We pre-filter oversize items and sub-batch by aggregate token count.
#
# Budget set to 100k (33% of OpenAI's 300k cap) because tiktoken's count
# can underestimate the actual server-side count by 25-30% on real web
# content (markup residue, mixed unicode, etc.) — saw a 250k budget
# blow past 327k in production. 100k absorbs up to a 3x undercount.
_EMBED_BATCH_TOKEN_BUDGET = 100_000
_EMBED_PER_INPUT_TOKEN_CAP = 8000  # margin under OpenAI's 8192
_OPENAI_TOKENIZER = None  # lazy: tiktoken is a chromadb dep but loading the
                          # encoding dict allocates ~MB of memory.


def _get_openai_tokenizer():
    global _OPENAI_TOKENIZER
    if _OPENAI_TOKENIZER is None:
        import tiktoken
        _OPENAI_TOKENIZER = tiktoken.get_encoding("cl100k_base")
    return _OPENAI_TOKENIZER


def _expand_oversize(items, splitter, per_input_cap=_EMBED_PER_INPUT_TOKEN_CAP):
    """For each (text, metadata) item, if the text exceeds the per-input
    token cap, split it via `splitter` into chunks and emit one
    (chunk_text, chunk_metadata) per chunk — chunk_idx / chunk_total are
    written into the metadata so retrieval can reassemble passages.
    Items within the cap pass through unchanged. Returns the expanded
    list AND a per-item token count list aligned to it.

    Without this expansion an oversize doc would 400 the whole batch
    (per-input cap = 8192 tokens), losing both itself and its smaller
    batch-mates."""
    enc = _get_openai_tokenizer()
    out_items = []
    out_counts = []
    for text, metadata in items:
        n = len(enc.encode(text, disallowed_special=()))
        if n <= per_input_cap or splitter is None:
            out_items.append((text, metadata))
            out_counts.append(n)
            continue
        chunks = splitter.split_text(text)
        for idx, chunk in enumerate(chunks):
            chunk_meta = dict(metadata or {})
            chunk_meta["chunk_idx"] = idx
            chunk_meta["chunk_total"] = len(chunks)
            out_items.append((chunk, chunk_meta))
            out_counts.append(len(enc.encode(chunk, disallowed_special=())))
    return out_items, out_counts


def _token_batched_indices(token_counts, budget=_EMBED_BATCH_TOKEN_BUDGET):
    """Yield (start, end) index slices over `token_counts` so that the
    cumulative token count of items in each slice stays within `budget`.
    A single item over the budget is emitted as its own slice."""
    start = 0
    running = 0
    for i, n in enumerate(token_counts):
        if i > start and running + n > budget:
            yield start, i
            start = i
            running = 0
        running += n
    if start < len(token_counts):
        yield start, len(token_counts)

# Models that accept the `reasoning_effort` parameter. Bare identifiers only —
# dated suffix variants (e.g. "gpt-5.4-2025-12-01") need to be added explicitly
# or the parameter will be silently dropped at LLM construction time.
# Verified against OpenAI's API docs (developers.openai.com/api/docs/models).
REASONING_MODELS = {
    # GPT-5 series
    "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro",
    "gpt-5.1", "gpt-5.1-codex-max",
    "gpt-5.2", "gpt-5.2-pro",
    "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4-pro",
    "gpt-5.5",
    # o-series
    "o1", "o1-mini", "o1-pro",
    "o3", "o3-mini", "o3-pro",
    "o4-mini",
}


class ChromaClientManager:
    """Enhanced ChromaDB + LlamaIndex knowledge base with reranking"""

    def __init__(self, agent):
        # `agent` is required — every config lookup, log dir, role injection,
        # and chat_completion call below assumes it exists. The previous
        # `agent=None` default was misleading: line 138 crashed before any
        # of the dead "if self.agent" fallbacks could run.
        self.agent = agent
        self.config = agent.config.get('ChromaClientManager', {}) or {}
        self.server_config = agent.config.get('ChromaServerManager', {}) or {}
        self.logger = get_logger()

        # Set attributes from config
        set_attributes_from_config(self, self.config,
            DEFAULT_CONFIG['ChromaClientManager'].keys())

        self.logger.assert_true(self.embedding_model in EMBEDDING_MODELS,
            f"Invalid embedding model: {self.embedding_model}")

        # Get or create server manager instance
        if self.use_shared_server:
            # Use shared singleton server
            self.server = ChromaServerManager.get_instance(self.server_config)
            self._owns_server = False
        else:
            # Create dedicated server instance
            self.server = ChromaServerManager(self.server_config)
            self._owns_server = True

        if not self.server.is_running():
            self.logger.info("ChromaDB server not running, starting it...")
            if not self.server.start():
                raise RuntimeError(f"Failed to start ChromaDB server at {self.server.server_url}")

        # Initialize ChromaDB client and collection
        self._setup_chroma_client()

        # Initialize LlamaIndex components
        self._setup_llamaindex()

        # Initialize reranker if enabled
        self._setup_reranker()

        self.logger.info(f"ChromaClientManager initialized: collection={self.collection_name}, reranking={self.enable_reranking}")

    def _path_to_collection_name(self, file_path: str, max_len: int = 128) -> str:
        """Convert file path to valid Chroma collection name.

        Chroma names must be 3-63 chars; we cap at max_len. Short or empty
        stems get prefixed with `doc_` (4 chars) to satisfy the minimum.
        """
        name = re.sub(r'[^a-z0-9._-]', '_', Path(file_path).stem.lower())
        name = re.sub(r'(^[^a-z0-9]+|[^a-z0-9]+$|_{2,}|\.{2,})', '_', name).strip('_')
        return (name if len(name) >= 3 else f"doc_{name}")[:max_len]

    def _validate_dimensions(self, expected_dim):
        """Validate collection embedding dimensions"""
        if self.collection.count() == 0:
            return

        result = self.collection.get(limit=1, include=["embeddings"])

        # More defensive approach - handle various array types
        embeddings = result.get("embeddings")
        # Check for None first
        if embeddings is None:
            return

        # Convert to list if it's a numpy array or similar
        if hasattr(embeddings, 'tolist'):
            embeddings = embeddings.tolist()

        # Now safely check length and content
        if len(embeddings) > 0 and embeddings[0] is not None:
            actual_dim = len(embeddings[0])
            if actual_dim != expected_dim:
                compatible = [m for m, d in EMBEDDING_MODELS.items() if d == actual_dim]
                raise ValueError(
                    f"Dimension mismatch: collection={actual_dim}d, model={expected_dim}d. "
                    f"Compatible models: {compatible} or clear collection."
                )
            else:
                self.logger.debug(f"Collection/model embedding dimension validated: {actual_dim}")

    def _create_collection(self):
        """Create collection with appropriate embedding function"""
        expected_dim = EMBEDDING_MODELS[self.embedding_model]
        is_sentence_transformer = expected_dim == 384

        if is_sentence_transformer:
            embedding_fn = SentenceTransformerEmbeddingFunction(model_name=self.embedding_model)
        else:
            if not os.getenv('OPENAI_API_KEY'):
                raise ValueError(f"OPENAI_API_KEY required for {self.embedding_model}")
            embedding_fn = OpenAIEmbeddingFunction(model_name=self.embedding_model)

        try:
            self.collection = self.client.create_collection(
                name=self.collection_name, embedding_function=embedding_fn)
            self.logger.debug(f"Created new collection: {self.collection_name} ({self.embedding_model} | {expected_dim}d)")

        except Exception as e:
            if "already exists" in str(e).lower():
                self.collection = self.client.get_collection(
                    name=self.collection_name, embedding_function=embedding_fn)
                self.logger.debug(f"Using existing collection: {self.collection_name} ({self.embedding_model} | {expected_dim}d | {self.collection.count()}n)")
            else: raise

        self._validate_dimensions(expected_dim)

    def _setup_chroma_client(self):
        """Setup ChromaDB client and collection with validation"""
        # Validate model and get expected dimensions
        if not self.embedding_model or self.embedding_model not in EMBEDDING_MODELS:
            models = list(EMBEDDING_MODELS.keys())
            raise ValueError(f"Invalid embedding_model '{self.embedding_model}'. Supported: {models}")

        if not self.collection_name:
            self.collection_name = self._path_to_collection_name(self.agent.repository)

        self.client = self.server.get_client()

        # Chromadb's HttpClient hardcodes httpx Timeout(timeout=None) on its
        # internal session (chromadb/api/fastapi.py, both branches). With no
        # timeout, a stalled chroma upsert handler hangs the client forever
        # (saw this consistently after ~6 small upserts, even on a fresh
        # 2-collection store). Override the internal session to give every
        # client→server call a finite read budget.
        try:
            import httpx as _httpx
            self.client._server._session.timeout = _httpx.Timeout(
                connect=5.0, read=300.0, write=300.0, pool=10.0)
        except (AttributeError, ImportError) as e:
            self.logger.error(
                f"Could not override chromadb httpx timeout (private API "
                f"may have changed): {e}")

        self._create_collection()

    def _setup_llamaindex(self):
        """Setup LlamaIndex components with instance isolation"""
        self.llm = OpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=DEFAULT_CONFIG['LLMHandler']['max_completion_tokens'],
            additional_kwargs={"reasoning_effort": self.reasoning_effort}
                if (self.reasoning_effort and self.model in REASONING_MODELS) else None,
        )
        self.embed_model = OpenAIEmbedding(model=self.embedding_model)
        self.logger.debug(
            f"LlamaIndex OpenAI config: model={self.model}, "
            f"embed={self.embedding_model}")

        # Configure chunking if specified
        if self.chunk_size and self.chunk_overlap is not None:
            self.node_parser = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

            self.logger.debug(f"Configured chunking: size={self.chunk_size}, overlap={self.chunk_overlap}")

        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # Create index with explicit embed model to avoid global setting conflicts
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=self.storage_context,
            embed_model=self.embed_model  # Use instance-specific embed model
        )

    def _setup_reranker(self):
        """Setup LLMRerank if enabled"""
        if self.enable_reranking:
            self.reranker = LLMRerank(
                # choice_batch_size=self.rerank_batch_size,
                top_n=self.rerank_top_n,
                llm=self.llm
            )
            self.response_synthesizer = get_response_synthesizer(
                response_mode="compact",
                llm=self.llm,
                # streaming=False,
                # use_async=False
            )
            self.logger.debug("LLMRerank enabled")
        else:
            self.reranker = None
            self.response_synthesizer = None
            self.logger.debug("Reranking disabled")

    def size(self):
        return self.collection.count()

    def add_text(self, text, metadata=None):
        """Add a single text document with automatic deduplication.

        Delegates to `add_texts` so a >8192-token doc gets chunked via the
        configured SentenceSplitter, the new/updated split comes from the
        same bulk-get pattern as the batch path, and empty inputs are
        skipped. Returns True if a row was written, False if input was empty.
        """
        if not text:
            return False
        return self.add_texts([(text, metadata)]) > 0

    def add_texts(self, items):
        """Batch upsert. `items` is a list of (text, metadata) tuples.

        Pages that exceed OpenAI's 8192-tokens-per-input cap are split into
        sentence-sized chunks (via the configured SentenceSplitter) and
        stored as multiple documents with chunk_idx/chunk_total metadata,
        so retrieval can still reach long sources. The aggregate batch is
        then split into sub-batches under the 300k-tokens-per-request cap.
        Each sub-batch is one `collection.upsert(...)` call.

        Empty / duplicate-hash items are deduped before upsert.
        """
        if not items:
            return 0

        # Expand oversize docs into chunks before deduping — different
        # chunks of the same page have different content hashes, so they
        # don't collide.
        splitter = getattr(self, "node_parser", None)
        expanded, expanded_counts = _expand_oversize(items, splitter)
        n_expanded = len(expanded) - len(items)

        seen: set[str] = set()
        ids: list[str] = []
        texts: list[str] = []
        metas: list[dict] = []
        token_counts: list[int] = []
        for (text, metadata), n in zip(expanded, expanded_counts):
            if not text:
                continue
            h = hashlib.sha256(text.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            ids.append(h)
            texts.append(text)
            metas.append(metadata or {})
            token_counts.append(n)
        if not ids:
            return 0

        # Snapshot existing ids only when needed (log_db on, or to count
        # how many are truly new). One bulk get is cheaper than N counts.
        # `None` sentinel = the snapshot is UNKNOWN (not "empty"); downstream
        # callers must not mistake "we don't know" for "all are new".
        existing_ids: Optional[set[str]] = set()
        try:
            res = self.collection.get(ids=ids)
            existing_ids = set(res.get("ids", []) or [])
        except Exception as e:
            self.logger.error(
                f"Pre-upsert get() failed; new/updated split and log_db "
                f"new-only filter are unavailable for this batch: {e}")
            existing_ids = None

        # Sub-batch by aggregate token count to stay under OpenAI's 300k-
        # tokens-per-request cap. Budget is set conservatively (33% of
        # the cap) since tiktoken's count can underestimate the actual
        # server-side count by 25-30% on real web content.
        n_batches = 0
        for s, e in _token_batched_indices(token_counts):
            self.collection.upsert(
                ids=ids[s:e], documents=texts[s:e], metadatas=metas[s:e])
            n_batches += 1

        if existing_ids is None:
            split_msg = "new/updated split unknown — pre-upsert get() failed"
        else:
            new_count = sum(1 for h in ids if h not in existing_ids)
            split_msg = f"{new_count} new, {len(ids) - new_count} updated"
        self.logger.info(
            f"Batch upserted {len(ids)} document(s) in {n_batches} "
            f"request(s) ({split_msg}"
            + (f", +{n_expanded} chunks from oversize docs" if n_expanded else "")
            + ")"
        )

        if self.log_db and existing_ids is not None:
            # When the snapshot failed, skip the doc log rather than re-log
            # every doc (including pre-existing ones) as if they were new.
            new_triples = [
                (h, t, m) for h, t, m in zip(ids, texts, metas)
                if h not in existing_ids
            ]
            if new_triples:
                with open(os.path.join(self.agent.get_log_dir(),
                    self.agent.get_id() + ".db-doc.log"), "a") as f:
                    for h, t, m in new_triples:
                        f.write("=" * 80)
                        f.write(f"\n\nDOCUMENT: {t}\n\nMETADATA: {m}\n\n")

        return len(ids)

    def query(self, question, top_k=None, top_n=None, return_sources=False, filters=None):
        """Enhanced query with optional source URLs"""
        try:
            if self.size() == 0:
                return ("", []) if return_sources else ""

            should_rerank = self.reranker is not None
            top_k = max([top_k or self.rerank_top_k, self.rerank_top_n, 1]) if should_rerank else max(top_k or self.top_k, 1)

            nodes = self._retrieve_nodes(question, top_k, filters)

            if should_rerank:
                # Single top_n value used for both diversification and rerank,
                # floored at rerank_top_n so the rerank pool stays viable.
                top_n = max(top_n or self.rerank_top_n, self.rerank_top_n, 1)
                # Per-URL cap on the rerank pool — needed because LLMRerank
                # scores in independent batches and cannot itself recognize
                # one source dominating the pool. Disabled on the no-rerank
                # path (which does its own retrieval inside the query engine).
                nodes = self._cap_per_source(nodes, max(1, top_n // 3))
                response, nodes = self._rerank_and_respond(question, nodes, top_n)
            else:
                response, nodes = self._standard_query(question, top_k, filters)

            if self.log_db:
                with open(os.path.join(self.agent.get_log_dir(),
                    self.agent.get_id() + ".db-query.log"), "a") as f:
                    f.write("="*80)
                    f.write(f"\n\nQUERY: {question}\n\nRESPONSE: {response}\n\n")

            if return_sources:
                sources = [
                    {'url': m['url'], 'depth': m.get('depth'), 'iteration': m.get('iteration')}
                    for node in nodes
                    if (m := getattr(node.node if hasattr(node, 'node') else node, 'metadata', {}))
                    and 'url' in m
                ]
                return response, sources

            return response

        except FatalLLMError:
            # Auth / quota / cost-limit must bubble to the agent, never
            # silently degrade to empty results on a fatal LLM condition.
            raise
        except openai.AuthenticationError as e:
            # llama_index's embeddings call goes through the openai SDK
            # directly (not LLMHandler), so auth errors surface as openai.*
            # exceptions instead of FatalLLMError. Reclassify so the run
            # surfaces the actual root cause instead of degrading to empty
            # results and a confusing downstream "No synthesis artifacts" error.
            self.logger.error(f"KB embedding auth error (fatal): {e}")
            raise FatalLLMError(f"Authentication failed: {e}") from e
        except openai.RateLimitError as e:
            # 429 covers both rate-limit and insufficient_quota. Quota
            # exhaustion is fatal (retry won't help); rate-limit is transient
            # (llama_index already retries internally via tenacity, so by the
            # time we see it the retries are exhausted, but the next iteration
            # may succeed).
            code = getattr(e, 'code', None) or ''
            msg = str(e).lower()
            if (code == 'insufficient_quota' or 'insufficient_quota' in msg
                    or 'billing' in msg):
                self.logger.error(f"KB embedding quota exhausted (fatal): {e}")
                raise FatalLLMError(
                    f"Insufficient quota / billing issue: {e}") from e
            self.logger.error(f"KB embedding rate-limited (transient): {e}")
            return ("", []) if return_sources else ""
        except Exception as e:
            self.logger.error(f"KB query error: {e}")
            self.logger.error(traceback.format_exc())
            return ("", []) if return_sources else ""

    def _cap_per_source(self, nodes, cap: int):
        """Hard-cap each source URL's share of the rerank pool.

        Why URL and not domain: the failure mode this exists to address is
        ONE source page being split into many chunks (a 30-chunk Wikipedia
        article dominates the top-K via repeated near-identical embeddings).
        That failure is keyed on the URL, not the netloc — capping by
        netloc unfairly restricts legitimate multi-page coverage of one
        documentation site.

        Why this exists at all: LLMRerank processes nodes in independent
        batches (choice_batch_size), so the reranker never sees the full
        pool and cannot itself recognize "one source is over-represented."
        Diversification has to happen pre-rerank.

        Nodes whose metadata lacks `url` are bucketed under `<unknown>`
        and capped together so url-less floods can't bypass the cap.
        """
        counts, kept = {}, []
        for n in nodes:
            meta = getattr(n.node if hasattr(n, 'node') else n, 'metadata', {}) or {}
            key = meta.get('url') or '<unknown>'
            if counts.get(key, 0) < cap:
                kept.append(n)
                counts[key] = counts.get(key, 0) + 1
        return kept

    def _retrieve_nodes(self, question: str, retrieval_k: int, filters: MetadataFilters):
        """Retrieve nodes with instance-specific embed model"""
        retriever = self.index.as_retriever(
            similarity_top_k=retrieval_k,
            embed_model=self.embed_model,
            filters=filters)

        self.logger.debug(f"Using retriever (n={retrieval_k}) for query")
        return retriever.retrieve(question)

    def _rerank_and_respond(self, question: str, nodes: list, top_n: int):
        """Handle reranking with LLMRerank and response generation.

        Caller is responsible for computing the floored top_n. This method
        rebuilds self.reranker only when its current top_n no longer matches
        the requested top_n (LLMRerank fixes top_n at construction).
        """
        if getattr(self.reranker, 'top_n', None) != top_n:
            self.reranker = LLMRerank(top_n=top_n, llm=self.llm)

        # Use LLMRerank to rerank nodes
        reranked_nodes = self.reranker.postprocess_nodes(nodes, query_str=question)

        # Never let the reranker throw away the whole retrieval. LLMRerank
        # parses the LLM's "Doc: N, Relevance: M" lines, so anything that makes
        # a batch unparseable -- an off-format reply, or empty content because a
        # reasoning model spent its completion budget on reasoning tokens --
        # silently yields zero nodes. The synthesizer is then handed no context
        # at all and answers "Empty Response" with zero sources, which is
        # indistinguishable from a genuinely empty knowledge base.
        #
        # Falling back to the embedding order is strictly better than answering
        # from nothing: these nodes are the top similarity matches that were
        # about to be reranked, not arbitrary ones. A real "nothing is relevant"
        # verdict on an on-topic query is far less likely than a parse failure,
        # and this is logged at warning so the distinction stays visible.
        if not reranked_nodes and nodes:
            self.logger.warning(
                "LLMRerank returned 0 of %d nodes; falling back to the top %d "
                "by embedding similarity so the answer keeps its sources.",
                len(nodes), top_n)
            reranked_nodes = nodes[:top_n]

        # Build context from reranked nodes
        # context = "\n\n".join([node.node.get_content() for node in reranked_nodes])

        # Generate response
        question = f"{question}\n\nIMPORTANT: Keep your response under {self.response_max_length} words!" if self.response_max_length else question
        response = self.response_synthesizer.synthesize(question, nodes=reranked_nodes)

        self.logger.debug(f"Using reranker (n={top_n}) for query")
        return str(response), reranked_nodes

    def _standard_query(self, question: str, top_k: int, filters: MetadataFilters):
        """Standard query with optional system prompt prepended.

        Returns (response_text, source_nodes) so the caller can build an
        accurate `sources` list reflecting what the engine actually retrieved
        — not a separately-diversified pool that may diverge.
        """

        # Simply prepend system prompt to the question
        if self.agent.role:
            question = f"{self.agent.role}\n\n{question}"

        engine = self.index.as_query_engine(
            similarity_top_k=top_k,
            response_mode="compact",
            llm=self.llm,
            embed_model=self.embed_model,
            filters=filters)
        question = f"{question}\n\nIMPORTANT: Keep your response under {self.response_max_length} words!" if self.response_max_length else question
        response = engine.query(question)
        return str(response), getattr(response, "source_nodes", []) or []

    def multi_query(self, question, n_queries=3, top_k=None, top_n=None,
                    return_sources=False, return_all=False, filters=None):
        """Run N diverse rewrites of the question and fuse the answers.

        Uses the agent's chat_completion to paraphrase the question into
        semantically varied rewrites, reuses self.query() for each rewrite,
        then fuses the per-rewrite answers into a single response.

        If return_all=True, skip fusion and return the per-rewrite answers as
        a list (or a list of (answer, sources) tuples when return_sources=True).
        """
        try:
            if self.size() == 0:
                if return_all:
                    return []
                return ("", []) if return_sources else ""

            n_queries = max(int(n_queries), 1)

            # 1. Ask the agent's LLM for diverse rewrites
            queries = [question]
            n_rewrites = n_queries - 1
            if n_rewrites > 0:
                example = ", ".join(['"..."'] * n_rewrites)
                rewrite_prompt = (
                    f"Rewrite the following question in {n_rewrites} diverse way(s). "
                    f"Preserve the original intent, but vary wording, angle, and emphasis "
                    f"so that each rewrite surfaces different relevant context when used "
                    f"for retrieval.\n\n"
                    f"Question: {question}\n\n"
                    f'Respond as JSON: {{"rewrites": [{example}]}}'
                )
                try:
                    raw = self.agent.chat_completion(
                        prompt=rewrite_prompt,
                        response_format={"type": "json_object"})
                    parsed = parse_json_response(raw) or {}
                    rewrites = parsed.get("rewrites") if isinstance(parsed, dict) else None
                    if isinstance(rewrites, list):
                        for r in rewrites:
                            if isinstance(r, str) and r.strip() and r not in queries:
                                queries.append(r.strip())
                            if len(queries) >= n_queries:
                                break
                except Exception as e:
                    self.logger.error(f"multi_query rewrite failed, using original only: {e}")

            # 2. Reuse self.query() for each rewrite (in parallel — I/O bound).
            # Per-rewrite isolation: a single failing rewrite must not poison
            # the batch. ex.map() would propagate the first exception and
            # discard every successful answer; wrap each call so a failure
            # yields ("", []) and the downstream `if resp:` filter drops it.
            def _safe_query(q):
                try:
                    return self.query(q, top_k=top_k, top_n=top_n,
                                      return_sources=True, filters=filters)
                except FatalLLMError:
                    # Auth/quota/cost-limit must still bubble — those mean
                    # every subsequent rewrite would fail the same way.
                    raise
                except Exception as e:
                    self.logger.error(f"multi_query rewrite failed for {q!r}: {e}")
                    return "", []

            with ThreadPoolExecutor(max_workers=len(queries)) as ex:
                results = list(ex.map(_safe_query, queries))

            answers, per_query_sources, all_sources = [], [], []
            for resp, srcs in results:
                if resp:
                    answers.append(resp)
                    per_query_sources.append(srcs)
                    all_sources.extend(srcs)

            if not answers:
                if return_all:
                    return []
                return ("", []) if return_sources else ""

            # Early return: caller wants raw per-rewrite answers, skip fusion
            if return_all:
                if self.log_db:
                    with open(os.path.join(self.agent.get_log_dir(),
                        self.agent.get_id() + ".db-query.log"), "a") as f:
                        f.write("=" * 80)
                        f.write(f"\n\nMULTI_QUERY (return_all): {question}\n\nREWRITES: {queries}\n\nANSWERS: {answers}\n\n")
                if return_sources:
                    return list(zip(answers, per_query_sources))
                return answers

            # 3. Fuse per-rewrite answers into a single response
            if len(answers) == 1:
                fused = answers[0]
            else:
                joined = "\n\n---\n\n".join(
                    f"Answer {i + 1}: {a}" for i, a in enumerate(answers))
                fuse_prompt = (
                    f"Question: {question}\n\n"
                    f"Below are {len(answers)} answers produced from diverse "
                    f"retrievals. Synthesize a single coherent answer that integrates "
                    f"the most useful and non-redundant information across them.\n\n"
                    f"{joined}"
                )
                if self.response_max_length:
                    fuse_prompt += f"\n\nIMPORTANT: Keep your response under {self.response_max_length} words!"
                try:
                    fused = self.agent.chat_completion(prompt=fuse_prompt) or answers[0]
                except Exception as e:
                    self.logger.error(f"multi_query fusion failed, returning first answer: {e}")
                    fused = answers[0]

            if self.log_db:
                with open(os.path.join(self.agent.get_log_dir(),
                    self.agent.get_id() + ".db-query.log"), "a") as f:
                    f.write("=" * 80)
                    f.write(f"\n\nMULTI_QUERY: {question}\n\nREWRITES: {queries}\n\nRESPONSE: {fused}\n\n")

            if return_sources:
                seen, dedup = set(), []
                for s in all_sources:
                    url = s.get('url')
                    if url and url not in seen:
                        seen.add(url)
                        dedup.append(s)
                return fused, dedup
            return fused

        except Exception as e:
            self.logger.error(f"KB multi_query error: {e}")
            self.logger.error(traceback.format_exc())
            if return_all:
                return []
            return ("", []) if return_sources else ""

    def info(self):
        """Get knowledge base information"""
        try:
            return {
                "name": self.collection_name,
                "count": self.collection.count(),
                "url": self.server.server_url,
                "running": self.server.is_running(),
                "reranking": self.reranker is not None,
                "chunk_size": getattr(self, 'chunk_size', None),
                "chunk_overlap": getattr(self, 'chunk_overlap', None)
            }
        except Exception as e:
            self.logger.error(f"Failed to get info: {e}")
            return {"error": str(e)}

    def shutdown(self):
        """Shutdown the knowledge base"""
        # Release client from server tracking
        if hasattr(self, 'client'):
            self.server.release_client(self.client)

        # Only stop server if we own it (not shared)
        if self._owns_server and hasattr(self, 'server'):
            self.server.stop()

        self.logger.info("ChromaClientManager shutdown completed")