# Maximum number of links to consider when selecting next webpage to go to
MAX_NUM_LINKS = 1000
# Maximum number of visited links to consider when selecting next webpage
MAX_NUM_VISITED_LINKS = 500
# Max graph distance for collecting related insights in think
NEIGHBORHOOD_DISTANCE = 3
# Maximum number of neighboring nodes to use for related insights in think
MAX_NUM_NEIGHBORS = 24
# Maximum number of past nodes to use for past insights in think
MAX_NUM_ANCESTORS = 8
# Maximum length (characters) of text to extract from a webpage
MAX_TEXT_LENGTH = 100000
# Timeout for fetching webpage html using requests
# 25s accommodates slow academic/gov sites (arxiv, springer, nih) that
# regularly take 15+s on first load. 10s caused false failures.
REQUESTS_TIMEOUT = 30
# Cap on per-retry exponential backoff (s), shared by the API/search clients
# (brave_search, semantic_scholar). Without it, 2^N growth turns a wedged
# endpoint into an unkillable multi-hour hang; this bounds retry wall-clock so
# the retry budget actually terminates.
MAX_BACKOFF_DELAY = 30
# Static headers for requests when fetching html.
# Referer and Sec-Fetch-Site are NOT included here — they depend on the
# previous URL and are computed per-request by _compute_nav_headers().
REQUESTS_HEADERS = {
    # 1. Standard Chrome Accept Header
    # Must include image formats (avif, webp, apng) because real Chrome always requests them.
    # Missing these is a high-confidence bot signal when impersonating a browser.
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',

    # 2. Accept-Encoding (CRITICAL)
    # You MUST include 'br' (Brotli) and 'zstd'.
    # Because you use impersonate="chrome", your TLS fingerprint claims you support these.
    # If you remove them here, the server sees a "TLS vs Header Mismatch" and blocks you instantly.
    # curl_cffi handles the decompression automatically, so this is safe to include.
    'Accept-Encoding': 'gzip, deflate, br, zstd',

    # 3. Standard Language & Cache
    'Accept-Language': 'en-US,en;q=0.9',
    'Cache-Control': 'max-age=0',

    # 4. Fetch Metadata (Sec-Fetch-Site is computed per-request)
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
}
# Baseline blocked domains: anti-bot gates, academic paywalls, and auth walls
# These are always blocked; user config can add more via blocked_domains.
BASELINE_BLOCKED_DOMAINS = [
    # Anti-bot / aggressive rate-limiting
    "scholar.google.com",
    "scholar.google.co.uk",
    "scholar.google.ca",
    "ui.adsabs.harvard.edu",
    # Academic paywalls (consistently fail without auth)
    "sciencedirect.com",
    "link.springer.com",
    "onlinelibrary.wiley.com",
    "jstor.org",
    "academic.oup.com",
    "journals.sagepub.com",
    "cell.com",
    "jneurosci.org",
    "royalsocietypublishing.org",
    "direct.mit.edu",
    # Auth walls
    "researchgate.net",
    "academia.edu",
]
# Maximum number of citation sources per query during synthesis
MAX_SYNTHESIS_QUERY_SOURCES = 5
# Max QA pairs to keep in context during iterative synthesis
MAX_SYNTHESIS_QA_CONTEXT = 50
# Number of retries for artifact synthesis before failing
NUM_SYNTHESIS_RETRIES = 5
# Whether to save synthesis artifact as JSON (in addition to text)
SYNTHESIS_SAVE_JSON = False
# Max times the starting page can be regenerated ("reseeded") via a fresh
# Brave search when its link pool is exhausted (separate from max_web_searches)
MAX_RESEEDS = 5

CAESAR_CONFIG = {
    "CaesarAgent": {
        # Maximum number of words for adapting agent role description
        "role_max_length": 350,
        # Total number of pages to explore before stopping
        "max_iterations": 1000,
        # Maximum depth of exploration tree before backtracking
        "max_depth": 1000,
        # Initial URL to begin exploration (enabling means start_query must be None)
        "starting_url": None,
        # Initial query to being exploration (enabling means starting_url must be None)
        "starting_query": None,
        # Additional queries to generate from initial query to help agent
        "additional_starting_queries": 5,
        # How strongly the starting_query steers exploration:
        #   "none"     — query never enters strategy or link-select prompts (pure drift)
        #   "strategy" — query steers the meta strategy decision only (partial drift)
        #   "full"     — query anchors every per-hop link selection too (no drift)
        "query_influence": "full",
        # Exploration mode: "web" (Brave/DDGS + open-web crawl, default) or
        # "arxiv" (Semantic Scholar + arxiv citation-graph traversal; needs a
        # starting_query; tunables in the "SemanticScholar" section below).
        "mode": "web",
        # Domains to allow exploration; empty list uses starting_url domain; use ["*"] to allow any
        "allowed_domains": [],
        # Domain-level blocks: substring match against URL netloc (merged with BASELINE_BLOCKED_DOMAINS)
        "blocked_domains": [],
        # URL-level blocks: case-insensitive substring match against full URL (path + query)
        "blocked_urls": [],
        # Generate visual graph representations (requires pygraphviz)
        "draw_graph": False,
        # Save exploration graph every N iterations
        "save_graph_interval": 1,
        # Save checkpoint every N iterations for resumption
        "checkpoint_interval": 1,

        # Ablation knob: True falls back to the orig think (1-hop, no history).
        "use_orig_think": False,
        # Whether to follow links that point to the same page (different fragments)
        "same_page_links": False,
        # Maximum times to revisiting page it has seen before during exploration
        "max_allowed_revisits": 20,
        # Whether to use new link display format or not
        "fancy_link_display": True,
        # Whether to dynamically determine to explore or go back to visited pages
        "use_explore_strategy": True,
        # Maximum web searches allowed during exploration (not including starting search)
        "max_web_searches": 0,
        # Whether to enable quick exploration, analyze all search results instead
        "use_quick_explore": False,
        # Whether to use insight generation for quick explore mode
        "quick_explore_insights": False,
        # Number of parallel workers for quick explore mode
        "quick_explore_workers": 20,
        # LLM config for agent's ACT/THINK phases to encourage exploration
        "exploration_llm_config": {
            "model": "gpt-5.4",
            "reasoning_effort": "low",
            "temperature": 0.9,
            "max_completion_tokens": 5000,
            "timeout": 120,
        },

        # Overwrites the default role with new role from file, order is [overwrite] -> [adapt]
        "overwrite_role_file": None,
        # Whether to modify agent role based on starting URL and/or insights
        "adapt_role": False,
        # Insights file path used to adapt the role and change it
        "adapt_role_file": None,
        # Whether to load role from checkpoint, or recreate it when agent starts
        "load_saved_role": True,
    },

    "ArtifactSynthesizer": {
        # Set to True for classic mode (ask all queries at once)
        "synthesis_classic_mode": False,
        # Number of drafts of synthesizing artifact
        "synthesis_drafts": 3,
        # Number of Q/A iterations per draft
        "synthesis_iterations": 20,
        # Reasoning effort for the artifact-generating SYNTHESIS + MERGE calls;
        # steps down high->medium->low on timeout (see _llm_call). The light
        # post-processes (clarify, eli5, human_eval) ignore this and run at the
        # model's default effort.
        "synthesis_reasoning_effort": "high",
        # Top_k for synthesis query and retrieval from DB
        "synthesis_top_k": 50,
        # Top_n for synthesis query and retrieval from DB
        "synthesis_top_n": 10,
        # Optional path to an external answer file used as the "previous artifact"
        # context for the FIRST draft only. Subsequent drafts continue to chain
        # off the prior draft as usual. Set to None to disable.
        "synthesis_reference_draft": None,
        # Optional query the reference draft answered. Surfaces as a
        # "previous query" block in the synthesis prompt so the LLM can
        # distinguish what was already asked from the current query.
        "synthesis_reference_query": None,
        # Maximum of words for generating final synthesis artifact
        "synthesis_max_length": None,
        # Whether to merge all draft artifacts into final artifact
        "synthesis_merge_artifacts": False,
        # Whether to run a clarity/formatting post-process on the merged artifact
        "synthesis_merge_clarify": True,
        # Whether to generate ELI5 explanation for artifact text
        "synthesis_eli5": False,
        # Maximum words for ELI5 explanation (None = unspecified word limit)
        "synthesis_eli5_length": None,
        # Whether to enable human evaluation summary for artifact text
        "synthesis_human_eval": False,
        # Maximum exploration iterations to filter knowledge base for queries
        "synthesis_iteration_filter": None,
        # Number of images to generate from the final artifact and embed as
        # markdown (in <artifact_dir>/images/). 0 disables; N>0 generates N
        # images in parallel via caesar.image_generator and re-saves the
        # synthesis with image refs spliced across paragraph boundaries.
        "synthesis_generate_images": 0,
    },

    "AgentMemory": {
        # Whether to clear memory or not on startup
        "clear_memory": False,
        # Whether to enable memory for Caesar agent
        "enabled": True,
        # Whether to use vector DB or graph DB
        "use_graph": False,
        # Whether to enable inference to consolidate memories
        "infer": False,
    },

    "BraveSearch": {
        # Total number of search results to fetch (paginated in batches of 20)
        "num_results": 20,
        # Number of tries to attempt to search. Combined with MAX_BACKOFF_DELAY=30s
        # in brave_search.py, the worst case wall-clock is ~5 * 30 = 150s before
        # the search gives up. 1000 (old default) plus uncapped exponential
        # backoff made retries unkillable when an endpoint was down.
        "max_retries": 5,
        # Delay between search retries
        "retry_delay": 1,
        # Request timeout in seconds
        "timeout": 30,
        # Filter duplicate URLs across multiple queries
        "filter_duplicates": True,
        # Shorten queries to fit API limits (400 char/50 word), options: truncation/summary
        "shorten_query": "summary",
        # When False, DDGS is used automatically if BRAVE_API_KEY not set
        # When True, force the DDGS backend regardless of BRAVE_API_KEY
        "use_ddgs": False,
    },

    # Semantic Scholar Graph API (only when CaesarAgent.mode == "arxiv").
    "SemanticScholar": {
        # Papers fetched for the initial search seed (S2 page cap: 100).
        "num_results": 20,
        # References / citations pulled per node. These are page SIZES, not call
        # counts (one call each regardless, S2 max 1000/page): raising them costs
        # response + prompt size, not rate-limited calls (429 pressure unchanged).
        # Measured on a live run: references rarely reach even 100 (papers cite
        # <~80), so refs_limit is headroom; citations are unbounded, so hub
        # papers truncate -- a large citations_limit samples more before the
        # client-side ranking. /citations is ordered RECENCY-descending, not by
        # relevance: on ARXIV:1706.03762 the first arXiv citer is at index 457,
        # so a shallow page returns nothing usable under arxiv_only. 800 keeps a
        # deep sample past that, at ~1 KB/neighbour (~0.8 MB/node, under S2's
        # 10 MB response cap). Same call count either way.
        "refs_limit": 200,
        "citations_limit": 800,
        # True = pure arxiv graph; False also keeps DOI/other cited papers.
        "arxiv_only": True,
        # Fetch + parse each visited paper's PDF for full-text content (via the
        # web fetcher's pypdf path). False = S2 abstract only (cheaper, faster,
        # shallower). Falls back to the abstract when a PDF is missing.
        "arxiv_fetch_pdf": True,
        # Min seconds between S2 calls, shared process-wide. Tune to your key
        # tier (~1 rps individual). Request timeout + retry budget are constants
        # in semantic_scholar.py, not knobs (a raisable retry is a stall risk).
        "min_request_interval": 1.1,
    },

    # Default config for LLM outside of agent exploration
    "LLMHandler": {
        # Total API cost limit in dollars
        "cost_limit": 300.0,
        # Model name for LLM
        "model": "gpt-5.4",
        # Reasoning effort for GPT-5/O models
        "reasoning_effort": "medium",
        # Base temperature for LLM (overridden by exploration_llm_config for ACT/THINK)
        "temperature": 0.1,
        # Maximum tokens per LLM response
        "max_completion_tokens": 50000,
        # API timeout in seconds. 900s = OpenAI-community spec for
        # reasoning_effort=high. Pairs with the synthesizer wrapper's
        # high→medium step-down retry (artifact_synthesis._llm_call) so
        # a timed-out high attempt fails fast and lets medium take over
        # — net worst case ~20 min for 2 attempts vs ~25 min at 1200s.
        "timeout": 900,
    },

    "ImageGenerator": {
        # Max cited URLs to scrape for candidate refs (latency/cost guard)
        "max_cited_urls": 50,
        # OpenAI image-gen model
        "image_model": "gpt-image-2",
        # VLM for scoring + captioning refs. gpt-4o collapses scores to the
        # rubric's middle (no spread) so the rerank signal dies — keep at gpt-5.4.
        "vlm_model": "gpt-5.4",
        # LLM for artifact → image-prompt synthesis
        "prompt_model": "gpt-5.4",
        # gpt-image-2 sizes: 1024x1024, 1024x1792, 1792x1024
        "size": "1792x1024",
        # gpt-image-2 quality: low/medium/high
        "quality": "high",
        # Use refs via images.edit only if top score >= this (0-10); below this,
        # mid-quality refs bias the model more than they help.
        "use_refs_top_score": 7.0,
        # Cap VLM scoring cost independent of max_cited_urls (one page can yield 50+ images).
        # Hard upper bound on VLM scoring calls. Per-domain cap (PER_DOMAIN_CAP=2)
        # plus downstream pool_size (4-7) mean only the top ~30 candidates ever
        # survive past scoring, so 60 keeps headroom while cutting VLM spend ~60%.
        "max_candidates_to_score": 60,
        # Heuristic prefilter (cheap text-only rank on alt + URL keywords) runs
        # before VLM scoring. Top vlm_score_pool_size by heuristic form the merit
        # pool; vlm_score_explore_size additional candidates are random-sampled
        # from the remainder as a control against systematic heuristic failure on
        # opaque-URL CDNs. Sum is the typical VLM call count; max_candidates_to_score
        # remains the hard upper bound above the prefilter.
        "vlm_score_pool_size": 50,
        "vlm_score_explore_size": 10,
        # Saved image format. webp is the default — smaller than png/jpg with
        # comparable quality and supported natively by browsers + gpt-image API.
        # Supported: png, jpg/jpeg, webp.
        "output_format": "webp",
        # Refs passed to each images.edit call. 3-5 is the gpt-image-2 sweet
        # spot (refs compete for influence). For n>1 each image picks a
        # chunk-specific subset by caption-vs-section cosine similarity;
        # the candidate pool is auto-sized to refs_per_image + n - 1
        # (capped at the OpenAI 16-ref API limit).
        "refs_per_image": 4,
    },

    # Default config for vector store knowledge base
    "ChromaClientManager": {
        # Model name for LLM
        "model": "gpt-5.4",
        # Reasoning effort for GPT-5/O models (ignored for non-reasoning models)
        "reasoning_effort": "low",
         # LLM model temperature (for query/reranking)
        "temperature": 0.1,
        # Maximum number of words for query response
        "response_max_length": 350,
    }
}