# config.py
import ast
import copy
import numbers
import os
import sys
import yaml
from types import NoneType
from typing import Dict, Any
from .logger import get_logger

######## These constants are not intended to be user modifiable ########
# Agent API allowed port range
API_PORT_RANGE = (40000, 41000)

# Min and max allowed lengths for agent names
AGENT_NAME_LENGTH = (8, 32)

# Default hash function to use for check file versions and AST cache
DEFAULT_HASH_FUNC = "sha256"

# How long to display long strings for console output
SHORT_SUMMARY_LEN = 50
SUMMARY_LENGTH = 100
LONG_SUMMARY_LEN = 200
LONGER_SUMMARY_LEN = 400
LONGEST_SUMMARY_LEN = 800

# Meta directory for file specific information
META_DIR_EXT = 'rome'

# Default logging base directory name for agents
LOG_DIR_NAME = "__rome__"

# Name of evaluation directory with GD test results
EVAL_DIR_NAME = "evaluation"

# Name of JSON with evaluation results in it
EVAL_RESULTS_NAME = "eval_results.json"

# Default extension for test files:
TEST_FILE_EXT = '_test.py'

# Graphviz DPI for drawing stuff
GRAPHVIZ_DPI = 200
######## These constants are not intended to be user modifiable ########

# Define the default configuration structure as a dictionary
# This should show the complete configuration of every class
DEFAULT_CONFIG = {
    # CORE AGENT CONFIGURATION

    "Agent": {
        "name": None,  # Agent name, can be overwritten by constructor
        "role": None,  # Agent description, can be overwritten by constructor
        "repository": None,  # Repository base directory, can be overwritten by constructor

        "fsm_type": "simple",  # Which FSM to load (see fsm_factory.py)
        "action_select_strat": "original",  # Which action selector to use (original or smart)
        "history_context_len": 24,  # Length of history to use when selecting action
        "patience": 4,  # If same state/action chosen repeatedly, prompt to choose different action

        "agent_api": True,  # Launch an REST API server for agent's internal state
        "log_pid": False,  # Whether to include PID in agent log file names
        "save_hist_interval": 10, # Interval to save summary history
        "draw_fsm": True, # Whether to draw FSM graph when initialized

        "use_ground_truth": False, # Whether to use EvalPlus results to determine if work is complete
        "save_insights": False, # Whether to save insights to knowledge base or not
        "query_insights": False, # Whether to query knowledge base for insights or not

    },

    "AgentMemory": {
        "enabled": False,  # Enable/disable agent memory system
        "auto_recall": False,  # Automatically inject relevant memories into chat_completion
        "recall_limit": 10,  # Number of most relevant memories to retrieve in recall()
        "auto_remember": False,  # Automatically extract and store important interactions
        "auto_remember_len": 100,  # Minimum characters (prompt + response) to consider storing
        "clear_memory": False,  # Whether to automatically clear memory on startup
        "infer": False, # Whether to enable inference to consolidate and process memories
        # DB settings (for tracking entities/relationships like file visits)
        "use_graph": False,  # Enable graph memory (neo4j/memgraph required)
        "embedding_model": "text-embedding-3-small",  # OpenAI embedding model for semantic search
        "graph_url": "bolt://localhost:7687",  # Graph database connection URL
        "graph_username": "neo4j",  # Graph database username
        # No default: a baked-in password made agent_memory.py's "graph enabled but no
        # password" guard unreachable, and shipped a real literal inside the wheel.
        # Set NEO4J_PASSWORD (read in agent_memory.py) or override here.
        "graph_password": None,
        "vector_host": "localhost",  # Chroma vector DB host
        "vector_port": 8000,  # Chroma vector DB port
    },

    "AgentApi": {
        "host": None,  # API url/hostname will be set to localhost if None
        "port": None  # API port, will be set automatically from 40000 if None
    },

    "MultiAgent": {
        "agent_role_json": None,  # Multi agent json, can be overwritten by constructor
        "repository": None,  # Multi agent repo, can be overwritten by constructor
        "suppress_output": True, # Suppress individual agent console output
    },

    # LLM HANDLER CONFIGURATION

    "LLMHandler": {
        # API Configuration
        "provider": "openai",  # LLM provider: "openai", "anthropic", "gemini"
        "base_url": "https://api.openai.com/v1",  # Url for chat completion API
        "key_name": "OPENAI_API_KEY",  # API key name to find in env
        "timeout": 60,  # Time for allowing responses from API
        "cost_limit": 500.0,  # Max budget ($) for running LLM chat completion
        "max_retries": 2,  # litellm/openai-SDK retry count on transient errors (timeouts, 5xx). Matches openai-python's own default; was 10 historically, which silently amplified hung calls into multi-hour stalls. Callers that own their own retry strategy should pass num_retries=0 explicitly.

        # Model Parameters
        "model": "gpt-4o",  # LLM model name
        "reasoning_effort": None,  # Reasoning effort for reasoning models (None to disable)
        "temperature": 0.1,  # LLM temperature
        "max_completion_tokens": 10000,  # LLM response max tokens
        "top_p": 1.0,  # LLM top-P
        "seed": None,  # LLM random seed

        # System Configuration
        "system_message": "You are a helpful assistant",  # Fall back system message

        # Context Management
        "manage_context": True,  # Prevent messages from overfilling context window
        "max_input_tokens": None,  # Set to manually override auto-calculated input token limit
        "token_count_thres": 0.5,  # Threshold when to use slow method to count tokens
        "chars_per_token": 4  # Used for fast calculation of number of tokens
    },

    # STATE MACHINE & ACTION SELECTION

    "FSM": {},

    "ActionSelector": {
        "min_interval": 1,  # Backoff detection interval (also backoff initial delay time)
        "max_tries": 5,  # Number of times to try before backoff error raised
        "backoff_enabled": True,  # Whether to enable backoff or not

        "original": {},  # Config for original selector (empty for now)
        "smart": {  # Config for smart selector
            "loop_detection_window": None,  # Override window based on agent.history_context_len
            "exploration_rate": 0.15,  # Base probability for encouraging non-optimal actions
        }
    },

    # STATES

    "IdleState": {},
    "CodeLoadedState": {},
    "CodeEditedState": {},
    "TestEditedState": {},
    "CodeExecutedPassState": {},
    "CodeExecutedFailState": {},

    # ACTIONS

    # Basic Actions
    "ResetAction": {},
    "AdvancedResetAction": {
        "completion_confidence": 90,  # Minimum completion confidence threshold to save insights
        "max_versions": 30,  # Maximum versions before forcing to save insights
    },
    "TransitionAction": {},

    # Search Actions
    "PrioritySearchAction": {
        "randomness": 15,
        "batch_size": 10,  # Candidate batch size for selection
        "selection_criteria": "Select the file that you have the most confidence in improving or fixing.",
    },
    "TournamentSearchAction": {
        "batch_size": 0.1,  # Number of files to randomly choose
        "selection_criteria": "Select the file that you have the most confidence in improving or fixing."
    },

    # Code Manipulation Actions
    "EditCodeAction": {},
    "EditCodeAction2": {},
    "EditTestAction": {},
    "EditTestAction2": {},
    "ExecuteCodeAction": {},
    "RevertCodeAction": {
        "num_versions": 10, # Number of versions to look back to revert
    },

    # Knowledge Base Actions
    "SaveKBAction": {
        "completion_confidence": 90,  # Minimum completion confidence threshold to save insights
        "max_versions": 30,  # Maximum versions before forcing to save insights
    },

    # REPOSITORY & FILE MANAGEMENT

    "RepositoryManager": {
        "file_types": [".py"],  # File types to search for
        "max_files": sys.maxsize,  # Maximum number of candidates to select from after search
        "exclude_types": [],  # File types to not search for
        "exclude_dirs": [".*", "venv", "__*__"],  # Directories to not search in
        "easy_check_finish": False,  # Mark file as finished for all agents if any agent marks it
    },

    "DatabaseManager": {
        "lock_timeout": 5.0,  # Timeout for read/write lock
        "max_retries": 5,  # Num tries to acquire lock
        "retry_delay": 0.05  # Delay between tries, exponential increase
    },

    "VersionManager": {
        "lock_active_file": False, # Set to true to make each file have only 1 agent working on it
    },

    # CODE EXECUTION

    "Executor": {
        "timeout": 10,  # Maximum time for code to run
        "virtual_env_context": None,  # Name of virtual env to run in
        "work_dir": "./", # Working directory when running code directly (not code file)
        "max_output_len": 5000,  # Maximum tokens for execution output
        "cmd_args": {  # Additional useful flags when executing code
            "pytest": ["-s", "--tb=short", "--no-header"],
            "python": ["-u"],
        }
    },

    # LOGGING & MONITORING

    "Logger": {
        "level": "DEBUG",  # Log level in increasing verbosity: INFO -> ERROR -> DEBUG
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Formatting
        "console": True,  # Print to console if true
        "include_caller_info": None,  # Can be "rome", "rich", or None
        "base_dir": None,  # Directory for log files, overwritten by agent if set to None
        "filename": None,  # Log file name, overwritten by agent if set to None
        "max_size_kb": 10000,  # Max log size in kb, truncates after exceeding
        "timezone": 'US/Pacific',  # Default timezone to display
    },

    # KNOWLEDGE BASE MANAGEMENT

    "ChromaClientManager": {
        # Note: LLM model and temperature are inherited from agent's LLMHandler config
        "collection_name": None,  # ChromaDB collection name, set to None to use agent repo
        "enable_reranking": True,  # Enable LLMRerank reranking
        "use_shared_server": True,  # Use shared server instance across KB instances
        "log_db": True,  # Log adding docs and queries to file

        # LlamaIndex configuration
        "embedding_model": "text-embedding-3-small",  # OpenAI embedding model
        "model": "gpt-5-mini",  # OpenAI llm model (for query/reranking)
        "temperature": 0.1,  # OpenAI llm model (for query/reranking)
        "reasoning_effort": "low",  # Reasoning effort for reasoning models (None to disable)

        "chunk_size": 400,
        "chunk_overlap": 80,
        "top_k": 10,  # Number of docs to retrieve (without reranking)
        "response_max_length": None,  # Maximum number of tokens for query response

        # Reranking configuration
        # "rerank_batch_size": 10,  # Batch size for reranking choices
        "rerank_top_k": 15,  # Number of docs to retrieve before reranking
        "rerank_top_n": 5,  # Number of top docs to return after reranking
    },


    "ChromaServerManager": {
        "host": "localhost",  # ChromaDB server host
        "port": 8000,  # ChromaDB server port
        "persist_path": None,  # Data persistence directory (None = auto-set user data dir)
        "startup_timeout": 5,  # Server startup timeout in seconds
        "shutdown_timeout": 2,  # Server shutdown timeout in seconds
    },
}


### Helper methods for setting/loading config parameters ###
def check_attrs(obj, required_attrs):
    """Helper function to check if required attributes have been set"""
    logger = get_logger()
    for attr in required_attrs:
        logger.assert_attribute(obj, attr)
        # logger.debug(f"'{attr}' provided in {obj.__class__.__name__} config")


def check_opt_attrs(obj, optional_attrs):
    """Helper function to check if optional attributes have been set"""
    logger = get_logger()
    for attr in optional_attrs:
        if not hasattr(obj, attr):
            setattr(obj, attr, None)
            # logger.debug(f"'{attr}' (optional) not provided in {obj.__class__.__name__} config")


def set_attributes_from_config(obj, config=None, required_attrs=None, optional_attrs=None):
    """
    Helper function to convert configuration dictionary entries to object attributes

    Args:
        obj: The object to set attributes on
        config: The configuration dictionary
        required_attrs: List of attributes that must be set from config
        optional_attrs: List of attributes that are optional
    """
    logger = get_logger()

    # Set each config parameter as an attribute
    if config:
        for key, value in config.items():
            setattr(obj, key, value)

    if required_attrs:
        check_attrs(obj, required_attrs)

    if optional_attrs:
        check_opt_attrs(obj, optional_attrs)

    logger.debug(f"Applied configuration to {obj.__class__.__name__} attributes")


def generate_default_config(output_path="config.yaml"):
    """Generate a default configuration YAML file"""
    logger = get_logger()

    with open(output_path, 'w') as f:
        yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Default configuration saved to {output_path}")
    print(f"Default configuration saved to {output_path}")


def load_config(config_path="config.yaml", create_if_missing=False):
    """Load configuration from a YAML file or Python file, creating it if it doesn't exist"""
    logger = get_logger(); config_path = str(config_path)

    if os.path.exists(config_path):
        logger.info(f"Loading configuration from {config_path}")
        _, ext = os.path.splitext(config_path.lower())

        if ext in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        elif ext == '.py':
            with open(config_path, 'r') as f:
                tree = ast.parse(f.read())

            # Check for first dictionary that has META_DIR_EXT (rome) in its name
            for node in tree.body:
                if (isinstance(node, ast.Assign) and
                    any(isinstance(target, ast.Name) and META_DIR_EXT.lower() in target.id.lower()
                        for target in node.targets)):
                    config = ast.literal_eval(node.value)
                    if not isinstance(config, dict):
                        raise TypeError(f"Config must be a dictionary in {config_path}")
                    return config

            raise ValueError(f"No ROME_CONFIG found in {config_path}")
        else:
            raise ValueError(f"Unsupported config file format: {ext}")

    elif create_if_missing:
        logger.info(f"Config file {config_path} not found. Creating default config...")
        generate_default_config(config_path)
        return DEFAULT_CONFIG
    else:
        raise FileNotFoundError(f"Config file {config_path} not found.")


def merge_with_default_config(custom_config):
    """Merge a custom config with the default config and do some basic validation"""

    def error_msg(key, val):
        return f"Invalid value {val} ({type(val).__name__}) for config parameter '{key}'"

    def same_type(x, y):
        return type(x) == type(y) or (isinstance(x, numbers.Number) and isinstance(y, numbers.Number))

    def validate(key, old_v, new_v):
        # Easy to way in include key but still use default
        if new_v == "default":
            new_v = old_v

        # Check old and new value type matches
        if old_v is not None:
            logger.assert_true(same_type(old_v, new_v), f"Type mismatch for old {type(old_v).__name__} and new value {type(new_v).__name__} for config parameter '{key}'")
        else:
            logger.debug(f"Type checking skipped for config parameter '{key}' ({old_v} -> {new_v})")

        # Key value must be >= 0 or length >= 0 (exception for 'exclude_' keys)
        for v in [old_v, new_v]:
            if isinstance(v, (list, tuple, str)):
                if not key.startswith("exclude_"):
                    logger.assert_true(len(v) > 0, f"{error_msg(key, v)} — non-empty value")
            elif isinstance(v, (int, float)):
                logger.assert_true(v >= 0, f"{error_msg(key, v)} — non-negative value")
            else:
                logger.assert_true(isinstance(v, (bool, NoneType)),
                    f"{error_msg(key, v)} - invalid type")
        return new_v

    def update_dict(d, u):
        for k, v in u.items():
            # If dict, recursively update it
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = update_dict(d[k], v)
            elif k in d: # If key in orig dict, validate it
                d[k] = validate(k, d[k], v)
            else: # If key not in orig dict, just add it
                d[k] = v
        return d

    logger = get_logger()

    # Backward compat: map old config key to new one
    if "OpenAIHandler" in custom_config and "LLMHandler" not in custom_config:
        custom_config["LLMHandler"] = custom_config.pop("OpenAIHandler")

    merged_config = copy.deepcopy(DEFAULT_CONFIG)
    result = update_dict(merged_config, custom_config)
    logger.info("Configuration merged with defaults")
    return result


def format_yaml_like(data, indent=0):
    """Recursive function to format dict as YAML-like string"""
    lines = []
    spaces = "    " * indent

    for key, value in sorted(data.items()):
        if isinstance(value, dict):
            lines.append(f"{spaces}{key}:")
            lines.extend(format_yaml_like(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{spaces}{key}:")
            for item in value:
                if isinstance(item, dict):
                    lines.append(f"{spaces}  -")
                    lines.extend(format_yaml_like(item, indent + 2))
                else:
                    lines.append(f"{spaces}  - {item}")
        else:
            # Handle None values and convert to string
            display_value = value if value is not None else "null"
            lines.append(f"{spaces}{key}: {display_value}")

    return lines


if __name__ == "__main__":
    pass