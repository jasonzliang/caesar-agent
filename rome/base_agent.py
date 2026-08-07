"""Base Agent class providing core functionality for all agent types"""
import copy
import os
import json
import time
from datetime import datetime
from typing import Dict, Optional, Callable
import yaml

from .llm_handler import LLMHandler
from .config import (DEFAULT_CONFIG, LOG_DIR_NAME, AGENT_NAME_LENGTH, LONGEST_SUMMARY_LEN,
                     set_attributes_from_config, merge_with_default_config, format_yaml_like)
from .logger import get_logger
from .parsing import parse_python_response, parse_json_response
from .process import process_managed
from .agent_memory import AgentMemory

yaml.add_representer(list, lambda dumper, data: dumper.represent_sequence(
    'tag:yaml.org,2002:seq', data, flow_style=True))


@process_managed
class BaseAgent:
    """Base agent providing core infrastructure for all agent types"""

    def __init__(self, name: str = None, role: str = None,
                 repository: str = None, config: Dict = None):
        """Initialize base agent with core components"""
        self.logger = get_logger()
        self._setup_config(config)
        self._validate_name_role(name, role)
        self._setup_repository_and_logging(repository)
        self._setup_llm_handler()
        self._setup_agent_memory()
        self._register_cleanup()
        self.export_config()

        self.logger.info(f"BaseAgent '{self.name}' initialized")

    def _setup_config(self, config_dict: Dict = None) -> None:
        """Setup and validate configuration"""
        self.config = merge_with_default_config(config_dict) if config_dict else \
            copy.deepcopy(DEFAULT_CONFIG)
        agent_config = self.config.get('Agent', {})
        set_attributes_from_config(self, agent_config,
            ['name', 'role', 'repository', 'log_pid'])

    def _validate_name_role(self, name: str, role: str) -> None:
        """Validates and formats agent name and role"""
        if role: self.role = role
        if name: self.name = name
        self.logger.assert_true(self.role and self.name,
            f"Invalid name or role: {name}, {role}")

        if "your role" not in self.role.lower():
            self.role = f"Your role:\n{self.role}"

        a, b = AGENT_NAME_LENGTH
        clean_name = ''.join(c for c in self.name if c.isalnum())
        clean_name = clean_name.ljust(a, '0') if len(clean_name) < a else clean_name[:b]

        self.logger.assert_true(clean_name and a <= len(clean_name) <= b,
            f"Agent name must be {a}-{b} alphanumeric characters")
        self.name = clean_name

    def _setup_repository_and_logging(self, repository: str = None) -> None:
        """Validate repository and configure logging"""
        if repository: self.repository = str(repository)
        self.logger.assert_true(
            self.repository and os.path.exists(self.repository),
            f"Repository path does not exist: {self.repository}")

        log_config = self.config.get('Logger', {})
        if not log_config.get('base_dir'):
            log_config['base_dir'] = self.get_log_dir()
        if not log_config.get('filename'):
            log_config['filename'] = f"{self.get_id()}.console.log"
        get_logger().configure(log_config)

        self.logger.info(f"Logging configured: {log_config['base_dir']}")

    def _setup_llm_handler(self) -> None:
        """Initialize LLM handler"""
        llm_config = self.config.get('LLMHandler', {})
        self.llm_handler = LLMHandler(config=llm_config)
        self.logger.debug("LLM handler initialized")

    def _setup_agent_memory(self) -> None:
        """Initialize agent memory system"""
        memory_config = self.config.get('AgentMemory', {})

        self.agent_memory = AgentMemory(
            agent=self,
            config=memory_config
        )

        if self.agent_memory.is_enabled():
            self.logger.info(f"Agent memory enabled (auto_recall={self.agent_memory.auto_recall}, auto_remember={self.agent_memory.auto_remember})")
        else:
            self.logger.debug("Agent memory disabled")


    def _register_cleanup(self) -> None:
        """Register cleanup handlers"""
        self.shutdown_called = False

    def get_id(self) -> str:
        """Unique identifier for agent in file system"""
        return f'agent_{self.name}_{os.getpid()}' if self.log_pid else f'agent_{self.name}'

    def get_repo(self) -> str:
        """Get agent repository, creating if needed"""
        os.makedirs(self.repository, exist_ok=True)
        return self.repository

    def get_log_dir(self) -> str:
        """Get agent log directory, creating if needed"""
        log_dir = os.path.join(self.repository, LOG_DIR_NAME)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def export_config(self, filepath: str = None) -> None:
        """Export agent configuration to YAML file.

        Secret-looking values (api_key, password, token, ...) are redacted so
        the on-disk config never persists credentials such as a per-run user
        OpenAI key. The in-memory self.config is untouched.
        """
        secret_names = ("api_key", "apikey", "secret", "password", "token")

        def redact(obj):
            if isinstance(obj, dict):
                return {
                    k: ("[REDACTED]"
                        if isinstance(k, str)
                        and any(s in k.lower() for s in secret_names) and v
                        else redact(v))
                    for k, v in obj.items()
                }
            if isinstance(obj, list):
                return [redact(v) for v in obj]
            return obj

        filepath = filepath or os.path.join(self.get_log_dir(), f"{self.get_id()}.config.yaml")
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(redact(self.config), f, default_flow_style=False, sort_keys=False, indent=4)
        self.logger.info(f"Configuration exported: {filepath}")

    def parse_json_response(self, response: str) -> Dict:
        """Parse JSON response"""
        result = parse_json_response(response)
        if result is None:
            self.logger.error(f"Failed to parse JSON from response: {response}")
        return result

    def parse_python_response(self, response: str) -> str:
        """Parse Python code from response"""
        result = parse_python_response(response)
        if result is None:
            self.logger.error(f"Failed to parse Python code from response: {response}")
        return result

    def shutdown(self) -> None:
        """Clean up resources before termination"""
        if self.shutdown_called: return
        self.shutdown_called = True

        try:
            from posthog import posthog
            if posthog: posthog.shutdown()
        except: pass

        try:
            self.logger.info("BaseAgent shutdown completed")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise

    def remember(self, content: str, context: str = None, metadata: Dict = None) -> bool:
        """Store content in long-term memory"""
        return self.agent_memory.remember(content, context, metadata)

    def recall(self, query: str, context: str = None) -> str:
        """Query long-term memory and get most relevant results"""
        return self.agent_memory.recall(query, context)

    def summarize_past(self, total_items: int = 40, recent_ratio: float = 0.5) -> str:
        """Recall both short term and long term memory"""
        return self.agent_memory.summarize_past(total_items, recent_ratio)

    def chat_completion(self, prompt: str, system_message: str = None,
                       override_config: Dict = None, response_format: Dict = None,
                       num_retries: Optional[int] = None) -> str:
        """Direct access to chat completion with configuration options"""
        if self.agent_memory.is_enabled():
            return self.agent_memory.chat_completion(
                prompt=prompt, system_message=system_message,
                override_config=override_config, response_format=response_format,
                num_retries=num_retries)
        else:
            system_message = system_message or self.role
            return self.llm_handler.chat_completion(
                prompt=prompt, system_message=system_message,
                override_config=override_config, response_format=response_format,
                num_retries=num_retries)
