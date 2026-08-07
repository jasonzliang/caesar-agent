from functools import lru_cache
import json
import os
import re
import threading
import warnings
from typing import Dict, Optional, Any, Union, List

# Mute litellm's `asyncio.get_event_loop()` DeprecationWarning fired at import
# time on Python 3.12+. Upstream issue; filter is scoped to the litellm import
# so we don't muffle anyone else's deprecation warnings.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r".*There is no current event loop.*",
    )
    import litellm

from .logger import get_logger
from .config import set_attributes_from_config, DEFAULT_CONFIG

# Suppress litellm's verbose logging and drop unsupported params gracefully
litellm.suppress_debug_info = True
litellm.drop_params = True


class FatalLLMError(BaseException):
    """Non-recoverable LLM error: auth failure, insufficient quota, permission denied,
    or our own cost limit hit. Callers should propagate (not retry, not swallow).

    Inherits from BaseException (not Exception) so generic `except Exception`
    handlers in retry/recovery code do NOT catch it. Top-level run-failure
    handlers must catch it explicitly via `except FatalLLMError` or
    `except (Exception, FatalLLMError)` to surface it as the run error."""


class CostLimitExceededException(FatalLLMError):
    """Raised when our own accumulated-cost cap would be exceeded."""
    def __init__(self, estimated_cost: float, cost_limit: float, accumulated_cost: float = None):
        self.estimated_cost = estimated_cost
        self.cost_limit = cost_limit
        self.accumulated_cost = accumulated_cost
        if accumulated_cost is not None:
            super().__init__(f"Estimated cost ${estimated_cost:.4f} would bring total to ${accumulated_cost + estimated_cost:.2f}, exceeding limit ${cost_limit:.2f}")
        else:
            super().__init__(f"Estimated cost ${estimated_cost:.4f} exceeds limit ${cost_limit:.2f}")


# Default API key environment variable per provider
PROVIDER_KEY_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}

# Reasoning effort to Anthropic thinking budget mapping
ANTHROPIC_THINKING_BUDGET = {
    # "minimal": 1000,
    "low": 8000,
    "medium": 16000,
    "high": 32000,
}


class LLMHandler:
    """Handler for LLM API interactions via litellm with multi-provider support"""

    # Model pricing (per 1M tokens) - fallback when litellm doesn't have pricing
    MODEL_PRICING = {
        # OpenAI GPT-5 series. gpt-5.5 surcharges 2x input / 1.5x output when
        # the prompt exceeds 272K tokens for the whole session; that band is
        # not modelled here (estimates would be off only in long-context runs).
        # GPT-5.6 family (Jul 2026). The bare `gpt-5.6` alias is deliberately
        # absent: it routes to -sol, so carrying it here offered one model under
        # two names and hid which tier was being bought. Name the tier.
        "gpt-5.6-sol": {"input": 5.0, "output": 30.0},
        # luna/terra/sol are the 5.6 tiers, 1:10:25 on input. Both of these were
        # entered too high (luna 5x, terra 1.25x) which made every Luna run look
        # 5x pricier than it is wherever this table is consulted. Verified
        # 2026-08-03 against developers.openai.com/api/docs/pricing, which
        # litellm's live map also matches.
        "gpt-5.6-terra": {"input": 2.0, "output": 12.0},
        "gpt-5.6-luna": {"input": 0.2, "output": 1.2},
        "gpt-5.5-pro": {"input": 30.0, "output": 180.0},
        "gpt-5.5": {"input": 5.0, "output": 30.0},
        "gpt-5.4": {"input": 2.5, "output": 15.0},
        "gpt-5.2": {"input": 1.75, "output": 14.0},
        "gpt-5.1": {"input": 1.25, "output": 10.0},
        "gpt-5": {"input": 1.25, "output": 10.0},
        "gpt-5.4-mini": {"input": 0.75, "output": 4.5},
        "gpt-5-mini": {"input": 0.25, "output": 2.0},
        "gpt-5-nano": {"input": 0.05, "output": 0.40},
        "gpt-5-pro": {"input": 15.0, "output": 120.0},
        # OpenAI GPT-4.1 series
        "gpt-4.1": {"input": 2.0, "output": 8.0},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
        # OpenAI GPT-4o series
        "gpt-4o": {"input": 2.5, "output": 10.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        # OpenAI o-series reasoning models
        "o1": {"input": 15.0, "output": 60.0},
        "o1-mini": {"input": 1.10, "output": 4.40},
        "o1-pro": {"input": 150.0, "output": 600.0},
        "o3": {"input": 2.0, "output": 8.0},
        "o3-mini": {"input": 1.10, "output": 4.40},
        "o4-mini": {"input": 1.10, "output": 4.40},
        # OpenAI Realtime models
        "gpt-realtime": {"input": 4.0, "output": 16.0},
        "gpt-realtime-mini": {"input": 0.60, "output": 2.40},
        # Anthropic Claude models (latest generation)
        "claude-opus-4-6": {"input": 5.0, "output": 25.0},
        "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
        "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
        # Google Gemini models (latest generation)
        "gemini-3.1-pro-preview": {"input": 2.0, "output": 12.0},
        "gemini-3-flash-preview": {"input": 0.50, "output": 3.0},
        "gemini-3.1-flash-lite-preview": {"input": 0.25, "output": 1.50},
    }

    MODEL_CONTEXT_SIZE = {
        # OpenAI GPT-5 series
        # GPT-5.6 family: 1.05M context across sol/terra/luna (per OpenAI docs).
        "gpt-5.6-sol": 1050000,
        "gpt-5.6-terra": 1050000,
        "gpt-5.6-luna": 1050000,
        "gpt-5.5-pro": 1050000,
        "gpt-5.5": 1050000,
        "gpt-5.4": 922000,
        "gpt-5.2": 272000,
        "gpt-5.1": 272000,
        "gpt-5": 272000,
        "gpt-5.4-mini": 272000,
        "gpt-5-mini": 272000,
        "gpt-5-nano": 272000,
        "gpt-5-pro": 272000,
        # OpenAI GPT-4.1 series
        "gpt-4.1": 1047576,
        "gpt-4.1-mini": 1047576,
        "gpt-4.1-nano": 1047576,
        # OpenAI GPT-4o series
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        # OpenAI o-series reasoning models
        "o1": 200000,
        "o1-mini": 128000,
        "o1-pro": 200000,
        "o3": 200000,
        "o3-mini": 200000,
        "o4-mini": 200000,
        # OpenAI Realtime models
        "gpt-realtime": 128000,
        "gpt-realtime-mini": 128000,
        # Anthropic Claude models (latest generation)
        "claude-opus-4-6": 1000000,
        "claude-sonnet-4-6": 1000000,
        "claude-haiku-4-5-20251001": 200000,
        # Google Gemini models (latest generation)
        "gemini-3.1-pro-preview": 1048576,
        "gemini-3-flash-preview": 1048576,
        "gemini-3.1-flash-lite-preview": 1048576,
    }

    # Models that support reasoning_effort or equivalent thinking parameters.
    # _is_reasoning_model also falls back to a `gpt-5*` prefix match so any
    # future GPT-5.x release (e.g. 5.6, 5.7) is auto-detected.
    REASONING_MODELS = {
        "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5-pro", "gpt-5.5", "gpt-5.4", "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-5.4-mini", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro", "o1", "o1-mini", "o1-pro", "o3", "o3-mini", "o4-mini",
    }

    # Unique identifier for chat completion requests
    USER_ID = "tzeentch-nietzsche"

    @classmethod
    def synthesis_models(cls) -> list[str]:
        """OpenAI current-generation text models offered as synthesis targets:
        the GPT-5.x family and the o-series, in MODEL_PRICING order. The legacy
        GPT-4.x line, audio/realtime models, and non-OpenAI providers are all
        excluded by construction, so there is no second list to keep in sync
        with MODEL_PRICING — add a new gpt-5.x/o-series entry there and it is
        offered automatically."""
        return [
            m for m in cls.MODEL_PRICING
            if (m.startswith("gpt-5") or re.match(r"o\d", m)) and "realtime" not in m
        ]

    def __init__(self, config: Dict = None):
        # Layer user config OVER DEFAULT_CONFIG['LLMHandler'] so direct
        # instantiation (e.g. image_generator's CLI path) gets sane defaults
        # for provider / base_url / key_name / etc. without forcing every
        # caller to pre-merge. Callers that pass a full config are unaffected.
        self.config = {**DEFAULT_CONFIG['LLMHandler'], **(config or {})}
        self.logger = get_logger()

        # _cost_lock guards _add_cost so concurrent callers (ImageGenerator's
        # IMAGE_GEN_WORKERS pool routing through this handler) don't lose
        # increments via interleaved read-modify-write on accumulated_cost.
        self.accumulated_cost = 0.0
        self.call_count = 0
        self.cost_history = []
        self._cost_lock = threading.Lock()

        # Set attributes from config
        set_attributes_from_config(self, self.config, DEFAULT_CONFIG['LLMHandler'].keys())

        # Resolve API key based on provider
        # Auto-detect key name if user didn't explicitly override it for a non-OpenAI provider
        if self.provider != "openai" and self.key_name == "OPENAI_API_KEY":
            env_key = PROVIDER_KEY_MAP.get(self.provider, "OPENAI_API_KEY")
        else:
            env_key = self.key_name or PROVIDER_KEY_MAP.get(self.provider, "OPENAI_API_KEY")
        self.api_key = self.config.get("api_key") or os.getenv(env_key)
        if not self.api_key:
            raise ValueError(f"API key not found in environment (looked for {env_key})")

        self.logger.info(f"LLM handler initialized: provider={self.provider}, model={self.model}")
        if self.cost_limit:
            self.logger.info(f"Cost limit enabled: ${self.cost_limit:.2f}")

    def _get_litellm_model(self, model: str = None) -> str:
        """Get litellm-formatted model string with provider prefix."""
        model = model or self.model
        if "/" in model:
            return model
        if self.provider == "openai":
            return model
        return f"{self.provider}/{model}"

    def _get_base_model(self, model: str = None) -> str:
        """Strip provider prefix to get base model name for lookups."""
        model = model or self.model
        return model.split("/", 1)[-1] if "/" in model else model

    def _is_reasoning_model(self, model: str = None) -> bool:
        """Check if model supports reasoning parameters.

        Falls back to a `gpt-5*` prefix match so future GPT-5.x releases
        (5.5, 5.6, mini/pro variants…) are auto-detected as reasoning
        models without a code change. Without this, a new model bumped
        in a preset YAML fails synthesis with "Unsupported value:
        'temperature'" because the temperature-strip path at
        _build_kwargs (~line 490) gates on this check.
        """
        model = self._get_base_model(model)
        return model in self.REASONING_MODELS or model.startswith("gpt-5")

    def _get_max_input_tokens(self) -> int:
        """Get max input tokens."""
        return max(self.max_input_tokens or (self._get_model_context_length() - self.max_completion_tokens), 0)

    @lru_cache(maxsize=1)
    def _get_model_context_length(self) -> int:
        """Get context length for the current model."""
        try:
            return litellm.get_max_tokens(self._get_litellm_model())
        except Exception:
            pass
        base_model = self._get_base_model()
        if base_model in self.MODEL_CONTEXT_SIZE:
            return self.MODEL_CONTEXT_SIZE[base_model]
        self.logger.error(f"Unknown model {self.model}, using 128000 as fallback context size")
        return 128000

    def _count_tokens(self, messages: List[Dict], precise: bool = False) -> int:
        """Count tokens in messages with optional precision."""
        if not precise:
            total_chars = sum(len(str(msg.get('content', ''))) + len(str(msg.get('role', ''))) + 10 for msg in messages)
            return total_chars // self.chars_per_token

        try:
            return litellm.token_counter(model=self._get_litellm_model(), messages=messages)
        except Exception:
            total_chars = sum(len(str(msg.get('content', ''))) + len(str(msg.get('role', ''))) + 10 for msg in messages)
            return total_chars // self.chars_per_token

    def _should_use_precise_counting(self, messages: List[Dict]) -> bool:
        """Decide if precise token counting is needed."""
        if not self.manage_context:
            return False
        fast_estimate = self._count_tokens(messages, precise=False)
        threshold = self._get_max_input_tokens() * self.token_count_thres
        return fast_estimate > threshold

    def _prepare_messages(self, messages: List[Dict]) -> List[Dict]:
        """Prepare messages with smart context management and LLMLingua-2 compression."""
        if not self._should_use_precise_counting(messages):
            return messages

        max_input = self._get_max_input_tokens()
        if self._count_tokens(messages, precise=True) <= max_input:
            return messages

        self._init_compressor()

        system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
        other_msgs = messages[1:] if system_msg else messages

        result = [system_msg] if system_msg else []
        current_tokens = self._count_tokens(result, precise=True) if system_msg else 0

        temp_msgs, compressed_count = self._fit_messages_with_compression(
            reversed(other_msgs), max_input, current_tokens
        )

        result.extend(reversed(temp_msgs))
        self._log_context_changes(len(messages) - len(result), compressed_count)
        return result

    def _init_compressor(self):
        """Initialize LLMLingua-2 compressor (lazy loading)."""
        if hasattr(self, 'compressor'):
            return
        try:
            from llmlingua import PromptCompressor
            self.compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                use_llmlingua2=True,
                device_map="cpu"
            )
            self.logger.info("LLMLingua-2 compressor initialized")
        except ImportError:
            self.logger.error("Text compression unavailable, please install: 'pip install llmlingua'")
            self.compressor = None

    def _fit_messages_with_compression(self, messages, max_input, current_tokens):
        """Fit messages using compression when needed."""
        temp_msgs, compressed_count = [], 0

        for msg in messages:
            msg_tokens = self._count_tokens([msg], precise=True)

            if current_tokens + msg_tokens <= max_input:
                temp_msgs.append(msg)
                current_tokens += msg_tokens
            elif compressed_msg := self._try_compress_message(msg, max_input - current_tokens, msg_tokens):
                temp_msgs.append(compressed_msg)
                current_tokens += self._count_tokens([compressed_msg], precise=True)
                compressed_count += 1
            else:
                break

        return temp_msgs, compressed_count

    def _try_compress_message(self, msg, remaining_tokens, msg_tokens):
        """Try to compress a message to fit in remaining token budget."""
        if not (self.compressor and msg.get('content')):
            return None

        compression_rate = min(0.9, remaining_tokens / msg_tokens)
        if compression_rate <= 0.1:
            return None

        result = self.compressor.compress_prompt(
            msg['content'],
            rate=compression_rate,
            force_tokens=['\n', '?', '!', '.', ',']
        )
        compressed_msg = msg.copy()
        compressed_msg['content'] = result['compressed_prompt']

        if self._count_tokens([compressed_msg], precise=True) <= remaining_tokens:
            self.logger.debug(f"Compressed: {msg_tokens}->{self._count_tokens([compressed_msg], precise=True)} ({result.get('ratio', 'N/A')})")
            return compressed_msg

        return None

    def _log_context_changes(self, truncated, compressed_count):
        """Log context management changes."""
        if truncated or compressed_count:
            parts = []
            if compressed_count:
                parts.append(f"compressed {compressed_count}")
            if truncated:
                parts.append(f"truncated {truncated}")
            self.logger.info(f"Context management: {', '.join(parts)} messages")

    def _get_model_pricing(self, model: str = None) -> Dict[str, float]:
        """Get pricing for a model."""
        model = self._get_base_model(model)

        if model in self.MODEL_PRICING:
            return self.MODEL_PRICING[model]

        # Try litellm's pricing database
        try:
            litellm_model = self._get_litellm_model(model)
            input_cost, output_cost = litellm.cost_per_token(model=litellm_model, prompt_tokens=1, completion_tokens=1)
            return {"input": input_cost * 1_000_000, "output": output_cost * 1_000_000}
        except Exception:
            pass

        self.logger.error(f"Unknown model {model}, using gpt-4o pricing as fallback")
        return self.MODEL_PRICING["gpt-4o"]

    def _add_cost(self, actual_cost: float, input_tokens: int = 0, output_tokens: int = 0):
        """Add actual cost to accumulated total and update tracking."""
        import time

        with self._cost_lock:
            self.accumulated_cost += actual_cost
            self.call_count += 1
            self.cost_history.append({
                'timestamp': time.time(),
                'cost': actual_cost,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
                'accumulated_cost': self.accumulated_cost
            })

    def _check_and_log_cost(self, messages: List[Dict], max_completion_tokens: int, model: str):
        """Check cost limit including accumulated costs."""
        if not self.cost_limit:
            return

        input_tokens = self._count_tokens(messages, precise=True)
        estimated_cost = self._estimate_cost(input_tokens, max_completion_tokens, model)
        total_projected_cost = self.accumulated_cost + estimated_cost

        if total_projected_cost > self.cost_limit:
            raise CostLimitExceededException(estimated_cost, self.cost_limit, self.accumulated_cost)

    def _log_messages_with_multiline_support(self, messages):
        """Log messages with proper multiline string formatting."""
        self.logger.debug("Request messages:")

        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content') or ''

            self.logger.debug(f"[{i}] {role}:")

            if '\n' in content:
                for line in content.split('\n'):
                    self.logger.debug(f"    {line}")
            else:
                self.logger.debug(f"    {content}")

            if i < len(messages) - 1:
                self.logger.debug("----------")

    def _estimate_cost(self, input_tokens: int, output_tokens: int, model: str = None) -> float:
        """Estimate cost based on token usage (does not modify accumulated cost)."""
        pricing = self._get_model_pricing(model)
        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

    def _map_reasoning_params(self, kwargs: Dict) -> Dict:
        """Map reasoning_effort to provider-specific parameters."""
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        if not reasoning_effort:
            return kwargs

        if self.provider == "openai":
            if reasoning_effort == "minimal":
                reasoning_effort = "low"
            kwargs["reasoning_effort"] = reasoning_effort
        elif self.provider == "anthropic":
            budget = ANTHROPIC_THINKING_BUDGET.get(reasoning_effort,
                ANTHROPIC_THINKING_BUDGET['low'])
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            # max_completion_tokens is total (thinking + visible output), must exceed budget
            current_max = kwargs.get("max_completion_tokens") or self.max_completion_tokens
            if current_max <= budget:
                kwargs["max_completion_tokens"] = budget + current_max
        elif self.provider == "gemini":
            budget = ANTHROPIC_THINKING_BUDGET.get(reasoning_effort,
                ANTHROPIC_THINKING_BUDGET['low'])
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            current_max = kwargs.get("max_completion_tokens") or self.max_completion_tokens
            if current_max <= budget:
                kwargs["max_completion_tokens"] = budget + current_max

        # Remove temperature for reasoning models (most providers don't support it)
        if "temperature" in kwargs:
            del kwargs["temperature"]

        return kwargs

    # Public methods

    def get_model_pricing(self, model: str = None) -> Dict[str, float]:
        """Get pricing for a model (public interface)."""
        return self._get_model_pricing(model)

    def reset_cost_tracking(self):
        """Reset cost tracking to zero."""
        self.accumulated_cost = 0.0
        self.call_count = 0
        self.cost_history.clear()
        self.logger.info("Cost tracking reset to zero")

    def report_external_cost(self, cost_usd: float, model: str = "external") -> None:
        """Track a non-chat-completion cost (e.g. image generation) in the
        same accumulated_cost bucket. Caller computes USD from a known price
        table since image-gen responses don't carry token usage."""
        if cost_usd <= 0:
            return
        self._add_cost(cost_usd, input_tokens=0, output_tokens=0)
        self.logger.debug(f"External cost: ${cost_usd:.4f} ({model}), "
                          f"Total: ${self.accumulated_cost:.2f}")

    def chat_completion(self, prompt: str,
        system_message: str = None,
        override_config: Dict = None,
        response_format: Dict = None,
        conversation_history: List[Dict] = None,
        num_retries: Optional[int] = None,
        **kwargs) -> str:
        """Chat completion with cost limiting and context management."""

        # Build messages
        messages = []
        if system_message or self.system_message:
            messages.append({"role": "system", "content": system_message or self.system_message})
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": prompt})

        # Apply context management
        messages = self._prepare_messages(messages)

        # Build API parameters
        kwargs = {
            "model": self._get_litellm_model(),
            "messages": messages,
            "top_p": self.top_p,
            "max_completion_tokens": self.max_completion_tokens,
            "temperature": self.temperature,
            "user": self.USER_ID,
            "api_key": self.api_key,
        }

        # Apply override config (may contain bare model names)
        if override_config:
            kwargs.update(override_config)
            if "model" in override_config:
                kwargs["model"] = self._get_litellm_model(override_config["model"])

        # Pass custom base_url only if explicitly configured to non-default
        if self.base_url and self.base_url != "https://api.openai.com/v1":
            kwargs["api_base"] = self.base_url

        if self.seed:
            kwargs["seed"] = self.seed
        if response_format:
            kwargs["response_format"] = response_format

        # Handle reasoning parameters
        # For OpenAI: only apply reasoning_effort to known reasoning models
        # For other providers: apply if set (they handle thinking params differently)
        is_reasoning = self._is_reasoning_model(kwargs['model']) or self.provider != "openai"
        if self.reasoning_effort and "reasoning_effort" not in kwargs and is_reasoning:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if "reasoning_effort" in kwargs and not is_reasoning:
            del kwargs["reasoning_effort"]
        if "reasoning_effort" in kwargs:
            kwargs = self._map_reasoning_params(kwargs)
        elif self._is_reasoning_model(kwargs['model']):
            # OpenAI reasoning models don't support temperature even without reasoning_effort
            kwargs.pop("temperature", None)

        # Check cost limit (including accumulated costs)
        self._check_and_log_cost(messages, self.max_completion_tokens, kwargs["model"])

        # Log request
        self.logger.debug(f"LLM API request parameters: {json.dumps({k: v for k, v in kwargs.items() if k not in ('messages', 'api_key')}, indent=4)}")
        self._log_messages_with_multiline_support(messages)

        # Add retry/timeout.
        #
        # num_retries is the load-bearing retry knob: litellm/main.py:1355-1356
        # does `if num_retries is not None: max_retries = num_retries` and
        # forwards the result into openai.OpenAI(max_retries=N), which spawns
        # up to N invisible retries on httpx timeouts. Default comes from
        # self.max_retries (config default 2 — matches openai-python's own
        # default and is appropriate for short transient failures). Callers
        # owning their own retry strategy (e.g. artifact_synthesis._llm_call
        # with its reasoning_effort step-down) pass num_retries=0 explicitly
        # so timeouts surface immediately to the wrapper instead of being
        # silently amplified into multi-hour stalls.
        effective_retries = num_retries if num_retries is not None else self.max_retries
        kwargs.setdefault("num_retries", effective_retries)
        kwargs.setdefault("timeout", self.timeout)

        # Make API call via litellm
        try:
            response = litellm.completion(**kwargs)
        except litellm.AuthenticationError as e:
            self.logger.error(f"LLM auth error (fatal): {type(e).__name__}")
            raise FatalLLMError(f"Authentication failed: {e}") from e
        except litellm.RateLimitError as e:
            # OpenAI 429 covers both rate-limit and insufficient_quota — classify via code/message
            code = getattr(e, 'code', None) or ''
            msg = str(e).lower()
            if code == 'insufficient_quota' or 'insufficient_quota' in msg or 'billing' in msg:
                self.logger.error(f"LLM quota exhausted (fatal): {type(e).__name__}")
                raise FatalLLMError(f"Insufficient quota / billing issue: {e}") from e
            self.logger.error(f"LLM rate limit (transient): {type(e).__name__}")
            raise
        except litellm.APIError as e:
            if "maximum context length" in str(e).lower():
                self.logger.error("Context length exceeded")
            else:
                self.logger.error(f"LLM API error: {type(e).__name__}")
            raise

        content = response.choices[0].message.content
        content = content.strip() if content else ""

        # Log usage and cost
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage

            # Try litellm's cost calculation first, fall back to manual estimation
            try:
                actual_cost = litellm.completion_cost(completion_response=response)
            except Exception:
                actual_cost = self._estimate_cost(usage.prompt_tokens, usage.completion_tokens, kwargs["model"])

            self._add_cost(actual_cost, usage.prompt_tokens, usage.completion_tokens)

            pricing = self._get_model_pricing(kwargs["model"])
            input_cost = (usage.prompt_tokens * pricing["input"]) / 1_000_000
            output_cost = (usage.completion_tokens * pricing["output"]) / 1_000_000

            self.logger.debug(f"Tokens: {usage.prompt_tokens}->{usage.completion_tokens}, Sum: {usage.total_tokens}")
            self.logger.debug(f"Cost: ${actual_cost:.4f} ({input_cost:.4f}->{output_cost:.4f}), Total: ${self.accumulated_cost:.2f}/{f'${self.cost_limit:.2f}' if self.cost_limit else 'unlimited'}")

        self.logger.debug(f"Response: {content}")
        return content

    def completion(self,
                   messages: List[Dict],
                   *,
                   model: Optional[str] = None,
                   temperature: Optional[float] = None,
                   max_completion_tokens: Optional[int] = None,
                   num_retries: Optional[int] = None,
                   **kwargs) -> Any:
        """Pre-built-messages completion (text or multimodal). Adds api_key,
        retry/timeout, cost tracking, and auth/quota error classification on
        top of litellm.completion; skips _prepare_messages (LLMLingua can't
        compress image_url blocks). Returns the raw litellm response."""
        resolved_model = self._get_litellm_model(model or self.model)
        kwargs.update(
            model=resolved_model,
            messages=messages,
            # `is not None` (vs `or`) so an explicit 0 wouldn't fall back to
            # self.max_completion_tokens — no current caller passes 0, but it's
            # cheap defense against a future caller wanting unbounded output.
            max_completion_tokens=(max_completion_tokens
                                   if max_completion_tokens is not None
                                   else self.max_completion_tokens),
            api_key=self.api_key,
            user=self.USER_ID,
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        elif "temperature" not in kwargs and self.temperature is not None:
            kwargs["temperature"] = self.temperature
        kwargs.setdefault("num_retries",
                          num_retries if num_retries is not None else self.max_retries)
        kwargs.setdefault("timeout", self.timeout)
        # Reasoning models reject `temperature` — strip it (matches chat_completion).
        if self._is_reasoning_model(resolved_model):
            kwargs.pop("temperature", None)

        # Pre-check the projected cost against cost_limit so we never silently
        # blow past the cap mid-batch (parity with chat_completion). Cheap:
        # _count_tokens caches and litellm.token_counter handles multimodal.
        self._check_and_log_cost(messages, kwargs["max_completion_tokens"],
                                 resolved_model)

        try:
            response = litellm.completion(**kwargs)
        except litellm.AuthenticationError as e:
            self.logger.error(f"LLM auth error (fatal): {type(e).__name__}")
            raise FatalLLMError(f"Authentication failed: {e}") from e
        except litellm.RateLimitError as e:
            code = getattr(e, 'code', None) or ''
            msg = str(e).lower()
            if code == 'insufficient_quota' or 'insufficient_quota' in msg or 'billing' in msg:
                self.logger.error(f"LLM quota exhausted (fatal): {type(e).__name__}")
                raise FatalLLMError(f"Insufficient quota / billing issue: {e}") from e
            raise

        if hasattr(response, "usage") and response.usage:
            try:
                actual_cost = litellm.completion_cost(completion_response=response)
            except Exception:
                actual_cost = self._estimate_cost(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    resolved_model)
            self._add_cost(actual_cost,
                           response.usage.prompt_tokens,
                           response.usage.completion_tokens)
        return response

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get comprehensive cost summary including accumulated costs."""
        return {
            "cost_limit": self.cost_limit,
            "accumulated_cost": self.accumulated_cost,
            "remaining_budget": self.cost_limit - self.accumulated_cost if self.cost_limit else None,
            "call_count": self.call_count,
            "average_cost_per_call": self.accumulated_cost / self.call_count if self.call_count > 0 else 0.0,
            "provider": self.provider,
            "model": self.model,
            "model_pricing": self._get_model_pricing(),
            "pricing_per_1m_tokens": True,
            "cost_history": self.cost_history[-10:]
        }
