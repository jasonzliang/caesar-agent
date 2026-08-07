# action_selector.py - Enhanced with backoff decorator
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
import time
import functools
from typing import Dict, List, Tuple, Optional

from .logger import get_logger
from .config import set_attributes_from_config, check_attrs


class ActionSelectorError(Exception):
    """Custom exception for action selection errors"""
    pass


class ActionTooFrequentError(Exception):
    """Exception raised when max backoff attempts exceeded"""
    def __init__(self, action_name: str, attempts: int, max_tries: int):
        self.action_name = action_name
        self.attempts = attempts
        self.max_tries = max_tries
        super().__init__(f"Action '{action_name}' blocked after {attempts} backoff attempts (max: {max_tries})")


def action_frequency_backoff(func=None):
    """
    Compact backoff: doubles wait time on retry, raises exception after max_tries
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(self, agent, iteration: int, stop_on_error: bool = True):
            # Initialize tracking
            if not hasattr(self, 'last_execution'):
                self.last_execution = {}
            if not hasattr(self, 'skip_next_check'):
                self.skip_next_check = set()
            if not hasattr(self, 'backoff_attempts'):
                self.backoff_attempts = {}

            if not getattr(self, 'backoff_enabled', True):
                return f(self, agent, iteration, stop_on_error)

            # Get action result
            result = f(self, agent, iteration, stop_on_error)
            if not result or len(result) != 3:
                return result

            action_name, reason, should_continue = result
            if not action_name or not should_continue:
                return result

            # Skip check if action just completed backoff
            if action_name in self.skip_next_check:
                self.skip_next_check.remove(action_name)
                self.last_execution[action_name] = time.time()
                self.logger.debug(f"Skipping frequency check for {action_name} (post-backoff)")
                return result

            # Check frequency
            now = time.time()
            last_time = self.last_execution.get(action_name, 0)
            elapsed = now - last_time

            if elapsed < self.min_interval:
                # Track backoff attempts
                attempts = self.backoff_attempts.get(action_name, 0) + 1
                self.backoff_attempts[action_name] = attempts

                # Check if max tries exceeded
                if attempts > self.max_tries:
                    self.logger.error(f"Action {action_name} blocked after {attempts} attempts")
                    raise ActionTooFrequentError(action_name, attempts, self.max_tries)

                # Calculate exponentially increasing wait time: min_interval * 2^attempts
                wait_time = self.min_interval * (2 ** (attempts - 1))

                self.logger.debug(f"Action {action_name} backing off {wait_time:.1f}s "
                                f"(attempt {attempts}/{self.max_tries})")

                # Sleep and mark to skip next check
                time.sleep(wait_time)
                self.skip_next_check.add(action_name)
            else:
                # Reset attempts counter on successful execution within interval
                self.backoff_attempts[action_name] = 0

            # Record execution time
            self.last_execution[action_name] = time.time()
            return result

        return wrapper

    return decorator if func is None else decorator(func)


class ActionSelectorBase(ABC):
    """Abstract base class for action selection strategies"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = get_logger()
        set_attributes_from_config(self, self.config)

    @abstractmethod
    def get_action_selection_prompt(self, fsm, agent) -> str:
        """Generate action selection prompt based on current FSM state and agent context"""
        pass

    def _extract_action_from_response(self, agent, response: str) -> tuple:
        """Extract the action name from the LLM response"""
        reasoning = "No reasoning provided"

        # Try to parse as JSON first
        parsed_json = agent.parse_json_response(response)

        if parsed_json and 'selected_action' in parsed_json:
            action = parsed_json['selected_action']
            reasoning = parsed_json.get('reasoning', 'No reasoning provided')
            self.logger.info(f"Selected action: {action} - {reasoning}")
            return action, reasoning

        # Fallback: look for action names in the response text
        available_actions = agent.fsm.get_available_actions()
        for action in available_actions:
            if action.lower() in response.lower():
                self.logger.info(f"Extracted action from text: {action}")
                return action, reasoning

        # If no action found, log the issue and return None
        self.logger.error(f"Could not extract valid action from response: {response}")
        raise ValueError(f"Could not extract valid action from response: {response}")

    def select_action(self, agent, iteration: int, stop_on_error: bool = True) -> Tuple[str, str, bool]:
        """Select an action with frequency-based exponential backoff"""

        def handle_error(msg: str, exception: Exception = None) -> Tuple[str, str, bool]:
            """Unified error handling with reset logic"""
            self.logger.error(msg)
            agent.history.add_error(iteration, msg, agent.fsm.current_state, str(exception) if exception else None)

            if stop_on_error:
                raise ActionSelectorError(msg) from exception

            agent.fsm.reset(agent)
            agent.history.reset()
            return None, f"{msg} - reset and continue", True

        # Check available actions
        available_actions = agent.fsm.get_available_actions()
        if not available_actions:
            return handle_error("No available actions in current state")

        self.logger.info(f"Available actions from {agent.fsm.get_current_state()}: {available_actions}")

        # Auto-select if only one action
        if len(available_actions) == 1:
            chosen_action = available_actions[0]
            self.logger.info(f"Auto-selecting single available action: {chosen_action}")
            return chosen_action, "only one action available - auto-selected", True

        # Multi-action selection via LLM
        try:
            prompt = self.get_action_selection_prompt(agent.fsm, agent)
            response = agent.chat_completion(
                prompt=prompt,
                system_message=agent.role,
                response_format={"type": "json_object"}
            )
            chosen_action, reasoning = self._extract_action_from_response(agent, response)

        except Exception as e:
            return handle_error(f"LLM call failed during action selection: {str(e)}", e)

        # Validate action selection
        if not chosen_action:
            return handle_error(f"Could not determine action from LLM response: {response}")

        if chosen_action not in available_actions:
            return handle_error(
                f"Action '{chosen_action}' not available in state '{agent.fsm.current_state}'. "
                f"Available: {available_actions}"
            )

        return chosen_action, reasoning, True


class OriginalActionSelector(ActionSelectorBase):
    """Original action selection method - preserves existing behavior"""

    def get_action_selection_prompt(self, fsm, agent) -> str:
        """Original action selection prompt logic"""
        if fsm.current_state not in fsm.states:
            raise ValueError(f"Current state '{fsm.current_state}' is not a valid state in the FSM")

        state_summary = f"{fsm.current_state}: {fsm.states[fsm.current_state].summary(agent)}"
        available_actions = fsm.get_available_actions()

        # Build action list with summaries and transition information
        action_details = []
        future_states = set()  # Track unique future states for summaries

        for action_name in available_actions:
            assert action_name in fsm.actions
            action_summary = fsm.actions[action_name].summary(agent)

            # Get transition information
            assert fsm.current_state in fsm.transitions and action_name in fsm.transitions[fsm.current_state]
            target_state, fallback_state = fsm.transitions[fsm.current_state][action_name]
            transition_info = f"next state: {target_state}"
            if fallback_state:
                transition_info += f", fallback: {fallback_state}"
                future_states.add(fallback_state)
            future_states.add(target_state)

            action_details.append(f"- {action_name} ({transition_info}): {action_summary}")

        actions_text = "\n".join(action_details)

        # Build future states summaries, sort for consistent ordering
        future_summaries = []
        for state_name in sorted(future_states):
            assert state_name in fsm.states
            future_summary = fsm.states[state_name].future_summary(agent)
            future_summaries.append(f"- {state_name}: {future_summary}")

        future_states_text = "\n".join(future_summaries) if future_summaries else "No future states to display"

        # Get history summary based on config
        history_summary = agent.history.get_history_summary(agent.history_context_len)

        prompt_parts = []

        # Add overview if it's set
        if fsm.overview:
            prompt_parts.append(f"## FSM Overview ##\n{fsm.overview}")

        # Add other sections
        if history_summary:
            prompt_parts.append(f"## Recent agent history ##\n{history_summary}")

        prompt_parts.extend([
            f"## Current state ##\n{state_summary}",
            f"## Available actions ##\n{actions_text}",
            f"## Future states ##\n{future_states_text}"
        ])

        prompt_parts.append(
"""Please select one of the available actions to execute. Respond with a JSON object containing:
{
    "initial_action": "initial considered action name",
    "selected_action": "final selected action name",
    "reasoning": "Short reason of why you choose this final action and why it is different from initial action"
}"""
        )

        prompt_parts.append(f"IMPORTANT:\nChoose the most appropriate action using your role as a guide. First consider an initial action and if that action (and associated state) is getting repeated {agent.patience} times or more in recent history, try selecting an alternative action to avoid getting stuck. Diversity in action selection often leads to better outcomes. Please mention in reasoning if avoiding being stuck influenced the final action that was selected.")

        return "\n\n".join(prompt_parts)


class SmartActionSelector(ActionSelectorBase):
    """Compact, efficient action selection with intelligent loop detection"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        # Single cache for pattern analysis
        self._cache = {"iteration": -1, "analysis": {}}

    def get_action_selection_prompt(self, fsm, agent) -> str:
        """Generate enhanced prompt with context-aware intervention"""
        if fsm.current_state not in fsm.states:
            raise ValueError(f"Current state '{fsm.current_state}' not in FSM states")

        # Build base prompt efficiently
        prompt_parts = self._build_prompt_parts(fsm, agent)

        # Add intervention guidance if needed
        analysis = self._get_pattern_analysis(agent, fsm.get_available_actions())
        if analysis["needs_intervention"]:
            prompt_parts.append(self._build_intervention_prompt(analysis))
        else:
            prompt_parts.append(self._build_base_selection_prompt())

        return "\n\n".join(prompt_parts)

    def _build_prompt_parts(self, fsm, agent) -> List[str]:
        """Build core prompt components efficiently"""
        current_state = fsm.current_state
        state_obj = fsm.states[current_state]
        available_actions = fsm.get_available_actions()

        # Build action details with state transitions
        action_lines = []
        future_states = set()

        for action in available_actions:
            target, fallback = fsm.transitions[current_state][action]
            transition = f"next state: {target}"
            if fallback:
                transition += f", fallback: {fallback}"
                future_states.add(fallback)
            future_states.add(target)

            summary = fsm.actions[action].summary(agent)
            action_lines.append(f"- {action} ({transition}): {summary}")

        # Build future state summaries
        future_lines = [f"- {name}: {fsm.states[name].future_summary(agent)}"
                       for name in sorted(future_states)]

        # Assemble prompt sections
        parts = []

        if fsm.overview:
            parts.append(f"## FSM Overview ##\n{fsm.overview}")

        history = agent.history.get_history_summary(agent.history_context_len)
        if history:
            parts.append(f"## Recent agent history ##\n{history}")

        parts.extend([
            f"## Current state ##\n{current_state}: {state_obj.summary(agent)}",
            f"## Available actions ##\n" + "\n".join(action_lines),
            f"## Future states ##\n" + ("\n".join(future_lines) if future_lines else "No future states")
        ])

        return parts

    def _get_pattern_analysis(self, agent, available_actions: List[str]) -> Dict:
        """Cached, efficient pattern analysis"""
        # Return cached result if same iteration
        if self._cache["iteration"] == agent.curr_iteration:
            return self._cache["analysis"]

        # Compute new analysis
        window_size = self.loop_detection_window or agent.history_context_len
        recent_actions = agent.history.get_recent_actions(window_size)

        if len(recent_actions) < 3:
            analysis = {"needs_intervention": False, "severity": "none", "message": ""}
        else:
            analysis = self._analyze_patterns(recent_actions, available_actions,
                                            agent.fsm.current_state, agent.patience)

        # Cache and return
        self._cache = {"iteration": agent.curr_iteration, "analysis": analysis}
        return analysis

    def _analyze_patterns(self, recent_actions: List, available_actions: List[str],
                         current_state: str, patience: int) -> Dict:
        """Compact pattern analysis with both loop and cycle detection"""
        # Count state-action patterns efficiently
        counts = defaultdict(int)
        states = []

        for record in recent_actions:
            action, state = self._extract_action_state(record, current_state)
            if action and state:
                counts[f"{state}->{action}"] += 1
                states.append(state)

        # 1. Check for simple loops (existing logic)
        current_patterns = {
            action: counts.get(f"{current_state}->{action}", 0)
            for action in available_actions
        }
        max_reps = max(current_patterns.values()) if current_patterns else 0

        # 2. Check for cycles (new logic) - only if no simple loop detected
        if max_reps < patience and len(states) >= 6:
            cycle_info = self._detect_cycle(states)
            if cycle_info["detected"]:
                return {
                    "needs_intervention": True,
                    "severity": "critical" if cycle_info["repetitions"] >= 3 else "warning",
                    "message": f"🔄 CYCLE: {' → '.join(cycle_info['pattern'])} repeated {cycle_info['repetitions']} times. Try a different approach.",
                    "max_repetitions": cycle_info["repetitions"]
                }

        # 3. Fall back to simple loop logic
        if max_reps >= patience:
            return self._create_intervention("critical", max_reps, patience,
                                           current_patterns, available_actions)
        elif max_reps >= patience // 2 and len(available_actions) > 1:
            return self._create_intervention("mild", max_reps, patience,
                                           current_patterns, available_actions)
        else:
            return {"needs_intervention": False, "severity": "none", "message": ""}

    def _detect_cycle(self, states: List[str]) -> Dict:
        """Compact cycle detection using pattern matching"""
        if len(states) < 6:  # Need at least 2 cycles of length 3
            return {"detected": False}

        # Check cycle lengths 3-8 (covers typical FSM cycles)
        for cycle_len in range(3, min(9, len(states) // 2 + 1)):
            recent = states[-cycle_len:]
            prev = states[-cycle_len*2:-cycle_len] if len(states) >= cycle_len*2 else []

            if recent == prev:  # Found a repeating pattern
                # Count total repetitions by looking backwards
                reps = 2  # We already found 2
                pos = len(states) - cycle_len * 3
                while pos >= 0 and states[pos:pos + cycle_len] == recent:
                    reps += 1
                    pos -= cycle_len

                if reps >= 2:  # At least 2 complete cycles
                    return {
                        "detected": True,
                        "pattern": recent,
                        "repetitions": reps,
                        "length": cycle_len
                    }

        return {"detected": False}

    def _extract_action_state(self, record, fallback_state: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract action and state from history record"""
        if isinstance(record, dict):
            action = record.get('action')
            state = record.get('prev_state') or record.get('state') or (fallback_state if action else None)
        else:
            action, state = record, fallback_state
        return action, state

    def _create_intervention(self, severity: str, max_reps: int, patience: int,
                           patterns: Dict[str, int], available_actions: List[str]) -> Dict:
        """Create intervention message based on severity"""
        if severity == "critical":
            overused = [a for a, c in patterns.items() if c >= patience]
            alternatives = [a for a in available_actions if patterns.get(a, 0) <= 1]
            message = (f"🚨 CRITICAL: Breaking repetitive loop!\n"
                      f"Overused ({max_reps}+ times): {', '.join(overused)}\n"
                      f"Try alternatives: {', '.join(alternatives) if alternatives else 'other actions'}")

        elif severity == "warning":
            at_limit = [a for a, c in patterns.items() if c >= patience]
            alternatives = [a for a in available_actions if patterns.get(a, 0) < patience]
            message = (f"⚠️ WARNING: Patience threshold reached\n"
                      f"At limit ({patience}): {', '.join(at_limit)}\n"
                      f"Consider: {', '.join(alternatives) if alternatives else 'other approaches'}")

        else:  # mild
            most_used = max(patterns.items(), key=lambda x: x[1]) if patterns else ("none", 0)
            less_used = [a for a in available_actions if patterns.get(a, 0) <= most_used[1] // 2]
            message = (f"💡 SUGGESTION: Consider exploring alternatives\n"
                      f"Most used: {most_used[0]} ({most_used[1]} times)\n"
                      f"Less explored: {', '.join(less_used) if less_used else 'other options'}")

        return {
            "needs_intervention": True,
            "severity": severity,
            "message": message,
            "max_repetitions": max_reps
        }

    def _build_intervention_prompt(self, analysis: Dict) -> str:
        """Build prompt with intervention guidance"""
        base = ('Please select one action. Respond with JSON:\n'
               '{"selected_action": "action_name", "reasoning": "brief_reason"}\n'
               'IMPORTANT: You must select an action from "## Available actions ##", even if all of them are overused.')

        severity = analysis["severity"]
        message = analysis["message"]

        if severity == "critical":
            guidance = f"\n\nCRITICAL INTERVENTION:\n{message}\nSTRONGLY recommended to select a different action."
        elif severity == "warning":
            guidance = f"\n\nPATIENCE THRESHOLD:\n{message}\nConsider alternatives if current approach isn't working."
        else:  # mild
            guidance = f"\n\nDIVERSITY SUGGESTION:\n{message}\nVariety often improves outcomes."

        return base + guidance

    def _build_base_selection_prompt(self) -> str:
        """Build standard selection prompt without intervention"""
        base = ('Please select one action. Respond with JSON:\n'
               '{"selected_action": "action_name", "reasoning": "brief_reason"}')

        if self.exploration_rate > 0.1:
            return base + "\n\nChoose the most appropriate action. Occasional exploration can improve outcomes."
        else:
            return base + "\n\nChoose the most appropriate action for your role."


class ActionSelector:
    """Factory class for action selection strategies with backoff support"""
    SELECTORS = {
        "original": OriginalActionSelector,
        "smart": SmartActionSelector
    }

    def __init__(self, strategy: str = "original", config: Dict = None):
        """Initialize action selection with specified strategy"""
        self.strategy = strategy
        self.config = config or {}
        self.logger = get_logger()

        set_attributes_from_config(self, self.config, ['min_interval', 'max_tries', 'backoff_enabled'])

        # Run selector constructor
        if strategy not in self.SELECTORS:
            raise ValueError(f"Unknown action selection strategy: {strategy}. Available: {list(self.SELECTORS.keys())}")

        strategy_config = self.config.get(strategy, {})
        self.selector = self.SELECTORS[self.strategy](strategy_config)

        self.logger.info(f"Initialized {strategy} action selection strategy with backoff "
            f"enabled: {self.backoff_enabled}")

    def get_action_selection_prompt(self, fsm, agent) -> str:
        """Generate action selection prompt using the configured strategy"""
        return self.selector.get_action_selection_prompt(fsm, agent)

    @action_frequency_backoff()
    def select_action(self, agent, iteration: int, stop_on_error: bool = True) -> Tuple[str, str, bool]:
        """Select an action using the configured strategy with automatic backoff"""
        return self.selector.select_action(agent, iteration, stop_on_error)
