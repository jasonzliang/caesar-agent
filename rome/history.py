# history.py
from typing import Dict, List, Optional, Any

from .logger import get_logger
from .state import truncate_text

class AgentHistory:
    """Manages agent execution history and provides summary functionality"""

    def __init__(self):
        self.iteration = 0
        self.actions_executed: List[Dict] = []
        self.states_visited: List[str] = []
        self.errors: List[Dict] = []
        self.final_state: Optional[str] = None
        self.final_context: Optional[Dict] = None
        self.logger = get_logger()

    def add_initial_state(self, state: str):
        """Add the initial state to the history"""
        if not self.states_visited:
            self.states_visited.append(state)
            self.logger.info(f"Added initial state to history: {state}")

    def add_action_execution(self, iteration: int, action: str, action_result: bool,
                           action_reason: str, prev_state: str, curr_state: str):
        """Record an action execution"""
        action_result = "success" if action_result else "failure"

        action_record = {
            'iteration': iteration,
            'action': action,
            'action_result': action_result,
            'action_reason': action_reason,
            'prev_state': prev_state,
            'curr_state': curr_state,
        }

        self.actions_executed.append(action_record)
        self.states_visited.append(curr_state)
        self.logger.info(f"Recorded action execution: {action} ({action_result})")

    def add_error(self, iteration: int, error_msg: str, state: str, exception: str = None):
        """Record an error that occurred during execution"""
        error_record = {
            'iteration': iteration,
            'error': error_msg,
            'state': state
        }

        if exception:
            error_record['exception'] = exception

        self.errors.append(error_record)
        # self.logger.debug(f"Recorded error in iteration {iteration}: {error_msg}")

    def increment_iteration(self):
        """Increment the iteration counter"""
        self.iteration += 1

    def set_iteration(self, iteration):
        """Set current iteration"""
        self.iteration = iteration

    def get_iteration(self):
        """Get current iteration"""
        return self.iteration

    def set_final_state(self, state: str, context: Dict):
        """Set the final state and context when execution completes"""
        self.final_state = state
        self.final_context = context
        self.logger.info(f"Set final state: {state}")

    def get_recent_actions(self, count: int) -> List[Dict]:
        """Get the most recent N action executions"""
        if count <= 0:
            return []
        return self.actions_executed[-count:]

    def get_history_summary(self, context_length: int = 5) -> str:
        """
        Generate a summary of recent state/action transitions

        Args:
            context_length: Number of recent actions to include in summary

        Returns:
            Formatted string summary of recent history
        """
        if not self.actions_executed or context_length <= 0:
            return ""

        # Get the last K transitions
        recent_actions = self.get_recent_actions(context_length)

        # Build summary
        history_lines = []
        for action_info in recent_actions:
            iteration = action_info.get('iteration', '?')
            action_name = action_info.get('action', 'unknown')
            result = action_info.get('action_result', 'unknown')
            prev_state = action_info.get('prev_state', 'unknown')
            curr_state = action_info.get('curr_state', 'unknown')
            reason = action_info.get('action_reason', 'No reason provided')
            reason = truncate_text(reason)

            # Format the transition
            history_lines.extend([
                f"[Iteration {iteration}] - selected action: {action_name}, reason: {reason}, action result: {result}",
                f"[Iteration {iteration}] - changed state: {prev_state} --[{action_name}]--> {curr_state}"
            ])

        return "\n".join(history_lines)

    def get_execution_summary(self) -> str:
        """Get a high-level summary of the entire execution"""
        summary_lines = [
            f"Total iterations: {self.iteration}",
            f"Actions executed: {len(self.actions_executed)}",
            f"Errors encountered: {len(self.errors)}",
            f"Final state: {self.final_state or 'Unknown'}"
        ]

        if self.actions_executed:
            action_names = [action['action'] for action in self.actions_executed]
            summary_lines.append(f"Action sequence: {' -> '.join(action_names)}")

        return "\n".join(summary_lines)

    def has_errors(self) -> bool:
        """Check if any errors were recorded"""
        return len(self.errors) > 0

    def get_last_error(self) -> Optional[Dict]:
        """Get the most recent error, if any"""
        return self.errors[-1] if self.errors else None

    def get_action_success_rate(self) -> float:
        """Calculate the success rate of executed actions"""
        if not self.actions_executed:
            return 0.0

        successful_actions = sum(1 for action in self.actions_executed
                               if action.get('action_result') == 'success')
        return successful_actions / len(self.actions_executed)

    def reset(self):
        """Reset the history to initial state"""
        self.iteration = 0
        self.actions_executed.clear()
        self.states_visited.clear()
        self.errors.clear()
        self.final_state = None
        self.final_context = None
        self.logger.info("History reset to initial state")

    def to_dict(self) -> Dict[str, Any]:
        """Convert history to dictionary format for backward compatibility"""
        return {
            'iterations': self.iteration,
            'actions_executed': self.actions_executed.copy(),
            'states_visited': self.states_visited.copy(),
            'errors': self.errors.copy(),
            'final_state': self.final_state,
            'final_context': self.final_context.copy() if self.final_context else None
        }

    def __str__(self) -> str:
        """String representation of the history"""
        return self.get_execution_summary()
