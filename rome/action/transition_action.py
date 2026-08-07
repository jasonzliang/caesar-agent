from typing import Dict
from .action import Action
from ..logger import get_logger


class TransitionAction(Action):
    """
    Action that performs no operations but transitions between states.
    Preserves agent context and always succeeds. Useful for creating
    explicit state transitions without side effects.

    Target state information is automatically populated when the action
    is added to an FSM via add_action().
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the TransitionAction

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.logger = get_logger()
        self.target_state_name = "Unknown State"
        self.target_state_summary = "No description available"
        self.fsm = None

    def summary(self, agent) -> str:
        """Return a summary describing the state transition"""

        # Get the target state object and its summary
        if self.target_state_name in self.fsm.states:
            pass
            # target_state = self.fsm.states[self.target_state_name]
            # self.target_state_summary = target_state.summary(agent)
        else:
            self.logger.error(f"Cannot find target state in FSM: {target_state_name}")
            raise

        return f"transition to {self.target_state_name}"

    def execute(self, agent, **kwargs) -> bool:
        """
        Execute the transition action (no-op that preserves context)

        Args:
            agent: The agent instance
            **kwargs: Additional arguments (ignored)

        Returns:
            bool: Always returns True to indicate successful transition
        """
        self.logger.info(f"Executing TransitionAction: transitioning to {self.target_state_name}")
        self.logger.debug(f"Agent context preserved: {len(agent.context)} items in context")

        # Explicitly preserve context by doing nothing to it
        # This is a no-op action that maintains all existing agent state
        return True

    def set_target_state_info(self, fsm, target_state_name: str):
        """
        Override base class method to automatically set target state information from FSM
        This method is called by FSM.add_action() to populate state info

        Args:
            fsm: The FSM instance containing the target state
            target_state_name: Name of the target state
        """
        self.target_state_name = target_state_name
        self.fsm = fsm
