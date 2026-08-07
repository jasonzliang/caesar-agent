from abc import ABC, abstractmethod
import glob
import os
import sys
from typing import Dict, List, Any, Optional, Callable
from ..logger import get_logger
from ..config import set_attributes_from_config


class Action(ABC):
    """Abstract base class for all actions"""

    def __init__(self, config: Dict = None):
        """Initialize the action with a configuration dictionary"""
        self.name = self.__class__.__name__
        self.config = config or {}
        self.logger = get_logger()

        # Automatically set attributes from config
        set_attributes_from_config(self, self.config)

    def set_target_state_info(self, fsm, target_state_name: str):
        """
        Optional method to receive target state information when action is added to FSM.
        Base implementation does nothing - subclasses can override to use this information.

        Args:
            fsm: The FSM instance containing the target state
            target_state_name: Name of the target state this action transitions to
        """
        # Base implementation is a no-op
        # Subclasses that need target state info can override this method
        pass

    @abstractmethod
    def execute(self, agent, **kwargs) -> bool:
        """Execute the action and return True if action succeeded otherwise False"""
        pass

    @abstractmethod
    def summary(self, agent) -> str:
        """Return a short summary that describes the current action"""
        pass