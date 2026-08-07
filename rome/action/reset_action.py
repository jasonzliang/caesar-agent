import glob
import os
import sys
import traceback
from typing import Dict, List, Any, Optional

from .action import Action
from ..logger import get_logger


class ResetAction(Action):
    """Action to clear context and return to initial state"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = get_logger()

    def summary(self, agent) -> str:
        """Return a short summary of the reset action"""
        return "clear current context and return to initial state to start a new task"

    def execute(self, agent, **kwargs) -> bool:
        # No need to clear agent context manually since idle state does that
        self.logger.info("Agent context has been cleared")
        return True
