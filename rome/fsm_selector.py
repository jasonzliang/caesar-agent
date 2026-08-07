# fsm_selector.py
from abc import ABC, abstractmethod
import os
import sys
from typing import Dict, List, Optional, Callable, Union, Tuple

from .action import *
from .state import *
from .fsm import FSM
from .logger import get_logger
from .config import set_attributes_from_config


class FSMBuilder(ABC):
    """Abstract base class for FSM builders"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = get_logger()
        set_attributes_from_config(self, self.config)

    @abstractmethod
    def build_fsm(self, config: Dict) -> FSM:
        """Build and return a configured FSM instance"""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Return a description of this FSM type"""
        pass


class MinimalFSMBuilder(FSMBuilder):
    """Builder for minimal FSM with just two states"""

    def get_description(self) -> str:
        return ("This is a minimal code analysis worflow with a simple 2-state workflow: "
            "1) Start in IDLE state → Search and load code files → Move to CODE_LOADED state "
            "2) From CODE_LOADED state → Reset back to IDLE to start over. "
            "The agent's purpose is to help users locate, examine, and understand code within a codebase through an iterative search-and-reset cycle.")

    def build_fsm(self, config: Dict) -> FSM:
        """Create minimal FSM with just two states"""
        self.logger.info("Creating minimal FSM")

        fsm = FSM(config.get("FSM", {}))

        # Set overview for the minimal FSM
        fsm.set_overview(self.get_description())

        # Create and add states
        idle_state = fsm.add_state(IdleState(config.get('IdleState', {})))
        code_loaded_state = fsm.add_state(CodeLoadedState(config.get('CodeLoadedState', {})))

        # Create actions and add them to FSM
        fsm.add_action(idle_state, code_loaded_state,
                     PrioritySearchAction(config.get('PrioritySearchAction', {})),
                     fallback_state=idle_state)

        fsm.add_action(code_loaded_state, idle_state,
                     ResetAction(config.get('ResetAction', {})))

        # Set initial state and validate
        fsm.set_initial_state(idle_state)
        fsm.validate_fsm()

        self.logger.info("Minimal FSM created successfully")
        return fsm


class SimpleFSMBuilder(FSMBuilder):
    """Builder for simple FSM with code editing, testing, and execution"""

    def get_description(self) -> str:
        return ("This is a simple code development workflow with 6-states: "
            "1) IDLE → Search/load code → CODE_LOADED "
            "2) CODE_LOADED → Edit code OR write tests → CODE_EDITED or TEST_EDITED "
            "3) CODE_EDITED → Write/edit tests → TEST_EDITED "
            "4) TEST_EDITED → Execute code with tests → CODE_EXECUTED_PASS (success) or CODE_EXECUTED_FAIL (failure) "
            "5) CODE_EXECUTED_PASS → Reset → IDLE (complete successful cycle) "
            "6) CODE_EXECUTED_FAIL → Analyze version history and potentially revert → CODE_LOADED (retry with better version) OR Reset → IDLE (start over). "
            "The agent follows a complete development lifecycle with intelligent failure recovery: discover code → modify code → create tests → validate → recover smartly or complete.")

    def build_fsm(self, config: Dict) -> FSM:
        """Create simple FSM with code editing, writing tests, and executing tests"""
        self.logger.info("Creating comprehensive FSM with code editing, test writing and execution capabilities")

        fsm = FSM(config.get("FSM", {}))

        # Set overview for the simple FSM
        fsm.set_overview(self.get_description())

        # Create and add states
        idle_state = fsm.add_state(IdleState(config.get('IdleState', {})))
        code_loaded_state = fsm.add_state(CodeLoadedState(config.get('CodeLoadedState', {})))
        code_edited_state = fsm.add_state(CodeEditedState(config.get('CodeEditedState', {})))
        test_edited_state = fsm.add_state(TestEditedState(config.get('TestEditedState', {})))
        code_executed_pass_state = fsm.add_state(
            CodeExecutedPassState(config.get('CodeExecutedPassState', {})))
        code_executed_fail_state = fsm.add_state(
            CodeExecutedFailState(config.get('CodeExecutedFailState', {})))

        # Create actions with their respective configurations
        search_action = TournamentSearchAction(config.get('TournamentSearchAction', {}))
        # search_action = PrioritySearchAction(config.get('PrioritySearchAction', {}))
        reset_action = AdvancedResetAction(config.get('AdvancedResetAction', {}))
        edit_code_action = EditCodeAction(config.get('EditCodeAction', {}))
        edit_test_action = EditTestAction(config.get('EditTestAction', {}))
        execute_code_action = ExecuteCodeAction(config.get('ExecuteCodeAction', {}),
            config.get('Executor', {}))

        # Replace TransitionAction with RevertCodeAction for intelligent failure recovery
        revert_code_action = RevertCodeAction(config.get('RevertCodeAction', {}))

        # Add transitions from Idle state
        fsm.add_action(idle_state, code_loaded_state,
            search_action, fallback_state=idle_state)

        # Add transitions from CodeLoaded state
        fsm.add_action(code_loaded_state, code_edited_state, edit_code_action,
            fallback_state=idle_state)
        fsm.add_action(code_loaded_state, test_edited_state, edit_test_action,
            fallback_state=idle_state)

        # Add transitions from CodeEdited state
        fsm.add_action(code_edited_state, test_edited_state, edit_test_action,
            fallback_state=idle_state)

        # Add transitions from TestEdited state
        fsm.add_action(test_edited_state, code_executed_pass_state, execute_code_action,
            fallback_state=code_executed_fail_state)

        # Add transitions from CodeExecutedPass state
        fsm.add_action(code_executed_pass_state, idle_state, reset_action)

        # Add transitions from CodeExecutedFail state - INTELLIGENT RECOVERY
        fsm.add_action(code_executed_fail_state, code_loaded_state, revert_code_action)
        fsm.add_action(code_executed_fail_state, idle_state, reset_action)

        # Set initial state and validate
        fsm.set_initial_state(idle_state)
        fsm.validate_fsm()

        self.logger.info("Simple FSM created successfully with code editing, test writing, execution, and intelligent failure recovery")
        return fsm


class KnowledgeBaseFSMBuilder(FSMBuilder):
    """Builder for intermediate FSM with code editing, testing, and execution"""

    def get_description(self) -> str:
        return ("This is an intermediate code development workflow with 7-states and multiple action variants: "
            "1) IDLE → Search/load code (TournamentSearch OR PrioritySearch) → CODE_LOADED "
            "2) CODE_LOADED → Edit code (EditCode OR EditCode2) OR write tests (EditTest OR EditTest2) → CODE_EDITED or TEST_EDITED "
            "3) CODE_EDITED → Write/edit tests (EditTest OR EditTest2) → TEST_EDITED "
            "4) TEST_EDITED → Execute code with tests → CODE_EXECUTED_PASS (success) or CODE_EXECUTED_FAIL (failure) "
            "5) CODE_EXECUTED_PASS → Reset/cleanup → CODE_EXECUTED_PASS2 → Save knowledge base → IDLE (complete successful cycle) "
            "6) CODE_EXECUTED_FAIL → Analyze version history and potentially revert → CODE_LOADED (retry with better version) OR Reset → IDLE (start over). "
            "This intermediate-level agent provides multiple strategies for each development phase, allowing for more sophisticated code discovery, editing approaches, and test creation methods compared to the simple workflow.")

    def build_fsm(self, config: Dict) -> FSM:
        """Create simple FSM with code editing, writing tests, and executing tests"""
        self.logger.info("Creating comprehensive FSM with code editing, test writing and execution capabilities")

        fsm = FSM(config.get("FSM", {}))

        # Set overview for the simple FSM
        fsm.set_overview(self.get_description())

        # Create and add states
        idle_state = fsm.add_state(IdleState(config.get('IdleState', {})))
        code_loaded_state = fsm.add_state(CodeLoadedState(config.get('CodeLoadedState', {})))
        code_edited_state = fsm.add_state(CodeEditedState(config.get('CodeEditedState', {})))
        test_edited_state = fsm.add_state(TestEditedState(config.get('TestEditedState', {})))
        code_executed_pass_state = fsm.add_state(
            CodeExecutedPassState(config.get('CodeExecutedPassState', {})))
        code_executed_pass_state2 = fsm.add_state(
            CodeExecutedPassState2(config.get('CodeExecutedPassState', {})))
        code_executed_fail_state = fsm.add_state(
            CodeExecutedFailState(config.get('CodeExecutedFailState', {})))
        code_executed_fail_state2 = fsm.add_state(
            CodeExecutedFailState2(config.get('CodeExecutedFailState', {})))

        # Create actions with their respective configurations
        search_action = TournamentSearchAction(config.get('TournamentSearchAction', {}))
        edit_code_action = EditCodeAction(config.get('EditCodeAction', {}))
        edit_test_action = EditTestAction(config.get('EditTestAction', {}))
        execute_code_action = ExecuteCodeAction(config.get('ExecuteCodeAction', {}),
            config.get('Executor', {}))
        reset_action = AdvancedResetAction(config.get('AdvancedResetAction', {}))
        revert_code_action = RevertCodeAction(config.get('RevertCodeAction', {}))
        save_kb_action = SaveKBAction(config.get('SaveKBAction', {}))

        # Add transitions from Idle state
        fsm.add_action(idle_state, code_loaded_state,
            search_action, fallback_state=idle_state)

        # Add transitions from CodeLoaded state
        fsm.add_action(code_loaded_state, code_edited_state, edit_code_action,
            fallback_state=idle_state)

        fsm.add_action(code_loaded_state, test_edited_state, edit_test_action,
            fallback_state=idle_state)

        # Add transitions from CodeEdited state
        fsm.add_action(code_edited_state, test_edited_state, edit_test_action,
            fallback_state=idle_state)
        # fsm.add_action(code_edited_state, test_edited_state, edit_test_action2,
        #     fallback_state=idle_state)

        # Add transitions from TestEdited state
        fsm.add_action(test_edited_state, code_executed_pass_state, execute_code_action,
            fallback_state=code_executed_fail_state)

        # Add transitions from CodeExecutedPass state
        fsm.add_action(code_executed_pass_state, code_executed_pass_state2, reset_action)
        fsm.add_action(code_executed_pass_state2, idle_state, save_kb_action)

        # Add transitions from CodeExecutedFail state - INTELLIGENT RECOVERY
        fsm.add_action(code_executed_fail_state, code_loaded_state, revert_code_action)
        fsm.add_action(code_executed_fail_state, code_executed_fail_state2, reset_action)
        fsm.add_action(code_executed_fail_state2, idle_state, save_kb_action)

        # search_action2 = PrioritySearchAction(config.get('PrioritySearchAction', {}))
        # edit_code_action2 = EditCodeAction2(config.get('EditCodeAction2', {}))
        # edit_test_action2 = EditTestAction2(config.get('EditTestAction2', {}))
        # fsm.add_action(idle_state, code_loaded_state,
        #     search_action2, fallback_state=idle_state)
        # fsm.add_action(code_loaded_state, code_edited_state, edit_code_action2,
        #     fallback_state=idle_state)
        # fsm.add_action(code_loaded_state, test_edited_state, edit_test_action2,
        #     fallback_state=idle_state)

        # Set initial state and validate
        fsm.set_initial_state(idle_state)
        fsm.validate_fsm()

        self.logger.info("Intermediate FSM created successfully with code editing, test writing, execution, and intelligent failure recovery")
        return fsm


class FSMSelector:
    """Factory class for FSM creation strategies"""

    BUILDERS = {
        'minimal': MinimalFSMBuilder,
        'simple': SimpleFSMBuilder,
        'knowledge_base': KnowledgeBaseFSMBuilder,
    }

    def __init__(self, fsm_type: str = "simple", config: Dict = None):
        """
        Initialize FSM selector with specified type

        Args:
            fsm_type: FSM type ("minimal" or "simple")
            config: Configuration dictionary for the FSM builder
        """
        self.logger = get_logger()

        if fsm_type not in self.BUILDERS:
            raise ValueError(f"Unknown FSM type: {fsm_type}. Available: {list(self.BUILDERS.keys())}")

        self.fsm_type = fsm_type
        self.builder = self.BUILDERS[fsm_type](config)
        self.logger.info(f"Initialized {fsm_type} FSM builder")

    def create_fsm(self, config: Dict) -> FSM:
        """Create FSM using the configured builder"""
        return self.builder.build_fsm(config)

    def get_description(self) -> str:
        """Get description of the current FSM type"""
        return self.builder.get_description()

    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of available FSM types"""
        return list(cls.BUILDERS.keys())

    @classmethod
    def get_all_descriptions(cls) -> Dict[str, str]:
        """Get descriptions for all available FSM types"""
        descriptions = {}
        for fsm_type, builder_class in cls.BUILDERS.items():
            # Create temporary builder to get description
            temp_builder = builder_class({})
            descriptions[fsm_type] = temp_builder.get_description()
        return descriptions
