# Updated state.py with enhanced summary methods
from abc import ABC, abstractmethod
import os
import sys
from typing import Dict, List, Optional, Callable

from .config import LONG_SUMMARY_LEN, set_attributes_from_config
from .logger import get_logger


# Helper functions for common context validation patterns
def validate_context_keys(context: Dict, required_keys: List[str], context_name: str = "context") -> None:
    """Validate that all required keys exist in the given context"""
    for key in required_keys:
        assert key in context, f"Missing {key} in {context_name}"


def validate_selected_file_context(agent, required_keys: List[str]) -> Dict:
    """Validate and return selected_file from agent context with required keys"""
    selected_file = agent.context.get('selected_file')
    assert selected_file is not None, "selected_file is None in agent context"

    validate_context_keys(selected_file, required_keys, 'selected file')
    return selected_file


def validate_file(file_path: str, expected_content: str) -> str:
    """Validate that file exists and expected_content matches disk, return actual content"""
    logger = get_logger()

    # Check if file exists
    if not os.path.exists(file_path):
        raise AssertionError(f"File path does not exist: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            actual_content = f.read()

        if actual_content != expected_content:
            logger.error(f"Content mismatch in {file_path}! Expected content is out of sync with disk. "
                        f"Expected {len(expected_content)} chars, found {len(actual_content)} chars. "
                        f"Returning actual disk content.")

        return actual_content

    except Exception as e:
        logger.error(f"Failed to validate file for {file_path}: {e}")
        raise


def get_file_size_info(file_path: str) -> str:
    """Get formatted file size information"""
    try:
        size_kb = os.path.getsize(file_path) / 1024
        return f" ({size_kb:.1f}KB)"
    except:
        return ""


def truncate_text(text: str, length: int = LONG_SUMMARY_LEN) -> str:
    """Truncate text with ellipsis if it exceeds length"""
    return text[:length] + '...' if len(text) > length else text


class State(ABC):
    """Abstract state"""
    def __init__(self,
        actions: List[str] = None,
        config: Dict = None):

        self.name = self.__class__.__name__
        self.config = config or {}
        # Initialize with empty actions list - will be populated by FSM
        self.actions = actions or []  # Store available actions for this state
        self.logger = get_logger()

        # Set attributes from config if provided
        set_attributes_from_config(self, self.config)

    def get_available_actions(self) -> List[str]:
        """Get the list of available actions for this state"""
        return self.actions.copy()

    def add_action(self, action_name: str) -> None:
        """Add an action to this state if it doesn't already exist"""
        if action_name not in self.actions:
            self.actions.append(action_name)

    @abstractmethod
    def check_context(self, agent, **kwargs) -> bool:
        """Check if agent context has required values"""
        pass

    @abstractmethod
    def summary(self, agent) -> str:
        """A short summary description of the current state"""
        pass

    def future_summary(self, agent) -> str:
        """A short summary description of the state if it is not current state"""
        return self.summary(agent)


class IdleState(State):
    """Initial state where the agent waits for commands"""

    def __init__(self, config: Dict = None):
        super().__init__(actions=[], config=config)

    def check_context(self, agent, **kwargs) -> bool:
        """In idle state, clear any previous context and always return True"""
        agent.context.clear()
        return True

    def summary(self, agent) -> str:
        """Enhanced summary for idle state"""
        return f"you are idling in initial state and ready to start a new task"

    def future_summary(self, agent) -> str:
        return "you are in default initial state and ready to start a new task"


class CodeLoadedState(State):
    """State where code has been loaded and is ready for analysis"""

    def __init__(self, config: Dict = None):
        super().__init__(actions=[], config=config)

    def check_context(self, agent, **kwargs) -> bool:
        """Check if we have a selected file in context"""
        selected_file = validate_selected_file_context(agent, ['path', 'content', 'reason'])
        selected_file['content'] = validate_file(selected_file['path'], selected_file['content'])
        return True

    def summary(self, agent) -> str:
        """Enhanced summary for code loaded state"""
        selected_file = agent.context['selected_file']
        filename = os.path.basename(selected_file['path'])
        size_info = get_file_size_info(selected_file['path'])
        reason = truncate_text(selected_file['reason'])

        return f"you selected and loaded code file {filename}{size_info} for editing, selection reason: {reason}"

    def future_summary(self, agent) -> str:
        return "you have loaded a code file for editing and testing"


class CodeEditedState(State):
    """State where code has been edited and updated"""

    def __init__(self, config: Dict = None):
        super().__init__(actions=[], config=config)

    def check_context(self, agent, **kwargs) -> bool:
        """Check if we have a selected file in context"""
        selected_file = validate_selected_file_context(agent, ['path', 'content', 'change_record'])

        # Validate nested change_record structure
        validate_context_keys(selected_file['change_record'], ['changes', 'explanation'], 'change record')
        selected_file['content'] = validate_file(selected_file['path'], selected_file['content'])
        return True

    def summary(self, agent) -> str:
        """Enhanced summary for code edited state"""
        selected_file = agent.context['selected_file']
        filename = os.path.basename(selected_file['path'])
        change_record = selected_file['change_record']

        num_changes = len(change_record['changes'])
        explanation = truncate_text(change_record['explanation'])

        return f"you edited {filename} with {num_changes} change(s) with explanation: {explanation}"

    def future_summary(self, agent) -> str:
        return "you have edited the code file with changes and explanation"


class TestEditedState(State):
    """State where code tests has been edited and updated"""

    def __init__(self, config: Dict = None):
        super().__init__(actions=[], config=config)

    def check_context(self, agent, **kwargs) -> bool:
        """Check if we have a selected file in context"""
        required_keys = ['path', 'content', 'changes', 'test_path', 'test_content', 'test_changes']
        selected_file = validate_selected_file_context(agent, required_keys)

        # Validate both files exist and content matches disk
        selected_file['content'] = validate_file(selected_file['path'], selected_file['content'])
        selected_file['test_content'] = validate_file(selected_file['test_path'], selected_file['test_content'])

        return True

    def summary(self, agent) -> str:
        """Enhanced summary for test edited state"""
        selected_file = agent.context['selected_file']
        filename = os.path.basename(selected_file['path'])
        test_filename = os.path.basename(selected_file['test_path'])

        num_test_changes = len(selected_file['test_changes'])
        return f"you created/updated tests in {test_filename} for {filename} with {num_test_changes} change(s)"

    def future_summary(self, agent) -> str:
        return "you have created or updated tests for the code file"


class CodeExecutedPassState(State):
    """State where code or test file has been executed successfully"""

    def __init__(self, config: Dict = None):
        super().__init__(actions=[], config=config)

    def check_context(self, agent, **kwargs) -> bool:
        """Check if we have a selected file in context with successful execution"""
        required_keys = ['path', 'content', 'test_path', 'test_content', 'exec_output', 'exec_exit_code', 'exec_analysis']
        selected_file = validate_selected_file_context(agent, required_keys)

        # Validate both files exist and content matches disk
        selected_file['content'] = validate_file(selected_file['path'], selected_file['content'])
        selected_file['test_content'] = validate_file(selected_file['test_path'], selected_file['test_content'])

        # Validate that execution was successful
        exit_code = selected_file['exec_exit_code']
        assert exit_code == 0, f"Expected successful execution (exit code 0), got {exit_code}"
        return True

    def summary(self, agent) -> str:
        """Enhanced summary for code executed pass state"""
        selected_file = agent.context['selected_file']
        test_filename = os.path.basename(selected_file['test_path'])

        # Get brief output summary
        output = selected_file['exec_output'] or 'No output'
        analysis = selected_file['exec_analysis'] or 'No analysis'
        output_summary = truncate_text(output.split('\n')[0])
        analysis_summary = truncate_text(analysis.split('\n')[0])

        return f"you executed {test_filename}: ✓ PASSED (exit code: 0), output: {output_summary}"

    def future_summary(self, agent) -> str:
        return "you have executed test file and the output shows the tests have passed"


class CodeExecutedPassState2(CodeExecutedPassState):
    """Intermediate transition state after code/tests have executed successfully"""
    pass


class CodeExecutedFailState(State):
    """State where code or test file execution failed"""

    def __init__(self, config: Dict = None):
        super().__init__(actions=[], config=config)

    def check_context(self, agent, **kwargs) -> bool:
        """Check if we have a selected file in context with failed execution"""
        required_keys = ['path', 'content', 'test_path', 'test_content', 'exec_output', 'exec_exit_code', 'exec_analysis']
        selected_file = validate_selected_file_context(agent, required_keys)

        # Validate both files exist and content matches disk
        selected_file['content'] = validate_file(selected_file['path'], selected_file['content'])
        selected_file['test_content'] = validate_file(selected_file['test_path'], selected_file['test_content'])

        # Validate that execution failed
        exit_code = selected_file['exec_exit_code']
        assert exit_code != 0, f"Expected failed execution (exit code != 0), got {exit_code}"
        return True

    def summary(self, agent) -> str:
        """Enhanced summary for code executed fail state"""
        selected_file = agent.context['selected_file']
        test_filename = os.path.basename(selected_file['test_path'])
        exit_code = selected_file['exec_exit_code']

        # Get brief output summary focusing on error info
        output = selected_file['exec_output'] or 'No output'
        analysis = selected_file['exec_analysis'] or 'No analysis'
        output_summary = truncate_text(output.split('\n')[0])
        analysis_summary = truncate_text(analysis.split('\n')[0])

        return f"you executed {test_filename}: ✗ FAILED (exit code: {exit_code}), output: {output_summary}, analysis: {analysis_summary}"

    def future_summary(self, agent) -> str:
        return "you have executed test file and the output shows the tests have failed"

class CodeExecutedFailState2(CodeExecutedFailState):
    """Intermediate transition state after code/tests have failed"""
    pass
