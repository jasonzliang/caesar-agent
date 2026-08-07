import os
import traceback
from typing import Dict, List, Any, Optional

from .action import Action
from .edit_code_action import create_analysis_prompt
from ..config import check_attrs
from ..logger import get_logger
from ..parsing import hash_string

class EditTestAction(Action):
    """Action to create or edit tests for the selected file - now inherits from Action"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.logger = get_logger()

    def summary(self, agent) -> str:
        """Return a short summary of the test editing action"""
        selected_file = agent.context['selected_file']
        file_path = selected_file['path']
        filename = os.path.basename(file_path)
        test_exists = 'test_path' in selected_file and \
            os.path.exists(selected_file.get('test_path', ''))
        action_type = "update existing tests" if test_exists else "create comprehensive unit tests"
        return f"{action_type} for {filename}, covering edge cases, error conditions, and test failures"

    def execute(self, agent, **kwargs) -> bool:
        """Execute test editing action to create or improve tests for the current selected file"""

        selected_file = agent.context['selected_file']
        file_path = selected_file['path']
        file_content = selected_file['content']

        # Determine test file path
        base_path, extension = os.path.splitext(file_path)
        test_path = f"{base_path}_test.py"

        # Initialize changes and test-related fields if they don't exist
        if 'changes' not in selected_file:
            selected_file['changes'] = []

        if 'test_changes' not in selected_file:
            selected_file['test_changes'] = []

        # Check if test file already exists
        test_exists = os.path.exists(test_path)
        test_content = ""

        if test_exists:
            try:
                with open(test_path, 'r', encoding='utf-8') as f:
                    test_content = f.read()
                self.logger.info(f"Existing test file found: {test_path}")
            except Exception as e:
                self.logger.error(f"Error reading test file {test_path}: {e}, overwriting")
        else:
            self.logger.info(f"No existing test file found. Will create new test file at: {test_path}")

        # Prepare prompt for test creation/improvement
        prompt = self._create_test_prompt(agent, file_path, file_content, test_path,
            test_content, test_exists)

        # Get improved tests from LLM
        self.logger.info(f"Requesting test {'improvements' if test_exists else 'creation'} for {file_path}")

        response = agent.chat_completion(
            prompt=prompt,
            system_message=agent.role,
            response_format={"type": "json_object"}
        )

        # Parse the response to get test code
        result = agent.parse_json_response(response)

        if not result or "improved_test_code" not in result:
            self.logger.error(f"Invalid response format from LLM: {response}")
            return False

        improved_test_content = result.get('improved_test_code')
        explanation = result.get('explanation', 'No explanation provided')
        test_changes = result.get('changes', [])

        # Validate the test code
        if not improved_test_content or not isinstance(improved_test_content, str):
            self.logger.error("Test code is missing or invalid")
            return False

        if test_exists and improved_test_content == test_content:
            self.logger.error("No changes made to the test code")
            # Even if no changes, we'll continue to the next state

        # Update the selected file in the agent context with test information
        selected_file['test_path'] = test_path
        selected_file['test_content'] = improved_test_content

        # Record the explanation and changes
        change_record = {
            'explanation': explanation,
            'changes': test_changes
        }
        selected_file['test_changes'].append(change_record)

        # Write the test code to disk
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write(improved_test_content)

        # Note: Version saving is now handled in ExecuteCodeAction to include execution results
        self.logger.info(f"Successfully edited and wrote test code to {test_path}")
        self.logger.info(f"Test editing completed for {file_path}")
        return True

    def _create_test_prompt(self, agent, file_path: str, file_content: str, test_path: str,
                           test_content: str, test_exists: bool) -> str:
        """Create a prompt for the LLM to create or improve tests"""

        # Extract file and module information for proper imports
        file_name = os.path.basename(file_path)
        module_name, _ = os.path.splitext(file_name)

        # Base prompt
        base_prompt = """Create comprehensive unit tests for the provided code file. Focus on:
1. Testing all public methods and functions
2. Edge cases and error conditions
3. Integration between components
4. Good test organization and documentation
"""

        action_type = "improve the existing" if test_exists else "create new"

        prompt = f"""{base_prompt}
I need you to {action_type} unit tests for the following Python code.

# Code file path:
{file_path}
# Code file content:
```python
{file_content}
```
"""
        if test_exists:
            prompt += f"""
# Test file path:
{test_path}
# Test file content:
```python
{test_content}
```
"""
        else:
            prompt += f"""
Create a new test file that will be saved at: {test_path}
Include proper test setup, all necessary imports, and comprehensive test cases.
"""

        # Get analysis context using the static function
        analysis_prompt = create_analysis_prompt(agent, file_path, file_content, query_kb=False)
        if analysis_prompt:
            prompt += f"\n{analysis_prompt}\n"

        prompt += f"""
Respond with a JSON object containing:
{{
    "improved_test_code": "The new and improved test code with changes or edits made to it",
    "explanation": "A clear explanation of the changes you made and why",
    "changes": [
        {{
            "type": "test type (unit test, integration test, etc.)",
            "description": "Description of what this test verifies"
        }},
        ...
    ]
}}

IMPORTANT:
- Make sure the test code is valid Python syntax and contains no markdown formatting like ```python...```
- Tests should be compatible with pytest and be sure to do "import pytest" first
- Make sure to import the module being tested from the code file
- Assume the code file and test file are both located in the root of the current working directory
- Include all necessary imports that are required by the tests
- Avoid external dependencies that are unnecessary for testing
- List improvements you made in "changes" and summarize the changes in "explanation"
- If improved test code is unchanged, be sure give an empty list for "changes" and mention it in "explanation"
"""

        if analysis_prompt:
            prompt += "- Pay special attention to addressing any test failures or issues identified in the code analysis"

        return prompt
