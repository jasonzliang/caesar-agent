import os
import traceback
from typing import Dict, List, Any, Optional

from .action import Action
from ..config import SUMMARY_LENGTH, check_attrs
from ..logger import get_logger
from ..executor import CodeExecutor, CodeBlock


class ExecuteCodeAction(Action):
    """Action to execute test code for the selected file"""

    def __init__(self, config: Dict = None, executor_config: Dict = None):
        super().__init__(config)
        self.logger = get_logger()
        self.executor_config = executor_config
        # check_attrs(self, ['executor_config'])

        # Initialize code executor
        self.executor = CodeExecutor(self.executor_config)

    def summary(self, agent) -> str:
        """Return a short summary of the code execution action"""
        selected_file = agent.context['selected_file']
        test_path = selected_file['test_path']
        test_filename = os.path.basename(test_path)
        return f"execute {test_filename}, analyze results for failures/issues, and save execution data with version history"

    def _analyze_execution_result(self,
        agent,
        original_file_content: str,
        test_file_content: str,
        output: str,
        exit_code: int) -> str:
        """Generate analysis of test execution results using LLM."""
        prompt = f"""Analyze the following test execution results and provide comprehensive feedback:

# Code file content:
```python
{original_file_content}
```

# Test file content:
```python
{test_file_content}
```

# Execution output:
```
{output}
```

Exit code: {exit_code}

Please create an analysis covering:
1. Overall test execution status (passed/failed)
2. Specific test failures and their root causes
3. Any errors, exceptions, or warnings found
4. Code quality issues revealed by the tests
5. Suggestions for fixing identified problems
6. Recommendations for improving test coverage

IMPORTANT: Provide analysis as a numbered list with no other text before or after the list

# Your analysis:
"""
        return agent.chat_completion(prompt=prompt, system_message=agent.role)



    def execute(self, agent, **kwargs) -> bool:
        """Execute tests for the current selected file"""

        selected_file = agent.context['selected_file']
        file_path = selected_file['path']
        test_path = selected_file['test_path']

        self.logger.info(f"Executing test file: {test_path}")

        try:
            # Execute the test file
            result = self.executor.execute_file(test_path)
        except Exception as e:
            # Store error information in context
            error_msg = f"Error executing test file {test_path}: {str(e)}"
            selected_file['exec_output'] = error_msg
            selected_file['exec_exit_code'] = 1
            selected_file['exec_analysis'] = f"Test execution failed due to an error: {str(e)}"

            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            return False

        # Generate execution analysis
        analysis = self._analyze_execution_result(
            agent=agent,
            original_file_content=selected_file['content'],
            test_file_content=selected_file['test_content'],
            output=result.output,
            exit_code=result.exit_code
        )

        # Store execution results in context for immediate access
        selected_file['exec_output'] = result.output
        selected_file['exec_exit_code'] = result.exit_code
        selected_file['exec_analysis'] = analysis

        # Store execution results in TinyDB
        execution_data = {
            'test_path': test_path,
            'exec_output': result.output,
            'exec_analysis': analysis,
            'exec_exit_code': result.exit_code,
            'agent_id': agent.get_id()
        }
        agent.version_manager.store_data(file_path, 'exec_result', execution_data)

        # Save code version with execution results if we have change record
        if 'change_record' in selected_file:
            change_record = selected_file['change_record']
            code_version = agent.version_manager.save_version(
                file_path=file_path,
                content=selected_file['content'],
                changes=change_record.get('changes', []),
                explanation=change_record.get('explanation', 'No explanation provided'),
                execution_output=result.output,
                exit_code=result.exit_code,
                execution_analysis=analysis
            )
            self.logger.info(f"Saved code version {code_version} with execution results for {file_path}")

        # Save test version with execution results if we have test changes
        if 'test_changes' in selected_file and selected_file['test_changes']:
            # Get the most recent test change record
            latest_test_change = selected_file['test_changes'][-1]
            test_version = agent.version_manager.save_test_version(
                test_file_path=test_path,
                content=selected_file['test_content'],
                changes=latest_test_change.get('changes', []),
                explanation=latest_test_change.get('explanation', 'No explanation provided'),
                execution_output=result.output,
                exit_code=result.exit_code,
                execution_analysis=analysis
            )
            self.logger.info(f"Saved test version {test_version} with execution results for {test_path}")

        self.logger.info(f"Test execution completed with exit code: {result.exit_code}")
        self.logger.info(f"Analysis summary: {analysis[:SUMMARY_LENGTH]}...")

        # Return True if tests passed (exit code 0), False otherwise
        return result.exit_code == 0
