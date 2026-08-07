import os
import traceback
from typing import Dict, List, Any, Optional

from .action import Action
from .advanced_reset_action import analyze_execution_results, check_max_versions_reached, check_ground_truth
from ..config import check_attrs, LONGER_SUMMARY_LEN, LONGEST_SUMMARY_LEN
from ..logger import get_logger


class SaveKBAction(Action):
    """Action to extract and save reusable insights from code/test pairs to knowledge base, inheriting completion analysis from AdvancedResetAction"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.logger = get_logger()
        check_attrs(self, ['completion_confidence', 'max_versions'])
        self.completion_confidence = max(min(self.completion_confidence, 100), 1)
        self.max_versions = max(self.max_versions, 1)

    def summary(self, agent) -> str:
        """Return a short summary of the knowledge base save action"""
        selected_file = agent.context['selected_file']
        filename = os.path.basename(selected_file['path'])
        return f"extract reusable patterns and insights from {filename} and its tests, then save to knowledge base for future reference"

    def _extract_insights(self, agent, code_content: str, test_content: str,
                         file_path: str, test_path: str) -> Dict[str, str]:
        """Extract reusable insights and patterns from code and test files using LLM"""

        filename = os.path.basename(file_path)
        test_filename = os.path.basename(test_path)

        # Get execution context if available
        execution_context = ""
        if ('exec_output' in agent.context['selected_file'] and
            agent.context['selected_file']['exec_exit_code'] == 0):
            exec_code = agent.context['selected_file'].get('exec_exit_code', 'unknown')
            exec_output = agent.context['selected_file'].get('exec_output', '')[:200]  # Limit context
            exec_analysis = agent.context['selected_file'].get('exec_analysis', '')[:200]
            execution_context = f"Exit: {exec_code}\nOutput: {exec_output}\nAnalysis: {exec_analysis}"

        prompt = f"""Analyze the following code and test files to extract useful, reusable insights that could help with similar problems in the future.

# Code File: {filename}
```python
{code_content}
```

# Test File: {test_filename}
```python
{test_content}
```

{execution_context}

Please extract and provide insights in the following categories:

1. **Code Patterns**: What design patterns, algorithms, or strategies are demonstrated? What is the overall problem solving approach to use and what are some anti-patterns to avoid?

2. **Data Structures**: What data structures are used to help solve the problem? How do the data structures minimize the time or space complexity required?

3. **Reusable Code**: Find 1-3 useful/reusable functions or code snippets that could be used as library code for other similar problems. Use the following format for long code snippets or functions: ```python [code_snippet]```.

4. **Applicable Context**: What types of problems or domains would benefit from these insights?

5. **Testing Approaches**: What testing strategies, patterns, or best practices are shown?

IMPORTANT: Limit your overall response to maximum of 1200 tokens. Be concise but comprehensive.

Respond with a JSON object containing these insights:
{{
    "code_patterns": "Useful design patterns, algorithms, or strategies for solving the problem",
    "data_structures": "Useful data structures for solving the problem",
    "reusable_code": "1-3 most reusable functions or code snippets for library use",
    "applicable_context": "Types of problems or domains where these insights apply",
    "testing_approaches": "Testing strategies and best practices demonstrated"
}}
"""

        response = agent.chat_completion(
            prompt=prompt,
            system_message=agent.role,
            response_format={"type": "json_object"}
        )
        return agent.parse_json_response(response)

    def _create_knowledge_entry(self, insights: Dict, file_path: str, test_path: str) -> str:
        """Create a comprehensive knowledge base entry from insights"""

        filename = os.path.basename(file_path)
        test_filename = os.path.basename(test_path)

        # Create structured knowledge entry
        kb_entry = f"""# Code Insights: {filename}

## Overview
Extracted reusable patterns and insights from {filename} and its test file {test_filename}.

## Code Patterns
{insights.get('code_patterns', 'None identified')}

## Data Structures
{insights.get('data_structures', 'None identified')}

## Reusable Code
{insights.get('reusable_code', 'None identified')}

## Applicable Context
{insights.get('applicable_context', 'None identified')}

## Testing Approaches
{insights.get('testing_approaches', 'None identified')}

## Source Files
- Code: {file_path}
- Tests: {test_path}
"""

        return kb_entry

    def execute(self, agent, **kwargs) -> bool:
        """Extract insights and save them to the knowledge base only if work is complete"""
        if not agent.save_insights:
            self.logger.error(f"Saving insights to knowledge base disabled")
            return False

        # Get file information from context
        selected_file = agent.context['selected_file']
        file_path = selected_file['path']
        test_path = selected_file['test_path']
        code_content = selected_file['content']
        test_content = selected_file['test_content']

        # Use parent class's analysis to check if work is complete
        if agent.use_ground_truth:
            work_complete = check_ground_truth(agent, file_path)
        else:
            work_complete = analyze_execution_results(agent, selected_file,
                self.completion_confidence) or \
            check_max_versions_reached(agent, file_path, self.max_versions)

        if not work_complete:
            self.logger.info(f"Coding not finished for {file_path}, skipping knowledge base storage")
            return False
        self.logger.info(f"Coding finished, extracting insights from {file_path}")

        # Extract insights using LLM analysis
        insights = self._extract_insights(
            agent=agent,
            code_content=code_content,
            test_content=test_content,
            file_path=file_path,
            test_path=test_path
        )

        if not insights or "error" in insights:
            self.logger.error(f"Failed to extract insights: {insights}")
            return False

        # Create comprehensive knowledge base entry
        kb_entry = self._create_knowledge_entry(insights, file_path, test_path)

        # Save to knowledge base with relevant metadata
        filename = os.path.basename(file_path)
        metadata = {
            "source_file": file_path,
            "test_file": test_path,
            "filename": filename,
            "type": "code_insights",
            "agent_id": agent.get_id(),
            "work_complete": True,
        }

        # Store in knowledge base
        success = agent.kb_manager.add_text(kb_entry, metadata)

        if success:
            self.logger.info(f"Successfully saved insights for {filename} to knowledge base")

            # Log key insights for immediate visibility
            # problem_solutions = insights.get('problem_solutions', 'None identified')
            # applicable_context = insights.get('applicable_context', 'General programming')
            # self.logger.info(f"Problem solutions: {problem_solutions[:LONGER_SUMMARY_LEN]}...")
            # self.logger.info(f"Applicable to: {applicable_context}")
            return True
        else:
            self.logger.error(f"Failed to save insights to knowledge base")
            return False