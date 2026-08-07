import os
import traceback
from typing import Dict, List, Any, Optional

from .action import Action
from ..config import check_attrs, SUMMARY_LENGTH, LONGER_SUMMARY_LEN, LONGEST_SUMMARY_LEN
from ..logger import get_logger
from ..parsing import hash_string
from ..state import truncate_text


def create_analysis_prompt(agent, file_path: str, file_content: str, query_kb: bool = True) -> Optional[str]:
    """Create analysis context for code editing prompts by loading execution data and querying knowledge base."""

    # Load the latest execution results
    execution_data = agent.version_manager.get_data(file_path, 'exec_result')

    context_parts = []

    # Add execution output if available
    if execution_data:
        exec_agent_id = execution_data['agent_id']
        current_agent_id = agent.get_id()

        context_parts.append(f"# Exit code: {execution_data['exec_exit_code']}")
        context_parts.append(f"# Execution output:\n```\n{execution_data['exec_output']}\n```")
        context_parts.append(f"# Execution analysis:\n{execution_data['exec_analysis']}")

        if exec_agent_id != current_agent_id:
            context_parts.append(f"# Note: The above analysis was performed by a different agent ({exec_agent_id}). Please double-check and verify the analysis results.")

    # Query knowledge base for relevant insights
    if query_kb:
        kb_insights = _query_knowledge_base(agent, file_path, file_content, execution_data)
        if kb_insights:
            context_parts.extend(kb_insights)

    if not context_parts:
        return None

    # Create clearly separated analysis section
    analysis_context = "\n---\n"
    analysis_context += "Previous code/test execution results and analysis:\n"
    analysis_context += "---\n\n"
    analysis_context += "\n\n".join(context_parts)
    analysis_context += "\n---\n"
    analysis_context += "Please address any issues identified above\n"
    analysis_context += "---\n"

    return analysis_context


def _query_knowledge_base(agent, file_path: str, file_content: str,
    execution_data: Optional[Dict]) -> List[str]:
    """Query knowledge base for relevant insights using agent's kb_manager.query method."""

    insights = []
    filename = os.path.basename(file_path)

    # Build prioritized query terms
    query_terms = _build_kb_query_terms(filename, file_path, file_content, execution_data)

    # Try each query until we get useful results
    for query_text, context_desc in query_terms:
        if not query_text.strip():
            continue

        # Use the kb_manager.query method which returns a string response
        kb_response = agent.kb_manager.query(
            question=query_text,
            top_k=5,  # Get top K most relevant results
            show_scores=True  # Don't show scores in logs for cleaner output
        )

        # Check if we got a meaningful response
        if kb_response and len(kb_response.strip()) > SUMMARY_LENGTH:  # Ensure substantial content
            # Don't include error responses
            if not kb_response.startswith("Error:"):
                insights.append(f"# Insights from agent's knowledge base ({context_desc}):\n{kb_response}")

    return insights


def _build_kb_query_terms(filename: str, file_path: str, file_content: str,
    execution_data: Optional[Dict]) -> List[tuple]:
    """Build prioritized query terms for knowledge base search."""

    queries = []
    if file_content:
        exec_analysis = execution_data.get('exec_analysis', '') if execution_data else ''
        code_query = f"""Find relevant insights to improve/complete the following code. Insights can be code snippets, algorithms, design patterns, problem solving strategies, or anything that be helpful.

# Code:
```python
{file_content}
```

# Code execution analysis:
{truncate_text(exec_analysis, LONGEST_SUMMARY_LEN)}
"""

#         code_query = f"""Find relevant insights to improve/fix the following code:
# # Code:
# ```python
# {truncate_text(file_content, LONGEST_SUMMARY_LEN)}
# ```
        queries.append((code_query, "code analysis"))

    # Error-specific queries (highest priority if execution failed)
    # if execution_data and execution_data.get('exec_exit_code') != 0:
    #     exec_analysis = execution_data.get('exec_analysis', '')

    #     # Use execution analysis if available
    #     analysis_query = f"Find insights for following execution analysis: {exec_analysis}"
    #     queries.append((analysis_query, "code execution analysis"))

    return queries


class EditCodeAction(Action):
    """Base action class for editing and improving code files"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.logger = get_logger()

    def summary(self, agent) -> str:
        """Return a short summary of the code editing action"""
        selected_file = agent.context['selected_file']
        file_path = selected_file['path']
        filename = os.path.basename(file_path)
        return f"analyze and improve code in {filename}, addressing execution issues, bugs, performance, and readability"

    def execute(self, agent, **kwargs) -> bool:
        """Execute code editing action to improve the current selected file"""

        selected_file = agent.context['selected_file']
        file_path = selected_file['path']
        file_content = selected_file['content']
        test_path = selected_file.get('test_path')
        test_content = selected_file.get('test_content')

        # Prepare prompt for code improvement
        prompt = self._create_code_prompt(agent, file_path, file_content, test_path, test_content)

        # Get improved code from LLM
        self.logger.info(f"Requesting code improvements for {file_path}")

        response = agent.chat_completion(
            prompt=prompt,
            system_message=agent.role,
            response_format={"type": "json_object"}
        )

        # Parse the response to get improved code
        result = agent.parse_json_response(response)

        if not result or "improved_code" not in result:
            self.logger.error(f"Invalid response format from LLM: {response}")
            return False

        improved_code = result.get('improved_code')
        explanation = result.get('explanation', 'No explanation provided')
        changes = result.get('changes', [])

        # Validate the improved code
        if not improved_code or not isinstance(improved_code, str):
            self.logger.error("Improved code is missing or invalid")
            return False

        if improved_code == file_content:
            self.logger.info("No changes made to the code")
            # Even if no changes, we'll continue to the next state

        # Update the selected file in the agent context
        selected_file['content'] = improved_code

        # Record the explanation and changes
        change_record = {
            'explanation': explanation,
            'changes': changes
        }
        selected_file['change_record'] = change_record

        # Write the improved code back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(improved_code)

        # Note: Version saving is now handled in ExecuteCodeAction to include execution results
        self.logger.info(f"Code editing completed for {file_path}")
        self.logger.info(f"Successfully edited and wrote improved code to {file_path}")
        return True

    def _create_code_prompt(self, agent, file_path: str, file_content: str,
        test_path: str, test_content: str) -> str:
        """Create a prompt for the LLM to improve the code"""

        # Base prompt without relying on a configured improvement_prompt
        prompt = """Analyze the code file and suggest improvements. Focus on:
1. Implementing missing code and filling in empty functions
2. Code correctness, quality, and readability
3. Bug fixes and edge cases
4. Performance optimizations
5. Documentation improvements
"""

        # Load original version content
        original_version = agent.version_manager.load_original_version(file_path, include_content=True)

        # Add original version in its own section if available
        if original_version and original_version.get('content'):
            prompt += f"""
# Original file content:
```python
{original_version['content']}
```
"""

        # Add current file code + tests if they exist
        prompt += f"""
# Current code file path: {file_path}
# Current code file content:
```python
{file_content}
```
"""
        if test_path and os.path.exists(test_path):
            prompt += f"""
# Test file path:
{test_path}
# Test file content:
```python
{test_content}
```
"""
        # Get analysis context using the enhanced function
        analysis_prompt = create_analysis_prompt(agent, file_path, file_content,
            query_kb=agent.query_insights)
        if analysis_prompt:
            prompt += f"\n{analysis_prompt}\n"

        prompt += """
Respond with a JSON object containing:
{
    "improved_code": "The new and improved code with changes or edits made to it",
    "explanation": "A clear explanation of the changes you made and why",
    "changes": [
        {
            "type": "improvement type (bug fix, performance, readability, etc.)",
            "description": "Description of the specific change"
        },
        ...
    ]
}

IMPORTANT:
- Return the ENTIRE file content with your improvements, not just the changed parts
- Make sure the improved code is valid Python syntax and contains no markdown formatting like ```python...```
- Be careful with changes - prioritize correctness over style
- List improvements you made in "changes" and summarize the changes in "explanation"
- If improved code is unchanged, be sure give an empty list for "changes" and mention it in "explanation"
"""

        if analysis_prompt:
            prompt += "- Pay special attention to addressing any issues identified in the code analysis and insights from agent's knowledge base"

        if original_version and original_version.get('content'):
            prompt += "\n- Compare current code with the original version to understand the evolution and make informed improvements"

        return prompt
