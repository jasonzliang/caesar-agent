import os
import traceback
from typing import Dict, List, Any, Optional

from .action import Action
from .edit_code_action import EditCodeAction, create_analysis_prompt
from ..config import check_attrs
from ..logger import get_logger
from ..parsing import hash_string

class EditCodeAction2(EditCodeAction):
    """Action class for editing and improving code files"""

    def summary(self, agent) -> str:
        """Return a short summary of the code editing action"""
        selected_file = agent.context['selected_file']
        file_path = selected_file['path']
        filename = os.path.basename(file_path)
        return f"analyze and improve code in {filename}, addressing execution issues, bugs, performance, and readability using a more complex, detailed LLM prompt template"

    def _create_code_prompt(self, agent, file_path: str, content: str) -> str:
        """Create a structured prompt for the LLM to improve code using evidence-based strategies"""

        # Load original version content
        original_version = agent.version_manager.load_original_version(file_path, include_content=True)

        # Build dynamic sections
        analysis_context = self._build_analysis_context(agent, file_path)
        original_context = self._build_original_context(original_version)

        base_prompt = self._get_prompt_template()

        # Format the complete prompt
        return base_prompt.format(
            file_path=file_path,
            content=content,
            original_context=original_context,
            analysis_context=analysis_context
        )

    def _build_original_context(self, original_version) -> str:
        """Build original version context if available"""
        if original_version and original_version.get('content'):
            return f"""
    ## Original Version
    ```python
    {original_version['content']}
    ```
    **Context**: Compare current implementation with original to understand evolution and make informed improvements"""
        return ""

    def _build_analysis_context(self, agent, file_path: str) -> str:
        """Build analysis context if available"""
        analysis_prompt = create_analysis_prompt(agent, file_path)
        if analysis_prompt:
            return f"""
    ## Execution Analysis
    {analysis_prompt}
    **Priority**: Fix execution failures and address identified issues"""
        return ""

    def _get_prompt_template(self) -> str:
        """Return the optimized prompt template"""
        return """# Code Improvement Analysis

    ## Code Assessment
    **File**: {file_path}
    ```python
    {content}
    ```

    {original_context}

    {analysis_context}

    ## Improvement Strategy
    Apply systematic code enhancement across these dimensions:

    **Correctness & Stability**:
    - Fix syntax errors, logic bugs, and runtime exceptions
    - Handle edge cases and invalid inputs properly
    - Ensure proper error handling and recovery

    **Performance & Efficiency**:
    - Optimize algorithms and data structures
    - Eliminate redundant operations and memory leaks
    - Improve computational complexity where possible

    **Code Quality & Maintainability**:
    - Enhance readability with clear naming and structure
    - Add comprehensive documentation and type hints
    - Modularize code for better organization

    ## Technical Specifications
    - **Standards**: Follow PEP 8 and Python best practices
    - **Compatibility**: Maintain existing interface contracts
    - **Dependencies**: Preserve import structure and requirements
    - **Focus**: Address execution issues identified in analysis above

    ## Response Format
    ```json
    {{
        "improved_code": "Complete enhanced file content (no markdown blocks)",
        "explanation": "Summary of improvements and technical decisions",
        "changes": [
            {{
                "type": "correctness|performance|maintainability|feature",
                "description": "Short description of what code was added/improved",
                "reasoning": "Short reason of   why this coding approach was chosen"
            }}
        ]
    }}
    ```

    **Critical Requirements**:
    1. Return ENTIRE file content with all improvements integrated
    2. Ensure syntactic correctness and proper Python execution
    3. Preserve original functionality while enhancing implementation
    4. Address specific execution failures if analysis provided
    5. Document all changes with clear technical reasoning"""
