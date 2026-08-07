import os
import traceback
from typing import Dict, List, Any, Optional

from .action import Action
from .edit_test_action import EditTestAction
from .edit_code_action import create_analysis_prompt
from ..config import check_attrs
from ..logger import get_logger
from ..parsing import hash_string

class EditTestAction2(EditTestAction):
    """Action to create or edit tests for the selected file - now inherits from Action"""

    def summary(self, agent) -> str:
        """Return a short summary of the test editing action"""
        selected_file = agent.context['selected_file']
        file_path = selected_file['path']
        filename = os.path.basename(file_path)
        test_exists = 'test_path' in selected_file and \
            os.path.exists(selected_file.get('test_path', ''))
        action_type = "update existing tests" if test_exists else "create comprehensive unit tests"
        return f"{action_type} for {filename}, covering edge cases, error conditions, and test failures using a more complex, detailed LLM prompt template"

    def _create_test_prompt(self, agent, file_path: str, file_content: str, test_path: str,
                           test_content: str, test_exists: bool) -> str:
        """Create a structured prompt for comprehensive test creation using evidence-based strategies"""

        # Extract file and module information for proper imports
        file_name = os.path.basename(file_path)
        module_name, _ = os.path.splitext(file_name)

        # Build dynamic sections
        existing_test_section = self._build_existing_test_section(test_path, test_content, test_exists)
        analysis_context = self._build_analysis_context(agent, file_path)

        base_prompt = self._get_prompt_template()

        # Format the complete prompt
        return base_prompt.format(
            file_path=file_path,
            module_name=module_name,
            file_content=file_content,
            existing_test_section=existing_test_section,
            analysis_context=analysis_context
        )

    def _build_existing_test_section(self, test_path: str, test_content: str, test_exists: bool) -> str:
        """Build the existing test section conditionally"""
        if test_exists:
            return f"""
## Existing Test Analysis
**Test File**: {test_path}
```python
{test_content}
```
**Task**: Improve coverage and fix identified gaps"""
        else:
            return f"""
## Test Creation
**Target**: {test_path}
**Task**: Create comprehensive test suite from scratch"""

    def _build_analysis_context(self, agent, file_path: str) -> str:
        """Build analysis context if available"""
        analysis_prompt = create_analysis_prompt(agent, file_path)
        if analysis_prompt:
            return f"""
## Execution Analysis
{analysis_prompt}
**Priority**: Address execution failures and edge cases identified above"""
        return ""

    def _get_prompt_template(self) -> str:
        """Return the optimized prompt template"""
        return """# Test Generation Analysis

## Code Analysis
**File**: {file_path} | **Module**: {module_name}

```python
{file_content}
```

{existing_test_section}

{analysis_context}

## Test Strategy
Generate comprehensive tests using systematic coverage:

**Core Testing**:
- All public methods with varied inputs
- Return value validation and type checking
- State changes and side effects

**Robustness Testing**:
- Boundary conditions (empty, None, extremes)
- Invalid inputs and exception paths
- Resource constraints and error recovery

**Integration Testing**:
- Component interactions and dependencies
- Mock external calls where needed

## Technical Specifications
- **Framework**: pytest with fixtures and parametrization
- **Imports**: `import pytest` + `from {module_name} import *`
- **Structure**: Logical grouping with descriptive test names
- **Coverage**: Focus on preventing execution failures from analysis

## Response Format
```json
{{
    "improved_test_code": "Complete test file content (no markdown blocks)",
    "explanation": "Testing strategy and key decisions",
    "changes": [
        {{
            "type": "coverage_area",
            "description": "Short description of what test was added/improved",
            "reasoning": "Short reason of why this testing approach was chosen"
        }}
    ]
}}
```

**Critical Requirements**:
1. Return ENTIRE test file with all imports and structure
2. Ensure pytest compatibility and executable Python syntax
3. Files located in current working directory root
4. Address specific execution issues if analysis provided
5. Mention if no changes made to existing tests"""
