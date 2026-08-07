import glob
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Dict, List, Any, Optional

from .action import Action
from ..logger import get_logger
from ..config import check_attrs, LOG_DIR_NAME, EVAL_DIR_NAME, EVAL_RESULTS_NAME


def same_path(path1, path2):
    """Check if two path strings resolve to the same location."""
    return Path(path1).resolve() == Path(path2).resolve()


def check_ground_truth(agent, file_path, use_plus=False) -> bool:
    """Check if the evalplus result passes base tests"""
    agent.logger.info(f"Checking evaluation ground truth status for {file_path}")

    eval_results_file = Path(agent.repository) / LOG_DIR_NAME / EVAL_DIR_NAME / EVAL_RESULTS_NAME
    if not eval_results_file.exists():
        agent.logger.error(f"Eval results file not found: {eval_results_file}")
        return False

    with open(eval_results_file, 'r') as f:
        eval_results = json.load(f)

    found_file = None
    for eval_file_path in eval_results:
        if same_path(file_path, eval_file_path):
            found_file = eval_file_path
            break

    if not found_file:
        agent.logger.error(f"File {file_path} not found in eval results")
        return False

    result = eval_results[found_file]
    status_key = 'plus_status' if use_plus else 'base_status'
    is_passed = result.get(status_key) == 'pass'

    agent.logger.info(f"Ground truth status is {result.get(status_key)} for {found_file}")
    return is_passed


def analyze_execution_results(agent, selected_file: Dict, completion_conf: int) -> bool:
    """Use LLM to analyze execution results and determine if work is complete"""

    execution_data = agent.version_manager.get_data(selected_file['path'], 'exec_result')

    if execution_data:
        exit_code = execution_data.get('exec_exit_code', 'unknown')
        output = execution_data.get('exec_output', 'No output available')
        analysis = execution_data.get('exec_analysis', 'No analysis available')
    else:
        exit_code = selected_file.get('exec_exit_code', 'unknown')
        output = selected_file.get('exec_output', 'No output available')
        analysis = selected_file.get('exec_analysis', 'No analysis available')

    min_completion_conf = 0
    low_completion_conf = completion_conf // 2
    medium_completion_conf = completion_conf - 1
    high_completion_conf = completion_conf
    max_completion_conf = 100

    prompt = f"""Examine the test execution results and analysis, use them to determine if the code and tests are now correct and complete.

# Test exit code: {exit_code}

# Test execution output:
```
{output}
```

# Test analysis:
{analysis}

Based on the execution results and analysis, please determine how confident you are that the work is complete.

Consider:
1. Are all tests passing successfully?
2. Is the code functioning correctly?
3. Are there any remaining issues that need to be addressed?
4. Is the work on this file complete and ready to be marked as finished?

Respond with a JSON object:
{{
"confidence": 0-100,
"reasoning": "Explanation of your assessment",
"remaining_issues": "Short summary of remaining issues",
}}

For confidence (0-100):
- Represents how confident you are that the work is COMPLETE
- {max_completion_conf} = absolutely certain work is complete
- {high_completion_conf}+ = high confidence work is complete, should be flagged as finished
- {low_completion_conf} - {medium_completion_conf} = moderate confidence, some uncertainty remains
- {min_completion_conf} - {low_completion_conf} = low confidence, significant issues likely remain

Only give a confidence that is {completion_conf} or above if:
- Exit code is 0 (success)
- All tests are passing (unless the failing test(s) are clearly incorrect or unnecessary)
- No critical errors or failures
- Code appears to be functioning correctly
"""

    response = agent.chat_completion(
        prompt=prompt,
        system_message=agent.role,
        response_format={"type": "json_object"}
    )

    result = agent.parse_json_response(response)

    if not result or "error" in result:
        agent.logger.error(f"Error parsing LLM analysis response: {result}")
        return False

    confidence = result.get('confidence', 0)
    reasoning = result.get('reasoning', 'No reasoning provided')
    remaining_issues = result.get('remaining_issues', 'No issues provided')

    # Validate and convert confidence to number
    if isinstance(confidence, str):
        confidence = float(confidence) if confidence.replace('.', '').isdigit() else 0
    elif not isinstance(confidence, (int, float)):
        confidence = 0

    confidence = max(0, min(100, float(confidence)))
    work_complete = confidence >= completion_conf

    progress = {
        'work_complete': work_complete,
        'confidence': confidence,
        'reasoning': reasoning,
        'remaining_issues': remaining_issues,
        'agent_id': agent.get_id()
    }

    agent.version_manager.store_data(selected_file['path'], 'progress', progress)

    agent.logger.info(f"Confidence: {confidence}/100, Complete: \
        {work_complete}, Reasoning: {reasoning}")

    return work_complete


def check_max_versions_reached(agent, file_path: str, max_versions: int) -> bool:
    """Check if file has reached maximum number of versions"""

    versions = agent.version_manager.load_version(file_path,
        k=max_versions + 1, include_content=False)

    if versions is None:
        version_count = 0
    elif isinstance(versions, list):
        version_count = len(versions)
    else:
        version_count = 1

    if version_count >= max_versions:
        agent.logger.info(f"Max versions ({max_versions}) reached for {file_path}")
        return True

    return False


class AdvancedResetAction(Action):
    """Advanced reset action that analyzes execution results and manages file flags appropriately"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = get_logger()
        check_attrs(self, ['completion_confidence', 'max_versions'])
        self.completion_confidence = max(min(self.completion_confidence, 100), 1)
        self.max_versions = max(self.max_versions, 1)

    def summary(self, agent) -> str:
        """Return a short summary of the advanced reset action"""
        return ("analyze test execution results using LLM to determine if code/tests are correct, "
                "mark code/tests as finished if correct, and return to initial state "
                "to start a new coding task")


    def execute(self, agent, **kwargs) -> bool:
        """Execute advanced reset with analysis and appropriate flagging"""

        selected_file = agent.context['selected_file']
        file_path = selected_file['path']

        if agent.use_ground_truth:
            work_complete = check_ground_truth(agent, file_path)
        else:
            work_complete = analyze_execution_results(agent, selected_file,
                self.completion_confidence) or \
            check_max_versions_reached(agent, file_path, self.max_versions)

        agent.version_manager.unflag_active(agent, file_path)

        if work_complete:
            agent.version_manager.flag_finished(agent, file_path)
            self.logger.info(f"Flagged {file_path} as finished")
        else:
            self.logger.error(f"Flagged {file_path} as not finished")

        # No need to clear agent context manually since idle state does that
        return work_complete
