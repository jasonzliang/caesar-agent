import fnmatch
import glob
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Dict, Optional, Any, Union, List

from .action import Action
from ..repository import RepositoryManager  # New import
from ..logger import get_logger
from ..config import LOG_DIR_NAME, META_DIR_EXT, SUMMARY_LENGTH, TEST_FILE_EXT
from ..config import check_attrs
from ..parsing import extract_all_definitions
from ..state import truncate_text

class PrioritySearchAction(Action):
    """Action to search the repository for code files using OpenAI selection"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.logger = get_logger()

        check_attrs(self, ['selection_criteria', 'batch_size'])

    def summary(self, agent) -> str:
        """Return a detailed summary of the search action"""
        file_types_str = ', '.join(agent.repository_manager.file_types)
        exclude_dirs = agent.repository_manager.exclude_dirs
        max_files = agent.repository_manager.max_files
        excluded_dirs_str = ', '.join(exclude_dirs) if exclude_dirs else 'none'

        # sampling_mode = "weighted sampling" if self.batch_sampling else "sequential"

        return (
            f"Multi-stage LLM repository search for {file_types_str} files: "
            f"(1) scan repository and extract metadata (size/age/definitions), "
            f"(2) LLM prioritizes {self.batch_size} files 1-10 based on '{self.selection_criteria}', "
            f"(3) select best match from prioritized files after full content analysis "
            f"[max files: {max_files}, excluding: {excluded_dirs_str}]"
        )

    def _get_file_stats(self, file_path: str) -> Dict[str, Any]:
        """Get file statistics and content analysis"""
        try:
            stats = os.stat(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            all_definitions = extract_all_definitions(content)

            return {
                "path": file_path,
                "content": content,
                "size_kb": round(stats.st_size / 1024, 2),
                "modified_age": round(time.time() - stats.st_mtime, 1),
                "all_definitions": all_definitions,
                "definition_count": len(all_definitions),
                "function_count": len([d for d in all_definitions if d['type'] == 'function']),
                "class_count": len([d for d in all_definitions if d['type'] == 'class']),
            }
        except Exception as e:
            self.logger.info(f"Error reading {file_path}: {e}")
            return None

    def _create_definition_summary(self, definition: Dict) -> str:
        """Create a concise summary of a function/class definition"""
        summary = definition['signature']

        if definition['docstring']:
            # Allow multiple lines but limit total characters to SUMMARY_LENGTH
            docstring_lines = [line.strip() for line in definition['docstring'].split('\n') if line.strip()]
            if docstring_lines:
                # Join lines with space and truncate to SUMMARY_LENGTH
                docstring_text = ' '.join(docstring_lines)
                truncated_docstring = truncate_text(docstring_text)
                summary += f" # {truncated_docstring}"

        if definition['type'] == 'class' and definition.get('methods'):
            method_count = len(definition['methods'])
            summary += f" ({method_count} methods)"

        return summary

    def _create_global_overview(self, agent, files: List[str]) -> List[Dict]:
        """Create a high-level overview of all files in the repository"""
        self.logger.info(f"Creating global overview of {len(files)} files")

        file_overviews = []
        for file_path in files:
            file_stats = self._get_file_stats(file_path)
            if file_stats:
                # Create definition summaries
                definition_summaries = [
                    self._create_definition_summary(defn)
                    for defn in file_stats['all_definitions'][:10]
                ]

                file_overviews.append({
                    **{k: v for k, v in file_stats.items() if k != 'content'},
                    "definition_summaries": definition_summaries
                })

        return file_overviews

    def _create_prioritization_prompt(self, file_overviews: List[Dict], shuffle: bool = True) -> str:
        """Create prompt for file prioritization"""
        if shuffle: random.shuffle(file_overviews)

        prompt = f"""Return a JSON OBJECT with ALL {len(file_overviews)} files with priority scores.

Assign a priority score (1-10, 10 being highest) using the following criteria in descending importance:
1. File modification age (older files get higher priority)
2. File size in KB (smaller files get higher priority)
3. Function definitions that match your role and selection criteria get higher priority

Files to prioritize:
"""

        for j, file_info in enumerate(file_overviews):
            prompt += f"\n--- File {j+1}: {file_info['path']} ---\n"
            prompt += f"Size: {file_info['size_kb']} KB\n"
            prompt += f"Last modified: {file_info['modified_age']} seconds ago\n"
            prompt += f"Total definitions: {file_info['definition_count']} ({file_info['function_count']} functions, {file_info['class_count']} classes)\n"

            if file_info['definition_summaries']:
                prompt += "Code definitions:\n"
                for def_summary in file_info['definition_summaries']:
                    prompt += f"  {def_summary}\n"
            else:
                prompt += "No function/class definitions found\n"

        prompt += f"""
Return a JSON OBJECT with {min(len(file_overviews), self.batch_size)} files like below. These files should be the ones which you think have the highest priority. For files with same priority, randomly choose one.
{{
  "path/to/file1.py": 8,
  "path/to/file2.py": 6,
  "path/to/file3.py": 3,
  ...
}}

IMPORTANT: Your response MUST be a valid JSON OBJECT where keys are file paths and values are priority scores (1-10).
"""
        return prompt

    def _prioritize_files(self, agent, file_overviews: List[Dict]) -> List[Dict]:
        """Use LLM to prioritize all files at once based on size, age, and function definitions"""
        self.logger.info(f"Prioritizing {min(len(file_overviews), self.batch_size)} files at once")

        prompt = self._create_prioritization_prompt(file_overviews)
        response = agent.chat_completion(prompt=prompt,
            system_message=agent.role,
            response_format={"type": "json_object"})
        result = agent.parse_json_response(response)

        # Handle simplified JSON object response format {file_path: priority}
        prioritized_files = []

        if isinstance(result, dict):
            # Convert simplified format to expected list format
            for file_path, priority in result.items():
                if isinstance(priority, (int, float)):
                    prioritized_files.append({
                        "file_path": file_path,
                        "priority": int(priority)
                    })
        else:
            # Invalid response - create default priorities
            self.logger.error("Invalid response format, using default priorities for all files")
            prioritized_files = [{
                "file_path": file_info["path"],
                "priority": 1
            } for file_info in file_overviews]

        return sorted(prioritized_files, key=lambda x: x.get("priority", 0), reverse=True)

    def _create_batch_overview(self, batch_data: List[Dict]) -> str:
        """Create overview section for batch selection prompt"""
        overview = "File list:\n"
        for i, file_info in enumerate(batch_data):
            filename = os.path.basename(file_info['path'])
            def_count = len(file_info['all_definitions'])
            func_count = file_info['function_count']
            class_count = file_info['class_count']
            overview += (f"- File {i+1}: {filename} (Priority: {file_info['priority']}/10, "
                        f"{def_count} total: {func_count} functions, {class_count} classes) "
                        f"- {file_info['priority_reason']}\n")
        return overview

    def _create_selection_prompt(self, batch_data: List[Dict]) -> str:
        """Create a concise prompt for file selection"""
        prompt = f"""Select the most relevant file based on your role and selection criteria: {self.selection_criteria}

{self._create_batch_overview(batch_data)}

Detailed file contents:"""

        for i, file_info in enumerate(batch_data):
            prompt += f"\n\n--- File {i+1}: {file_info['path']} ---\n{file_info['content']}"

        prompt += """

Respond with a JSON object:
{
    "selected_file": {
        "file_number": <int>,
        "path": "<file_path>",
        "reason": "<brief reason>"
    }
}

- Set "selected_file" to null if no file meets the criteria
"""
        return prompt

    def _prepare_batch_data(self, batch_file_info: List[Dict]) -> List[Dict]:
        """Prepare batch data with file content and metadata"""
        batch_data = []

        for file_info in batch_file_info:
            file_path = file_info["file_path"]
            file_stats = self._get_file_stats(file_path)
            if not file_stats:
                continue

            batch_data.append({
                **file_stats,
                'priority': file_info.get("priority", 1),
                'priority_reason': file_info.get("reason", "")
            })

        return batch_data

    # def _weighted_random_sample(self, prioritized_files: List[Dict], batch_size: int) -> List[Dict]:
    #     """Sample complete file objects using weighted random sampling"""
    #     weights = [file_info.get("priority", 1) for file_info in prioritized_files]
    #     sample_size = min(batch_size, len(prioritized_files))

    #     sampled_indices = set()
    #     sampled_files = []

    #     for _ in range(sample_size):
    #         available_indices = [i for i in range(len(prioritized_files)) if i not in sampled_indices]
    #         if not available_indices:
    #             break

    #         available_weights = [weights[i] for i in available_indices]
    #         total_weight = sum(available_weights)

    #         if total_weight == 0:
    #             chosen_idx = random.choice(available_indices)
    #         else:
    #             probabilities = [w/total_weight for w in available_weights]
    #             chosen_relative_idx = random.choices(range(len(available_indices)),
    #                 weights=probabilities)[0]
    #             chosen_idx = available_indices[chosen_relative_idx]

    #         sampled_indices.add(chosen_idx)
    #         sampled_files.append(prioritized_files[chosen_idx])

    #     return sampled_files

    def _process_single_batch(self, agent, batch_file_info: List[Dict], batch_info: str) -> Optional[Dict]:
        """Process a single batch of files and return selected file if found"""
        self.logger.info(f"Processing {batch_info}: {len(batch_file_info)} files")

        current_batch_data = self._prepare_batch_data(batch_file_info)
        if not current_batch_data:
            self.logger.info(f"No readable files in {batch_info}")
            return None

        # Get LLM selection
        prompt = self._create_selection_prompt(current_batch_data)
        response = agent.chat_completion(prompt=prompt,
            system_message=agent.role,
            response_format={"type": "json_object"})
        result = agent.parse_json_response(response)

        if not result or "error" in result:
            self.logger.error(f"Error parsing response for {batch_info}: {result.get('error', 'Unknown')}")
            return None

        selected_info = result.get('selected_file')
        if not selected_info:
            return None

        file_number = selected_info.get('file_number')
        if not (file_number and 1 <= file_number <= len(current_batch_data)):
            self.logger.error(f"Invalid file_number {file_number} for {batch_info}")
            return None

        file_data = current_batch_data[file_number - 1]
        selected_file = {
            'path': file_data['path'],
            'content': file_data['content'],
            'reason': selected_info.get('reason', 'Selected by LLM'),
            'priority': file_data['priority'],
            'all_definitions': file_data['all_definitions']
        }

        self.logger.info(f"Selected: {file_data['path']} - {selected_file['reason']}")
        return selected_file

    # def _process_file_batches(self, agent, prioritized_files: List[Dict]) -> Optional[Dict]:
    #     """Process batches of files to find the most relevant match"""
    #     self.logger.info(f"Processing batches using {'weighted sampling' if self.batch_sampling else 'sequential'} mode")

    #     if self.batch_sampling:
    #         remaining_files = prioritized_files.copy()
    #         batch_count = 0

    #         while remaining_files:
    #             batch_count += 1
    #             batch_file_info = self._weighted_random_sample(remaining_files, self.batch_size)

    #             # Remove sampled files
    #             for file_info in batch_file_info:
    #                 remaining_files.remove(file_info)

    #             selected_file = self._process_single_batch(agent, batch_file_info, f"sampled batch {batch_count}")
    #             if selected_file:
    #                 return selected_file
    #     else:
    #         total_batches = (len(prioritized_files) + self.batch_size - 1) // self.batch_size

    #         for batch_idx in range(total_batches):
    #             start_idx = batch_idx * self.batch_size
    #             end_idx = min((batch_idx + 1) * self.batch_size, len(prioritized_files))
    #             batch_file_info = prioritized_files[start_idx:end_idx]

    #             batch_info = f"batch {batch_idx + 1}/{total_batches}"
    #             selected_file = self._process_single_batch(agent, batch_file_info, batch_info)
    #             if selected_file:
    #                 return selected_file

    #     return None

    def execute(self, agent, **kwargs) -> bool:
        """Execute the search action"""

        # Use the agent's repository manager to collect and filter files
        filtered_files = agent.repository_manager.collect_and_filter_files(agent=agent)

        if not filtered_files:
            self.logger.error("No files left after filtering")
            return False

        # Create overview, prioritize, and process batches
        file_overviews = self._create_global_overview(agent, filtered_files)
        prioritized_files = self._prioritize_files(agent, file_overviews)
        selected_file = self._process_single_batch(agent, prioritized_files, "single batch")
        # selected_file = self._process_file_batches(agent, prioritized_files)

        if selected_file:
            # Try to reserve the file with retry logic (3 attempts with backoff)
            if agent.version_manager.try_reserve_file(agent, selected_file['path']):
                agent.version_manager.save_original(selected_file['path'], selected_file['content'])
                agent.context['selected_file'] = selected_file
                self.logger.info(f"Search completed with selected file: {selected_file['path']}")
                return True
            else:
                # All attempts failed
                self.logger.error(f"Could not reserve {selected_file['path']} after multiple attempts")
                return False
        else:
            self.logger.error("Search completed but no file was selected by LLM")
            return False
