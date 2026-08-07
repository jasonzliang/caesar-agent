import fnmatch
import glob
import os
from pathlib import Path
import random
import re
import sys
import time
import pprint
from typing import Dict, Optional, Any, Union, List

from .action import Action
from ..repository import RepositoryManager
from ..logger import get_logger
from ..config import LOG_DIR_NAME, META_DIR_EXT, SUMMARY_LENGTH, TEST_FILE_EXT
from ..config import check_attrs
from ..parsing import extract_all_definitions
from ..state import truncate_text

class PrioritySearchAction(Action):
    """Action to search repository for code files using completion confidence-based prioritization"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.logger = get_logger()
        check_attrs(self, ['selection_criteria', 'batch_size', 'randomness'])

    def summary(self, agent) -> str:
        """Return a detailed summary of the search action"""
        file_types_str = ', '.join(agent.repository_manager.file_types)
        exclude_dirs = agent.repository_manager.exclude_dirs
        max_files = agent.repository_manager.max_files
        excluded_dirs_str = ', '.join(exclude_dirs) if exclude_dirs else 'none'

        return (
            f"Completion confidence-based repository search for {file_types_str} files: "
            f"(1) scan repository and extract metadata, "
            f"(2) LLM prioritizes {self.batch_size} files 1-10 based on completion confidence "
            f"(lower confidence = higher priority), version count, and definitions, "
            f"(3) select best match from prioritized files after full content analysis "
            f"[max files: {max_files}, excluding: {excluded_dirs_str}]"
        )

    def _get_file_metadata(self, file_path: str, agent) -> Dict[str, Any]:
        """Get comprehensive file metadata including completion confidence and version count"""
        try:
            stats = os.stat(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            all_definitions = extract_all_definitions(content)

            # Get completion confidence from progress data
            progress_data = agent.version_manager.get_data(file_path, 'progress')
            completion_confidence = progress_data.get('confidence', 0) if progress_data else 0

            # Get version count
            versions = agent.version_manager.load_version(file_path, k=sys.maxsize,
                include_content=False)
            version_count = max(len(versions), 1) if isinstance(versions, list) else 1

            return {
                "path": file_path,
                "content": content,
                "size_kb": round(stats.st_size / 1024, 2),
                "modified_age": round(time.time() - stats.st_mtime, 1),
                "all_definitions": all_definitions,
                "definition_count": len(all_definitions),
                "function_count": len([d for d in all_definitions if d['type'] == 'function']),
                "class_count": len([d for d in all_definitions if d['type'] == 'class']),
                "completion_confidence": completion_confidence,
                "version_count": version_count,
            }
        except Exception as e:
            self.logger.error(f"Failed to extract file metadata from {file_path} due to error: {e}")
            return None

    def _create_definition_summary(self, definition: Dict) -> str:
        """Create a concise summary of a function/class definition"""
        summary = definition['signature']
        if definition['docstring']:
            docstring_lines = [line.strip() for line in definition['docstring'].split('\n') if line.strip()]
            if docstring_lines:
                docstring_text = ' '.join(docstring_lines)
                truncated_docstring = truncate_text(docstring_text)
                summary += f" # {truncated_docstring}"
        if definition['type'] == 'class' and definition.get('methods'):
            summary += f" ({len(definition['methods'])} methods)"
        return summary

    def _create_global_overview(self, agent, files: List[str]) -> List[Dict]:
        """Create a high-level overview of all files with completion metrics"""
        self.logger.info(f"Creating global overview of {len(files)} files with completion metrics")

        file_overviews = []
        for file_path in files:
            file_metadata = self._get_file_metadata(file_path, agent)
            if file_metadata:
                # Create definition summaries (limit to 10 for overview)
                definition_summaries = [
                    self._create_definition_summary(defn)
                    for defn in file_metadata['all_definitions'][:10]
                ]

                file_overviews.append({
                    **{k: v for k, v in file_metadata.items() if k != 'content'},
                    "definition_summaries": definition_summaries
                })

        return file_overviews

    def _create_prioritization_prompt(self, file_overviews: List[Dict], shuffle: bool = True) -> str:
        """Create prompt for completion confidence-based file prioritization"""
        if shuffle:
            random.shuffle(file_overviews)
        # FIXED: Extract exact file paths upfront
        valid_file_paths = [overview['path'] for overview in file_overviews]

        randomness_note = ""
        if self.randomness > 0:
            randomness_note = f"\n 4. **Score noise**: Add or subtract random score 0-{self.randomness} for exploration."

        prompt = f"""Return a JSON OBJECT with priority scores for EXACTLY these {len(file_overviews)} files.

Assign priority scores (1-10, 10 being highest) using these criteria:
1. **Completion confidence** (MOST IMPORTANT): Lower confidence = higher priority
2. **Version count**: Fewer versions = higher priority
3. **Function definitions**: More relevant functions = higher priority{randomness_note}
CRITICAL: You MUST use the EXACT file paths listed below. Do NOT modify, generate, or create new paths.

File details:
"""

        for j, file_info in enumerate(file_overviews):
            confidence = file_info['completion_confidence']
            version_count = file_info['version_count']
            prompt += f"\n--- Code filepath {j+1}: {file_info['path']} ---\n"
            prompt += f"Completion confidence: {confidence}%\n"
            prompt += f"Version count: {version_count}\n"
            prompt += f"Total definitions: {file_info['definition_count']} ({file_info['function_count']} functions, {file_info['class_count']} classes)\n"
            if file_info['definition_summaries']:
                prompt += "Code definitions:\n"
                for def_summary in file_info['definition_summaries']:
                    prompt += f"  {def_summary}\n"
            else:
                prompt += "No function/class definitions found\n"
        prompt += f"""
Return a JSON OBJECT with up to {min(len(file_overviews), self.batch_size)} files from the EXACT paths listed above:
{{
  "{valid_file_paths[0] if valid_file_paths else 'example/path.py'}": 9,
  "{valid_file_paths[1] if len(valid_file_paths) > 1 else 'example/path2.py'}": 8
}}
REQUIREMENTS:
- Use ONLY the exact file paths shown above
- Do NOT create, modify, or generate new file paths
- Do NOT use paths not explicitly listed
"""
        return prompt

    def _prioritize_files(self, agent, file_overviews: List[Dict]) -> List[Dict]:
        """Use LLM to prioritize files with strict validation of returned paths"""

        # Extract valid file paths for validation
        valid_file_paths = {overview['path'] for overview in file_overviews}

        prompt = self._create_prioritization_prompt(file_overviews)
        response = agent.chat_completion(
            prompt=prompt,
            system_message=agent.role,
            response_format={"type": "json_object"}
        )
        result = agent.parse_json_response(response)

        prioritized_files = []
        invalid_paths = []

        if isinstance(result, dict):
            for file_path, priority in result.items():
                # ADDED: Strict validation of file paths
                if file_path not in valid_file_paths:
                    invalid_paths.append(file_path)
                    self.logger.error(f"LLM returned invalid file path: {file_path}")
                    continue

                # ADDED: Verify file still exists
                if not os.path.exists(file_path):
                    self.logger.error(f"LLM returned non-existent file: {file_path}")
                    continue

                if isinstance(priority, (int, float)):
                    prioritized_files.append({
                        "file_path": file_path,
                        "priority": int(priority)
                    })

            # Log results
            if invalid_paths:
                self.logger.error(f"LLM hallucinated {len(invalid_paths)} invalid paths: {invalid_paths}")
            self.logger.info(f"Valid prioritized files: {len(prioritized_files)}")

        return sorted(prioritized_files, key=lambda x: x.get("priority", 0), reverse=True)

    def _prepare_batch_data(self, batch_file_info: List[Dict], agent) -> List[Dict]:
        """Prepare batch data with full file content and metadata"""
        batch_data = []
        for file_info in batch_file_info:
            file_path = file_info["file_path"]
            file_metadata = self._get_file_metadata(file_path, agent)
            if file_metadata:
                batch_data.append({
                    **file_metadata,
                    'priority': file_info.get("priority", 1)
                })
        return batch_data

    def _create_batch_overview(self, batch_data: List[Dict]) -> str:
        """Create overview section for batch selection prompt"""
        overview = "File list:\n"
        for i, file_info in enumerate(batch_data):
            filename = os.path.basename(file_info['path'])
            confidence = file_info['completion_confidence']
            version_count = file_info['version_count']
            def_count = len(file_info['all_definitions'])
            func_count = file_info['function_count']
            class_count = file_info['class_count']

            overview += (f"- File {i+1}: {filename} (Priority: {file_info['priority']}/10, "
                        f"Completion confidence: {confidence}%, Versions: {version_count}, "
                        f"{def_count} total: {func_count} functions, {class_count} classes)\n")
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

    def _process_batch(self, agent, batch_file_info: List[Dict]) -> Optional[Dict]:
        """Process a batch of files and return selected file if found"""
        self.logger.info(f"Processing batch: {len(batch_file_info)} files")

        current_batch_data = self._prepare_batch_data(batch_file_info, agent)
        if not current_batch_data:
            self.logger.error(f"No readable files in file batch")
            return None

        # Get LLM selection
        prompt = self._create_selection_prompt(current_batch_data)
        response = agent.chat_completion(
            prompt=prompt,
            system_message=agent.role,
            response_format={"type": "json_object"}
        )
        result = agent.parse_json_response(response)
        selected_info = result.get('selected_file')

        if not result or "error" in result or not selected_info:
            self.logger.error(f"Error parsing response for file batch: {result.get('error', 'Unknown')}")
            return None

        file_number = selected_info.get('file_number')
        if not (file_number and 1 <= file_number <= len(current_batch_data)):
            self.logger.error(f"Invalid file_number {file_number} for file batch")
            return None

        file_data = current_batch_data[file_number - 1]
        selected_file = {
            'path': file_data['path'],
            'content': file_data['content'],
            'reason': selected_info.get('reason', 'Selected by LLM'),
            # 'priority': file_data['priority'],
            # 'all_definitions': file_data['all_definitions'],
            # 'completion_confidence': file_data['completion_confidence'],
            # 'version_count': file_data['version_count']
        }
        self.logger.info(f"Selected: {file_data['path']} (Completion confidence: {file_data['completion_confidence']}%, Versions: {file_data['version_count']}) - {selected_file['reason']}")
        return selected_file

    def execute(self, agent, **kwargs) -> bool:
        """Execute the completion confidence-based search action"""
        # Use the agent's repository manager to collect and filter files
        filtered_files = agent.repository_manager.collect_and_filter_files(agent=agent)

        if not filtered_files:
            self.logger.error("No files left after filtering")
            return False

        # Create overview, prioritize by completion confidence, and process
        file_overviews = self._create_global_overview(agent, filtered_files)
        prioritized_files = self._prioritize_files(agent, file_overviews)
        selected_file = self._process_batch(agent, prioritized_files)

        if selected_file:
            # Try to reserve the file with retry logic
            if agent.version_manager.try_reserve_file(agent, selected_file['path']):
                agent.version_manager.save_original(selected_file['path'], selected_file['content'])
                agent.context['selected_file'] = selected_file
                return True
            else:
                self.logger.error(f"Could not reserve {selected_file['path']} after multiple attempts")
                return False
        else:
            self.logger.error("Search completed but no file was selected by LLM")
            return False
