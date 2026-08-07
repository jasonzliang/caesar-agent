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
    """Action to search repository for code files using completion confidence-based scoring"""

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
            f"(1) score files by completion confidence (low confidence = high score), "
            f"(2) sample {self.batch_size} files weighted by score, "
            f"(3) LLM selects best match from batch "
            f"[max files: {max_files}, excluding: {excluded_dirs_str}]"
        )

    def _get_file_metadata(self, file_path: str, agent) -> Dict[str, Any]:
        """Get file metadata including completion confidence"""
        try:
            stats = os.stat(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            all_definitions = extract_all_definitions(content)

            # Get completion confidence from progress data
            progress_data = agent.version_manager.get_data(file_path, 'progress')
            completion_confidence = progress_data.get('confidence', 0) if progress_data else 0

            # Get version count
            versions = agent.version_manager.load_version(file_path, k=sys.maxsize, include_content=False)
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
            self.logger.error(f"Failed to extract file metadata from {file_path}: {e}")
            return None

    def _calculate_priority_score(self, file_metadata: Dict) -> float:
        """Calculate priority score based on completion confidence (low confidence = high score)"""
        confidence = file_metadata['completion_confidence']

        # Base score: invert confidence (0% confidence = 100 score, 100% confidence = 0 score)
        base_score = 100 - confidence

        # Optional: Add small bonuses for other factors
        version_bonus = max(0, 10 - file_metadata['version_count'])  # Fewer versions = small bonus
        definition_bonus = min(10, file_metadata['definition_count'])  # More definitions = small bonus

        total_score = base_score + version_bonus * 0.1 + definition_bonus * 0.1

        # Add randomness if configured
        if self.randomness > 0:
            noise = random.uniform(-self.randomness, self.randomness)
            total_score += noise

        return max(0, int(total_score))

    def _sample_batch_files(self, file_metadatas: List[Dict]) -> List[Dict]:
        """Sample batch_size files weighted by priority score"""
        # Calculate scores for all files first
        scored_files = []
        for metadata in file_metadatas:
            score = self._calculate_priority_score(metadata)
            scored_files.append({**metadata, 'priority_score': score})

        if len(scored_files) <= self.batch_size:
            self.logger.info(f"Using all {len(scored_files)} files (less than batch size)")
            return scored_files

        # Sort by score and take top candidates for weighted sampling
        # scored_files.sort(key=lambda x: x['priority_score'], reverse=True)

        # Use top 2x batch_size files for weighted sampling to balance exploration vs exploitation
        # candidates = scored_files[:min(len(scored_files), self.batch_size * 2)]
        candidates = scored_files
        weights = [f['priority_score'] for f in candidates]

        # Weighted sampling without replacement
        sampled = random.choices(candidates, weights=weights, k=min(self.batch_size, len(candidates)))

        self.logger.info(f"Sampled {len(sampled)} files from {len(file_metadatas)} total files")
        for f in sampled:
            self.logger.info(f"  {f['path']}: confidence={f['completion_confidence']}%, score={f['priority_score']:.1f}")

        return sampled

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

    def _create_batch_overview(self, batch_data: List[Dict]) -> str:
        """Create overview section for batch selection prompt"""
        overview = "Available files:\n"
        for i, file_info in enumerate(batch_data):
            filename = os.path.basename(file_info['path'])
            confidence = file_info['completion_confidence']
            score = file_info['priority_score']
            def_count = file_info['definition_count']
            func_count = file_info['function_count']
            class_count = file_info['class_count']

            overview += (f"- File {i+1}: {filename} "
                        f"(Completion confidence: {confidence}%, Score: {score:.1f}, "
                        f"{def_count} definitions: {func_count} functions, {class_count} classes)\n")
        return overview

    def _create_selection_prompt(self, batch_data: List[Dict]) -> str:
        """Create prompt for LLM file selection"""
        prompt = f"""Select the most relevant file based on your role and criteria: {self.selection_criteria}

Files are scored by completion confidence (lower confidence = higher priority for improvement).

{self._create_batch_overview(batch_data)}

Detailed file contents:"""

        for i, file_info in enumerate(batch_data):
            prompt += f"\n\n--- File {i+1}: {file_info['path']} ---\n"

            # # Add definition summaries if available
            # if file_info['all_definitions']:
            #     prompt += "Key definitions:\n"
            #     for defn in file_info['all_definitions'][:5]:  # Limit to top 5
            #         prompt += f"  {self._create_definition_summary(defn)}\n"
            #     prompt += "\n"

            prompt += file_info['content']
            prompt += "\n---\n"

        prompt += """
Respond with JSON:
{
    "selected_file": {
        "file_number": <int>,
        "path": "<exact_file_path>",
        "reason": "<brief_selection_reason>"
    }
}

Set "selected_file" to null if no file meets your criteria.
"""
        return prompt

    def execute(self, agent, **kwargs) -> bool:
        """Execute the completion confidence-based search action"""
        # Collect and filter files
        filtered_files = agent.repository_manager.collect_and_filter_files(agent=agent)
        if not filtered_files:
            self.logger.error("No files found after filtering")
            return False

        # Get metadata for all files
        file_metadatas = []
        for file_path in filtered_files:
            metadata = self._get_file_metadata(file_path, agent)
            if metadata:
                file_metadatas.append(metadata)

        if not file_metadatas:
            self.logger.error("No readable files found")
            return False

        # Sample batch based on completion confidence scores
        batch_files = self._sample_batch_files(file_metadatas)

        # Create selection prompt and get LLM choice
        prompt = self._create_selection_prompt(batch_files)
        response = agent.chat_completion(
            prompt=prompt,
            system_message=agent.role,
            response_format={"type": "json_object"}
        )

        result = agent.parse_json_response(response)
        selected_info = result.get('selected_file')

        if not selected_info:
            self.logger.error("No file selected by LLM")
            return False

        file_number = selected_info.get('file_number')
        if not (file_number and 1 <= file_number <= len(batch_files)):
            self.logger.error(f"Invalid file_number {file_number}")
            return False

        # Get selected file data
        selected_metadata = batch_files[file_number - 1]
        selected_file = {
            'path': selected_metadata['path'],
            'content': selected_metadata['content'],
            'reason': selected_info.get('reason', 'Selected by LLM'),
        }

        self.logger.info(f"Selected: {selected_file['path']} "
                        f"(Completion confidence: {selected_metadata['completion_confidence']}%, "
                        f"Score: {selected_metadata['priority_score']:.1f}) - {selected_file['reason']}")

        # Try to reserve the file
        if agent.version_manager.try_reserve_file(agent, selected_file['path']):
            agent.version_manager.save_original(selected_file['path'], selected_file['content'])
            agent.context['selected_file'] = selected_file
            return True
        else:
            self.logger.error(f"Could not reserve {selected_file['path']}")
            return False
