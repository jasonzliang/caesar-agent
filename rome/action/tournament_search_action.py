import random
import os
import time
import sys
from typing import Dict, List, Optional, Any

from .action import Action
from ..logger import get_logger
from ..config import check_attrs
from ..parsing import extract_all_definitions


class TournamentSearchAction(Action):
    """Tournament-style file selection: random K files → display full content → LLM picks best"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.logger = get_logger()
        check_attrs(self, ['selection_criteria', 'batch_size'])

    def summary(self, agent) -> str:
        """Return a detailed summary of the tournament search action"""
        repo_config = agent.repository_manager.config
        file_types_str = ', '.join(repo_config.get('file_types', ['.py']))
        exclude_dirs = repo_config.get('exclude_dirs', [])
        max_files = repo_config.get('max_files', 'unlimited')
        excluded_dirs_str = ', '.join(exclude_dirs) if exclude_dirs else 'none'

        # Handle fractional vs fixed tournament size
        if self.batch_size < 1:
            selection_desc = f"{self.batch_size:.1%} of available files"
        else:
            selection_desc = f"{int(self.batch_size)} files"

        return (
            f"Tournament search for {file_types_str} files: "
            f"(1) randomly select {selection_desc}, "
            f"(2) analyze full contents, "
            f"(3) LLM selects best match for '{self.selection_criteria}' "
            f"[max files: {max_files}, excluding: {excluded_dirs_str}]"
        )

    def _get_file_metadata(self, file_path: str, agent) -> Dict[str, Any]:
        """Get file metadata including completion confidence and version data"""
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

    def _randomly_select_files(self, all_files: List[str]) -> List[str]:
        """Step 1: Randomly choose K files or fraction of files from all available files"""
        if not all_files:
            self.logger.warning("No files available for tournament selection")
            return []

        if self.batch_size < 1:
            # Fractional selection: tournament_size is a percentage (0.0 - 1.0)
            fraction = self.batch_size
            k = max(1, int(len(all_files) * fraction))
        else:
            # Fixed number selection
            k = min(int(self.batch_size), len(all_files))

        selected_files = random.sample(all_files, k)

        self.logger.info(f"Randomly selected {len(selected_files)} files from {len(all_files)} total files")
        for i, file_path in enumerate(selected_files, 1):
            self.logger.debug(f"  {i}. {file_path}")

        return selected_files

    def _prepare_tournament_data(self, selected_files: List[str], agent) -> List[Dict]:
        """Step 2: Load full content and metadata for each selected file"""
        tournament_data = []

        for file_path in selected_files:
            file_metadata = self._get_file_metadata(file_path, agent)
            if file_metadata:
                tournament_data.append(file_metadata)
            else:
                self.logger.warning(f"Could not read file: {file_path}")

        self.logger.info(f"Successfully loaded {len(tournament_data)} files for tournament")
        return tournament_data

    def _create_tournament_prompt(self, tournament_data: List[Dict]) -> str:
        """Step 3: Create prompt with full file contents for LLM selection"""
        prompt = f"""Select the most relevant file based on your role and selection criteria: {self.selection_criteria}

Tournament Files ({len(tournament_data)} candidates):
"""

        # Add file overview with enhanced metadata
        for i, file_info in enumerate(tournament_data, 1):
            filename = os.path.basename(file_info['path'])
            confidence = file_info['completion_confidence']
            version_count = file_info['version_count']
            def_count = file_info['definition_count']
            func_count = file_info['function_count']
            class_count = file_info['class_count']
            size_kb = file_info['size_kb']

            prompt += (f"- File {i}: {filename} "
                      f"({size_kb} KB, {def_count} definitions: {func_count} functions, {class_count} classes, "
                      f"completion confidence: {confidence}%, versions: {version_count})\n")

        prompt += "\nFull file contents:\n"

        # Add complete file contents
        for i, file_info in enumerate(tournament_data, 1):
            prompt += f"\n---\nFile {i}: {file_info['path']}\n---\n"
            prompt += file_info['content']
            prompt += "\n---\n"

        prompt += f"""
Based on the selection criteria "{self.selection_criteria}", choose the most relevant file.

Respond with a JSON object:
{{
    "selected_file": {{
        "file_number": <int>,
        "path": "<file_path>",
        "reason": "<brief explanation for why this file was selected>"
    }}
}}

- Set "selected_file" to null if no file meets the criteria
- file_number should be between 1 and {len(tournament_data)}
"""
        return prompt

    def _select_winner(self, agent, tournament_data: List[Dict]) -> Optional[Dict]:
        """Use LLM to select the best file from tournament participants"""
        prompt = self._create_tournament_prompt(tournament_data)

        try:
            response = agent.chat_completion(
                prompt=prompt,
                system_message=agent.role,
                response_format={"type": "json_object"}
            )
            result = agent.parse_json_response(response)

            if not result or "error" in result:
                self.logger.error(f"Error parsing LLM response: {result.get('error', 'Unknown')}")
                return None

            selected_info = result.get('selected_file')
            if not selected_info:
                self.logger.error("LLM did not select any file")
                return None

            file_number = selected_info.get('file_number')
            if not (file_number and 1 <= file_number <= len(tournament_data)):
                self.logger.error(f"Invalid file_number {file_number}")
                return None

            # Get the winning file
            winner_data = tournament_data[file_number - 1]
            selected_file = {
                'path': winner_data['path'],
                'content': winner_data['content'],
                'reason': selected_info.get('reason', 'Tournament winner'),
                'all_definitions': winner_data['all_definitions'],
                'tournament_size': len(tournament_data)
            }

            self.logger.info(f"Tournament winner: {winner_data['path']} "
                           f"(completion confidence: {winner_data['completion_confidence']}%, "
                           f"versions: {winner_data['version_count']}) - {selected_file['reason']}")
            return selected_file

        except Exception as e:
            self.logger.error(f"LLM selection failed: {e}")
            return None

    def execute(self, agent, **kwargs) -> bool:
        """Execute the tournament search action"""
        # Use the agent's repository manager to collect and filter files
        filtered_files = agent.repository_manager.collect_and_filter_files(agent=agent)

        if not filtered_files:
            self.logger.error("No files left after filtering")
            return False

        # Step 1: Randomly select K files
        selected_files = self._randomly_select_files(filtered_files)

        if not selected_files:
            self.logger.error("Search completed but no file was selected by agent.")
            return False

        # Step 2: Load full content and metadata for selected files
        tournament_data = self._prepare_tournament_data(selected_files, agent)

        if not tournament_data:
            self.logger.error("No readable files in tournament")
            return False

        # Step 3: LLM selects the best file
        selected_file = self._select_winner(agent, tournament_data)

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
