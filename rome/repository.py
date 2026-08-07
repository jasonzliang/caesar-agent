import fnmatch
import glob
import os
import random
from typing import List, Dict
from pathlib import Path

from .logger import get_logger
from .config import LOG_DIR_NAME, META_DIR_EXT, TEST_FILE_EXT, set_attributes_from_config


class RepositoryManager:
    """Manages repository file operations including discovery, filtering, and statistics"""

    def __init__(self, repository_path: str, config: Dict = None):
        """
        Initialize the RepositoryManager

        Args:
            repository_path: Path to the repository root
            config: Configuration dictionary for the repository manager
        """
        self.logger = get_logger()
        self.repository_path = repository_path
        self.logger.assert_true(os.path.exists(self.repository_path),
            f"Repository path does not exist: {self.repository_path}")

        # Set configuration attributes
        set_attributes_from_config(
            self,
            config,
            required_attrs=['file_types', 'max_files', 'exclude_dirs', 'exclude_types'],
            optional_attrs=['easy_check_finish']
        )

        # Set to True/False value
        self.easy_check_finish == bool(self.easy_check_finish)

        # Store config for reference
        self.config = config or {}

        # Add default exclusions if not already present
        self.logger.assert_true(len(self.file_types) > 0,
            f"Provided file types is empty: {self.file_types}")
        if LOG_DIR_NAME not in self.exclude_dirs:
            self.exclude_dirs.append(LOG_DIR_NAME)
        if f"*.{META_DIR_EXT}" not in self.exclude_dirs:
            self.exclude_dirs.append(f"*.{META_DIR_EXT}")
        if TEST_FILE_EXT not in self.exclude_types:
            self.exclude_types.append(TEST_FILE_EXT)

        # Statistics tracking
        self.stats = {
            'total_files_found': 0,
            'files_after_filtering': 0,
            'excluded_by_dirs': 0,
            'excluded_by_types': 0,
            'excluded_by_flags': 0,
            'excluded_by_limit': 0
        }

    def collect_all_files(self) -> List[str]:
        """
        Collect all files matching the specified file types

        Returns:
            List of file paths
        """
        all_files = []

        for file_type in self.file_types:
            # Ensure file_type has a leading dot if needed
            if not file_type.startswith('.'):
                file_type = '.' + file_type

            search_path = os.path.join(self.repository_path, f'**/*{file_type}')
            self.logger.info(f"Searching for files in: {search_path}")

            files = glob.glob(search_path, recursive=True)
            self.logger.info(f"Found {len(files)} {file_type} files")
            all_files.extend(files)

        self.stats['total_files_found'] = len(all_files)
        return all_files

    def apply_filters(self, agent, all_files: List[str]) -> List[str]:
        """
        Apply all file filters in sequence

        Args:
            agent: Agent instance for checking file flags

        Returns:
            Filtered list of file paths
        """
        filters = [
            ("excluded directories", lambda files: self._filter_excluded_dirs(files)),
            ("excluded types", lambda files: self._filter_excluded_types(files)),
            ("flagged files", lambda files: self._filter_flagged_files(files, agent)),
            ("max limit", lambda files: self._filter_max_limit(files))
        ]

        filtered_files = all_files
        for filter_name, filter_func in filters:
            original_count = len(filtered_files)
            filtered_files = filter_func(filtered_files)
            excluded_count = original_count - len(filtered_files)

            # Update statistics
            if filter_name == "excluded directories":
                self.stats['excluded_by_dirs'] = excluded_count
            elif filter_name == "excluded types":
                self.stats['excluded_by_types'] = excluded_count
            elif filter_name == "flagged files":
                self.stats['excluded_by_flags'] = excluded_count
            elif filter_name == "max limit":
                self.stats['excluded_by_limit'] = excluded_count

            self.logger.info(f"Found {len(filtered_files)} files after {filter_name} filtering")

        self.stats['files_after_filtering'] = len(filtered_files)
        return filtered_files

    def _filter_excluded_dirs(self, files: List[str]) -> List[str]:
        """
        Filter out files in excluded directories

        Args:
            files: List of file paths
            exclude_dirs: List of directory patterns to exclude

        Returns:
            Filtered list of file paths
        """
        if not self.exclude_dirs:
            return files

        # Preprocess patterns for faster matching
        patterns = [p.replace('\\', '/') for p in self.exclude_dirs]

        def is_excluded(file_path: str) -> bool:
            norm_path = file_path.replace('\\', '/')
            path_parts = norm_path.split('/')

            # Check each directory component once
            for part in path_parts[:-1]:  # Exclude filename
                for pattern in patterns:
                    if fnmatch.fnmatch(part, pattern):
                        return True

            # Only check path patterns if they contain '/'
            path_patterns = [p for p in patterns if '/' in p]
            if path_patterns:
                for i in range(len(path_parts) - 1):
                    segment = '/'.join(path_parts[:i+1])
                    for pattern in path_patterns:
                        if fnmatch.fnmatch(segment, pattern) or fnmatch.fnmatch(norm_path, f"*{pattern}*"):
                            return True

            return False

        return [f for f in files if not is_excluded(f)]

    def _filter_excluded_types(self, files: List[str]) -> List[str]:
        """
        Filter out files with excluded file types

        Args:
            files: List of file paths
            exclude_types: List of file extensions to exclude

        Returns:
            Filtered list of file paths
        """
        if not self.exclude_types:
            return files

        return [f for f in files if not any(f.endswith(exclude_type) for exclude_type in self.exclude_types)]

    def _filter_flagged_files(self, files: List[str], agent) -> List[str]:
        """
        Filter out files that are currently active or already finished

        Args:
            files: List of file paths
            agent: Agent instance with version_manager

        Returns:
            Filtered list of file paths
        """
        if not files:
            return files

        filtered_files = []
        for file_path in files:
            is_available = agent.version_manager.check_avaliable(file_path)
            is_finished = agent.version_manager.check_finished(agent, file_path,
                easy_mode=self.easy_check_finish)

            if not is_available:
                self.logger.debug(f"Filtering out active file: {file_path}")
            elif is_finished:
                self.logger.debug(f"Filtering out finished file: {file_path}")
            else:
                filtered_files.append(file_path)

        return filtered_files

    def _filter_max_limit(self, files: List[str]) -> List[str]:
        """
        Filter out files if they exceed max file limit

        Args:
            files: List of file paths
            max_files: Maximum number of files to keep

        Returns:
            Filtered list of file paths (randomly sampled if over limit)
        """
        if len(files) > self.max_files:
            random.shuffle(files)
            return files[:self.max_files]
        return files

    def collect_and_filter_files(self, agent) -> List[str]:
        """
        Convenience method that combines file collection and filtering

        Returns:
            Filtered list of file paths
        """
        # Reset statistics
        self.stats = {key: 0 for key in self.stats}

        # Collect all files
        all_files = self.collect_all_files()
        self.logger.info(f"Found {len(all_files)} total files before filtering")

        # Apply filters
        filtered_files = self.apply_filters(agent, all_files)

        return filtered_files

    # === MOVED FROM VERSION MANAGER ===

    def get_repository_completion_stats(self, agent) -> Dict[str, int]:
        """
        Check overall completion status across all Python files in the repository.

        Args:
            agent: Agent instance with version_manager for checking finished status
            verbose: Whether to print stats to console/log

        Returns:
            Dictionary with completion statistics
        """
        # Use collect_all_files to get all Python files
        all_files = self.collect_all_files()

        # Apply directory and type filters but not file flags or max_files limit
        filtered_files = self._filter_excluded_dirs(all_files)
        filtered_files = self._filter_excluded_types(filtered_files)

        total_count = len(filtered_files)
        finished_count = 0

        # Check finished status for each file
        for file_path in filtered_files:
            if agent.version_manager.check_finished(agent, file_path):
                finished_count += 1

        unfinished_count = total_count - finished_count
        completion_percentage = (finished_count / total_count * 100) if total_count > 0 else 0

        result = {
            'finished_files': finished_count,
            'unfinished_files': unfinished_count,
            'total_files': total_count,
            'completion_percentage': round(completion_percentage, 2)
        }

        self.logger.info(f"Checked completion status in repository: {self.repository_path}")
        self.logger.info(f"Repository progress: {finished_count}/{total_count} files ({completion_percentage:.2f}%)")
        return result

    # === END MOVED METHODS ===

    def get_statistics(self) -> Dict:
        """
        Get current repository statistics

        Returns:
            Dictionary containing repository statistics
        """
        return self.stats.copy()

    def get_file_info(self, file_path: str) -> Dict:
        """
        Get detailed information about a specific file

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file information
        """
        try:
            stats = os.stat(file_path)
            return {
                'path': file_path,
                'size_bytes': stats.st_size,
                'size_kb': round(stats.st_size / 1024, 2),
                'modified_timestamp': stats.st_mtime,
                'exists': True
            }
        except Exception as e:
            self.logger.warning(f"Could not get info for file {file_path}: {e}")
            return {
                'path': file_path,
                'exists': False,
                'error': str(e)
            }
