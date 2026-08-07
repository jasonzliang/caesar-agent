import datetime
import fnmatch
import random
import json
import os
import time
from typing import List, Dict, Any, Optional, Set
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

import psutil

from .config import LOG_DIR_NAME, META_DIR_EXT, TEST_FILE_EXT, set_attributes_from_config
from .logger import get_logger
from .parsing import hash_string
from .database import DatabaseManager, locked_file_operation, locked_json_operation, get_locking_config


class FileType(Enum):
    INDEX = "index.json"
    ACTIVE = "active.json"
    FINISHED = "finished.json"
    DATABASE = "database.json"  # TinyDB database file


@dataclass
class ValidationError:
    """Structured validation error information"""
    file_path: str
    field: str
    message: str
    value: Any = None


class VersionManager:
    """Manages code/test file activity, ownership, and versioning for the agent"""

    def __init__(self, config: Dict = None, db_config: Dict = None):
        self.config = config or {}
        self.logger = get_logger()
        self.active_files: Set[str] = set()

        set_attributes_from_config(
            self,
            config,
            required_attrs=['lock_active_file']
        )

        # Initialize TinyDBManager with database path function
        # This will also initialize the global locking configuration
        self.db = DatabaseManager(
            get_db_path_func=self._get_database_path,
            config=db_config
        )

    # Core utility methods
    def _get_timestamp(self) -> str:
        return datetime.datetime.now().isoformat()

    def _get_meta_dir(self, file_path: str) -> str:
        meta_dir = f"{file_path}.{META_DIR_EXT}"
        # Another process created the directory
        try:
            os.makedirs(meta_dir, exist_ok=True)
        except FileExistsError:
            pass
        return meta_dir

    def _get_test_meta_dir(self, test_file_path: str, main_file_path: str) -> str:
        main_meta_dir = self._get_meta_dir(main_file_path)
        test_filename = os.path.basename(test_file_path)
        test_meta_dir = os.path.join(main_meta_dir, f"{test_filename}.{META_DIR_EXT}")
        # FIXED: Use try-except to handle race conditions
        try:
            os.makedirs(test_meta_dir, exist_ok=True)
        except FileExistsError:
            # Another process created the directory
            pass
        return test_meta_dir

    def _get_file_path(self, meta_dir: str, file_type: FileType) -> str:
        return os.path.join(meta_dir, file_type.value)

    def _get_database_path(self, file_path: str) -> str:
        """Get the TinyDB database path for a given file."""
        meta_dir = self._get_meta_dir(file_path)
        return self._get_file_path(meta_dir, FileType.DATABASE)

    def _infer_main_file_from_test(self, test_file_path: str) -> Optional[str]:
        """Infer main file path from test file using TEST_FILE_EXT convention."""
        if not test_file_path.endswith(TEST_FILE_EXT):
            return None

        test_dir = os.path.dirname(test_file_path)
        test_filename = os.path.basename(test_file_path)
        main_filename = test_filename[:-len(TEST_FILE_EXT)] + '.py'
        main_file_path = os.path.join(test_dir, main_filename)

        if os.path.exists(main_file_path):
            self.logger.debug(f"Inferred main file {main_file_path} from test file {test_file_path}")
            return main_file_path

        self.logger.warning(f"Could not infer main file from test file: {test_file_path}")
        return None

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with given PID is still running and is a Python process."""
        try:
            process = psutil.Process(pid)
            current_user = (os.getlogin() if hasattr(os, 'getlogin')
                          else os.environ.get('USER', os.environ.get('USERNAME')))

            return (process.username() == current_user and
                   ('python' in process.name().lower() or
                    any('python' in arg.lower() for arg in process.cmdline()[:2])))
        # FIXED: Simplified exception handling
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, AttributeError):
            return False
        except Exception as e:
            self.logger.warning(f"Unexpected error checking process {pid}: {e}")
            return False

    # Active file management
    def _set_active_file_pointer(self, file_path: str) -> None:
        """Clears set to make sure only 1 active file is stored"""
        self.active_files.clear()
        self.active_files.add(os.path.abspath(file_path))

    def _remove_active_file_pointer(self, file_path: str) -> bool:
        """Only removes if file path matches stored one"""
        abs_path = os.path.abspath(file_path)
        if abs_path in self.active_files:
            self.active_files.discard(abs_path)
            return True
        return False

    def _has_active_files_pointer(self) -> bool:
        return len(self.active_files) > 0

    def _get_active_files_pointer(self) -> Set[str]:
        return self.active_files.copy()

    # Version management utilities
    def _find_existing_version_by_hash(self, index: Dict, content_hash: str) -> Optional[int]:
        return next((v.get('version') for v in index.get('versions', [])
                    if v.get('hash') == content_hash), None)

    def _get_next_version_number(self, index: Dict) -> int:
        versions = index.get('versions', [])
        return max((v.get('version', 0) for v in versions), default=0) + 1

    def _clean_file_path(self, file_path: str) -> str:
        """Remove meta directory extension from file path if present."""
        meta_ext_suffix = f".{META_DIR_EXT}"
        if file_path.endswith(meta_ext_suffix):
            cleaned = file_path[:-len(meta_ext_suffix)]
            self.logger.warning(f"Stripped meta extension from file_path: {file_path} -> {cleaned}")
            return cleaned
        return file_path

    def _create_version_file_path(self, file_path: str, version_number: int, versions_dir: str) -> str:
        """Create the path for a version file."""
        clean_path = self._clean_file_path(file_path)
        file_name = os.path.basename(clean_path)
        file_base, file_ext = os.path.splitext(file_name)
        version_file_name = f"{file_base}_v{version_number}{file_ext}"
        return os.path.join(versions_dir, version_file_name)

    def _create_version_metadata(self, file_path: str, content_hash: str, version_number: int,
                               changes: Optional[List[Dict[str, str]]], explanation: Optional[str],
                               main_file_path: Optional[str] = None,
                               execution_output: Optional[str] = None,
                               exit_code: Optional[int] = None,
                               execution_analysis: Optional[str] = None) -> Dict:
        """Create metadata entry for a version."""
        metadata = {
            'version': version_number,
            'file_path': file_path,
            'timestamp': self._get_timestamp(),
            'hash': content_hash,
            'changes': changes or [],
            'explanation': explanation or "No explanation provided"
        }
        if main_file_path:
            metadata['main_file_path'] = main_file_path

        # Add execution results if provided
        if execution_output is not None:
            metadata['execution_output'] = execution_output
        if exit_code is not None:
            metadata['exit_code'] = exit_code
        if execution_analysis is not None:
            metadata['execution_analysis'] = execution_analysis

        return metadata

    def _save_version_internal(self, file_path: str, content: str, versions_dir: str,
                             changes: Optional[List[Dict[str, str]]] = None,
                             explanation: Optional[str] = None,
                             main_file_path: Optional[str] = None,
                             execution_output: Optional[str] = None,
                             exit_code: Optional[int] = None,
                             execution_analysis: Optional[str] = None) -> int:
        """Internal method to save a versioned snapshot."""
        assert os.path.exists(versions_dir), f"File meta dir {versions_dir} does not exist"

        content_hash = hash_string(content)
        index_file_path = self._get_file_path(versions_dir, FileType.INDEX)

        # Now uses unified locking configuration from DatabaseManager
        with locked_json_operation(index_file_path, {'versions': []}, logger=self.logger) as index:
            if main_file_path and 'main_file_path' not in index:
                index['main_file_path'] = main_file_path

            # Check for existing version with same hash
            existing_version = self._find_existing_version_by_hash(index, content_hash)
            if existing_version:
                self.logger.debug(f"Content already exists in version {existing_version}. Skipping save.")
                return existing_version

            # Create new version
            version_number = self._get_next_version_number(index)
            version_file_path = self._create_version_file_path(file_path, version_number, versions_dir)

            # Save version file
            with open(version_file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Update index
            metadata = self._create_version_metadata(
                file_path, content_hash, version_number, changes, explanation, main_file_path,
                execution_output, exit_code, execution_analysis
            )
            index['versions'].append(metadata)
            index['versions'].sort(key=lambda x: x.get('version', 0))

            self.logger.debug(f"Successfully saved version {version_number} with {len(changes or [])} documented changes")
            return version_number

    def save_version(self, file_path: str, content: str,
                    changes: Optional[List[Dict[str, str]]] = None,
                    explanation: Optional[str] = None,
                    execution_output: Optional[str] = None,
                    exit_code: Optional[int] = None,
                    execution_analysis: Optional[str] = None) -> int:
        """Save a versioned snapshot of a main file."""
        self.logger.info(f"Saving version for file: {file_path}")
        versions_dir = self._get_meta_dir(file_path)
        return self._save_version_internal(file_path, content, versions_dir, changes, explanation,
                                         execution_output=execution_output, exit_code=exit_code,
                                         execution_analysis=execution_analysis)

    def save_test_version(self, test_file_path: str, content: str,
                         changes: Optional[List[Dict[str, str]]] = None,
                         explanation: Optional[str] = None,
                         main_file_path: Optional[str] = None,
                         execution_output: Optional[str] = None,
                         exit_code: Optional[int] = None,
                         execution_analysis: Optional[str] = None) -> int:
        """Save a versioned snapshot of a test file."""
        self.logger.info(f"Saving test version for file: {test_file_path}")

        if main_file_path is None:
            main_file_path = self._infer_main_file_from_test(test_file_path)
            if main_file_path is None:
                raise ValueError(f"Could not infer main file path for test file {test_file_path}")

        if not os.path.exists(main_file_path):
            raise ValueError(f"Main file does not exist: {main_file_path}")

        test_meta_dir = self._get_test_meta_dir(test_file_path, main_file_path)
        return self._save_version_internal(
            test_file_path, content, test_meta_dir, changes, explanation, main_file_path,
            execution_output, exit_code, execution_analysis
        )

    def _load_version_internal(self, file_path: str, versions_dir: str, include_content: bool,
                              reverse: bool = True, start_idx: int = None, end_idx: int = None,
                              return_single: bool = False):
        """
        Internal method to load versions from any versions directory with slice-based selection.

        Args:
            file_path: Path to the file
            versions_dir: Directory containing version metadata
            include_content: Whether to load file content
            reverse: Whether to sort versions in descending order (newest first)
            start_idx: Start index for slice (after sorting)
            end_idx: End index for slice (after sorting)
            return_single: If True and result has one item, return Dict instead of List

        Returns:
            Single Dict if return_single=True and one result, List otherwise, or None/[] on error
        """
        index_file_path = self._get_file_path(versions_dir, FileType.INDEX)

        if not os.path.exists(index_file_path):
            return None if return_single else []

        try:
            with locked_json_operation(index_file_path, {'versions': []}, logger=self.logger) as index:
                versions = index.get('versions', [])
                if not versions:
                    return None if return_single else []

                # Sort versions by version number
                versions = sorted(versions, key=lambda x: x.get('version', 0), reverse=reverse)

                # Apply slicing
                versions = versions[start_idx:end_idx]

                if include_content:
                    for version_data in versions:
                        if version_num := version_data.get('version'):
                            version_file_path = self._create_version_file_path(file_path, version_num, versions_dir)
                            try:
                                with open(version_file_path, 'r', encoding='utf-8') as f:
                                    version_data['content'] = f.read()
                            except (FileNotFoundError, IOError) as e:
                                self.logger.warning(f"Could not load content for version {version_num}: {e}")
                                version_data['content'] = None

                # Return single item if requested and exactly one result
                if return_single and len(versions) == 1:
                    return versions[0]
                elif return_single and len(versions) == 0:
                    return None
                else:
                    return versions

        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.error(f"Error loading versions from {index_file_path}: {e}")
            return None if return_single else []

    def load_version(self, file_path: str, k: int = 1, include_content: bool = False):
        """Load the last K versions for a file. Returns Dict if k=1, List if k>1."""
        return self._load_version_internal(
            file_path, self._get_meta_dir(file_path),
            include_content=include_content, reverse=True,
            start_idx=None, end_idx=k,
            return_single=(k == 1)
        )

    def load_test_version(self, test_file_path: str, k: int = 1, include_content: bool = False,
                         main_file_path: Optional[str] = None):
        """Load the last K versions for a test file. Returns Dict if k=1, List if k>1."""
        if main_file_path is None:
            main_file_path = self._infer_main_file_from_test(test_file_path)
            if main_file_path is None:
                self.logger.error(f"Could not infer main file path for test file {test_file_path}")
                return None if k == 1 else []

        if not os.path.exists(main_file_path):
            self.logger.error(f"Main file does not exist: {main_file_path}")
            return None if k == 1 else []

        return self._load_version_internal(
            test_file_path, self._get_test_meta_dir(test_file_path, main_file_path),
            include_content=include_content, reverse=True,
            start_idx=None, end_idx=k,
            return_single=(k == 1)
        )

    def load_original_version(self, file_path: str, include_content: bool = False) -> Optional[Dict]:
        """Load the original (first) version for a file. Returns Dict or None."""
        return self._load_version_internal(
            file_path, self._get_meta_dir(file_path),
            include_content=include_content, reverse=False,
            start_idx=None, end_idx=1,
            return_single=True
        )

    def load_test_original_version(self, test_file_path: str, include_content: bool = False,
                                  main_file_path: Optional[str] = None) -> Optional[Dict]:
        """Load the original (first) version for a test file. Returns Dict or None."""
        if main_file_path is None:
            main_file_path = self._infer_main_file_from_test(test_file_path)
            if main_file_path is None:
                self.logger.error(f"Could not infer main file path for test file {test_file_path}")
                return None

        if not os.path.exists(main_file_path):
            self.logger.error(f"Main file does not exist: {main_file_path}")
            return None

        return self._load_version_internal(
            test_file_path, self._get_test_meta_dir(test_file_path, main_file_path),
            include_content=include_content, reverse=False,
            start_idx=None, end_idx=1,
            return_single=True
        )

    def save_original(self, file_path: str, content: str) -> int:
        """Save the original unedited file."""
        self.logger.info(f"Saving original version for file: {file_path}")
        return self.save_version(
            file_path, content,
            changes=[{"type": "initial", "description": "Original file content"}],
            explanation="Initial version: Original unedited file"
        )

    # File status management
    def _cleanup_stale_active_file(self, active_file_path: str) -> None:
        """Helper method to clean up stale active files."""
        try:
            os.remove(active_file_path)
        except (FileNotFoundError, PermissionError):
            pass

    def check_avaliable(self, file_path: str) -> bool:
        """Returns false if file is being worked on by an agent (active.json),
        will return true if being worked on by self"""
        if not self.lock_active_file: return True

        meta_dir = self._get_meta_dir(file_path)
        active_file_path = self._get_file_path(meta_dir, FileType.ACTIVE)

        if not os.path.exists(active_file_path):
            return True

        try:
            # Now uses unified locking configuration
            with locked_file_operation(active_file_path, 'r', logger=self.logger) as f:
                active_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            self._cleanup_stale_active_file(active_file_path)
            return True

        stored_pid = active_data.get('pid')
        if not stored_pid or not self._is_process_running(stored_pid):
            self._cleanup_stale_active_file(active_file_path)
            return True

        if stored_pid == os.getpid():
            return True

        return False

    def try_reserve_file(self, agent, file_path: str, max_attempts: int = None) -> bool:
        """Try to reserve a file with exponential backoff using global locking config."""
        if not self.lock_active_file:
            return True

        locking_config = get_locking_config()
        if max_attempts is None:
            max_attempts = locking_config.max_retries

        for attempt in range(max_attempts):
            try:
                return self.flag_active(agent, file_path)
            except RuntimeError as e:
                if "already being worked on" not in str(e):
                    raise  # Re-raise unexpected errors
                if attempt == max_attempts - 1:
                    return False  # Final attempt failed

                # Exponential backoff with jitter
                base_delay = locking_config.retry_delay
                backoff_time = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
                time.sleep(backoff_time)

        return False

    def flag_active(self, agent, file_path: str) -> bool:
        """Flag a file as being actively worked on."""
        meta_dir = self._get_meta_dir(file_path)
        active_file_path = self._get_file_path(meta_dir, FileType.ACTIVE)
        current_pid = os.getpid()

        with locked_json_operation(self._get_file_path(meta_dir, FileType.ACTIVE), {}, logger=self.logger) as active_data:
            # Check existing active state INSIDE the lock
            existing_pid = active_data.get('pid')
            if existing_pid and self._is_process_running(existing_pid):
                if self.lock_active_file and existing_pid != current_pid:
                    raise RuntimeError(f"File {file_path} is already being worked on by PID {existing_pid}")

            # Handle old active file pointer inside lock
            if self._has_active_files_pointer():
                old_filepath = list(self._get_active_files_pointer())[0]
                if old_filepath != file_path:
                    self.logger.error(f"Agent {agent.get_id()} already has active file: {old_filepath}, setting to new active file: {file_path}")
                    self.unflag_active(agent, old_filepath)

            # Set the active flag atomically
            active_data.clear()
            active_data.update({
                'agent_id': agent.get_id(),
                'pid': current_pid,
                'timestamp': self._get_timestamp(),
                'file_path': file_path
            })
            self._set_active_file_pointer(file_path)

        self.logger.debug(f"Flagged active {file_path} for agent {agent.get_id()} (PID {current_pid})")
        return True

    def unflag_active(self, agent, file_path: str) -> bool:
        """Remove the active flag for a file."""
        meta_dir = self._get_meta_dir(file_path)
        active_file_path = self._get_file_path(meta_dir, FileType.ACTIVE)
        current_pid = os.getpid()

        if not os.path.exists(active_file_path):
            self._remove_active_file_pointer(file_path)
            return False

        try:
            # Now uses unified locking configuration
            with locked_file_operation(active_file_path, 'r', logger=self.logger) as f:
                active_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            self._remove_active_file_pointer(file_path)
            return False

        # Check PID for ownership - this is sufficient since PID uniquely identifies the process
        stored_pid = active_data.get('pid')
        if stored_pid != current_pid:
            return False

        # Use helper method for consistent file cleanup
        self._cleanup_stale_active_file(active_file_path)
        self._remove_active_file_pointer(file_path)
        self.logger.debug(f"Unflagged active {file_path} for agent {agent.get_id()} (PID {current_pid})")
        return True

    def check_finished(self, agent, file_path: str, easy_mode: bool = False) -> bool:
        """Check if a file has been marked as finished by this agent (by name)."""
        meta_dir = self._get_meta_dir(file_path)
        finished_file_path = self._get_file_path(meta_dir, FileType.FINISHED)

        try:
            with open(finished_file_path, 'r', encoding='utf-8') as f:
                finished_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return False

        # Use agent name for comparison instead of full agent_id
        current_agent_name = agent.name

        agents = finished_data.get('agents', [])
        if easy_mode:
            self.logger.debug("File finished check is set to easy mode!")
            return len(agents) > 0
        else:
            return any(
                agent_entry.get('agent_name') == current_agent_name
                for agent_entry in agents
            )

    def flag_finished(self, agent, file_path: str) -> None:
        """Flag a file as finished, storing agent name for cross-session recognition."""
        meta_dir = self._get_meta_dir(file_path)

        # Now uses unified locking configuration
        with locked_json_operation(self._get_file_path(meta_dir, FileType.FINISHED),
            {'agents': []}, logger=self.logger) as finished_data:

            # Check for duplicates by agent name
            current_agent_name = agent.name
            if any(agent_entry.get('agent_name') == current_agent_name for agent_entry in finished_data['agents']):
                return

            finished_data['agents'].append({
                'agent_id': agent.get_id(),        # Keep full ID for debugging/history
                'agent_name': agent.name,          # Store name for comparison
                'timestamp': self._get_timestamp(),
                'file_path': file_path
            })

        self.logger.debug(f"Flagged {file_path} as finished by agent {agent.name} (ID: {agent.get_id()})")

    # Validation methods (simplified)
    def _validate_required_fields(self, data: Dict, required_fields: List[str], context: str) -> List[ValidationError]:
        """Generic field validation."""
        errors = []
        for field in required_fields:
            if field not in data:
                errors.append(ValidationError(context, field, f"Missing required field '{field}'"))
        return errors

    def _validate_timestamp(self, timestamp: str, context: str) -> Optional[ValidationError]:
        """Validate timestamp format."""
        try:
            datetime.datetime.fromisoformat(timestamp)
            return None
        except ValueError as e:
            return ValidationError(context, 'timestamp', f"Invalid timestamp: {e}", timestamp)

    def _validate_file_path_match(self, actual: str, expected: str, context: str) -> Optional[ValidationError]:
        """Validate file path matches expected."""
        if os.path.abspath(actual) != os.path.abspath(expected):
            return ValidationError(context, 'file_path', f"Path mismatch. Expected: {expected}, Found: {actual}")
        return None

    def validate_active_files(self, agent) -> None:
        """Validate agent's active files and meta directory structure."""
        if not self.lock_active_file: return

        agent_id = agent.get_id()
        active_files = self._get_active_files_pointer()

        if len(active_files) > 1:
            raise RuntimeError(f"Agent {agent_id} has {len(active_files)} active files, only 1 allowed: {list(active_files)}")

        if not active_files:
            self.logger.debug(f"Agent {agent_id} has no active files - validation passed")
            return

        active_file_path = list(active_files)[0]
        self.logger.debug(f"Validating active file: {active_file_path}")

        if not os.path.exists(active_file_path):
            raise FileNotFoundError(f"Active file does not exist: {active_file_path}")

        meta_dir = self._get_meta_dir(active_file_path)
        active_json_path = self._get_file_path(meta_dir, FileType.ACTIVE)

        if not os.path.exists(active_json_path):
            raise FileNotFoundError(f"Active JSON file missing: {active_json_path}")

        # Validate active.json content
        try:
            with open(active_json_path, 'r', encoding='utf-8') as f:
                active_data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Active JSON file is corrupted: {active_json_path}")

        # Basic validation
        errors = self._validate_required_fields(active_data, ['agent_id', 'timestamp', 'file_path'], active_json_path)

        if active_data.get('agent_id') != agent_id:
            errors.append(ValidationError(active_json_path, 'agent_id', f"Agent ID mismatch. Expected: {agent_id}"))

        timestamp_error = self._validate_timestamp(active_data.get('timestamp', ''), active_json_path)
        if timestamp_error:
            errors.append(timestamp_error)

        path_error = self._validate_file_path_match(active_data.get('file_path', ''), active_file_path, active_json_path)
        if path_error:
            errors.append(path_error)

        if errors:
            error_messages = [f"{err.field}: {err.message}" for err in errors]
            raise ValueError(f"Validation failed for {active_json_path}: {'; '.join(error_messages)}")

        self.logger.debug(f"Validation passed for active file: {active_file_path}")

    # ========== DATABASE INTERFACE METHODS ==========
    # Only methods actually used by action classes are exposed

    def store_data(self,
        file_path: str,
        table_name: str,
        data: Dict[str, Any],
        single_entry: bool = True) -> int:
        """Store data in the TinyDB database for a file, clearing existing data first."""
        # Clear the table before storing new data
        if single_entry:
            self.db.clear_table(file_path, table_name)
        # Store the new data
        return self.db.store_data(file_path, table_name, data)

    def get_data(self,
        file_path: str,
        table_name: str,
        query_filter: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Get the most recent record from a table."""
        return self.db.get_latest_data(file_path, table_name, query_filter)

    # ========== END DATABASE INTERFACE METHODS ==========

    def shutdown(self, agent):
        """Enhanced cleanup with database closure."""
        # Clear active files
        for file_path in self.active_files.copy():
            try:
                self.unflag_active(agent, file_path)
            except Exception as e:
                self.logger.debug(f"Failed to cleanup active file {file_path}: {e}")

        self.active_files.clear()
        self.db.shutdown()
