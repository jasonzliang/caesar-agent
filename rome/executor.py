#!/usr/bin/env python3
# Standalone Command Line Code Executor
# Simplified version with core functionality only

import os
import re
import subprocess
import sys
import platform
import shutil
from hashlib import md5
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Union

from .config import set_attributes_from_config
from .logger import get_logger

# Constants
WIN32 = platform.system() == "Windows"
PYTHON_VARIANTS = ["python", "py", "python3"]
PYTEST_VARIANTS = ["pytest", "py.test"]
TIMEOUT_MSG = "\nExecution timed out."


class CodeBlock:
    """Represents a block of code with its language or a path to a code file."""
    def __init__(self, code: Union[str, Path], language: str, is_filepath: bool = False):
        self.code = str(code) if isinstance(code, Path) else code
        self.language = language
        self.is_filepath = is_filepath


class CommandLineCodeResult:
    """Result of a command line code execution."""
    def __init__(self, exit_code: int, output: str, code_file: Optional[str] = None):
        self.exit_code = exit_code
        self.output = output
        self.code_file = code_file


def _cmd(lang: str) -> str:
    """Get the command to execute for a language."""
    if lang == "python":
        return "python" if WIN32 else "python3"
    if lang == "pytest":
        return "pytest"
    if lang in ["bash", "shell", "sh"]:
        return "bash" if not WIN32 else "powershell"
    if lang in ["pwsh", "powershell", "ps1"]:
        return "powershell" if WIN32 else "pwsh"
    # Default case for unsupported languages
    return lang


def _get_file_name_from_content(code: str, work_dir: Path) -> Optional[str]:
    """Extract filename from code comments."""
    file_pattern = r"#\s*filename:\s*([^\n]+)"
    match = re.search(file_pattern, code)
    if match:
        filename = match.group(1).strip()
        # Ensure filename is safe and within workspace
        if os.path.isabs(filename):
            normalized_path = os.path.normpath(filename)
            work_dir_path = os.path.normpath(str(work_dir))
            if not normalized_path.startswith(work_dir_path):
                raise ValueError(f"Filename cannot be outside of workspace: {filename}")

        # Additional safety check for path traversal
        safe_path = Path(work_dir) / filename
        if not safe_path.resolve().is_relative_to(work_dir.resolve()):
            raise ValueError(f"Path traversal detected in filename: {filename}")

        return filename
    return None


def silence_pip(code: str, lang: str) -> str:
    """Modify pip commands to be quiet."""
    if lang != "python":
        return code

    # Replace pip install with quiet flag
    pattern = r"(pip(?:3)?\s+install)"
    replacement = r"\1 -q"
    return re.sub(pattern, replacement, code)


def create_virtual_env(venv_path: Union[str, Path], with_pip: bool = True, install_pytest: bool = True) -> SimpleNamespace:
    """Create a virtual environment at the specified path."""
    import subprocess

    venv_path = Path(venv_path) if isinstance(venv_path, str) else venv_path

    if not venv_path.exists():
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_path), "--with-pip" if with_pip else ""],
                           check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error creating virtual environment: {e.stderr}")
            raise

    bin_path = venv_path / ("Scripts" if WIN32 else "bin")
    python_exe = bin_path / ("python.exe" if WIN32 else "python")

    # Check if the python executable exists
    if not python_exe.exists():
        raise FileNotFoundError(f"Python executable not found at {python_exe}")

    # Install pytest in the virtual environment if requested
    if install_pytest and with_pip:
        try:
            subprocess.run(
                [str(python_exe), "-m", "pip", "install", "pytest"],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install pytest in virtual environment: {e.stderr}")

    return SimpleNamespace(
        bin_path=bin_path,
        env_exe=python_exe,
        env_dir=venv_path,
    )


class CodeExecutor:
    """A code executor that executes or saves code in a local command line environment."""

    SUPPORTED_LANGUAGES = [
        "bash", "shell", "sh", "pwsh", "powershell", "ps1",
        "python", "javascript", "html", "css", "pytest"
    ]

    # Fixed dictionary definition
    DEFAULT_EXEC_POLICIES = {
        "bash": True, "shell": True, "sh": True,
        "pwsh": True, "powershell": True, "ps1": True,
        "python": True, "javascript": False, "html": False, "css": False,
        "pytest": True
    }

    def __init__(self, config: Dict = None):
        """Initialize the command line code executor.

        Args:
            config: Configuration dictionary with the following keys:
                - timeout: The timeout for code execution (seconds)
                - virtual_env_context: The virtual environment context
                - execution_policies: Language to execution policies mapping
                - work_dir: The working directory for code execution
        """
        # Initialize logger
        self.logger = get_logger()

        # Set default configuration
        self.config = config or {}

        # Update with provided config if any
        set_attributes_from_config(self, self.config,
            ['timeout', 'work_dir'],
            ['cmd_args', 'max_output_len', 'virtual_env_context', 'execution_policies'])

        # Create a custom execution_policies dict merging defaults with any provided in config
        execution_policies = self.DEFAULT_EXEC_POLICIES.copy()
        if self.execution_policies:
            execution_policies.update(self.execution_policies)
        self.execution_policies = execution_policies

        # Validate timeout
        if self.timeout < 1:
            raise ValueError("Timeout must be greater than or equal to 1.")

        # Ensure work_dir is a Path object
        try:
            self.work_dir = Path(self.work_dir)
            self.logger.info(f"Executor work dir: {self.work_dir}")
        except:
            self.work_dir = Path(os.getcwd())
            self.logger.error(f"Invalid executor work dir, using fallback: {os.getcwd()}")

        # Ensure work_dir exists
        self.work_dir.mkdir(exist_ok=True, parents=True)

        # self.logger.debug(f"Initialized {self.__class__.__name__} with config: {self.config}")

    def _truncate_output(self, output: str) -> str:
        """Truncate output if it exceeds max_output_len."""
        if not output or not self.max_output_len or len(output) <= self.max_output_len:
            return output

        truncated = output[:self.max_output_len * 4]
        truncation_msg = f"\n\n[OUTPUT TRUNCATED - showing first {self.max_output_len} characters of {len(output)} total]"

        # Try to find a good break point (newline) near the end
        last_newline = truncated.rfind('\n', max(0, self.max_output_len - 200))
        if last_newline > self.max_output_len * 0.8:  # If newline is in last 20%
            truncated = truncated[:last_newline]

        return truncated + truncation_msg

    @staticmethod
    def sanitize_command(lang: str, code: str) -> None:
        """Sanitize the code block to prevent dangerous commands."""
        dangerous_patterns = [
            (r"\brm\s+-rf\b", "Use of 'rm -rf' command is not allowed."),
            (r"\bmv\b.*?\s+/dev/null", "Moving files to /dev/null is not allowed."),
            (r"\bdd\b", "Use of 'dd' command is not allowed."),
            (r">\s*/dev/sd[a-z][1-9]?", "Overwriting disk blocks directly is not allowed."),
            (r":\(\)\{\s*:\|\:&\s*\};:", "Fork bombs are not allowed."),
        ]

        # Base patterns for all languages
        common_patterns = [
            (r"(?:import|require).*?(?:sys|os).*?(?:system|exec|spawn|popen)",
             "Executing shell commands from code is not allowed."),
            (r"(?:eval|exec)\s*\(", "Dynamic code execution is not allowed.")
        ]

        # Language-specific patterns
        if lang in ["bash", "shell", "sh"]:
            for pattern, message in dangerous_patterns:
                if re.search(pattern, code):
                    raise ValueError(f"Potentially dangerous command detected: {message}")

        # Python-specific patterns
        if lang == "python":
            python_patterns = [
                (r"(?:import|from).*?(?:subprocess|os|pty|commands)",
                 "Importing modules that can execute shell commands is not allowed."),
                (r"__import__\s*\(\s*['\"](?:subprocess|os|pty|commands)",
                 "Dynamic importing of dangerous modules is not allowed.")
            ]
            for pattern, message in python_patterns:
                if re.search(pattern, code):
                    raise ValueError(f"Potentially dangerous code detected: {message}")

        # Check common patterns for all languages
        for pattern, message in common_patterns:
            if re.search(pattern, code):
                raise ValueError(f"Potentially dangerous code detected: {message}")

    def execute_code_blocks(self,
        code_blocks: List[CodeBlock],
        work_dir: Optional[Union[Path, str]] = None) -> CommandLineCodeResult:
        """Execute the code blocks and return the result.

        Args:
            code_blocks: List of code blocks to execute (can contain code or filepaths)
            work_dir: The working directory for code execution

        Returns:
            Result of the execution
        """
        if not work_dir:
            work_dir = self.work_dir
        if isinstance(work_dir, str):
            work_dir = Path(work_dir)

        # Create work directory if it doesn't exist
        try:
            work_dir.mkdir(exist_ok=True, parents=True)
        except PermissionError:
            error_msg = f"Permission denied: Cannot create directory {work_dir}"
            self.logger.error(error_msg)
            return CommandLineCodeResult(exit_code=1, output=error_msg)
        except Exception as e:
            error_msg = f"Failed to create work directory {work_dir}: {str(e)}"
            self.logger.error(error_msg)
            return CommandLineCodeResult(exit_code=1, output=error_msg)

        logs_all = ""
        file_names = []
        exitcode = 0

        self.logger.debug(f"Executing {len(code_blocks)} code blocks in {work_dir}")

        for code_block in code_blocks:
            lang, code, is_filepath = code_block.language, code_block.code, code_block.is_filepath
            lang = lang.lower()

            # Handle empty code blocks or normalize language
            if not is_filepath and not code.strip():
                self.logger.info("Empty code block detected, skipping")
                continue

            if lang in PYTHON_VARIANTS:
                lang = "python"
            elif lang in PYTEST_VARIANTS:
                lang = "pytest"
            elif WIN32 and lang in ["sh", "shell"]:
                lang = "ps1"

            if lang not in self.SUPPORTED_LANGUAGES:
                exitcode = 1
                logs_all += f"Unknown language: {lang}"
                self.logger.error(f"Unknown language: {lang}")
                break

            execute_code = self.execution_policies.get(lang, False)

            # Process the code/filepath to get the file to execute
            written_file = None
            if is_filepath:
                # Handle filepath case - use the existing file
                file_path = Path(code)
                if not file_path.exists():
                    error_msg = f"File not found: {file_path}"
                    self.logger.error(error_msg)
                    return CommandLineCodeResult(exit_code=1, output=error_msg)

                written_file = file_path.resolve()
                file_names.append(written_file)
            else:
                # Handle code case - create a new file
                try:
                    self.sanitize_command(lang, code)
                    code = silence_pip(code, lang)
                except ValueError as e:
                    self.logger.error(f"Command sanitization failed: {str(e)}")
                    return CommandLineCodeResult(exit_code=1, output=str(e))

                # Get filename from content or generate one
                try:
                    filename = _get_file_name_from_content(code, work_dir)
                except ValueError as e:
                    self.logger.error(f"Invalid filename: {str(e)}")
                    return CommandLineCodeResult(exit_code=1, output=f"Invalid filename: {str(e)}")

                if filename is None:
                    code_hash = md5(code.encode()).hexdigest()
                    ext = 'py' if lang == 'python' else ('py' if lang == 'pytest' else lang)
                    filename = f"tmp_code_{code_hash}.{ext}"

                written_file = (work_dir / filename).resolve()

                # Safety check for path traversal
                if not written_file.is_relative_to(work_dir.resolve()):
                    error_msg = f"File path is outside of working directory: {written_file}"
                    self.logger.error(error_msg)
                    return CommandLineCodeResult(exit_code=1, output=error_msg)

                # Write code to file
                try:
                    with written_file.open("w", encoding="utf-8") as f:
                        f.write(code)
                    file_names.append(written_file)
                    self.logger.debug(f"Saved code to {written_file}")
                except (PermissionError, IOError) as e:
                    error_msg = f"Failed to write file {written_file}: {str(e)}"
                    self.logger.error(error_msg)
                    return CommandLineCodeResult(exit_code=1, output=error_msg)

            # If execution is disabled, just log that the file was saved
            if not execute_code:
                logs_all += f"\nCannot run due to execution policy: {written_file!s}"
                self.logger.error(f"Cannot run due to execution policy: {written_file!s}")
                continue

            # Prepare execution command
            program = _cmd(lang)
            if not shutil.which(program):
                error_msg = f"Program '{program}' not found in PATH"
                self.logger.error(error_msg)
                return CommandLineCodeResult(exit_code=1, output=error_msg)

            # Add additional argument if running pytest
            cmd = [program, str(written_file.absolute())]
            if self.cmd_args and len(self.cmd_args.get(program, [])) > 0:
                cmd_args = self.cmd_args.get(program)
                if type(cmd_args) is str:
                    cmd_args = cmd_args.split()
                cmd = [cmd[0]] + cmd_args + [cmd[1]]

            env = os.environ.copy()
            # Configure virtual environment if available
            if self.virtual_env_context:
                virtual_env_abs_path = os.path.abspath(str(self.virtual_env_context.bin_path))
                path_with_virtualenv = f"{virtual_env_abs_path}{os.pathsep}{env.get('PATH', '')}"
                env["PATH"] = path_with_virtualenv

                # Windows-specific handling for virtual environments
                if WIN32:
                    temp_batch = work_dir / f"run_in_venv_{md5(str(written_file).encode()).hexdigest()}.bat"
                    activation_script = os.path.join(virtual_env_abs_path, "activate.bat")

                    with open(temp_batch, 'w') as f:
                        f.write(f'@echo off\n')
                        f.write(f'call "{activation_script}"\n')
                        f.write(f'{" ".join(cmd)}\n')

                    cmd = [str(temp_batch)]

            # Execute the command
            self.logger.debug(f"Executing command: {cmd}")
            try:
                result = subprocess.run(
                    cmd,
                    cwd=work_dir,
                    stdout=subprocess.PIPE,     # Instead of capture_output=True
                    stderr=subprocess.STDOUT,   # Redirect stderr to stdout
                    text=True,
                    timeout=float(self.timeout),
                    env=env,
                    encoding="utf-8",
                    shell=WIN32,  # Use shell on Windows for better compatibility
                )
                logs_all += "\n" + self._truncate_output(result.stdout)
                exitcode = result.returncode

                if exitcode != 0:
                    self.logger.error(f"Command exited with non-zero code: {exitcode}")
                    break

            except subprocess.TimeoutExpired:
                logs_all += "\n" + TIMEOUT_MSG
                exitcode = 124  # Same exit code as the timeout command on linux
                self.logger.error(f"Command execution timed out after {self.timeout} seconds")
                break
            except FileNotFoundError:
                logs_all += f"\nError: Command '{cmd[0]}' not found."
                exitcode = 127  # Standard exit code for command not found
                self.logger.error(f"Command not found: {cmd[0]}")
                break
            except Exception as e:
                logs_all += f"\nError executing command: {str(e)}"
                exitcode = 1
                self.logger.error(f"Error executing command: {str(e)}")
                break

            # Clean up temporary batch file if created
            if WIN32 and self.virtual_env_context and 'temp_batch' in locals():
                try:
                    temp_batch.unlink()
                except:
                    pass

        code_file = str(file_names[0]) if file_names else None
        return CommandLineCodeResult(exit_code=exitcode, output=logs_all, code_file=code_file)

    @staticmethod
    def _detect_file_language(file_path: Union[str, Path]) -> str:
        """Detect the language of a file based on its extension and content."""
        file_path = Path(file_path) if isinstance(file_path, str) else file_path

        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file extension
        extension = file_path.suffix.lower().lstrip('.')

        # Extension-based detection
        extension_mapping = {
            'py': 'python',
            'js': 'javascript',
            'html': 'html',
            'css': 'css',
            'sh': 'bash',
            'bash': 'bash',
            'ps1': 'powershell',
        }

        if extension in extension_mapping:
            language = extension_mapping[extension]

            # Additional check for pytest files
            if language == 'python':
                # Check if it's a pytest file based on naming convention
                file_name = file_path.name.lower()
                if file_name.startswith('test_') or file_name.endswith('_test.py'):
                    # Read first few lines to confirm it's a pytest file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read(4096)  # Read first 4KB
                            if 'import pytest' in content or '@pytest' in content or 'pytest.fixture' in content:
                                return 'pytest'
                    except UnicodeDecodeError:
                        pass

            return language

        # Content-based detection for files without recognizable extensions
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(4096)  # Read first 4KB

                if '#!/usr/bin/env python' in content or 'import ' in content:
                    # Check if it's a pytest file
                    if 'import pytest' in content or '@pytest' in content or 'pytest.fixture' in content:
                        return 'pytest'
                    return 'python'
                elif '#!/bin/bash' in content or '#!/usr/bin/env bash' in content:
                    return 'bash'
                elif '<html' in content.lower():
                    return 'html'
        except UnicodeDecodeError:
            pass

        # Default fallback
        return 'python'

    def execute_file(self,
        file_path: Union[str, Path],
        work_dir: Optional[str] = None,
        language: Optional[str] = None) -> CommandLineCodeResult:
        """Execute a file and return the result."""
        file_path = Path(file_path) if isinstance(file_path, str) else file_path

        # Check if file exists
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return CommandLineCodeResult(exit_code=1, output=f"File not found: {file_path}")

        # Use the file's directory as the working directory if not provided
        if not work_dir: work_dir = file_path.parent

        # Detect language if not provided
        if language is None:
            try:
                language = self._detect_file_language(file_path)
                self.logger.debug(f"Detected language for {file_path}: {language}")
            except Exception as e:
                self.logger.error(f"Error detecting language: {str(e)}")
                return CommandLineCodeResult(exit_code=1, output=f"Error detecting language: {str(e)}")

        # Create a code block with the file path and execute it
        code_block = CodeBlock(code=file_path, language=language, is_filepath=True)
        return self.execute_code_blocks([code_block], work_dir)


if __name__ == "__main__":
    pass
