import logging
import sys
import threading
import tempfile
import os
import subprocess
import pytz
from datetime import datetime
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape

# Per-thread Logger instances. CLI is single-threaded; the web server runs
# each Caesar agent in its own asyncio.to_thread worker, and per-thread
# isolation prevents concurrent agents from clobbering each other's handlers
# when each calls Logger.configure() with its own file path.
#
# threading.local() (rather than a TID-keyed dict) is intentional: it stores
# the instance on the thread itself, so it's garbage-collected when the
# thread exits. A dict keyed on threading.get_ident() would (a) grow
# unboundedly in a long-lived server and (b) hand stale Logger instances
# back on TID reuse, briefly logging to the previous run's console.log
# until the next agent's BaseAgent.__init__ calls configure().
_thread_loggers = threading.local()


class SizeRotatingFileHandler(logging.FileHandler):
    """Deadlock-safe size-rotating file handler with cross-platform support"""

    def __init__(self, filename, max_size_kb, keep_ratio=0.5, *args, **kwargs):
        self.max_size_kb = max_size_kb
        self.max_size_bytes = max_size_kb * 1024
        self.keep_ratio = max(0.1, min(0.9, keep_ratio))
        self._lock = threading.RLock()
        self._rotating = False
        self._rotation_count = 0
        self._last_rotation_size = 0
        super().__init__(filename, *args, **kwargs)

    def emit(self, record):
        """Emit record with deadlock-safe rotation"""
        with self._lock:
            try:
                if self._should_rotate():
                    self._rotate()

                # Check if stream is valid before emitting
                if self.stream is None:
                    # Last resort: write to stderr
                    sys.stderr.write(f"{record.levelname}: {record.getMessage()}\n")
                    return

                super().emit(record)
            except Exception as e:
                self.handleError(record)

    def _should_rotate(self):
        """Check if rotation is needed"""
        if self._rotating or not self.max_size_bytes:
            return False

        try:
            if not os.path.exists(self.baseFilename):
                return False

            current_size = os.path.getsize(self.baseFilename)
            return current_size >= self.max_size_bytes
        except (OSError, IOError):
            return False

    def _rotate(self):
        """Rotate log using Unix tools"""
        try:
            self._rotating = True

            # Close stream before any file operations
            if self.stream:
                try:
                    self.stream.flush()
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None

            # Try Unix-based rotation, emergency truncate on failure
            if os.name == 'posix':
                success = self._unix_rotate()
            else:
                sys.stderr.write("Log rotation only supported on Unix systems\n")
                success = False

            if success:
                self._rotation_count += 1
                try:
                    self._last_rotation_size = os.path.getsize(self.baseFilename)
                except Exception:
                    self._last_rotation_size = 0
            else:
                self._emergency_truncate()

        except Exception as e:
            sys.stderr.write(f"Log rotation failed: {e}\n")
            self._emergency_truncate()
        finally:
            # Always reopen stream
            if not self.stream:
                try:
                    self.stream = self._open()
                except Exception as e:
                    sys.stderr.write(f"Failed to reopen log stream: {e}\n")
            self._rotating = False

    def _unix_rotate(self) -> bool:
        """Fast Unix-based rotation with timeout protection"""
        tmp = None
        try:
            lines = max(int(self.max_size_bytes * self.keep_ratio) // 80, 50)

            # Create temp file in same directory for atomic rename
            dir_name = os.path.dirname(self.baseFilename) or '.'
            fd, tmp = tempfile.mkstemp(dir=dir_name, suffix='.log')

            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(f"[ROTATED - {self.max_size_kb}KB limit, kept {int(self.keep_ratio*100)}%]\n")

                    # Use tail to get last N lines
                    result = subprocess.run(
                        ['tail', '-n', str(lines), self.baseFilename],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=5
                    )
                    f.write(result.stdout)
            except Exception:
                # fd was already closed by os.fdopen, just cleanup tmp file
                if tmp and os.path.exists(tmp):
                    os.unlink(tmp)
                return False

            # Atomic rename
            os.rename(tmp, self.baseFilename)
            tmp = None  # Prevent cleanup since rename succeeded
            return True

        except Exception:
            return False
        finally:
            # Clean up temp file if it still exists
            if tmp and os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except Exception:
                    pass

    def _emergency_truncate(self):
        """Emergency truncate if rotation fails"""
        try:
            with open(self.baseFilename, 'w', encoding='utf-8') as f:
                f.write(f"[EMERGENCY TRUNCATE - {self.max_size_kb}KB limit]\n")
                f.write(f"[Previous rotation attempts: {self._rotation_count}]\n")
        except Exception as e:
            sys.stderr.write(f"Emergency truncate failed: {e}\n")
            sys.stderr.write("File logging disabled - falling back to stderr only\n")
            # Don't try to use this handler anymore
            self.stream = None

    def close(self):
        """Thread-safe handler close"""
        with self._lock:
            try:
                if self.stream:
                    self.stream.flush()
                    self.stream.close()
            except Exception:
                pass
            finally:
                self.stream = None
                super().close()

    def get_rotation_stats(self):
        """Get rotation statistics for monitoring"""
        return {
            'rotation_count': self._rotation_count,
            'last_rotation_size': self._last_rotation_size,
            'current_size': os.path.getsize(self.baseFilename) if os.path.exists(self.baseFilename) else 0,
            'max_size_bytes': self.max_size_bytes
        }


class ParentPathRichHandler(RichHandler):
    """Custom RichHandler that shows the path of the parent function (1 level above caller)."""

    def __init__(self, *args, **kwargs):
        kwargs['show_path'] = False
        super().__init__(*args, **kwargs)
        self._cache = {}
        self._cache_size = 1000

    def emit(self, record):
        """Add parent path info before emitting."""
        import inspect

        caller_key = (record.pathname, record.lineno)

        if caller_key not in self._cache:
            if len(self._cache) >= self._cache_size:
                self._cache.clear()

            try:
                stack = inspect.stack()
                user_frames = [
                    {'filename': os.path.basename(f.filename), 'lineno': f.lineno}
                    for f in stack
                    if '__init__.py' not in f.filename and
                       'logging' not in f.filename and
                       f.function not in ['emit', 'handle', 'callHandlers', '_log']
                ]
                self._cache[caller_key] = user_frames[1] if len(user_frames) >= 2 else None
            except Exception:
                self._cache[caller_key] = None

        parent_info = self._cache[caller_key]

        if parent_info:
            parent_path = f"[{parent_info['filename']}:{parent_info['lineno']}]"
            record.msg = f"{parent_path} {escape(str(record.msg))}"

        super().emit(record)


class Logger:
    """Thread-safe logger with Rich console output and caller information"""

    def __init__(self, logger_name='Agent'):
        """Initialize logger with specified name"""
        self.logger_name = logger_name
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._configured = False
        self._lock = threading.Lock()

        # Setup basic console handler immediately for early logging
        self._setup_basic_console()

    def _setup_basic_console(self):
        """Setup basic console handler for immediate use"""
        if not self._logger.handlers:
            console = Console()
            basic_handler = RichHandler(
                console=console,
                show_time=True,
                show_path=False,
                show_level=True,
                rich_tracebacks=True,
                markup=True,
                log_time_format="[%H:%M:%S]"
            )
            basic_handler.setLevel(logging.INFO)
            self._logger.addHandler(basic_handler)

    def disable(self):
        """Temporarily disable all logging output"""
        self._logger.disabled = True

    def enable(self):
        """Re-enable logging output"""
        self._logger.disabled = False

    def is_configured(self) -> bool:
        """Check if the logger has been properly configured"""
        return self._configured

    def configure(self, log_config: dict):
        """Configure logger with provided settings, sets up console and file handlers."""
        with self._lock:
            # Clear existing handlers and filters
            for handler in self._logger.handlers[:]:
                try:
                    handler.close()
                except Exception:
                    pass
                self._logger.removeHandler(handler)

            for filter_obj in self._logger.filters[:]:
                self._logger.removeFilter(filter_obj)

            # Set attributes from config with defaults
            self.level = log_config.get('level', 'INFO')
            self.format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.console = log_config.get('console', True)
            self.include_caller_info = log_config.get('include_caller_info', None)
            self.base_dir = log_config.get('base_dir', None)
            self.filename = log_config.get('filename', None)
            self.max_size_kb = log_config.get('max_size_kb', None)
            self.timezone = log_config.get('timezone', 'US/Pacific')

            try:
                # Set log level
                level = getattr(logging, self.level.upper(), logging.INFO)
                self._logger.setLevel(level)

                # Set up timezone filter
                if self.timezone:
                    try:
                        tz = pytz.timezone(self.timezone)

                        class TimezoneFilter(logging.Filter):
                            def filter(self, record):
                                dt = datetime.fromtimestamp(record.created, tz=tz)
                                record.created = dt.timestamp()
                                return True

                        self._logger.addFilter(TimezoneFilter())
                    except Exception as e:
                        sys.stderr.write(f"Invalid timezone '{self.timezone}': {e}\n")

                # Create formatter
                formatter = logging.Formatter(self.format, datefmt="%H:%M:%S")

                # Add file handler if base_dir and filename are specified
                if self.base_dir and self.filename:
                    try:
                        os.makedirs(self.base_dir, exist_ok=True)
                        log_file_path = os.path.join(self.base_dir, self.filename)

                        if self.max_size_kb:
                            file_handler = SizeRotatingFileHandler(
                                log_file_path,
                                max_size_kb=max(self.max_size_kb, 1024),
                                mode='a',
                                encoding='utf-8'
                            )
                        else:
                            file_handler = logging.FileHandler(
                                log_file_path,
                                mode='a',
                                encoding='utf-8'
                            )

                        file_handler.setFormatter(formatter)
                        file_handler.setLevel(level)
                        self._logger.addHandler(file_handler)
                    except Exception as e:
                        sys.stderr.write(f"Failed to create file handler: {e}\n")

                # Add Rich console handler if enabled
                if self.console:
                    self._add_console_handler(level)

                # Mark as configured
                self._configured = True

                # Log successful configuration
                self.info("Logger configured successfully")

            except Exception as e:
                # Emergency fallback - create basic working logger
                sys.stderr.write(f"Logger configuration failed: {e}\n")
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(levelname)s: %(message)s')
                handler.setFormatter(formatter)
                handler.setLevel(logging.INFO)
                self._logger.addHandler(handler)
                self._configured = True

    def _add_console_handler(self, level):
        """Add appropriate console handler based on caller_info setting"""
        console = Console()
        caller_info = self.include_caller_info

        if caller_info not in ["rome", "rich", None]:
            sys.stderr.write(f"Warning: Invalid caller_info '{caller_info}', using None\n")
            caller_info = None

        try:
            if caller_info == "rome":
                handler = ParentPathRichHandler(
                    console=console,
                    show_time=True,
                    show_path=False,
                    show_level=True,
                    rich_tracebacks=True,
                    markup=True,
                    log_time_format="[%H:%M:%S]"
                )
            elif caller_info == "rich":
                handler = RichHandler(
                    console=console,
                    show_time=True,
                    show_path=True,
                    show_level=True,
                    rich_tracebacks=True,
                    markup=True,
                    log_time_format="[%H:%M:%S]"
                )
            else:
                handler = RichHandler(
                    console=console,
                    show_time=True,
                    show_path=False,
                    show_level=True,
                    rich_tracebacks=True,
                    markup=True,
                    log_time_format="[%H:%M:%S]"
                )

            handler.setLevel(level)
            self._logger.addHandler(handler)

        except Exception as e:
            sys.stderr.write(f"Failed to create RichHandler: {e}\n")
            # Fallback to basic handler
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            handler.setLevel(level)
            self._logger.addHandler(handler)

    def get_file_handler_stats(self):
        """Get rotation statistics from file handler if available"""
        for handler in self._logger.handlers:
            if isinstance(handler, SizeRotatingFileHandler):
                return handler.get_rotation_stats()
        return None

    # Standard logging methods. *args/**kwargs are forwarded to the stdlib
    # logger so %-style calls (e.g. logger.error("x: %s", val)) don't raise a
    # TypeError. This matters most in error-handling paths: a logging crash
    # there would otherwise mask the real exception (see the LLM auth/quota
    # handlers in llm_handler.py).
    def info(self, message, *args, **kwargs):
        self._logger.info(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self._logger.warning(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self._logger.error(message, *args, **kwargs)

    def debug(self, message, *args, **kwargs):
        self._logger.debug(message, *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        self._logger.critical(message, *args, **kwargs)

    def exception(self, message, *args, **kwargs):
        """Log exception with traceback"""
        self._logger.exception(message, *args, **kwargs)

    def assert_true(self, condition: bool, message: str,
                    exception_type=ValueError,
                    log_only: bool = True) -> None:
        """Assert condition is true, log error with traceback if false."""
        if condition:
            return

        if log_only:
            try:
                self.error(f"ASSERTION FAILED: {message}")
            except Exception:
                sys.stderr.write(f"ASSERTION FAILED: {message}\n")
            sys.exit(1)
        else:
            raise exception_type(message)

    def assert_attribute(self, obj, attr_name: str,
                         message: str = None,
                         exception_type=ValueError) -> None:
        """Assert object has specified attribute"""
        if message is None:
            obj_name = obj.__class__.__name__
            message = f"'{attr_name}' not provided in {obj_name} configuration"

        self.assert_true(hasattr(obj, attr_name), message, exception_type)


def get_logger():
    """Per-thread Logger instance.

    Each calling thread gets its own Logger backed by its own named
    ``logging.Logger`` (``Agent.t<TID>``) so concurrent agents don't share
    a handler list — ``Logger.configure()`` only affects the calling
    thread's handlers. Stored on a ``threading.local()`` so the instance
    is cleaned up when the thread exits.
    """
    inst = getattr(_thread_loggers, "instance", None)
    if inst is None:
        inst = Logger(f"Agent.t{threading.get_ident()}")
        _thread_loggers.instance = inst
    return inst
