"""
Compact process management with automatic cleanup, signal handling, and zombie prevention.

Usage:
    @process_managed
    class MyClass:
        def shutdown(self): pass  # Optional existing cleanup

    # Auto-cleanup on SIGTERM/SIGINT with zombie prevention
    # Manual: obj.shutdown() or obj.cleanup_zombies()

Dependencies: pip install psutil
"""
import signal, sys, os, threading, traceback, io
from typing import List, Callable, Optional
import psutil


class ProcessManager:
    """Compact process manager with zombie prevention"""

    def __init__(self, name: str = "ProcessManager", timeout: int = 10):
        self.name, self.timeout = name, timeout
        self.callbacks: List[Callable] = []
        self.shutdown_called = False
        self._lock = threading.Lock()
        self.parent_pid = os.getpid()

    def add_callback(self, callback: Callable) -> None:
        if callback and callable(callback):
            self.callbacks.append(callback)

    def setup_signals(self, logger=None) -> None:
        """Setup signal handlers - only works in main thread"""
        import threading

        # Skip if not in main thread
        if threading.current_thread() is not threading.main_thread():
            if logger:
                logger.debug(f"{self.name}: Skipping signal setup (worker thread)")
            return

        def handler(signum, frame):
            if logger:
                logger.info(f"{self.name} received signal {signum}, initiating shutdown")
                if frame:
                    filename = frame.f_code.co_filename
                    line_number = frame.f_lineno
                    function_name = frame.f_code.co_name
                    logger.info(f"Interrupted at: {filename}:{line_number} in {function_name}()")
                try:
                    buf = io.StringIO()
                    traceback.print_stack(frame, file=buf)
                    logger.info("Execution stack when interrupted:")
                    logger.info(buf.getvalue())
                    buf.close()
                except Exception as e:
                    logger.warning(f"Failed to capture stack trace: {e}")

            try:
                self.shutdown()
            except Exception as e:
                if logger: logger.error(f"Shutdown error: {e}")
                self._force_cleanup()
            finally:
                sys.exit(0)

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def get_children(self) -> List[psutil.Process]:
        try:
            return psutil.Process(self.parent_pid).children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return []

    def get_zombies_python(self, children_only: bool = True) -> List[psutil.Process]:
        zombies = []
        try:
            if children_only:
                # Check children only
                processes = [p for p in self.get_children() if self._is_zombie_python(p)]
                return processes
            else:
                # Check system-wide (children + others)
                processes = psutil.process_iter(['pid', 'name', 'cmdline', 'status'])

            for proc_info in processes:
                try:
                    if proc_info.info['status'] != psutil.STATUS_ZOMBIE:
                        continue

                    proc = psutil.Process(proc_info.info['pid'])
                    if self._is_python_info(proc_info.info):
                        zombies.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except Exception:
            # Fallback: just check children
            zombies.extend([p for p in self.get_children() if self._is_zombie_python(p)])
        return zombies

    def _is_python_info(self, proc_info: dict) -> bool:
        """Check if process info indicates Python process"""
        name = proc_info.get('name', '').lower()
        if any(p in name for p in ['python', 'py']):
            return True
        cmdline = proc_info.get('cmdline', [])
        if cmdline:
            cmd_str = ' '.join(cmdline).lower()
            return any(p in cmd_str for p in ['python', '.py'])
        return False

    def _is_zombie_python(self, proc: psutil.Process) -> bool:
        """Check if process is zombie Python"""
        try:
            return proc.status() == psutil.STATUS_ZOMBIE and self._is_python(proc)
        except:
            return False

    def _is_python(self, proc: psutil.Process) -> bool:
        """Check if process is Python"""
        try:
            name = proc.name().lower()
            if any(p in name for p in ['python', 'py']):
                return True
            cmdline = ' '.join(proc.cmdline()).lower()
            return any(p in cmdline for p in ['python', '.py'])
        except:
            return False

    def _terminate_process(self, proc: psutil.Process) -> bool:
        """Terminate a process"""
        try:
            if not proc.is_running():
                return True

            proc.terminate()
            try:
                proc.wait(timeout=min(3, self.timeout))
                return True
            except psutil.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=self.timeout)
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return True
        except:
            return False

    def cleanup_zombies(self, children_only: bool = True, logger=None) -> int:
        """"Finad and terminate all zombie processes"""
        zombies = self.get_zombies_python(children_only)
        if not zombies:
            return 0

        if logger:
            logger.info(f"Cleaning {len(zombies)} zombie Python processes")

        cleaned = 0
        for zombie in zombies:
            try:
                if zombie.is_running():
                    zombie.terminate()
                    zombie.wait(timeout=1)
                cleaned += 1
            except:
                try:
                    zombie.kill()
                    cleaned += 1
                except:
                    pass

        if logger and cleaned:
            logger.info(f"Cleaned {cleaned} zombies")
        return cleaned

    def _force_cleanup(self):
        """Emergency cleanup without error handling"""
        try:
            for child in self.get_children():
                try:
                    child.kill()
                    child.wait(timeout=1)
                except:
                    pass
        except:
            pass

    def shutdown_processes(self, cleanup_zombies: bool = True, logger=None):
        if cleanup_zombies:
            self.cleanup_zombies(logger=logger)

        for child in self.get_children():
            self._terminate_process(child)

    def shutdown(self, cleanup_zombies: bool = True):
        with self._lock:
            if self.shutdown_called:
                return
            self.shutdown_called = True

        logger = getattr(self, 'logger', None)

        # Run callbacks
        for callback in self.callbacks:
            try:
                callback()
            except:
                pass

        # Cleanup processes
        self.shutdown_processes(cleanup_zombies, logger)


def process_managed(timeout: int = 10, auto_signals: bool = True, cleanup_zombies: bool = True):
    """
    Compact decorator for automatic process management with zombie prevention.

    Args:
        timeout: Process termination timeout (default: 10)
        auto_signals: Setup SIGTERM/SIGINT handlers (default: True)
        cleanup_zombies: Cleanup zombie processes (default: True)
    """
    def decorator(cls):
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)

            # Setup manager
            name = f"{cls.__name__}_{getattr(self, 'name', 'unknown')}"
            self.process_manager = ProcessManager(name, timeout)

            # Preserve existing shutdown
            if hasattr(self, 'shutdown') and callable(self.shutdown):
                self.process_manager.add_callback(self.shutdown)

            # Replace shutdown
            self.shutdown = lambda: self.process_manager.shutdown(cleanup_zombies)

            # Setup signals
            if auto_signals:
                self.process_manager.setup_signals(getattr(self, 'logger', None))

        def cleanup_zombies_method(self, children_only: bool = True):
            logger = getattr(self, 'logger', None)
            return self.process_manager.cleanup_zombies(children_only, logger)

        cls.__init__ = new_init
        cls.cleanup_zombies = cleanup_zombies_method
        return cls

    # Support @process_managed and @process_managed()
    if callable(timeout):
        cls = timeout
        timeout = 10
        return decorator(cls)
    return decorator


# Standalone functions
def cleanup_child_processes(timeout: int = 10, cleanup_zombies: bool = True):
    """Cleanup all child processes and optionally zombies"""
    manager = ProcessManager("manual", timeout)
    manager.shutdown_processes(cleanup_zombies)


def cleanup_zombie_python_processes(children_only: bool = True) -> int:
    """Find and cleanup zombie Python processes"""
    return ProcessManager("zombie").cleanup_zombies(children_only)
