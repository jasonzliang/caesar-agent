import multiprocessing as mp
import json
import os
import pprint
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from .agent import Agent
from .config import DEFAULT_CONFIG, LOG_DIR_NAME, set_attributes_from_config, merge_with_default_config
from .logger import get_logger
from .process import process_managed


def _agent_worker(agent_name: str, agent_role: str, repository: str, config: Dict,
                  max_iterations: int, result_queue: mp.Queue, suppress_output: bool = False):
    """Simplified worker - process_managed decorator handles main process cleanup"""
    agent = None

    # Suppress output if requested
    if suppress_output:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        try:
            devnull = open(os.devnull, 'w')
            sys.stdout = sys.stderr = devnull
        except:
            pass  # Fallback to normal operation if redirection fails

    try:
        result_queue.put(('starting', agent_name, 'Initializing'))
        agent = Agent(name=agent_name, role=agent_role, repository=repository, config=config)
        result_queue.put(('initialized', agent_name, 'Running'))

        results = agent.run_loop(max_iterations=max_iterations, raise_exception=True)
        results['agent_name'] = agent_name
        result_queue.put(('success', agent_name, results))

    except Exception as e:
        result_queue.put(('error', agent_name, {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'traceback': traceback.format_exc(),
            'agent_name': agent_name
        }))
    finally:
        if agent:
            try:
                agent.shutdown()
            except:
                pass

        # Restore output streams if suppressed
        if suppress_output:
            try:
                sys.stdout, sys.stderr = original_stdout, original_stderr
                devnull.close()
            except:
                pass


@process_managed  # Let decorator handle process cleanup
class MultiAgent:
    """Compact multi-agent manager leveraging process_managed decorator"""

    def __init__(self, agent_role_json: str, repository: str, config: Dict = None):
        self.logger = get_logger()

        # Config setup - pass the config parameter correctly
        self._setup_config(config)

        # Use processed config values, with parameter fallbacks
        self.agent_role_json = agent_role_json or getattr(self, 'agent_role_json', None)
        self.repository = repository or getattr(self, 'repository', None)
        os.makedirs(self.repository, exist_ok=True)

        # Validate and load
        self._validate_paths()
        self.agent_role_config = self._load_role_config()

        # Runtime state - process_managed will handle cleanup automatically
        self.agent_processes: Dict[str, mp.Process] = {}
        self.result_queue = None
        self.results = {}
        self.agent_status = {}

    def _setup_config(self, config_dict: Dict = None) -> Dict:
        """Setup and validate configuration"""
        if config_dict:
            self.config = merge_with_default_config(config_dict)
        else:
            self.logger.info("Using DEFAULT_CONFIG, no config dict provided")
            self.config = DEFAULT_CONFIG.copy()

        # Apply MultiAgent-specific config
        multiagent_config = self.config.get('MultiAgent', {})
        set_attributes_from_config(self, multiagent_config,
            required_attrs=['agent_role_json', 'repository', 'suppress_output'])

        self.logger.info(f"MultiAgent full config:\n{pprint.pformat(self.config)}")
        return self.config

    def _validate_paths(self):
        """Validate required paths exist"""
        for path, name in [(self.agent_role_json, "Agent config")]:
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found: {path}")

    def _load_role_config(self) -> Dict[str, str]:
        """Load and validate agent configuration"""
        try:
            with open(self.agent_role_json, 'r') as f:
                config = json.load(f)
            if not isinstance(config, dict) or not config:
                raise ValueError("Agent config must be non-empty dictionary")
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.agent_role_json}: {e}")

    def run_loop(self, max_iterations: int = 10, timeout_per_agent: int = 0) -> Dict:
        """Execute all agents in parallel - cleanup handled by decorator"""
        # Reset state
        self.results.clear()
        self.agent_status.clear()
        self.agent_processes.clear()

        # Setup fresh queue
        self._setup_queue()

        # Initialize agent status
        self.agent_status = {name: 'pending' for name in self.agent_role_config}

        try:
            self._start_processes(max_iterations)
            self._collect_results(timeout_per_agent)
        except Exception as e:
            self.logger.error(f"Agent execution failed: {e}")
            raise
        # No finally block needed - process_managed handles cleanup

        return {
            'total_agents': len(self.agent_role_config),
            'completed_agents': len(self.results),
            'agent_results': self.results,
            'agent_status': self.agent_status,
            'repository': self.repository,
            'log_dir': self.get_log_dir(),
            'summary': self._create_summary()
        }

    def _setup_queue(self):
        """Setup or reset result queue"""
        if self.result_queue:
            try:
                self.result_queue.close()
                self.result_queue.join_thread()
            except:
                pass
        self.result_queue = mp.Queue()

    def _start_processes(self, max_iterations: int):
        """Start all agent processes"""
        for agent_name, agent_role in self.agent_role_config.items():
            try:
                process = mp.Process(
                    target=_agent_worker,
                    args=(agent_name, agent_role, self.repository, self.config,  # Use processed config
                          max_iterations, self.result_queue, self.suppress_output),
                    daemon=False
                )
                process.start()
                self.agent_processes[agent_name] = process
                self.agent_status[agent_name] = 'starting'
            except Exception as e:
                self._mark_failed(agent_name, 'ProcessStartError', str(e))

    def _collect_results(self, timeout_per_agent: int):
        """Collect results from all agents with timeout handling"""
        completed = 0
        total = len(self.agent_role_config)
        start_time = time.time()

        while completed < total:
            try:
                # Calculate remaining timeout
                elapsed = time.time() - start_time
                if timeout_per_agent > 0:
                    remaining = max(30, timeout_per_agent - elapsed)
                else:
                    remaining = 1e6

                result_type, agent_name, data = self.result_queue.get(timeout=remaining)

                if result_type in ['starting', 'initialized']:
                    self.agent_status[agent_name] = 'running' if result_type == 'initialized' else 'initializing'

                elif result_type == 'success':
                    self.results[agent_name] = data
                    self.agent_status[agent_name] = 'completed'
                    completed += 1

                elif result_type == 'error':
                    self.results[agent_name] = {'error': data}
                    self.agent_status[agent_name] = 'failed'
                    self._log_error(agent_name, data)
                    completed += 1

            except Exception:
                # Timeout - mark remaining agents as failed
                for name, status in self.agent_status.items():
                    if status not in ['completed', 'failed']:
                        self._mark_failed(name, 'TimeoutError', f'Timed out after {timeout_per_agent}s')
                        completed += 1
                break

    def _mark_failed(self, agent_name: str, error_type: str, error_message: str):
        """Mark agent as failed with error details"""
        self.results[agent_name] = {
            'error': {
                'error_type': error_type,
                'error_message': error_message,
                'agent_name': agent_name
            }
        }
        self.agent_status[agent_name] = 'failed'
        self.logger.error(f"Agent '{agent_name}' {error_type}: {error_message}")

    def _log_error(self, agent_name: str, error_data: Dict):
        """Log detailed error information"""
        error_type = error_data.get('error_type', 'Unknown')
        error_msg = error_data.get('error_message', 'No message')
        self.logger.error(f"Agent '{agent_name}' failed: {error_type} - {error_msg}")
        if traceback_str := error_data.get('traceback'):
            self.logger.error(f"Traceback:\n{traceback_str}")

    def _create_summary(self) -> Dict:
        """Generate execution summary with statistics"""
        if not self.results:
            return {"status": "No results available"}

        successful = sum(1 for r in self.results.values() if 'error' not in r)

        # Agent summaries
        agent_summaries = {}
        for name, result in self.results.items():
            if 'error' in result:
                error = result['error']
                agent_summaries[name] = {
                    "status": "FAILED",
                    "error_type": error.get('error_type', 'Unknown'),
                    "error_message": error.get('error_message', 'No message')
                }
            else:
                agent_summaries[name] = {
                    "status": "SUCCESS",
                    "actions_executed": result.get('execution_stats', {}).get('actions_executed', 0),
                    "final_state": result.get('agent_info', {}).get('current_state', 'Unknown')
                }

        summary = {
            "type": "MultiAgent Summary",
            "repository": self.repository,
            "total_agents": len(self.agent_role_config),
            "successful_agents": successful,
            "failed_agents": len(self.results) - successful,
            "success_rate": f"{(successful/len(self.results)*100):.1f}%" if self.results else "0.0%",
            "agent_summaries": agent_summaries
        }

        # Save summary
        try:
            summary_file = Path(self.get_log_dir()) / "multi_agent_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=4, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save summary: {e}")

        return summary

    def get_log_dir(self) -> str:
        """Get or create log directory"""
        log_dir = Path(self.repository) / LOG_DIR_NAME
        log_dir.mkdir(exist_ok=True)
        return str(log_dir)

    def shutdown(self):
        """Cleanup queue resources - processes handled by decorator"""
        if self.result_queue:
            try:
                # Drain queue
                while True:
                    try:
                        self.result_queue.get_nowait()
                    except:
                        break
                self.result_queue.close()
                self.result_queue.join_thread()
            except:
                pass
            self.result_queue = None
