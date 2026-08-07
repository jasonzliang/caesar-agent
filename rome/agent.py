"""Agent class with FSM-based action execution"""
import json
import os
import time
import traceback
from datetime import datetime
from typing import Dict, Callable

from .base_agent import BaseAgent
from .config import DEFAULT_CONFIG, set_attributes_from_config, format_yaml_like
from .fsm_selector import FSMSelector
from .history import AgentHistory
from .repository import RepositoryManager
from .metadata import VersionManager
from .action_selector import ActionSelector
from .kb_client import ChromaClientManager


class Agent(BaseAgent):
    """Agent with FSM-based action execution and repository management"""

    def __init__(self, name: str = None, role: str = None,
                 repository: str = None, config: Dict = None):
        """Initialize Agent with FSM and repository management"""
        super().__init__(name, role, repository, config)

        self._setup_agent_config()
        self._setup_fsm()
        self._setup_context_history()
        self._setup_repository_manager()
        self._setup_version_manager()
        self._setup_action_selection()
        self._setup_knowledge_base()
        self._setup_callback()
        self._setup_agent_api()

        self.draw_fsm_graph()
        self.logger.info(f"Agent '{self.name}' initialized with role:\n{self.role}")

    def _setup_agent_config(self) -> None:
        """Setup Agent-specific configuration attributes"""
        # Exclude keys already set by BaseAgent from constructor args;
        # re-applying them here would clobber them with default-None values.
        base_keys = {'name', 'role', 'repository', 'log_pid'}
        full_agent_config = self.config.get('Agent', {})
        agent_config = {k: v for k, v in full_agent_config.items() if k not in base_keys}
        set_attributes_from_config(self, agent_config,
            [k for k in DEFAULT_CONFIG['Agent'].keys() if k not in base_keys])

        self.logger.assert_true(self.history_context_len > 0,
            "history_context_len must be > 0")
        self.logger.assert_true(self.save_hist_interval > 0,
            "save_hist_interval must be > 0")

    def _setup_fsm(self) -> None:
        """Initialize FSM using FSMSelector"""
        fsm_selector = FSMSelector(fsm_type=self.fsm_type, config=self.config)
        self.fsm = fsm_selector.create_fsm(self.config)
        self.logger.info(f"Initialized {self.fsm_type} FSM: {self.fsm.get_current_state()}")
        self.logger.info(f"FSM Description: {fsm_selector.get_description()}")

    def _setup_context_history(self) -> None:
        """Initialize context and history"""
        self._load_summary_history()
        self.context = {}
        self.history = AgentHistory()
        self.history.set_iteration(self.curr_iteration)
        if self.fsm and self.fsm.current_state:
            self.history.add_initial_state(self.fsm.current_state)

    def _load_summary_history(self) -> None:
        """Load iteration number and summary history from saved files"""
        try:
            summary_file = os.path.join(self.get_log_dir(), f"{self.get_id()}.summary.json")
            history_file = os.path.join(self.get_log_dir(), f"{self.get_id()}.summary_history.json")

            iteration = None; accumulated_cost = None; call_count = None

            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
                iteration = summary_data.get('iteration')
                openai_cost = summary_data.get('summary', {}).get('openai_cost', {})
                accumulated_cost = openai_cost.get('accumulated_cost')
                call_count = openai_cost.get('call_count')
            else:
                self.logger.info("No summary file found, starting fresh at iteration 1")

            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                self.summary_history = history_data.get('summary_history', [])
                self.logger.info(f"Loaded {len(self.summary_history)} summary history entries")
            else:
                self.summary_history = []
                self.logger.info("No summary history file found, starting with empty history")

            if accumulated_cost is not None and call_count is not None:
                self.llm_handler.accumulated_cost = accumulated_cost
                self.llm_handler.call_count = call_count
                self.logger.info(f"Loaded LLM API cost from summary: ${accumulated_cost:.4f}")

            self.curr_iteration = iteration if iteration else 1

        except Exception as e:
            self.logger.error(f"Error loading summary data: {e}, defaulting to iteration 1")
            self.summary_history = []
            self.curr_iteration = 1

    def _setup_repository_manager(self) -> None:
        """Initialize repository manager"""
        repository_config = self.config.get('RepositoryManager', {})
        self.repository_manager = RepositoryManager(
            repository_path=self.repository, config=repository_config)
        self.logger.debug("Repository manager initialized")

    def _setup_version_manager(self) -> None:
        """Initialize version manager"""
        version_config = self.config.get('VersionManager', {})
        db_config = self.config.get('DatabaseManager', {})
        self.version_manager = VersionManager(config=version_config, db_config=db_config)
        self.logger.debug("Version manager initialized")

    def _setup_action_selection(self) -> None:
        """Setup action selection strategy"""
        action_selector_config = self.config.get('ActionSelector', {})
        self.action_selector = ActionSelector(
            strategy=self.action_select_strat, config=action_selector_config)
        self.logger.info(f"Action selector: {self.action_select_strat}")

    def _setup_knowledge_base(self) -> None:
        """Setup knowledge base"""
        self.kb_manager = ChromaClientManager(agent=self)
        self.logger.info(f"Knowledge base initialized: {self.kb_manager.info()}")

    def _setup_agent_api(self) -> None:
        """Setup agent API if enabled"""
        if not self.agent_api:
            self.logger.debug("Agent API disabled, skipping initialization")
            return
        from .agent_api import AgentApi
        api_config = self.config.get('AgentApi', {})
        self.agent_api = AgentApi(agent=self, config=api_config)
        self.agent_api.run()
        self.logger.info("Agent API initialized and running")

    def _setup_callback(self) -> None:
        """Setup callback variable"""
        self.callback = None

    def draw_fsm_graph(self, output_path: str = None) -> str:
        """Draw FSM graph to PNG file"""
        if not self.draw_fsm: return None
        output_path = output_path or os.path.join(
            self.get_log_dir(), f"{self.get_id()}.fsm.png")
        self.logger.info(f"Drawing FSM graph to {output_path}")
        return self.fsm.draw_graph(output_path)

    def set_callback(self, callback: Callable[[int], None]) -> None:
        """Set callback function for end of each iteration"""
        self.logger.assert_true(callable(callback),
            f"Callback must be callable, got {type(callback)}")
        self.callback = callback

    def run_loop(self, max_iterations: int = 10, stop_on_error: bool = True,
                 raise_exception: bool = False, auto_resume: bool = True) -> Dict:
        """
        Main execution loop using FSM and ActionSelector

        Args:
            max_iterations: Maximum number of iterations
            stop_on_error: Whether to stop on errors or continue
            raise_exception: Whether to reraise exception if error encountered
            auto_resume: Whether to automatically resume from last iteration

        Returns:
            Dict containing loop execution results
        """
        def process_summary():
            summary = self.get_summary()
            self._print_summary(summary)
            self._save_summary(summary)
            return summary

        self.logger.assert_true(self.fsm is not None,
            "FSM has not been properly initialized")

        self.logger.info(f"Starting agent loop for {max_iterations} iterations")
        start_iteration = self.curr_iteration
        end_iteration = max_iterations
        summary = None

        for iteration in range(start_iteration, end_iteration + 1):
            try:
                self.curr_iteration = iteration
                self.history.set_iteration(iteration)
                self.logger.info(f"Loop iteration {iteration}/{end_iteration}")
                self.logger.info(f"Current state: {self.fsm.current_state}")

                # Validate active file state
                self.version_manager.validate_active_files(self)

                # Check context on first iteration
                if iteration == start_iteration:
                    self.fsm.check_context(self)

                # Use ActionSelector for action selection
                chosen_action, reason, should_continue = self.action_selector.select_action(
                    agent=self, iteration=iteration, stop_on_error=stop_on_error)

                # Handle action selection results
                if not should_continue:
                    if chosen_action is None:
                        break  # Exit loop
                    else:
                        continue  # Continue to next iteration

                # Store previous state
                prev_state = self.fsm.current_state

                # Execute action through FSM
                self.logger.info(f"Executing {chosen_action} action")
                success = self.fsm.execute_action(chosen_action, self)
                self.logger.info(f"Action executed (success: {success}), new state: {self.fsm.current_state}")

                # Record execution in history
                self.history.add_action_execution(
                    iteration=iteration,
                    action=chosen_action,
                    action_result=success,
                    action_reason=reason,
                    prev_state=prev_state,
                    curr_state=self.fsm.current_state
                )

                # Print and save summary
                summary = process_summary()

                # Handle callbacks
                if self.callback:
                    try:
                        self.callback(self, iteration)
                    except Exception as e:
                        self.logger.error(f"Callback error: {e}")
                        self.logger.error(traceback.format_exc())

            except Exception as e:
                if raise_exception:
                    raise
                error_msg = f"Agent run_loop error at iteration {iteration}: {str(e)}"
                stack_trace = traceback.format_exc()
                self.logger.error(error_msg)
                self.logger.error(stack_trace)
                self.history.add_error(iteration, f"{error_msg}\n{stack_trace}",
                    self.fsm.current_state, str(e))
                if stop_on_error:
                    break
                else:
                    self.fsm.reset(self)
                    self.history.reset()
                    continue

        # Record final state
        self.history.set_final_state(self.fsm.current_state, self.context)
        if self.history.has_errors():
            self.logger.info(f"Loop completed with {len(self.history.errors)} errors")

        # Set current iteration for next run_loop call
        self.curr_iteration = end_iteration + 1

        return summary if summary else process_summary()

    def get_summary(self, recent_hist_len: int = None) -> Dict:
        """Get comprehensive summary of agent state"""
        def _get_context_keys(obj, prefix="", depth=0):
            if depth >= 2 or not isinstance(obj, dict):
                return []
            keys = []
            for k, v in obj.items():
                full_key = f"{prefix}{k}"
                if isinstance(v, dict):
                    keys.extend(_get_context_keys(v, f"{full_key}.", depth + 1))
                else:
                    keys.append(full_key)
            return keys

        recent_hist_len = recent_hist_len or self.history_context_len

        # Context info
        ctx_keys = _get_context_keys(self.context)

        # History summary
        actions_count = len(self.history.actions_executed)
        errors_count = len(self.history.errors)
        success_rate = 0.0
        recent_actions_summary = {}

        if actions_count:
            successful = sum(1 for a in self.history.actions_executed
                           if a['action_result'] == 'success')
            success_rate = (successful / actions_count) * 100

            recent = self.history.actions_executed[-recent_hist_len:]
            for action_data in recent:
                action_name = action_data.get('action', 'unknown')
                recent_actions_summary[action_name] = recent_actions_summary.get(action_name, 0) + 1

        # Repository completion stats
        completion = self.repository_manager.get_repository_completion_stats(self)

        # LLM cost summary
        cost_summary = self.llm_handler.get_cost_summary()

        return {
            "agent_info": {
                "name": self.name,
                "current_state": self.fsm.current_state,
                "iteration": self.curr_iteration,
                "repository": self.repository
            },
            "context": {
                "keys": ctx_keys,
                "key_count": len(ctx_keys)
            },
            "execution_stats": {
                "actions_executed": actions_count,
                "errors": errors_count,
                "success_rate": f"{success_rate:.1f}%",
                "recent_actions": recent_actions_summary,
                "recent_actions_count": len(recent) if actions_count else 0
            },
            "repository_progress": {
                "finished_files": completion['finished_files'],
                "total_files": completion['total_files'],
                "completion_percentage": completion['completion_percentage']
            },
            "openai_cost": {
                "model": cost_summary['model'],
                "call_count": cost_summary['call_count'],
                "accumulated_cost": cost_summary['accumulated_cost'],
                "cost_limit": cost_summary.get('cost_limit'),
                "remaining_budget": cost_summary.get('remaining_budget'),
                "usage_percentage": f"{(cost_summary['accumulated_cost'] / cost_summary['cost_limit']) * 100:.1f}%" if cost_summary.get('cost_limit') else None
            }
        }

    def _print_summary(self, summary: Dict) -> None:
        """Print summary to console in YAML-like format"""
        self.logger.info("="*80)
        self.logger.info("AGENT SUMMARY")
        self.logger.info("="*80)
        for line in format_yaml_like(summary):
            self.logger.info(line)
        self.logger.info("="*80)

    def _save_summary(self, summary: Dict) -> None:
        """Save full summary history and most recent summary to JSON files"""
        log_dir = self.get_log_dir()

        # Check if we should save full history this iteration
        save_full_history = (self.curr_iteration == 1 or
                            self.curr_iteration % self.save_hist_interval == 0)

        # Create recent summary data
        recent_summary_data = {
            'iteration': self.curr_iteration,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'epoch_time': time.time(),
            'summary': summary
        }

        # Only append to summary_history when saving full history
        if save_full_history:
            self.summary_history.append(recent_summary_data)

        try:
            # Always save most recent summary
            recent_summary_file = os.path.join(log_dir, f"{self.get_id()}.summary.json")
            with open(recent_summary_file, 'w', encoding='utf-8') as f:
                json.dump(recent_summary_data, f, indent=4, default=str)
            self.logger.debug(f"Latest summary saved to: {recent_summary_file}")

            # Conditionally save full history
            if save_full_history:
                full_history_file = os.path.join(log_dir, f"{self.get_id()}.summary_history.json")
                full_summary_data = {
                    "agent_id": self.get_id(),
                    "summary_count": len(self.summary_history),
                    "summary_history": self.summary_history
                }
                with open(full_history_file, 'w', encoding='utf-8') as f:
                    json.dump(full_summary_data, f, indent=4, default=str)
                self.logger.debug(f"Full summary history ({len(self.summary_history)} entries) saved to: {full_history_file}")
            else:
                self.logger.debug(f"Skipping full history save (iteration {self.curr_iteration}, interval {self.save_hist_interval})")

        except Exception as e:
            self.logger.error(f"Failed to save summary files: {e}")

    def shutdown(self) -> None:
        """Clean up Agent resources"""
        if self.shutdown_called: return
        self.shutdown_called = True
        try:
            if hasattr(self, 'agent_api') and self.agent_api:
                self.agent_api.shutdown()
            if hasattr(self, 'version_manager'):
                self.version_manager.shutdown(self)
            if hasattr(self, 'kb_manager'):
                self.kb_manager.shutdown()
            super().shutdown()
            self.logger.info("Agent shutdown completed successfully")
        except Exception as e:
            self.logger.error(f"Error during Agent shutdown: {e}")
            raise
