"""Checkpoint Manager - Save and load exploration state"""
import gzip
import json
import os
from datetime import datetime

import networkx as nx

from rome.logger import get_logger


class CheckpointManager:
    """Handles checkpoint save/load for CaesarAgent"""

    def __init__(self, agent):
        self.agent = agent
        self.logger = get_logger()

    def get_path(self) -> str:
        """Get checkpoint file path"""
        return os.path.join(self.agent.get_log_dir(), f"{self.agent.get_id()}.checkpoint.json")

    def save(self, iteration: int, save_graph_interval: int | None = None) -> None:
        """Save exploration state with optional graph data.

        ``save_graph_interval`` overrides the agent's default for this call.
        Pass ``1`` to force a graph snapshot on every save (used by
        quick_explore where ``iteration`` is a per-URL id in arbitrary
        completion order, so the agent's modulo gate rarely fires).
        """
        try:
            if not self.agent.url_stack:
                error_msg = f"Cannot save checkpoint on iteration {iteration}: empty url_stack"
                raise RuntimeError(error_msg)
            graph_interval = (
                save_graph_interval
                if save_graph_interval is not None
                else self.agent.save_graph_interval
            )

            checkpoint_data = {
                'role': self.agent.role,
                'iteration': iteration,
                'current_url': self.agent.current_url,
                'current_depth': self.agent.current_depth,
                'url_stack': self.agent.url_stack,
                'failed_urls': list(self.agent.failed_urls),
                'visited_urls': self.agent.visited_urls,
                'web_searches_used': self.agent.web_searches_used,
                'reseeds_used': self.agent.reseeds_used,
                'traversal_history': self.agent.traversal_history,
                'graph': nx.node_link_data(self.agent.graph, edges="edges"),
                'config': {
                    'starting_url': self.agent.starting_url,
                    'starting_query': self.agent.starting_query,
                    'allowed_domains': self.agent.allowed_domains,
                    'max_iterations': self.agent.max_iterations,
                    'max_depth': self.agent.max_depth,
                },
                'timestamp': datetime.now().isoformat(),
                'cost': {
                    'accumulated_cost': self.agent.llm_handler.accumulated_cost,
                    'call_count': self.agent.llm_handler.call_count,
                    'sessions': self.agent.session_costs + [{
                        'session_cost': self.agent.llm_handler.accumulated_cost - self.agent.session_start_cost,
                        'session_calls': self.agent.llm_handler.call_count - self.agent.session_start_calls,
                        'timestamp': datetime.now().isoformat(),
                    }],
                },
            }

            # Save checkpoint
            with open(self.get_path(), 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Checkpoint saved on iteration {iteration}")

            # Save separate graph file if at interval
            if iteration % graph_interval == 0 or iteration == self.agent.max_iterations:
                knowledge_graph = checkpoint_data['graph']
                knowledge_graph['iteration'] = iteration
                knowledge_graph['starting_url'] = self.agent.starting_url
                # Atomic write: temp + rename. gzip.open(target, 'wt')
                # truncates the destination before filling it, so a reader
                # tailing the dir for new graph_iter files can see a
                # zero-byte / partially-written file.
                graph_path = os.path.join(self.agent.get_repo(),
                    f"{self.agent.get_id()}.graph_iter{iteration}.json.gz")
                tmp_path = graph_path + ".tmp"
                with gzip.open(tmp_path, 'wt', encoding='utf-8') as f:
                    json.dump(knowledge_graph, f, indent=4, ensure_ascii=False)
                os.replace(tmp_path, graph_path)
                self.logger.info(f"Knowledge graph saved on iteration {iteration}")

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint on iteration {iteration}: {e}")

    def load(self) -> bool:
        """Load exploration state from checkpoint"""
        checkpoint_path = self.get_path()
        if not os.path.exists(checkpoint_path):
            return False

        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check config changes
            config = data.get('config', {})
            if config.get('starting_query') != self.agent.starting_query:
                self.logger.error(f"Checkpoint starting_query mismatch: '{config.get('starting_query')}' vs '{self.agent.starting_query}'")
            if config.get('starting_url') != self.agent.starting_url:
                self.logger.error(f"Checkpoint starting_url mismatch")
            if config.get('allowed_domains') != self.agent.allowed_domains:
                self.logger.error(f"Checkpoint allowed_domains mismatch: {config.get('allowed_domains')} vs {self.agent.allowed_domains}")

            # Restore state
            self.agent.current_iteration = data.get('iteration', self.agent.current_iteration)
            self.agent.web_searches_used = data.get('web_searches_used', self.agent.web_searches_used)
            self.agent.reseeds_used = data.get('reseeds_used', self.agent.reseeds_used)
            self.agent.traversal_history = data.get('traversal_history', self.agent.traversal_history)
            self.agent.visited_urls = data.get('visited_urls', self.agent.visited_urls)
            self.agent.url_stack = data.get('url_stack', self.agent.url_stack)
            if not self.agent.url_stack:
                self.logger.error("Invalid checkpoint: empty url_stack")
                return False
            self.agent.current_depth = len(self.agent.url_stack)
            self.agent.current_url = self.agent.url_stack[-1]
            # Disabled due to having separating loading mechanism or not necessary
            # self.agent.failed_urls = set(data.get('failed_urls', self.agent.failed_urls))

            # Restore graph inline
            self.agent.graph = nx.node_link_graph(data.get('graph', self.agent.graph), edges="edges")

            # Restore accumulated cost from checkpoint
            cost_data = data.get('cost', {})
            self.agent.llm_handler.accumulated_cost = cost_data.get('accumulated_cost', 0.0)
            self.agent.llm_handler.call_count = cost_data.get('call_count', 0)
            self.agent.session_costs = cost_data.get('sessions', [])
            self.agent.session_start_cost = self.agent.llm_handler.accumulated_cost
            self.agent.session_start_calls = self.agent.llm_handler.call_count

            # Restore role from checkpoint if enabled
            if self.agent.load_saved_role:
                self.agent.role = data.get('role', self.agent.role)
            else:
                self.agent._update_role()

            self.logger.info(f"Checkpoint loaded from {data.get('timestamp')}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False
