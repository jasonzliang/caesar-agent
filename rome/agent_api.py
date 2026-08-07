import logging
import socket
from typing import Optional, Dict, Any
import threading

from .logger import get_logger
from .config import API_PORT_RANGE, set_attributes_from_config


def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


class AgentApi:
    def __init__(self, agent: Any, config: Dict = None):
        """
        Initialize the Agent API with configuration

        Args:
            agent: The agent instance (type annotation is Any to avoid circular imports)
            config: Configuration dictionary
        """
        self.agent = agent
        self.config = config or {}

        # Initialize logger
        self.logger = get_logger()

        # Set attributes from config if provided
        set_attributes_from_config(self, self.config, ['host', 'port'])

        self._setup_host_port()
        self._setup_fastapi()
        self._setup_server()
        self.logger.info(f"AgentApi initialized")

    def _setup_server(self):
        import uvicorn

        # Disable Uvicorn's default loggers
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

        # Configure Uvicorn
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="warning",  # Minimize Uvicorn's own logging
            access_log=False,     # Disable access logs entirely
        )

        self.server = uvicorn.Server(config)
        self.server_thread: Optional[threading.Thread] = None

    def _setup_host_port(self):
        """Check for a valid port to use"""
        min_port, max_port = API_PORT_RANGE

        if not self.host:
            self.host = "localhost"
        if not self.port:
            self.port = min_port

        self.logger.assert_true(min_port <= self.port <= max_port,
            f"Ports must be within {min_port}-{max_port} range")

        orig_port = self.port
        while _is_port_in_use(self.port):
            self.port += 1
            if self.port > max_port:
                raise ValueError("Cannot find a valid unused port for API server")
        if orig_port != self.port:
            self.logger.error(f"Port {orig_port} is used already, using {self.port} instead")

    def _setup_fastapi(self):
        """Add API routes to the FastAPI app"""
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        self.app = FastAPI(title="Agent API", description="API to expose agent state.", version="1.2")

        @self.app.get("/agent", response_class=JSONResponse)
        async def get_agent_state():
            # Disable logging for this endpoint
            self.logger.disable()
            try:
                agent, fsm, hist = self.agent, self.agent.fsm, self.agent.history
                summary = agent.get_summary()
                cost = agent.llm_handler.get_cost_summary()

                return {
                    "status": {
                        "current_state": fsm.current_state,
                        "current_action": fsm.current_action,
                        "available_actions": fsm.get_available_actions(),
                        "iteration": agent.curr_iteration
                    },
                    "performance": {
                        "actions_executed": len(hist.actions_executed),
                        "success_rate": round(hist.get_action_success_rate() * 100, 1),
                        "errors_count": len(hist.errors),
                        "repository_progress": {
                            k: summary["repository_progress"][k]
                            for k in ["finished_files", "total_files", "completion_percentage"]
                        }
                    },
                    "recent_activity": {
                        "last_5_actions": [
                            {
                                "iteration": a.get("iteration"),
                                "action": a.get("action"),
                                "result": a.get("action_result"),
                                "transition": f"{a.get('prev_state')} → {a.get('curr_state')}"
                            } for a in hist.get_recent_actions(5)
                        ],
                        "recent_errors": [
                            {
                                "iteration": e.get("iteration"),
                                "state": e.get("state"),
                                "exception": (e.get("exception", "Unknown")[:150] + "...")
                                            if len(e.get("exception", "")) > 150 else e.get("exception", "Unknown")
                            } for e in hist.errors[-3:]
                        ] if hist.has_errors() else []
                    },
                    "resources": {
                        "openai_cost": cost["accumulated_cost"],
                        "cost_limit": cost.get("cost_limit"),
                        "budget_remaining": cost.get("remaining_budget"),
                        "api_calls": cost["call_count"],
                        "active_files": len(getattr(agent.version_manager, 'active_files', {}))
                    },
                    "config": {
                        "name": agent.name,
                        "fsm_type": getattr(agent, 'fsm_type', 'unknown'),
                        "repository": agent.repository.split('/')[-1],
                        "strategy": getattr(agent, 'action_select_strat', 'unknown')
                    },
                    "debug": {
                        "context_keys": list(agent.context.keys())[:10],
                        "agent_id": agent.get_id()
                    }
                }
            except Exception:
                self.logger.exception("/agent/info failed")
                return JSONResponse(content={"error": "Internal server error"}, status_code=500)
            finally:
                # Re-enable logging after endpoint execution
                self.logger.enable()

        @self.app.get("/agent/fsm", response_class=JSONResponse)
        async def get_agent_fsm():
            """Get detailed FSM structure and workflow information"""
            # Disable logging for this endpoint
            self.logger.disable()
            try:
                fsm = self.agent.fsm
                return {
                    "fsm_info": {
                        "type": getattr(self.agent, 'fsm_type', 'unknown'),
                        "current_state": fsm.current_state,
                        "default_state": fsm.default_state,
                        "overview": getattr(fsm, 'overview', 'No overview available')
                    },
                    "states": {
                        name: {
                            "actions": state.get_available_actions(),
                            "description": getattr(state, 'description', 'No description')
                        } for name, state in fsm.states.items()
                    },
                    "graph": fsm.get_graph(),
                    "validation": {
                        "is_valid": fsm.validate_fsm(),
                        "total_states": len(fsm.states),
                        "total_actions": len(fsm.actions)
                    }
                }
            except Exception:
                self.logger.exception("/agent/fsm failed")
                return JSONResponse(content={"error": "Internal server error"}, status_code=500)
            finally:
                # Re-enable logging after endpoint execution
                self.logger.enable()

    def run(self):
        """Run FastAPI in a background thread (non-blocking)."""
        self.logger.info(f"Starting AgentApi server on {self.host}:{self.port}")

        def run_server():
            """Internal method to run the server loop."""
            self.server.run()

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.logger.info("AgentApi server started in background thread")

    def shutdown(self):
        """Gracefully shutdown the Uvicorn server."""
        self.logger.info("Shutting down AgentApi server")
        if hasattr(self, 'server'):
            self.server.should_exit = True
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=1)  # 1 second is plenty
        self.logger.info("AgentApi server shutdown complete")