<p align="center">
  <img src="https://img.freepik.com/premium-photo/ancient-rome-sunset-scenery-buildings-ruins-summer-beautiful-sunny-panorama-historical-city-houses-sun-sky-concept-roman-empire-antique-painting-background_788189-12960.jpg" alt="Rome" width="500"/>
</p>

# Rome

Rome is a Python library for building intelligent AI agents with Finite State Machine (FSM) architecture. This library provides a structured framework for creating AI-powered agents that can analyze, edit, test, and execute code with LLM assistance, specifically designed for automated software development workflows.

> For the project overview and the Caesar research agent built on Rome, see the [repo root README](../README.md).

## Overview

Rome is built around a configurable Finite State Machine architecture that enables AI agents to navigate through different operational states while performing complex software development tasks. The library includes comprehensive components for:

- **FSM-based agent orchestration** with configurable workflows
- **Code repository analysis and management**
- **Intelligent code editing and improvement**
- **Automated test creation and execution**
- **Version control and file management**
- **Multi-agent coordination** for collaborative workflows
- **Benchmark evaluation** against coding datasets like HumanEval+

### FSM Workflows

Rome supports multiple FSM types for different use cases:

#### Simple FSM (6-State Development Lifecycle)
The comprehensive workflow for full development cycles:

1. **IDLE → CODE_LOADED**: Search and load code files using tournament/priority search
2. **CODE_LOADED → CODE_EDITED or TEST_EDITED**: Edit code or write tests based on analysis
3. **CODE_EDITED → TEST_EDITED**: After code changes, create comprehensive tests
4. **TEST_EDITED → CODE_EXECUTED_PASS/FAIL**: Execute code with tests and validate results
5. **CODE_EXECUTED_PASS → IDLE**: Successful completion, reset for next task
6. **CODE_EXECUTED_FAIL → CODE_LOADED/IDLE**: Intelligent recovery with version revert or restart

#### Minimal FSM (2-State Analysis Cycle)
Lightweight workflow for code exploration and analysis:

1. **IDLE → CODE_LOADED**: Search and examine code files
2. **CODE_LOADED → IDLE**: Reset and repeat for iterative analysis

## Key Features

### Advanced Agent Capabilities

- **Smart Action Selection**: Tournament-based and priority-driven action selection strategies
- **Context Management**: Maintains execution context and history across iterations
- **Intelligent Recovery**: Automatic fallback states and version management for failure scenarios
- **Cost Management**: Built-in OpenAI API cost tracking and limits
- **Patience System**: Prevents infinite loops with configurable retry limits

### Repository Management

- **File Type Filtering**: Configurable file type support (`.py`, etc.)
- **Version Control**: Automatic versioning of code changes with metadata tracking
- **Completion Tracking**: Progress monitoring across repository files
- **Database Integration**: Persistent storage for agent state and history

### Benchmarking & Evaluation

- **HumanEval+ Integration**: Built-in support for coding benchmark evaluation
- **EvalPlus Compatibility**: Seamless integration with EvalPlus evaluation framework
- **Periodic Evaluation**: Background evaluation during agent execution
- **Multi-Dataset Support**: Support for HumanEval and MBPP datasets

## Architecture

### Core Components

- **Agent**: Main orchestrator with FSM management, context handling, and execution control
- **FSMSelector**: Factory for creating different FSM types (`minimal`, `simple`)
- **ActionSelector**: Configurable strategies for intelligent action selection
- **RepositoryManager**: File system operations and repository analysis
- **VersionManager**: Code versioning and change tracking with database backend
- **LLMHandler**: LLM interface with cost management and response parsing
- **AgentHistory**: Execution tracking and performance analytics

### States & Actions

**Available States:**
- `IdleState`: Waiting for tasks, ready to search
- `CodeLoadedState`: File selected and loaded for processing
- `CodeEditedState`: Code has been modified
- `TestEditedState`: Tests created or updated
- `CodeExecutedPassState`: Successful test execution
- `CodeExecutedFailState`: Failed execution requiring recovery

**Key Actions:**
- `TournamentSearchAction`: Advanced search with ranking algorithms
- `PrioritySearchAction`: Priority-based file selection
- `EditCodeAction`: AI-assisted code modification
- `EditTestAction`: Comprehensive test generation
- `ExecuteCodeAction`: Code and test execution with validation
- `RevertCodeAction`: Intelligent version recovery
- `AdvancedResetAction`: Context cleanup and state reset

## Getting Started

### Prerequisites

- Linux or macOS with Python 3.10+
- OpenAI API key or compatible LLM endpoint
- Optional: `graphviz` for FSM visualization

### Installation

#### From Source

```bash
git clone https://github.com/cognizant-ai-lab/caesar-agent.git
cd caesar-agent
pip install -e .
pip install -e .[dev]  # Include development dependencies
```

#### Dependencies

```bash
# Core dependencies
pip install openai pyyaml

# Benchmarking (optional)
pip install evalplus

# Visualization (optional)
pip install graphviz streamlit
```

### Environment Setup

```bash
export OPENAI_API_KEY=your_api_key_here
```

### Basic Usage

#### Single Agent Workflow

```python
from rome.agent import Agent

# Create agent with simple FSM
agent = Agent(
    name="CodeExpert",
    role="You are an expert Python developer focused on writing clean, efficient, and well-tested code",
    repository="./my_project",
    config={
        "Agent": {
            "fsm_type": "simple",
            "patience": 3,
            "action_select_strat": "smart"
        },
        "LLMHandler": {
            "model": "gpt-4o",
            "cost_limit": 10.0
        }
    }
)

# Run agent for specified iterations
results = agent.run_loop(max_iterations=20, stop_on_error=False)
agent.shutdown()
```

#### Multi-Agent Coordination

```python
from rome.multi_agent import MultiAgent

# Define agent roles
agent_roles = {
    "MathSolver": "Expert in mathematical algorithms and numerical computations",
    "StringProcessor": "Specialist in text processing and string manipulation",
    "TestExpert": "Focused on comprehensive test coverage and edge cases"
}

# Create multi-agent system
multi_agent = MultiAgent(
    agent_role_json="agent_roles.json",
    repository="./project",
    config=config
)

results = multi_agent.run_loop(max_iterations=30)
```

#### Benchmark Evaluation

```python
from benchmark.evalplus_single_agent import EvalPlusBenchmark

# Run HumanEval+ benchmark
benchmark = EvalPlusBenchmark(
    benchmark_dir="./benchmark_results",
    config_path="benchmark/config/single_agent_basic.yaml",
    dataset="humaneval"
)

# Complete benchmark with evaluation
results, results_file = benchmark.run_benchmark(
    max_iterations=50,
    num_problems=10,
    run_evaluation=True,
    eval_interval=1800  # Periodic evaluation every 30 minutes
)

benchmark.print_summary(results)
```

### Configuration

Rome uses YAML-based configuration with extensive customization options:

```yaml
# Agent Configuration
Agent:
  name: "CodeExpert"
  fsm_type: "simple"  # or "minimal"
  patience: 3
  action_select_strat: "smart"
  agent_api: true
  history_context_len: 15

# LLM Configuration
LLMHandler:
  model: "gpt-4o"
  temperature: 0.1
  max_tokens: 8192
  cost_limit: 50.0

# Action Configuration
TournamentSearchAction:
  batch_size: 5

EditCodeAction:
  max_iterations: 3

ExecuteCodeAction:
  timeout: 30

# Repository Settings
RepositoryManager:
  file_types: [".py", ".js", ".ts"]
  exclude_dirs: ["__pycache__", ".git"]

# Logging
Logger:
  level: "DEBUG"
  console: true
  include_caller_info: "rome"
```

## Advanced Features

### FSM Visualization

Generate visual representations of your agent's state machine:

```python
# Automatic graph generation
agent.draw_fsm_graph()  # Saves to <repository>/__rome__/agent_<name>_<pid>.fsm.png

# Custom path
agent.draw_fsm_graph("/path/to/output.png")
```

### Custom FSM Development

Extend Rome with custom state machines:

```python
from rome.fsm_selector import FSMBuilder
from rome.fsm import FSM

class CustomFSMBuilder(FSMBuilder):
    def get_description(self) -> str:
        return "Custom FSM for specialized workflow"

    def build_fsm(self, config: Dict) -> FSM:
        fsm = FSM(config.get("FSM", {}))
        # Add custom states and transitions
        return fsm

# Register custom FSM
FSMSelector.BUILDERS['custom'] = CustomFSMBuilder
```

### Action Selection Strategies

Configure different action selection approaches:

```python
# Smart selection with context awareness
"action_select_strat": "smart"

# Original (default) sequential selection strategy
"action_select_strat": "original"
```

### Callback Integration

Add custom behavior during agent execution:

```python
def evaluation_callback(agent, iteration):
    if iteration % 10 == 0:
        # Custom evaluation logic
        results = evaluate_agent_progress(agent)
        log_results(results)

agent.set_callback(evaluation_callback)
agent.run_loop(max_iterations=100)
```

## Benchmarking & Evaluation

### HumanEval+ Integration

Rome includes built-in support for evaluating agents against coding benchmarks:

```bash
# Run benchmark from command line
python benchmark/evalplus_single_agent.py ./results benchmark/config/single_agent_basic.yaml \
    --dataset humaneval \
    --num-problems 20 \
    --max-iterations 50 \
    --eval-interval 1800
```

### Performance Metrics

Track agent performance with comprehensive metrics:

- **Success Rate**: Percentage of successful action executions
- **Completion Rate**: Repository file completion percentage
- **Cost Efficiency**: API cost per successful task
- **Execution Time**: Time per iteration and total runtime
- **Error Recovery**: Success rate of failure recovery mechanisms

## Development & Testing

### Running Tests

```bash
# Manual testing scripts
python test/test_single_agent.py   # Single agent testing
python test/test_multi_agent.py    # Multi-agent testing

# Custom test scenarios
pytest test/ -v
```

### Development Tools

```bash
# Code formatting
black rome/
isort rome/

# Type checking
mypy rome/

# Linting
flake8 rome/
```

## API Reference

### Agent API

When `agent_api: true`, Rome exposes an HTTP API:

```bash
# Agent introspection
GET /agent       # Current agent state and info
GET /agent/fsm   # FSM structure and current state
```

### Key Classes

- **Agent**: Main orchestrator class
- **MultiAgent**: Coordination of multiple agents
- **FSMSelector**: FSM factory and builder
- **EvalPlusBenchmark**: Benchmark evaluation framework
- **ActionSelector**: Configurable action selection strategies

## Examples & Use Cases

### Code Quality Improvement
- Analyze existing codebases for improvement opportunities
- Automatically generate comprehensive test suites
- Refactor code following best practices

### Automated Development
- Complete partial function implementations
- Generate documentation and comments
- Fix bugs and logical errors

### Benchmark Evaluation
- Evaluate agent performance on coding challenges
- Compare different FSM configurations
- Measure improvement over time

## Limitations & Considerations

- **LLM Dependencies**: Requires access to OpenAI API or compatible endpoints
- **Cost Management**: Monitor API usage with built-in cost limits and tracking
- **File System Access**: Agents modify files directly (use with version control)
- **Execution Safety**: Code execution happens in local environment (use containers for isolation)
- **Port Requirements**: Agent API uses ports 40000-41000 range for web interface
- **Real-time Monitoring**: Streamlit dashboard requires agent API to be enabled

## Contributing

Contributions are welcome once the project reaches stable release. Focus areas:

- New FSM types and workflows
- Additional action implementations
- Enhanced benchmarking capabilities
- Performance optimizations

## License

See LICENSE file for details.
