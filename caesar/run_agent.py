#!/usr/bin/env python3
"""
CaesarAgent CLI Runner

Single experiment:
    python run_agent.py [repository] config [--max-iterations N] [-q QUERY]

Batch experiments (parallel):
    python run_agent.py -b batch.jsonl [-n NUM_WORKERS]
"""

import argparse
import hashlib
import json
import os
import pprint
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import traceback

# Add the parent directory to sys.path to import from the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from caesar.caesar_agent import CaesarAgent
from rome.config import LOG_DIR_NAME, load_config, format_yaml_like
from rome.llm_handler import FatalLLMError
from rome.logger import get_logger


# =============================================================================
# Path and config resolution
# =============================================================================

CONFIG_PRESETS = {'regular', 'mini', 'nano'}

USER_DATA_DIR = Path.home() / ".caesar"
USER_CONFIG_DIR = USER_DATA_DIR / "configs"


def _is_source_checkout() -> bool:
    # Source-checkout: caesar/ sits next to a pyproject.toml. Pip installs land
    # in site-packages/caesar/ with no pyproject.toml above them.
    return (Path(__file__).resolve().parent.parent / "pyproject.toml").exists()


def default_result_root() -> Path:
    env = os.environ.get("CAESAR_RESULT_DIR")
    if env:
        return Path(env).expanduser().resolve()
    if _is_source_checkout():
        return Path(__file__).resolve().parent / "result"
    return USER_DATA_DIR / "result"


def resolve_config_path(config_path):
    """Resolve a config name or path.

    Precedence: bundled preset name → literal path that exists → ~/.caesar/configs/<name>.yaml.
    Bundled wins, so a user can't shadow a preset by name (use a different name to override)."""
    if config_path in CONFIG_PRESETS:
        return str(Path(__file__).resolve().parent / "config" / "config_preset" / f"{config_path}.yaml")
    literal = Path(config_path).expanduser()
    if literal.exists():
        return str(literal)
    user_cfg = USER_CONFIG_DIR / (config_path if config_path.endswith(('.yaml', '.yml')) else f"{config_path}.yaml")
    if user_cfg.exists():
        return str(user_cfg)
    return config_path


def resolve_experiment_repository(query=None, max_iterations=None, exp_id=None, config_name=None):
    """Generate an experiment directory path from query/iterations.

    Returns: <result-root>/<date>_<config>_Q-<hash>_T-<iters>[_ID-<id>]
    Source checkouts write to caesar/result/ (historical). Pip installs write to
    ~/.caesar/result/. Override either with CAESAR_RESULT_DIR.
    exp_id is included in batch mode to avoid collisions between experiments."""
    date_str = datetime.now().strftime("%m-%d-%y")
    query = query or ''
    query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
    cfg = config_name or 'unknown'
    dir_name = f"{date_str}_{cfg}_q-{query_hash}_t-{max_iterations}"
    if exp_id is not None:
        dir_name += f"_id-{exp_id}"
    return str(default_result_root() / dir_name)


def get_version() -> str:
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version("caesar-agent")
        except PackageNotFoundError:
            return "dev"
    except ImportError:
        return "dev"


LLM_KEY_ENV_VARS = (
    ("OPENAI_API_KEY", "OpenAI", "https://platform.openai.com/api-keys"),
    ("ANTHROPIC_API_KEY", "Anthropic", "https://console.anthropic.com/settings/keys"),
    ("GOOGLE_API_KEY", "Gemini", "https://aistudio.google.com/apikey"),
    ("GEMINI_API_KEY", "Gemini", "https://aistudio.google.com/apikey"),
)


def _build_missing_key_message() -> str:
    lines = ["No LLM provider API key found. Set at least one of:", ""]
    seen = set()
    for env_name, provider, url in LLM_KEY_ENV_VARS:
        if provider in seen:
            continue
        seen.add(provider)
        lines.append(f"  {env_name:<22}  # {provider} — get a key at {url}")
    lines.append("")
    lines.append("Example:  export OPENAI_API_KEY=sk-...")
    return "\n".join(lines)


# Precomputed help string. Separating the message constant from the env-var
# read in has_any_llm_key keeps the printed output free of taint from
# os.environ, which silences CodeQL's clear-text-logging warning.
MISSING_KEY_MESSAGE = _build_missing_key_message()


def has_any_llm_key() -> bool:
    return any(os.environ.get(name) for name, _, _ in LLM_KEY_ENV_VARS)


def validate_repository(repo_path: str, logger) -> str:
    """Validate and return absolute repository path, creating if needed."""
    path = Path(repo_path).resolve()

    if not path.exists():
        logger.info(f"Creating repository directory: {path}")
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create repository: {e}")
            sys.exit(1)
    elif not path.is_dir():
        logger.error(f"Path exists but is not a directory: {path}")
        sys.exit(1)

    return str(path)


# =============================================================================
# CLI argument parsing
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run CaesarAgent for web exploration and insight synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Single experiment:
  python run_agent.py ./result/my_exp config.yaml             # explicit output directory
  python run_agent.py config.yaml                              # auto-names: result/MMDD_<hash>_<iters>
  python run_agent.py mini -q "my query"                       # use preset config (regular, mini, nano)
  python run_agent.py config.yaml -q "my query" --max-iterations 50

Batch mode (runs experiments as parallel subprocesses):
  python run_agent.py -b experiments.jsonl                     # 10 workers (default)
  python run_agent.py -b experiments.jsonl -n 4                # 4 workers

Batch management (requires -b to identify the batch):
  python run_agent.py -b experiments.jsonl --status            # show status of all experiments
  python run_agent.py -b experiments.jsonl --stop 3            # stop experiment 3
  python run_agent.py -b experiments.jsonl --restart 3         # restart experiment 3 (resumes from checkpoint)

JSONL format (one JSON object per line, "config" is required):
  {"config": "config.yaml", "query": "topic A", "max_iterations": 50, "repository": "result/exp1"}
  {"config": "config.yaml", "query": "topic B"}
        """
    )

    parser.add_argument('positional', type=str, nargs='*',
                       help='[repository] config — repository is auto-generated under result/ if omitted')
    parser.add_argument('--max-iterations', type=int, default=None,
                       help='Override max_iterations from config (default: use config value)')
    parser.add_argument('-q', '--query', type=str, default=None,
                       help='Override starting_query from config')
    parser.add_argument('-b', '--batch', type=str, default=None,
                       help='Path to JSONL file for batch experiments')
    parser.add_argument('-n', '--num-workers', type=int, default=10,
                       help='Number of parallel workers for batch mode (default: 10, minimum: 1)')
    parser.add_argument('--stop', type=int, default=None, metavar='ID',
                       help='Stop a running batch experiment by ID (requires -b)')
    parser.add_argument('--restart', type=int, default=None, metavar='ID',
                       help='Restart a stopped/failed batch experiment by ID (requires -b)')
    parser.add_argument('--status', action='store_true',
                       help='Show status of all batch experiments (requires -b)')
    parser.add_argument('--generate-image', type=int, default=0, metavar='N',
                       help='Generate N standalone images into <run_dir>/images/ after '
                            'a successful run. Artifact text is NOT modified. For inline '
                            'embed, use synthesis_generate_images (preset YAML) or run '
                            '`python -m caesar.image_generator <run_dir>` post-hoc. '
                            'Default 0 skips.')
    parser.add_argument('--image-model', type=str, default=None,
                       help='Image-gen model for --generate-image (default: gpt-image-1)')
    parser.add_argument('--image-refs', type=int, default=None,
                       help='Refs per image for --generate-image (default: 4)')
    parser.add_argument('--version', action='version',
                       version=f'caesar-agent {get_version()}')

    args = parser.parse_args()

    if args.generate_image < 0:
        parser.error("--generate-image must be >= 0")

    if args.batch:
        # Batch mode takes priority — ignore positional arguments
        args.repository = None
        args.config = None
        if args.num_workers < 1:
            parser.error("-n/--num-workers must be at least 1")
    elif args.stop is not None or args.restart is not None or args.status:
        parser.error("--stop, --restart, and --status require -b")
    elif len(args.positional) == 1:
        args.repository = None
        args.config = args.positional[0]
    elif len(args.positional) == 2:
        args.repository = args.positional[0]
        args.config = args.positional[1]
    else:
        parser.error("Expected 1 or 2 positional arguments: [repository] config")

    return args


# =============================================================================
# Display helpers
# =============================================================================

def build_config_summary(agent):
    """Build configuration summary dict (used for display and JSON export)."""
    memory = agent.agent_memory
    synth = agent.synthesizer
    return {
        "Agent Configuration": {
            "Name": agent.name,
            "Repository": agent.repository,
            "Max Iterations": agent.max_iterations,
        },
        "Caesar Settings": {
            "Starting URL": agent.starting_url,
            "Starting Query": agent.starting_query,
            "Additional Queries": agent.additional_starting_queries,
            "Allowed Domains": ', '.join(agent.allowed_domains) if not agent.allow_all_domains else "* (ALL DOMAINS)",
            "Blocked Domains": ', '.join(agent.blocked_domains) if (agent.allow_all_domains and agent.blocked_domains) else "(none)",
            "Max Web Searches": agent.max_web_searches,
            "Max Depth": agent.max_depth,
            "Role Customized": agent.adapt_role or bool(agent.overwrite_role_file),
            "Checkpoint Interval": agent.checkpoint_interval,
            "Exploration LLM": f"{agent.exploration_llm_config['model']} (temp={agent.exploration_llm_config['temperature']})",
        },
        "Memory": {
            "Enabled": memory.enabled,
            "Type": "Vector & Graph DB" if memory.use_graph else "Vector DB",
        },
        "Synthesis": {
            "Mode": "Classic" if synth.synthesis_classic_mode else "Iterative",
            "Drafts": synth.synthesis_drafts,
            "Iterations/Draft": synth.synthesis_iterations,
            "Max Length (Words)": synth.synthesis_max_length,
            "Max ELI5 Length (Words)": synth.synthesis_eli5_length,
        },
        "Usage": {
            "Model": agent.llm_handler.model,
            "Cost Limit": f"${agent.llm_handler.cost_limit:.2f}" if agent.llm_handler.cost_limit else "None",
        }
    }


def print_config_summary(agent, logger):
    """Print configuration summary before starting."""
    summary = build_config_summary(agent)
    logger.info("="*80)
    logger.info("CAESAR AGENT CONFIGURATION")
    logger.info("="*80)
    for line in format_yaml_like(summary):
        logger.info(line)
    logger.info("="*80)


def write_experiment_summary(agent, artifact, start_time, logger):
    """Write a JSON summary of the experiment to the repo directory."""
    repo = Path(agent.get_repo())
    exp_id = agent.get_id()
    wall_time = time.time() - start_time

    # Aggregate token usage from llm_handler.cost_history
    history = getattr(agent.llm_handler, 'cost_history', []) or []
    input_tokens = sum(h.get('input_tokens', 0) for h in history)
    output_tokens = sum(h.get('output_tokens', 0) for h in history)

    artifact_info = artifact if isinstance(artifact, dict) else {}

    summary = {
        "timestamp": datetime.now().isoformat(),
        "wall_time": round(wall_time, 2),
        "tokens_used": input_tokens + output_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "token_cost": round(agent.llm_handler.accumulated_cost, 6),
        "api_calls": agent.llm_handler.call_count,
        "iterations_elapsed": agent.current_iteration,
        "webpages_visited": len(agent.visited_urls),
        "artifact_dir": artifact_info.get("artifact_dir"),
        "num_drafts": artifact_info.get("num_drafts"),
        "config_summary": build_config_summary(agent),
    }

    log_dir = repo / LOG_DIR_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{exp_id}.exp_summary.json"
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Experiment summary saved to: {path}")
    except Exception as e:
        logger.error(f"Failed to write experiment summary: {e}")


def print_final_summary(agent, artifact, logger):
    """Print exploration completion summary."""
    cost = agent.llm_handler.get_cost_summary()

    logger.info("\n" + "="*80)
    logger.info("EXPLORATION COMPLETE - FINAL SYNTHESIS")
    logger.info("="*80)
    logger.info(artifact)
    logger.info("="*80)

    logger.info("\nAPI COST SUMMARY:")
    logger.info(f"  Total Cost:    ${cost['accumulated_cost']:.4f}")
    logger.info(f"  API Calls:     {cost['call_count']}")
    logger.info(f"  Avg per Call:  ${cost['average_cost_per_call']:.4f}")
    if cost.get('cost_limit'):
        remaining = cost.get('remaining_budget', 0)
        usage_pct = (cost['accumulated_cost'] / cost['cost_limit']) * 100
        logger.info(f"  Budget Used:   {usage_pct:.1f}% (${remaining:.2f} remaining)")

    logger.info(f"\nLogs saved to: {agent.get_log_dir()}")
    logger.info("Exploration completed successfully!")


# =============================================================================
# Single experiment
# =============================================================================

def run_single(config_path, logger, repository=None, query=None, max_iterations=None,
               repository_out=None):
    """Run a single experiment. Returns 0 on success, 1 on error, 130 on interrupt.

    If repository_out is a list, the resolved repository path is appended to it on
    successful resolution (before exploration starts) so callers can post-process
    the run dir even if exploration is interrupted."""
    agent = None
    try:
        # Load configuration
        config_path = resolve_config_path(config_path)
        config = load_config(config_path)
        logger.info(f"Configuration loaded: {config_path}")
        logger.info(pprint.pformat(config))

        # Apply overrides
        if max_iterations is not None:
            if 'CaesarAgent' not in config:
                config['CaesarAgent'] = {}
            config['CaesarAgent']['max_iterations'] = max_iterations
            logger.info(f"Overriding max_iterations: {max_iterations}")

        if query is not None:
            if 'CaesarAgent' not in config:
                config['CaesarAgent'] = {}
            config['CaesarAgent']['starting_query'] = query
            logger.info(f"Overriding starting_query: {query}")

        # Resolve repository path (auto-generate if not provided)
        if repository is not None:
            repository = validate_repository(repository, logger)
        else:
            caesar_cfg = config.get('CaesarAgent', {})
            repository = validate_repository(
                resolve_experiment_repository(
                    query=caesar_cfg.get('starting_query'),
                    max_iterations=caesar_cfg.get('max_iterations'),
                    config_name=Path(config_path).stem,
                ), logger
            )
        if isinstance(repository_out, list):
            repository_out.append(repository)

        # Get agent name from config
        agent_name = config.get('Agent', {}).get('name', 'CaesarAgent')

        # Initialize agent
        logger.info("Initializing CaesarAgent...")
        agent = CaesarAgent(name=agent_name, repository=repository, config=config)

        # Print configuration
        print_config_summary(agent, logger)

        # Run exploration
        logger.info("Starting exploration...\n")
        start_time = time.time()
        artifact = agent.explore()

        # Print final summary + persist JSON summary
        print_final_summary(agent, artifact, logger)
        write_experiment_summary(agent, artifact, start_time, logger)

        agent.shutdown()
        return 0

    except KeyboardInterrupt:
        logger.error("\nExploration interrupted by user")
        if agent is not None:
            try:
                agent.shutdown()
            except Exception:
                pass
        return 130

    except (Exception, FatalLLMError) as e:
        # FatalLLMError is BaseException-derived (so intermediate `except
        # Exception` retry layers can't swallow auth/quota errors), which means
        # this top-level handler must catch it explicitly to exit cleanly with
        # an error message rather than crash with an unhandled traceback.
        logger.error(f"\nAgent stopped due to error: {e}")
        traceback.print_exc()
        if agent is not None:
            try:
                agent.shutdown()
            except Exception:
                pass
        return 1


# =============================================================================
# Batch status file helpers
# =============================================================================

def get_status_path(batch_path):
    """Derive status file path from batch JSONL path."""
    return str(Path(batch_path).with_suffix('.status.json'))


def load_status(status_path):
    """Load batch status from file. Returns empty dict if missing or corrupt."""
    if os.path.exists(status_path):
        try:
            with open(status_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def save_status(status_path, status_data):
    """Save batch status to file atomically."""
    tmp_path = status_path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(status_data, f, indent=2)
    os.replace(tmp_path, status_path)


# =============================================================================
# Batch helpers
# =============================================================================

def parse_batch_file(batch_path, logger):
    """Parse JSONL batch file. Returns list of entries or None on error."""
    experiments = []
    with open(batch_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON on line {line_num}: {e}")
                return None
            if 'config' not in entry:
                logger.error(f"Missing required 'config' field on line {line_num}")
                return None
            experiments.append(entry)
    return experiments


def build_experiment_cmd(entry):
    """Build the run_agent.py command for a single experiment subprocess."""
    cmd = [sys.executable, os.path.abspath(__file__)]
    # repository is always present — resolved at status init time
    cmd.append(entry['repository'])
    cmd.append(entry['config'])
    if entry.get('query'):
        cmd.extend(['-q', entry['query']])
    if entry.get('max_iterations') is not None:
        cmd.extend(['--max-iterations', str(entry['max_iterations'])])
    return cmd


# =============================================================================
# Batch execution and management
# =============================================================================

def run_batch(batch_path, num_workers, logger):
    """Run batch experiments as parallel subprocesses. Returns number of failures."""
    batch_file = Path(batch_path)
    if not batch_file.is_file():
        logger.error(f"Batch file not found: {batch_path}")
        return 1

    experiments = parse_batch_file(batch_file, logger)
    if experiments is None:
        return 1

    total = len(experiments)
    status_path = get_status_path(batch_path)
    status = load_status(status_path)

    # Initialize status for new or stale (interrupted while running) experiments
    for i, entry in enumerate(experiments, 1):
        exp_id = str(i)
        existing = status.get(exp_id, {}).get('status')
        if existing in ('completed', 'failed', 'stopped'):
            continue
        if exp_id in status:
            # Entry exists (stale running or pending from --restart) — keep entry, reset status
            status[exp_id]['status'] = 'pending'
            status[exp_id]['pid'] = None
        else:
            # New experiment — resolve repository once so restarts use the same directory
            entry = dict(entry)
            entry['repository'] = entry.get('repository') or resolve_experiment_repository(
                query=entry.get('query'),
                max_iterations=entry.get('max_iterations'),
                exp_id=exp_id,
                config_name=Path(entry['config']).stem,
            )
            status[exp_id] = {'status': 'pending', 'pid': None, 'entry': entry}
    save_status(status_path, status)

    pending = [str(i) for i in range(1, total + 1) if status[str(i)]['status'] == 'pending']
    skipped = total - len(pending)
    if skipped:
        logger.info(f"Skipping {skipped} already completed/failed/stopped experiments")

    effective_workers = min(num_workers, len(pending)) if pending else 0
    logger.info(f"Batch: {len(pending)} pending, {total} total ({effective_workers} workers)")

    running = {}  # exp_id -> Popen

    try:
        while pending or running:
            # Poll running processes
            for exp_id in list(running):
                ret = running[exp_id].poll()
                if ret is None:
                    continue
                del running[exp_id]
                if ret == 0:
                    status[exp_id]['status'] = 'completed'
                    logger.info(f"Experiment {exp_id}/{total} completed")
                elif ret in (-signal.SIGTERM, -signal.SIGKILL):
                    status[exp_id]['status'] = 'stopped'
                    logger.info(f"Experiment {exp_id}/{total} stopped")
                else:
                    status[exp_id]['status'] = 'failed'
                    logger.error(f"Experiment {exp_id}/{total} failed (exit code {ret})")
                status[exp_id]['pid'] = None
                save_status(status_path, status)

            # Start new processes up to worker limit
            while pending and len(running) < effective_workers:
                exp_id = pending.pop(0)
                entry = status[exp_id]['entry']
                cmd = build_experiment_cmd(entry)
                try:
                    proc = subprocess.Popen(cmd)
                except OSError as e:
                    logger.error(f"Failed to start experiment {exp_id}/{total}: {e}")
                    status[exp_id]['status'] = 'failed'
                    status[exp_id]['pid'] = None
                    save_status(status_path, status)
                    continue
                running[exp_id] = proc
                status[exp_id]['status'] = 'running'
                status[exp_id]['pid'] = proc.pid
                save_status(status_path, status)
                logger.info(f"Started experiment {exp_id}/{total} (PID {proc.pid})")

            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("\nInterrupted — stopping all running experiments...")
        for exp_id, proc in running.items():
            proc.terminate()
            status[exp_id]['status'] = 'stopped'
            status[exp_id]['pid'] = None
        for proc in running.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        save_status(status_path, status)
        logger.info(f"Stopped {len(running)} experiments. Resume with: python run_agent.py -b {batch_path}")
        return -1  # signal interrupted

    # Final summary
    completed = sum(1 for s in status.values() if s['status'] == 'completed')
    failed = sum(1 for s in status.values() if s['status'] == 'failed')
    stopped = sum(1 for s in status.values() if s['status'] == 'stopped')

    logger.info("\n" + "#"*80)
    logger.info(f"BATCH COMPLETE: {completed} completed, {failed} failed, {stopped} stopped (of {total})")
    logger.info("#"*80)
    return failed  # 0 = all succeeded, >0 = number of failures


def batch_stop(batch_path, experiment_id, logger):
    """Stop a running batch experiment by ID."""
    status_path = get_status_path(batch_path)
    status = load_status(status_path)

    exp_id = str(experiment_id)
    if exp_id not in status:
        logger.error(f"Experiment {exp_id} not found")
        return 1

    exp = status[exp_id]
    if exp['status'] != 'running':
        logger.error(f"Experiment {exp_id} is not running (status: {exp['status']})")
        return 1

    pid = exp.get('pid')
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            logger.info(f"Sent SIGTERM to experiment {exp_id} (PID {pid})")
        except ProcessLookupError:
            logger.info(f"Process {pid} already exited")

    status[exp_id]['status'] = 'stopped'
    status[exp_id]['pid'] = None
    save_status(status_path, status)
    return 0


def batch_restart(batch_path, experiment_id, logger):
    """Mark a stopped/failed experiment for restart."""
    status_path = get_status_path(batch_path)
    status = load_status(status_path)

    exp_id = str(experiment_id)
    if exp_id not in status:
        logger.error(f"Experiment {exp_id} not found")
        return 1

    exp = status[exp_id]
    if exp['status'] == 'running':
        logger.error(f"Experiment {exp_id} is still running (PID {exp.get('pid')})")
        return 1
    if exp['status'] == 'pending':
        logger.info(f"Experiment {exp_id} is already pending")
        return 0

    status[exp_id]['status'] = 'pending'
    status[exp_id]['pid'] = None
    save_status(status_path, status)
    logger.info(f"Experiment {exp_id} marked for restart (will resume from checkpoint)")
    logger.info(f"Run: python run_agent.py -b {batch_path}")
    return 0


def batch_status(batch_path, logger):
    """Show status of all batch experiments."""
    status_path = get_status_path(batch_path)
    status = load_status(status_path)

    if not status:
        logger.info("No batch status found. Run a batch first.")
        return 0

    logger.info(f"\nBatch: {batch_path}")
    logger.info(f"{'ID':>4}  {'Status':<12}  {'PID':<8}  {'Config':<30}  {'Query'}")
    logger.info("-" * 80)
    for exp_id in sorted(status.keys(), key=int):
        s = status[exp_id]
        entry = s.get('entry', {})
        pid_str = str(s['pid']) if s.get('pid') else '-'
        config = entry.get('config', '-')
        query = entry.get('query') or '-'
        logger.info(f"{exp_id:>4}  {s['status']:<12}  {pid_str:<8}  {config:<30}  {query}")

    counts = {}
    for s in status.values():
        counts[s['status']] = counts.get(s['status'], 0) + 1
    summary = ', '.join(f"{v} {k}" for k, v in sorted(counts.items()))
    logger.info(f"\nTotal: {len(status)} ({summary})")
    return 0


# =============================================================================
# Entry point
# =============================================================================

def main():
    """Main execution function."""
    # Configure logger
    logger = get_logger()
    logger.configure({
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "console": True
    })

    try:
        args = parse_args()

        # Friendly fail-fast if no LLM provider key is configured. argparse runs
        # before this for --version/--help so those still work.
        if not has_any_llm_key():
            print(MISSING_KEY_MESSAGE, file=sys.stderr)
            return 1

        if args.batch:
            if args.status:
                return batch_status(args.batch, logger)
            if args.stop is not None:
                return batch_stop(args.batch, args.stop, logger)
            if args.restart is not None:
                return batch_restart(args.batch, args.restart, logger)
            result = run_batch(args.batch, args.num_workers, logger)
            if result < 0:
                return 130  # interrupted
            return 1 if result else 0

        repo_holder: list = []
        rc = run_single(
            config_path=args.config,
            logger=logger,
            repository=args.repository,
            query=args.query,
            max_iterations=args.max_iterations,
            repository_out=repo_holder,
        )

        # Post-process: image generation. Opt-in; failure does not flip the run's rc
        # (the exploration succeeded; image-gen is a separate side step).
        if rc == 0 and args.generate_image > 0 and repo_holder:
            try:
                from caesar.image_generator import generate_image_from_run
                kwargs = {}
                if args.image_model:
                    kwargs["image_model"] = args.image_model
                if args.image_refs is not None:
                    kwargs["refs_per_image"] = args.image_refs
                logger.info(f"Post-processing: generating {args.generate_image} "
                            f"image(s) from artifact...")
                generate_image_from_run(repo_holder[0], n=args.generate_image, **kwargs)
            except Exception as e:
                logger.error(f"Image generation failed (run itself succeeded): {e}")

        return rc

    except KeyboardInterrupt:
        logger.error("\nInterrupted by user")
        return 130


if __name__ == '__main__':
    sys.exit(main())
