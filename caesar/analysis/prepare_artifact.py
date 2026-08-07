"""Prepare Caesar artifacts for LLM as Judge"""
import glob
import os
import re
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default Configuration Dictionary
DEFAULT_CONFIG = {
    # Base directory filled with answers from Caesar agent
    "CAESAR_AGENT_BASE_DIR": os.path.abspath("result"),
    # Filename for Caesar agent answer in answer directory
    "CAESAR_AGENT_FILENAME": "answer_cat_cam.txt",
    # Base directory filled with answers from other agents
    "OTHER_AGENT_BASE_DIR": os.path.abspath("query_result/other_agent_answers"),

    # Clean up the output directory before answer file is copied
    "CLEAR_OUTPUT_DIR": True,
    # Strip ABSTRACT section from start and SOURCES section from end of artifact
    "STRIP_ABSTRACT_AND_SOURCES": True,

    # Base directory to output all agent answers
    "ALL_AGENT_BASE_DIR": os.path.abspath("query_result/all_agent_answers"),
    # Whether to determine output format directory from artifact draft names
    "OUTPUT_DIR_FROM_PATTERN": False
    }

CATEGORIES = ['constrained_creativity', 'counterfactual_reasoning',
    'crossdomain_synthesis', 'meta_creativity', 'openended_creativity']


def strip_abstract_and_sources(text):
    """Remove ABSTRACT section from start and SOURCES section from end."""
    # Strip abstract: everything before "ARTIFACT:" line
    artifact_match = re.search(r'^ARTIFACT:\s*\n', text, re.MULTILINE)
    if artifact_match:
        text = text[artifact_match.end():]
    # Strip sources: everything from "SOURCES:" line onward
    sources_match = re.search(r'\n\s*SOURCES:\s*\n', text)
    if sources_match:
        text = text[:sources_match.start()]
    return text.strip() + '\n'


def find_synthesis(experiment_glob, file_pattern="*merged-3*",
                   base_dir=None, synthesis_id=None):
    """Find file_pattern in a synthesis folder.

    Args:
        experiment_glob: Glob for experiment dirs (e.g. '03_28_*')
        file_pattern: Filename glob inside the synthesis folder
        base_dir: Override for CAESAR_AGENT_BASE_DIR
        synthesis_id: Specific synthesis folder ID (e.g. '01110646').
                      If None, uses the highest-numbered synthesis folder.

    Returns:
        Relative glob string usable as caesar_sources value.
    """
    base = base_dir or DEFAULT_CONFIG["CAESAR_AGENT_BASE_DIR"]

    if synthesis_id:
        synth_dirs = sorted(glob.glob(os.path.join(base, experiment_glob,
            f"*{synthesis_id}*")))
        if not synth_dirs:
            raise FileNotFoundError(
                f"No synthesis dirs matching {synthesis_id!r} for {experiment_glob!r} in {base}")
    else:
        synth_dirs = sorted(glob.glob(os.path.join(base, experiment_glob,
            "*.synthesis.*")))
        if not synth_dirs:
            raise FileNotFoundError(
                f"No synthesis dirs found for {experiment_glob!r} in {base}")
        def _synth_key(p):
            m = re.search(r'\.synthesis\.(\d+)', p)
            return int(m.group(1)) if m else 0
        synth_dirs.sort(key=_synth_key, reverse=True)

    for candidate in synth_dirs:
        rel = os.path.relpath(candidate, base)
        result = os.path.join(rel, file_pattern)
        matches = glob.glob(os.path.join(base, result))
        if matches:
            break
    else:
        raise FileNotFoundError(
            f"No synthesis dir with files matching {file_pattern!r} "
            f"for {experiment_glob!r} in {base}")
    print(f"Auto-resolved: {result}  ({len(matches)} file(s))")
    return result


def _match_other_source(caesar_file, other_source_dirs):
    """Find the other_source_dir whose category name appears in the caesar file path."""
    for d in other_source_dirs:
        if os.path.normpath(d).split(os.sep)[-1] in caesar_file:
            return d
    return None


def _resolve_output_dir(config, caesar_file, other_source_dir, td):
    """Determine output directory."""
    if other_source_dir:
        parts = os.path.normpath(other_source_dir).split(os.sep)
        category, parent_dir = parts[-1], parts[-2]
    else:
        category = next((c for c in CATEGORIES if c in caesar_file), 'unknown')
        parent_dir = td['other_sources'].split('/')[0]

    if config['OUTPUT_DIR_FROM_PATTERN']:
        draft_type = os.path.basename(td['caesar_sources']).strip('*').strip('.') or 'default'
        parent_dir = f"{draft_type}_answers"

    return os.path.join(config['ALL_AGENT_BASE_DIR'], parent_dir, category)


def _touch_files(directory, filenames):
    """Create or update timestamps on files in a directory."""
    os.makedirs(directory, exist_ok=True)
    for f in filenames:
        path = os.path.join(directory, f)
        with open(path, 'a'):
            os.utime(path, None)


def _ensure_query_file(output_dir):
    """Copy query.txt from empty_agent_answers if missing from output directory."""
    query_file = os.path.join(output_dir, 'query.txt')
    if not os.path.exists(query_file):
        category = os.path.basename(os.path.normpath(output_dir))
        src = os.path.join('query_result/empty_agent_answers/full_answers', category, 'query.txt')
        if os.path.exists(src):
            shutil.copy2(src, query_file)
            print(f'  Copied missing query.txt from {src}')


def prepare_artifact(transfer_func):
    """Copy Caesar artifacts to query answer directory for LLM judging."""
    transfer_dicts, config_overrides = transfer_func()
    config = {**DEFAULT_CONFIG, **(config_overrides or {})}
    os.makedirs(config['ALL_AGENT_BASE_DIR'], exist_ok=True)
    print(f"Active Output Dir: {config['ALL_AGENT_BASE_DIR']}")

    for td in transfer_dicts:
        other_source_dirs = glob.glob(os.path.join(
            config['OTHER_AGENT_BASE_DIR'], td['other_sources']))
        caesar_files = glob.glob(os.path.join(
            config['CAESAR_AGENT_BASE_DIR'], td['caesar_sources']))

        for caesar_file in caesar_files:
            other_dir = _match_other_source(caesar_file, other_source_dirs)
            print(f"Caesar answer: {caesar_file}")
            if other_dir:
                print(f"Matched: {other_dir}")
            else:
                print(f"No match, copying caesar answer only")

            output_dir = _resolve_output_dir(config, caesar_file, other_dir, td)
            meta_dir = os.path.dirname(os.path.normpath(output_dir))

            if config['CLEAR_OUTPUT_DIR'] and os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            if other_dir:
                shutil.copytree(other_dir, output_dir, dirs_exist_ok=True)
            else:
                os.makedirs(output_dir, exist_ok=True)

            _ensure_query_file(output_dir)
            caesar_filename = td.get("caesar_filename") or config['CAESAR_AGENT_FILENAME']
            dst = os.path.join(output_dir, caesar_filename)
            print(f'cp {caesar_file} {dst}\n')
            shutil.copy2(caesar_file, dst)

            if config['STRIP_ABSTRACT_AND_SOURCES']:
                with open(dst, 'r') as f:
                    original = f.read()
                stripped = strip_abstract_and_sources(original)
                if stripped != original:
                    with open(dst, 'w') as f:
                        f.write(stripped)
                    print(f'  Stripped abstract/sources from {caesar_filename}')


if __name__ == '__main__':
    from transfer_configs import TRANSFER_CONFIGS

    keys = list(TRANSFER_CONFIGS.keys())
    print("Select a transfer config:")
    for i, key in enumerate(keys):
        print(f"  [{i}] {key}")
    choice = input(f"\nEnter number (0-{len(keys)-1}): ")
    try:
        selected = keys[int(choice)]
    except (ValueError, IndexError):
        print("Invalid selection")
        sys.exit(1)
    print(f"Using: {selected}\n")
    prepare_artifact(TRANSFER_CONFIGS[selected])
