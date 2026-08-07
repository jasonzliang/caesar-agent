"""Transfer configuration functions for prepare_artifact.py"""
import glob
import json
import os

import networkx as nx

from prepare_artifact import find_synthesis, DEFAULT_CONFIG, CATEGORIES


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------

OTHER_SOURCES = {
    'full':     'full_answers/*/',
    'eli5':     'eli5_answers/*/',
    'eli5_450': 'eli5_450w_answers/*/',
}


def _transfer(caesar_sources, other_sources, caesar_filename=None):
    """Build a single transfer dict."""
    d = {'caesar_sources': caesar_sources, 'other_sources': other_sources}
    if caesar_filename:
        d['caesar_filename'] = caesar_filename
    return d


def _ablation_overrides(**extra):
    """Overrides for ablation configs (empty baseline + no clearing)."""
    return {
        "OTHER_AGENT_BASE_DIR": os.path.abspath("query_result/empty_agent_answers"),
        "CLEAR_OUTPUT_DIR": False, **extra}


def _build_simple(variants, overrides=None):
    """Build transfer config from (format_key, caesar_glob) pairs.

    format_key indexes into OTHER_SOURCES for the baseline comparison dir.
    """
    return (
        [_transfer(src, OTHER_SOURCES[fmt]) for fmt, src in variants],
        overrides or {},
    )


def _resolve_transfers(variants, categories=CATEGORIES, allow_missing=False):
    """Build transfer list by resolving variants across categories via find_synthesis.

    Args:
        variants: list of (exp_template, synthesis_id, filename, file_patterns) where:
            exp_template: experiment glob with {cat} placeholder for category
            synthesis_id: synthesis folder ID (or None for latest)
            filename: caesar_filename for output
            file_patterns: list of (other_sources, file_glob) tuples
        categories: category names to substitute into {cat}
        allow_missing: if True, skip FileNotFoundError instead of raising
    """
    transfer_list = []
    for exp_template, synthesis_id, filename, file_patterns in variants:
        for cat in categories:
            exp = exp_template.format(cat=cat)
            for other_sources, pattern in file_patterns:
                try:
                    source = find_synthesis(exp, pattern, synthesis_id=synthesis_id)
                except FileNotFoundError:
                    if allow_missing:
                        continue
                    raise
                transfer_list.append(_transfer(source, other_sources, filename))
    return transfer_list


# ---------------------------------------------------------------------------
# Simple configs: full / eli5 / eli5-sized variants
# ---------------------------------------------------------------------------

# Outdated, do not use
def setup_transfer_dict_11_17():
    return _build_simple([
        ('full',      '11_17_*/*12130155/*merged-3*'),
        ('eli5',      '11_17_*/*12130155/*merged-eli5-3*'),
        ('eli5_600', '11_17_*/*12130327/*merged-eli5-3*'),
    ])


def setup_transfer_dict_12_13():
    return _build_simple([
        ('full',      '12_13_*/*12160*/*merged-3*'),
        ('eli5',      '12_13_*/*12160*/*merged-eli5-3.1*'),
        ('eli5_450', '12_13_*/*12160*/*merged-eli5-3.450w.1*'),
    ])


# Outdated, do not use
def setup_transfer_dict_12_13_v2():
    return _build_simple([
        ('full',      '12_13_*/*12161*/*merged-3*'),
        ('eli5',      '12_13_*/*12161*/*merged-eli5-3.1*'),
        ('eli5_450', '12_13_*/*12161*/*merged-eli5-3.450w.1*'),
    ])


# ---------------------------------------------------------------------------
# Ablation configs
# ---------------------------------------------------------------------------

def setup_transfer_dict_12_13_syn_ablation():
    """Synthesis method ablation: syn1 vs syn3 vs merged, across full/eli5/eli5_450."""
    baselines = [OTHER_SOURCES['full'], OTHER_SOURCES['eli5'], OTHER_SOURCES['eli5_450']]
    # (label, full_pattern, eli5_pattern, eli5_450_pattern)
    specs = [
        ('syn1',  '*synthesis-1*', '*synth-eli5-1.1*',  '*synth-eli5-1.450w*'),
        ('syn3',  '*synthesis-3*', '*synth-eli5-3.1*',  '*synth-eli5-3.450w*'),
        ('merge', '*merged-3*',    '*merged-eli5-3.1*', '*merged-eli5-3.450w*'),
    ]
    variants = [
        ('12_13_{cat}', '12160', f'answer_cat_{label}.txt', list(zip(baselines, patterns)))
        for label, *patterns in specs
    ]
    return _resolve_transfers(variants), _ablation_overrides()


def setup_transfer_dict_12_13_iter_ablation():
    """Iteration count ablation: 250 vs 500 vs 1000, across full/eli5/eli5_450."""
    file_patterns = [
        (OTHER_SOURCES['full'],     '*merged-3*'),
        (OTHER_SOURCES['eli5'],     '*merged-eli5-3.0*'),
        (OTHER_SOURCES['eli5_450'], '*merged-eli5-3.450w*'),
    ]
    # (label, synthesis_id)
    variants = [
        ('12_13_{cat}', '01072',    'answer_cat_250.txt',  file_patterns),
        ('12_13_{cat}', '01071',    'answer_cat_500.txt',  file_patterns),
        ('12_13_{cat}', '01110646', 'answer_cat_1000.txt', file_patterns),
    ]
    return _resolve_transfers(variants), _ablation_overrides()


# ---------------------------------------------------------------------------
# Dynamic graph ablation config
# ---------------------------------------------------------------------------

def _count_drafts(exp_glob, synthesis_id=None, base=None):
    """Count synthesis draft files in the latest (or specified) synthesis folder."""
    base = base or DEFAULT_CONFIG["CAESAR_AGENT_BASE_DIR"]
    if synthesis_id:
        sdirs = glob.glob(os.path.join(base, exp_glob, f"*{synthesis_id}*"))
    else:
        sdirs = sorted(glob.glob(os.path.join(base, exp_glob, "*.synthesis.*")))
    if not sdirs:
        return 0
    return len(glob.glob(os.path.join(sdirs[-1], "*synthesis-[0-9]*")))


def _has_merged(exp_glob, base=None):
    """Check if any synthesis folder for this experiment has merged files."""
    base = base or DEFAULT_CONFIG["CAESAR_AGENT_BASE_DIR"]
    return bool(glob.glob(os.path.join(base, exp_glob, "*.synthesis.*", "*merged*")))


def _variant_filename(variant):
    """Derive base filename from a variant tuple."""
    if len(variant) > 2 and variant[2]:
        return variant[2]
    name = variant[0].rsplit('/', 1)[-1].replace('{cat}', '').strip('_')
    while '__' in name:
        name = name.replace('__', '_')
    return f"answer_cat_{name}.txt"


def _draft_filename(base_filename, pattern):
    """Derive per-draft filename from base filename and a file pattern."""
    stem = base_filename.rsplit('.', 1)[0]
    if 'merged' in pattern:
        return f"{stem}_merge.txt"
    draft_num = pattern.strip('*').strip('-').replace('synthesis-', 'syn')
    return f"{stem}_{draft_num}.txt"


def _build_draft_patterns(num_drafts, has_merged):
    """Build file pattern list for all drafts and optionally merged."""
    patterns = [(OTHER_SOURCES['full'], f'*synthesis-{i}*') for i in range(1, num_drafts + 1)]
    if has_merged:
        patterns.append((OTHER_SOURCES['full'], f'*merged-{num_drafts}*'))
    return patterns


def setup_transfer_dict_compare_artifacts(
    categories=CATEGORIES,  # Query categories
    compare_all=False,      # Compare every artifact draft
    caesar_only=False):     # Compare against baseline agents

    raw_variants = [
        ('4_3_{cat}', '0427', 'answer_cat_cam.txt'),
        # ('3_28_{cat}',),
        # ('exp_03_2026/3_29_{cat}', '04031'),
        # ('4_1_{cat}', '04031', 'answer_cat_cam.txt'),
        # ('4_5_{cat}', '040406', 'answer_cat_cam.txt')
        # ('4_5_{cat}', '040408', 'answer_cat_cam.txt')
        # ('4_2_{cat}_qe',),
        # ('4_3_{cat}', '040311'),
        # ('4_3_{cat}', '04030'),
        # ('4_4_{cat}_qe', '04031'),
        # ('12_13_{cat}', '04030'),
        # ('12_13_{cat}', '12160'),
        # ('12_13_{cat}', '01110646', 'answer_cat_1000.txt'),
        # ('12_13_{cat}', '01072', 'answer_cat_250.txt'),
    ]
    file_patterns = [
        ('full_answers/*/',         '*merged-?.*'),
        ('eli5_answers/*/',         '*merged-eli5-?.[01]*'),
        ('eli5_450w_answers/*/',    '*merged-eli5-?.450w*'),
    ]

    if caesar_only:
        overrides = _ablation_overrides(OUTPUT_DIR_FROM_PATTERN=bool(compare_all))
    else:
        overrides = {"CLEAR_OUTPUT_DIR": False, "OUTPUT_DIR_FROM_PATTERN": bool(compare_all)}

    # Auto-detect drafts from the first variant
    num_drafts = 0
    all_draft_patterns = []
    if compare_all:
        first_exp = raw_variants[0][0].format(cat=categories[0])
        first_synth_id = raw_variants[0][1] if len(raw_variants[0]) > 1 else None
        num_drafts = _count_drafts(first_exp, first_synth_id)
        has_merged = _has_merged(first_exp)
        all_draft_patterns = _build_draft_patterns(num_drafts, has_merged)
        print(f"Auto-detected {num_drafts} synthesis drafts from {first_exp}")

    transfer_list = []
    for variant in raw_variants:
        exp_template = variant[0]
        synthesis_id = variant[1] if len(variant) > 1 else None
        base_filename = _variant_filename(variant)

        # Check if this variant's draft count matches for comparison
        use_comparison = compare_all and num_drafts > 0
        if use_comparison:
            check_exp = exp_template.format(cat=categories[0])
            variant_drafts = _count_drafts(check_exp, synthesis_id)
            if variant_drafts != num_drafts:
                print(f"Skipping comparison for {exp_template} ({variant_drafts} drafts != {num_drafts})")
                use_comparison = False

        if use_comparison:
            # Each draft pattern gets its own derived filename
            expanded = [
                (exp_template, synthesis_id, _draft_filename(base_filename, pat), [(other, pat)])
                for other, pat in all_draft_patterns
            ]
            transfer_list.extend(_resolve_transfers(expanded, categories, allow_missing=True))
        else:
            resolved = [(exp_template, synthesis_id, base_filename, file_patterns)]
            transfer_list.extend(_resolve_transfers(resolved, categories))

    return transfer_list, overrides


# ---------------------------------------------------------------------------
# Graph insight extraction
# ---------------------------------------------------------------------------

def setup_transfer_dict_compare_insights(base_dir=None, output_dir=None, exp_glob='4_3_*'):
    """Parse checkpoint graphs, find top/bottom pages by neighbor count, write to files.

    Top 10: pages with most neighbors (visit_count > 1) — high-connectivity hubs.
    Bottom 10: pages with fewest neighbors (visit_count = 1) — leaf/dead-end pages.
    """
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'result')
    if output_dir is None:
        output_dir = os.path.join(base_dir, 'hub_leaf_insights')
    os.makedirs(output_dir, exist_ok=True)

    experiment_dirs = sorted(glob.glob(os.path.join(base_dir, exp_glob)))
    transfer_list = []

    for exp_dir in experiment_dirs:
        exp_name = os.path.basename(exp_dir)
        checkpoint_path = os.path.join(exp_dir, '__rome__', 'agent_CaesarExplorer.checkpoint.json')
        if not os.path.exists(checkpoint_path):
            continue

        with open(checkpoint_path) as f:
            data = json.load(f)

        graph = nx.node_link_graph(data['graph'], edges='edges')

        # Build (url, neighbor_count, visit_count, insights) tuples
        node_stats = []
        for node in graph.nodes:
            visit_count = graph.nodes[node].get('visit_count', 1)
            neighbors = (set(graph.successors(node)) | set(graph.predecessors(node))) - {node}
            insights = graph.nodes[node].get('insights', '')
            node_stats.append((node, len(neighbors), visit_count, insights))

        # Top 10: most neighbors, visit_count > 1
        hub_pages = sorted(
            [s for s in node_stats if s[2] > 1 and s[3]],
            key=lambda x: x[1], reverse=True)[:10]

        # Bottom 10: fewest neighbors, visit_count = 1
        leaf_pages = sorted(
            [s for s in node_stats if s[2] == 1 and s[3]],
            key=lambda x: x[1])[:10]

        for _, pages, suffix in [('top10', hub_pages, 'hubs'), ('bottom10', leaf_pages, 'leaves')]:
            filename = f'{exp_name}_{suffix}.txt'
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w') as f:
                for _, _, _, insights in pages:
                    f.write(f"Insights: {insights or '(none)'}\n")
                    f.write("-" * 80 + "\n\n")

            transfer_list.append(_transfer(
                f'hub_leaf_insights/{filename}', OTHER_SOURCES['full'],
                caesar_filename=f'answer_cat_{suffix}.txt'))

    return transfer_list, {"CLEAR_OUTPUT_DIR": False}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TRANSFER_CONFIGS = {
    '11_17': setup_transfer_dict_11_17,
    '12_13': setup_transfer_dict_12_13,
    '12_13_v2': setup_transfer_dict_12_13_v2,
    '12_13_syn': setup_transfer_dict_12_13_syn_ablation,
    '12_13_iter': setup_transfer_dict_12_13_iter_ablation,
    'compare_artifacts': setup_transfer_dict_compare_artifacts,
    'compare_insights': setup_transfer_dict_compare_insights,
}
