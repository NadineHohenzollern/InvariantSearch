"""Reporting for IVSN invariance experiments."""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from .runtime import ensure_dir


def summarize_subset(rows: List[dict]) -> dict:
    if not rows:
        return {'n_trials': 0, 'mean_fixations': None, 'std_fixations': None, 'accuracy_by_n_fixations': None, 'found_within_3_fixations_rate': None, 'found_within_3_fixations_count': 0, 'not_found_within_3_fixations_count': 0, 'mean_score_target': None, 'mean_score_max_distractor': None, 'mean_score_mean_distractor': None, 'mean_score_margin': None, 'std_score_margin': None, 'top1_rate': None, 'mean_target_rank': None, 'mean_p_target': None}
    fix = np.array([r['n_fixations'] for r in rows], dtype=np.float32)
    found = np.array([r['found'] for r in rows], dtype=np.float32)
    found3 = np.array([r['found_within_3_fixations'] for r in rows], dtype=np.float32)
    score_target = np.array([r['score_target'] for r in rows], dtype=np.float32)
    score_max_distractor = np.array([r['score_max_distractor'] for r in rows], dtype=np.float32)
    score_mean_distractor = np.array([r['score_mean_distractor'] for r in rows], dtype=np.float32)
    score_margin = np.array([r['score_margin'] for r in rows], dtype=np.float32)
    is_top1 = np.array([r['is_top1'] for r in rows], dtype=np.float32)
    target_rank = np.array([r['target_rank'] for r in rows], dtype=np.float32)
    p_target = np.array([r['p_target'] for r in rows], dtype=np.float32)
    return {'n_trials': int(len(rows)), 'mean_fixations': float(fix.mean()), 'std_fixations': float(fix.std(ddof=0)), 'accuracy_by_n_fixations': float(found.mean()), 'found_within_3_fixations_rate': float(found3.mean()), 'found_within_3_fixations_count': int(found3.sum()), 'not_found_within_3_fixations_count': int(len(rows) - found3.sum()), 'mean_score_target': float(score_target.mean()), 'mean_score_max_distractor': float(score_max_distractor.mean()), 'mean_score_mean_distractor': float(score_mean_distractor.mean()), 'mean_score_margin': float(score_margin.mean()), 'std_score_margin': float(score_margin.std(ddof=0)), 'top1_rate': float(is_top1.mean()), 'mean_target_rank': float(target_rank.mean()), 'mean_p_target': float(p_target.mean())}


def build_grouped_summary(results: List[dict], n_objects: int) -> List[dict]:
    grouped = []
    by_condition = {}
    for r in results:
        by_condition.setdefault(r['condition_name'], []).append(r)
    for cond_name, rows in sorted(by_condition.items(), key=lambda kv: kv[0]):
        overall = summarize_subset(rows)
        diff = summarize_subset([r for r in rows if r['trial_type'] == 'different'])
        ident = summarize_subset([r for r in rows if r['trial_type'] == 'identical'])
        grouped.append({'condition_name': cond_name, 'condition_group': rows[0]['condition_group'], 'condition_value': rows[0]['condition_value'], 'n_trials': overall['n_trials'], 'n_objects': n_objects, 'mean_fixations_all': overall['mean_fixations'], 'std_fixations_all': overall['std_fixations'], f'accuracy_by_{n_objects}_fixations_all': overall['accuracy_by_n_fixations'], 'found_within_3_fixations_rate_all': overall['found_within_3_fixations_rate'], 'found_within_3_fixations_count_all': overall['found_within_3_fixations_count'], 'not_found_within_3_fixations_count_all': overall['not_found_within_3_fixations_count'], 'mean_fixations_target_different': diff['mean_fixations'], 'std_fixations_target_different': diff['std_fixations'], 'accuracy_target_different': diff['accuracy_by_n_fixations'], 'found_within_3_fixations_rate_target_different': diff['found_within_3_fixations_rate'], 'found_within_3_fixations_count_target_different': diff['found_within_3_fixations_count'], 'not_found_within_3_fixations_count_target_different': diff['not_found_within_3_fixations_count'], 'mean_fixations_target_identical': ident['mean_fixations'], 'std_fixations_target_identical': ident['std_fixations'], 'accuracy_target_identical': ident['accuracy_by_n_fixations'], 'found_within_3_fixations_rate_target_identical': ident['found_within_3_fixations_rate'], 'found_within_3_fixations_count_target_identical': ident['found_within_3_fixations_count'], 'not_found_within_3_fixations_count_target_identical': ident['not_found_within_3_fixations_count'], 'mean_score_target_all': overall['mean_score_target'], 'mean_score_max_distractor_all': overall['mean_score_max_distractor'], 'mean_score_mean_distractor_all': overall['mean_score_mean_distractor'], 'mean_score_margin_all': overall['mean_score_margin'], 'std_score_margin_all': overall['std_score_margin'], 'mean_target_rank_all': overall['mean_target_rank'], 'top1_rate_all': overall['top1_rate'], 'mean_p_target_all': overall['mean_p_target'], 'mean_score_target_target_different': diff['mean_score_target'], 'mean_score_max_distractor_target_different': diff['mean_score_max_distractor'], 'mean_score_mean_distractor_target_different': diff['mean_score_mean_distractor'], 'mean_score_margin_target_different': diff['mean_score_margin'], 'std_score_margin_target_different': diff['std_score_margin'], 'mean_target_rank_target_different': diff['mean_target_rank'], 'top1_rate_target_different': diff['top1_rate'], 'mean_p_target_target_different': diff['mean_p_target'], 'mean_score_target_target_identical': ident['mean_score_target'], 'mean_score_max_distractor_target_identical': ident['mean_score_max_distractor'], 'mean_score_mean_distractor_target_identical': ident['mean_score_mean_distractor'], 'mean_score_margin_target_identical': ident['mean_score_margin'], 'std_score_margin_target_identical': ident['std_score_margin'], 'mean_target_rank_target_identical': ident['mean_target_rank'], 'top1_rate_target_identical': ident['top1_rate'], 'mean_p_target_target_identical': ident['mean_p_target']})
    grouped.sort(key=lambda x: (str(x['condition_group']), float(x['condition_value'])))
    return grouped


def write_trial_csv(results: List[dict], out_path: Path):
    fieldnames = ['condition_name', 'condition_group', 'condition_value', 'unique_id', 'repeat_id', 'trial_type', 'target_category', 'target_position', 'n_fixations', 'found', 'found_within_3_fixations', 'score_target', 'score_max_distractor', 'score_mean_distractor', 'score_margin', 'target_rank', 'is_top1', 'p_target', 'cue_path', 'target_path', 'distractor_paths', 'cue_transform', 'target_transform', 'distractor_transforms', 'fixation_positions', 'fixation_centers', 'scores_initial', 'score_history']
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            row = dict(r)
            for k in ['distractor_paths', 'cue_transform', 'target_transform', 'distractor_transforms', 'fixation_positions', 'fixation_centers', 'scores_initial', 'score_history']:
                row[k] = json.dumps(row[k])
            writer.writerow(row)


def write_grouped_summary_csv(grouped: List[dict], out_path: Path):
    if not grouped:
        return
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(grouped[0].keys()), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(grouped)


def pick_x_label(condition_group: str) -> str:
    mapping = {'rotation_deg': 'Rotation (degrees)', 'scale': 'Scale factor', 'shift_x': 'Horizontal shift (px)', 'shift_y': 'Vertical shift (px)', 'skew_x_deg': 'Horizontal skew (degrees)', 'skew_y_deg': 'Vertical skew (degrees)', 'noise_std': 'Gaussian noise std', 'blur_radius': 'Blur radius', 'mixed': 'Condition index'}
    return mapping.get(condition_group, 'Condition value')


def save_grouped_bar_plot(df, x_col: str, y_cols: list, labels: list, title: str, x_label: str, y_label: str, out_path: Path, y_lim: tuple=None):
    x_vals = df[x_col].tolist()
    n_groups = len(x_vals)
    n_series = len(y_cols)
    x = np.arange(n_groups)
    width = 0.24 if n_series == 3 else 0.35
    plt.figure(figsize=(8, 4.8))
    offsets = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * width
    for i, (y, label) in enumerate(zip(y_cols, labels)):
        plt.bar(x + offsets[i], df[y], width=width, label=label)
    plt.xticks(x, [f'{v:g}' for v in x_vals])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, axis='y', alpha=0.3)
    if n_series > 1:
        plt.legend()
    if y_lim is not None:
        plt.ylim(*y_lim)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_grouped_plots(grouped: List[dict], out_dir: Path, transform_mode: str):
    if not grouped:
        return
    plot_dir = out_dir / 'plots'
    ensure_dir(plot_dir)
    import pandas as pd
    df = pd.DataFrame(grouped).sort_values('condition_value')
    x_label = pick_x_label(str(df['condition_group'].iloc[0]))
    rank_ylim = (0, df['n_objects'].iloc[0])
    save_grouped_bar_plot(df, 'condition_value', ['mean_fixations_all', 'mean_fixations_target_different', 'mean_fixations_target_identical'], ['All', 'Target-different', 'Target-identical'], f'{transform_mode}: Mean fixations', x_label, 'Mean fixations', plot_dir / 'mean_fixations.png', y_lim=(0, df['n_objects'].iloc[0]))
    save_grouped_bar_plot(df, 'condition_value', ['found_within_3_fixations_rate_all', 'found_within_3_fixations_rate_target_different', 'found_within_3_fixations_rate_target_identical'], ['All', 'Target-different', 'Target-identical'], f'{transform_mode}: Found within 3 fixations', x_label, 'Rate', plot_dir / 'found_within_3_fixations_rate.png', y_lim=(0, 1))
    save_grouped_bar_plot(df, 'condition_value', ['top1_rate_all', 'top1_rate_target_different', 'top1_rate_target_identical'], ['All', 'Target-different', 'Target-identical'], f'{transform_mode}: Top-1 accuracy', x_label, 'Rate', plot_dir / 'top1_rate.png', y_lim=(0, 1))


def smoothing_suffix(args) -> str:
    if args.smoothing_mode == 'none':
        return 'unsmoothed'
    role_bits = []
    if args.smooth_target:
        role_bits.append('target')
    if args.smooth_cue:
        role_bits.append('cue')
    if args.smooth_distractors:
        role_bits.append('distractors')
    role_tag = '-'.join(role_bits) if role_bits else 'none'
    if args.smoothing_mode == 'alpha':
        return f'smoothed_alpha_{role_tag}_b{args.alpha_soften_blur_radius:g}'
    if args.smoothing_mode == 'cosine':
        return f'smoothed_cosine_{role_tag}_w{args.edge_taper_width_px:g}'
    raise ValueError(f'Unsupported smoothing mode: {args.smoothing_mode}')


def build_dynamic_out_dir(base_out_dir: Path, args) -> Path:
    if args.no_dynamic_out_dir:
        return base_out_dir
    suffix = smoothing_suffix(args)
    return base_out_dir / f'{base_out_dir.name}_{args.model_kind}_g{args.gist_image_size}_{suffix}'
