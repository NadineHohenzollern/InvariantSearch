"""Post-process existing target ERFs into inverse-aligned visualizations."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np

from .feature_reporting import write_rows_csv
from .target_feature_imaging import render_paired_targets
from .target_feature_reporting import (
    align_erf_to_original,
    aligned_erf_metrics,
    build_grouped_aligned_erf_summary,
    save_aligned_erf_comparison_figure,
    save_aligned_erf_summary_plots,
)
from .trials import make_condition_specs


def parse_args(argv: Sequence[str] = None):
    parser = argparse.ArgumentParser(
        description=(
            'Create inverse-aligned ERF comparison figures from an existing '
            'analyze_target_features.py result directory.'
        )
    )
    parser.add_argument('--result-dir', type=str, required=True)
    return parser.parse_args(argv)


def _read_csv(path: Path) -> list:
    with open(path, 'r', newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _existing_shape_lookup(path: Path) -> dict:
    lookup = {}
    for row in _read_csv(path):
        key = (
            str(row['condition_name']),
            int(row['unique_id']),
            int(row['layer']),
        )
        lookup[key] = (
            int(row['channels']),
            int(row['activation_height']),
            int(row['activation_width']),
        )
    return lookup


def main(argv: Sequence[str] = None) -> None:
    cli_args = parse_args(argv)
    result_dir = Path(cli_args.result_dir)
    with open(result_dir / 'config.json', 'r', encoding='utf-8') as handle:
        config = json.load(handle)
    args = SimpleNamespace(**config)
    layers = [int(layer) for layer in args.layers]
    selected_records = _read_csv(result_dir / 'selected_erf_targets.csv')
    shape_lookup = _existing_shape_lookup(result_dir / 'target_erf_metrics.csv')
    condition_specs = make_condition_specs(args)
    aligned_rows = []

    for condition_index, condition_spec in enumerate(condition_specs, start=1):
        condition_name, condition_group, condition_value, transformed_spec = (
            condition_spec
        )
        print(f'Aligning {condition_name}...')
        for record in selected_records:
            unique_id = int(record['unique_id'])
            target_path = Path(record['target_path'])
            category = str(record['target_category'])
            noise_seed = (
                int(args.seed)
                + 1_000_003 * unique_id
                + 10_007 * condition_index
            )
            original_image, transformed_image = render_paired_targets(
                target_path,
                transformed_spec,
                args,
                noise_seed=noise_seed,
            )
            original_erfs = {}
            transformed_erfs = {}
            aligned_erfs = {}
            layer_shapes = {}
            for layer in layers:
                stem = f'trial_{unique_id}_{target_path.stem}_layer_{layer}'
                npz_path = (
                    result_dir / 'erf_examples' / condition_name / category
                    / f'{stem}.npz'
                )
                with np.load(npz_path) as arrays:
                    original_erf = arrays['original_erf'].astype(np.float32)
                    transformed_erf = arrays['transformed_erf'].astype(np.float32)
                aligned_erf = align_erf_to_original(
                    transformed_erf,
                    transformed_spec,
                )
                original_erfs[layer] = original_erf
                transformed_erfs[layer] = transformed_erf
                aligned_erfs[layer] = aligned_erf
                layer_shapes[layer] = shape_lookup[
                    (condition_name, unique_id, layer)
                ]
                channels, height, width = layer_shapes[layer]
                aligned_rows.append({
                    'condition_name': condition_name,
                    'condition_group': condition_group,
                    'condition_value': condition_value,
                    'unique_id': unique_id,
                    'repeat_id': int(record['repeat_id']),
                    'trial_type': record['trial_type'],
                    'target_category': category,
                    'target_path': str(target_path),
                    'layer': layer,
                    'input_size': int(args.input_size),
                    'channels': channels,
                    'activation_height': height,
                    'activation_width': width,
                    'n_superimposed_cell_maps': height * width,
                    **aligned_erf_metrics(
                        original_erf,
                        transformed_erf,
                        aligned_erf,
                    ),
                })
            output_dir = (
                result_dir / 'erf_aligned_examples' / condition_name / category
            )
            save_aligned_erf_comparison_figure(
                original_image,
                transformed_image,
                original_erfs,
                transformed_erfs,
                aligned_erfs,
                layer_shapes,
                title=(
                    f'{condition_name} | {category} | '
                    'inverse-aligned ERF comparison'
                ),
                path=output_dir / f'trial_{unique_id}_{target_path.stem}.png',
            )

    grouped_rows = build_grouped_aligned_erf_summary(aligned_rows)
    write_rows_csv(
        aligned_rows,
        result_dir / 'aligned_target_erf_metrics.csv',
    )
    write_rows_csv(
        grouped_rows,
        result_dir / 'grouped_aligned_target_erf_metrics.csv',
    )
    save_aligned_erf_summary_plots(
        grouped_rows,
        result_dir / 'erf_alignment_plots',
    )
    print(f'Aligned ERF visualizations written to: {result_dir.resolve()}')


if __name__ == '__main__':
    main()
