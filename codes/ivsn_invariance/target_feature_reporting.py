"""CSV summaries and figures for target-only VGG analyses."""

from pathlib import Path
from typing import List, Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .feature_analysis import normalized_map
from .feature_reporting import GROUPS, condition_x_label, confidence_interval_95


ACTIVATION_METRICS = (
    'mean_absolute_difference',
    'rms_difference',
    'relative_mean_absolute_difference',
    'cosine_distance',
)


def build_grouped_target_activation_summary(
        rows: Sequence[dict],
    ) -> List[dict]:
    """Aggregate trial-level activation differences with mean, SD and CI."""
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    keys = [
        'condition_name', 'condition_group', 'condition_value',
        'layer', 'channels', 'activation_height', 'activation_width',
    ]
    grouped_rows = []
    for key_values, condition_frame in frame.groupby(keys, sort=False):
        row = dict(zip(keys, key_values))
        for group_name, _ in GROUPS:
            if group_name == 'all':
                subset = condition_frame
            else:
                trial_type = group_name[len('target_'):]
                subset = condition_frame[
                    condition_frame['trial_type'] == trial_type
                ]
            row[f'n_{group_name}'] = int(len(subset))
            for metric in ACTIVATION_METRICS:
                values = subset[metric].to_numpy(dtype=np.float64)
                if len(values):
                    row[f'{metric}_{group_name}'] = float(values.mean())
                    row[f'std_{metric}_{group_name}'] = float(
                        values.std(ddof=1) if len(values) > 1 else 0.0
                    )
                    row[f'ci95_{metric}_{group_name}'] = (
                        confidence_interval_95(values)
                    )
                else:
                    row[f'{metric}_{group_name}'] = float('nan')
                    row[f'std_{metric}_{group_name}'] = float('nan')
                    row[f'ci95_{metric}_{group_name}'] = float('nan')
        grouped_rows.append(row)
    return sorted(
        grouped_rows,
        key=lambda row: (
            str(row['condition_group']),
            float(row['condition_value']),
            int(row['layer']),
        ),
    )


def save_grouped_target_activation_plots(
        grouped_rows: Sequence[dict],
        plot_dir: Path,
    ) -> None:
    """Plot per-layer transformation sweeps with trial SD error bars."""
    if not grouped_rows:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(grouped_rows)
    metric_labels = {
        'mean_absolute_difference': 'Mean absolute activation difference',
        'rms_difference': 'RMS activation difference',
        'relative_mean_absolute_difference': 'Relative mean absolute difference',
        'cosine_distance': 'Cosine distance',
    }
    for layer in sorted(frame['layer'].unique()):
        layer_frame = frame[frame['layer'] == layer].sort_values('condition_value')
        values = layer_frame['condition_value'].to_numpy(dtype=np.float64)
        condition_group = str(layer_frame['condition_group'].iloc[0])
        x = np.arange(len(values))
        width = 0.24
        for metric, label in metric_labels.items():
            fig, axis = plt.subplots(figsize=(8, 4.8))
            for group_index, (group_name, group_label) in enumerate(GROUPS):
                offset = (group_index - 1) * width
                axis.bar(
                    x + offset,
                    layer_frame[f'{metric}_{group_name}'],
                    width=width,
                    yerr=layer_frame[f'std_{metric}_{group_name}'],
                    capsize=3,
                    label=group_label,
                )
            axis.set_xticks(x, [f'{value:g}' for value in values])
            axis.set_xlabel(condition_x_label(condition_group))
            axis.set_ylabel(label)
            channels = int(layer_frame['channels'].iloc[0])
            height = int(layer_frame['activation_height'].iloc[0])
            width_value = int(layer_frame['activation_width'].iloc[0])
            axis.set_title(
                f'VGG layer {layer} ({channels}x{height}x{width_value})'
            )
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
            axis.legend()
            fig.tight_layout()
            fig.savefig(
                plot_dir / f'layer_{layer}_{metric}.png',
                dpi=300,
            )
            plt.close(fig)


def save_target_erf_figure(
        original_image: Image.Image,
        transformed_image: Image.Image,
        original_erf: np.ndarray,
        transformed_erf: np.ndarray,
        title: str,
        path: Path,
    ) -> None:
    """Save original/transformed ERF overlays and normalized difference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    display_size = (original_erf.shape[1], original_erf.shape[0])
    original_rgb = np.asarray(
        original_image.resize(display_size, Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    transformed_rgb = np.asarray(
        transformed_image.resize(display_size, Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    original_normalized = normalized_map(original_erf)
    transformed_normalized = normalized_map(transformed_erf)
    difference = np.abs(original_normalized - transformed_normalized)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(original_rgb)
    axes[0].imshow(original_normalized, cmap='magma', alpha=0.55, vmin=0, vmax=1)
    axes[0].set_title('Original image + ERF')
    axes[1].imshow(transformed_rgb)
    axes[1].imshow(
        transformed_normalized,
        cmap='magma',
        alpha=0.55,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title('Transformed image + ERF')
    axes[2].imshow(difference, cmap='viridis', vmin=0, vmax=1)
    axes[2].set_title('Absolute ERF difference')
    for axis in axes:
        axis.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches='tight')
    plt.close(fig)


def save_target_erf_arrays(
        original_erf: np.ndarray,
        transformed_erf: np.ndarray,
        path: Path,
    ) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        original_erf=original_erf,
        transformed_erf=transformed_erf,
        original_erf_normalized=normalized_map(original_erf),
        transformed_erf_normalized=normalized_map(transformed_erf),
        absolute_normalized_difference=np.abs(
            normalized_map(original_erf) - normalized_map(transformed_erf)
        ),
    )
