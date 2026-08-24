"""Target-only activation and ERF computations for transformed images."""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .feature_analysis import effective_receptive_field


def activation_difference_metrics(
        original: torch.Tensor,
        transformed: torch.Tensor,
    ) -> Dict[str, float]:
    """Summarize elementwise activation changes for one VGG layer."""
    if original.shape != transformed.shape:
        raise ValueError(
            f'Activation shapes differ: {tuple(original.shape)} vs '
            f'{tuple(transformed.shape)}.'
        )

    original_flat = original.detach().reshape(-1).float()
    transformed_flat = transformed.detach().reshape(-1).float()
    absolute_difference = (original_flat - transformed_flat).abs()
    squared_difference = (original_flat - transformed_flat).square()
    eps = torch.finfo(original_flat.dtype).eps

    original_norm = torch.linalg.vector_norm(original_flat)
    transformed_norm = torch.linalg.vector_norm(transformed_flat)
    if original_norm <= eps and transformed_norm <= eps:
        cosine_distance = original_flat.new_tensor(0.0)
    elif original_norm <= eps or transformed_norm <= eps:
        cosine_distance = original_flat.new_tensor(1.0)
    else:
        cosine_distance = 1.0 - F.cosine_similarity(
            original_flat.unsqueeze(0),
            transformed_flat.unsqueeze(0),
            dim=1,
        )[0]

    original_abs_mean = original_flat.abs().mean()
    return {
        'n_activations': int(original_flat.numel()),
        'original_mean_activation': float(original_flat.mean().cpu()),
        'transformed_mean_activation': float(transformed_flat.mean().cpu()),
        'mean_absolute_difference': float(absolute_difference.mean().cpu()),
        'std_absolute_difference': float(
            absolute_difference.std(unbiased=False).cpu()
        ),
        'sum_absolute_difference': float(absolute_difference.sum().cpu()),
        'max_absolute_difference': float(absolute_difference.max().cpu()),
        'rms_difference': float(torch.sqrt(squared_difference.mean()).cpu()),
        'relative_mean_absolute_difference': float(
            (absolute_difference.mean() / torch.clamp(original_abs_mean, min=eps)).cpu()
        ),
        'cosine_distance': float(torch.clamp(cosine_distance, 0.0, 2.0).cpu()),
    }


def full_layer_erf(
        input_tensor: torch.Tensor,
        activation: torch.Tensor,
        retain_graph: bool = False,
    ) -> np.ndarray:
    """Aggregate per-cell ERFs across every spatial cell of a VGG layer."""
    if activation.ndim != 4 or activation.shape[0] != 1:
        raise ValueError(
            'Expected activation shape [1, channels, height, width], got '
            f'{tuple(activation.shape)}.'
        )
    height, width = (int(value) for value in activation.shape[-2:])
    coordinates: Tuple[Tuple[int, int], ...] = tuple(
        (row, col)
        for row in range(height)
        for col in range(width)
    )
    return effective_receptive_field(
        input_tensor,
        activation,
        coordinates,
        retain_graph=retain_graph,
    )
