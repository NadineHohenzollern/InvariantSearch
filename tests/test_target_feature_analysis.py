"""Tests for isolated-target VGG activation and ERF analysis."""

import unittest

import numpy as np
import torch

from codes.ivsn_invariance.domain import TransformSpec
from codes.ivsn_invariance.target_feature_analysis import (
    activation_difference_metrics,
    full_layer_erf,
)
from codes.ivsn_invariance.target_feature_cli import TargetRecord, select_erf_records
from codes.ivsn_invariance.target_feature_reporting import (
    align_erf_to_original,
    aligned_erf_metrics,
    build_grouped_target_activation_summary,
)


class TargetFeatureAnalysisTests(unittest.TestCase):

    def test_activation_difference_uses_all_tensor_elements(self):
        original = torch.zeros((1, 2, 2, 2))
        transformed = torch.ones_like(original)
        metrics = activation_difference_metrics(original, transformed)
        self.assertEqual(metrics['n_activations'], 8)
        self.assertAlmostEqual(metrics['mean_absolute_difference'], 1.0)
        self.assertAlmostEqual(metrics['std_absolute_difference'], 0.0)
        self.assertAlmostEqual(metrics['sum_absolute_difference'], 8.0)
        self.assertAlmostEqual(metrics['rms_difference'], 1.0)
        self.assertAlmostEqual(metrics['cosine_distance'], 1.0)

    def test_full_layer_erf_superimposes_every_spatial_cell(self):
        input_tensor = torch.ones((1, 1, 2, 2), requires_grad=True)
        activation = 3.0 * input_tensor
        erf = full_layer_erf(input_tensor, activation)
        np.testing.assert_allclose(erf, np.full((2, 2), 3.0))

    def test_erf_selection_uses_unique_paths_per_category(self):
        records = [
            TargetRecord(0, 0, 'identical', 'cats', 'cat_a.png'),
            TargetRecord(1, 0, 'different', 'cats', 'cat_a.png'),
            TargetRecord(2, 0, 'different', 'cats', 'cat_b.png'),
            TargetRecord(3, 0, 'identical', 'dogs', 'dog_a.png'),
        ]
        selected = select_erf_records(records, per_class=2)
        self.assertEqual(
            [(row.target_category, row.target_path) for row in selected],
            [('cats', 'cat_a.png'), ('cats', 'cat_b.png'), ('dogs', 'dog_a.png')],
        )

    def test_grouped_summary_reports_across_trial_standard_deviation(self):
        base = {
            'condition_name': 'rotation_30',
            'condition_group': 'rotation_deg',
            'condition_value': 30.0,
            'layer': 16,
            'channels': 256,
            'activation_height': 8,
            'activation_width': 8,
            'rms_difference': 0.0,
            'relative_mean_absolute_difference': 0.0,
            'cosine_distance': 0.0,
        }
        rows = [
            {**base, 'trial_type': 'identical', 'mean_absolute_difference': 1.0},
            {**base, 'trial_type': 'different', 'mean_absolute_difference': 3.0},
        ]
        summary = build_grouped_target_activation_summary(rows)[0]
        self.assertEqual(summary['n_all'], 2)
        self.assertAlmostEqual(summary['mean_absolute_difference_all'], 2.0)
        self.assertAlmostEqual(
            summary['std_mean_absolute_difference_all'],
            np.sqrt(2.0),
        )

    def test_identity_erf_alignment_preserves_map(self):
        values = np.zeros((32, 32), dtype=np.float32)
        values[8:14, 18:24] = 1.0
        aligned = align_erf_to_original(values, TransformSpec())
        np.testing.assert_allclose(aligned, values, atol=0.06)

    def test_inverse_rotation_alignment_recovers_spatial_pattern(self):
        original = np.zeros((32, 32), dtype=np.float32)
        original[5:11, 19:25] = 1.0
        transformed = align_erf_to_original(
            original,
            TransformSpec(rotation_deg=-90.0),
        )
        recovered = align_erf_to_original(
            transformed,
            TransformSpec(rotation_deg=90.0),
        )
        metrics = aligned_erf_metrics(original, transformed, recovered)
        self.assertGreater(metrics['aligned_erf_mass_overlap'], 0.90)
        self.assertGreater(metrics['erf_alignment_gain_mass_overlap'], 0.50)


if __name__ == '__main__':
    unittest.main()
