"""Focused tests for VGG feature-analysis computations."""

import unittest

import numpy as np
import torch

from codes.ivsn_invariance.feature_analysis import (
    centered_region,
    cue_target_distance_rows,
    effective_receptive_field,
    erf_distance_metrics,
    feature_distance_rows,
    image_point_to_feature_cell,
    parse_layer_windows,
    pooled_and_spatially_tolerant_metrics,
)


class FeatureAnalysisTests(unittest.TestCase):

    def test_parse_default_layer_windows(self):
        self.assertEqual(parse_layer_windows([]), {16: 5, 23: 3, 30: 1})
        self.assertEqual(parse_layer_windows(['10:7', '20:1']), {10: 7, 20: 1})
        with self.assertRaises(ValueError):
            parse_layer_windows(['16:4'])

    def test_pixel_location_maps_to_feature_cell_and_region_stays_in_bounds(self):
        self.assertEqual(
            image_point_to_feature_cell((360, 360), (720, 720), (14, 14)),
            (7, 7),
        )
        region = centered_region(16, 5, (0, 0), (720, 720), (56, 56))
        self.assertEqual(len(region.coordinates), 25)
        self.assertEqual(region.coordinates[0], (0, 0))
        self.assertEqual(region.coordinates[-1], (4, 4))

    def test_feature_distances_are_cellwise_and_channel_normalized(self):
        original = torch.zeros((1, 4, 3, 3))
        transformed = original.clone()
        transformed[0, :, 1, 1] = 1.0
        region = centered_region(30, 1, (1, 1), (3, 3), (3, 3))
        rows = feature_distance_rows(original, transformed, region)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]['euclidean_distance'], 2.0)
        self.assertAlmostEqual(rows[0]['rms_euclidean_distance'], 1.0)
        self.assertAlmostEqual(rows[0]['cosine_distance'], 1.0)

    def test_erf_superimposes_absolute_per_cell_gradients(self):
        input_tensor = torch.ones((1, 1, 2, 2), requires_grad=True)
        activation = 2.0 * input_tensor
        erf = effective_receptive_field(
            input_tensor,
            activation,
            coordinates=[(0, 0), (1, 1)],
        )
        np.testing.assert_allclose(erf, np.array([[2.0, 0.0], [0.0, 2.0]]))

    def test_spatially_tolerant_distance_ignores_cell_permutation(self):
        original = torch.zeros((1, 2, 3, 3))
        transformed = original.clone()
        original[0, :, 0, 0] = torch.tensor([1.0, 0.0])
        original[0, :, 0, 1] = torch.tensor([0.0, 1.0])
        transformed[0, :, 0, 0] = torch.tensor([0.0, 1.0])
        transformed[0, :, 0, 1] = torch.tensor([1.0, 0.0])
        region = centered_region(16, 3, (1, 1), (3, 3), (3, 3))
        strict = feature_distance_rows(original, transformed, region)
        tolerant = pooled_and_spatially_tolerant_metrics(
            original,
            transformed,
            region,
        )
        self.assertGreater(
            np.mean([row['cosine_distance'] for row in strict]),
            0.0,
        )
        self.assertAlmostEqual(tolerant['best_match_cosine_distance'], 0.0)

    def test_cue_target_metrics_distinguish_transformed_target(self):
        cue = torch.tensor([[[[1.0]], [[0.0]]]])
        original = torch.zeros((1, 2, 3, 3))
        transformed = original.clone()
        original[0, :, 1, 1] = torch.tensor([1.0, 0.0])
        transformed[0, :, 1, 1] = torch.tensor([0.0, 1.0])
        region = centered_region(30, 1, (1, 1), (3, 3), (3, 3))
        row = cue_target_distance_rows(
            cue,
            original,
            transformed,
            region,
        )[0]
        self.assertAlmostEqual(row['cue_original_cosine_similarity'], 1.0)
        self.assertAlmostEqual(row['cue_transformed_cosine_similarity'], 0.0)
        self.assertAlmostEqual(row['cue_similarity_drop'], 1.0)

    def test_erf_metrics_ignore_global_gradient_scale(self):
        original = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        transformed = 3.0 * original
        metrics = erf_distance_metrics(original, transformed)
        self.assertAlmostEqual(metrics['erf_cosine_distance'], 0.0)
        self.assertAlmostEqual(metrics['erf_total_variation_distance'], 0.0)
        self.assertAlmostEqual(metrics['erf_centroid_shift_px'], 0.0)
        self.assertAlmostEqual(metrics['erf_mass_overlap'], 1.0)


if __name__ == '__main__':
    unittest.main()
