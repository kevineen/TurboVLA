"""Unit tests for the vendored GR-BC weight module (starVLA.training.grbc_weights).

Run from the TurboVLA repo root:
  PYTHONPATH=$PWD:$PWD/third_party/starvla_runtime .venv310/bin/python -m unittest tests.test_grbc_weights -v
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from starVLA.training.grbc_weights import GRBCWeights, weighted_loss


def write_weight_table(dirpath, indices, weights, n_action_step=12):
    df = pd.DataFrame(
        {
            "index": np.asarray(indices, dtype=np.int64),
            "episode_index": np.zeros(len(indices), dtype=np.int64),
            "frame_index": np.arange(len(indices), dtype=np.int64),
            "graph_transition_weight": np.asarray(weights, dtype=np.float32),
        }
    )
    path = Path(dirpath) / "transition_weights.parquet"
    df.to_parquet(path)
    (Path(dirpath) / "summary.json").write_text(
        json.dumps({"config": {"n_action_step": n_action_step}, "weight_column": "graph_transition_weight"})
    )
    return path


class GRBCWeightsTest(unittest.TestCase):
    def test_load_and_batch_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_weight_table(tmp, [0, 1, 2, 3], [1.5, 0.0, 0.5, 1.0])
            provider = GRBCWeights(path)
            examples = [{"index": 3}, {"index": 1}, {"index": 0}]
            weights, stats = provider.compute_batch_weights(examples)
            self.assertTrue(torch.equal(weights, torch.tensor([1.0, 0.0, 1.5])))
            self.assertAlmostEqual(stats["grbc_mean_weight"], 2.5 / 3, places=5)
            self.assertAlmostEqual(stats["grbc_zero_fraction"], 1 / 3, places=5)

    def test_missing_weight_index_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_weight_table(tmp, [0, 1], [1.0, 1.0])
            provider = GRBCWeights(path)
            with self.assertRaises(KeyError):
                provider.compute_batch_weights([{"index": 5}])

    def test_batch_requires_index_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = GRBCWeights(write_weight_table(tmp, [0], [1.0]))
            with self.assertRaises(KeyError):
                provider.compute_batch_weights([{"action": np.zeros((12, 6), dtype=np.float16)}])

    def test_invalid_table_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                GRBCWeights(write_weight_table(tmp, [0, 0], [1.0, 1.0]))  # duplicate index
            with self.assertRaises(ValueError):
                GRBCWeights(write_weight_table(tmp, [0], [-1.0]))  # negative weight

    def test_validate_indices_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = GRBCWeights(write_weight_table(tmp, [0, 1, 2], [1.0, 1.0, 1.0]))
            provider.validate_indices([2, 0])  # subset is fine
            with self.assertRaises(ValueError):
                provider.validate_indices([0, 7])

    def test_validate_horizon(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = GRBCWeights(write_weight_table(tmp, [0], [1.0], n_action_step=12))
            provider.validate_horizon(12)
            with self.assertRaises(ValueError):
                provider.validate_horizon(30)


class WeightedLossTest(unittest.TestCase):
    def test_mean_reduction_matches_uniform_mean(self):
        per_sample = torch.tensor([1.0, 2.0, 4.0])
        weights = torch.ones(3)
        self.assertAlmostEqual(weighted_loss(per_sample, weights).item(), per_sample.mean().item(), places=6)

    def test_mean_reduction_weights_batch_size_denominator(self):
        per_sample = torch.tensor([1.0, 2.0, 4.0])
        weights = torch.tensor([2.0, 0.0, 0.0])
        # denominator is batch size (3), so loss = (1*2)/3
        self.assertAlmostEqual(weighted_loss(per_sample, weights).item(), 2.0 / 3, places=6)

    def test_self_normalized_reduction(self):
        per_sample = torch.tensor([1.0, 2.0, 4.0])
        weights = torch.tensor([2.0, 0.0, 0.0])
        self.assertAlmostEqual(
            weighted_loss(per_sample, weights, reduction="self_normalized").item(), 1.0, places=6
        )

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            weighted_loss(torch.zeros(2, 3), torch.ones(2))
        with self.assertRaises(ValueError):
            weighted_loss(torch.zeros(3), torch.ones(2))

    def test_l1_per_sample_matches_flat_mean(self):
        # Mirrors the patched TurboVLA.forward: per-sample L1 over (T, D) equals the
        # uniform-mean scalar loss when every weight is 1.
        predicted = torch.randn(4, 12, 6)
        targets = torch.randn(4, 12, 6)
        losses = torch.nn.functional.l1_loss(predicted, targets, reduction="none")
        per_sample = losses.mean(dim=tuple(range(1, losses.ndim)))
        self.assertEqual(per_sample.shape, (4,))
        self.assertAlmostEqual(
            weighted_loss(per_sample, torch.ones(4)).item(), losses.mean().item(), places=6
        )


if __name__ == "__main__":
    unittest.main()
