from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.training.mask_value_data import CachedMaskValueCollator, CachedMaskValueDataset


class CachedMaskValueDataTest(unittest.TestCase):
    def test_cache_ids_and_collation_are_exact(self) -> None:
        rows = [{
            "example_id": "x",
            "active_levels": [0.6, 0.7, 0.8, 0.9, 0.95],
            "mask_values": [
                {"mask": 0, "attained_levels": 0, "sampling_probability": 1.0},
                {"mask": 1, "attained_levels": 1, "sampling_probability": 0.25},
            ],
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pt"
            torch.save({
                "example_ids": ["x"],
                "packets": torch.zeros(1, 12, 512, dtype=torch.bfloat16),
                "questions": torch.zeros(1, 512, dtype=torch.bfloat16),
            }, path)
            dataset = CachedMaskValueDataset(rows, str(path))
            batch = CachedMaskValueCollator()([dataset[0]])
        self.assertEqual(tuple(batch["packets"].shape), (1, 12, 512))
        self.assertEqual(batch["sampling_probabilities"].tolist(), [1.0, 0.25])

    def test_cache_id_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pt"
            torch.save({
                "example_ids": ["wrong"],
                "packets": torch.zeros(1, 12, 512),
                "questions": torch.zeros(1, 512),
            }, path)
            with self.assertRaises(ValueError):
                CachedMaskValueDataset([{"example_id": "x"}], str(path))


if __name__ == "__main__":
    unittest.main()
