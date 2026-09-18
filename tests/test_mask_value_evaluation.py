from __future__ import annotations

import unittest

import torch

from src.evaluation.mask_value_evaluation import logits_to_attainment


class MaskValueEvaluationTest(unittest.TestCase):
    def test_attainment_always_uses_five_fixed_levels(self) -> None:
        logits = torch.tensor([[2.0, 1.0, -1.0, -2.0, -3.0], [2.0, 1.0, 0.5, 0.1, -1.0]])
        self.assertEqual(logits_to_attainment(logits).tolist(), [2, 4])


if __name__ == "__main__":
    unittest.main()
