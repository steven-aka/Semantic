import unittest

import torch

from src.training.train_v17act_c1_factorized_omission_cv import outcome, pair_feature


class FactorizedOmissionTest(unittest.TestCase):
    def test_features_depend_on_omitted_pair(self):
        base = list(range(12))
        alternative = list(range(9)) + [10, 9, 11]
        torch.manual_seed(5)
        packets = torch.randn(12, 8)
        query = torch.randn(8)
        feature = pair_feature(base, alternative, packets, query, [10] * 12)
        self.assertEqual(feature.ndim, 1)
        self.assertTrue(torch.isfinite(feature).all())
        self.assertGreater(torch.linalg.vector_norm(feature).item(), 0)

    def test_unattainable_level_is_masked(self):
        hits, cumulative = outcome({"f1": 0.9, "tokens": 100}, {0.6, 0.7, 0.8, 0.9})
        self.assertEqual(hits, (True, True, True, True, None))
        self.assertEqual(cumulative, 400)


if __name__ == "__main__":
    unittest.main()
