import unittest

import torch

from src.data.build_v17sel_c0_train_candidates import outcome
from src.model.topk_set_selector import TopKSetSelector, successful_set_loss, FEATURE_DIM


class SelC0ContractTests(unittest.TestCase):
    def test_earliest_success_survives_later_fidelity_rollback(self):
        fidelity = [0.0] * 4096
        tokens = list(range(4096))
        fidelity[1] = 0.9
        fidelity[3] = 0.2
        fidelity[7] = 0.95
        result = outcome(list(range(12)), fidelity, tokens, {.6, .7, .8, .9, .95})
        self.assertEqual(result["success"], [True] * 5)
        self.assertEqual(result["earliest_tokens"][3], 1)
        self.assertTrue(result["complete"])
        self.assertEqual(result["oracle_ordered_complete_cumulative_tokens"], 11)


    def test_duplicate_projected_order_cannot_gain_probability_mass(self):
        scores = torch.tensor([[0.0, 0.0, -torch.inf]], requires_grad=True)
        success = torch.tensor([[False, True, False]])
        loss = successful_set_loss(scores, success)
        self.assertTrue(torch.allclose(loss, torch.tensor(0.6931472), atol=1e-6))
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad[:, :2]).all())
        self.assertEqual(scores.grad[0, 2], 0)


    def test_selector_masks_duplicate_slot_during_inference(self):
        model = TopKSetSelector().eval()
        features = torch.randn(2, 3, FEATURE_DIM)
        unique = torch.tensor([[True, True, False], [True, False, True]])
        scores = model(features, unique)
        self.assertTrue(torch.isneginf(scores[~unique]).all())
        self.assertTrue(torch.isfinite(scores[unique]).all())


if __name__ == "__main__":
    unittest.main()
