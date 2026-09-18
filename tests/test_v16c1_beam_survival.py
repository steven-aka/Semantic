import unittest
import torch

from src.training.train_v16c1_beam_survival import beam_survival_loss


class V16C1LossTest(unittest.TestCase):
    def test_survival_margin_uses_cumulative_beam_threshold(self):
        parents = [{"history": (), "viable_actions": (2,), "optimal_actions": (2,)}]
        logits = [torch.tensor([3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
        loss, diagnostic = beam_survival_loss([torch.tensor(0.0)], logits, parents, beam_width=2, margin=.1)
        self.assertGreater(float(loss), 0.0)
        self.assertLess(diagnostic["margin"], 0.0)

    def test_zero_when_viable_child_clears_margin(self):
        parents = [{"history": (), "viable_actions": (0,), "optimal_actions": (0,)}]
        logits = [torch.tensor([3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
        loss, _ = beam_survival_loss([torch.tensor(0.0)], logits, parents, beam_width=2, margin=.1)
        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
