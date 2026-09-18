from __future__ import annotations

import unittest

import torch

from src.model.cost_sensitive_policy_loss import cost_sensitive_policy_loss


class CostSensitivePolicyLossTest(unittest.TestCase):
    def test_severe_action_receives_larger_gradient_than_mild_action(self) -> None:
        logits = torch.zeros((1, 12), requires_grad=True)
        progress = torch.zeros((1, 5), requires_grad=True)
        advantages = [[0.0, 0.1, 2.0] + [None] * 9]
        result = cost_sensitive_policy_loss(
            logits,
            progress,
            advantages,
            torch.tensor([0]),
            torch.tensor([3]),
        )
        result.total.backward()
        self.assertGreater(logits.grad[0, 2].item(), logits.grad[0, 1].item())
        self.assertGreater(result.action.item(), 0.0)

    def test_tied_optimal_actions_are_aggregated(self) -> None:
        logits = torch.tensor([[0.0, 1.0, -1.0] + [-100.0] * 9], requires_grad=True)
        progress = torch.zeros((1, 5), requires_grad=True)
        advantages = [[0.0, 0.0, 0.2] + [None] * 9]
        result = cost_sensitive_policy_loss(
            logits,
            progress,
            advantages,
            torch.tensor([0]),
            torch.tensor([2]),
        )
        self.assertEqual(result.zero_advantage_accuracy, 1.0)
        result.total.backward()
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
