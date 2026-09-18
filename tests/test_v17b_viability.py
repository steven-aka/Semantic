import unittest

import torch
from torch import nn

from src.data.build_v17b0_viability_artifact import build_state
from src.search.sequential_trajectory_dp import SequentialTrajectoryDP
from src.training.v17b_viability import StratifiedStateSampler, ViabilityResidualHead, masked_viability_loss, set_valued_action_loss
from src.model.v17b_viability_policy import V17BViabilityPolicy
from tests.test_sequential_trajectory_dp import row


class V17BViabilityTest(unittest.TestCase):
    def test_unattainable_and_resolved_anchors_are_masked(self):
        exact = [row(mask, value) for mask, value in enumerate([0.0, .6, 0.0, .9, 0.0, .6, 0.0, 0.0])]
        dp = SequentialTrajectoryDP(exact, [.6, .7, .8, .9])
        record = build_state(dp, (), [.6, .7, .8, .9], "deployed")
        self.assertIsNotNone(record)
        self.assertEqual(record["anchor_status"][4], "undefined")
        self.assertFalse(record["active_target_mask"][4])
        self.assertTrue(all(action["future_viability"][4] is None for action in record["actions"]))

    def test_losses_and_zero_initialized_residual_have_gradients(self):
        head = ViabilityResidualHead(7, 4)
        features = torch.randn(2, 3, 7)
        target_mask = torch.tensor([[[1, 1, 1, 1, 0]] * 3] * 2, dtype=torch.bool)
        logits, residual = head(features, target_mask)
        targets = torch.randint(0, 2, logits.shape).float()
        action_logits = torch.randn(2, 3, requires_grad=True)
        optimal = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
        legal = torch.ones_like(optimal)
        loss = masked_viability_loss(logits, targets, target_mask) + set_valued_action_loss(action_logits, optimal, legal) + residual.square().mean()
        self.assertTrue(torch.isfinite(loss)); loss.backward()
        self.assertIsNotNone(head.viability[0].weight.grad)
        self.assertIsNotNone(head.residual_scale.grad)

    def test_sampler_is_exactly_half_each_layer(self):
        rows = [{"provenance": layer, "example_id": f"q{i}"} for layer in ("deployed", "one_hop") for i in range(3)]
        indices = StratifiedStateSampler(rows, 7).sample(8, 0)
        self.assertEqual(sum(rows[i]["provenance"] == "deployed" for i in indices), 4)

    def test_v17_policy_freezes_v8_and_starts_as_exact_residual_noop(self):
        lm = nn.Module(); lm.config = type("Config", (), {"hidden_size": 8})()
        model = V17BViabilityPolicy(lm, model_dim=8, set_layers=1, set_heads=1, viability_hidden_dim=4)
        model.freeze_v8(); model.eval()
        self.assertTrue(all(p.requires_grad == name.startswith("viability_head.") for name, p in model.named_parameters()))
        packets = torch.randn(1, 12, 8); question = torch.randn(1, 8); history = torch.randn(1, 8)
        selected = torch.zeros(1, 12, dtype=torch.bool); indices = torch.zeros(1, dtype=torch.long); fractions = torch.full((1, 12), 1 / 12)
        features, _ = model.action_features(packets, question, history, selected, indices, fractions)
        base = model.action_head(features).squeeze(-1)
        active = torch.ones(1, 12, 5, dtype=torch.bool)
        scored, _, _ = model.score_states_with_viability(packets, question, history, selected, indices, fractions, active)
        self.assertTrue(torch.equal(base, scored))
        progress = torch.tensor([[10.0, -10.0, -10.0, -10.0, -10.0]])
        deployed_mask = model.deployment_active_target_mask(progress, torch.tensor([4]))
        self.assertEqual(deployed_mask[0, 0].tolist(), [False, True, True, True, False])


if __name__ == "__main__": unittest.main()
