import unittest

import torch

from src.model.v17d2_branch_adapter import Branch090AdaptedViabilityHead
from src.training.v17b_viability import ViabilityResidualHead


class V17D2BranchAdapterTest(unittest.TestCase):
    def test_zero_init_is_exact_noop_and_only_adapter_is_trainable(self):
        torch.manual_seed(9)
        head = ViabilityResidualHead(17, 11)
        head.residual_scale.data.fill_(0.2)
        wrapped = Branch090AdaptedViabilityHead(head, 17, rank=8)
        features = torch.randn(3, 4, 17)
        mask = torch.ones(3, 4, 5, dtype=torch.bool)
        expected = head(features, mask)
        actual = wrapped(features, mask)
        self.assertTrue(torch.equal(expected[0], actual[0]))
        self.assertTrue(torch.equal(expected[1], actual[1]))
        self.assertEqual(sum(p.numel() for p in wrapped.trainable_parameters()), 2 * 17 * 8)
        self.assertTrue(all(not p.requires_grad for p in head.parameters()))

    def test_nonzero_adapter_changes_only_primary_logit(self):
        torch.manual_seed(10)
        head = ViabilityResidualHead(13, 7)
        wrapped = Branch090AdaptedViabilityHead(head, 13, rank=3)
        wrapped.adapter.up.weight.data.normal_()
        features = torch.randn(2, 5, 13)
        mask = torch.ones(2, 5, 5, dtype=torch.bool)
        original = head(features, mask)[0]
        routed = wrapped(features, mask)[0]
        self.assertTrue(torch.equal(original[..., [0, 1, 2, 4]], routed[..., [0, 1, 2, 4]]))
        self.assertGreater(float((original[..., 3] - routed[..., 3]).abs().max().detach()), 0.0)

    def test_gradient_is_confined_to_adapter(self):
        head = ViabilityResidualHead(9, 5)
        wrapped = Branch090AdaptedViabilityHead(head, 9, rank=2)
        wrapped.adapter.up.weight.data.normal_(std=0.01)
        features = torch.randn(2, 4, 9)
        mask = torch.ones(2, 4, 5, dtype=torch.bool)
        wrapped(features, mask)[0][..., 3].sum().backward()
        self.assertTrue(all(p.grad is None for p in head.parameters()))
        self.assertTrue(all(p.grad is not None for p in wrapped.trainable_parameters()))


if __name__ == "__main__":
    unittest.main()
