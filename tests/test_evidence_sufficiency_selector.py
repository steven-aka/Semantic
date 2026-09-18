import unittest
import torch

from src.model.evidence_sufficiency_selector import conservative_sufficiency_loss


class EvidenceSufficiencyLossTest(unittest.TestCase):
    def test_loss_backpropagates(self):
        logits = torch.zeros((5, 5), requires_grad=True)
        targets = torch.tensor([[1,1,1,0,0],[1,1,1,1,0],[1,1,1,0,0],[1,1,0,0,0],[1,1,1,0,0]], dtype=torch.bool)
        active = torch.ones_like(targets)
        loss = conservative_sufficiency_loss(logits, targets, active, torch.zeros(5, dtype=torch.long))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
