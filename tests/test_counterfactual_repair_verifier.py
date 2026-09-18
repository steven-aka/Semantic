import unittest
import torch
from src.model.counterfactual_repair_verifier import verifier_loss

class CounterfactualVerifierTest(unittest.TestCase):
 def test_balanced_loss_backpropagates(self):
  logits=torch.zeros((4,4),requires_grad=True);delta=torch.zeros(4,requires_grad=True);target=torch.arange(4);loss=verifier_loss(logits,delta,target,torch.zeros(4),torch.zeros(4,dtype=torch.long),torch.ones(4));loss.backward();self.assertTrue(torch.isfinite(loss));self.assertIsNotNone(logits.grad)
