import unittest
import torch
from src.model.trajectory_candidate_head import TrajectoryCandidateHead,candidate_set_loss
class TestTrajectoryCandidateHead(unittest.TestCase):
 def test_shapes_and_set_loss(self):
  torch.manual_seed(1);m=TrajectoryCandidateHead(model_dim=16,layers=1,heads=4,dropout=0)
  logits=m(torch.randn(2,12,16),torch.randn(2,16),torch.tensor([0,3,7,1,9]),torch.tensor([0,0,0,1,1]))
  self.assertEqual(tuple(logits.shape),(5,));loss=candidate_set_loss(logits,torch.tensor([1,0,1,0,1],dtype=torch.bool),torch.tensor([0,0,0,1,1]));loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(any(p.grad is not None for p in m.parameters()))
 def test_missing_optimum_rejected(self):
  with self.assertRaises(ValueError):candidate_set_loss(torch.zeros(2),torch.zeros(2,dtype=torch.bool),torch.zeros(2,dtype=torch.long))
if __name__=='__main__':unittest.main()
