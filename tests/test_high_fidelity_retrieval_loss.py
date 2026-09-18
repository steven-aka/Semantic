import unittest,torch
from src.model.high_fidelity_retrieval_loss import high_fidelity_retrieval_loss
class T(unittest.TestCase):
 def test_positive_score_lowers_loss(self):
  labels=torch.tensor([0,4,0,4]);idx=torch.tensor([0,0,1,1]);a=torch.zeros(4,5,requires_grad=True);x=high_fidelity_retrieval_loss(a,labels,idx);b=a.detach().clone();b[[1,3],3]=3;y=high_fidelity_retrieval_loss(b,labels,idx);self.assertLess(y,x);x.backward();self.assertTrue(torch.isfinite(a.grad).all())
 def test_missing_positive(self):
  with self.assertRaises(ValueError):high_fidelity_retrieval_loss(torch.zeros(2,5),torch.zeros(2,dtype=torch.long),torch.zeros(2,dtype=torch.long))
