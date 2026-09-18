import unittest
import torch

from src.model.packet_interaction_probe import PacketInteractionProbe,interaction_loss


class PacketInteractionProbeTest(unittest.TestCase):
    def test_shapes_and_loss(self):
        model=PacketInteractionProbe(dim=32,layers=1,heads=4,dropout=0);n=8
        args=(torch.randn(2,12,32),torch.randn(2,32),torch.tensor([3]*n),torch.tensor([5]*n),torch.tensor([2]*n),torch.tensor([4]*n),torch.tensor([1]*n),torch.randn(n,5),torch.rand(n),torch.tensor([0]*4+[1]*4))
        classes,anchors=model(*args);self.assertEqual(tuple(classes.shape),(n,4));self.assertEqual(tuple(anchors.shape),(n,5,3))
        loss=interaction_loss(classes,anchors,torch.arange(n)%4,torch.ones((n,5),dtype=torch.long),torch.ones((n,5),dtype=torch.bool),args[-1],torch.ones(4));loss.backward();self.assertTrue(torch.isfinite(loss))
