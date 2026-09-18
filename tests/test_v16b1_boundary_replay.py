import random,unittest
from src.training.train_v16b1_boundary_replay import sample_batch


class FakeData:
    def __init__(self):
        def h(depth,reached):return {"history":list(range(depth)),"reached_levels":reached,"optimal_actions":[depth],"source":"model_induced_v8_beam8"}
        self.items=[]
        for i in range(8):
            row={"example_id":str(i),"active_levels":[.6,.7,.8,.9],"packet_tokens":[1]*12,"boundary_histories":[h(2,1)],"retention_histories":[h(2,1),h(9,4)]}
            self.items.append((row,object()))
    def __getitem__(self,index):return self.items[index]


class BoundaryReplayTest(unittest.TestCase):
    def test_half_boundary_half_depth_matched_replay(self):
        batch=sample_batch(FakeData(),list(range(8)),list(range(8)),random.Random(1),histories_per_query=2)
        self.assertEqual(len(batch["optimal_actions"]),16)
        self.assertTrue((batch["history_lengths"]==2).all())
