import unittest
import json
from pathlib import Path

from src.evaluation.v17d0_two_bottleneck_audit import action_targets, state_signature, trajectory_stats
from src.search.sequential_trajectory_dp import SequentialTrajectoryDP
from tests.test_sequential_trajectory_dp import row


class V17D0AuditTest(unittest.TestCase):
    def test_trajectory_attainment_depends_on_order(self):
        exact=[row(mask,value) for mask,value in enumerate([0,.9,0,.9,0,0,0,0])]
        dp=SequentialTrajectoryDP(exact,[.6,.9])
        early=trajectory_stats(dp,[0,1,2],1); late=trajectory_stats(dp,[1,0,2],1)
        self.assertEqual(early["target_first_attainment_depth"],1)
        self.assertEqual(late["target_first_attainment_depth"],2)

    def test_signature_ignores_packet_identity(self):
        first={0:[1,0,None],1:[0,1,None]}; second={8:[0,1,None],3:[1,0,None]}
        self.assertEqual(state_signature(first,2),state_signature(second,2))

    def test_canonical_manifest_freezes_causal_counts(self):
        path=Path("results/v2_rank_then_cut/v17d0_canonical_harness_and_two_bottleneck_audit/canonical_replay_manifest.json")
        manifest=json.loads(path.read_text())
        self.assertEqual(manifest["expected_regression"]["residual_off_primary_090"],279)
        self.assertEqual(manifest["expected_regression"]["v17b1_primary_090"],279)
        self.assertEqual(manifest["expected_regression"]["shared_failure_set"],21)
        self.assertEqual(manifest["beam"]["width"],8)


if __name__=="__main__": unittest.main()
