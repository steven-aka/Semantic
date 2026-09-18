import unittest

from src.evaluation.v17a_counterfactual_branch_audit import state_consequences, superset_minimum_tokens
from src.search.sequential_trajectory_dp import SequentialTrajectoryDP
from tests.test_sequential_trajectory_dp import row


class V17AAuditTest(unittest.TestCase):
    def test_detects_action_consequence_boundary(self):
        fidelity = [0.0, 0.6, 0.0, 0.9, 0.0, 0.6, 0.0, 0.0]
        exact = [row(mask, value) for mask, value in enumerate(fidelity)]
        dp = SequentialTrajectoryDP(exact, [0.6, 0.9])
        result = state_consequences(dp, superset_minimum_tokens(exact, dp.levels), (), dp.tokens[-1], cost_gap=.03)
        self.assertTrue(result["feasibility_critical"])
        self.assertTrue(result["primary_090_critical"])


if __name__ == "__main__": unittest.main()
