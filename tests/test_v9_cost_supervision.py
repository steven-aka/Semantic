from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data.build_v9_cost_supervision import add_costs
from src.data.schemas import ExactSearchResult, write_jsonl
from src.search.atomic_nested_chain import mask_to_state
from src.search.sequential_trajectory_dp import SequentialTrajectoryDP


class V9CostSupervisionTest(unittest.TestCase):
    def test_costs_match_exact_optimal_action_set(self) -> None:
        fidelity = [0.0, 0.6, 0.0, 0.8, 0.7, 0.6, 0.9, 0.95]
        exact = [
            ExactSearchResult(
                example_id="example",
                state=mask_to_state(mask, 3),
                tokens=mask.bit_count() + 1,
                prediction="",
                answer_em=0.0,
                answer_f1=0.0,
                fact_recall=0.0,
                fidelity=value,
            )
            for mask, value in enumerate(fidelity)
        ]
        dp = SequentialTrajectoryDP(exact, [0.6, 0.7, 0.8, 0.9])
        source = {
            "example_id": "example",
            "active_levels": [0.6, 0.7, 0.8, 0.9],
            "histories": [
                {
                    "history": [],
                    "selected_mask": 0,
                    "reached_levels": 0,
                    "optimal_actions": list(dp.optimal_actions(0, 0)),
                    "source": "oracle_0",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            write_jsonl(Path(directory) / "example.jsonl", exact)
            result, counts = add_costs(source, directory)
        costs = result["histories"][0]["action_advantages"]
        zeros = {index for index, value in enumerate(costs) if value == 0.0}
        self.assertEqual(zeros, set(dp.optimal_actions(0, 0)))
        self.assertEqual(counts["histories"], 1)


if __name__ == "__main__":
    unittest.main()
