import unittest

from src.data.schemas import QAExample
from src.data.select_v3_rank_confirm import select_role_reserve


class SelectV3RankConfirmTest(unittest.TestCase):
    def test_selects_only_eligible_rows_after_consumed_prefix(self) -> None:
        rows = [
            QAExample(
                example_id=str(index),
                dataset="test",
                question="q",
                context="c",
                answer="a",
            )
            for index in range(6)
        ]
        maximum = {
            "0": 0.95,
            "1": 0.2,
            "2": 0.90,
            "3": 0.89,
            "4": 1.0,
            "5": 0.91,
        }
        selected = select_role_reserve(
            rows, maximum, minimum_fidelity=0.90, consumed=2
        )
        self.assertEqual([row.example_id for row in selected], ["4", "5"])


if __name__ == "__main__":
    unittest.main()
