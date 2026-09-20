import unittest

from src.training.train_v17sel_d1_conservative_switch import preferred_actions


def action(rank: int, value: str, *, complete: bool = False, repair_090: bool = False,
           cost: int | None = None) -> dict:
    return {"rank": rank, "unique_order": True, "value_class": value,
            "complete": complete, "success": [True, True, True, repair_090, None],
            "oracle_legal_cumulative_tokens": cost}


class ConservativeSwitchTests(unittest.TestCase):
    def test_stay_when_no_strong_benefit(self):
        row = {"anchor": {"complete": False, "success": [True, True, True, False, None]},
               "actions": [action(1, "break"), action(2, "no_gain")]}
        self.assertEqual(preferred_actions(row), [0])

    def test_complete_and_090_repair_precedes_token_saving(self):
        row = {"anchor": {"complete": False, "success": [True, True, True, False, None]},
               "actions": [action(1, "repair", complete=True),
                           action(2, "repair", complete=True, repair_090=True),
                           action(3, "safe_token_saving", complete=True, cost=10)]}
        self.assertEqual(preferred_actions(row), [2])

    def test_cheapest_safe_saving_retains_ties(self):
        row = {"anchor": {"complete": True, "success": [True, True, True, True, None]},
               "actions": [action(1, "safe_token_saving", complete=True, cost=100),
                           action(2, "safe_token_saving", complete=True, cost=80),
                           action(3, "safe_token_saving", complete=True, cost=80)]}
        self.assertEqual(preferred_actions(row), [2, 3])


if __name__ == "__main__":
    unittest.main()
