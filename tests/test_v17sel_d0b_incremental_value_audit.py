import unittest

from src.evaluation.v17sel_d0b_incremental_value_audit import value_class, window_090


def candidate(success: list[bool], complete: bool, cost: int | None) -> dict:
    return {"success": success, "complete": complete,
            "oracle_ordered_complete_cumulative_tokens": cost}


class IncrementalValueAuditTests(unittest.TestCase):
    def setUp(self):
        self.active = {.6, .7, .8, .9}
        self.base = candidate([True, True, True, False, False], False, None)

    def test_repair_requires_no_anchor_or_complete_break(self):
        improved = candidate([True, True, True, True, False], False, None)
        mixed = candidate([False, True, True, True, False], False, None)
        self.assertEqual(value_class(self.base, improved, self.active), "repair")
        self.assertEqual(value_class(self.base, mixed, self.active), "mixed_break")

    def test_token_saving_uses_legal_cumulative_cost_only_when_complete(self):
        base = candidate([True, True, True, True, False], True, 300)
        cheap = candidate([True, True, True, True, False], True, 280)
        incomplete = candidate([True, True, True, True, False], False, None)
        self.assertEqual(value_class(base, cheap, self.active), "safe_token_saving")
        self.assertEqual(value_class(base, incomplete, self.active), "break")

    def test_window_reports_rollback_after_first_success(self):
        masks = list(range(13))
        fidelity = [0.0] * 13
        fidelity[4], fidelity[5], fidelity[6] = 0.9, 0.91, 0.85
        result = window_090(masks, fidelity)
        self.assertEqual(result["first_depth"], 4)
        self.assertEqual(result["first_run_length"], 2)
        self.assertEqual(result["successful_prefix_count"], 2)


if __name__ == "__main__":
    unittest.main()
