import unittest

from src.evaluation.v17sel_d0_tail_opportunity_audit import outcome, summarize


def candidate(rank: int, success: bool = False, tokens: int | None = None) -> dict:
    return {"rank": rank, "success": [False, False, False, success, False],
            "earliest_tokens": [None, None, None, tokens, None], "complete": success}


class TailOpportunityAuditTests(unittest.TestCase):
    def test_one_adaptive_tail_can_reproduce_full_pool_oracle(self):
        row = {"full_tokens": 100, "candidates": [candidate(i) for i in range(11)]}
        row["candidates"][9] = candidate(9, True, 80)
        self.assertFalse(outcome(row, ())["success"])
        self.assertFalse(outcome(row, (5,))["success"])
        self.assertEqual(outcome(row, (9,)), outcome(row, tuple(range(5, 11))))

    def test_preserved_top4_pool_cannot_lose_oracle_success_or_cost(self):
        row = {"full_tokens": 100, "candidates": [candidate(i) for i in range(11)]}
        row["candidates"][2] = candidate(2, True, 80)
        row["candidates"][7] = candidate(7, True, 60)
        base = outcome(row, ())
        extended = outcome(row, (7,))
        self.assertTrue(base["success"] and extended["success"])
        self.assertEqual(base["earliest_090_tokens"], 80)
        self.assertEqual(extended["earliest_090_tokens"], 60)
        result = summarize([row], (7,))
        self.assertEqual(result["new_success_vs_top4"], 0)
        self.assertEqual(result["top4_success_with_token_saving"], 1)
        self.assertAlmostEqual(result["mean_failure_penalized_090_token_fraction"], 0.6)


if __name__ == "__main__":
    unittest.main()
