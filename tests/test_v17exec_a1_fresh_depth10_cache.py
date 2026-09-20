import unittest

from src.evaluation.v17exec_a1_fresh_depth10_cache import choose, score


class FreshDepth10ContractTest(unittest.TestCase):
    def test_unattainable_level_is_masked_and_quality_break_is_forbidden(self):
        active = {0.6, 0.7, 0.8, 0.9}
        stay = score({"f1": 0.8, "tokens": 100, "full_tokens": 200}, active)
        cheaper_break = score({"f1": 0.6, "tokens": 50, "full_tokens": 200}, active)
        quality_repair = score({"f1": 0.9, "tokens": 95, "full_tokens": 200}, active)
        for mask, item in enumerate((stay, cheaper_break, quality_repair)):
            item["mask"] = mask
        self.assertIsNone(stay["hits"][4])
        self.assertEqual(choose([stay, cheaper_break, quality_repair])["mask"], 2)

    def test_oracle_never_spends_more_tokens_for_repair(self):
        stay = {"mask": 1, "hits": [True, True, True, False, None], "tokens": 100}
        costly = {"mask": 2, "hits": [True, True, True, True, None], "tokens": 101}
        self.assertEqual(choose([stay, costly])["mask"], 1)


if __name__ == "__main__":
    unittest.main()
