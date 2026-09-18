import unittest

from src.evaluation.v17d1b_frozen_boundary_separability import stable_fold, strict_signature


class V17D1BTest(unittest.TestCase):
    def test_stable_fold_is_deterministic(self):
        self.assertEqual(stable_fold("query-a",3),stable_fold("query-a",3))

    def test_signature_includes_contract_and_pair_counts(self):
        row={"anchor_status":["active"]*5,"reached_level_count":0,"depth":2,"actions":[{"future_viability":[1,1,1,1,0]},{"future_viability":[1,1,1,0,0]}]}
        self.assertNotEqual(strict_signature(row),strict_signature({**row,"depth":3}))


if __name__=="__main__":unittest.main()
