import unittest

from src.evaluation.v17act_c0_depth10_omission_audit import FULL, PAIRS, choose, scored


class OmissionAuditTests(unittest.TestCase):
    def test_pair_universe_and_cost_bounded_choice(self):
        self.assertEqual(len(PAIRS), 66)
        baseline_mask = FULL ^ 3
        alternate_mask = FULL ^ 5
        fidelity = [0.0] * 4096
        tokens = [mask.bit_count() for mask in range(4096)]
        fidelity[baseline_mask] = .9
        fidelity[alternate_mask] = .95
        active = {.6, .7, .8, .9, .95}
        baseline = scored(baseline_mask, fidelity, tokens, active)
        selected = choose([baseline, scored(alternate_mask, fidelity, tokens, active)], baseline)
        self.assertEqual(selected["mask"], alternate_mask)
        self.assertTrue(selected["complete"])


if __name__ == "__main__":
    unittest.main()
