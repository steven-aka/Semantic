import unittest

from src.evaluation.critical_boundary_diagnostic import (
    derive_critical_boundary_preferences,
    linear_extension_contract_statistics,
)


class CriticalBoundaryDiagnosticTest(unittest.TestCase):
    def test_set_valued_boundary_variants_are_conservative(self) -> None:
        fidelity = {
            0b000: 0.0,
            0b001: 0.9,
            0b010: 0.0,
            0b011: 0.2,
            0b100: 0.0,
            0b101: 0.9,
            0b110: 0.0,
            0b111: 0.2,
        }
        result = derive_critical_boundary_preferences(
            fidelity, [[0b001]], [0.9], 3, [[0, 1], [0, 2]]
        )
        self.assertEqual(result["robust_membership"], [(0, 1), (0, 2)])
        self.assertEqual(result["stable_counterfactual"], [(0, 1)])
        self.assertEqual(result["conflicting_local_unordered_pairs"], 0)

    def test_linear_extension_success_is_counted_exactly(self) -> None:
        fidelity = {
            mask: float(bool(mask & 0b001) and not bool(mask & 0b010))
            for mask in range(8)
        }
        result = linear_extension_contract_statistics(fidelity, 3, [(0, 1)], 0.9)
        self.assertEqual(result["linear_extensions"], 3)
        self.assertEqual(result["successful_linear_extensions"], 3)
        self.assertTrue(result["all_extensions_successful"])

    def test_unconstrained_extensions_can_miss_the_feasible_prefix(self) -> None:
        fidelity = {
            mask: float(bool(mask & 0b001) and not bool(mask & 0b010))
            for mask in range(8)
        }
        result = linear_extension_contract_statistics(fidelity, 3, [], 0.9)
        self.assertEqual(result["linear_extensions"], 6)
        self.assertEqual(result["successful_linear_extensions"], 3)
        self.assertEqual(result["successful_linear_extension_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
