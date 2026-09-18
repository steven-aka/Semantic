from __future__ import annotations

import unittest

from src.evaluation.rank_projection_diagnostic import (
    never_reveal_tail_projection,
    stable_partial_order_projection,
)


class RankProjectionDiagnosticTests(unittest.TestCase):
    def test_stable_projection_obeys_preferences_and_preserves_ties(self) -> None:
        order = (0, 1, 2, 3)
        projected = stable_partial_order_projection(order, ((2, 1), (3, 1)))
        self.assertEqual(projected, (0, 2, 3, 1))

    def test_never_reveal_projection_is_stable(self) -> None:
        order = (2, 0, 3, 1)
        labels = (True, False, None, True)
        self.assertEqual(never_reveal_tail_projection(order, labels), (2, 1, 0, 3))


if __name__ == "__main__":
    unittest.main()
