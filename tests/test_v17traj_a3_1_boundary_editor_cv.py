import unittest

from src.training.train_v17traj_a3_1_boundary_editor_cv import (
    canonical_choices,
    training_label,
)
from src.evaluation.v17traj_a2_stop_aligned_local_oracle import path


class BoundaryEditorTests(unittest.TestCase):
    def test_canonical_choices_are_distinct_outcomes(self):
        base = list(range(12))
        choices = canonical_choices(base)
        self.assertEqual(len(choices), 4)
        self.assertEqual(choices[0], tuple(base))
        self.assertEqual(len({path(order)[10] for order in choices}), 4)

    def test_quality_break_is_negative_even_when_tokens_fall(self):
        baseline = ((True, True, False, False, None), 100)
        self.assertEqual(training_label(((False, True, False, False, None), 90), baseline), -1)
        self.assertEqual(training_label(((True, True, True, False, None), 100), baseline), 1)
        self.assertEqual(training_label(((True, True, True, False, None), 110), baseline), 0)


if __name__ == "__main__":
    unittest.main()
