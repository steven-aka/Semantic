import unittest

from src.evaluation.v17b0_recompute_order_baseline import summarize_orders


class V17B0BaselineTest(unittest.TestCase):
    def test_module_exports_summary_function(self):
        self.assertTrue(callable(summarize_orders))


if __name__ == "__main__": unittest.main()
