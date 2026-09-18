import unittest

from src.data.build_v13_internal_split import split_rows, stratum


def row(index: int, family: str, positives: int):
    return {
        "example_id": f"{index}__{family}__train",
        "class_population": {"4": positives, "5": 0},
    }


class V13InternalSplitTest(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_stratified(self):
        rows = [row(i, "simple", 2) for i in range(10)] + [row(i + 10, "table", 20) for i in range(10)]
        train, validation = split_rows(rows, validation_size=6, seed=7)
        train2, validation2 = split_rows(list(reversed(rows)), validation_size=6, seed=7)
        self.assertTrue({x["example_id"] for x in train}.isdisjoint(x["example_id"] for x in validation))
        self.assertEqual({x["example_id"] for x in validation}, {x["example_id"] for x in validation2})
        self.assertEqual({x["example_id"] for x in train}, {x["example_id"] for x in train2})
        self.assertEqual(sum(stratum(x) == ("simple", "le4") for x in validation), 3)
        self.assertEqual(sum(stratum(x) == ("table", "gt10") for x in validation), 3)
