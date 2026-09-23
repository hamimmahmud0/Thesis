import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "sam31_tiled.py"
SPEC = importlib.util.spec_from_file_location("sam31_tiled", MODULE_PATH)
sam31_tiled = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = sam31_tiled
SPEC.loader.exec_module(sam31_tiled)


def instance(category_id, score, mask=None, x=0, y=0):
    if mask is None:
        mask = np.ones((10, 10), dtype=bool)
    return sam31_tiled.Instance(category_id, score, x, y, mask)


class CrossCategoryDeduplicationTests(unittest.TestCase):
    categories = {1: "Vehicle", 2: "Car", 3: "Bus", 4: "Pedestrian"}
    non_vehicle = {"pedestrian", "dog", "cat"}

    def suppress(self, instances, threshold=0.85):
        return sam31_tiled.suppress_cross_category_duplicates(
            instances,
            self.categories,
            threshold,
            self.non_vehicle,
        )

    def test_vehicle_subclass_wins_even_with_lower_score(self):
        kept, generic_removed, score_suppressed = self.suppress(
            [instance(1, 0.99), instance(2, 0.70)]
        )
        self.assertEqual([item.category_id for item in kept], [2])
        self.assertEqual(generic_removed, 1)
        self.assertEqual(score_suppressed, 0)

    def test_other_overlapping_classes_keep_highest_score(self):
        kept, generic_removed, score_suppressed = self.suppress(
            [instance(2, 0.70), instance(3, 0.92)]
        )
        self.assertEqual([item.category_id for item in kept], [3])
        self.assertEqual(generic_removed, 0)
        self.assertEqual(score_suppressed, 1)

    def test_non_vehicle_is_not_treated_as_vehicle_subclass(self):
        kept, generic_removed, score_suppressed = self.suppress(
            [instance(1, 0.95), instance(4, 0.70)]
        )
        self.assertEqual([item.category_id for item in kept], [1])
        self.assertEqual(generic_removed, 0)
        self.assertEqual(score_suppressed, 1)

    def test_instances_below_iou_threshold_are_kept(self):
        kept, generic_removed, score_suppressed = self.suppress(
            [instance(2, 0.9, x=0), instance(3, 0.8, x=5)]
        )
        self.assertEqual(len(kept), 2)
        self.assertEqual(generic_removed, 0)
        self.assertEqual(score_suppressed, 0)

    def test_non_vehicle_parser_is_case_insensitive_and_trims(self):
        self.assertEqual(
            sam31_tiled.parse_category_names(" Pedestrian;DOG | cat "),
            {"pedestrian", "dog", "cat"},
        )


if __name__ == "__main__":
    unittest.main()
