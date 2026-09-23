import importlib.util
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

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


class Sam31InvocationTests(unittest.TestCase):
    def test_inner_cli_uses_the_current_python_environment(self):
        args = Namespace(
            prompts=["car"],
            prompts_file=None,
            confidence=0.5,
            batch_size=1,
            devices="all",
            checkpoint=None,
            token=None,
            resume=False,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tile_dir = root / "tiles"
            tile_dir.mkdir()
            raw_out = root / "output"

            def complete_run(command, check):
                self.assertTrue(check)
                coco = raw_out / "tiles" / "annotations" / "instances.json"
                coco.parent.mkdir(parents=True)
                coco.write_text("{}", encoding="utf-8")

            with mock.patch.object(
                sam31_tiled.subprocess,
                "run",
                side_effect=complete_run,
            ) as run:
                result = sam31_tiled.run_sam31(args, tile_dir, raw_out)

            command = run.call_args.args[0]
            self.assertEqual(command[:4], [sys.executable, "-m", "sam31.cli", "run"])
            self.assertEqual(result, raw_out / "tiles" / "annotations" / "instances.json")


if __name__ == "__main__":
    unittest.main()
