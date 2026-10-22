"""Regression coverage for mixed result schemas and safe completion-marker reuse.

Written for server execution; not executed during the static-only change.
"""

import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .results import ACCURACY, load_completed, write_csv


class ResultPersistenceTests(unittest.TestCase):
    def test_native_and_controlled_fields_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "per_seed.csv"
            write_csv(path, [{"variant": "simatta_original", "slow_updates": 10},
                             {"variant": "calibration_only", "slow_updates": 10,
                              "LabelMe_first10_accuracy_pct": 62.5, "slow_backward_calls": 20}])
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["slow_backward_calls"], "")
            self.assertEqual(rows[1]["slow_backward_calls"], "20")
            self.assertEqual(rows[1]["LabelMe_first10_accuracy_pct"], "62.5")

    def test_failed_write_keeps_previous_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "per_seed.csv"
            path.write_text("previous result\n")
            with patch("csv.DictWriter.writerows", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    write_csv(path, [{"variant": "simatta_original"}])
            self.assertEqual(path.read_text(), "previous result\n")
            self.assertFalse(path.with_name("per_seed.csv.tmp").exists())

    def make_completed(self, root, mode="simatta_original"):
        folder = root / mode / "seed_110"
        folder.mkdir(parents=True)
        identity = {"method": "VLCS-calibration", "dataset": "VLCS", "variant": mode,
                    "seed": 110, "delay": 0}
        manifest = {"checkpoint_sha256": {"encoder": "same"}, "dataset_root": str(root), "num_workers": 0}
        settings = {**identity, **manifest, "batch_size": 32, "fast_adapter_enabled": False,
                    "targeted_replay": False, "teacher_calibration": mode == "calibration_only",
                    "reference_hashes": {"trace": "same"}, "simatta": {"cold_start": 100},
                    "fcra": {"gate_prior": 4}, "train_batch_size": 16}
        result = {**identity, **{key: 80.0 for key in ACCURACY}, "queried": 2, "returned": 2,
                  "pending": 0, "slow_updates": 10, "fast_updates": 0, "reference_updates": 10}
        (folder / "settings.json").write_text(json.dumps(settings))
        (folder / "result.json").write_text(json.dumps(result))
        for name in ("batches.csv", "domains.csv", "feedback.csv", "updates.csv"):
            (folder / name).write_text("header\n")
        return folder, manifest, settings

    def test_completed_result_is_reused_without_relying_on_aggregate_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, manifest, _ = self.make_completed(Path(directory))
            (Path(directory) / "per_seed.csv").write_text("truncated,")
            row = load_completed(folder, "simatta_original", 110, manifest)
            self.assertEqual(row["slow_updates"], 10)

    def test_missing_result_is_not_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "settings.json").write_text("{}")
            self.assertIsNone(load_completed(folder, "simatta_original", 110, {}))

    def test_mismatched_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, manifest, _ = self.make_completed(Path(directory))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                load_completed(folder, "simatta_original", 123, manifest)

    def test_changed_native_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, manifest, settings = self.make_completed(Path(directory), "calibration_only")
            trace = SimpleNamespace(hashes={"trace": "changed"}, settings=settings, total=10,
                                    records=[{}, {}])
            with self.assertRaisesRegex(ValueError, "native-reference mismatch"):
                load_completed(folder, "calibration_only", 110, manifest, trace)


if __name__ == "__main__":
    unittest.main()
