"""Static-only delivery: these tests are supplied but have not been executed."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from .recover_diagnostics import SCHEMAS, recover, write_csv


class RecoveryTests(unittest.TestCase):
    def test_existing_diagnostic_is_copied_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "experiment"
            folder = source / "calibration_only" / "seed_123"
            folder.mkdir(parents=True)
            path = folder / "updates.csv"
            write_csv(path, [{"step": 0, "iteration": 0, "gate_mean": 0.83}], SCHEMAS["updates.csv"])
            original = path.read_bytes()
            inventory = recover(source, root / "export")
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual((root / "export/calibration_only/seed_123/updates.csv").read_bytes(), original)
            row = next(row for row in inventory if row["file"] == "updates.csv")
            self.assertEqual(row["status"], "copied_original")

    def test_schedule_recovery_never_fabricates_gate_or_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "experiment"
            folder = source / "calibration_only/seed_456"
            folder.mkdir(parents=True)
            (folder / "settings.json").write_text(json.dumps({
                "variant": "calibration_only", "seed": 456, "dataset": "VLCS", "train_batch_size": 16}))
            write_csv(folder / "budget.csv", [{"step": 0, "slow_actual": 2}], ("step", "slow_actual"))
            write_csv(folder / "batches.csv", [{"step": 0, "returned": 10, "domain": "LabelMe",
                                               "samples": 32, "correct": 16}],
                      ("step", "returned", "domain", "samples", "correct"))
            inventory = recover(source, root / "export")
            with (root / "export/calibration_only/seed_456/updates.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["returned_memory"], "10")
            for row in rows:
                self.assertEqual(row["gate_mean"], "")
                self.assertEqual(row["supervised_loss"], "")
                self.assertEqual(row["pseudo_loss"], "")
            calibration = next(row for row in inventory if row["file"] == "calibration.csv")
            self.assertEqual(calibration["status"], "unrecoverable")
            self.assertEqual(calibration["rows"], 0)

    def test_flat_summary_can_only_generate_explicit_missing_placeholders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "experiment"
            source.mkdir()
            write_csv(source / "per_seed.csv", [{"variant": "simatta_control", "seed": 110}], ("variant", "seed"))
            inventory = recover(source, root / "export")
            self.assertTrue(all(row["status"] == "unrecoverable" for row in inventory))
            self.assertTrue(all(row["rows"] == 0 for row in inventory))

    def test_export_does_not_overwrite_original(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            with self.assertRaises(ValueError):
                recover(source, source)


if __name__ == "__main__":
    unittest.main()
