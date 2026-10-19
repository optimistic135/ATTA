"""Aggregation checks; run explicitly on the experiment machine."""

import csv
from pathlib import Path
import tempfile
import unittest

from .run import DELAYS, METRICS, summarize


class SummaryTests(unittest.TestCase):
    def rows(self):
        rows = []
        for delay in DELAYS:
            for seed, value in enumerate((60, 70, 80)):
                row = dict.fromkeys(METRICS, value)
                row.update(seed=seed, delay=delay)
                rows.append(row)
        return rows

    def test_mean_and_sample_std(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            summarize(self.rows(), path)
            with (path / "summary.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual([int(row["delay"]) for row in rows], [0, 20, 50, 100])
            self.assertIn("batch size=32", (path / "summary.md").read_text())
            self.assertEqual(float(rows[0]["Art_accuracy_pct_mean"]), 70)
            self.assertEqual(float(rows[0]["Art_accuracy_pct_std"]), 10)

    def test_no_return_ratio_stays_undefined(self):
        rows = self.rows()
        for row in rows:
            row["cross_domain_pct"] = None
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            summarize(rows, path)
            with (path / "summary.csv").open(newline="") as stream:
                result = next(csv.DictReader(stream))
            self.assertEqual(result["cross_domain_pct_n"], "0")
            self.assertEqual(result["cross_domain_pct_mean"], "")
            self.assertIn("N/A", (path / "summary.md").read_text())

    def test_incomplete_sweep_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                summarize(self.rows()[:-1], Path(folder))


if __name__ == "__main__":
    unittest.main()
