"""VLCS routing and output checks, for explicit use on the experiment machine."""

import csv
from pathlib import Path
import tempfile
import unittest

from .datasets import EXPERIMENTS
from .run import DELAYS, parser, summarize, validate_inputs


class VLCSTests(unittest.TestCase):
    def test_defaults_select_caltech_source_weights(self):
        experiment = EXPERIMENTS["VLCS"]
        self.assertEqual(experiment.source, "Caltech101")
        self.assertEqual(experiment.targets, ("LabelMe", "SUN09", "VOC2007"))
        self.assertEqual(experiment.checkpoint_names, ("encoder_0.pth", "fc_0.pth"))
        self.assertEqual(experiment.checkpoint_dir.parts[-5], "VLCS")
        self.assertEqual(parser("VLCS").parse_args(["--gpus", "0", "1"]).dataset, "VLCS")
        self.assertEqual(parser().parse_args([]).dataset, "PACS")
        self.assertEqual(parser().parse_args(["--dataset", "VLCS"]).dataset, "VLCS")

    def test_supported_folders_and_wrong_source_checkpoint(self):
        experiment = EXPERIMENTS["VLCS"]
        for layout in experiment.layouts:
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                for domain in layout:
                    (root / "VLCS" / domain).mkdir(parents=True)
                checkpoint = root / "weights"
                checkpoint.mkdir()
                # A PACS checkpoint must not satisfy a VLCS run's requirements.
                for name in ("encoder_2.pth", "fc_2.pth"):
                    (checkpoint / name).touch()
                args = parser("VLCS").parse_args([
                    "--dataset-root", str(root), "--checkpoint-dir", str(checkpoint)])
                with self.assertRaises(FileNotFoundError):
                    validate_inputs(args)
                for name in experiment.checkpoint_names:
                    (checkpoint / name).touch()
                validate_inputs(args)
                (root / "VLCS" / "extra_domain").mkdir()
                with self.assertRaises(ValueError):
                    validate_inputs(args)

    def test_summary_uses_vlcs_domain_names(self):
        experiment = EXPERIMENTS["VLCS"]
        rows = []
        for delay in DELAYS:
            for seed, value in enumerate((60, 70, 80)):
                row = dict.fromkeys(experiment.metrics, value)
                row.update(seed=seed, delay=delay)
                rows.append(row)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            summarize(rows, output, dataset="VLCS")
            with (output / "summary.csv").open(newline="") as stream:
                summary = list(csv.DictReader(stream))
            self.assertEqual(len(summary), 4)
            self.assertEqual(float(summary[0]["LabelMe_accuracy_pct_mean"]), 70)
            self.assertEqual(float(summary[0]["VOC2007_accuracy_pct_std"]), 10)
            self.assertNotIn("Art_accuracy_pct_mean", summary[0])
            markdown = (output / "summary.md").read_text(encoding="utf-8")
            self.assertIn("# VLCS", markdown)
            self.assertIn("Caltech101 → LabelMe → SUN09 → VOC2007", markdown)


if __name__ == "__main__":
    unittest.main()
