"""Continuous-need export and replay-alignment checks; run explicitly on server."""

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from .backfill_need import align_events
from .export_features import export_features


class ContinuousNeedTests(unittest.TestCase):
    def fixture(self, folder, save_probability=True):
        root = Path(folder)
        (root / "manifest.json").write_text(json.dumps({"delays": [0]}))
        domain = root / "D0"
        domain.mkdir()
        (domain / "result.json").write_text(json.dumps({"status": "complete"}))
        data = pd.DataFrame({
            "delay": [0, 0, 0], "sample_id": [10, 11, 12], "return_step": [1, 2, 3],
            "complete_window": [True, True, False], "utility": [0.01, -0.02, 0.03],
            "return_prediction": [0, 1, 0], "returned_label": [0, 0, 0],
            "prediction_js": [0.1, 0.2, 0.3], "gradient_reference_cosine": [0.2, -0.3, 0.4],
            "feature_batch_cosine": [0.4, 0.5, 0.6],
        })
        # Correct but uncertain is not the same thing as need=0.
        if save_probability:
            data["return_true_probability"] = [0.2, 0.1, 1.0]
        data.to_csv(domain / "utilities.csv", index=False)
        return root

    def test_need_uses_true_class_probability_not_binary_correctness(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.fixture(folder)
            result = export_features(root)
            self.assertEqual(result["sample_id"].tolist(), [10, 11])
            self.assertAlmostEqual(result["need"].iloc[0], 0.8)
            self.assertAlmostEqual(result["need"].iloc[1], 0.9)
            self.assertEqual(result["need_binary"].tolist(), [0, 1])
            self.assertEqual(result["utility"].tolist(), [0.01, -0.02])
            with self.assertRaises(FileExistsError):
                export_features(root)

    def test_old_results_require_probabilities_and_can_use_aligned_sidecar(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.fixture(folder, save_probability=False)
            with self.assertRaisesRegex(ValueError, "cannot be recovered"):
                export_features(root)
            pd.DataFrame({
                "delay": [0, 0], "sample_id": [11, 10], "return_step": [2, 1],
                "return_true_probability": [0.1, 0.2],
            }).to_csv(root / "need_backfill.csv", index=False)
            result = export_features(root)
            self.assertAlmostEqual(result.loc[result.sample_id == 10, "need"].iloc[0], 0.8)

    def test_invalid_probability_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.fixture(folder)
            path = root / "D0/utilities.csv"
            data = pd.read_csv(path)
            data.loc[0, "return_true_probability"] = 1.1
            data.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "invalid true-class probabilities"):
                export_features(root)

    def test_replay_must_match_original_events_and_observables(self):
        reference = pd.DataFrame({
            "delay": [20], "sample_id": [10], "return_step": [21], "query_step": [1],
            "anchor_index": [0], "returned_label": [0], "query_prediction": [1],
            "return_prediction": [0], "query_domain": ["Art"], "return_domain": ["Cartoon"],
            "prediction_changed": [1], "query_entropy": [0.7], "return_entropy": [0.5],
            "prediction_js": [0.2], "feature_temporal_cosine": [0.8],
            "feature_batch_cosine": [0.6], "gradient_temporal_cosine": [0.4],
            "gradient_reference_cosine": [-0.1],
        })
        replayed = reference.assign(return_true_probability=0.8, need=0.2, need_binary=0)
        self.assertEqual(len(align_events(reference, replayed)), 1)
        changed = replayed.copy()
        changed.loc[0, "return_entropy"] = 0.6
        with self.assertRaisesRegex(ValueError, "return_entropy"):
            align_events(reference, changed)
        changed = replayed.copy()
        changed.loc[0, "return_step"] = 22
        with self.assertRaisesRegex(ValueError, "events differ"):
            align_events(reference, changed)


if __name__ == "__main__":
    unittest.main()
