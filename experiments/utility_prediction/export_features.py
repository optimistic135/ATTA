"""Export event features with continuous need=1-p(true class), without model runs."""

import argparse
import json
from pathlib import Path


KEYS = ["delay", "sample_id", "return_step"]


def export_features(output_dir, output_file=None, need_features=None):
    import numpy as np
    import pandas as pd

    root = Path(output_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    destination = Path(output_file) if output_file else root / "utility_features_continuous.csv"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}; choose --output-file")
    frames = []
    for delay in manifest["delays"]:
        folder = root / f"D{delay}"
        if json.loads((folder / "result.json").read_text())["status"] != "complete":
            raise ValueError(f"Incomplete experiment: {folder}")
        frame = pd.read_csv(folder / "utilities.csv")
        if not frame["delay"].eq(delay).all():
            raise ValueError(f"Delay mismatch in {folder}")
        complete = frame["complete_window"].astype(str).str.lower().eq("true")
        frames.append(frame.loc[complete & frame["utility"].notna()].copy())
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(KEYS).any():
        raise ValueError("Duplicate event identities in utility files")
    if "return_true_probability" not in data or data["return_true_probability"].isna().any():
        sidecar = Path(need_features) if need_features else root / "need_backfill.csv"
        if not sidecar.is_file():
            raise ValueError(
                "True-class probabilities were not saved. They cannot be recovered from "
                "argmax or entropy. Run experiments.utility_prediction.backfill_need first.")
        extra = pd.read_csv(sidecar)
        extra = extra[KEYS + ["return_true_probability"]]
        if extra.duplicated(KEYS).any():
            raise ValueError(f"Duplicate event identities in {sidecar}")
        data = data.merge(extra, on=KEYS, how="left", validate="one_to_one",
                          suffixes=("", "_backfill"))
        if "return_true_probability_backfill" in data:
            recorded = data["return_true_probability"]
            replayed = data["return_true_probability_backfill"]
            both = recorded.notna() & replayed.notna()
            if not np.allclose(recorded[both], replayed[both], rtol=1e-5, atol=1e-7):
                raise ValueError("Recorded and backfilled true-class probabilities disagree")
            data["return_true_probability"] = recorded.fillna(replayed)
    probability = pd.to_numeric(data["return_true_probability"], errors="raise")
    if not np.isfinite(probability).all() or not probability.between(0, 1).all():
        raise ValueError("Missing/invalid true-class probabilities; all events need values in [0,1]")
    result = data[KEYS].copy()
    result["utility"] = data["utility"]  # Accuracy scale 0--1; multiply by 100 for pp.
    result["need"] = 1.0 - probability
    result["pred_drift"] = data["prediction_js"]
    result["grad_cos"] = data["gradient_reference_cosine"]
    result["feature_similarity"] = data["feature_batch_cosine"]
    result["return_true_probability"] = probability
    result["need_binary"] = (data["return_prediction"] != data["returned_label"]).astype(int)
    result.to_csv(destination, index=False)
    print(result.groupby("delay").size().rename("events"))
    print(f"Exported {len(result)} events: {destination}", flush=True)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", type=Path, required=True)
    cli.add_argument("--output-file", type=Path)
    cli.add_argument("--need-features", type=Path)
    args = cli.parse_args()
    export_features(args.output_dir, args.output_file, args.need_features)


if __name__ == "__main__":
    main()
