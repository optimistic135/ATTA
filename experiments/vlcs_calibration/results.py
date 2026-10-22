"""Result persistence independent of model code; heterogeneous columns are retained."""

import csv
import json
import math
from pathlib import Path


ACCURACY = ("LabelMe_accuracy_pct", "SUN09_accuracy_pct", "VOC2007_accuracy_pct",
            "stream_accuracy_pct", "domain_macro_accuracy_pct")
COSTS = ("queried", "returned", "slow_updates")


def write_csv(path, rows):
    """Use the union of keys and replace atomically; never truncate a good CSV."""
    if not rows:
        return
    path = Path(path)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_completed(folder, mode, seed, manifest, trace=None):
    """Only a valid final result is reusable; a log/partial CSV is not completion."""
    folder = Path(folder)
    if not (folder / "result.json").is_file():
        return None
    result = json.loads((folder / "result.json").read_text())
    settings = json.loads((folder / "settings.json").read_text())
    for record in (result, settings):
        if (record.get("method") != "VLCS-calibration" or record.get("dataset") != "VLCS"
                or record.get("variant") != mode or record.get("seed") != seed
                or record.get("delay") != 0):
            raise ValueError(f"Completed run identity mismatch: {folder}")
    if (settings.get("batch_size") != 32
            or settings.get("checkpoint_sha256") != manifest["checkpoint_sha256"]
            or Path(settings["dataset_root"]).resolve() != Path(manifest["dataset_root"]).resolve()
            or settings.get("num_workers") != manifest["num_workers"]
            or settings.get("teacher_calibration") != (mode == "calibration_only")
            or settings.get("targeted_replay") is not False
            or settings.get("fast_adapter_enabled") is not False
            or result.get("fast_updates") != 0):
        raise ValueError(f"Completed run configuration mismatch: {folder}")
    for metric in ACCURACY:
        value = result[metric]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"Invalid accuracy: {folder}: {metric}")
    for metric in COSTS:
        value = result[metric]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Invalid counter: {folder}: {metric}")
    if result["queried"] != result["returned"] or result.get("pending") != 0:
        raise ValueError(f"Invalid D=0 feedback totals: {folder}")
    if any(not (folder / name).is_file() for name in (
            "batches.csv", "domains.csv", "feedback.csv", "updates.csv")):
        raise ValueError(f"Completed run lacks diagnostic files: {folder}")
    if mode != "simatta_original":
        if trace is None:
            raise ValueError("A completed controlled run needs its completed native reference")
        if (settings.get("reference_hashes") != trace.hashes
                or settings["simatta"] != trace.settings["simatta"]
                or settings["train_batch_size"] != trace.settings["train_batch_size"]
                or settings["fcra"] != trace.settings["fcra"]
                or result.get("reference_updates") != trace.total
                or result["slow_updates"] != trace.total
                or result["queried"] != len(trace.records)):
            raise ValueError(f"Completed run native-reference mismatch: {folder}")
    return result
