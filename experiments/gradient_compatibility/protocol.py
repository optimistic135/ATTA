"""Read-only reference validation and reusable completed-run checks."""

import csv
import json
import math
from pathlib import Path

from experiments.delayed_feedback.run import sha256
from experiments.vlcs_calibration import run as vlcs
from experiments.vlcs_calibration.reference import NativeTrace
from experiments.vlcs_calibration.results import load_completed


METHOD = "VLCS-gradient-compatibility"
VARIANTS = ("direction_check", "uniform_relaxation")


def validate_variant(folder, mode, seed, reference, trace, entropy_gap, method=METHOD, donor_hash=None):
    """A result.json is the completion marker; incomplete runs are never reused."""
    folder = Path(folder)
    if not (folder / "result.json").is_file():
        return None
    result = json.loads((folder / "result.json").read_text())
    settings = json.loads((folder / "settings.json").read_text())
    for record in (result, settings):
        if (record.get("method") != method or record.get("variant") != mode
                or record.get("seed") != seed or record.get("dataset") != "VLCS"
                or record.get("delay") != 0 or record.get("entropy_gap") != entropy_gap):
            raise ValueError(f"Completed identity mismatch: {folder}")
    if (settings.get("batch_size") != 32 or settings.get("num_workers") != reference["num_workers"]
            or settings.get("checkpoint_sha256") != reference["checkpoint_sha256"]
            or Path(settings["dataset_root"]).resolve() != Path(reference["dataset_root"]).resolve()
            or settings.get("simatta") != trace.settings["simatta"]
            or settings.get("fcra") != trace.settings["fcra"]
            or settings.get("train_batch_size") != trace.settings["train_batch_size"]
            or settings.get("reference_hashes") != trace.hashes
            or settings.get("teacher_calibration") is not True
            or settings.get("targeted_replay") is not False
            or settings.get("fast_adapter_enabled") is not False
            or result.get("fast_updates") != 0 or result.get("pending") != 0
            or result.get("queried") != len(trace.records) or result.get("returned") != len(trace.records)
            or result.get("reference_updates") != trace.total or result.get("slow_updates") != trace.total):
        raise ValueError(f"Completed protocol mismatch: {folder}")
    for key in vlcs.ACCURACY:
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"Invalid accuracy in {folder}: {key}")
    required = ["batches.csv", "domains.csv", "feedback.csv", "updates.csv", "budget.csv", "state_matching.csv"]
    if method == METHOD:
        required.append("direction.csv")
        if settings.get("donor_sha256") != donor_hash:
            raise ValueError(f"Completed run used a different donor: {folder}")
    if any(not (folder / name).is_file() for name in required):
        raise ValueError(f"Completed run lacks diagnostic files: {folder}")
    if method == METHOD:
        schedule = load_donor(folder / "direction.csv", trace)
        with (folder / "updates.csv").open(newline="", encoding="utf-8") as stream:
            expected = {(int(row["step"]), int(row["iteration"])) for row in csv.DictReader(stream)
                        if row.get("pseudo_loss", "") != ""}
        if set(schedule) != expected or result.get("pseudo_gate_updates") != len(schedule):
            raise ValueError(f"Incomplete pseudo-gate diagnostics: {folder}")
    return result


def load_references(args):
    """Require both old sweeps to be complete. Never launch an old variant."""
    native_manifest = args.reference_dir / "manifest.json"
    match_manifest = args.confidence_dir / "manifest.json"
    reference = json.loads(native_manifest.read_text())
    match = json.loads(match_manifest.read_text())
    if (reference.get("method") != "VLCS-calibration" or reference.get("status") != "complete"
            or reference.get("dataset") != "VLCS" or reference.get("delay") != 0
            or reference.get("batch_size") != 32 or len(reference["seeds"]) != 3
            or len(set(reference["seeds"])) != 3 or reference.get("num_workers") != args.num_workers
            or Path(reference["dataset_root"]).resolve() != args.dataset_root):
        raise ValueError("Expected completed VLCS D=0 three-group references with matching data/workers")
    if (match.get("method") != "VLCS-confidence-match" or match.get("status") != "complete"
            or match.get("seeds") != reference["seeds"]
            or match.get("checkpoint_sha256") != reference["checkpoint_sha256"]
            or Path(match["reference_dir"]).resolve() != args.reference_dir
            or match.get("reference_manifest_sha256") != sha256(native_manifest)):
        raise ValueError("Confidence-matching reference differs from the native reference")
    gap = match["entropy_gap"]
    if not math.isfinite(gap) or not 0 < gap <= 1:
        raise ValueError("Invalid reference entropy gap")
    for name in vlcs.EXPERIMENT.checkpoint_names:
        if sha256(args.checkpoint_dir / name) != reference["checkpoint_sha256"][name]:
            raise ValueError("Source weights differ from completed references")
    existing, traces = {}, {}
    paths = [native_manifest, match_manifest]
    for seed in reference["seeds"]:
        trace = NativeTrace(args.reference_dir / "simatta_original" / f"seed_{seed}", seed)
        traces[seed] = trace
        if trace.hashes != match["native_hashes"][str(seed)]:
            raise ValueError("Confidence run used a different query/update schedule")
        for mode in (*vlcs.VARIANTS, "confidence_match"):
            root = args.confidence_dir if mode == "confidence_match" else args.reference_dir
            folder = root / mode / f"seed_{seed}"
            result = (validate_variant(folder, mode, seed, reference, trace, gap, "VLCS-confidence-match")
                      if mode == "confidence_match" else load_completed(folder, mode, seed, reference, trace))
            if result is None:
                raise ValueError(f"Missing completed reference: {folder}; it will not be rerun")
            if mode != "confidence_match" and sha256(folder / "result.json") != match["existing_results_sha256"][f"{mode}/seed_{seed}"]:
                raise ValueError("Baseline changed after confidence experiment")
            existing[mode, seed] = result
            paths.extend(folder / name for name in ("result.json", "settings.json"))
        paths.extend(args.reference_dir / "simatta_original" / f"seed_{seed}" / name for name in trace.hashes)
    snapshots = {str(path): sha256(path) for path in paths}
    return reference, gap, existing, traces, snapshots


def verify_snapshots(snapshots):
    for path, expected in snapshots.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Reference changed: {path}")


def load_donor(path, trace):
    """Only per-update training strength/draws are read, never evaluation outcomes."""
    schedule, previous = {}, (-1, -1)
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["step"]), int(row["iteration"]))
            target = float(row["removed_weight_mean"])
            if (key <= previous or not 0 <= key[0] < trace.steps
                    or not 0 <= key[1] < trace.slow[key[0]] or int(row["pseudo_draws"]) <= 0
                    or not math.isfinite(target) or not 0 <= target <= 1):
                raise ValueError("Invalid donor training schedule")
            if any(len(row[name]) != 64 or any(c not in "0123456789abcdef" for c in row[name])
                   for name in ("supervised_draw_hash", "pseudo_draw_hash")):
                raise ValueError("Invalid donor draw identity")
            schedule[key], previous = row, key
    if not schedule:
        raise ValueError("Empty direction-check donor")
    return schedule
