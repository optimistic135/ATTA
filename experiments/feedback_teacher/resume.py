"""Validate completed runs and archive incomplete attempts before task-level resume."""
import json
from datetime import datetime, timezone

from experiments.delayed_feedback.run import dump_json
from experiments.sp_atta.reference import read_csv
from .spec import DEFAULTS, GROUPS, METRICS

REQUIRED_FILES = ("settings.json", "result.json", "batches.csv", "domains.csv",
    "feedback.csv", "gates.csv", "updates.csv", "budget.csv", "compute.json",
    "local_future.csv", "teacher_feedback.csv", "current_targets.csv")
MANIFEST_KEYS = ("dataset", "groups", "seeds", "delays", "batch_size", "reference_dir",
                 "reference_hashes", "defaults", "checkpoint_sha256", "dataset_root", "checkpoint_dir")


def completed_result(folder, group, seed, args, trace):
    """No final result or truncated outputs mean incomplete, not a completed task."""
    path = folder / "result.json"
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text())
    except (ValueError, UnicodeError):
        return None
    if not isinstance(result, dict):
        return None
    if (result.get("method"), result.get("seed"), result.get("delay")) != (group, seed, 100):
        raise ValueError(f"Completed task identity mismatch: {folder}")
    if any(not (folder / name).is_file() or not (folder / name).stat().st_size for name in REQUIRED_FILES):
        return None
    settings = json.loads((folder / "settings.json").read_text())
    expected = {"method": group, "seed": seed, "delay": 100, "dataset": "PACS", "batch_size": 32,
        "dataset_root": str(args.dataset_root), "checkpoint_dir": str(args.checkpoint_dir),
        "reference_dir": str(args.reference_dir), "reference_hashes": trace.hashes,
        "defaults": DEFAULTS, "simatta": trace.simatta, "train_batch_size": trace.train_batch_size}
    for key, value in expected.items():
        if settings.get(key) != value:
            raise ValueError(f"Completed task configuration mismatch at {key}: {folder}")
    required = set(METRICS) | {"formal_draws_sha256", "pseudo_pool_sha256", "reference_hashes"}
    if not required.issubset(result):
        return None
    if result["reference_hashes"] != trace.hashes:
        raise ValueError(f"Completed task reference mismatch: {folder}")
    if (result["main_updates"] != sum(trace.quotas)
            or result["student_updates"] + result["head_updates"] != result["main_updates"]):
        raise ValueError(f"Completed task budget mismatch: {folder}")
    batches = read_csv(folder / "batches.csv")
    if len(batches) != trace.steps:
        return None
    for actual, reference in zip(batches, trace.batches):
        if any(actual.get(key) != reference.get(key) for key in ("step", "domain", "samples", "new_queries", "new_returns")):
            raise ValueError(f"Completed task stream mismatch: {folder}")
    trace.verify_feedback(read_csv(folder / "feedback.csv"))
    if batches:
        for key in ("queried", "returned", "pending"):
            if result[key] != int(batches[-1][key]):
                raise ValueError(f"Completed task feedback count mismatch: {folder}")
    return result


def prepare_resume(output, manifest, args, references):
    """First validate everything, then archive only tasks selected for rerun."""
    previous = json.loads((output / "manifest.json").read_text())
    # JSON round-trip normalizes tuple-valued fields from the new in-memory manifest.
    normalized = json.loads(json.dumps(manifest))
    for key in MANIFEST_KEYS:
        if previous.get(key) != normalized[key]:
            raise ValueError(f"Cannot resume with different {key}; use the original configuration")
    rows, pending = [], []
    for group in GROUPS:
        for seed in args.seeds:
            folder = output / group / f"seed_{seed}"
            result = completed_result(folder, group, seed, args, references[seed])
            if result is None:
                pending.append((group, seed))
            else:
                rows.append(result)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    archive = output / "_resume_archive" / stamp
    archive.mkdir(parents=True, exist_ok=False)
    dump_json(archive / "manifest.json", previous)
    for name in ("per_seed.csv", "summary.csv", "summary.md", "paired.csv", "paired_summary.csv"):
        path = output / name
        if path.exists():
            path.rename(archive / name)
    for group, seed in pending:
        folder = output / group / f"seed_{seed}"
        if folder.exists():
            destination = archive / group / f"seed_{seed}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            folder.rename(destination)
        log = output / f"{group}_seed_{seed}.log"
        if log.exists():
            log.rename(archive / log.name)
    manifest["resume_history"] = previous.get("resume_history", []) + [{
        "at_utc": stamp, "archive": str(archive), "previous_git_commit": previous.get("git_commit"),
        "skipped_runs": len(rows), "pending_jobs": pending}]
    return rows, pending
