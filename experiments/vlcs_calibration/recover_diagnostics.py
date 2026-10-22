"""Recover saved diagnostics or explicitly partial records; never recreate model outputs."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil


VARIANTS = ("simatta_control", "calibration_only")
SCHEMAS = {
    "updates.csv": ("step", "iteration", "returned_memory", "replay_batch", "source_memory", "alpha",
                    "supervised_loss", "pseudo_loss", "gate_mean", "gradient_norm"),
    "calibration.csv": ("step", "anchor_index", "sample_id", "teacher_prediction", "label", "correct",
                        "cache_size", "query_step", "return_step", "query_domain", "return_domain"),
    "replay.csv": ("step", "returned_memory", "updates", "coverage", "effective_sample_size",
                   "need_mean", "relevance_mean", "score_mean"),
}
RUN = re.compile(r"(?:^|/)(simatta_control|calibration_only)/seed_(\d+)(?:/|$)")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recover(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir() or output == source or source.is_relative_to(output):
        raise ValueError("Choose a source directory and a separate new output directory")
    # Snapshot inventory before creating output; never recursively reuse previous
    # recovery directories as evidence for original observations.
    files = [path for path in source.rglob("*") if path.is_file()
             and not any(part in ("resume_history", "diagnostics_recovery") for part in path.relative_to(source).parts)
             and not (path.parent / "RECOVERY.json").is_file()
             and not path.is_relative_to(output)]
    candidates = {}
    identities = set()
    for path in files:
        match = RUN.search(path.relative_to(source).as_posix())
        if match:
            key = (match.group(1), int(match.group(2)))
            identities.add(key)
            candidates.setdefault((key, path.name), []).append(path)
    # Also support a flat export containing only root summaries and logs.
    summaries = [path for path in files if path.name == "per_seed.csv" and path.parent == source]
    summary_rows = read_csv(summaries[0]) if summaries else []
    for row in summary_rows:
        if row.get("variant") in VARIANTS:
            identities.add((row["variant"], int(row["seed"])))
    for path in files:
        match = re.fullmatch(r"(calibration_only|simatta_control)_seed_(\d+)\.log", path.name)
        if match:
            identities.add((match.group(1), int(match.group(2))))
    if not identities:
        raise ValueError("No calibration_only/simatta_control run directories, per_seed.csv or logs found")

    def find(key, name):
        found = candidates.get((key, name), [])
        if not found:
            return None
        if len({sha256(path) for path in found}) != 1:
            raise ValueError(f"Conflicting source files for {key}/{name}; use one experiment root")
        return found[0]

    # Resolve conflicts before writing anything.
    for key in identities:
        for name in (*SCHEMAS, "budget.csv", "batches.csv", "settings.json", "result.json", "feedback.csv"):
            find(key, name)
    output.mkdir(parents=True, exist_ok=False)
    inventory, domain_rows = [], []
    for key in sorted(identities):
        variant, seed = key
        folder = output / variant / f"seed_{seed}"
        folder.mkdir(parents=True)
        settings_path = find(key, "settings.json")
        settings = json.loads(settings_path.read_text()) if settings_path else {}
        if settings and (settings.get("variant") != variant or settings.get("seed") != seed
                         or settings.get("dataset") != "VLCS"):
            raise ValueError(f"Wrong run identity: {settings_path}")
        if settings.get("diagnostic_recovery"):
            raise ValueError("Recovered exports cannot be treated as raw experiment evidence")
        budget_path = find(key, "budget.csv")
        budgets = read_csv(budget_path) if budget_path else []
        batches_path = find(key, "batches.csv")
        batches = read_csv(batches_path) if batches_path else []
        by_step = {int(row["step"]): row for row in batches}
        update_path = find(key, "updates.csv")
        raw_updates = read_csv(update_path) if update_path else []
        counts = Counter(int(row["step"]) for row in raw_updates if row.get("step") not in (None, ""))
        if budgets:
            budget_counts = {int(row["step"]): int(row["slow_actual"]) for row in budgets}
            if update_path and dict(counts) != {step: n for step, n in budget_counts.items() if n}:
                raise ValueError(f"Update/budget disagreement for {key}")
            counts = Counter(budget_counts)
        for name, fields in SCHEMAS.items():
            original = find(key, name)
            record = {"variant": variant, "seed": seed, "file": name, "status": "unrecoverable",
                      "source": "", "source_sha256": "", "rows": 0, "known_fields": "",
                      "missing_fields": ",".join(fields), "reason": "Not recorded in available artifacts"}
            if original:
                saved_rows = read_csv(original)
                with original.open(newline="", encoding="utf-8-sig") as stream:
                    header = next(csv.reader(stream), [])
                if not set(fields) <= set(header):
                    raise ValueError(f"Unexpected diagnostic schema: {original}")
                shutil.copyfile(original, folder / name)
                record.update(status="copied_original", source=str(original), source_sha256=sha256(original),
                              rows=len(saved_rows), known_fields=",".join(header), missing_fields="",
                              reason="Copied saved raw observations; no model outputs reconstructed")
            else:
                recovered = []
                known = []
                if name == "updates.csv" and budgets:
                    known = ["step", "iteration", "returned_memory", "replay_batch"]
                    for step in sorted(counts):
                        batch = by_step.get(step, {})
                        for iteration in range(counts[step]):
                            recovered.append({"step": step, "iteration": iteration,
                                              "returned_memory": batch.get("returned", ""),
                                              "replay_batch": settings.get("train_batch_size", "")})
                elif name == "replay.csv" and (budgets or update_path):
                    known = ["step", "returned_memory", "updates"]
                    for step, count in sorted(counts.items()):
                        if count:
                            recovered.append({"step": step, "returned_memory": by_step.get(step, {}).get("returned", ""),
                                              "updates": count})
                # Feedback IDs/times do not contain teacher predictions, labels
                # or features. Never substitute student predictions for teacher.
                write_csv(folder / name, recovered, fields)
                if recovered:
                    known = [field for field in fields
                             if any(row.get(field) not in (None, "") for row in recovered)]
                    evidence = [path for path in (budget_path, batches_path, settings_path, update_path) if path]
                    record.update(status="partial_derived", source=";".join(str(path) for path in evidence),
                                  source_sha256=";".join(sha256(path) for path in evidence), rows=len(recovered),
                                  known_fields=",".join(known), missing_fields=",".join(field for field in fields if field not in known),
                                  reason="Only schedule/memory-count metadata derived; empty cells mean unavailable, never zero")
            inventory.append(record)
        # Domain accuracy is recoverable from actual per-batch sample counts,
        # not from guessed dataset sizes or cumulative progress percentages.
        for domain in dict.fromkeys(row["domain"] for row in batches):
            group = [row for row in batches if row["domain"] == domain]
            n, correct = sum(int(row["samples"]) for row in group), sum(int(row["correct"]) for row in group)
            domain_rows.append({"variant": variant, "seed": seed, "domain": domain,
                                "samples": n, "correct": correct, "accuracy_pct": 100 * correct / n})
        (folder / "RECOVERY.json").write_text(json.dumps({
            "diagnostic_recovery": True, "variant": variant, "seed": seed,
            "warning": "Only copied_original files are raw diagnostics. Partial/empty files cannot establish gate behavior.",
        }, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "availability.csv", inventory, list(inventory[0]))
    if domain_rows:
        write_csv(output / "domain_accuracy.csv", domain_rows, list(domain_rows[0]))
    lines = ["# Existing VLCS diagnostics: recovery inventory", "",
             "No model was loaded, no training or inference was run. Input files were not modified.",
             "copied_original = saved raw records; partial_derived = schedule metadata only;",
             "unrecoverable = header-only placeholder, not evidence of zero events.", "",
             "| Variant | Seed | File | Status | Rows |", "| --- | ---: | --- | --- | ---: |"]
    for row in inventory:
        lines.append(f"| {row['variant']} | {row['seed']} | {row['file']} | {row['status']} | {row['rows']} |")
    lines += ["", "## What cannot be reconstructed", "",
              "Progress logs and per_seed.csv do not contain per-SGD losses/gates, teacher predictions,",
              "teacher features or replay draws. They cannot recreate gate_mean, gradient_norm, teacher",
              "correctness events, need/relevance scores, sampled coverage or actual loss values.",
              "feedback.csv alone supplies query/return metadata, not teacher predictions or true classes.",
              "No missing values were replaced by 0, and control gate=1 was not fabricated as an observation.", "",
              "The controlled VLCS implementation normally writes these files under variant/seed_*/.",
              "First search the original server output tree or backups. Native SimATTA writes updates.csv",
              "but does not generate FCRA calibration/replay diagnostics.",
              "If raw files are genuinely absent, the missing mechanism statistics require a future instrumented run.",
              "Recovered exports are separate artifacts; do not use them as experiment completion/resume files."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return inventory


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--results-dir", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path)
    args = cli.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output_dir or args.results_dir / "diagnostics_recovery" / stamp
    recover(args.results_dir, output)
    print(f"Recovery report: {output.resolve() / 'report.md'}")


if __name__ == "__main__":
    main()
