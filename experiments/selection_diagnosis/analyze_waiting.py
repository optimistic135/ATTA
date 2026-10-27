"""Offline elite-target precision before first feedback; no model or GPU needed."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from .analyze_batches import load_seed, sha256, dump_json, write_csv, cell, SOURCE_METHOD
from .offline_metrics import BATCH_FIELDS, FRACTION, aggregate, batch_rankings

METHOD = "SP-waiting-elite-offline-analysis"
VERSION = 1
FILES = ("per_batch.csv", "per_seed.csv", "summary.csv", "summary.md")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--input-dir", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path)
    cli.add_argument("--resume", action="store_true")
    args = cli.parse_args()
    root = args.input_dir.resolve()
    output = args.output_dir.resolve() if args.output_dir else root / "waiting_analysis"
    source = json.loads((root / "manifest.json").read_text())
    seeds = source["seeds"]
    if source.get("status") != "complete" or source.get("method") != SOURCE_METHOD or source.get("delays") != [100] or source.get("batch_size") != 32:
        raise ValueError("Use the completed selection_diagnosis D100 result root")
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Require all three distinct seeds")
    inputs = [root / "manifest.json"] + [root / f"seed_{seed}" / name for seed in seeds for name in
        ("settings.json", "result.json", "batches.csv", "score_observations.csv", "query_scores.csv", "feedback.csv", "shadow_updates.csv")]
    if any(output == p.resolve() or output in p.resolve().parents for p in inputs):
        raise ValueError("Output may not replace input files")
    hashes = {str(p.relative_to(root)): sha256(p) for p in inputs}
    config = {"method": METHOD, "version": VERSION, "fraction": FRACTION,
              "scope": "through first return batch inclusive; predictions precede return"}
    output.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output.parent / f".{output.name}.analysis.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another waiting analysis uses this output") from None
        if output.exists():
            if not args.resume:
                raise FileExistsError("Use --resume or a new --output-dir")
            previous = json.loads((output / "manifest.json").read_text())
            if previous.get("config") != config or previous.get("input_hashes") != hashes:
                raise ValueError("Different input/settings; use a new directory")
            if previous.get("status") == "complete":
                if any(not (output / name).is_file() or sha256(output / name) != previous.get("output_hashes", {}).get(name) for name in FILES):
                    raise ValueError("Completed reports changed; use a new directory")
                print(f"Already complete: {output / 'summary.md'}", flush=True)
                return
            archive = output.parent / "_waiting_analysis_archive" / (output.name + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
            archive.parent.mkdir(parents=True, exist_ok=True)
            output.rename(archive)
        output.mkdir(parents=True, exist_ok=False)
        manifest = {"status": "running", "method": METHOD, "input_hashes": hashes, "seeds": seeds,
                    "input_dir": str(root), "config": config, "python": sys.version,
                    "source_git_commit": source.get("git_commit")}
        dump_json(output / "manifest.json", manifest)
        per_batch, per_seed = [], []
        try:
            for seed in seeds:
                print(f"Reading waiting-phase CSVs: seed={seed}", flush=True)
                observations, _, batches, feedback, _, _ = load_seed(root, seed, source)
                returns = [r["return_step"] for r in feedback if r["return_step"] is not None]
                first_return = min(returns) if returns else None
                selected = {b["step"] for b in batches if first_return is None or b["step"] <= first_return}
                rows, summaries, _ = batch_rankings(seed, observations, batches, float(source["defaults"]["confidence"]), selected_steps=selected)
                per_batch.extend(rows)
                for row in summaries:
                    row["evaluated_batches"] = row.pop("warm_batches")
                    row["mean_selected_per_batch"] = row.pop("mean_selected_per_warm_batch")
                    row["first_feedback_step"] = first_return
                per_seed.extend(summaries)
            if any(sha256(p) != hashes[str(p.relative_to(root))] for p in inputs):
                raise ValueError("Input changed during analysis")
            summaries = aggregate(per_seed, ("pool", "domain", "period", "branch"))
            write_csv(output / "per_batch.csv", per_batch, BATCH_FIELDS)
            write_csv(output / "per_seed.csv", per_seed)
            write_csv(output / "summary.csv", summaries)
            text = "# Waiting-phase within-batch elite precision: PACS D=100\n\n"
            text += "| Domain | Period | Batches | Selected weight | Empty (%) | Source confidence acc (%) | Random expectation (%) |\n| :--- | :--- | ---: | ---: | ---: | ---: | ---: |\n"
            text += "\n".join(f"| {r['domain']} | {r['period']} | {cell(r, 'evaluated_batches')} | {cell(r, 'selected_weight')} | {cell(r, 'zero_candidate_pct')} | {cell(r, 'confidence_high_accuracy_pct')} | {cell(r, 'random_expected_accuracy_pct')} |" for r in summaries if r["pool"] == "current_batch" and r["branch"] == "source")
            text += ("\n\nIncludes the first return batch because prediction/target selection precede release. "
                "No warmup filter; thirds refer to the waiting part of each domain. "
                "Primary current_batch contains only new images; rolling_window is supplementary. "
                "Selection is top ceil(0.1N) after confidence>=0.9 and flip agreement, with fractional boundary ties. "
                "Actual coverage may exceed10%. Expected correctness is scored offline; no labels train models or set thresholds. "
                "Three-seed mean ± sample SD, valid n in CSV. Precision does not establish waiting-period adaptation gains.\n")
            (output / "summary.md").write_text(text, encoding="utf-8")
        except BaseException:
            manifest["status"] = "failed"
            dump_json(output / "manifest.json", manifest)
            raise
        manifest.update(status="complete", output_hashes={name: sha256(output / name) for name in FILES})
        dump_json(output / "manifest.json", manifest)
        print(text, flush=True)
        print(f"Complete: {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
