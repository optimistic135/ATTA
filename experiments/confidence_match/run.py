"""Run only confidence_match; reuse completed VLCS baseline result/trace files."""

import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone

from experiments.vlcs_calibration import run as vlcs
from experiments.vlcs_calibration.reference import NativeTrace
from experiments.vlcs_calibration.results import load_completed, write_csv
from experiments.delayed_feedback.run import dump_json, sha256, validate_inputs
from experiments.delayed_feedback.progress import ProgressDisplay, emit_progress, parse_progress
from experiments.delayed_feedback.scheduler import run_jobs


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset-root", type=Path, default=vlcs.ROOT / "storage/datasets")
    cli.add_argument("--checkpoint-dir", type=Path, default=vlcs.EXPERIMENT.checkpoint_dir)
    cli.add_argument("--reference-dir", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path)
    cli.add_argument("--entropy-gap", type=float, default=0.05)
    devices = cli.add_mutually_exclusive_group()
    devices.add_argument("--gpus", type=int, nargs="+")
    devices.add_argument("--gpu-idx", type=int)
    cli.add_argument("--num-workers", type=int, default=0)
    cli.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    cli.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    return cli


def main():
    args = parser().parse_args()
    args.dataset, args.variant = "VLCS", "confidence_match"
    for key in ("dataset_root", "checkpoint_dir", "reference_dir", "output_dir"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    if not math.isfinite(args.entropy_gap) or not 0 < args.entropy_gap <= 1:
        raise ValueError("entropy-gap must be in (0,1]")
    validate_inputs(args)
    if args.worker:
        from .algorithm import ConfidenceMatch
        native = args.reference_dir / "simatta_original" / f"seed_{args.seed}"
        parent = json.loads((args.output_dir.parent.parent / "manifest.json").read_text())
        if (parent["entropy_gap"] != args.entropy_gap
                or NativeTrace(native, args.seed).hashes != parent["native_hashes"][str(args.seed)]):
            raise ValueError("Reference or matching rule changed after preflight")
        vlcs.worker(args, native_folder=native,
                    algorithm_factory=lambda config, trace: ConfidenceMatch(config, trace, args.entropy_gap),
                    settings_extra={"method": "VLCS-confidence-match", "teacher_calibration": True,
                                    "entropy_gap": args.entropy_gap,
                                    "state_definition": "frozen-teacher entropy / log(number of classes)"})
        return
    if args.seed is not None:
        raise ValueError("seed is worker-only")
    reference = json.loads((args.reference_dir / "manifest.json").read_text())
    if (reference.get("method") != "VLCS-calibration" or reference.get("status") != "complete"
            or reference.get("dataset") != "VLCS" or reference.get("delay") != 0
            or reference.get("batch_size") != 32 or len(reference["seeds"]) != 3
            or len(set(reference["seeds"])) != 3
            or Path(reference["dataset_root"]).resolve() != args.dataset_root
            or reference["num_workers"] != args.num_workers):
        raise ValueError("Use the completed VLCS three-group experiment with matching data/workers")
    for name in vlcs.EXPERIMENT.checkpoint_names:
        if sha256(args.checkpoint_dir / name) != reference["checkpoint_sha256"][name]:
            raise ValueError("Source checkpoint differs from completed experiment")
    seeds, hashes, existing = reference["seeds"], {}, {}
    for seed in seeds:
        trace = NativeTrace(args.reference_dir / "simatta_original" / f"seed_{seed}", seed)
        hashes[str(seed)] = trace.hashes
        for mode in vlcs.VARIANTS:
            result = load_completed(args.reference_dir / mode / f"seed_{seed}", mode, seed, reference, trace)
            if result is None:
                raise ValueError(f"Missing completed baseline: {mode}, {seed}; no baseline will be rerun")
            existing[mode, seed] = result
    gpus = args.gpus if args.gpus is not None else [args.gpu_idx if args.gpu_idx is not None else 0]
    if len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("Choose distinct nonnegative GPUs")
    output = args.output_dir or vlcs.ROOT / "storage/vlcs_confidence_match" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"method": "VLCS-confidence-match", "status": "running", "dataset": "VLCS",
                "delay": 0, "seeds": seeds, "entropy_gap": args.entropy_gap,
                "checkpoint_sha256": reference["checkpoint_sha256"], "native_hashes": hashes,
                "reference_dir": str(args.reference_dir), "reference_manifest_sha256": sha256(args.reference_dir / "manifest.json"),
                "existing_results_sha256": {f"{mode}/seed_{seed}": sha256(args.reference_dir / mode / f"seed_{seed}" / "result.json")
                                            for mode, seed in existing},
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=vlcs.ROOT, text=True).strip()}
    dump_json(output / "manifest.json", manifest)
    results = []

    def launch(seed, gpu):
        folder = output / "confidence_match" / f"seed_{seed}"
        command = [sys.executable, "-m", "experiments.confidence_match.run", "--worker", "--seed", str(seed),
                   "--gpu-idx", str(gpu), "--entropy-gap", str(args.entropy_gap),
                   "--dataset-root", str(args.dataset_root), "--checkpoint-dir", str(args.checkpoint_dir),
                   "--reference-dir", str(args.reference_dir), "--output-dir", str(folder),
                   "--num-workers", str(args.num_workers)]
        progress.start(gpu, 0, seed)
        with (output / f"confidence_match_seed_{seed}.log").open("w", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=vlcs.ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    event = parse_progress(line)
                    if event is not None:
                        progress.update(gpu, event)
        if process.returncode:
            raise RuntimeError(f"New variant failed for seed {seed}; existing baselines were not rerun")
        result = json.loads((folder / "result.json").read_text())
        if (result.get("method") != "VLCS-confidence-match" or result["seed"] != seed
                or result.get("entropy_gap") != args.entropy_gap):
            raise ValueError("Wrong worker result")
        for key in ("queried", "returned", "slow_updates"):
            if result[key] != existing["calibration_only", seed][key]:
                raise ValueError(f"Budget mismatch for seed {seed}: {key}")
        return result

    def save(result):
        results.append(result)
        write_csv(output / "per_seed.csv", results)
        progress.finish(result)

    try:
        with ProgressDisplay(gpus, 3) as progress:
            run_jobs(seeds, gpus, launch, save)
        for (mode, seed), _ in existing.items():
            if sha256(args.reference_dir / mode / f"seed_{seed}" / "result.json") != manifest["existing_results_sha256"][f"{mode}/seed_{seed}"]:
                raise ValueError("Existing baseline changed during experiment")
        comparisons = []
        for mode in vlcs.VARIANTS:
            for row in results:
                comparisons.append({"seed": row["seed"], "baseline": mode, **{
                    f"{key}_delta_pp": row[key] - existing[mode, row["seed"]][key]
                    for key in vlcs.ACCURACY}})
        write_csv(output / "paired_deltas.csv", comparisons)
        lines = ["# VLCS: teacher confidence-state matching", "",
                 f"Only the new variant ran; three existing groups reused. Entropy gap={args.entropy_gap}.", "",
                 "| Variant | LabelMe | SUN09 | VOC2007 | Stream | Macro |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for mode in (*vlcs.VARIANTS, "confidence_match"):
            rows = results if mode == "confidence_match" else [existing[mode, seed] for seed in seeds]
            cells = [f"{statistics.mean(row[key] for row in rows):.2f} ± {statistics.stdev(row[key] for row in rows):.2f}" for key in vlcs.ACCURACY]
            lines.append(f"| {mode} | " + " | ".join(cells) + " |")
        lines += ["", "| Compared with | Paired Stream delta (pp) | Positive seeds |", "| --- | ---: | ---: |"]
        for mode in vlcs.VARIANTS:
            values = [row["stream_accuracy_pct_delta_pp"] for row in comparisons if row["baseline"] == mode]
            lines.append(f"| {mode} | {statistics.mean(values):+.3f} ± {statistics.stdev(values):.3f} | {sum(x > 0 for x in values)}/3 |")
        lines += ["", "Matching filters feedback before the same top-k and unchanged gate formula.",
                  "No matching feedback means weight=1. No hidden truth, random audit, replay change or adapter.",
                  "Performance alone cannot prove reduced harm to correct pseudo-labels; sample-level correctness is not collected here.",
                  "Three development seeds do not establish significance or confirm novelty."]
        (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except BaseException:
        manifest.update(status="failed", completed_runs=len(results))
        dump_json(output / "manifest.json", manifest)
        raise
    manifest.update(status="complete", completed_runs=3)
    dump_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
