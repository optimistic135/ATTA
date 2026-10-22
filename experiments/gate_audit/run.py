"""Paired baseline/oracle/random gate diagnostics; never a benchmark leaderboard."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from experiments.fcra_ablation import run as ablation
from experiments.fcra_dual import run as dual
from experiments.fcra_dual.reference import ReferenceTrace
from experiments.delayed_feedback.run import dump_json, validate_inputs
from experiments.delayed_feedback.progress import ProgressDisplay, parse_progress
from experiments.delayed_feedback.scheduler import run_jobs


MODES = ("baseline", "oracle_correct", "random_count")


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset-root", type=Path, default=dual.ROOT / "storage/datasets")
    cli.add_argument("--checkpoint-dir", type=Path, default=dual.EXPERIMENT.checkpoint_dir)
    cli.add_argument("--reference-dir", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path)
    cli.add_argument("--audit-seeds", type=int, nargs="+", default=[123, 456])
    devices = cli.add_mutually_exclusive_group()
    devices.add_argument("--gpus", type=int, nargs="+")
    devices.add_argument("--gpu-idx", type=int)
    cli.add_argument("--num-workers", type=int, default=0)
    cli.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    cli.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    cli.add_argument("--variant", choices=MODES, help=argparse.SUPPRESS)
    return cli


def main():
    args = parser().parse_args()
    args.dataset, args.seeds = "PACS", None
    for key in ("dataset_root", "checkpoint_dir", "reference_dir", "output_dir"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    validate_inputs(args)
    if args.worker:
        from .algorithm import GateAudit
        ablation.worker(args, algorithm_factory=lambda config, trace: GateAudit(
            config, trace, args.variant, args.output_dir), settings_extra={
                "method": "GateAudit", "mode": args.variant,
                "diagnostic_only": args.variant != "baseline",
                "intervention_scope": "Sketch onward; earlier batches unchanged",
                "random_control": "matches corrected draw count, not restored weight mass",
                "diagnostic_rng_seed": args.seed + 89017 if args.seed is not None else None,
            })
        return
    if args.variant is not None or args.seed is not None:
        raise ValueError("variant/seed are worker-only")
    reference, all_seeds = dual.load_reference(args)
    if (len(set(args.audit_seeds)) != len(args.audit_seeds)
            or not args.audit_seeds or not set(args.audit_seeds) <= set(all_seeds)):
        raise ValueError("Audit seeds must be distinct members of the reference")
    gpus = args.gpus if args.gpus is not None else [args.gpu_idx if args.gpu_idx is not None else 0]
    if len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("Provide distinct nonnegative GPU indices")
    hashes = {str(seed): ReferenceTrace(args.reference_dir / "D0" / f"seed_{seed}", 0, seed).hashes
              for seed in args.audit_seeds}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output_dir or dual.ROOT / "storage/gate_audit" / stamp
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "method": "GateAudit", "status": "running", "dataset": "PACS", "delay": 0,
        "seeds": args.audit_seeds, "modes": MODES, "reference_hashes": hashes,
        "reference_manifest_sha256": ablation.sha256(args.reference_dir / "manifest.json"),
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "reference_dir": str(args.reference_dir), "dataset_root": str(args.dataset_root),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dual.ROOT, text=True).strip(),
        "diagnostic_warning": "Oracle and random-count arms consume hidden labels; not valid ATTA methods",
        "fork_protocol": "deterministic prefix replay; require identical model/memory/RNG fingerprints",
        "group_thresholds": {"neighbors": [0, 2, 8], "support": [0, 1, 3],
                             "cross_source_fraction": 0.5},
    }
    dump_json(output / "manifest.json", manifest)
    results = []

    def launch(job, gpu):
        mode, seed = job
        folder = output / mode / f"seed_{seed}"
        command = [sys.executable, "-m", "experiments.gate_audit.run", "--worker",
                   "--variant", mode, "--seed", str(seed), "--gpu-idx", str(gpu),
                   "--dataset-root", str(args.dataset_root), "--checkpoint-dir", str(args.checkpoint_dir),
                   "--reference-dir", str(args.reference_dir), "--output-dir", str(folder),
                   "--num-workers", str(args.num_workers)]
        progress.start(gpu, 0, seed)
        print(f"Gate audit mode={mode}, seed={seed}", flush=True)
        with (output / f"{mode}_seed_{seed}.log").open("w", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=dual.ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    event = parse_progress(line)
                    if event is not None:
                        progress.update(gpu, event)
        if process.returncode:
            raise RuntimeError(f"Diagnostic failed: mode={mode}, seed={seed}; see log")
        result = json.loads((folder / "result.json").read_text())
        if result.get("method") != "GateAudit" or result["mode"] != mode or result["seed"] != seed:
            raise ValueError("Worker identity mismatch")
        return result

    def save(result):
        results.append(result)
        dump_json(output / "runs.json", results)
        progress.finish(result)

    try:
        jobs = [(mode, seed) for mode in MODES for seed in args.audit_seeds]
        with ProgressDisplay(gpus, len(jobs)) as progress:
            run_jobs(jobs, gpus, launch, save)
        for seed in args.audit_seeds:
            group = [row for row in results if row["seed"] == seed]
            for key in ("prefix_sha256", "train_draw_sha256", "slow_updates", "queried", "returned"):
                values = {row[key] for row in group}
                if len(values) != 1 or None in values:
                    raise ValueError(f"Unpaired diagnostic for seed={seed}: {key}")
        from .analyze import analyze
        analyze(output)
    except BaseException:
        manifest.update(status="failed", completed_runs=len(results))
        dump_json(output / "manifest.json", manifest)
        raise
    manifest.update(status="complete", completed_runs=len(results))
    dump_json(output / "manifest.json", manifest)
    print(f"Diagnostic report: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
