"""Run only direction_check and a training-strength-matched relaxation control."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import statistics
import subprocess
import sys

from experiments.delayed_feedback.run import dump_json, sha256, validate_inputs
from experiments.delayed_feedback.progress import ProgressDisplay, emit_progress, parse_progress
from experiments.delayed_feedback.scheduler import run_jobs
from experiments.vlcs_calibration import run as vlcs
from experiments.vlcs_calibration.reference import NativeTrace
from experiments.vlcs_calibration.results import write_csv
from .protocol import METHOD, VARIANTS, load_references, validate_variant, verify_snapshots, load_donor


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset-root", type=Path, default=vlcs.ROOT / "datasets")
    cli.add_argument("--checkpoint-dir", type=Path, default=vlcs.EXPERIMENT.checkpoint_dir)
    cli.add_argument("--reference-dir", type=Path, required=True, help="Completed VLCS three-group sweep")
    cli.add_argument("--confidence-dir", type=Path, required=True, help="Completed confidence_match sweep")
    cli.add_argument("--output-dir", type=Path, required=True)
    cli.add_argument("--resume", action="store_true", help="Reuse valid completed NEW runs; archive partial runs")
    cli.add_argument("--num-workers", type=int, default=0)
    devices = cli.add_mutually_exclusive_group()
    devices.add_argument("--gpus", type=int, nargs="+")
    devices.add_argument("--gpu-idx", type=int)
    cli.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    cli.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    cli.add_argument("--variant", choices=VARIANTS, help=argparse.SUPPRESS)
    return cli


def implementation_hashes():
    names = ("gradient_compatibility/algorithm.py", "gradient_compatibility/math.py",
             "gradient_compatibility/protocol.py", "gradient_compatibility/run.py",
             "confidence_match/algorithm.py", "fcra_dual/algorithm.py", "fcra/algorithm.py",
             "fcra_ablation/algorithm.py", "vlcs_calibration/algorithm.py", "vlcs_calibration/run.py")
    return {name: sha256(vlcs.ROOT / "experiments" / name) for name in names}


def donor_path(output, seed):
    return output / "direction_check" / f"seed_{seed}" / "direction.csv"


def worker(args):
    from .algorithm import GradientCompatibility
    parent = json.loads((args.output_dir.parent.parent / "manifest.json").read_text())
    if (args.seed not in parent["seeds"] or parent["reference_dir"] != str(args.reference_dir)
            or parent["confidence_dir"] != str(args.confidence_dir)
            or parent["dataset_root"] != str(args.dataset_root) or parent["num_workers"] != args.num_workers
            or parent["implementation_sha256"] != implementation_hashes()):
        raise ValueError("Worker differs from the saved experiment protocol")
    verify_snapshots(parent["reference_snapshots"])
    native = args.reference_dir / "simatta_original" / f"seed_{args.seed}"
    trace = NativeTrace(native, args.seed)
    if trace.hashes != parent["native_hashes"][str(args.seed)]:
        raise ValueError("Worker native reference changed")
    donor, donor_hash = None, None
    path = donor_path(args.output_dir.parent.parent, args.seed)
    if args.variant == "uniform_relaxation":
        donor_hash = sha256(path)
        if donor_hash != parent["donor_sha256"][str(args.seed)]:
            raise ValueError("Donor changed after the direction-check stage")
        donor = load_donor(path, trace)
    vlcs.worker(args, native_folder=native,
                algorithm_factory=lambda config, native_trace: GradientCompatibility(
                    config, native_trace, args.variant, parent["entropy_gap"], donor),
                settings_extra={"method": METHOD, "teacher_calibration": True,
                                "entropy_gap": parent["entropy_gap"], "donor_sha256": donor_hash,
                                "direction_parameters": "current linear classifier weight and bias",
                                "direction_loss": "raw CE, supervised mean vs each pseudo draw",
                                "strength_matching": "per-update mean removed weight, clipped to feasible range"})
    if donor_hash is not None and sha256(path) != donor_hash:
        # Do not leave a reusable completion marker if a donor was edited mid-run.
        (args.output_dir / "result.json").rename(args.output_dir / "invalid_result.json")
        raise ValueError("Donor changed while the control was running")


def summarize(output, results, existing, seeds):
    indexed = {(row["variant"], row["seed"]): row for row in results}
    if set(indexed) != {(mode, seed) for mode in VARIANTS for seed in seeds}:
        raise ValueError("Need both new variants for all three seeds")
    all_rows = {**existing, **indexed}
    old_modes = (*vlcs.VARIANTS, "confidence_match")
    comparisons = []
    for treatment in VARIANTS:
        baselines = old_modes + (("uniform_relaxation",) if treatment == "direction_check" else ())
        for baseline in baselines:
            for seed in seeds:
                comparisons.append({"variant": treatment, "baseline": baseline, "seed": seed, **{
                    f"{metric}_delta_pp": all_rows[treatment, seed][metric] - all_rows[baseline, seed][metric]
                    for metric in vlcs.ACCURACY}})
    write_csv(output / "paired_deltas.csv", comparisons)
    ordered = [indexed[mode, seed] for mode in VARIANTS for seed in seeds]
    write_csv(output / "per_seed.csv", ordered)
    lines = ["# VLCS D=0: classifier-head gradient compatibility", "",
             "Only two new groups ran. Four old groups were reused without retraining.",
             "Same native queries, per-batch update quotas, batch=32; no adapter or targeted replay.", "",
             "| Variant | LabelMe | SUN09 | VOC2007 | Stream | Macro |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for mode in (*old_modes, *VARIANTS):
        cells = [f"{statistics.mean(all_rows[mode, seed][key] for seed in seeds):.2f} ± "
                 f"{statistics.stdev(all_rows[mode, seed][key] for seed in seeds):.2f}" for key in vlcs.ACCURACY]
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    lines += ["", "| New variant | Compared with | Paired Stream delta (pp) | Positive seeds |",
              "| --- | --- | ---: | ---: |"]
    for treatment, baseline in dict.fromkeys((row["variant"], row["baseline"]) for row in comparisons):
        values = [row["stream_accuracy_pct_delta_pp"] for row in comparisons
                  if row["variant"] == treatment and row["baseline"] == baseline]
        lines.append(f"| {treatment} | {baseline} | {statistics.mean(values):+.3f} ± "
                     f"{statistics.stdev(values):.3f} | {sum(value > 0 for value in values)}/3 |")
    lines += ["", "| Seed | Control unmatched updates (>1e-6) | Mean absolute strength gap | Maximum gap |",
              "| ---: | ---: | ---: | ---: |"]
    for seed in seeds:
        row = indexed["uniform_relaxation", seed]
        lines.append(f"| {seed} | {row['unmatched_updates']} | {row['mean_abs_matching_gap']:.8f} | "
                     f"{row['max_abs_matching_gap']:.8f} |")
    lines += ["", "| New variant | Seed | Queries | Updates | Final mean gate | Conflict fraction |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for mode in VARIANTS:
        for seed in seeds:
            row = indexed[mode, seed]
            conflict = f"{row['conflict_fraction']:.6f}" if mode == "direction_check" else "n/a"
            lines.append(f"| {mode} | {seed} | {row['queried']} | {row['slow_updates']} | "
                         f"{row['final_gate_mean']:.6f} | {conflict} |")
    lines += ["", "Control matches the donor's per-update mean(1-g) whenever its own gate permits it.",
              "Saturated control steps are reported; do not claim exact strength matching when gaps remain.",
              "The control replays training statistics from a separate direction-check run, not hidden truth.",
              "It is a diagnostic control, not a standalone deployable ATTA method.",
              "Both new arms verify identical supervised/pseudo draw indices on every pseudo update.",
              "Alignment uses current raw-CE head gradients, not encoder gradients or momentum updates.",
              "Gradient conflict is not proof of incorrect pseudo-labels; no hidden correctness was collected.",
              "Equal encoder calls/update counts do not imply equal FLOPs; analytic head calculations add cost.",
              "Mean ± sample SD over three development seeds; no significance or novelty claim."]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args, reference, gap, existing, traces, snapshots, gpus):
    output, seeds = args.output_dir, reference["seeds"]
    protocol = {"method": METHOD, "dataset": "VLCS", "delay": 0, "batch_size": 32,
                "seeds": seeds, "variants": list(VARIANTS), "entropy_gap": gap,
                "dataset_root": str(args.dataset_root), "num_workers": args.num_workers,
                "reference_dir": str(args.reference_dir), "confidence_dir": str(args.confidence_dir),
                "checkpoint_sha256": reference["checkpoint_sha256"], "reference_snapshots": snapshots,
                "native_hashes": {str(seed): traces[seed].hashes for seed in seeds},
                "implementation_sha256": implementation_hashes()}
    completed = {}
    if args.resume:
        previous = json.loads((output / "manifest.json").read_text())
        if any(previous.get(key) != value for key, value in protocol.items()):
            raise ValueError("Resume protocol/code/references differ; use a new output directory")
        for mode in VARIANTS:
            for seed in seeds:
                path = donor_path(output, seed)
                donor_hash = sha256(path) if mode == "uniform_relaxation" and path.is_file() else None
                saved_hash = previous.get("donor_sha256", {}).get(str(seed))
                if mode == "uniform_relaxation" and saved_hash is not None and donor_hash != saved_hash:
                    raise ValueError("Direction-check donor changed since the previous attempt")
                result = validate_variant(output / mode / f"seed_{seed}", mode, seed, reference,
                                          traces[seed], gap, donor_hash=donor_hash)
                if result is not None:
                    load_donor(output / mode / f"seed_{seed}" / "direction.csv", traces[seed])
                    if mode == "uniform_relaxation" and ("direction_check", seed) not in completed:
                        raise ValueError("Completed control has no completed donor")
                    completed[mode, seed] = result
        backup = output / "resume_history" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        backup.mkdir(parents=True, exist_ok=False)
        for name in ("manifest.json", "per_seed.csv", "paired_deltas.csv", "summary.md"):
            if (output / name).is_file():
                (backup / name).write_bytes((output / name).read_bytes())
        for mode in VARIANTS:
            for seed in seeds:
                if (mode, seed) in completed:
                    continue
                folder = output / mode / f"seed_{seed}"
                if folder.exists():
                    target = backup / mode / folder.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    folder.rename(target)
                log = output / f"{mode}_seed_{seed}.log"
                if log.exists():
                    log.rename(backup / log.name)
    manifest = {**protocol, "status": "running", "completed_runs": len(completed),
                "reused_runs": len(completed), "donor_sha256": {},
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=vlcs.ROOT, text=True).strip()}
    dump_json(output / "manifest.json", manifest)
    results = list(completed.values())
    print(f"Reusing 12 old reference runs and {len(completed)} completed new runs; "
          f"scheduling {6 - len(completed)} new runs", flush=True)

    def launch(job, gpu):
        mode, seed = job
        folder = output / mode / f"seed_{seed}"
        command = [sys.executable, "-u", "-m", "experiments.gradient_compatibility.run", "--worker",
                   "--variant", mode, "--seed", str(seed), "--gpu-idx", str(gpu),
                   "--dataset-root", str(args.dataset_root), "--checkpoint-dir", str(args.checkpoint_dir),
                   "--reference-dir", str(args.reference_dir), "--confidence-dir", str(args.confidence_dir),
                   "--output-dir", str(folder), "--num-workers", str(args.num_workers)]
        progress.start(gpu, 0, seed)
        print(f"VLCS variant={mode}, seed={seed}, GPU={gpu}", flush=True)
        with (output / f"{mode}_seed_{seed}.log").open("w", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=vlcs.ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    event = parse_progress(line)
                    if event is not None:
                        progress.update(gpu, event)
        if process.returncode:
            raise RuntimeError(f"{mode}, seed={seed} failed; see its log; --resume skips finished runs")
        donor_hash = manifest["donor_sha256"].get(str(seed)) if mode == "uniform_relaxation" else None
        result = validate_variant(folder, mode, seed, reference, traces[seed], gap, donor_hash=donor_hash)
        if result is None:
            raise ValueError("Worker exited without a completed result")
        load_donor(folder / "direction.csv", traces[seed])
        return result

    def save(result):
        results.append(result)
        write_csv(output / "per_seed.csv", results)
        manifest["completed_runs"] = len(results)
        dump_json(output / "manifest.json", manifest)
        progress.finish(result)

    try:
        with ProgressDisplay(gpus, 6 - len(completed)) as progress:
            for mode in VARIANTS:
                if mode == "uniform_relaxation":
                    manifest["donor_sha256"] = {str(seed): sha256(donor_path(output, seed)) for seed in seeds}
                    dump_json(output / "manifest.json", manifest)
                pending = [(mode, seed) for seed in seeds if (mode, seed) not in completed]
                if pending:
                    run_jobs(pending, gpus, launch, save)
        verify_snapshots(snapshots)
        if implementation_hashes() != protocol["implementation_sha256"]:
            raise ValueError("Implementation changed during the experiment")
        summarize(output, results, existing, seeds)
    except BaseException:
        manifest.update(status="failed", completed_runs=len(results))
        dump_json(output / "manifest.json", manifest)
        raise
    manifest.update(status="complete", completed_runs=6)
    dump_json(output / "manifest.json", manifest)


def main():
    args = parser().parse_args()
    args.dataset = "VLCS"
    for key in ("dataset_root", "checkpoint_dir", "reference_dir", "confidence_dir", "output_dir"):
        setattr(args, key, getattr(args, key).resolve())
    validate_inputs(args)
    if args.worker:
        if args.variant is None or args.seed is None:
            raise ValueError("Worker requires variant and seed")
        worker(args)
        return
    if args.variant is not None or args.seed is not None:
        raise ValueError("variant/seed are worker-only")
    gpus = args.gpus if args.gpus is not None else [args.gpu_idx if args.gpu_idx is not None else 0]
    if len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("Use distinct nonnegative GPUs")
    reference, gap, existing, traces, snapshots = load_references(args)
    # Output must be separate from both read-only reference trees in either direction.
    for root in (args.reference_dir, args.confidence_dir):
        if args.output_dir == root or root in args.output_dir.parents or args.output_dir in root.parents:
            raise ValueError("Output and reference directories must not overlap")
    if args.resume:
        if not args.output_dir.is_dir():
            raise ValueError("--resume requires an existing output directory")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another runner is using this output directory") from None
        run(args, reference, gap, existing, traces, snapshots, gpus)


if __name__ == "__main__":
    main()
