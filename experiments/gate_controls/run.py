"""Run two new immediate-feedback controls; never train existing baselines."""

import argparse
import csv
import fcntl
import json
import math
import multiprocessing
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone

from experiments.delayed_feedback.datasets import EXPERIMENTS, ROOT
from experiments.delayed_feedback.run import dump_json, sha256, validate_inputs
from experiments.delayed_feedback.progress import ProgressDisplay, emit_progress, parse_progress
from experiments.delayed_feedback.scheduler import run_jobs
from experiments.fcra_ablation.run import completed_result
from experiments.fcra_dual.reference import ReferenceTrace
from experiments.fcra_dual.run import load_reference
from experiments.vlcs_calibration.reference import NativeTrace
from experiments.vlcs_calibration.results import load_completed, write_csv


METHOD = "source-gate-controls"
VARIANTS = ("uniform_gate", "shuffled_gate")
BASELINES = ("simatta_control", "calibration_only")
REQUIRED = ("settings.json", "result.json", "batches.csv", "domains.csv", "feedback.csv",
            "updates.csv", "gate_updates.csv", "gate_samples.csv")


def read_json(path):
    return json.loads(path.read_text())


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset", required=True, choices=EXPERIMENTS)
    cli.add_argument("--dataset-root", type=Path, default=ROOT / "storage/datasets")
    cli.add_argument("--checkpoint-dir", type=Path)
    cli.add_argument("--reference-dir", type=Path, required=True)
    cli.add_argument("--baseline-dir", type=Path,
                     help="PACS: completed FCRA-ablation; VLCS: defaults to reference-dir")
    cli.add_argument("--output-dir", type=Path, required=True)
    cli.add_argument("--resume", action="store_true")
    devices = cli.add_mutually_exclusive_group()
    devices.add_argument("--gpus", type=int, nargs="+")
    devices.add_argument("--gpu-idx", type=int)
    cli.add_argument("--num-workers", type=int, default=0)
    cli.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    cli.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    cli.add_argument("--variant", choices=VARIANTS, help=argparse.SUPPRESS)
    return cli


def preflight(args):
    """Use the original dataset-specific validators and pin read-only inputs."""
    reference = read_json(args.reference_dir / "manifest.json")
    baseline = read_json(args.baseline_dir / "manifest.json")
    experiment = EXPERIMENTS[args.dataset]
    if args.dataset == "PACS":
        args.seeds = None
        reference, seeds = load_reference(args)
        if (baseline.get("method") != "FCRA-ablation" or baseline.get("status") != "complete"
                or baseline.get("dataset") != "PACS" or baseline.get("delay") != 0
                or baseline.get("seeds") != seeds
                or baseline.get("reference_manifest_sha256") != sha256(args.reference_dir / "manifest.json")):
            raise ValueError("PACS requires completed ablations from exactly this FCRA reference")
        traces = {seed: ReferenceTrace(args.reference_dir / "D0" / f"seed_{seed}", 0, seed)
                  for seed in seeds}
    else:
        seeds = reference["seeds"]
        if (reference.get("method") != "VLCS-calibration" or reference.get("status") != "complete"
                or reference.get("dataset") != "VLCS" or reference.get("delay") != 0
                or reference.get("batch_size") != 32 or args.baseline_dir != args.reference_dir):
            raise ValueError("VLCS requires its completed native/control/calibration experiment")
        traces = {seed: NativeTrace(args.reference_dir / "simatta_original" / f"seed_{seed}", seed)
                  for seed in seeds}
    if (len(seeds) != 3 or len(set(seeds)) != 3
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError("Reference must have three distinct valid seeds")
    for manifest in (reference, baseline):
        if (Path(manifest["dataset_root"]).resolve() != args.dataset_root
                or manifest["num_workers"] != args.num_workers):
            raise ValueError("Use identical dataset path and num-workers as existing runs")
    checkpoints = {name: sha256(args.checkpoint_dir / name) for name in experiment.checkpoint_names}
    if any(manifest["checkpoint_sha256"] != checkpoints for manifest in (reference, baseline)):
        raise ValueError("Source weights do not match both references")
    artifacts = {}

    def pin(path):
        artifacts[str(path)] = sha256(path)

    pin(args.reference_dir / "manifest.json")
    pin(args.baseline_dir / "manifest.json")
    existing = []
    for seed, trace in traces.items():
        if args.dataset == "VLCS" and trace.settings["checkpoint_sha256"] != checkpoints:
            raise ValueError("Trace source weights mismatch")
        if (Path(trace.settings["dataset_root"]).resolve() != args.dataset_root
                or trace.settings["num_workers"] != args.num_workers):
            raise ValueError("Trace dataset/workers mismatch")
        if args.dataset == "PACS" and baseline["reference_hashes"][str(seed)] != trace.hashes:
            raise ValueError("PACS baseline trace mismatch")
        native = (args.reference_dir / "D0" / f"seed_{seed}" if args.dataset == "PACS" else
                  args.reference_dir / "simatta_original" / f"seed_{seed}")
        for name in trace.hashes:
            pin(native / name)
        for mode in BASELINES:
            folder = args.baseline_dir / mode / f"seed_{seed}"
            result = (completed_result(folder, mode, seed, reference, trace, args.dataset_root)
                      if args.dataset == "PACS" else load_completed(folder, mode, seed, reference, trace))
            if result is None:
                raise ValueError(f"Missing completed baseline: {folder}; no baseline will be rerun")
            settings = read_json(folder / "settings.json")
            if settings["num_workers"] != args.num_workers:
                raise ValueError("Baseline worker count differs")
            for key in accuracy_keys(args.dataset):
                value = result[key]
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100:
                    raise ValueError(f"Invalid baseline accuracy: {folder}/{key}")
            for name in ("settings.json", "result.json", "batches.csv", "domains.csv", "feedback.csv", "updates.csv"):
                pin(folder / name)
            existing.append(result)
    protocol = {"method": METHOD, "dataset": args.dataset, "delay": 0, "batch_size": 32,
                "seeds": seeds, "variants": list(VARIANTS), "dataset_root": str(args.dataset_root),
                "checkpoint_sha256": checkpoints, "num_workers": args.num_workers,
                "reference_dir": str(args.reference_dir), "baseline_dir": str(args.baseline_dir),
                "reference_hashes": {str(seed): trace.hashes for seed, trace in traces.items()},
                "input_sha256": artifacts,
                "code_sha256": code_hashes()}
    return protocol, traces, existing


def code_hashes():
    # Pin inherited implementation too; resume must not mix code versions.
    files = [path for directory in (ROOT / "ATTA", ROOT / "experiments")
             for path in directory.rglob("*.py")]
    files += [EXPERIMENTS[name].config_path for name in EXPERIMENTS]
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(files)}


def verify_inputs(protocol):
    for path, digest in protocol["input_sha256"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Read-only reference changed: {path}")
    if code_hashes() != protocol["code_sha256"]:
        raise ValueError("Experiment code changed; use a new output directory")


def validate_new(folder, mode, seed, protocol, trace):
    if not (folder / "result.json").is_file():
        return None
    settings, result = read_json(folder / "settings.json"), read_json(folder / "result.json")
    for record in (settings, result):
        if any(record.get(key) != value for key, value in {
                "method": METHOD, "dataset": protocol["dataset"], "variant": mode,
                "seed": seed, "delay": 0}.items()):
            raise ValueError(f"Wrong completed identity: {folder}")
    if (settings.get("protocol") != protocol or settings.get("simatta") != trace.settings["simatta"]
            or settings.get("train_batch_size") != trace.settings["train_batch_size"]
            or settings.get("fcra") != trace.settings["fcra"]
            or result.get("slow_updates") != trace.total or result.get("reference_updates") != trace.total
            or result.get("queried") != len(trace.records) or result.get("returned") != len(trace.records)
            or result.get("pending") != 0 or result.get("fast_updates") != 0
            or result.get("fast_adapter_enabled") is not False):
        raise ValueError(f"Completed run protocol/budget mismatch: {folder}")
    for metric in accuracy_keys(protocol["dataset"]):
        value = result[metric]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"Invalid accuracy: {folder}/{metric}")
    for name in REQUIRED:
        if not (folder / name).is_file():
            raise ValueError(f"Missing diagnostic: {folder}/{name}")
    with (folder / "feedback.csv").open(newline="") as stream:
        trace.verify_feedback(list(csv.DictReader(stream)))
    return result


def worker(args):
    multiprocessing.set_start_method("spawn", force=True)
    import torch
    from ATTA import args_parser, config_summoner
    from ATTA.utils.initial import reset_random_seed
    from experiments.fcra.algorithm import DEFAULTS
    from .algorithm import GateControl

    parent = read_json(args.output_dir.parent.parent / "manifest.json")
    protocol, traces, _ = preflight(args)
    if protocol != parent["protocol"]:
        raise ValueError("Worker inputs differ from coordinator preflight")
    verify_inputs(protocol)
    if (args.seed not in traces or args.variant not in VARIANTS or args.gpus is not None
            or args.gpu_idx is None or not torch.cuda.is_available()
            or not 0 <= args.gpu_idx < torch.cuda.device_count()):
        raise ValueError("Worker requires valid identity and one visible CUDA GPU")
    torch.cuda.set_device(args.gpu_idx)
    experiment, trace = EXPERIMENTS[args.dataset], traces[args.seed]
    config = config_summoner(args_parser([
        "--config_path", str(experiment.config_path), "--exp_round", "1",
        "--ckpt_dir", str(args.checkpoint_dir), "--dataset.dataset_root", str(args.dataset_root),
        "--gpu_idx", str(args.gpu_idx), "--num_workers", str(args.num_workers), "--atta.batch_size", "32"]))
    config.random_seed = args.seed
    if (config.dataset.name != args.dataset or tuple(config.dataset.test_envs) != experiment.environments
            or config.atta.batch_size != 32 or dict(config.atta.SimATTA) != trace.settings["simatta"]
            or config.train.train_bs != trace.settings["train_batch_size"] or DEFAULTS != trace.settings["fcra"]):
        raise ValueError("Current configuration differs from existing runs")
    reset_random_seed(config)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dump_json(args.output_dir / "settings.json", {
        "method": METHOD, "dataset": args.dataset, "variant": args.variant, "seed": args.seed,
        "delay": 0, "protocol": protocol, "simatta": dict(config.atta.SimATTA), "fcra": DEFAULTS,
        "train_batch_size": config.train.train_bs, "source_reference": "frozen source model",
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(args.gpu_idx)})
    result = GateControl(config, trace, args.variant).run_stream(args.output_dir, progress=emit_progress)
    verify_inputs(protocol)
    dump_json(args.output_dir / "result.json", result)


def accuracy_keys(dataset):
    return tuple(f"{domain}_accuracy_pct" for domain in EXPERIMENTS[dataset].targets) + (
        "stream_accuracy_pct", "domain_macro_accuracy_pct")


def summarize(output, protocol, rows, existing):
    seeds, dataset = protocol["seeds"], protocol["dataset"]
    indexed = {(row["variant"], row["seed"]): row for row in rows + existing}
    modes = BASELINES + VARIANTS
    if set(indexed) != {(mode, seed) for mode in modes for seed in seeds}:
        raise ValueError("Incomplete results; summary is not valid")
    for seed in seeds:
        for key in ("queried", "returned", "slow_updates"):
            if len({indexed[mode, seed][key] for mode in modes}) != 1:
                raise ValueError(f"Budget mismatch: {seed}/{key}")
        if indexed[VARIANTS[0], seed]["train_draw_sha256"] != indexed[VARIANTS[1], seed]["train_draw_sha256"]:
            raise ValueError("New controls consumed different supervised/pseudo draw sequences")
    metrics = accuracy_keys(dataset) + ("queried", "returned", "slow_updates")
    summary = [{"variant": mode, "runs": len(seeds), **{
        f"{key}_{stat}": function(indexed[mode, seed][key] for seed in seeds)
        for key in metrics for stat, function in (("mean", statistics.mean), ("std", statistics.stdev))}}
        for mode in modes]
    pairs = []
    for treatment, control in (("calibration_only", "simatta_control"),
                               ("calibration_only", "uniform_gate"),
                               ("calibration_only", "shuffled_gate"),
                               ("uniform_gate", "simatta_control"),
                               ("shuffled_gate", "simatta_control")):
        for seed in seeds:
            pairs.append({"treatment": treatment, "control": control, "seed": seed, **{
                f"{key}_delta_pp": indexed[treatment, seed][key] - indexed[control, seed][key]
                for key in accuracy_keys(dataset)}})
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired_deltas.csv", pairs)
    write_csv(output / "all_results.csv", existing + rows)
    summarize_diagnostics(output, rows)
    header = [*EXPERIMENTS[dataset].targets, "Stream", "Macro"]
    lines = [f"# {dataset}: immediate-feedback gate controls", "",
             "Two new controls; existing control/calibration runs reused read-only.", "",
             "| Variant | " + " | ".join(header) + " |", "| --- |" + " ---: |" * len(header)]
    for row in summary:
        lines.append(f"| {row['variant']} | " + " | ".join(
            f"{row[key+'_mean']:.3f} ± {row[key+'_std']:.3f}" for key in accuracy_keys(dataset)) + " |")
    lines += ["", "| Treatment - control | Stream delta (pp) | Positive seeds |",
              "| --- | ---: | ---: |"]
    for treatment, control in dict.fromkeys((row["treatment"], row["control"]) for row in pairs):
        values = [row["stream_accuracy_pct_delta_pp"] for row in pairs
                  if (row["treatment"], row["control"]) == (treatment, control)]
        lines.append(f"| {treatment} - {control} | {statistics.mean(values):+.3f} ± "
                     f"{statistics.stdev(values):.3f} | {sum(value > 0 for value in values)}/{len(seeds)} |")
    lines += ["", "Mean ± sample SD; development seeds do not establish significance.",
              "Each new trajectory computes its own raw gates. Matching means within a trajectory",
              "does not match the calibration-only trajectory, CE values or gradient magnitudes.",
              "Shuffling preserves the draw-position weight multiset; duplicate images may receive different weights.",
              "Sample diagnostics describe only new trajectories; old baseline diagnostics are not reconstructed.",
              "Hidden truth is used only for evaluation diagnostics, never gate transformation.",
              "Equal update counts do not imply equal FLOPs. See summary.csv for all cost counters."]
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def summarize_diagnostics(output, rows):
    """Keep image-equal and draw-weighted descriptions separate, per seed/domain."""
    from collections import defaultdict

    reports = []
    for result in rows:
        groups = defaultdict(list)
        folder = output / result["variant"] / f"seed_{result['seed']}"
        with (folder / "gate_samples.csv").open(newline="") as stream:
            for record in csv.DictReader(stream):
                if int(record["draws"]) == 0:
                    continue
                groups[record["domain"], int(record["correct"])].append(record)
        for (domain, correct), samples in sorted(groups.items()):
            for weighting in ("sample_equal", "draw_weighted"):
                weights = [1 if weighting == "sample_equal" else int(item["draws"]) for item in samples]
                denominator = sum(weights)
                report = {"variant": result["variant"], "seed": result["seed"], "domain": domain,
                          "correct": correct, "weighting": weighting, "unique_samples": len(samples),
                          "draws": sum(int(item["draws"]) for item in samples)}
                for key in ("raw_gate_mean", "effective_gate_mean", "raw_removed_ce_mean",
                            "effective_removed_ce_mean", "covered_fraction"):
                    report[key] = sum(weight * float(item[key]) for weight, item in zip(weights, samples)) / denominator
                reports.append(report)
    write_csv(output / "diagnostics_summary.csv", reports or [{"draws": 0}])


def main():
    args = parser().parse_args()
    args.checkpoint_dir = args.checkpoint_dir or EXPERIMENTS[args.dataset].checkpoint_dir
    if args.baseline_dir is None:
        if args.dataset == "PACS":
            raise ValueError("PACS needs --baseline-dir for the completed ablation experiment")
        args.baseline_dir = args.reference_dir
    for name in ("dataset_root", "checkpoint_dir", "reference_dir", "baseline_dir", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    validate_inputs(args)
    if args.worker:
        worker(args)
        return
    if args.seed is not None or args.variant is not None:
        raise ValueError("seed/variant are worker-only")
    for source in (args.reference_dir, args.baseline_dir, args.dataset_root, args.checkpoint_dir):
        if args.output_dir == source or source in args.output_dir.parents or args.output_dir in source.parents:
            raise ValueError("Output must be separate from inputs")
    protocol, traces, existing = preflight(args)
    gpus = args.gpus if args.gpus is not None else [args.gpu_idx if args.gpu_idx is not None else 0]
    if not gpus or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("Choose distinct nonnegative GPUs")
    if args.resume:
        if not args.output_dir.is_dir():
            raise ValueError("--resume requires an existing output directory")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    # Exclusive coordinator lock, released automatically even after failure.
    with (args.output_dir / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        coordinate(args, protocol, traces, existing, gpus)


def coordinate(args, protocol, traces, existing, gpus):
    output, seeds = args.output_dir, protocol["seeds"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    rows, jobs = [], []
    if args.resume:
        previous = read_json(output / "manifest.json")
        if previous["protocol"] != protocol:
            raise ValueError("Resume protocol/code/input mismatch; use a new directory")
    for mode in VARIANTS:
        for seed in seeds:
            folder = output / mode / f"seed_{seed}"
            result = validate_new(folder, mode, seed, protocol, traces[seed]) if args.resume else None
            if result is None:
                jobs.append((mode, seed))
            else:
                rows.append(result)
    if args.resume:
        backup = output / "resume_history" / stamp
        backup.mkdir(parents=True, exist_ok=False)
        for name in ("manifest.json", "per_seed.csv", "summary.csv", "summary.md", "paired_deltas.csv", "all_results.csv", "diagnostics_summary.csv"):
            path = output / name
            if path.is_file():
                path.rename(backup / name)
        for mode, seed in jobs:
            folder = output / mode / f"seed_{seed}"
            if folder.exists():
                target = backup / mode / folder.name
                target.parent.mkdir(parents=True, exist_ok=True)
                folder.rename(target)
            log = output / f"{mode}_seed_{seed}.log"
            if log.exists():
                log.rename(backup / log.name)
    manifest = {"protocol": protocol, "status": "running", "completed_runs": len(rows),
                "reused_runs": len(rows), "gpus": gpus,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    dump_json(output / "manifest.json", manifest)
    print(f"Reusing {len(rows)} new controls; scheduling {len(jobs)} missing controls; baselines are read-only", flush=True)

    def launch(job, gpu):
        mode, seed = job
        folder = output / mode / f"seed_{seed}"
        command = [sys.executable, "-m", "experiments.gate_controls.run", "--worker",
                   "--dataset", args.dataset, "--variant", mode, "--seed", str(seed), "--gpu-idx", str(gpu),
                   "--dataset-root", str(args.dataset_root), "--checkpoint-dir", str(args.checkpoint_dir),
                   "--reference-dir", str(args.reference_dir), "--baseline-dir", str(args.baseline_dir),
                   "--output-dir", str(folder), "--num-workers", str(args.num_workers)]
        progress.start(gpu, 0, seed)
        print(f"{args.dataset} variant={mode}, seed={seed}, GPU={gpu}", flush=True)
        with (output / f"{mode}_seed_{seed}.log").open("w") as log:
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    event = parse_progress(line)
                    if event is not None:
                        progress.update(gpu, event)
        if process.returncode:
            raise RuntimeError(f"{mode}/{seed} failed; see its log; no baselines rerun")
        return validate_new(folder, mode, seed, protocol, traces[seed])

    def save(result):
        if result is None:
            raise ValueError("Worker did not write a completion marker")
        rows.append(result)
        write_csv(output / "per_seed.csv", rows)
        progress.finish(result)

    try:
        with ProgressDisplay(gpus, len(jobs)) as progress:
            run_jobs(jobs, gpus, launch, save)
        verify_inputs(protocol)
        summarize(output, protocol, rows, existing)
    except BaseException:
        manifest.update(status="failed", completed_runs=len(rows))
        dump_json(output / "manifest.json", manifest)
        raise
    manifest.update(status="complete", completed_runs=len(rows))
    dump_json(output / "manifest.json", manifest)
    write_csv(output / "per_seed.csv", rows)
    print(f"Complete: {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
