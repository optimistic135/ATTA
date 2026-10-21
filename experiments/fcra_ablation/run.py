"""Controlled D=0 ablations plus original SimATTA at matched public settings."""

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone

from experiments.fcra_dual import run as dual
from experiments.fcra_dual.reference import ReferenceTrace
from experiments.delayed_feedback.progress import ProgressDisplay, emit_progress, parse_progress
from experiments.delayed_feedback.scheduler import run_jobs
from experiments.delayed_feedback.run import dump_json, sha256, validate_inputs, write_csv


CONTROLLED_VARIANTS = ("simatta_control", "replay_only", "calibration_only", "full_fcra")
VARIANTS = CONTROLLED_VARIANTS + ("simatta_original",)
METRICS = tuple(f"{domain}_accuracy_pct" for domain in dual.EXPERIMENT.targets) + (
    "stream_accuracy_pct", "domain_macro_accuracy_pct", "queried", "returned", "slow_updates")
CONTRASTS = (
    ("replay_alone", "replay_only", "simatta_control"),
    ("calibration_alone", "calibration_only", "simatta_control"),
    ("replay_with_calibration", "full_fcra", "calibration_only"),
    ("calibration_with_replay", "full_fcra", "replay_only"),
    ("full_vs_control", "full_fcra", "simatta_control"),
    ("full_vs_original", "full_fcra", "simatta_original"),
    ("control_vs_original", "simatta_control", "simatta_original"),
)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset-root", type=Path, default=dual.ROOT / "storage/datasets")
    cli.add_argument("--checkpoint-dir", type=Path, default=dual.EXPERIMENT.checkpoint_dir)
    cli.add_argument("--reference-dir", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path)
    cli.add_argument("--resume", action="store_true",
                     help="Reuse completed runs in output-dir and update its summary with missing runs")
    devices = cli.add_mutually_exclusive_group()
    devices.add_argument("--gpu-idx", type=int)
    devices.add_argument("--gpus", type=int, nargs="+")
    cli.add_argument("--num-workers", type=int, default=0)
    cli.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    cli.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    cli.add_argument("--variant", choices=VARIANTS, help=argparse.SUPPRESS)
    return cli


def worker(args):
    multiprocessing.set_start_method("spawn", force=True)
    import torch
    from ATTA import args_parser, config_summoner
    from ATTA.utils.initial import reset_random_seed
    from experiments.fcra.algorithm import DEFAULTS
    from .algorithm import AblationFCRA, OriginalSimATTA

    if args.seed is None or args.variant is None or args.output_dir is None or args.gpus is not None:
        raise ValueError("Worker needs seed, variant, output-dir and one gpu-idx")
    if (args.gpu_idx is None or not torch.cuda.is_available()
            or not 0 <= args.gpu_idx < torch.cuda.device_count()):
        raise RuntimeError("A visible CUDA GPU is required")
    torch.cuda.set_device(args.gpu_idx)
    reference, seeds = dual.load_reference(args)
    if args.seed not in seeds:
        raise ValueError("Worker seed differs from reference")
    trace = ReferenceTrace(args.reference_dir / "D0" / f"seed_{args.seed}", 0, args.seed)
    parent = json.loads((args.output_dir.parent.parent / "manifest.json").read_text())
    if (trace.hashes != parent["reference_hashes"][str(args.seed)]
            or sha256(args.reference_dir / "manifest.json") != parent["reference_manifest_sha256"]):
        raise ValueError("Reference changed after preflight")
    config = config_summoner(args_parser([
        "--config_path", str(dual.EXPERIMENT.config_path), "--exp_round", "1",
        "--ckpt_dir", str(args.checkpoint_dir), "--dataset.dataset_root", str(args.dataset_root),
        "--gpu_idx", str(args.gpu_idx), "--num_workers", str(args.num_workers),
        "--atta.batch_size", "32",
    ]))
    config.random_seed = args.seed
    if (config.dataset.name != "PACS" or tuple(config.dataset.test_envs) != dual.EXPERIMENT.environments
            or config.atta.batch_size != 32
            or config.train.train_bs != trace.settings["train_batch_size"]
            or dict(config.atta.SimATTA) != trace.settings["simatta"]
            or any(trace.settings["fcra"][key] != DEFAULTS[key] for key in (
                "replay_mix", "replay_neighbors", "epsilon", "teacher_cache_size",
                "gate_neighbors", "gate_similarity", "gate_temperature", "gate_prior"))):
        raise ValueError("Configuration differs from the reference")
    reset_random_seed(config)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    algorithm = (OriginalSimATTA(config) if args.variant == "simatta_original"
                 else AblationFCRA(config, trace, args.variant))
    original = args.variant == "simatta_original"
    dump_json(args.output_dir / "settings.json", {
        "method": "FCRA-ablation", "dataset": "PACS",
        "variant": args.variant, "seed": args.seed, "delay": 0, "batch_size": 32,
        "dataset_root": str(args.dataset_root), "checkpoint_dir": str(args.checkpoint_dir),
        "source": dual.EXPERIMENT.source, "target_order": list(dual.EXPERIMENT.targets),
        "train_batch_size": config.train.train_bs, "simatta": dict(config.atta.SimATTA),
        "fcra": DEFAULTS, "targeted_replay": False if original else algorithm.use_replay,
        "teacher_calibration": False if original else algorithm.use_calibration,
        "fast_adapter_enabled": False,
        "query_policy": "original_simatta" if original else "reference_trace",
        "update_policy": "original_loss_early_stop" if original else "reference_quota",
        "reference_hashes": trace.hashes, "reference_updates": None if original else trace.total,
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.gpu_idx),
        "num_workers": args.num_workers,
    })
    result = algorithm.run_stream(args.output_dir, progress=emit_progress)
    dump_json(args.output_dir / "result.json", result)


def summarize(rows, output, seeds):
    indexed = {(row["variant"], row["seed"]): row for row in rows}
    if len(rows) != len(VARIANTS) * len(seeds) or set(indexed) != {
            (variant, seed) for variant in VARIANTS for seed in seeds}:
        raise ValueError("Incomplete five-variant experiment")
    for seed in seeds:
        controls = [indexed[variant, seed] for variant in CONTROLLED_VARIANTS]
        for key in ("queried", "returned", "reference_updates", "slow_updates"):
            if len({row[key] for row in controls}) != 1:
                raise ValueError(f"Unmatched budget for seed {seed}: {key}")
    summaries = []
    for variant in VARIANTS:
        record = {"variant": variant, "runs": len(seeds)}
        for metric in METRICS:
            values = [indexed[variant, seed][metric] for seed in seeds]
            record[f"{metric}_mean"] = statistics.mean(values)
            record[f"{metric}_std"] = statistics.stdev(values)
        summaries.append(record)
    write_csv(output / "summary.csv", summaries)
    paired = []
    for name, treatment, control in CONTRASTS:
        for seed in seeds:
            row = {"contrast": name, "seed": seed}
            for metric in METRICS[:5]:
                row[f"{metric}_delta_pp"] = indexed[treatment, seed][metric] - indexed[control, seed][metric]
            paired.append(row)
    for seed in seeds:
        row = {"contrast": "interaction", "seed": seed}
        for metric in METRICS[:5]:
            row[f"{metric}_delta_pp"] = (
                indexed["full_fcra", seed][metric] - indexed["replay_only", seed][metric]
                - indexed["calibration_only", seed][metric] + indexed["simatta_control", seed][metric])
        paired.append(row)
    write_csv(output / "paired_deltas.csv", paired)
    lines = ["# PACS D=0: controlled ablation and original SimATTA", "",
             "All groups share source weights, stream seeds, batch=32 and training hyperparameters.",
             "Only the four controlled groups share queries and per-batch update quotas.",
             "SimATTA-control uses its uncalibrated joint objective with uniform replay in a shared runner;",
             "it is not an end-to-end reproduction of the original query/early-stop implementation.", "",
             "simatta_original uses the original selection, joint training and loss early stopping,",
             "with the D=0 stream adapter's ID deduplication/tiny-cluster guards and pre-update scoring.",
             "Its label count and update count are measured, not forced to match FCRA.", "",
             "| Variant | Art | Cartoon | Sketch | Stream | Macro |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summaries:
        cells = [f"{row[m + '_mean']:.2f} ± {row[m + '_std']:.2f}" for m in METRICS[:5]]
        lines.append("| " + row["variant"] + " | " + " | ".join(cells) + " |")
    lines += ["", "| Variant | Queried | Returned | Actual updates |",
              "| --- | ---: | ---: | ---: |"]
    for row in summaries:
        cells = [f"{row[m + '_mean']:.2f} ± {row[m + '_std']:.2f}"
                 for m in ("queried", "returned", "slow_updates")]
        lines.append("| " + row["variant"] + " | " + " | ".join(cells) + " |")
    lines += ["", "| Paired contrast | Stream delta (pp) | Positive seeds |",
              "| --- | ---: | ---: |"]
    for name in [item[0] for item in CONTRASTS] + ["interaction"]:
        values = [row["stream_accuracy_pct_delta_pp"] for row in paired if row["contrast"] == name]
        lines.append(f"| {name} | {statistics.mean(values):+.2f} ± {statistics.stdev(values):.2f} | "
                     f"{sum(value > 0 for value in values)}/{len(values)} |")
    lines += ["", "Mean ± sample SD; three paired seeds do not establish significance.",
              "Positive interaction means the combined gain exceeds the sum of isolated gains.",
              "Contrasts involving simatta_original compare complete adaptation policies, not isolated modules;",
              "interpret them together with label/update costs. Module attribution uses only controlled groups.",
              "Equal update counts do not imply equal FLOPs. No evaluation labels select modules or parameters."]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def completed_result(folder, variant, seed, reference, trace, dataset_root):
    """A result written after run_stream is the completion marker, not a CSV mean."""
    path = folder / "result.json"
    if not path.is_file():
        return None
    result = json.loads(path.read_text())
    settings = json.loads((folder / "settings.json").read_text())
    for record in (result, settings):
        if (record.get("method") != "FCRA-ablation" or record.get("variant") != variant
                or record.get("seed") != seed or record.get("delay") != 0):
            raise ValueError(f"Cannot reuse mismatched run: {folder}")
    if (settings["reference_hashes"] != trace.hashes
            or settings["checkpoint_sha256"] != reference["checkpoint_sha256"]
            or Path(settings["dataset_root"]).resolve() != dataset_root
            or settings["batch_size"] != 32
            or settings["simatta"] != trace.settings["simatta"]
            or settings["train_batch_size"] != trace.settings["train_batch_size"]
            or settings["fast_adapter_enabled"] or result["fast_updates"]):
        raise ValueError(f"Cannot reuse a run with different settings: {folder}")
    if variant in CONTROLLED_VARIANTS:
        expected_flags = {
            "simatta_control": (False, False), "replay_only": (True, False),
            "calibration_only": (False, True), "full_fcra": (True, True),
        }[variant]
        if ((settings["targeted_replay"], settings["teacher_calibration"]) != expected_flags
                or settings["fcra"] != trace.settings["fcra"]
                or result["slow_updates"] != trace.total
                or result["reference_updates"] != trace.total
                or result["queried"] != len(trace.records) or result["returned"] != len(trace.records)):
            raise ValueError(f"Cannot reuse incompatible controlled run: {folder}")
    elif (settings.get("query_policy") != "original_simatta"
          or settings.get("update_policy") != "original_loss_early_stop"):
        raise ValueError(f"Cannot reuse a different SimATTA policy: {folder}")
    if any(not (folder / name).is_file() for name in (
            "batches.csv", "domains.csv", "feedback.csv", "updates.csv")):
        raise ValueError(f"Completed run is missing diagnostics: {folder}")
    return result


def main():
    args = parser().parse_args()
    args.dataset, args.seeds = "PACS", None
    for name in ("dataset_root", "checkpoint_dir", "reference_dir", "output_dir"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    validate_inputs(args)
    if args.worker:
        worker(args)
        return
    if args.seed is not None or args.variant is not None:
        raise ValueError("seed and variant are worker-only arguments")
    if args.resume and (args.output_dir is None or not args.output_dir.is_dir()):
        raise ValueError("--resume requires an existing --output-dir")
    reference, seeds = dual.load_reference(args)
    traces = {seed: ReferenceTrace(args.reference_dir / "D0" / f"seed_{seed}", 0, seed)
              for seed in seeds}
    hashes = {str(seed): trace.hashes for seed, trace in traces.items()}
    gpus = args.gpus if args.gpus is not None else [args.gpu_idx if args.gpu_idx is not None else 0]
    if len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("Provide distinct nonnegative GPU indices")
    output = args.output_dir or dual.ROOT / "storage/fcra_ablation" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    rows, jobs = [], []
    if args.resume:
        previous = json.loads((output / "manifest.json").read_text())
        if (previous.get("method") != "FCRA-ablation" or previous.get("dataset") != "PACS"
                or previous.get("delay") != 0 or previous["seeds"] != seeds
                or previous["reference_hashes"] != hashes
                or previous["checkpoint_sha256"] != reference["checkpoint_sha256"]
                or previous["reference_manifest_sha256"] != sha256(args.reference_dir / "manifest.json")
                or Path(previous["dataset_root"]).resolve() != args.dataset_root):
            raise ValueError("Existing experiment does not match the current protocol")
        # Validate every completed run before changing any experiment files.
        for variant in VARIANTS:
            for seed in seeds:
                folder = output / variant / f"seed_{seed}"
                result = completed_result(folder, variant, seed, reference, traces[seed], args.dataset_root)
                if result is None:
                    jobs.append((variant, seed))
                else:
                    rows.append(result)
        backup = output / "resume_history" / stamp
        backup.mkdir(parents=True, exist_ok=False)
        for name in ("manifest.json", "summary.md", "summary.csv", "per_seed.csv", "paired_deltas.csv"):
            path = output / name
            if path.is_file():
                (backup / name).write_bytes(path.read_bytes())
        for variant, seed in jobs:
            folder = output / variant / f"seed_{seed}"
            if folder.exists():
                target = backup / variant / folder.name
                target.parent.mkdir(parents=True, exist_ok=True)
                folder.rename(target)
    else:
        output.mkdir(parents=True, exist_ok=False)
        jobs = [(variant, seed) for variant in VARIANTS for seed in seeds]
    manifest = {
        "method": "FCRA-ablation", "status": "running", "dataset": "PACS", "delay": 0,
        "seeds": seeds, "variants": VARIANTS, "reference_hashes": hashes,
        "reference_manifest_sha256": sha256(args.reference_dir / "manifest.json"),
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "dataset_root": str(args.dataset_root), "reference_dir": str(args.reference_dir),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dual.ROOT, text=True).strip(),
        "gpus": gpus, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "num_workers": args.num_workers,
        "protocol": "D0; four fixed-query/update ablations plus original SimATTA selection/early stopping",
        "resumed": args.resume, "reused_runs": len(rows), "completed_runs": len(rows),
        "reused_artifact_hashes": {
            f"{row['variant']}/seed_{row['seed']}": {
                name: sha256(output / row["variant"] / f"seed_{row['seed']}" / name)
                for name in ("settings.json", "result.json")}
            for row in rows},
    }
    dump_json(output / "manifest.json", manifest)
    print(f"Reusing {len(rows)} completed runs; scheduling {len(jobs)} missing runs", flush=True)
    if rows:
        write_csv(output / "per_seed.csv", rows)

    def launch(job, gpu):
        variant, seed = job
        folder = output / variant / f"seed_{seed}"
        command = [sys.executable, "-m", "experiments.fcra_ablation.run", "--worker",
                   "--variant", variant, "--seed", str(seed), "--gpu-idx", str(gpu),
                   "--dataset-root", str(args.dataset_root), "--checkpoint-dir", str(args.checkpoint_dir),
                   "--reference-dir", str(args.reference_dir), "--output-dir", str(folder),
                   "--num-workers", str(args.num_workers)]
        progress.start(gpu, 0, seed)
        print(f"Variant={variant}, seed={seed}, GPU={gpu}", flush=True)
        with (output / f"{variant}_seed_{seed}.log").open("w", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=dual.ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    event = parse_progress(line)
                    if event is not None:
                        progress.update(gpu, event)
        if process.returncode:
            raise RuntimeError(f"{variant}, seed={seed} failed; see its log")
        result = json.loads((folder / "result.json").read_text())
        if result.get("method") != "FCRA-ablation" or result["variant"] != variant or result["seed"] != seed:
            raise ValueError("Worker identity mismatch")
        return result

    def save(result):
        rows.append(result)
        write_csv(output / "per_seed.csv", rows)
        progress.finish(result)

    try:
        with ProgressDisplay(gpus, len(jobs)) as progress:
            run_jobs(jobs, gpus, launch, save)
        summarize(rows, output, seeds)
    except BaseException:
        manifest.update(status="failed", completed_runs=len(rows))
        dump_json(output / "manifest.json", manifest)
        raise
    manifest.update(status="complete", completed_runs=len(rows))
    dump_json(output / "manifest.json", manifest)
    print(f"Complete: {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
