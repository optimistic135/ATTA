"""Four controlled PACS D=0 learning variants, with three paired seeds."""

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


VARIANTS = ("simatta_control", "replay_only", "calibration_only", "full_fcra")
METRICS = tuple(f"{domain}_accuracy_pct" for domain in dual.EXPERIMENT.targets) + (
    "stream_accuracy_pct", "domain_macro_accuracy_pct", "queried", "returned", "slow_updates")
CONTRASTS = (
    ("replay_alone", "replay_only", "simatta_control"),
    ("calibration_alone", "calibration_only", "simatta_control"),
    ("replay_with_calibration", "full_fcra", "calibration_only"),
    ("calibration_with_replay", "full_fcra", "replay_only"),
    ("full_vs_control", "full_fcra", "simatta_control"),
)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset-root", type=Path, default=dual.ROOT / "storage/datasets")
    cli.add_argument("--checkpoint-dir", type=Path, default=dual.EXPERIMENT.checkpoint_dir)
    cli.add_argument("--reference-dir", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path)
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
    from .algorithm import AblationFCRA

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
    algorithm = AblationFCRA(config, trace, args.variant)
    dump_json(args.output_dir / "settings.json", {
        "method": "FCRA-ablation", "dataset": "PACS",
        "variant": args.variant, "seed": args.seed, "delay": 0, "batch_size": 32,
        "dataset_root": str(args.dataset_root), "checkpoint_dir": str(args.checkpoint_dir),
        "source": dual.EXPERIMENT.source, "target_order": list(dual.EXPERIMENT.targets),
        "train_batch_size": config.train.train_bs, "simatta": dict(config.atta.SimATTA),
        "fcra": DEFAULTS, "targeted_replay": algorithm.use_replay,
        "teacher_calibration": algorithm.use_calibration, "fast_adapter_enabled": False,
        "reference_hashes": trace.hashes, "reference_updates": trace.total,
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
        raise ValueError("Incomplete four-variant paired experiment")
    for seed in seeds:
        controls = [indexed[variant, seed] for variant in VARIANTS]
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
    lines = ["# PACS D=0: no-adapter controlled ablation", "",
             "Same source weights, stream seeds, batch=32, queries and per-batch update quotas.",
             "SimATTA-control uses its uncalibrated joint objective with uniform replay in a shared runner;",
             "it is not an end-to-end reproduction of the original query/early-stop implementation.", "",
             "| Variant | Art | Cartoon | Sketch | Stream | Macro |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summaries:
        cells = [f"{row[m + '_mean']:.2f} ± {row[m + '_std']:.2f}" for m in METRICS[:5]]
        lines.append("| " + row["variant"] + " | " + " | ".join(cells) + " |")
    lines += ["", "| Paired contrast | Stream delta (pp) | Positive seeds |",
              "| --- | ---: | ---: |"]
    for name in [item[0] for item in CONTRASTS] + ["interaction"]:
        values = [row["stream_accuracy_pct_delta_pp"] for row in paired if row["contrast"] == name]
        lines.append(f"| {name} | {statistics.mean(values):+.2f} ± {statistics.stdev(values):.2f} | "
                     f"{sum(value > 0 for value in values)}/{len(values)} |")
    lines += ["", "Mean ± sample SD; three paired seeds do not establish significance.",
              "Positive interaction means the combined gain exceeds the sum of isolated gains.",
              "Equal update counts do not imply equal FLOPs. No evaluation labels select modules or parameters."]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    reference, seeds = dual.load_reference(args)
    hashes = {str(seed): ReferenceTrace(args.reference_dir / "D0" / f"seed_{seed}", 0, seed).hashes
              for seed in seeds}
    gpus = args.gpus if args.gpus is not None else [args.gpu_idx if args.gpu_idx is not None else 0]
    if len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("Provide distinct nonnegative GPU indices")
    output = args.output_dir or dual.ROOT / "storage/fcra_ablation" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "method": "FCRA-ablation", "status": "running", "dataset": "PACS", "delay": 0,
        "seeds": seeds, "variants": VARIANTS, "reference_hashes": hashes,
        "reference_manifest_sha256": sha256(args.reference_dir / "manifest.json"),
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "dataset_root": str(args.dataset_root), "reference_dir": str(args.reference_dir),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dual.ROOT, text=True).strip(),
        "gpus": gpus, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "num_workers": args.num_workers,
        "protocol": "D0; no adapter; fixed queries and per-batch updates; two-factor learning ablation",
    }
    dump_json(output / "manifest.json", manifest)
    rows = []

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
        with ProgressDisplay(gpus, len(VARIANTS) * len(seeds)) as progress:
            run_jobs([(variant, seed) for variant in VARIANTS for seed in seeds], gpus, launch, save)
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
