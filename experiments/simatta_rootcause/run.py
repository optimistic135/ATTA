"""Diagnose original SimATTA at D=100 on PACS with three paired-objective seed runs."""

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone

from experiments.delayed_feedback.datasets import EXPERIMENTS, ROOT
from experiments.delayed_feedback.progress import ProgressDisplay, emit_progress, parse_progress
from experiments.delayed_feedback.run import dump_json, sha256, validate_inputs, write_csv
from experiments.delayed_feedback.scheduler import run_jobs
from .spec import ARMS, BASELINE_METRICS, PROBE_METRICS, DEFAULTS, METHOD

DELAYS = (100,)
METRICS = BASELINE_METRICS + PROBE_METRICS
STAGES = (("Art", 0), ("Cartoon", 1), ("Sketch", 3))


BATCH_SIZE = 32
EXPERIMENT = EXPERIMENTS["PACS"]


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset-root", type=Path, default=ROOT / "storage/datasets")
    cli.add_argument("--checkpoint-dir", type=Path, default=EXPERIMENT.checkpoint_dir)
    cli.add_argument("--output-dir", type=Path,
                     help="New results directory; existing results are never overwritten")
    cli.add_argument("--seeds", type=int, nargs=3, default=[110, 123, 456],
                     help="Three distinct D=100 seeds (default: 110 123 456)")
    devices = cli.add_mutually_exclusive_group()
    devices.add_argument("--gpu-idx", type=int, help="One GPU (default: 0)")
    devices.add_argument("--gpus", type=int, nargs="+", help="One worker per GPU, e.g. 0 1")
    cli.add_argument("--num-workers", type=int, default=0)
    cli.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    cli.add_argument("--delay", type=int, choices=DELAYS, help=argparse.SUPPRESS)
    cli.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    return cli


def run_worker(args):
    multiprocessing.set_start_method("spawn", force=True)
    import torch
    from ATTA import args_parser, config_summoner
    from ATTA.utils.initial import reset_random_seed
    from .algorithm import RootCauseSimATTA

    if (args.delay is None or args.seed is None or args.output_dir is None
            or args.gpu_idx is None or args.gpu_idx < 0 or args.gpus is not None):
        raise ValueError("Internal worker requires delay, seed, output-dir and one gpu-idx")
    if not 0 <= args.seed < 2**32:
        raise ValueError("Seed must be in [0, 2**32)")
    if not torch.cuda.is_available() or args.gpu_idx >= torch.cuda.device_count():
        raise RuntimeError(f"A visible CUDA GPU is required; requested GPU {args.gpu_idx}")
    torch.cuda.set_device(args.gpu_idx)
    validate_inputs(args)
    config = config_summoner(args_parser([
        "--config_path", str(EXPERIMENT.config_path), "--exp_round", "1",
        "--ckpt_dir", str(args.checkpoint_dir),
        "--dataset.dataset_root", str(args.dataset_root),
        "--gpu_idx", str(args.gpu_idx), "--num_workers", str(args.num_workers),
        "--atta.batch_size", str(BATCH_SIZE),
    ]))
    # exp_round selects the fixed round1 weights but also derives seed=110.
    # Override only after config processing, before model and loaders are built.
    config.random_seed = args.seed
    if (config.dataset.name != "PACS"
            or tuple(config.dataset.test_envs) != EXPERIMENT.environments
            or config.atta.batch_size != BATCH_SIZE):
        raise ValueError("Diagnosis requires the PACS Photo-source configuration and batch size 32")
    reset_random_seed(config)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dump_json(args.output_dir / "settings.json", {
        "method": "SimATTA-D100-diagnosis", "dataset": "PACS", "seed": args.seed, "delay": args.delay,
        "batch_size": BATCH_SIZE, "train_batch_size": config.train.train_bs,
        "source": EXPERIMENT.source, "target_order": [name for name, _ in STAGES],
        "dataset_root": str(args.dataset_root), "checkpoint_dir": str(args.checkpoint_dir),
        "diagnosis": DEFAULTS,
        "simatta": dict(config.atta.SimATTA),
        "gpu_clustering": config.atta.gpu_clustering,
        "clustering": "GPU KMeans" if config.atta.gpu_clustering else "sklearn CPU KMeans",
        "num_workers": config.num_workers, "device": str(config.device),
        "gpu_idx": args.gpu_idx, "gpu_name": torch.cuda.get_device_name(args.gpu_idx),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
    })
    algorithm = RootCauseSimATTA(config)
    result = algorithm.run_stream(args.output_dir, progress=emit_progress)
    dump_json(args.output_dir / "result.json", result)


def summarize(rows, output, seeds):
    """Average paired effects within seed first; seed is the replicate unit."""
    expected = {(100, seed) for seed in seeds}
    actual = [(row["delay"], row["seed"]) for row in rows]
    if len(actual) != 3 or set(actual) != expected or any(row["method"] != METHOD for row in rows):
        raise ValueError("Diagnosis requires three completed original SimATTA D=100 seed runs")
    summary = {"method": METHOD, "delay": 100, "runs": 3}
    for metric in METRICS:
        values = [row[metric] for row in rows if row.get(metric) is not None]
        summary[f"{metric}_mean"] = statistics.mean(values) if values else None
        summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
        summary[f"{metric}_n"] = len(values)
    write_csv(output / "per_seed.csv", rows)
    write_csv(output / "summary.csv", [summary])
    write_csv(output / "paired_per_seed.csv", [
        {"seed": row["seed"], "delay": 100, **{key: row.get(key) for key in PROBE_METRICS}} for row in rows])

    def cell(key):
        mean, std = summary[f"{key}_mean"], summary[f"{key}_std"]
        if mean is None:
            return "N/A"
        return f"{mean:.2f} ± {std:.2f}" if std is not None else f"{mean:.2f}"

    baseline_columns = [("Art (%)", "Art_accuracy_pct"), ("Cartoon (%)", "Cartoon_accuracy_pct"),
                        ("Sketch (%)", "Sketch_accuracy_pct"), ("Stream (%)", "stream_accuracy_pct"),
                        ("Macro (%)", "domain_macro_accuracy_pct"), ("Queried", "queried"),
                        ("Returned", "returned"), ("Pending", "pending"), ("Returned (%)", "returned_pct"),
                        ("Cross-domain (%)", "cross_domain_pct"), ("Main updates", "main_updates")]
    baseline = ["## Original SimATTA main trajectory\n",
                "| D | " + " | ".join(name for name, _ in baseline_columns) + " |",
                "| " + " | ".join(["---:"] * (len(baseline_columns) + 1)) + " |",
                "| 100 | " + " | ".join(cell(key) for _, key in baseline_columns) + " |"]
    arms = ["## Frozen short-update branches: next ten batches\n",
            "| Arm | Future accuracy (%) | Gain vs SKIP (pp) |",
            "| :--- | ---: | ---: |"]
    for arm in ARMS:
        gain = ("0.00" if summary["SKIP_future_accuracy_pct_mean"] is not None else "N/A") if arm == "SKIP" else cell(f"{arm}_minus_SKIP_pp")
        arms.append(f"| {arm} | {cell(f'{arm}_future_accuracy_pct')} | {gain} |")
    notes = (f"\n\nTRUE − MIX: {cell('TRUE_minus_MIX_pp')} pp. "
             f"Initial true/pseudo gradient cosine: {cell('gradient_cosine')}. "
             f"Weighted pseudo/true gradient norm ratio: {cell('weighted_pseudo_to_true_norm_ratio')}. "
             f"Complete probes per seed: {cell('complete_probes')}.\n\n"
             "Probe branches do not change the main trajectory. TRUE/MIX share the complete returned truth pool "
             "and draws; PSEUDO/MIX share the original pseudo pool and draws. SKIP receives no update. "
             "Each other branch receives ten SGD steps, then is frozen for ten future batches. "
             "Future truth is used only for scoring. These accuracies are conditional local diagnostics, "
             "not alternative deployed algorithms or whole-stream gains.\n\n"
             "Effects are sample-weighted over complete windows within each seed, then averaged across seeds "
             "with sample standard deviation. Probe events are not independent seed replicates. "
             "Missing or incomplete windows are excluded; summary.csv records valid n. "
             "A TRUE/MIX difference can reflect both weighting and gradient direction, not just conflict.\n")
    text = ("# SimATTA D=100: simple root-cause diagnosis\n\n"
            "Photo(source) → Art → Cartoon → Sketch; batch=32. "
            f"Seeds: {seeds}. Mean ± sample standard deviation.\n\n"
            + "\n".join(baseline) + "\n\n" + "\n".join(arms) + notes)
    (output / "summary.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main():
    args = parser().parse_args()
    args.dataset = "PACS"
    args.dataset_root = args.dataset_root.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()
    if args.worker:
        run_worker(args)
        return
    if args.delay is not None or args.seed is not None:
        raise ValueError("Use --seeds for the sweep; delay/seed are worker-only options")
    validate_inputs(args)
    gpus = args.gpus if args.gpus is not None else [args.gpu_idx if args.gpu_idx is not None else 0]
    if len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("Provide distinct nonnegative GPU indices")
    seeds = (args.seeds if args.seeds is not None
             else random.SystemRandom().sample(range(1, 2**31), 3))
    if len(set(seeds)) != 3 or any(not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Provide three distinct seeds in [0, 2**32)")
    output = args.output_dir or (ROOT / "storage/simatta_rootcause" /
                                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "method": "SimATTA-D100-diagnosis", "dataset": "PACS", "status": "running",
        "seeds": seeds, "delays": list(DELAYS), "batch_size": BATCH_SIZE,
        "git_commit": revision, "dataset_root": str(args.dataset_root),
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_sha256": {name: sha256(args.checkpoint_dir / name)
                              for name in EXPERIMENT.checkpoint_names},
        "python": sys.version, "gpus": gpus, "num_workers": args.num_workers,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source": EXPERIMENT.source, "target_order": [name for name, _ in STAGES],
        "protocol": "original delayed SimATTA prediction/query/early-stop training; D=100; "
                    "up to five isolated TRUE/PSEUDO/MIX/SKIP probes; ten short updates; "
                    "freeze and evaluate next ten batches; no future truth in training; no final flush",
        "diagnosis": DEFAULTS,
    }
    dump_json(output / "manifest.json", manifest)
    print(f"Method: SimATTA-D100-diagnosis\nSeeds: {seeds}\nGPUs: {gpus}\nResults: {output}", flush=True)
    rows = []

    def launch(job, gpu):
        delay, seed = job
        run_dir = output / f"D{delay}" / f"seed_{seed}"
        command = [sys.executable, "-m", "experiments.simatta_rootcause.run", "--worker",
                   "--delay", str(delay), "--seed", str(seed),
                   "--dataset-root", str(args.dataset_root),
                   "--checkpoint-dir", str(args.checkpoint_dir),
                   "--output-dir", str(run_dir), "--gpu-idx", str(gpu),
                   "--num-workers", str(args.num_workers)]
        progress.start(gpu, delay, seed)
        log_path = output / f"D{delay}_seed_{seed}.log"
        with log_path.open("w", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True,
                                  encoding="utf-8", errors="replace", bufsize=1) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    event = parse_progress(line)
                    if event is not None:
                        progress.update(gpu, event)
        if process.returncode:
            raise RuntimeError(f"SimATTA-D100-diagnosis D={delay}, seed={seed}, GPU={gpu} failed "
                               f"(exit {process.returncode}); see {log_path}")
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        if result.get("method") != "SimATTA-D100-diagnosis" or result["delay"] != delay or result["seed"] != seed:
            raise ValueError(f"Worker result identity mismatch: {run_dir}")
        return result

    def save_result(result):
        rows.append(result)
        rows.sort(key=lambda row: (row["delay"], seeds.index(row["seed"])))
        write_csv(output / "per_seed.csv", rows)
        progress.finish(result)

    try:
        jobs = [(delay, seed) for delay in DELAYS for seed in seeds]
        with ProgressDisplay(gpus, len(jobs)) as progress:
            run_jobs(jobs, gpus, launch, save_result)
        summarize(rows, output, seeds)
    except BaseException:
        manifest["status"] = "failed"
        manifest["completed_runs"] = len(rows)
        dump_json(output / "manifest.json", manifest)
        raise
    manifest["status"] = "complete"
    manifest["completed_runs"] = len(rows)
    dump_json(output / "manifest.json", manifest)
    print(f"Complete: {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
