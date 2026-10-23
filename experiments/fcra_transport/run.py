"""Run only calibrated transport on recurrent PACS: D=0,50,100 and three seeds."""

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
from .spec import METRICS, STAGES


DELAYS = (0, 50, 100)
BATCH_SIZE = 32
EXPERIMENT = EXPERIMENTS["PACS"]


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset-root", type=Path, default=ROOT / "storage/datasets")
    cli.add_argument("--checkpoint-dir", type=Path, default=EXPERIMENT.checkpoint_dir)
    cli.add_argument("--output-dir", type=Path,
                     help="New results directory; existing results are never overwritten")
    cli.add_argument("--seeds", type=int, nargs=3, default=[110, 123, 456],
                     help="Three distinct seeds shared by all delays (default: 110 123 456)")
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
    from .algorithm import CalibratedTransport
    from .policy import ACTIONS, DEFAULTS
    from .stream import run_stream

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
        raise ValueError("Transport requires the PACS Photo-source configuration and batch size 32")
    reset_random_seed(config)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dump_json(args.output_dir / "settings.json", {
        "method": "FCRA-Transport", "dataset": "PACS", "seed": args.seed, "delay": args.delay,
        "batch_size": BATCH_SIZE, "train_batch_size": config.train.train_bs,
        "source": EXPERIMENT.source, "target_order": [name for name, _ in STAGES],
        "dataset_root": str(args.dataset_root), "checkpoint_dir": str(args.checkpoint_dir),
        "transport": DEFAULTS, "actions": ACTIONS,
        "simatta": dict(config.atta.SimATTA),
        "gpu_clustering": config.atta.gpu_clustering,
        "clustering": "GPU KMeans" if config.atta.gpu_clustering else "sklearn CPU KMeans",
        "num_workers": config.num_workers, "device": str(config.device),
        "gpu_idx": args.gpu_idx, "gpu_name": torch.cuda.get_device_name(args.gpu_idx),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
    })
    algorithm = CalibratedTransport(config, args.delay)
    result = run_stream(algorithm, args.output_dir, emit_progress)
    dump_json(args.output_dir / "result.json", result)


def summarize(rows, output, seeds):
    """Only the requested method, three delays and three common seeds."""
    expected = {(delay, seed) for delay in DELAYS for seed in seeds}
    actual = [(row["delay"], row["seed"]) for row in rows]
    if (len(actual) != len(expected) or set(actual) != expected
            or any(row.get("method") != "FCRA-Transport" for row in rows)):
        raise ValueError("Summary requires the 9 distinct, completed FCRA jobs")
    summaries = []
    for delay in DELAYS:
        group = [row for row in rows if row["delay"] == delay]
        summary = {"method": "FCRA-Transport", "delay": delay, "runs": len(group)}
        for metric in METRICS:
            values = [row[metric] for row in group if row[metric] is not None]
            summary[f"{metric}_mean"] = statistics.mean(values) if values else None
            summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
            summary[f"{metric}_n"] = len(values)
        summaries.append(summary)
    write_csv(output / "per_seed.csv", rows)
    write_csv(output / "summary.csv", summaries)
    columns = [(f"{domain} accuracy (%)", f"{domain}_accuracy_pct")
               for domain, _ in STAGES] + [
        ("Stream accuracy (%)", "stream_accuracy_pct"),
        ("Domain macro accuracy (%)", "domain_macro_accuracy_pct"),
        ("Stage macro accuracy (%)", "stage_macro_accuracy_pct"),
        ("Queried", "queried"), ("Returned", "returned"),
        ("Cross-domain returned", "cross_domain_returned"),
        ("Cross-domain (%)", "cross_domain_pct"), ("Pending", "pending"),
        ("Cross-stage (%)", "cross_stage_pct"),
        ("Main updates", "main_updates"), ("Audit updates", "audit_updates"),
        ("Transport updates", "transport_updates"),
        ("Audited queries", "audited_queries"), ("Returned audits", "returned_audits"),
        ("Before feedback (%)", "before_feedback_accuracy_pct"),
        ("After feedback (%)", "after_feedback_accuracy_pct"),
    ]

    def cell(row, key):
        mean, std = row[f"{key}_mean"], row[f"{key}_std"]
        if mean is None:
            return "N/A"
        return f"{mean:.2f} ± {std:.2f}" if std is not None else f"{mean:.2f}"

    table = ["| D (batch) | " + " | ".join(name for name, _ in columns) + " |",
             "| " + " | ".join(["---:"] * (len(columns) + 1)) + " |"]
    table.extend(f"| {row['delay']} | " + " | ".join(cell(row, key) for _, key in columns) + " |"
                 for row in summaries)
    text = "\n".join(table) + "\n"
    (output / "summary.md").write_text(
        "# FCRA-Transport: recurrent PACS\n\nPhoto(source) → Art_1 → Cartoon → Sketch → Art_2 → Photo; batch size=32.\n\n"
        f"Common seeds: {', '.join(str(seed) for seed in seeds)}. "
        "Values are mean ± sample standard deviation (ddof=1).\n\n" + text +
        "\nAccuracy is scored before feedback/update. Stream accuracy is sample-weighted; "
        "domain macro first pools both Art visits and gives Art/Cartoon/Sketch/Photo equal weight; "
        "stage macro gives five visits equal weight. "
        "Cross-domain percentage is computed per seed as cross-domain returned / returned; "
        "pending labels are excluded. A seed with no returns has N/A for this percentage; "
        "summary.csv records the valid n for every metric. "
        "Cross-stage counts different visits even when their semantic domain is the same. "
        "Art_2 repeats physical Art images, without re-querying previously queried images; "
        "final Photo is source-return performance, not held-out source generalization. "
        "Only this method is run; no baseline gain or budget-equivalence claim is made.\n",
        encoding="utf-8")
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
    output = args.output_dir or (ROOT / "storage/fcra_transport" /
                                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "method": "FCRA-Transport", "dataset": "PACS", "status": "running",
        "seeds": seeds, "delays": list(DELAYS), "batch_size": BATCH_SIZE,
        "git_commit": revision, "dataset_root": str(args.dataset_root),
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_sha256": {name: sha256(args.checkpoint_dir / name)
                              for name in EXPERIMENT.checkpoint_names},
        "python": sys.version, "gpus": gpus, "num_workers": args.num_workers,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source": EXPERIMENT.source, "target_order": [name for name, _ in STAGES],
        "protocol": "predict before release/update; audit before new release including D=0; "
                    "one query per physical image; A1,C,S,A2,P; no final flush; "
                    "30 combined main/audit optimizer steps per batch upper bound",
    }
    dump_json(output / "manifest.json", manifest)
    print(f"Method: FCRA-Transport\nSeeds: {seeds}\nGPUs: {gpus}\nResults: {output}", flush=True)
    rows = []

    def launch(job, gpu):
        delay, seed = job
        run_dir = output / f"D{delay}" / f"seed_{seed}"
        command = [sys.executable, "-m", "experiments.fcra_transport.run", "--worker",
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
            raise RuntimeError(f"FCRA-Transport D={delay}, seed={seed}, GPU={gpu} failed "
                               f"(exit {process.returncode}); see {log_path}")
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        if result.get("method") != "FCRA-Transport" or result["delay"] != delay or result["seed"] != seed:
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
