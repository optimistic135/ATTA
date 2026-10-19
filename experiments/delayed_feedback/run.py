"""Run twelve isolated PACS or VLCS experiments and summarize their measured results.

Usage (from repository root): python -m experiments.delayed_feedback.run
Model dependencies are imported only inside a worker; --help uses the standard library.
"""

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone

from .scheduler import run_jobs
from .progress import ProgressDisplay, emit_progress, parse_progress
from .datasets import EXPERIMENTS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = EXPERIMENTS["PACS"].checkpoint_dir
DELAYS = (0, 10, 20, 30)
METRICS = EXPERIMENTS["PACS"].metrics


def parser(default_dataset="PACS"):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset", choices=tuple(EXPERIMENTS), default=default_dataset)
    cli.add_argument("--dataset-root", type=Path, default=ROOT / "storage/datasets",
                     help="Parent of the PACS/ or VLCS/ directory")
    cli.add_argument("--checkpoint-dir", type=Path,
                     help="Source checkpoint directory (default: dataset-specific round1)")
    cli.add_argument("--output-dir", type=Path,
                     help="New directory; existing directories are never overwritten")
    cli.add_argument("--seeds", type=int, nargs=3,
                     help="Optional three distinct seeds to reproduce a previous sweep")
    devices = cli.add_mutually_exclusive_group()
    devices.add_argument("--gpu-idx", type=int, default=None,
                         help="Single GPU index (default: 0)")
    devices.add_argument("--gpus", type=int, nargs="+",
                         help="Run one independent experiment per GPU, e.g. --gpus 0 1")
    cli.add_argument("--num-workers", type=int, default=0)
    cli.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    cli.add_argument("--delay", type=int, choices=DELAYS, help=argparse.SUPPRESS)
    cli.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    cli.add_argument("--require-cuda", action="store_true", help=argparse.SUPPRESS)
    return cli


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(args):
    experiment = EXPERIMENTS[args.dataset]
    folder = args.dataset_root / experiment.name
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing dataset: {folder}; automatic download is disabled")
    # The original ImageFolder loader uses sorted folder names as numeric IDs.
    # Reject extra/reordered domains so source weights cannot silently mismatch.
    actual = tuple(sorted(path.name for path in folder.iterdir() if path.is_dir()))
    if actual not in experiment.layouts:
        raise ValueError(f"Unexpected {experiment.name} domains in {folder}: {actual}; "
                         f"supported sorted layouts: {experiment.layouts}")
    for name in experiment.checkpoint_names:
        path = args.checkpoint_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing source checkpoint: {path}; automatic training is disabled")
    if args.num_workers < 0:
        raise ValueError("num-workers must be nonnegative")


def run_worker(args):
    multiprocessing.set_start_method("spawn", force=True)
    import torch
    from ATTA import args_parser, config_summoner
    from ATTA.utils.initial import reset_random_seed
    from .simatta import DelayedSimATTA

    if args.seed is None or args.delay is None or args.output_dir is None:
        raise ValueError("Internal worker requires seed, delay and output-dir")
    if args.gpu_idx is None or args.gpu_idx < 0 or args.gpus is not None:
        raise ValueError("Internal worker requires one nonnegative --gpu-idx")
    if torch.cuda.is_available():
        if args.gpu_idx >= torch.cuda.device_count():
            raise ValueError(f"GPU {args.gpu_idx} is not visible to this process")
        torch.cuda.set_device(args.gpu_idx)
    elif args.require_cuda:
        raise RuntimeError("--gpus requires CUDA; no CUDA GPU is available")
    validate_inputs(args)
    experiment = EXPERIMENTS[args.dataset]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = config_summoner(args_parser([
        "--config_path", str(experiment.config_path),
        "--exp_round", "1", "--ckpt_dir", str(args.checkpoint_dir),
        "--dataset.dataset_root", str(args.dataset_root),
        "--gpu_idx", str(args.gpu_idx), "--num_workers", str(args.num_workers),
    ]))
    # process_configs derives a seed from exp_round. Override AFTER it runs so
    # all workers load round1 weights while honoring the requested stream seed.
    config.random_seed = args.seed
    if (config.dataset.name != experiment.name
            or tuple(config.dataset.test_envs) != experiment.environments):
        raise ValueError(f"This experiment requires {experiment.name}: "
                         f"{' -> '.join((experiment.source,) + experiment.targets)}")
    if config.atta.batch_size != 100:
        raise ValueError("This experiment requires batch_size=100")
    reset_random_seed(config)
    dump_json(args.output_dir / "settings.json", {
        "seed": args.seed, "delay": args.delay, "batch_size": 100,
        "dataset": experiment.name,
        "dataset_root": str(args.dataset_root), "checkpoint_dir": str(args.checkpoint_dir),
        "device": str(config.device), "gpu_idx": args.gpu_idx,
        "torch_version": torch.__version__,
        "simatta": dict(config.atta.SimATTA), "num_workers": config.num_workers,
        "source": experiment.source, "target_order": list(experiment.targets),
    })
    algorithm = DelayedSimATTA(config, args.delay)
    result = algorithm.run_stream(args.output_dir, progress=emit_progress)
    dump_json(args.output_dir / "result.json", result)


def summarize(rows, output_dir, dataset="PACS"):
    experiment = EXPERIMENTS[dataset]
    summaries = []
    for delay in DELAYS:
        group = [row for row in rows if row["delay"] == delay]
        if len(group) != 3 or len({row["seed"] for row in group}) != 3:
            raise ValueError(f"D={delay} must have three distinct, completed seeds")
        summary = {"delay": delay, "runs": len(group)}
        for metric in experiment.metrics:
            values = [row[metric] for row in group if row[metric] is not None]
            summary[f"{metric}_mean"] = statistics.mean(values) if values else None
            summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
            summary[f"{metric}_n"] = len(values)
        summaries.append(summary)
    write_csv(output_dir / "per_seed.csv", rows)
    write_csv(output_dir / "summary.csv", summaries)

    def cell(summary, metric):
        mean, std = summary[f"{metric}_mean"], summary[f"{metric}_std"]
        if mean is None:
            return "N/A"
        return f"{mean:.2f} ± {std:.2f}" if std is not None else f"{mean:.2f}"

    columns = [(f"{domain}准确率(%)", f"{domain}_accuracy_pct")
               for domain in experiment.targets] + [
        ("全流准确率(%)", "stream_accuracy_pct"),
        ("查询数", "queried"), ("返回数", "returned"),
        ("跨域返回数", "cross_domain_returned"), ("跨域比例(%)", "cross_domain_pct"),
        ("未返回数", "pending"),
    ]
    table = ["| D(batch) | " + " | ".join(name for name, _ in columns) + " |",
             "| " + " | ".join(["---:"] * (len(columns) + 1)) + " |"]
    for row in summaries:
        table.append(f"| {row['delay']} | " + " | ".join(cell(row, key) for _, key in columns) + " |")
    text = "\n".join(table) + "\n"
    (output_dir / "summary.md").write_text(
        f"# {experiment.name} SimATTA 标签延迟反馈\n\n"
        + " → ".join((experiment.source,) + experiment.targets)
        + "；batch size=100。数值为三个 seed 的均值 ± 样本标准差。\n\n"
        + text + "\n跨域比例＝跨域返回数/已返回数；未返回标签不计入分母。"
        "比例先按 seed 计算再取均值；无返回的 seed 记为 N/A，summary.csv 记录有效 n。\n",
        encoding="utf-8",
    )
    print(text, flush=True)


def main(default_dataset="PACS"):
    args = parser(default_dataset).parse_args()
    experiment = EXPERIMENTS[args.dataset]
    args.dataset_root = args.dataset_root.resolve()
    args.checkpoint_dir = (args.checkpoint_dir or experiment.checkpoint_dir).resolve()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()
    if args.worker:
        run_worker(args)
        return
    validate_inputs(args)
    gpus = args.gpus if args.gpus is not None else [args.gpu_idx if args.gpu_idx is not None else 0]
    if len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("Provide distinct nonnegative GPU indices")
    seeds = args.seeds if args.seeds is not None else random.SystemRandom().sample(range(1, 2**31), 3)
    if len(set(seeds)) != 3 or any(not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Provide three distinct seeds in [0, 2**32)")
    output_base = ROOT / "storage/delayed_feedback"
    if experiment.name != "PACS":
        output_base = output_base / experiment.name
    output = args.output_dir or (output_base /
                                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    dump_json(output / "manifest.json", {
        "seeds": seeds, "delays": list(DELAYS), "git_commit": revision,
        "dataset": experiment.name,
        "dataset_root": str(args.dataset_root), "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_sha256": {name: sha256(args.checkpoint_dir / name)
                              for name in experiment.checkpoint_names},
        "python": sys.version, "batch_size": 100,
        "gpus": gpus, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source": experiment.source, "target_order": list(experiment.targets),
    })
    print(f"Seeds: {seeds}\nGPUs: {gpus}\nResults: {output}", flush=True)
    rows = []

    def launch(job, gpu):
        delay, seed = job
        run_dir = output / f"D{delay}" / f"seed_{seed}"
        command = [sys.executable, "-m", "experiments.delayed_feedback.run", "--worker",
                   "--dataset", experiment.name,
                   "--delay", str(delay), "--seed", str(seed),
                   "--dataset-root", str(args.dataset_root),
                   "--checkpoint-dir", str(args.checkpoint_dir),
                   "--output-dir", str(run_dir), "--gpu-idx", str(gpu),
                   "--num-workers", str(args.num_workers)]
        if args.gpus is not None:
            command.append("--require-cuda")
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
            raise RuntimeError(f"D={delay}, seed={seed}, GPU={gpu} failed "
                               f"(exit {process.returncode}); see {log_path}")
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        if result["delay"] != delay or result["seed"] != seed:
            raise ValueError(f"Worker result identity mismatch: {run_dir}")
        return result

    def save_result(result):
        rows.append(result)
        # Only the parent coordinator writes shared tables, regardless of finish order.
        rows.sort(key=lambda row: (row["delay"], seeds.index(row["seed"])))
        write_csv(output / "per_seed.csv", rows)
        progress.finish(result)

    jobs = [(delay, seed) for delay in DELAYS for seed in seeds]
    with ProgressDisplay(gpus, len(jobs)) as progress:
        run_jobs(jobs, gpus, launch, save_result)
    summarize(rows, output, dataset=experiment.name)


if __name__ == "__main__":
    main()
