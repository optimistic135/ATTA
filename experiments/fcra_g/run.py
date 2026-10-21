"""Run FCRA-G on PACS with recorded FCRA queries and per-step update budgets."""

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone

from experiments.delayed_feedback.datasets import EXPERIMENTS, ROOT
from experiments.delayed_feedback.progress import ProgressDisplay, emit_progress, parse_progress
from experiments.delayed_feedback.run import dump_json, sha256, validate_inputs, write_csv
from experiments.delayed_feedback.scheduler import run_jobs
from experiments.fcra_dual.reference import ReferenceTrace


METHOD = "FCRA-G"
BASE_METHOD = "FCRA"
DELAYS = (0, 20, 50, 100)
BATCH_SIZE = 32
EXPERIMENT = EXPERIMENTS["PACS"]
EXTRA_METRICS = ("reference_updates", "reference_queries", "actual_supervised_updates")


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset-root", type=Path, default=ROOT / "storage/datasets")
    cli.add_argument("--checkpoint-dir", type=Path, default=EXPERIMENT.checkpoint_dir)
    cli.add_argument("--output-dir", type=Path,
                     help="New results directory; existing results are never overwritten")
    cli.add_argument("--seeds", type=int, nargs=3,
                     help="Exactly the reference seeds; default: reference manifest seeds")
    devices = cli.add_mutually_exclusive_group()
    devices.add_argument("--gpu-idx", type=int, help="One GPU (default: 0)")
    devices.add_argument("--gpus", type=int, nargs="+", help="One worker per GPU, e.g. 0 1")
    cli.add_argument("--num-workers", type=int, default=0)
    cli.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    cli.add_argument("--delay", type=int, choices=DELAYS, help=argparse.SUPPRESS)
    cli.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    cli.add_argument("--reference-dir", type=Path, required=True,
                     help="Completed full-memory FCRA sweep: manifest + D*/seed_* traces")
    return cli


def load_reference(args):
    manifest = json.loads((args.reference_dir / "manifest.json").read_text())
    seeds = manifest["seeds"]
    if (manifest.get("method") != BASE_METHOD or manifest.get("status") != "complete"
            or manifest["dataset"] != "PACS" or manifest["batch_size"] != BATCH_SIZE
            or tuple(manifest["delays"]) != DELAYS or len(seeds) != 3
            or len(set(seeds)) != 3 or any(not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError("Reference must be a complete PACS FCRA four-delay/three-seed sweep")
    for name in EXPERIMENT.checkpoint_names:
        if sha256(args.checkpoint_dir / name) != manifest["checkpoint_sha256"][name]:
            raise ValueError(f"Reference source checkpoint hash differs: {name}")
    if args.seeds is not None and (len(set(args.seeds)) != 3 or set(args.seeds) != set(seeds)):
        raise ValueError("Use exactly the reference's three seeds; a new seed breaks matching")
    return manifest, args.seeds if args.seeds is not None else seeds


def run_worker(args):
    multiprocessing.set_start_method("spawn", force=True)
    import torch
    from ATTA import args_parser, config_summoner
    from ATTA.utils.initial import reset_random_seed
    from experiments.fcra.algorithm import DEFAULTS as CORE_DEFAULTS
    from .algorithm import DEFAULTS, GraphFCRA

    if (args.delay is None or args.seed is None or args.output_dir is None
            or args.gpu_idx is None or args.gpu_idx < 0 or args.gpus is not None):
        raise ValueError("Internal worker requires delay, seed, output-dir and one gpu-idx")
    if not 0 <= args.seed < 2**32:
        raise ValueError("Seed must be in [0, 2**32)")
    if not torch.cuda.is_available() or args.gpu_idx >= torch.cuda.device_count():
        raise RuntimeError(f"A visible CUDA GPU is required; requested GPU {args.gpu_idx}")
    torch.cuda.set_device(args.gpu_idx)
    validate_inputs(args)
    _, seeds = load_reference(args)
    if args.seed not in seeds:
        raise ValueError("Worker seed is not in the reference sweep")
    trace = ReferenceTrace(args.reference_dir / f"D{args.delay}" / f"seed_{args.seed}",
                           args.delay, args.seed)
    parent = json.loads((args.output_dir.parent.parent / "manifest.json").read_text())
    key = f"D{args.delay}/seed_{args.seed}"
    if (parent["reference_hashes"][key] != trace.hashes
            or parent["reference_manifest_sha256"] != sha256(args.reference_dir / "manifest.json")):
        raise ValueError("Reference changed after sweep preflight")
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
            or config.atta.batch_size != BATCH_SIZE
            or config.train.train_bs != trace.settings["train_batch_size"]
            or dict(config.atta.SimATTA) != trace.settings["simatta"]):
        raise ValueError("FCRA-G requires PACS Photo -> Art -> Cartoon -> Sketch, batch size 32")
    for name in ("replay_mix", "replay_neighbors", "epsilon", "teacher_cache_size",
                 "gate_neighbors", "gate_similarity", "gate_temperature", "gate_prior"):
        if trace.settings["fcra"][name] != CORE_DEFAULTS[name]:
            raise ValueError(f"Inherited FCRA setting differs: {name}")
    reset_random_seed(config)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dump_json(args.output_dir / "settings.json", {
        "method": METHOD, "base_method": BASE_METHOD,
        "dataset": "PACS", "seed": args.seed, "delay": args.delay,
        "batch_size": BATCH_SIZE, "train_batch_size": config.train.train_bs,
        "source": EXPERIMENT.source, "target_order": list(EXPERIMENT.targets),
        "dataset_root": str(args.dataset_root), "checkpoint_dir": str(args.checkpoint_dir),
        "reference_dir": str(args.reference_dir), "reference_hashes": trace.hashes,
        "reference_updates": trace.total, "reference_queries": len(trace.records),
        "fcra": CORE_DEFAULTS, "fcra_g": DEFAULTS, "simatta": dict(config.atta.SimATTA),
        "gpu_clustering": config.atta.gpu_clustering,
        "clustering": "GPU KMeans" if config.atta.gpu_clustering else "sklearn CPU KMeans",
        "num_workers": config.num_workers, "device": str(config.device),
        "gpu_idx": args.gpu_idx, "gpu_name": torch.cuda.get_device_name(args.gpu_idx),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
    })
    write_csv(args.output_dir / "reference_budget.csv", [
        {"step": step, "reference_updates": trace.old_updates[step],
         "reference_queries": len(trace.queries[step])}
        for step in range(trace.steps)
    ])
    algorithm = GraphFCRA(config, args.delay, trace)
    result = algorithm.run_stream(args.output_dir, progress=emit_progress)
    dump_json(args.output_dir / "result.json", result)


def summarize(rows, output, seeds):
    """Only complete, common-three-seed FCRA-G results can form the main table."""
    expected = {(delay, seed) for delay in DELAYS for seed in seeds}
    actual = [(row["delay"], row["seed"]) for row in rows]
    if (len(actual) != len(expected) or set(actual) != expected
            or any(row.get("method") != METHOD for row in rows)):
        raise ValueError("Summary requires the 12 distinct, completed FCRA-G jobs")
    summaries = []
    for delay in DELAYS:
        group = [row for row in rows if row["delay"] == delay]
        summary = {"method": METHOD, "delay": delay, "runs": len(group)}
        for metric in EXPERIMENT.metrics + EXTRA_METRICS:
            values = [row[metric] for row in group if row[metric] is not None]
            summary[f"{metric}_mean"] = statistics.mean(values) if values else None
            summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
            summary[f"{metric}_n"] = len(values)
        summaries.append(summary)
    write_csv(output / "per_seed.csv", rows)
    write_csv(output / "summary.csv", summaries)
    columns = [(f"{domain} accuracy (%)", f"{domain}_accuracy_pct")
               for domain in EXPERIMENT.targets] + [
        ("Stream accuracy (%)", "stream_accuracy_pct"),
        ("Domain macro accuracy (%)", "domain_macro_accuracy_pct"),
        ("Queried", "queried"), ("Returned", "returned"),
        ("Cross-domain returned", "cross_domain_returned"),
        ("Cross-domain (%)", "cross_domain_pct"), ("Pending", "pending"),
        ("Reference updates", "reference_updates"),
        ("Actual updates", "actual_supervised_updates"),
        ("Reference queries", "reference_queries"),
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
        "# FCRA-G: PACS delayed feedback\n\nPhoto → Art → Cartoon → Sketch; batch size=32.\n\n"
        f"Common seeds: {', '.join(str(seed) for seed in seeds)}. "
        "Values are mean ± sample standard deviation (ddof=1).\n\n" + text +
        "\nAccuracy is scored before feedback/update. Stream accuracy is sample-weighted; "
        "domain macro accuracy gives each target domain equal weight. "
        "Cross-domain percentage is computed per seed as cross-domain returned / returned; "
        "pending labels are excluded. A seed with no returns has N/A for this percentage; "
        "summary.csv records the valid n for every metric. Query IDs/times and per-step "
        "supervised update counts are replayed from the reference FCRA sweep. D=0 uses the "
        "original FCRA prediction/training path without graph correction or expert fusion.\n",
        encoding="utf-8")
    print(text, flush=True)


def main():
    args = parser().parse_args()
    args.dataset = "PACS"
    args.dataset_root = args.dataset_root.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.reference_dir = args.reference_dir.resolve()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()
    if args.worker:
        run_worker(args)
        return
    if args.delay is not None or args.seed is not None:
        raise ValueError("Use --seeds for the sweep; delay/seed are worker-only options")
    validate_inputs(args)
    reference_manifest, seeds = load_reference(args)
    gpus = args.gpus if args.gpus is not None else [args.gpu_idx if args.gpu_idx is not None else 0]
    if len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("Provide distinct nonnegative GPU indices")
    output = args.output_dir or (ROOT / "storage/fcra_g" /
                                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    reference_hashes = {
        f"D{delay}/seed_{seed}": ReferenceTrace(
            args.reference_dir / f"D{delay}" / f"seed_{seed}", delay, seed).hashes
        for delay in DELAYS for seed in seeds
    }
    output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "method": METHOD, "base_method": BASE_METHOD, "dataset": "PACS", "status": "running",
        "seeds": seeds, "delays": list(DELAYS), "batch_size": BATCH_SIZE,
        "git_commit": revision, "dataset_root": str(args.dataset_root),
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_sha256": {name: sha256(args.checkpoint_dir / name)
                              for name in EXPERIMENT.checkpoint_names},
        "reference_dir": str(args.reference_dir),
        "reference_manifest_sha256": sha256(args.reference_dir / "manifest.json"),
        "reference_hashes": reference_hashes,
        "reference_git_commit": reference_manifest.get("git_commit"),
        "python": sys.version, "gpus": gpus, "num_workers": args.num_workers,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source": EXPERIMENT.source, "target_order": list(EXPERIMENT.targets),
        "protocol": "recorded FCRA query IDs/times and per-step update counts; "
                    "predict before release/update; complete returned-label memory; "
                    "delayed-label anchored graph correction before release; "
                    "D=0 bypasses graph/experts; no final flush",
    }
    dump_json(output / "manifest.json", manifest)
    print(f"Method: {METHOD}\nSeeds: {seeds}\nGPUs: {gpus}\nResults: {output}", flush=True)
    rows = []

    def launch(job, gpu):
        delay, seed = job
        run_dir = output / f"D{delay}" / f"seed_{seed}"
        command = [sys.executable, "-m", "experiments.fcra_g.run", "--worker",
                   "--delay", str(delay), "--seed", str(seed),
                   "--dataset-root", str(args.dataset_root),
                   "--checkpoint-dir", str(args.checkpoint_dir),
                   "--reference-dir", str(args.reference_dir),
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
            raise RuntimeError(f"FCRA-G D={delay}, seed={seed}, GPU={gpu} failed "
                               f"(exit {process.returncode}); see {log_path}")
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        if (result.get("method") != METHOD or result["delay"] != delay
                or result["seed"] != seed):
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
