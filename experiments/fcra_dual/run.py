"""PACS dual-timescale FCRA using an existing FCRA sweep as a fixed protocol."""

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone

from experiments.fcra import run as baseline
from experiments.delayed_feedback.progress import ProgressDisplay, emit_progress, parse_progress
from experiments.delayed_feedback.run import dump_json, sha256, validate_inputs, write_csv
from experiments.delayed_feedback.scheduler import run_jobs
from .reference import ReferenceTrace


ROOT, EXPERIMENT = baseline.ROOT, baseline.EXPERIMENT
DELAYS, BATCH_SIZE = baseline.DELAYS, baseline.BATCH_SIZE
METHOD = "FCRA-dual"
EXTRA_METRICS = tuple(f"{domain}_first10_accuracy_pct" for domain in EXPERIMENT.targets) + (
                 "pre_feedback_accuracy_pct", "post_feedback_accuracy_pct",
                 "reference_updates", "slow_updates", "fast_updates", "elapsed_seconds",
                 "slow_forward_calls", "teacher_forward_calls", "slow_backward_calls")


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset-root", type=Path, default=ROOT / "storage/datasets")
    cli.add_argument("--checkpoint-dir", type=Path, default=EXPERIMENT.checkpoint_dir)
    cli.add_argument("--output-dir", type=Path, help="New output directory; never overwrite")
    cli.add_argument("--seeds", type=int, nargs=3, help="Exactly the reference seeds; default: manifest seeds")
    devices = cli.add_mutually_exclusive_group()
    devices.add_argument("--gpu-idx", type=int, help="One GPU (default: 0)")
    devices.add_argument("--gpus", type=int, nargs="+", help="One independent worker per GPU, e.g. 0 1")
    cli.add_argument("--num-workers", type=int, default=0)
    cli.add_argument("--delays", type=int, nargs="+", choices=DELAYS, default=list(DELAYS),
                     help="Delays to run; use --delays 0 for immediate-label experiments")
    cli.add_argument("--enable-fast-d0", action="store_true",
                     help="Enable the fast adapter and reserve its update budget also at D=0")
    cli.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    cli.add_argument("--delay", type=int, choices=DELAYS, help=argparse.SUPPRESS)
    cli.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    cli.add_argument("--reference-dir", type=Path, required=True,
                     help="Completed full-memory FCRA sweep: manifest + D*/seed_* traces")
    return cli


def load_reference(args):
    manifest = json.loads((args.reference_dir / "manifest.json").read_text())
    seeds = manifest["seeds"]
    if (manifest.get("method") != "FCRA" or manifest.get("status") != "complete"
            or manifest["dataset"] != "PACS" or manifest["batch_size"] != BATCH_SIZE
            or tuple(manifest["delays"]) != DELAYS or len(seeds) != 3
            or len(set(seeds)) != 3 or any(not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError("Reference must be a complete PACS FCRA four-delay/three-seed sweep")
    for name in EXPERIMENT.checkpoint_names:
        if sha256(args.checkpoint_dir / name) != manifest["checkpoint_sha256"][name]:
            raise ValueError(f"Reference source checkpoint hash differs: {name}")
    if args.seeds is not None and (len(set(args.seeds)) != 3 or set(args.seeds) != set(seeds)):
        raise ValueError("Use exactly the reference's three seeds; a new random seed breaks matching")
    return manifest, args.seeds if args.seeds is not None else seeds


def run_worker(args):
    multiprocessing.set_start_method("spawn", force=True)
    import torch
    from ATTA import args_parser, config_summoner
    from ATTA.utils.initial import reset_random_seed
    from experiments.fcra.algorithm import DEFAULTS as CORE_DEFAULTS
    from .adapter import DEFAULTS
    from .algorithm import DualFCRA

    if (args.delay is None or args.seed is None or args.output_dir is None
            or args.gpu_idx is None or args.gpu_idx < 0 or args.gpus is not None):
        raise ValueError("Worker requires delay, seed, output-dir and one gpu-idx")
    if not torch.cuda.is_available() or args.gpu_idx >= torch.cuda.device_count():
        raise RuntimeError("Choose a visible CUDA GPU")
    torch.cuda.set_device(args.gpu_idx)
    validate_inputs(args)
    _, seeds = load_reference(args)
    if args.seed not in seeds:
        raise ValueError("Worker seed is not in reference sweep")
    trace = ReferenceTrace(args.reference_dir / f"D{args.delay}" / f"seed_{args.seed}",
                           args.delay, args.seed, enable_fast_d0=args.enable_fast_d0)
    parent = json.loads((args.output_dir.parent.parent / "manifest.json").read_text())
    key = f"D{args.delay}/seed_{args.seed}"
    if (parent["enable_fast_d0"] != args.enable_fast_d0
            or args.delay not in parent["delays"]
            or parent["reference_hashes"][key] != trace.hashes
            or parent["reference_manifest_sha256"] != sha256(args.reference_dir / "manifest.json")):
        raise ValueError("Reference changed after sweep preflight")
    config = config_summoner(args_parser([
        "--config_path", str(EXPERIMENT.config_path), "--exp_round", "1",
        "--ckpt_dir", str(args.checkpoint_dir), "--dataset.dataset_root", str(args.dataset_root),
        "--gpu_idx", str(args.gpu_idx), "--num_workers", str(args.num_workers),
        "--atta.batch_size", str(BATCH_SIZE),
    ]))
    config.random_seed = args.seed
    if (config.dataset.name != "PACS" or tuple(config.dataset.test_envs) != EXPERIMENT.environments
            or config.atta.batch_size != BATCH_SIZE
            or config.train.train_bs != trace.settings["train_batch_size"]
            or dict(config.atta.SimATTA) != trace.settings["simatta"]):
        raise ValueError("Current PACS training configuration differs from reference")
    for name in ("replay_mix", "replay_neighbors", "epsilon", "teacher_cache_size",
                 "gate_neighbors", "gate_similarity", "gate_temperature", "gate_prior"):
        if trace.settings["fcra"][name] != CORE_DEFAULTS[name]:
            raise ValueError(f"Inherited FCRA setting differs: {name}")
    reset_random_seed(config)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dump_json(args.output_dir / "settings.json", {
        "method": METHOD, "dataset": "PACS", "seed": args.seed, "delay": args.delay,
        "batch_size": BATCH_SIZE, "train_batch_size": config.train.train_bs,
        "source": EXPERIMENT.source, "target_order": list(EXPERIMENT.targets),
        "dataset_root": str(args.dataset_root), "checkpoint_dir": str(args.checkpoint_dir),
        "reference_dir": str(args.reference_dir), "reference_hashes": trace.hashes,
        "fcra": CORE_DEFAULTS, "dual": DEFAULTS, "simatta": dict(config.atta.SimATTA),
        "reference_updates": trace.total, "reserved_fast": trace.fast_total,
        "enable_fast_d0": args.enable_fast_d0, "fast_adapter_enabled": trace.fast_enabled,
        "gpu_idx": args.gpu_idx, "gpu_name": torch.cuda.get_device_name(args.gpu_idx),
        "num_workers": args.num_workers, "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    })
    write_csv(args.output_dir / "budget_plan.csv", trace.budget_rows())
    result = DualFCRA(config, args.delay, trace).run_stream(args.output_dir, progress=emit_progress)
    dump_json(args.output_dir / "result.json", result)


def summarize(rows, output, seeds, delays):
    expected = {(delay, seed) for delay in delays for seed in seeds}
    identities = [(row["delay"], row["seed"]) for row in rows]
    if (set(identities) != expected or len(identities) != len(expected)
            or any(row.get("method") != METHOD for row in rows)):
        raise ValueError("Summary requires all selected delays and three reference seeds")
    summaries = []
    metrics = EXPERIMENT.metrics + EXTRA_METRICS
    for delay in delays:
        group = [row for row in rows if row["delay"] == delay]
        summary = {"method": METHOD, "delay": delay, "runs": 3}
        for metric in metrics:
            values = [row[metric] for row in group if row[metric] is not None]
            summary[f"{metric}_mean"] = statistics.mean(values) if values else None
            summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
            summary[f"{metric}_n"] = len(values)
        summaries.append(summary)
    write_csv(output / "summary.csv", summaries)

    def table(columns):
        lines = ["| D (batch) | " + " | ".join(name for name, _ in columns) + " |",
                 "| " + " | ".join(["---:"] * (len(columns) + 1)) + " |"]
        for summary in summaries:
            cells = []
            for _, metric in columns:
                mean, std = summary[f"{metric}_mean"], summary[f"{metric}_std"]
                cells.append("N/A" if mean is None else (f"{mean:.2f} ± {std:.2f}"
                                                         if std is not None else f"{mean:.2f}"))
            lines.append(f"| {summary['delay']} | " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"

    accuracy = table([(f"{domain} (%)", f"{domain}_accuracy_pct") for domain in EXPERIMENT.targets] + [
        ("Stream (%)", "stream_accuracy_pct"), ("Domain macro (%)", "domain_macro_accuracy_pct"),
        ("Before first feedback (%)", "pre_feedback_accuracy_pct"),
        ("After first feedback (%)", "post_feedback_accuracy_pct")])
    costs = table([(name, metric) for name, metric in (
        ("Queried", "queried"), ("Returned", "returned"), ("Cross returned", "cross_domain_returned"),
        ("Cross (%)", "cross_domain_pct"), ("Pending", "pending"),
        ("Reference updates", "reference_updates"), ("Slow updates", "slow_updates"),
        ("Fast updates", "fast_updates"), ("Stream seconds", "elapsed_seconds"))])
    transitions = table([(f"{domain}: first 10 batches (%)", f"{domain}_first10_accuracy_pct")
                         for domain in EXPERIMENT.targets])
    text = ("# Dual-timescale FCRA: PACS\n\nPhoto → Art → Cartoon → Sketch; batch size=32. "
            f"Fast adapter enabled at D=0: {next((row['fast_adapter_enabled'] for row in rows if row['delay'] == 0), 'N/A')}. "
            f"Common reference seeds: {seeds}. Mean ± sample standard deviation.\n\n" + accuracy +
            "\n" + costs + "\n" + transitions +
            "\nAccuracy is recorded before updates. The first return batch is included "
            "in the before-feedback phase. Query IDs/times and total optimization steps match "
            "the reference. Equal steps do not imply equal FLOPs; per-run budget.csv records "
            "encoder forward/backward calls and image counts. Cross-domain percentages exclude "
            "pending labels and are averaged per seed.\n")
    (output / "summary.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main():
    args = parser().parse_args()
    if len(set(args.delays)) != len(args.delays):
        raise ValueError("Provide distinct delays")
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
        raise ValueError("delay/seed are worker-only arguments")
    validate_inputs(args)
    reference, seeds = load_reference(args)
    reference_hashes = {}
    for delay in args.delays:
        for seed in seeds:
            trace = ReferenceTrace(args.reference_dir / f"D{delay}" / f"seed_{seed}", delay, seed,
                                   enable_fast_d0=args.enable_fast_d0)
            reference_hashes[f"D{delay}/seed_{seed}"] = trace.hashes
    gpus = args.gpus if args.gpus is not None else [args.gpu_idx if args.gpu_idx is not None else 0]
    if len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("Provide distinct nonnegative GPU indices")
    output = args.output_dir or (ROOT / "storage/fcra_dual" /
                                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "method": METHOD, "dataset": "PACS", "status": "running", "seeds": seeds,
        "delays": args.delays, "batch_size": BATCH_SIZE,
        "enable_fast_d0": args.enable_fast_d0,
        "reference_dir": str(args.reference_dir), "reference_hashes": reference_hashes,
        "reference_manifest_sha256": sha256(args.reference_dir / "manifest.json"),
        "reference_commit": reference["git_commit"],
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "checkpoint_dir": str(args.checkpoint_dir), "dataset_root": str(args.dataset_root),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "gpus": gpus, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": sys.version, "protocol": "fixed reference queries and total updates; two timescales",
    }
    dump_json(output / "manifest.json", manifest)
    print(f"Method: {METHOD}\nSeeds: {seeds}\nGPUs: {gpus}\nResults: {output}", flush=True)
    rows = []

    def launch(job, gpu):
        delay, seed = job
        folder = output / f"D{delay}" / f"seed_{seed}"
        command = [sys.executable, "-m", "experiments.fcra_dual.run", "--worker",
                   "--reference-dir", str(args.reference_dir), "--delay", str(delay), "--seed", str(seed),
                   "--dataset-root", str(args.dataset_root), "--checkpoint-dir", str(args.checkpoint_dir),
                   "--output-dir", str(folder), "--gpu-idx", str(gpu), "--num-workers", str(args.num_workers)]
        if args.enable_fast_d0:
            command.append("--enable-fast-d0")
        progress.start(gpu, delay, seed)
        log_path = output / f"D{delay}_seed_{seed}.log"
        with log_path.open("w", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    event = parse_progress(line)
                    if event is not None:
                        progress.update(gpu, event)
        if process.returncode:
            raise RuntimeError(f"D={delay}, seed={seed}, GPU={gpu} failed; see {log_path}")
        result = json.loads((folder / "result.json").read_text())
        if result.get("method") != METHOD or result["seed"] != seed or result["delay"] != delay:
            raise ValueError("Worker result identity mismatch")
        return result

    def save_result(result):
        rows.append(result)
        rows.sort(key=lambda row: (row["delay"], seeds.index(row["seed"])))
        write_csv(output / "per_seed.csv", rows)
        progress.finish(result)

    try:
        with ProgressDisplay(gpus, len(args.delays) * len(seeds)) as progress:
            run_jobs([(delay, seed) for delay in args.delays for seed in seeds], gpus, launch, save_result)
        summarize(rows, output, seeds, args.delays)
    except BaseException:
        manifest.update(status="failed", completed_runs=len(rows))
        dump_json(output / "manifest.json", manifest)
        raise
    manifest.update(status="complete", completed_runs=len(rows))
    dump_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
