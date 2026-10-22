"""Minimal frozen-source diagnostics; run from the repository root."""

import argparse
from datetime import datetime, timezone
import multiprocessing
from pathlib import Path
import subprocess
import sys

from experiments.delayed_feedback.datasets import EXPERIMENTS, ROOT
from experiments.delayed_feedback.run import config_arguments, dump_json, sha256, validate_inputs


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset", choices=tuple(EXPERIMENTS), default="PACS")
    cli.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    cli.add_argument("--checkpoint-dir", type=Path)
    cli.add_argument("--output-dir", type=Path)
    cli.add_argument("--seeds", type=int, nargs="+", default=[110])
    cli.add_argument("--gpu-idx", type=int, default=0)
    cli.add_argument("--num-workers", type=int, default=0)
    cli.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    cli.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = cli.parse_args()
    experiment = EXPERIMENTS[args.dataset]
    args.dataset_root = args.dataset_root.resolve()
    args.checkpoint_dir = (args.checkpoint_dir or experiment.checkpoint_dir).resolve()
    if args.gpu_idx < 0 or len(set(args.seeds)) != len(args.seeds) or any(
            not 0 <= seed < 2**32 for seed in args.seeds):
        raise ValueError("Use distinct seeds in [0,2**32) and a nonnegative GPU index")
    validate_inputs(args)
    if args.worker:
        if args.seed is None or args.output_dir is None:
            raise ValueError("Worker requires seed and output-dir")
        if not 0 <= args.seed < 2**32:
            raise ValueError("Worker seed must be in [0,2**32)")
        multiprocessing.set_start_method("spawn", force=True)
        import numpy as np
        import sklearn
        import torch
        from ATTA import args_parser, config_summoner
        from ATTA.utils.initial import reset_random_seed
        from experiments.fcra.algorithm import DEFAULTS
        from .collect import FrozenProbe

        if not torch.cuda.is_available() or args.gpu_idx >= torch.cuda.device_count():
            raise RuntimeError("Select an available CUDA GPU")
        torch.cuda.set_device(args.gpu_idx)
        config = config_summoner(args_parser(config_arguments(args)))
        config.random_seed = args.seed
        if tuple(config.dataset.test_envs) != experiment.environments or config.dataset.name != args.dataset:
            raise ValueError("Unexpected domain stream")
        reset_random_seed(config)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        hashes = {name: sha256(args.checkpoint_dir / name) for name in experiment.checkpoint_names}
        dump_json(args.output_dir / "settings.json", {
            "seed": args.seed, "dataset": args.dataset, "delay": 0, "updates": 0,
            "batch_size": config.atta.batch_size, "simatta": dict(config.atta.SimATTA),
            "gate": DEFAULTS, "checkpoint_sha256": hashes,
            "torch_version": torch.__version__, "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__, "num_workers": args.num_workers,
            "checkpoint_dir": str(args.checkpoint_dir),
            "device": str(config.device), "dataset_root": str(args.dataset_root),
            "source": experiment.source, "targets": experiment.targets,
            "protocol": "Frozen selector; score before current returns; held-out low-entropy pool",
        })
        FrozenProbe(config, 0).collect(args.output_dir)
        if hashes != {name: sha256(args.checkpoint_dir / name) for name in experiment.checkpoint_names}:
            raise RuntimeError("Source checkpoint changed during collection")
        from .analyze import analyze
        analyze(args.output_dir, args.dataset)
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (args.output_dir or ROOT / "storage/local_reliability" / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "dataset": args.dataset, "seeds": args.seeds,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "checkpoint_sha256": {name: sha256(args.checkpoint_dir / name) for name in experiment.checkpoint_names},
                "dataset_root": str(args.dataset_root), "python": sys.version, "updates": 0}
    dump_json(output / "manifest.json", manifest)
    try:
        for seed in args.seeds:
            run_dir = output / f"seed_{seed}"
            command = [sys.executable, "-u", "-m", "experiments.local_reliability.run", "--worker",
                       "--dataset", args.dataset, "--dataset-root", str(args.dataset_root),
                       "--checkpoint-dir", str(args.checkpoint_dir), "--output-dir", str(run_dir),
                       "--seed", str(seed), "--gpu-idx", str(args.gpu_idx), "--num-workers", str(args.num_workers)]
            subprocess.run(command, cwd=ROOT, check=True)
            import json
            settings = json.loads((run_dir / "settings.json").read_text())
            if settings["seed"] != seed or settings["checkpoint_sha256"] != manifest["checkpoint_sha256"]:
                raise ValueError("Worker identity or checkpoint differs from manifest")
    except BaseException:
        manifest["status"] = "failed"
        dump_json(output / "manifest.json", manifest)
        raise
    manifest["status"] = "complete"
    dump_json(output / "manifest.json", manifest)
    reports = [f"- seed {seed}: [report](seed_{seed}/report.md)" for seed in args.seeds]
    (output / "report.md").write_text(
        "# Frozen local reliability diagnostics\n\n" + "\n".join(reports)
        + "\n\nNo training or causal performance claim. Read every domain, coverage and counts; "
        "do not select only positive seeds. Repeated seeds reuse the same dataset.\n", encoding="utf-8")
    print(f"Results: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
