"""PACS utility experiment: python -m experiments.utility_prediction.run."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from experiments.delayed_feedback.datasets import EXPERIMENTS, ROOT
from experiments.delayed_feedback.run import dump_json, sha256, validate_inputs


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset-root", type=Path, required=True)
    cli.add_argument("--checkpoint-dir", type=Path, default=EXPERIMENTS["PACS"].checkpoint_dir)
    cli.add_argument("--output-dir", type=Path)
    cli.add_argument("--gpu-idx", type=int, default=0)
    cli.add_argument("--seed", type=int, default=110)
    cli.add_argument("--delays", type=int, nargs="+", default=[0, 20, 50, 100])
    cli.add_argument("--batch-size", type=int, default=32)
    cli.add_argument("--steps", type=int, default=30)
    cli.add_argument("--horizon", type=int, default=20)
    args = cli.parse_args()
    if (args.gpu_idx < 0 or args.steps < 1 or args.horizon < 1 or args.batch_size < 1
            or not 0 <= args.seed < 2**32 or any(d < 0 for d in args.delays)
            or len(set(args.delays)) != len(args.delays)):
        cli.error("Use positive steps/horizon/batch-size, unique nonnegative delays and a valid seed/GPU")
    args.dataset, args.num_workers = "PACS", 0
    args.dataset_root = args.dataset_root.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    validate_inputs(args)
    import subprocess
    import torch
    from ATTA import args_parser, config_summoner
    from ATTA.utils.initial import reset_random_seed
    from .engine import UtilitySimATTA

    if not torch.cuda.is_available() or args.gpu_idx >= torch.cuda.device_count():
        raise RuntimeError("Choose a visible CUDA GPU; this entrypoint does not silently fall back to CPU")
    torch.cuda.set_device(args.gpu_idx)
    output = args.output_dir or (ROOT / "storage/utility_prediction" /
                                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    manifest.update({"output_dir": str(output), "torch_version": torch.__version__,
                     "cuda_version": torch.version.cuda,
                     "gpu_name": torch.cuda.get_device_name(args.gpu_idx),
                     "git_commit": subprocess.check_output(
                         ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                     "checkpoint_sha256": {name: sha256(args.checkpoint_dir / name)
                                           for name in EXPERIMENTS["PACS"].checkpoint_names},
                     "protocol": "fixed factual query/arrival schedule; per-label loss masking",
                     "status": "running"})
    dump_json(output / "manifest.json", manifest)
    print(f"Results: {output}", flush=True)
    try:
        for delay in args.delays:
            folder = output / f"D{delay}"
            folder.mkdir()
            config = config_summoner(args_parser([
                "--config_path", str(EXPERIMENTS["PACS"].config_path),
                "--exp_round", "1", "--ckpt_dir", str(args.checkpoint_dir),
                "--dataset.dataset_root", str(args.dataset_root),
                "--gpu_idx", str(args.gpu_idx), "--num_workers", "0",
                "--atta.batch_size", str(args.batch_size),
            ]))
            config.random_seed = args.seed
            config.atta.gpu_clustering = False
            reset_random_seed(config)
            algorithm = UtilitySimATTA(config, delay, folder, args.steps, args.horizon)
            if len(algorithm.stream) <= delay + args.horizon:
                raise ValueError(f"Only {len(algorithm.stream)} batches: D={delay} has no complete "
                                 f"future-{args.horizon} window; reduce --batch-size")
            dump_json(folder / "settings.json", {
                "seed": args.seed, "delay": delay, "batch_size": args.batch_size,
                "steps": args.steps, "horizon": args.horizon,
                "simatta": dict(config.atta.SimATTA), "clustering": "sklearn CPU",
                "target_order": ["Art", "Cartoon", "Sketch"],
                "stream_batches": len(algorithm.stream),
            })
            factual = algorithm.run_factual()
            dump_json(folder / "factual_result.json", factual)
            events = algorithm.evaluate_utilities()
            dump_json(folder / "result.json", {
                "delay": delay, "returned_events": len(events),
                "complete_events": sum(row["complete_window"] for row in events),
                "pending_labels": algorithm.oracle.counts()["pending"],
                "status": "complete",
            })
            del algorithm
            torch.cuda.empty_cache()
        from .analyze import analyze
        analyze(output)
    except Exception:
        manifest["status"] = "failed"
        dump_json(output / "manifest.json", manifest)
        raise
    manifest["status"] = "complete"
    dump_json(output / "manifest.json", manifest)
    print(f"Complete: {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
