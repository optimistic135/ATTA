"""Nine matched D100 jobs: SP, current soft targets, and risk rectification."""
import argparse
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

from experiments.delayed_feedback.datasets import EXPERIMENTS, ROOT
from experiments.delayed_feedback.progress import ProgressDisplay, emit_progress, parse_progress
from experiments.delayed_feedback.run import dump_json, sha256, validate_inputs, write_csv
from experiments.delayed_feedback.scheduler import run_jobs
from experiments.sp_atta.reference import SimATTATrace
from .spec import METHOD, DEFAULTS, GROUPS
from .report import summarize

EXPERIMENT = EXPERIMENTS["PACS"]


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset-root", type=Path, default=ROOT / "storage/datasets")
    cli.add_argument("--checkpoint-dir", type=Path, default=EXPERIMENT.checkpoint_dir)
    cli.add_argument("--reference-dir", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path)
    cli.add_argument("--seeds", type=int, nargs=3, default=[110, 123, 456])
    devices = cli.add_mutually_exclusive_group()
    devices.add_argument("--gpus", type=int, nargs="+")
    devices.add_argument("--gpu-idx", type=int)
    cli.add_argument("--num-workers", type=int, default=0)
    cli.add_argument("--resume", action="store_true")
    cli.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    cli.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    cli.add_argument("--group", choices=GROUPS, help=argparse.SUPPRESS)
    return cli


def worker(args):
    multiprocessing.set_start_method("spawn", force=True)
    import torch
    from ATTA import args_parser, config_summoner
    from ATTA.utils.initial import reset_random_seed
    from .algorithm import RiskRectification
    if args.group is None or args.seed is None or args.output_dir is None or args.gpu_idx is None or args.gpus is not None:
        raise ValueError("Incomplete worker identity")
    if not 0 <= args.seed < 2**32 or not torch.cuda.is_available() or not 0 <= args.gpu_idx < torch.cuda.device_count():
        raise ValueError("Worker needs a valid seed and CUDA GPU")
    torch.cuda.set_device(args.gpu_idx)
    validate_inputs(args)
    config = config_summoner(args_parser([
        "--config_path", str(EXPERIMENT.config_path), "--exp_round", "1",
        "--ckpt_dir", str(args.checkpoint_dir), "--dataset.dataset_root", str(args.dataset_root),
        "--gpu_idx", str(args.gpu_idx), "--num_workers", str(args.num_workers), "--atta.batch_size", "32",
    ]))
    config.random_seed = args.seed
    if config.dataset.name != "PACS" or tuple(config.dataset.test_envs) != (2, 0, 1, 3) or config.atta.batch_size != 32:
        raise ValueError("Require PACS Photo -> Art -> Cartoon -> Sketch, batch32")
    reset_random_seed(config)
    trace = SimATTATrace(args.reference_dir, args.seed, args.checkpoint_dir)
    algorithm = RiskRectification(config, args.group, trace)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dump_json(args.output_dir / "settings.json", {"method": args.group, "seed": args.seed, "delay": 100,
        "defaults": DEFAULTS, "reference_hashes": trace.hashes, "simatta": dict(config.atta.SimATTA),
        "train_batch_size": config.train.train_bs, "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda})
    def progress(event):
        emit_progress({**event, "domain": f"{args.group}/{event['domain']}"})
    result = algorithm.run_stream(args.output_dir, progress=progress)
    result["reference_hashes"] = trace.hashes
    # Result is written last and is the task completion marker.
    dump_json(args.output_dir / "result.json", result)


def load_completed(folder, group, seed, trace):
    required = ("result.json", "settings.json", "batches.csv", "feedback.csv", "risk_batches.csv",
        "risk_losses.csv", "density_weights.csv", "gates.csv", "updates.csv", "budget.csv", "compute.json")
    if any(not (folder / name).is_file() or not (folder / name).stat().st_size for name in required):
        return None
    try:
        result = json.loads((folder / "result.json").read_text())
    except (ValueError, UnicodeError):
        return None
    if not isinstance(result, dict):
        return None
    if (result.get("method"), result.get("seed"), result.get("delay")) != (group, seed, 100):
        raise ValueError(f"Task identity mismatch: {folder}")
    settings = json.loads((folder / "settings.json").read_text())
    if (settings.get("method") != group or settings.get("seed") != seed or settings.get("delay") != 100
            or settings.get("defaults") != json.loads(json.dumps(DEFAULTS))
            or settings.get("reference_hashes") != trace.hashes or result.get("reference_hashes") != trace.hashes
            or result.get("main_updates") != sum(trace.quotas)):
        raise ValueError(f"Task settings, reference or budget mismatch: {folder}")
    from experiments.sp_atta.reference import read_csv
    trace.verify_feedback(read_csv(folder / "feedback.csv"))
    return result


def prepare_resume(output, manifest, args, references):
    previous = json.loads((output / "manifest.json").read_text())
    normalized = json.loads(json.dumps(manifest))
    for key in ("method", "seeds", "delays", "batch_size", "defaults", "reference_dir", "reference_hashes",
                "checkpoint_sha256", "dataset_root", "checkpoint_dir", "groups", "code_sha256"):
        if previous.get(key) != normalized[key]:
            raise ValueError(f"Resume configuration differs: {key}")
    completed, pending = [], []
    for group in GROUPS:
        for seed in args.seeds:
            saved = load_completed(output / group / f"seed_{seed}", group, seed, references[seed])
            if saved is None:
                pending.append((group, seed))
            else:
                completed.append(saved)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    archive = output / "_resume_archive" / stamp
    archive.mkdir(parents=True, exist_ok=False)
    dump_json(archive / "manifest.json", previous)
    reports = ("per_seed.csv", "summary.csv", "summary.md", "paired_gains.csv", "paired_summary.csv", "risk_by_domain.csv")
    for name in reports:
        path = output / name
        if path.exists():
            path.rename(archive / name)
    for group, seed in pending:
        for path in (output / group / f"seed_{seed}", output / f"{group}_seed_{seed}.log"):
            if path.exists():
                destination = archive / group / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.rename(destination)
    manifest["resume_history"] = previous.get("resume_history", []) + [{"at_utc": stamp,
        "archive": str(archive), "previous_git_commit": previous.get("git_commit"),
        "skipped_jobs": [(r["method"], r["seed"]) for r in completed], "pending_jobs": pending}]
    return completed, pending


def main():
    args = parser().parse_args()
    args.dataset = "PACS"
    args.dataset_root, args.checkpoint_dir = args.dataset_root.resolve(), args.checkpoint_dir.resolve()
    args.reference_dir = args.reference_dir.resolve()
    if args.resume and (args.output_dir is None or args.worker):
        raise ValueError("--resume requires an explicit existing --output-dir and is parent-only")
    if args.output_dir:
        args.output_dir = args.output_dir.resolve()
    if args.worker:
        worker(args)
        return
    if args.seed is not None or args.group is not None:
        raise ValueError("--seed/--group are internal; use --seeds")
    validate_inputs(args)
    seeds = args.seeds
    gpus = args.gpus if args.gpus is not None else [args.gpu_idx if args.gpu_idx is not None else 0]
    if len(set(seeds)) != 3 or any(not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Use three distinct valid seeds")
    if len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("Use unique nonnegative GPUs")
    references = {seed: SimATTATrace(args.reference_dir, seed, args.checkpoint_dir) for seed in seeds}
    output = args.output_dir or ROOT / "storage/risk_rectification" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    if args.resume:
        if not (output / "manifest.json").is_file():
            raise ValueError("Resume needs an existing experiment manifest")
    else:
        output.mkdir(parents=True, exist_ok=False)
    import fcntl
    output_lock = (output / ".run.lock").open("a")
    try:
        fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        output_lock.close()
        raise ValueError("Another launcher is using this output") from None
    manifest = {"status": "running", "method": METHOD, "seeds": seeds, "delays": [100], "batch_size": 32,
        "gpus": gpus, "defaults": DEFAULTS, "reference_dir": str(args.reference_dir),
        "reference_hashes": {str(s): t.hashes for s, t in references.items()},
        "checkpoint_sha256": {name: sha256(args.checkpoint_dir / name) for name in EXPERIMENT.checkpoint_names},
        "dataset_root": str(args.dataset_root), "checkpoint_dir": str(args.checkpoint_dir),
        "groups": GROUPS,
        "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
            list(Path(__file__).parent.glob("*.py")) + [ROOT / name for name in (
                "experiments/sp_atta/algorithm.py", "experiments/sp_atta/reference.py",
                "experiments/delayed_feedback/simatta.py", "experiments/delayed_feedback/protocol.py",
                "experiments/simatta_rootcause/algorithm.py")]},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": sys.version, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    completed, pending = prepare_resume(output, manifest, args, references) if args.resume else ([], [(g, s) for g in GROUPS for s in seeds])
    results = list(completed)
    manifest["completed_runs"] = len(results)
    dump_json(output / "manifest.json", manifest)
    print(f"Method: {METHOD}\nSeeds: {seeds}\nGPUs: {gpus}\nResults: {output}\nSkipping: {len(results)}; pending: {len(pending)}", flush=True)
    def launch(job, gpu):
        group, seed = job
        folder = output / group / f"seed_{seed}"
        command = [sys.executable, "-u", "-m", "experiments.risk_rectification.run", "--worker", "--group", group,
            "--seed", str(seed), "--gpu-idx", str(gpu), "--dataset-root", str(args.dataset_root),
            "--checkpoint-dir", str(args.checkpoint_dir), "--reference-dir", str(args.reference_dir),
            "--output-dir", str(folder), "--num-workers", str(args.num_workers)]
        progress.start(gpu, 100, seed)
        log_path = output / f"{group}_seed_{seed}.log"
        with log_path.open("w", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    event = parse_progress(line)
                    if event:
                        progress.update(gpu, event)
        if process.returncode:
            raise RuntimeError(f"seed={seed} GPU={gpu} failed: see {log_path}")
        saved = load_completed(folder, group, seed, references[seed])
        if saved is None:
            raise ValueError(f"Missing complete worker output: {folder}")
        return saved
    def save(result):
        results.append(result)
        results.sort(key=lambda r: (GROUPS.index(r["method"]), seeds.index(r["seed"])))
        write_csv(output / "per_seed.csv", results)
        progress.finish(result)
    try:
        with ProgressDisplay(gpus, len(GROUPS) * 3) as progress:
            for result in results:
                progress.finish(result)
            if pending:
                run_jobs(pending, gpus, launch, save)
        summarize(output, results, seeds)
    except BaseException:
        manifest.update(status="failed", completed_runs=len(results))
        dump_json(output / "manifest.json", manifest)
        raise
    manifest.update(status="complete", completed_runs=len(results))
    dump_json(output / "manifest.json", manifest)
    print(f"Complete: {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
