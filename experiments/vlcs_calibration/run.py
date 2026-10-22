"""VLCS D=0: native SimATTA, quota-matched control and calibration-only."""

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
from experiments.delayed_feedback.run import dump_json, sha256, validate_inputs, write_csv
from experiments.delayed_feedback.progress import ProgressDisplay, emit_progress, parse_progress
from experiments.delayed_feedback.scheduler import run_jobs
from .reference import NativeTrace


EXPERIMENT = EXPERIMENTS["VLCS"]
VARIANTS = ("simatta_original", "simatta_control", "calibration_only")
ACCURACY = tuple(f"{domain}_accuracy_pct" for domain in EXPERIMENT.targets) + (
    "stream_accuracy_pct", "domain_macro_accuracy_pct")
METRICS = ACCURACY + ("queried", "returned", "slow_updates")


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset-root", type=Path, default=ROOT / "storage/datasets")
    cli.add_argument("--checkpoint-dir", type=Path, default=EXPERIMENT.checkpoint_dir)
    cli.add_argument("--output-dir", type=Path)
    cli.add_argument("--seeds", type=int, nargs=3, default=[110, 123, 456])
    devices = cli.add_mutually_exclusive_group()
    devices.add_argument("--gpus", type=int, nargs="+")
    devices.add_argument("--gpu-idx", type=int)
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
    from .algorithm import NativeSimATTA, ControlledCalibration

    if (args.seed is None or args.variant is None or args.output_dir is None
            or args.gpus is not None or args.gpu_idx is None
            or not torch.cuda.is_available() or not 0 <= args.gpu_idx < torch.cuda.device_count()):
        raise ValueError("Worker requires identity, output and one visible CUDA GPU")
    torch.cuda.set_device(args.gpu_idx)
    parent = json.loads((args.output_dir.parent.parent / "manifest.json").read_text())
    if args.seed not in parent["seeds"]:
        raise ValueError("Wrong worker seed")
    for name in EXPERIMENT.checkpoint_names:
        if sha256(args.checkpoint_dir / name) != parent["checkpoint_sha256"][name]:
            raise ValueError("Source checkpoint changed")
    config = config_summoner(args_parser([
        "--config_path", str(EXPERIMENT.config_path), "--exp_round", "1",
        "--ckpt_dir", str(args.checkpoint_dir), "--dataset.dataset_root", str(args.dataset_root),
        "--gpu_idx", str(args.gpu_idx), "--num_workers", str(args.num_workers),
        "--atta.batch_size", "32",
    ]))
    config.random_seed = args.seed
    if (config.dataset.name != "VLCS" or tuple(config.dataset.test_envs) != EXPERIMENT.environments
            or config.atta.batch_size != 32):
        raise ValueError("Expected VLCS Caltech101 -> LabelMe -> SUN09 -> VOC2007, batch=32")
    native = args.variant == "simatta_original"
    trace = None
    if not native:
        trace = NativeTrace(args.output_dir.parent.parent / "simatta_original" / f"seed_{args.seed}", args.seed)
        if (dict(config.atta.SimATTA) != trace.settings["simatta"]
                or config.train.train_bs != trace.settings["train_batch_size"]
                or trace.settings["checkpoint_sha256"] != parent["checkpoint_sha256"]):
            raise ValueError("Controlled configuration differs from native run")
    reset_random_seed(config)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    algorithm = NativeSimATTA(config) if native else ControlledCalibration(config, trace, args.variant)
    dump_json(args.output_dir / "settings.json", {
        "method": "VLCS-calibration", "dataset": "VLCS", "variant": args.variant,
        "seed": args.seed, "delay": 0, "batch_size": 32, "train_batch_size": config.train.train_bs,
        "source": EXPERIMENT.source, "target_order": list(EXPERIMENT.targets),
        "simatta": dict(config.atta.SimATTA), "fcra": DEFAULTS,
        "dataset_root": str(args.dataset_root), "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_sha256": parent["checkpoint_sha256"],
        "reference_hashes": trace.hashes if trace else None,
        "teacher_calibration": args.variant == "calibration_only", "targeted_replay": False,
        "fast_adapter_enabled": False, "num_workers": args.num_workers,
        "gpu_name": torch.cuda.get_device_name(args.gpu_idx),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
    })
    result = algorithm.run_stream(args.output_dir, progress=emit_progress)
    if trace is not None:
        current = NativeTrace(args.output_dir.parent.parent / "simatta_original" / f"seed_{args.seed}", args.seed)
        if trace.hashes != current.hashes:
            raise ValueError("Native reference changed during controlled run")
    dump_json(args.output_dir / "result.json", result)


def summarize(results, output, seeds):
    indexed = {(row["variant"], row["seed"]): row for row in results}
    if len(results) != 9 or set(indexed) != {(mode, seed) for mode in VARIANTS for seed in seeds}:
        raise ValueError("Need all three variants and three seeds")
    for seed in seeds:
        group = [indexed[mode, seed] for mode in VARIANTS]
        for key in ("queried", "returned", "slow_updates"):
            if len({row[key] for row in group}) != 1:
                raise ValueError(f"Unmatched label/update costs: seed={seed}, {key}")
    summaries = []
    for mode in VARIANTS:
        row = {"variant": mode, "runs": 3}
        for metric in METRICS:
            values = [indexed[mode, seed][metric] for seed in seeds]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_std"] = statistics.stdev(values)
        summaries.append(row)
    contrasts = (("calibration_vs_control", "calibration_only", "simatta_control"),
                 ("calibration_vs_original", "calibration_only", "simatta_original"),
                 ("control_vs_original", "simatta_control", "simatta_original"))
    paired = []
    for name, treatment, control in contrasts:
        for seed in seeds:
            paired.append({"contrast": name, "seed": seed, **{
                f"{metric}_delta_pp": indexed[treatment, seed][metric] - indexed[control, seed][metric]
                for metric in ACCURACY}})
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "paired_deltas.csv", paired)
    lines = ["# VLCS D=0: calibration-only validation", "",
             f"Caltech101 -> LabelMe -> SUN09 -> VOC2007; batch=32; seeds={seeds}.",
             "Native SimATTA selects queries and stops by its original loss window.",
             "Other variants replay its exact queries and per-batch update counts; no fast adapter or targeted replay.",
             "VLCS official SimATTA parameters are preserved; calibration defaults are transferred without tuning.", "",
             "| Variant | LabelMe | SUN09 | VOC2007 | Stream | Macro |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summaries:
        cells = [f"{row[m+'_mean']:.2f} ± {row[m+'_std']:.2f}" for m in ACCURACY]
        lines.append(f"| {row['variant']} | " + " | ".join(cells) + " |")
    lines += ["", "| Variant | Queried | Returned | Updates |", "| --- | ---: | ---: | ---: |"]
    for row in summaries:
        cells = [f"{row[m+'_mean']:.2f} ± {row[m+'_std']:.2f}" for m in METRICS[5:]]
        lines.append(f"| {row['variant']} | " + " | ".join(cells) + " |")
    lines += ["", "| Contrast | Paired Stream delta (pp) | Positive seeds |", "| --- | ---: | ---: |"]
    for name, _, _ in contrasts:
        values = [row["stream_accuracy_pct_delta_pp"] for row in paired if row["contrast"] == name]
        lines.append(f"| {name} | {statistics.mean(values):+.3f} ± {statistics.stdev(values):.3f} | "
                     f"{sum(value > 0 for value in values)}/3 |")
    lines += ["", "Use calibration_vs_control to assess the gate. Native vs controlled additionally changes",
              "the training sampler/loop and empty-pseudo fallback, so is not pure gate attribution.",
              "Native selection retains query-ID deduplication and tiny-cluster guards from the stream adapter.",
              "Accuracy is scored before feedback/update; Stream is sample-weighted and Macro domain-equal.",
              "Mean ± sample SD; three seeds do not establish significance. Equal updates do not imply equal FLOPs."]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parser().parse_args()
    args.dataset = "VLCS"
    for key in ("dataset_root", "checkpoint_dir", "output_dir"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    validate_inputs(args)
    if args.worker:
        worker(args)
        return
    if args.seed is not None or args.variant is not None:
        raise ValueError("seed/variant are worker-only")
    if len(set(args.seeds)) != 3 or any(not 0 <= seed < 2**32 for seed in args.seeds):
        raise ValueError("Provide three distinct seeds in [0, 2**32)")
    gpus = args.gpus if args.gpus is not None else [args.gpu_idx if args.gpu_idx is not None else 0]
    if len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("Choose distinct nonnegative GPUs")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output_dir or ROOT / "storage/vlcs_calibration" / stamp
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"method": "VLCS-calibration", "dataset": "VLCS", "delay": 0,
                "status": "running", "seeds": args.seeds, "variants": VARIANTS, "batch_size": 32,
                "dataset_root": str(args.dataset_root), "checkpoint_dir": str(args.checkpoint_dir),
                "checkpoint_sha256": {name: sha256(args.checkpoint_dir / name) for name in EXPERIMENT.checkpoint_names},
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "gpus": gpus, "num_workers": args.num_workers,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    dump_json(output / "manifest.json", manifest)
    results = []

    def launch(seed, gpu):
        group = []
        for mode in VARIANTS:
            folder = output / mode / f"seed_{seed}"
            command = [sys.executable, "-m", "experiments.vlcs_calibration.run", "--worker",
                       "--variant", mode, "--seed", str(seed), "--gpu-idx", str(gpu),
                       "--dataset-root", str(args.dataset_root), "--checkpoint-dir", str(args.checkpoint_dir),
                       "--output-dir", str(folder), "--num-workers", str(args.num_workers)]
            progress.start(gpu, 0, seed)
            print(f"VLCS variant={mode}, seed={seed}, GPU={gpu}", flush=True)
            with (output / f"{mode}_seed_{seed}.log").open("w", encoding="utf-8") as log:
                with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        event = parse_progress(line)
                        if event is not None:
                            progress.update(gpu, event)
            if process.returncode:
                raise RuntimeError(f"VLCS {mode}, seed={seed} failed; see log")
            result = json.loads((folder / "result.json").read_text())
            if result.get("method") != "VLCS-calibration" or result["variant"] != mode or result["seed"] != seed:
                raise ValueError("Worker identity mismatch")
            group.append(result)
            progress.finish(result)
        return group

    def save(group):
        results.extend(group)
        write_csv(output / "per_seed.csv", results)

    try:
        with ProgressDisplay(gpus, 9) as progress:
            run_jobs(args.seeds, gpus, launch, save)
        summarize(results, output, args.seeds)
    except BaseException:
        manifest.update(status="failed", completed_runs=sum((output / mode / f"seed_{seed}" / "result.json").is_file()
                                                           for mode in VARIANTS for seed in args.seeds))
        dump_json(output / "manifest.json", manifest)
        raise
    manifest.update(status="complete", completed_runs=len(results))
    dump_json(output / "manifest.json", manifest)
    print(f"Complete: {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
