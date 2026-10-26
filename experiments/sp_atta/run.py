"""Run the fixed-trace D100 four-group study, or the separate scaled diagnosis."""

import argparse
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import statistics
import subprocess
import sys

from experiments.delayed_feedback.datasets import EXPERIMENTS, ROOT
from experiments.delayed_feedback.progress import ProgressDisplay, emit_progress, parse_progress
from experiments.delayed_feedback.run import dump_json, sha256, validate_inputs, write_csv
from experiments.delayed_feedback.scheduler import run_jobs
from experiments.simatta_rootcause.spec import BASELINE_METRICS, PROBE_METRICS
from .reference import SimATTATrace
from .spec import GROUPS, DIAGNOSTIC, ARMS, DEFAULTS, METRICS

EXPERIMENT = EXPERIMENTS["PACS"]


def parser(diagnostic):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.set_defaults(diagnostic=diagnostic)
    cli.add_argument("--dataset-root", type=Path, default=ROOT / "storage/datasets")
    cli.add_argument("--checkpoint-dir", type=Path, default=EXPERIMENT.checkpoint_dir)
    cli.add_argument("--reference-dir", type=Path, help="Main study: completed original simatta_rootcause result ROOT")
    cli.add_argument("--output-dir", type=Path, help="New directory only")
    cli.add_argument("--seeds", type=int, nargs=3, default=[110, 123, 456])
    devices = cli.add_mutually_exclusive_group()
    devices.add_argument("--gpus", type=int, nargs="+")
    devices.add_argument("--gpu-idx", type=int)
    cli.add_argument("--num-workers", type=int, default=0)
    cli.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    cli.add_argument("--group", choices=GROUPS + (DIAGNOSTIC,), help=argparse.SUPPRESS)
    cli.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    cli.add_argument("--diagnostic", action="store_true", help=argparse.SUPPRESS)
    return cli


def worker(args):
    multiprocessing.set_start_method("spawn", force=True)
    import torch
    from ATTA import args_parser, config_summoner
    from ATTA.utils.initial import reset_random_seed
    if args.group is None or args.seed is None or args.output_dir is None or args.gpu_idx is None or args.gpus is not None:
        raise ValueError("Incomplete internal worker identity")
    if not 0 <= args.seed < 2**32 or not torch.cuda.is_available() or not 0 <= args.gpu_idx < torch.cuda.device_count():
        raise ValueError("Worker requires a valid seed and a visible CUDA GPU")
    torch.cuda.set_device(args.gpu_idx)
    validate_inputs(args)
    config = config_summoner(args_parser([
        "--config_path", str(EXPERIMENT.config_path), "--exp_round", "1",
        "--ckpt_dir", str(args.checkpoint_dir), "--dataset.dataset_root", str(args.dataset_root),
        "--gpu_idx", str(args.gpu_idx), "--num_workers", str(args.num_workers), "--atta.batch_size", "32",
    ]))
    config.random_seed = args.seed
    if config.dataset.name != "PACS" or tuple(config.dataset.test_envs) != (2, 0, 1, 3) or config.atta.batch_size != 32:
        raise ValueError("This study requires PACS Photo -> Art -> Cartoon -> Sketch, batch32")
    reset_random_seed(config)
    if args.diagnostic:
        if args.group != DIAGNOSTIC:
            raise ValueError("Diagnostic worker group mismatch")
        from .scaled import ScaledTruthDiagnosis
        algorithm = ScaledTruthDiagnosis(config)
        trace = None
    else:
        if args.group not in GROUPS or args.reference_dir is None:
            raise ValueError("Main study requires a fixed-trace reference and valid objective")
        from .algorithm import SPATTA
        trace = SimATTATrace(args.reference_dir, args.seed, args.checkpoint_dir)
        algorithm = SPATTA(config, args.group, trace)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dump_json(args.output_dir / "settings.json", {
        "method": args.group, "dataset": "PACS", "delay": 100, "seed": args.seed,
        "batch_size": 32, "dataset_root": str(args.dataset_root), "checkpoint_dir": str(args.checkpoint_dir),
        "reference_dir": str(args.reference_dir) if trace else None,
        "reference_hashes": trace.hashes if trace else None, "defaults": DEFAULTS,
        "simatta": dict(config.atta.SimATTA), "train_batch_size": config.train.train_bs,
        "gpu_idx": args.gpu_idx, "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
    })
    def progress(event):
        emit_progress({**event, "domain": f"{args.group}/{event['domain']}"})
    result = algorithm.run_stream(args.output_dir, progress=progress)
    if trace:
        result["reference_hashes"] = trace.hashes
    dump_json(args.output_dir / "result.json", result)


def aggregate(group, rows, metrics):
    summary = {"method": group, "delay": 100, "runs": len(rows)}
    for key in metrics:
        values = [row[key] for row in rows if row.get(key) is not None]
        summary[f"{key}_mean"] = statistics.mean(values) if values else None
        summary[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else None
        summary[f"{key}_n"] = len(values)
    return summary


def cell(row, key):
    mean, std = row[f"{key}_mean"], row[f"{key}_std"]
    if mean is None:
        return "N/A"
    return f"{mean:.2f} ± {std:.2f}" if std is not None else f"{mean:.2f}"


def summarize(output, rows, seeds, diagnostic, traces):
    groups = (DIAGNOSTIC,) if diagnostic else GROUPS
    if {(row["method"], row["seed"]) for row in rows} != {(group, seed) for group in groups for seed in seeds} or len(rows) != len(groups) * 3:
        raise ValueError("Summary requires all distinct, completed jobs")
    if diagnostic:
        metrics = BASELINE_METRICS + PROBE_METRICS + (
            "TRUE-scaled_future_accuracy_pct", "TRUE_minus_TRUE-scaled_pp", "TRUE-scaled_minus_MIX_pp")
    else:
        metrics = METRICS
        for seed in seeds:
            matched = [row for row in rows if row["seed"] == seed]
            for key in ("formal_draws_sha256", "pseudo_pool_sha256", "main_updates", "candidate_updates", "gate_evaluations"):
                if len({row[key] for row in matched}) != 1:
                    raise ValueError(f"Paired protocol mismatch at {key}, seed={seed}")
            if any(row["reference_hashes"] != traces[str(seed)] for row in matched):
                raise ValueError("Reference files changed during the sweep")
    summaries = [aggregate(group, [row for row in rows if row["method"] == group], metrics) for group in groups]
    write_csv(output / "per_seed.csv", rows)
    write_csv(output / "summary.csv", summaries)
    if diagnostic:
        row = summaries[0]
        table = ["| Arm | Future accuracy (%) |", "| :--- | ---: |"]
        table += [f"| {arm} | {cell(row, f'{arm}_future_accuracy_pct')} |" for arm in ARMS]
        text = ("# TRUE-scaled diagnosis: PACS D=100\n\n" + "\n".join(table)
                + f"\n\nTRUE − TRUE-scaled: {cell(row, 'TRUE_minus_TRUE-scaled_pp')} pp. "
                + f"TRUE-scaled − MIX: {cell(row, 'TRUE-scaled_minus_MIX_pp')} pp.\n"
                + "\nFrozen branches, same ten-step draws and next ten batches; these are local diagnostic effects, not deployed whole-stream accuracy.\n")
    else:
        accuracy = [("Art (%)", "Art_accuracy_pct"), ("Cartoon (%)", "Cartoon_accuracy_pct"),
                    ("Sketch (%)", "Sketch_accuracy_pct"), ("Macro (%)", "domain_macro_accuracy_pct"),
                    ("Stream (%)", "stream_accuracy_pct")]
        feedback = [("Queried", "queried"), ("Returned", "returned"), ("Returned (%)", "returned_pct"),
                    ("Cross-domain (%)", "cross_domain_pct"), ("Pending", "pending"),
                    ("Formal updates", "main_updates"), ("Shadow updates", "candidate_updates"),
                    ("Gate accepts (%)", "gate_acceptance_pct"), ("Actual auxiliary steps (%)", "aux_update_pct")]
        blocks = []
        for title, columns in (("Accuracy", accuracy), ("Feedback and mechanism", feedback)):
            table = [f"## {title}\n", "| Method | " + " | ".join(name for name, _ in columns) + " |",
                     "| :--- | " + " | ".join("---:" for _ in columns) + " |"]
            table += [f"| {row['method']} | " + " | ".join(cell(row, key) for _, key in columns) + " |" for row in summaries]
            blocks.append("\n".join(table))
        paired = []
        for seed in seeds:
            lookup = {row["method"]: row for row in rows if row["seed"] == seed}
            paired.append({"seed": seed,
                           "SP_minus_TRUE_stream_pp": lookup["SP-ATTA"]["stream_accuracy_pct"] - lookup["TRUE-only"]["stream_accuracy_pct"],
                           "SP_minus_Fixed_stream_pp": lookup["SP-ATTA"]["stream_accuracy_pct"] - lookup["Fixed-Aux"]["stream_accuracy_pct"],
                           "SP_minus_Original_stream_pp": lookup["SP-ATTA"]["stream_accuracy_pct"] - lookup["Original-MIX"]["stream_accuracy_pct"]})
        write_csv(output / "paired.csv", paired)
        paired_summary = []
        for comparison, key in (("SP − TRUE", "SP_minus_TRUE_stream_pp"),
                                ("SP − Fixed", "SP_minus_Fixed_stream_pp"),
                                ("SP − Original", "SP_minus_Original_stream_pp")):
            values = [row[key] for row in paired]
            paired_summary.append({"comparison": comparison, "stream_gain_pp_mean": statistics.mean(values),
                                   "stream_gain_pp_std": statistics.stdev(values), "n": len(values)})
        write_csv(output / "paired_summary.csv", paired_summary)
        text = "# SP-ATTA controlled study: PACS D=100\n\n" + "\n\n".join(blocks)
        text += "\n\n## Paired stream gains\n\n| Comparison | Gain (pp) |\n| :--- | ---: |\n"
        text += "\n".join(f"| {row['comparison']} | {row['stream_gain_pp_mean']:.2f} ± {row['stream_gain_pp_std']:.2f} |"
                          for row in paired_summary)
        text += "\n\nQuery/return schedules, formal quotas, formal draws and pseudo pools are matched across groups. Shadow tests run in all groups; only SP-ATTA follows their decision.\n"
    text += f"\nPhoto(source) → Art → Cartoon → Sketch; batch32; seeds={seeds}; mean ± sample standard deviation. "
    text += "Updates are recorded separately from shadow overhead; equal steps do not imply equal FLOPs. No superiority or significance claim is automatic.\n"
    (output / "summary.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main(diagnostic=False):
    args = parser(diagnostic).parse_args()
    args.dataset = "PACS"
    args.dataset_root, args.checkpoint_dir = args.dataset_root.resolve(), args.checkpoint_dir.resolve()
    if args.reference_dir:
        args.reference_dir = args.reference_dir.resolve()
    if args.output_dir:
        args.output_dir = args.output_dir.resolve()
    if args.worker:
        worker(args)
        return
    if args.group is not None or args.seed is not None:
        raise ValueError("Group and seed are worker-only; use --seeds")
    if not args.diagnostic and args.reference_dir is None:
        raise ValueError("Main study needs --reference-dir pointing to the completed original simatta_rootcause result root")
    if args.diagnostic and args.reference_dir is not None:
        raise ValueError("Scaled diagnosis runs the original trajectory directly and does not need --reference-dir")
    validate_inputs(args)
    seeds = args.seeds
    gpus = args.gpus if args.gpus is not None else [args.gpu_idx if args.gpu_idx is not None else 0]
    if len(set(seeds)) != 3 or any(not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Provide three distinct seeds in [0,2**32)")
    if len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("Provide unique nonnegative GPU indices")
    traces = ({str(seed): SimATTATrace(args.reference_dir, seed, args.checkpoint_dir).hashes for seed in seeds}
              if not args.diagnostic else {})
    name = "sp_atta_scaled" if args.diagnostic else "sp_atta"
    output = args.output_dir or ROOT / "storage" / name / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    groups = (DIAGNOSTIC,) if args.diagnostic else GROUPS
    manifest = {"status": "running", "mode": "scaled-diagnosis" if args.diagnostic else "controlled-study",
                "dataset": "PACS", "groups": groups, "seeds": seeds, "delays": [100], "batch_size": 32,
                "gpus": gpus, "reference_dir": str(args.reference_dir) if args.reference_dir else None,
                "reference_hashes": traces, "defaults": DEFAULTS,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "checkpoint_sha256": {name: sha256(args.checkpoint_dir / name) for name in EXPERIMENT.checkpoint_names},
                "dataset_root": str(args.dataset_root), "checkpoint_dir": str(args.checkpoint_dir),
                "python": sys.version, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    dump_json(output / "manifest.json", manifest)
    print(f"Mode: {manifest['mode']}\nGroups: {groups}\nSeeds: {seeds}\nGPUs: {gpus}\nResults: {output}", flush=True)
    rows = []
    def launch(job, gpu):
        group, seed = job
        run_dir = output / group / f"seed_{seed}"
        command = [sys.executable, "-u", "-m", "experiments.sp_atta.run", "--worker", "--group", group,
                   "--seed", str(seed), "--gpu-idx", str(gpu), "--dataset-root", str(args.dataset_root),
                   "--checkpoint-dir", str(args.checkpoint_dir), "--output-dir", str(run_dir),
                   "--num-workers", str(args.num_workers)]
        if args.diagnostic:
            command.append("--diagnostic")
        else:
            command.extend(("--reference-dir", str(args.reference_dir)))
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
            raise RuntimeError(f"{group} seed={seed} GPU={gpu} failed: see {log_path}")
        result = json.loads((run_dir / "result.json").read_text())
        if result["method"] != group or result["seed"] != seed or result["delay"] != 100:
            raise ValueError("Worker identity mismatch")
        return result
    def save(result):
        rows.append(result)
        rows.sort(key=lambda row: (groups.index(row["method"]), seeds.index(row["seed"])))
        write_csv(output / "per_seed.csv", rows)
        progress.finish(result)
    try:
        with ProgressDisplay(gpus, len(groups) * 3) as progress:
            run_jobs([(group, seed) for group in groups for seed in seeds], gpus, launch, save)
        summarize(output, rows, seeds, args.diagnostic, traces)
    except BaseException:
        manifest.update(status="failed", completed_runs=len(rows))
        dump_json(output / "manifest.json", manifest)
        raise
    manifest.update(status="complete", completed_runs=len(rows))
    dump_json(output / "manifest.json", manifest)
    print(f"Complete: {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
