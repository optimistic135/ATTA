"""PACS D100: original SP, current-source targets, feedback-repaired targets."""
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
from experiments.sp_atta.reference import SimATTATrace
from experiments.sp_atta.run import aggregate, cell
from .spec import GROUPS, DEFAULTS, METRICS

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
    cli.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    cli.add_argument("--group", choices=GROUPS, help=argparse.SUPPRESS)
    cli.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    return cli


def worker(args):
    multiprocessing.set_start_method("spawn", force=True)
    import torch
    from ATTA import args_parser, config_summoner
    from ATTA.utils.initial import reset_random_seed
    from .algorithm import FeedbackTeacher
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
        raise ValueError("Study requires PACS Photo -> Art -> Cartoon -> Sketch, batch32")
    reset_random_seed(config)
    trace = SimATTATrace(args.reference_dir, args.seed, args.checkpoint_dir)
    algorithm = FeedbackTeacher(config, args.group, trace)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dump_json(args.output_dir / "settings.json", {
        "method": args.group, "dataset": "PACS", "delay": 100, "seed": args.seed,
        "batch_size": 32, "dataset_root": str(args.dataset_root), "checkpoint_dir": str(args.checkpoint_dir),
        "reference_dir": str(args.reference_dir), "reference_hashes": trace.hashes,
        "defaults": DEFAULTS, "simatta": dict(config.atta.SimATTA), "train_batch_size": config.train.train_bs,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
    })
    def progress(event):
        emit_progress({**event, "domain": f"{args.group}/{event['domain']}"})
    result = algorithm.run_stream(args.output_dir, progress=progress)
    result["reference_hashes"] = trace.hashes
    dump_json(args.output_dir / "result.json", result)


def summarize(output, rows, seeds, traces):
    if len(rows) != 9 or {(r["method"], r["seed"]) for r in rows} != {(g, s) for g in GROUPS for s in seeds}:
        raise ValueError("Summary needs nine distinct completed runs")
    for seed in seeds:
        matched = [r for r in rows if r["seed"] == seed]
        for key in ("formal_draws_sha256", "pseudo_pool_sha256", "main_updates", "queried", "returned", "pending"):
            if len({r[key] for r in matched}) != 1:
                raise ValueError(f"Matched trace mismatch: {key}, seed={seed}")
        if any(r["reference_hashes"] != traces[str(seed)] for r in matched):
            raise ValueError("Reference changed during sweep")
        current = [r for r in matched if r["method"] != "SP-ATTA"]
        for key in ("student_updates", "head_updates"):
            if len({r[key] for r in current}) != 1:
                raise ValueError(f"Current-arm budget mismatch: {key}, seed={seed}")
    summaries = [aggregate(group, [r for r in rows if r["method"] == group], METRICS) for group in GROUPS]
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "per_seed.csv", rows)
    sections = []
    for title, columns in (
        ("Accuracy", [("Art (%)", "Art_accuracy_pct"), ("Cartoon (%)", "Cartoon_accuracy_pct"),
            ("Sketch (%)", "Sketch_accuracy_pct"), ("Macro (%)", "domain_macro_accuracy_pct"),
            ("Stream (%)", "stream_accuracy_pct")]),
        ("Feedback and costs", [("Queried", "queried"), ("Returned", "returned"),
            ("Returned (%)", "returned_pct"), ("Cross returned", "cross_domain_returned"),
            ("Cross (%)", "cross_domain_pct"), ("Pending", "pending"),
            ("Total formal", "main_updates"), ("Student", "student_updates"),
            ("Head", "head_updates"), ("Shadow", "candidate_updates"), ("Seconds", "stream_seconds")]),
        ("Mechanism", [("Current coverage (%)", "current_coverage_pct"), ("Repair use (%)", "repair_use_pct"),
            ("Source query acc (%)", "source_query_accuracy_pct"), ("Repair query acc (%)", "repair_query_accuracy_pct"),
            ("Repair-source query (pp)", "repair_minus_source_query_pp"),
            ("Pre-update return CE gain", "return_teacher_ce_gain"),
            ("Gate accepts (%)", "gate_acceptance_pct"), ("Aux steps (%)", "aux_update_pct"),
            ("Local AUX-TRUE (pp)", "local_aux_minus_true_pp")]),
    ):
        table = [f"## {title}\n", "| Method | " + " | ".join(label for label, _ in columns) + " |",
                 "| :--- | " + " | ".join("---:" for _ in columns) + " |"]
        table += [f"| {r['method']} | " + " | ".join(cell(r, key) for _, key in columns) + " |" for r in summaries]
        sections.append("\n".join(table))
    paired = []
    for seed in seeds:
        lookup = {r["method"]: r for r in rows if r["seed"] == seed}
        for left, right in (("Current-Source", "SP-ATTA"), ("Current-Feedback", "Current-Source"),
                            ("Current-Feedback", "SP-ATTA")):
            paired.append({"seed": seed, "comparison": f"{left} - {right}",
                "stream_gain_pp": lookup[left]["stream_accuracy_pct"] - lookup[right]["stream_accuracy_pct"]})
    write_csv(output / "paired.csv", paired)
    comparisons = []
    for comparison in dict.fromkeys(r["comparison"] for r in paired):
        values = [r["stream_gain_pp"] for r in paired if r["comparison"] == comparison]
        comparisons.append({"comparison": comparison, "mean_pp": statistics.mean(values),
                            "std_pp": statistics.stdev(values), "n": len(values)})
    write_csv(output / "paired_summary.csv", comparisons)
    sections.append("## Paired stream gains\n\n| Comparison | Gain (pp) |\n| :--- | ---: |\n" +
        "\n".join(f"| {r['comparison']} | {r['mean_pp']:.2f} ± {r['std_pp']:.2f} |" for r in comparisons))
    text = "# Feedback-teacher study: PACS D=100\n\n" + "\n\n".join(sections)
    text += (f"\n\nPhoto(source) → Art → Cartoon → Sketch; batch32; seeds={seeds}. Mean ± sample SD (ddof=1). "
        "Predictions precede feedback/update. Query schedules and total formal step quotas match the original reference. "
        "Formal = student + teacher-head; shadow costs and FLOPs differ and are reported separately. "
        "Current-Source also trains a diagnostic repair head but never uses its targets, matching head costs to Current-Feedback. "
        "SP keeps its original objective and full student quota. Current-vs-SP changes buffer, targets and budget allocation jointly. "
        "Query-time teacher scores cover only queried images whose labels returned; they are not unbiased stream accuracy. "
        "Local frozen AUX-TRUE probes are conditional one-step effects, not deployed whole-stream gains. "
        "No significance or superiority claim is automatic.\n")
    (output / "summary.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main():
    args = parser().parse_args()
    args.dataset = "PACS"
    args.dataset_root, args.checkpoint_dir = args.dataset_root.resolve(), args.checkpoint_dir.resolve()
    args.reference_dir = args.reference_dir.resolve()
    if args.output_dir:
        args.output_dir = args.output_dir.resolve()
    if args.worker:
        worker(args)
        return
    if args.group is not None or args.seed is not None:
        raise ValueError("--group and --seed are internal worker options")
    validate_inputs(args)
    seeds = args.seeds
    gpus = args.gpus if args.gpus is not None else [args.gpu_idx if args.gpu_idx is not None else 0]
    if len(set(seeds)) != 3 or any(not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Use three distinct valid seeds")
    if len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("Use unique nonnegative GPU indices")
    traces = {str(seed): SimATTATrace(args.reference_dir, seed, args.checkpoint_dir).hashes for seed in seeds}
    output = args.output_dir or ROOT / "storage/feedback_teacher" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "dataset": "PACS", "groups": GROUPS, "seeds": seeds,
        "delays": [100], "batch_size": 32, "gpus": gpus, "reference_dir": str(args.reference_dir),
        "reference_hashes": traces, "defaults": DEFAULTS,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "checkpoint_sha256": {name: sha256(args.checkpoint_dir / name) for name in EXPERIMENT.checkpoint_names},
        "dataset_root": str(args.dataset_root), "checkpoint_dir": str(args.checkpoint_dir),
        "python": sys.version, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    dump_json(output / "manifest.json", manifest)
    print(f"Groups: {GROUPS}\nSeeds: {seeds}\nGPUs: {gpus}\nResults: {output}", flush=True)
    rows = []
    def launch(job, gpu):
        group, seed = job
        run_dir = output / group / f"seed_{seed}"
        command = [sys.executable, "-u", "-m", "experiments.feedback_teacher.run", "--worker",
            "--group", group, "--seed", str(seed), "--gpu-idx", str(gpu),
            "--dataset-root", str(args.dataset_root), "--checkpoint-dir", str(args.checkpoint_dir),
            "--reference-dir", str(args.reference_dir), "--output-dir", str(run_dir),
            "--num-workers", str(args.num_workers)]
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
        rows.sort(key=lambda r: (GROUPS.index(r["method"]), seeds.index(r["seed"])))
        write_csv(output / "per_seed.csv", rows)
        progress.finish(result)
    try:
        with ProgressDisplay(gpus, 9) as progress:
            run_jobs([(g, s) for g in GROUPS for s in seeds], gpus, launch, save)
        summarize(output, rows, seeds, traces)
    except BaseException:
        manifest.update(status="failed", completed_runs=len(rows))
        dump_json(output / "manifest.json", manifest)
        raise
    manifest.update(status="complete", completed_runs=len(rows))
    dump_json(output / "manifest.json", manifest)
    print(f"Complete: {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
