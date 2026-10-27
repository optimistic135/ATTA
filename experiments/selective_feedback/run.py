"""PACS D100: SP, reliability-filtered source targets and selective correction."""
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
    cli.add_argument("--resume", action="store_true", help="Reuse completed tasks in --output-dir; rerun incomplete tasks")
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
    from .algorithm import SelectiveFeedback
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
    algorithm = SelectiveFeedback(config, args.group, trace)
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
        for key in ("student_updates", "head_updates", "reliability_updates", "feedback_training_sha256", "current_draws_sha256"):
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
            ("Head", "head_updates"), ("Reliability", "reliability_updates"),
            ("SP shadow", "candidate_updates"), ("Local shadow", "diagnostic_updates"), ("Seconds", "stream_seconds")]),
        ("Selected targets", [("Coverage (%)", "selected_coverage_pct"),
            ("Training draws", "selected_training_draws"), ("Target acc (%)", "selected_target_accuracy_pct"),
            ("Source on same images (%)", "selected_source_accuracy_pct"),
            ("Net correction (pp)", "target_net_correction_pp"), ("Repair draws (%)", "selected_repair_pct"),
            ("Current student steps (%)", "current_update_pct")]),
        ("Reliability and local effects", [("Source risk Brier", "prequential_source_brier"),
            ("Source confidence Brier", "prequential_source_confidence_brier"),
            ("Repair risk Brier", "prequential_repair_brier"),
            ("Repair confidence Brier", "prequential_repair_confidence_brier"),
            ("Selected-TRUE (pp)", "local_selected_minus_true_pp"),
            ("Selected-HISTORY (pp)", "local_selected_minus_history_pp"),
            ("Active selected-HISTORY (pp)", "active_local_selected_minus_history_pp")]),
    ):
        table = [f"## {title}\n", "| Method | " + " | ".join(label for label, _ in columns) + " |",
                 "| :--- | " + " | ".join("---:" for _ in columns) + " |"]
        table += [f"| {r['method']} | " + " | ".join(cell(r, key) for _, key in columns) + " |" for r in summaries]
        sections.append("\n".join(table))
    paired = []
    for seed in seeds:
        lookup = {r["method"]: r for r in rows if r["seed"] == seed}
        for left, right in (("SP-Select", "SP-ATTA"), ("SP-Correct", "SP-Select"),
                            ("SP-Correct", "SP-ATTA")):
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
    from .report import domain_reports
    sections.append(domain_reports(output, rows))
    text = "# Selective-feedback study: PACS D=100\n\n" + "\n\n".join(sections)
    text += (f"\n\nPhoto(source) → Art → Cartoon → Sketch; batch32; seeds={seeds}. Mean ± sample SD (ddof=1). "
        "Predictions precede feedback/update. Query schedules and total formal step quotas match the original reference. "
        "Formal = student + source-copy head + reliability steps. SP retains its full student quota. "
        "SP-Select and SP-Correct share feedback descriptors, training, current draws and module budgets. "
        "Correction changes which current targets are allowed; both replace only 25% of historical auxiliary weight when SP accepts. "
        "When no reliable current targets exist, student loss falls back to historical SP, but reserved module costs still apply. "
        "Brier scores use stored query-time predictions scored on return, not current population calibration. "
        "Target accuracy is weighted by actual current training draws; coverage counts selected window-image observations. "
        "Local TRUE/HISTORY/SELECTED probes always test hypothetical auxiliary coefficient0.25; "
        "only SELECTED-HISTORY isolates replacing history, and active excludes probes with zero selected samples. "
        "Shadow costs are separate and equal steps do not imply equal FLOPs. "
        "No significance or superiority claim is automatic.\n")
    (output / "summary.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main():
    args = parser().parse_args()
    args.dataset = "PACS"
    args.dataset_root, args.checkpoint_dir = args.dataset_root.resolve(), args.checkpoint_dir.resolve()
    args.reference_dir = args.reference_dir.resolve()
    if args.resume and (args.output_dir is None or args.worker):
        raise ValueError("--resume requires an explicit --output-dir and is parent-only")
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
    references = {seed: SimATTATrace(args.reference_dir, seed, args.checkpoint_dir) for seed in seeds}
    traces = {str(seed): trace.hashes for seed, trace in references.items()}
    output = args.output_dir or ROOT / "storage/selective_feedback" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    if args.resume:
        if not (output / "manifest.json").is_file():
            raise ValueError("--resume needs an existing experiment directory containing manifest.json")
    else:
        output.mkdir(parents=True, exist_ok=False)
    # Linux/macOS advisory lock: prevents two updated launchers sharing this output.
    # The local handle remains alive throughout main; process exit releases it.
    import fcntl
    output_lock = (output / ".run.lock").open("a")
    try:
        fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        output_lock.close()
        raise ValueError("Another launcher is using this output directory") from None
    manifest = {"status": "running", "dataset": "PACS", "groups": GROUPS, "seeds": seeds,
        "delays": [100], "batch_size": 32, "gpus": gpus, "reference_dir": str(args.reference_dir),
        "reference_hashes": traces, "defaults": DEFAULTS,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "checkpoint_sha256": {name: sha256(args.checkpoint_dir / name) for name in EXPERIMENT.checkpoint_names},
        "dataset_root": str(args.dataset_root), "checkpoint_dir": str(args.checkpoint_dir),
        "python": sys.version, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    rows = []
    jobs = [(g, s) for g in GROUPS for s in seeds]
    if args.resume:
        from .resume import prepare_resume
        rows, jobs = prepare_resume(output, manifest, args, references)
    manifest["completed_runs"] = len(rows)
    dump_json(output / "manifest.json", manifest)
    print(f"Groups: {GROUPS}\nSeeds: {seeds}\nGPUs: {gpus}\nResults: {output}", flush=True)
    print(f"Reusing {len(rows)} completed tasks; pending {len(jobs)}", flush=True)
    def launch(job, gpu):
        group, seed = job
        run_dir = output / group / f"seed_{seed}"
        command = [sys.executable, "-u", "-m", "experiments.selective_feedback.run", "--worker",
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
            for result in rows:
                print(f"Skipping completed: {result['method']} seed={result['seed']}", flush=True)
                progress.finish(result)
            if rows:
                write_csv(output / "per_seed.csv", rows)
            if jobs:
                run_jobs(jobs, gpus, launch, save)
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
