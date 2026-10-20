"""Replay only the factual trajectory to add continuous need to completed runs.

No USE/SKIP branches are rerun and no existing experiment files are replaced.
All saved events and observable features must align before writing the sidecar.
"""

import argparse
import json
from pathlib import Path
import tempfile

from .export_features import KEYS


def align_events(reference, replayed):
    import numpy as np

    exact = ["anchor_index", "query_step", "returned_label", "query_prediction",
             "return_prediction", "query_domain", "return_domain", "prediction_changed"]
    numeric = ["query_entropy", "return_entropy", "prediction_js", "feature_temporal_cosine",
               "feature_batch_cosine", "gradient_temporal_cosine", "gradient_reference_cosine"]
    if reference.duplicated(KEYS).any() or replayed.duplicated(KEYS).any():
        raise ValueError("Duplicate events prevent factual-trajectory alignment")
    combined = reference[KEYS + exact + numeric].merge(
        replayed[KEYS + exact + numeric + ["return_true_probability", "need", "need_binary"]],
        on=KEYS, how="outer", suffixes=("_saved", "_replayed"),
        validate="one_to_one", indicator=True)
    if not combined["_merge"].eq("both").all():
        mismatch = combined.loc[combined["_merge"] != "both", KEYS + ["_merge"]].head(5)
        raise ValueError("Replayed query/return events differ from the completed run: "
                         f"{mismatch.to_dict('records')}")
    for column in exact:
        if not combined[f"{column}_saved"].eq(combined[f"{column}_replayed"]).all():
            raise ValueError(f"Factual trajectory differs in {column}; original results are unchanged")
    for column in numeric:
        if not np.allclose(combined[f"{column}_saved"], combined[f"{column}_replayed"],
                           rtol=1e-5, atol=1e-7, equal_nan=False):
            raise ValueError(f"Factual trajectory differs in {column}; check original data, "
                             "dependencies and configuration. Original results are unchanged.")
    return combined[KEYS + ["return_true_probability", "need", "need_binary"]].copy()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", type=Path, required=True, help="Completed utility experiment")
    cli.add_argument("--gpu-idx", type=int, default=0)
    cli.add_argument("--dataset-root", type=Path, help="Override original dataset location")
    cli.add_argument("--checkpoint-dir", type=Path, help="Override original source-weight location")
    cli.add_argument("--output-file", type=Path, help="New sidecar (default: output-dir/need_backfill.csv)")
    cli.add_argument("--reselect", action="store_true",
                     help="Slow diagnostic mode: repeat original online clustering instead of recorded queries")
    cli.add_argument("--cache-dir", type=Path,
                     help="Verified per-delay results (default: output-dir/need_backfill_parts)")
    args = cli.parse_args()
    root = args.output_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Backfill requires a completed utility experiment")
    destination = args.output_file or root / "need_backfill.csv"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}; choose --output-file")
    args.dataset_root = (args.dataset_root or Path(manifest["dataset_root"])).resolve()
    args.checkpoint_dir = (args.checkpoint_dir or Path(manifest["checkpoint_dir"])).resolve()
    args.dataset, args.num_workers = "PACS", 0
    from experiments.delayed_feedback.datasets import EXPERIMENTS
    from experiments.delayed_feedback.run import sha256, validate_inputs
    validate_inputs(args)
    for name in EXPERIMENTS["PACS"].checkpoint_names:
        if sha256(args.checkpoint_dir / name) != manifest["checkpoint_sha256"][name]:
            raise ValueError(f"Source checkpoint differs from original run: {name}")

    import pandas as pd
    import torch
    from ATTA import args_parser, config_summoner
    from ATTA.utils.initial import reset_random_seed
    from .engine import UtilitySimATTA
    from .scheduled_replay import ScheduledNeedReplay

    if (not torch.cuda.is_available() or args.gpu_idx < 0
            or args.gpu_idx >= torch.cuda.device_count()):
        raise RuntimeError("Choose a visible CUDA GPU for factual replay")
    torch.cuda.set_device(args.gpu_idx)
    frames = []
    cache = args.cache_dir or root / "need_backfill_parts"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="need_replay_", dir=root) as temporary:
        for delay in manifest["delays"]:
            original = root / f"D{delay}"
            settings = json.loads((original / "settings.json").read_text())
            if json.loads((original / "result.json").read_text())["status"] != "complete":
                raise ValueError(f"Incomplete experiment: {original}")
            inputs = {name: sha256(original / name)
                      for name in ("utilities.csv", "settings.json")}
            inputs["manifest.json"] = sha256(root / "manifest.json")
            if (original / "feedback.csv").is_file():
                inputs["feedback.csv"] = sha256(original / "feedback.csv")
            cached_csv, cached_meta = cache / f"D{delay}.csv", cache / f"D{delay}.json"
            if cached_csv.exists() or cached_meta.exists():
                metadata = json.loads(cached_meta.read_text()) if cached_meta.exists() else {}
                if (cached_csv.exists() and metadata.get("inputs") == inputs
                        and metadata.get("sha256") == sha256(cached_csv)):
                    frames.append(pd.read_csv(cached_csv))
                    print(f"D={delay}: reused previously verified need values", flush=True)
                    continue
                raise ValueError(f"Stale/incomplete need cache for D={delay}: {cache}; "
                                 "choose a new --cache-dir. Original results are unchanged.")
            config = config_summoner(args_parser([
                "--config_path", str(EXPERIMENTS["PACS"].config_path), "--exp_round", "1",
                "--ckpt_dir", str(args.checkpoint_dir),
                "--dataset.dataset_root", str(args.dataset_root),
                "--gpu_idx", str(args.gpu_idx), "--num_workers", "0",
                "--atta.batch_size", str(settings["batch_size"]),
            ]))
            config.random_seed = manifest["seed"]
            config.atta.gpu_clustering = False
            for name, value in settings["simatta"].items():
                config.atta.SimATTA[name] = value
            reset_random_seed(config)
            folder = Path(temporary) / f"D{delay}"
            folder.mkdir()
            reference = pd.read_csv(original / "utilities.csv")
            expected_by_step = {int(step): group for step, group in reference.groupby("return_step")}

            def verify_batch(step, events):
                expected = expected_by_step.get(step)
                if expected is None and not events:
                    return
                if expected is None or not events:
                    raise ValueError(f"D={delay}, return step={step}: expected "
                                     f"{0 if expected is None else len(expected)} returns, got {len(events)}")
                try:
                    align_events(expected, pd.DataFrame(events))
                except ValueError as error:
                    raise ValueError(f"D={delay}, return step={step}: {error}") from error
                print(f"D={delay} step={step}: aligned {len(events)} returned events", flush=True)

            if args.reselect:
                algorithm = UtilitySimATTA(config, delay, folder, settings["steps"],
                                           settings["horizon"], save_snapshots=False,
                                           event_observer=verify_batch)
            else:
                feedback_path = original / "feedback.csv"
                if not feedback_path.is_file():
                    raise FileNotFoundError(f"Recorded query schedule is missing: {feedback_path}. "
                                            "Use --reselect only if the original schedule is unavailable.")
                algorithm = ScheduledNeedReplay(
                    config, delay, folder, pd.read_csv(feedback_path), settings["steps"],
                    settings["horizon"], event_observer=verify_batch)
            mode = "reselecting queries" if args.reselect else "fixed recorded queries, no K-means/cache refresh"
            print(f"Replaying factual trajectory: D={delay} ({mode})", flush=True)
            algorithm.run_factual()
            replayed = pd.read_csv(folder / "return_features.csv")
            aligned = align_events(reference, replayed)
            aligned.to_csv(cached_csv, index=False)
            cached_meta.write_text(json.dumps({"inputs": inputs, "sha256": sha256(cached_csv)}, indent=2))
            frames.append(aligned)
            del algorithm
            torch.cuda.empty_cache()
    output = pd.concat(frames, ignore_index=True)
    output.to_csv(destination, index=False)
    print(f"Aligned all delays; saved {len(output)} returned events: {destination}", flush=True)
    print("Original utilities and results were preserved. Now run export_features.", flush=True)


if __name__ == "__main__":
    main()
