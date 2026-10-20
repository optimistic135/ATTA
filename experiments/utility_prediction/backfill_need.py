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
        raise ValueError("Replayed query/return events differ from the completed run")
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

    if (not torch.cuda.is_available() or args.gpu_idx < 0
            or args.gpu_idx >= torch.cuda.device_count()):
        raise RuntimeError("Choose a visible CUDA GPU for factual replay")
    torch.cuda.set_device(args.gpu_idx)
    frames = []
    with tempfile.TemporaryDirectory(prefix="need_replay_", dir=root) as temporary:
        for delay in manifest["delays"]:
            original = root / f"D{delay}"
            settings = json.loads((original / "settings.json").read_text())
            if json.loads((original / "result.json").read_text())["status"] != "complete":
                raise ValueError(f"Incomplete experiment: {original}")
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
            algorithm = UtilitySimATTA(config, delay, folder, settings["steps"],
                                       settings["horizon"], save_snapshots=False)
            print(f"Replaying factual trajectory only: D={delay}", flush=True)
            algorithm.run_factual()
            reference = pd.read_csv(original / "utilities.csv")
            replayed = pd.read_csv(folder / "return_features.csv")
            frames.append(align_events(reference, replayed))
            del algorithm
            torch.cuda.empty_cache()
    output = pd.concat(frames, ignore_index=True)
    output.to_csv(destination, index=False)
    print(f"Aligned all delays; saved {len(output)} returned events: {destination}", flush=True)
    print("Original utilities and results were preserved. Now run export_features.", flush=True)


if __name__ == "__main__":
    main()
