"""Streaming offline summaries; evaluation truth is never sent back to training."""

import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import statistics

MODES = ("baseline", "oracle_correct", "random_count")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Aggregate:
    def __init__(self):
        self.n = 0
        self.gate = self.removed = self.loss = self.removed_loss = self.restored = 0.0

    def add(self, gate, effective, loss):
        self.n += 1
        self.gate += gate
        self.removed += 1 - gate
        self.loss += loss
        self.removed_loss += (1 - gate) * loss
        self.restored += effective - gate

    def row(self):
        return {"n": self.n, "gate_mean": self.gate / self.n,
                "suppression_mean": self.removed / self.n,
                "raw_ce_mean": self.loss / self.n,
                "removed_ce_mean": self.removed_loss / self.n,
                "restored_weight_mean": self.restored / self.n}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def analyze(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("method") != "GateAudit" or tuple(manifest["modes"]) != MODES:
        raise ValueError("Expected the three-arm GateAudit experiment")
    conditions, unique, paired, costs = [], [], [], []
    for seed in manifest["seeds"]:
        runs, restores, prefix_batches = {}, {}, {}
        for mode in MODES:
            folder = output / mode / f"seed_{seed}"
            result = json.loads((folder / "result.json").read_text())
            if result.get("method") != "GateAudit" or result["mode"] != mode or result["seed"] != seed:
                raise ValueError("Wrong diagnostic identity")
            runs[mode] = result
            prefix_batches[mode] = [(row["step"], row["samples"], row["correct"])
                                    for row in read_csv(folder / "batches.csv") if row["domain"] != "Sketch"]
            aggregates = defaultdict(Aggregate)
            sample_aggregates = defaultdict(Aggregate)
            restored_by_update = defaultdict(int)
            restored_mass = 0.0
            with gzip.open(folder / "pseudo_draws.csv.gz", "rt", newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    gate, effective, loss = float(row["gate"]), float(row["effective_gate"]), float(row["raw_ce"])
                    if row["domain"] == "Sketch":
                        restored_by_update[(int(row["step"]), int(row["iteration"]))] += int(effective > gate)
                        restored_mass += effective - gate
                    # Baseline alone is used to diagnose conditions; interventions
                    # are assessed separately and must not redefine strata.
                    if mode != "baseline":
                        continue
                    cross = ("no_support" if not row["cross_source_fraction"] else
                             "cross_majority" if float(row["cross_source_fraction"]) >= 0.5 else "same_majority")
                    for axis, value in (("all", "all"), ("neighbors", row["neighbor_bin"]),
                                        ("support", row["support_bin"]), ("cross_source", cross)):
                        key = (row["domain"], row["source_domain"], axis, value, int(row["correct"]))
                        aggregates[key].add(gate, effective, loss)
                        sample_aggregates[key + (int(row["sample_id"]),)].add(gate, effective, loss)
            restores[mode] = restored_by_update
            costs.append({"seed": seed, "mode": mode, "restored_draws": sum(restored_by_update.values()),
                          "restored_weight_mass": restored_mass})
            for key, aggregate in aggregates.items():
                domain, source, axis, value, correct = key
                conditions.append({"seed": seed, "domain": domain, "source_domain": source,
                                   "axis": axis, "bin": value, "correct": correct, **aggregate.row()})
            # Within each stratum, average each distinct sample's recorded draws
            # first, then give samples equal weight. A sample may visit several strata.
            grouped = defaultdict(list)
            for key, aggregate in sample_aggregates.items():
                grouped[key[:-1]].append(aggregate.row())
            for key, values in grouped.items():
                domain, source, axis, value, correct = key
                unique.append({"seed": seed, "domain": domain, "source_domain": source,
                               "axis": axis, "bin": value, "correct": correct,
                               "unique_samples": len(values),
                               **{name: statistics.mean(row[name] for row in values)
                                  for name in ("gate_mean", "suppression_mean", "removed_ce_mean")}})
        for key in ("prefix_sha256", "train_draw_sha256", "slow_updates", "queried", "returned"):
            values = {row[key] for row in runs.values()}
            if len(values) != 1 or None in values:
                raise ValueError(f"Unpaired state/budget for seed {seed}: {key}")
        if not prefix_batches["baseline"] == prefix_batches["oracle_correct"] == prefix_batches["random_count"]:
            raise ValueError("Pre-Sketch predictions diverged; no common prefix")
        if restores["oracle_correct"] != restores["random_count"]:
            raise ValueError("Random control failed per-update restoration-count matching")
        for treatment, control in (("oracle_correct", "baseline"), ("random_count", "baseline"),
                                   ("oracle_correct", "random_count")):
            paired.append({"seed": seed, "contrast": f"{treatment}_minus_{control}",
                           "Sketch_delta_pp": runs[treatment]["Sketch_accuracy_pct"] - runs[control]["Sketch_accuracy_pct"],
                           "Stream_delta_pp": runs[treatment]["stream_accuracy_pct"] - runs[control]["stream_accuracy_pct"]})
    write_csv(output / "conditions_draw_weighted.csv", conditions)
    write_csv(output / "conditions_sample_weighted.csv", unique)
    write_csv(output / "intervention_deltas.csv", paired)
    write_csv(output / "restoration_costs.csv", costs)
    lines = ["# Local pseudo-supervision gate: diagnostic report", "",
             "Oracle-correct and random-count use hidden correctness. They are diagnostic interventions, not ATTA methods.",
             "Interventions start only at Sketch. Prefix model/memory/RNG hashes, sampling hashes, pre-Sketch",
             "correct counts and update/label budgets match across arms. Prefixes are replayed, not loaded from one checkpoint.", "",
             "| Seed | Contrast | Sketch delta (pp) | Stream delta (pp) |",
             "| ---: | --- | ---: | ---: |"]
    for row in paired:
        lines.append(f"| {row['seed']} | {row['contrast']} | {row['Sketch_delta_pp']:+.3f} | {row['Stream_delta_pp']:+.3f} |")
    lines += ["", "## Baseline training-draw gates", "",
              "| Seed | Current domain | Correct pseudo gate | Incorrect pseudo gate |",
              "| ---: | --- | ---: | ---: |"]
    for seed in manifest["seeds"]:
        for domain in ("Art", "Cartoon", "Sketch"):
            cells = []
            for correct in (1, 0):
                group = [row for row in conditions if row["seed"] == seed and row["domain"] == domain
                         and row["axis"] == "all" and row["correct"] == correct]
                count = sum(row["n"] for row in group)
                cells.append(f"{sum(row['n'] * row['gate_mean'] for row in group) / count:.4f}"
                             if count else "N/A")
            lines.append(f"| {seed} | {domain} | {' | '.join(cells)} |")
    lines += ["", "Inspect baseline correct vs incorrect rows in conditions_draw_weighted.csv and",
              "conditions_sample_weighted.csv. Suppression of a correct label is not itself proof of harm.",
              "Support bins: zero=0, weak=(0,1], medium=(1,3], strong>3. Neighbor bins: 0, 1-2, 3-8.",
              "Cross-source bins use weighted neighbor-domain mismatch to the pseudo sample, threshold=0.5.",
              "Group definitions are fixed; strata are descriptive and do not establish causality.",
              "Random control matches restored draw count per update, not weight mass; inspect restoration_costs.csv.",
              "A single randomization per seed cannot rule out randomness; oracle advantage is suggestive, not conclusive.",
              "Training CE is evaluated on the evolving model; removed CE is not a gradient norm or a risk estimate.",
              "No claim of significance; selected seeds are development cases. Validate changes on untouched seeds/datasets.",
              "The audit has no per-sample model counterfactual; it cannot identify individual repaired/broken predictions."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--results-dir", type=Path, required=True)
    analyze(cli.parse_args().results_dir)


if __name__ == "__main__":
    main()
