"""Offline, sample-weighted diagnostics. Hidden truth is used only here."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.delayed_feedback.datasets import EXPERIMENTS


def table(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return float(np.mean(values)) if len(values) else None


def local_structure(data, experiment):
    """Leave-one-out same-domain/class top-8 vs exact matched random null."""
    errors = data["prediction"] != data["truth"]
    rows = []
    for domain in experiment.environments[1:]:
        for label in np.unique(data["prediction"][data["domain"] == domain]):
            indices = np.flatnonzero((data["domain"] == domain) & (data["prediction"] == label))
            if len(indices) < 2:
                continue
            features = data["features"][indices]
            group_errors = errors[indices]
            k = min(8, len(indices) - 1)
            for start in range(0, len(indices), 256):
                stop = min(start + 256, len(indices))
                similarity = features[start:stop] @ features.T
                similarity[np.arange(stop - start), np.arange(start, stop)] = -np.inf
                # Stable stream-order tie break for exact feature ties.
                neighbors = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
                for offset, position in enumerate(range(start, stop)):
                    local_error = float(group_errors[neighbors[offset]].mean())
                    # Exact expectation for uniformly sampling any k of the
                    # other same-domain/class members, without replacement.
                    random_error = float((group_errors.sum() - int(group_errors[position]))
                                         / (len(indices) - 1))
                    rows.append({
                        "sample_id": int(data["sample_id"][indices[position]]),
                        "domain": experiment.domains[domain], "prediction": int(label),
                        "center_error": int(group_errors[position]), "neighbors": k,
                        "neighbor_error_rate": local_error, "matched_random_error_rate": random_error,
                        "excess_error_rate": local_error - random_error,
                        "mean_cosine": float(similarity[offset, neighbors[offset]].mean()),
                    })
    summary = []
    for domain in experiment.targets:
        for center_error in (0, 1):
            group = [row for row in rows if row["domain"] == domain and row["center_error"] == center_error]
            total = int(((data["domain"] == experiment.domains.index(domain))
                         & (errors == bool(center_error))).sum())
            summary.append({"domain": domain, "center_error": center_error,
                            "centers_with_neighbors": len(group), "all_centers": total,
                            "neighbor_error_rate": mean([r["neighbor_error_rate"] for r in group]),
                            "matched_random_error_rate": mean([r["matched_random_error_rate"] for r in group]),
                            "excess_error_rate": mean([r["excess_error_rate"] for r in group])})
    return rows, summary


def ranking(errors, scores, fraction):
    """Expected top-fraction errors under uniform tie breaking at boundary."""
    count = len(errors)
    if not count:
        return {"selected": 0, "precision": None, "recall": None, "lift": None}
    take = max(1, math.ceil(count * fraction))
    boundary = np.sort(scores)[-take]
    above, tied = scores > boundary, scores == boundary
    weight = (take - int(above.sum())) / int(tied.sum())
    captured = float(errors[above].sum() + weight * errors[tied].sum())
    total_errors = int(errors.sum())
    precision = captured / take
    return {"selected": take, "precision": precision,
            "recall": captured / total_errors if total_errors else None,
            "lift": precision / (total_errors / count) if total_errors else None}


def feedback_identification(data, experiment):
    errors = data["prediction"] != data["truth"]
    rows, gates = [], []
    for domain_id in experiment.environments[1:]:
        domain = experiment.domains[domain_id]
        domain_mask = data["domain"] == domain_id
        domain_n = int(domain_mask.sum())
        for scope in ("all", "covered"):
            mask = domain_mask & (data["covered"].astype(bool) if scope == "covered" else True)
            truth = errors[mask]
            covered_n = int((domain_mask & data["covered"].astype(bool)).sum())
            for error in (0, 1):
                weights = data["gate"][mask & (errors == bool(error))]
                gates.append({"domain": domain, "scope": scope, "pseudo_error": error,
                              "samples": len(weights), "mean_gate": mean(weights)})
            # Higher score always means greater predicted error risk.
            for method, scores in (("gate", 1 - data["gate"][mask]),
                                   ("entropy", data["entropy"][mask]),
                                   ("random_expectation", np.zeros(len(truth)))):
                auroc = float(roc_auc_score(truth, scores)) if len(np.unique(truth)) == 2 else None
                for fraction in (0.1, 0.2):
                    rows.append({"domain": domain, "scope": scope, "method": method,
                                 "fraction": fraction, "samples": len(truth),
                                 "errors": int(truth.sum()), "base_error_rate": mean(truth),
                                 "coverage": covered_n / domain_n if domain_n else None,
                                 "auroc": auroc, **ranking(truth, scores, fraction)})
    return rows, gates


def fmt(value):
    return "N/A" if value is None else f"{100 * value:.2f}%"


def analyze(folder, dataset):
    experiment = EXPERIMENTS[dataset]
    with np.load(folder / "evidence.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    count = len(data["sample_id"])
    if any(len(value) != count for value in data.values()):
        raise ValueError("Evidence arrays have unequal lengths")
    if len(np.unique(data["sample_id"])) != count:
        raise ValueError("Evidence contains repeated samples")
    if any(not np.isfinite(data[key]).all() for key in ("features", "entropy", "gate")):
        raise ValueError("Non-finite evidence")
    if ((data["gate"] < 0) | (data["gate"] > 1)).any():
        raise ValueError("Gate outside [0,1]")
    local, local_summary = local_structure(data, experiment)
    identification, gates = feedback_identification(data, experiment)
    table(folder / "local_neighbors.csv", local, [
        "sample_id", "domain", "prediction", "center_error", "neighbors", "neighbor_error_rate",
        "matched_random_error_rate", "excess_error_rate", "mean_cosine"])
    table(folder / "local_summary.csv", local_summary, [
        "domain", "center_error", "centers_with_neighbors", "all_centers", "neighbor_error_rate",
        "matched_random_error_rate", "excess_error_rate"])
    table(folder / "identification.csv", identification, [
        "domain", "scope", "method", "fraction", "samples", "errors", "base_error_rate",
        "coverage", "auroc", "selected", "precision", "recall", "lift"])
    table(folder / "gate_groups.csv", gates, ["domain", "scope", "pseudo_error", "samples", "mean_gate"])
    lines = ["# Frozen-source local reliability", "", f"Dataset: {dataset}; evaluation samples: {count}.",
             "No training. This is a descriptive feasibility probe, not causal performance evidence.", "",
             "## 1. Local error structure", "",
             "| Domain | Center | N with neighbors / all | Neighbor error | Matched random | Excess |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in local_summary:
        lines.append(f"| {row['domain']} | {'incorrect' if row['center_error'] else 'correct'} | "
                     f"{row['centers_with_neighbors']}/{row['all_centers']} | "
                     f"{fmt(row['neighbor_error_rate'])} | {fmt(row['matched_random_error_rate'])} | "
                     f"{fmt(row['excess_error_rate'])} |")
    lines += ["", "Stronger local-error evidence requires higher excess for incorrect centers than "
              "correct centers, consistently across domains/seeds. Inspect counts and class rows; "
              "no significance claim is made. Neighbors overlap and are not independent.", "",
              "## 2. Online feedback identification", "",
              "All held-out low-entropy samples, top 20% predicted risk. "
              "CSV also includes top 10% and covered-only rows.", "",
              "| Domain | Method | N | Errors | Coverage | AUROC | Precision@20% | Recall@20% |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in identification:
        if row["scope"] == "all" and row["fraction"] == 0.2:
            auc = "N/A" if row["auroc"] is None else f"{row['auroc']:.3f}"
            lines.append(f"| {row['domain']} | {row['method']} | {row['samples']} | {row['errors']} | "
                         f"{fmt(row['coverage'])} | {auc} | {fmt(row['precision'])} | {fmt(row['recall'])} |")
    lines += ["", "Gate should exceed random AND entropy at predeclared cutoffs, with meaningful coverage. "
              "A positive covered-only result with near-zero coverage does not establish broad utility. "
              "Tied scores are fractionally allocated at the cutoff; random is an exact expectation.", "",
              "No p values, threshold search, training benefit or forgetting claim. "
              "A negative result must be retained. Validate on data not used for method development.", ""]
    (folder / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run-dir", type=Path, required=True)
    args = cli.parse_args()
    settings = json.loads((args.run_dir / "settings.json").read_text())
    analyze(args.run_dir, settings["dataset"])


if __name__ == "__main__":
    main()
