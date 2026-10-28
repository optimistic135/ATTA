"""Three-seed controlled TAST effects and same-snapshot label-only diagnostics."""
import statistics

from experiments.delayed_feedback.run import write_csv
from experiments.sp_atta.reference import read_csv
from .spec import GROUPS, METRICS, DEFAULTS as D


def stats(values):
    values = [v for v in values if v is not None]
    return {"mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None, "n": len(values)}


def cell(row, key):
    mean, std = row[key + "_mean"], row[key + "_std"]
    return "N/A" if mean is None else f"{mean:.2f}" + (f" ± {std:.2f}" if std is not None else "")


def summarize(output, results, seeds):
    if len(results) != 9 or {(r["method"], r["seed"]) for r in results} != {(g, s) for g in GROUPS for s in seeds}:
        raise ValueError("Need all nine completed jobs")
    lookup = {(r["method"], r["seed"]): r for r in results}
    comparisons = ((GROUPS[1], GROUPS[0]), (GROUPS[2], GROUPS[1]), (GROUPS[2], GROUPS[0]))
    pairs, domain_rows = [], []
    for seed in seeds:
        matched = [lookup[g, seed] for g in GROUPS]
        for key in ("reference_hashes", "initial_head_sha256", "frozen_source_sha256", "source_features_sha256",
                    "replay_draws_sha256", "support_ids_sha256", "formal_updates", "queried", "returned", "pending"):
            if any(r[key] != matched[0][key] for r in matched[1:]):
                raise ValueError(f"Matched protocol differs at {key}, seed={seed}")
        if matched[1]["replay_updates"] != matched[2]["replay_updates"] or matched[1]["replay_draws"] != matched[2]["replay_draws"]:
            raise ValueError("Replay budgets differ")
        if matched[0]["replay_updates"] or matched[0]["repaired_labels"] or matched[1]["repaired_labels"]:
            raise ValueError("Truth was used in an unintended arm")
        batches = [read_csv(output / g / f"seed_{seed}" / "batches.csv") for g in GROUPS]
        if len({len(b) for b in batches}) != 1:
            raise ValueError("Batch lengths differ")
        for observations in zip(*batches):
            a = observations[0]
            for b in observations[1:]:
                for key in ("step", "domain", "samples", "before_feedback", "released", "returned_memory", "source_correct", "repair_exposed"):
                    if a[key] != b[key]:
                        raise ValueError(f"Paired batch protocol differs at {key}")
                if a["before_feedback"].lower() == "true" and a["online_predictions_sha256"] != b["online_predictions_sha256"]:
                    raise ValueError("Branches differ before first feedback")
        for left, right in comparisons:
            for metric in ("Art_accuracy_pct", "Cartoon_accuracy_pct", "Sketch_accuracy_pct", "stream_accuracy_pct",
                           "domain_macro_accuracy_pct", "before_feedback_accuracy_pct", "after_feedback_accuracy_pct"):
                a, b = lookup[left, seed][metric], lookup[right, seed][metric]
                pairs.append({"seed": seed, "comparison": f"{left} − {right}", "metric": metric,
                              "gain_pp": a - b if a is not None and b is not None else None})
        for group, observations in zip(GROUPS, batches):
            for phase in ("all", "before_feedback", "after_feedback"):
                for domain in ("ALL", "Art", "Cartoon", "Sketch"):
                    rows = [r for r in observations if (domain == "ALL" or r["domain"] == domain)
                        and (phase == "all" or (r["before_feedback"].lower() == "true") == (phase == "before_feedback"))]
                    n = sum(int(r["samples"]) for r in rows)
                    row = {"seed": seed, "method": group, "phase": phase, "domain": domain, "samples": n}
                    for key, field in (("plain_target_accuracy_pct", "label_repair_plain_correct"),
                        ("repaired_target_accuracy_pct", "label_repair_corrected_correct"),
                        ("fixed_pct", "label_repair_fixed"), ("broken_pct", "label_repair_broken"),
                        ("exposure_pct", "repair_exposed"), ("label_only_prediction_gain_pp", "label_repair_prediction_gain")):
                        row[key] = 100 * sum(int(r[field]) for r in rows) / n if n else None
                    domain_rows.append(row)
    summaries = []
    for group in GROUPS:
        rows = [r for r in results if r["method"] == group]
        summary = {"method": group, "delay": 100, "runs": len(rows)}
        for metric in METRICS:
            for suffix, value in stats([r[metric] for r in rows]).items():
                summary[f"{metric}_{suffix}"] = value
        summaries.append(summary)
    paired_summary = []
    for left, right in comparisons:
        comparison = f"{left} − {right}"
        for metric in dict.fromkeys(r["metric"] for r in pairs):
            paired_summary.append({"comparison": comparison, "metric": metric,
                **stats([r["gain_pp"] for r in pairs if r["comparison"] == comparison and r["metric"] == metric])})
    for name, rows in (("per_seed.csv", results), ("summary.csv", summaries), ("paired_gains.csv", pairs),
                       ("paired_summary.csv", paired_summary), ("repair_by_domain.csv", domain_rows)):
        write_csv(output / name, rows)
    lines = ["# Delayed TAST-style controlled study: PACS D=100", "", "Photo(source) → Art → Cartoon → Sketch; batch32. "
        f"Seeds={seeds}; mean ± sample SD (ddof=1).", ""]
    tables = (
        ("Online accuracy", (("Art (%)", "Art_accuracy_pct"), ("Cartoon (%)", "Cartoon_accuracy_pct"),
            ("Sketch (%)", "Sketch_accuracy_pct"), ("Macro (%)", "domain_macro_accuracy_pct"),
            ("Stream (%)", "stream_accuracy_pct"), ("Before feedback (%)", "before_feedback_accuracy_pct"),
            ("After feedback (%)", "after_feedback_accuracy_pct"))),
        ("Waiting phase and label repair diagnostics", (("Source before feedback (%)", "source_before_feedback_accuracy_pct"),
            ("Waiting gain vs source (pp)", "waiting_gain_vs_source_pp"), ("Repaired labels", "repaired_labels"),
            ("Wrong source labels repaired", "repaired_wrong_source_labels"), ("Truth neighbor exposure (%)", "repair_exposure_pct"),
            ("Label-only target gain (pp)", "label_repair_target_gain_pp"),
            ("Label-only prediction gain (pp)", "label_repair_prediction_gain_pp"))),
        ("Feedback and costs", (("Queried", "queried"), ("Returned", "returned"), ("Returned (%)", "returned_pct"),
            ("Cross returned", "cross_domain_returned"), ("Cross (%)", "cross_domain_pct"), ("Pending", "pending"),
            ("Head steps", "formal_updates"), ("Replay steps", "replay_updates"), ("Replay draws", "replay_draws"),
            ("Replay unique labels", "replay_unique_labels"), ("Seconds", "stream_seconds"))),
    )
    for title, columns in tables:
        lines += [f"## {title}", "", "| Method | " + " | ".join(n for n, _ in columns) + " |",
                  "| :--- | " + " | ".join("---:" for _ in columns) + " |"]
        for row in summaries:
            lines.append("| " + " | ".join([row["method"]] + [cell(row, key) for _, key in columns]) + " |")
        lines.append("")
    lines += ["## Paired stream gains", "", "| Comparison | Gain (pp) |", "| :--- | ---: |"]
    for row in paired_summary:
        if row["metric"] == "stream_accuracy_pct":
            value = "N/A" if row["mean"] is None else f"{row['mean']:.2f}" + (f" ± {row['std']:.2f}" if row["std"] is not None else "")
            lines.append(f"| {row['comparison']} | {value} |")
    lines += ["", "TAST-Online is a controlled online TAST-style implementation, NOT an exact reproduction of the original paper. "
        "All groups predict before current support insertion, release and adaptation, retain state across domains, "
        "freeze the same source encoder/classifier, and use the same initialized lightweight ensemble. "
        "All groups share source-entropy support IDs union returned query features, even if evicted from the entropy bank. "
        "Online/Replayed supports keep source labels; only Repair uses true prototype labels and authoritative neighbor votes. "
        "This common returned-feature capacity is a deliberate control differing from plain original TAST. "
        "Online does not use true categories in predictions or updates; returned categories may appear in evaluation-only diagnostics. "
        "Replay and Repair share actual supervised draws and optimizer quotas. One head update per batch includes waiting updates; "
        "the old SP per-batch optimizer quotas are NOT reused. Equal steps do not imply equal FLOPs. "
        "Label-only diagnostics compare unrepaired/repaired target votes and soft outputs using the SAME current head, "
        "features and support IDs before release/update. They are not cross-trajectory gains or proof of future benefit. "
        "Repair changes prototypes, target votes and inference votes jointly; these three arms do not separate those channels. "
        "No unseen current/future truth enters training, pruning, support retention or queries. "
        "First return batch belongs to before-feedback. Cross percentages exclude pending labels, computed per seed. "
        "The baseline still executes matched query events but never learns their true categories; its query cost represents "
        "the common study protocol, not the intrinsic labeling requirement of unsupervised TAST. "
        "Three seeds are independent replicates; no performance, significance or novelty claim is automatic."]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
