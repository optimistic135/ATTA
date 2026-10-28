"""Seed-paired online gains and evaluation-only risk estimation diagnostics."""
import statistics

from experiments.delayed_feedback.run import write_csv
from experiments.sp_atta.reference import read_csv
from .spec import GROUPS, METRICS


def stats(values):
    values = [v for v in values if v is not None]
    return {"mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None, "n": len(values)}


def number(value):
    return None if value in (None, "", "None") else float(value)


def cell(row, key):
    mean, std = row[key + "_mean"], row[key + "_std"]
    if mean is None:
        return "N/A"
    precision = 4 if "risk" in key or "residual" in key or "fraction" in key or "density_holdout" in key else 2
    return f"{mean:.{precision}f}" + (f" ± {std:.{precision}f}" if std is not None else "")


def summarize(output, results, seeds):
    if len(results) != 9 or {(r["method"], r["seed"]) for r in results} != {(g, s) for g in GROUPS for s in seeds}:
        raise ValueError("Need all nine distinct completed jobs")
    lookup = {(r["method"], r["seed"]): r for r in results}
    paired, domain_rows = [], []
    comparisons = ((GROUPS[1], GROUPS[0]), (GROUPS[2], GROUPS[1]), (GROUPS[2], GROUPS[0]))
    for seed in seeds:
        matched = [lookup[g, seed] for g in GROUPS]
        for key in ("reference_hashes", "formal_draws_sha256", "pseudo_pool_sha256", "source_cache_sha256",
                    "main_updates", "queried", "returned", "cross_domain_returned", "pending", "density_updates"):
            if any(r[key] != matched[0][key] for r in matched[1:]):
                raise ValueError(f"Matched protocol differs at {key}, seed={seed}")
        if matched[1]["current_aux_updates"] != matched[2]["current_aux_updates"]:
            raise ValueError("Current-soft and rectified activation budgets differ")
        batches = [read_csv(output / g / f"seed_{seed}" / "risk_batches.csv") for g in GROUPS]
        if len({len(b) for b in batches}) != 1:
            raise ValueError("Batch lengths differ")
        for observations in zip(*batches):
            a = observations[0]
            for b in observations[1:]:
                for key in ("step", "domain", "samples", "before_feedback", "released", "memory", "eligible",
                            "reason", "ess", "ess_fraction", "raw_ess", "clip_fraction", "holdout_accuracy"):
                    if a[key] != b[key]:
                        raise ValueError(f"Batch pairing differs at {key}, seed={seed}")
                if a["before_feedback"].lower() == "true" and a["online_predictions_sha256"] != b["online_predictions_sha256"]:
                    raise ValueError("Predictions changed before first feedback")
        for left, right in comparisons:
            for metric in ("stream_accuracy_pct", "domain_macro_accuracy_pct", "Art_accuracy_pct", "Cartoon_accuracy_pct",
                           "Sketch_accuracy_pct", "before_feedback_accuracy_pct", "after_feedback_accuracy_pct"):
                a, b = lookup[left, seed][metric], lookup[right, seed][metric]
                paired.append({"seed": seed, "comparison": f"{left} − {right}", "metric": metric,
                               "gain_pp": a - b if a is not None and b is not None else None})
        for group, observations in zip(GROUPS, batches):
            for domain in ("ALL", "Art", "Cartoon", "Sketch"):
                rows = [r for r in observations if r["eligible"].lower() == "true" and (domain == "ALL" or r["domain"] == domain)]
                n = sum(int(r["samples"]) for r in rows)
                row = {"seed": seed, "method": group, "domain": domain, "eligible_batches": len(rows), "samples": n}
                for key, field in (("proxy_mae", "proxy_risk"), ("rectified_mae", "rectified_risk")):
                    row[key] = sum(abs(number(r[field]) - number(r["actual_current_risk"])) * int(r["samples"]) for r in rows) / n if n else None
                for key in ("ess", "ess_fraction", "raw_ess", "clip_fraction", "holdout_accuracy"):
                    row["mean_" + key] = statistics.mean(number(r[key]) for r in rows) if rows else None
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
        for metric in dict.fromkeys(r["metric"] for r in paired):
            paired_summary.append({"comparison": comparison, "metric": metric,
                **stats([r["gain_pp"] for r in paired if r["comparison"] == comparison and r["metric"] == metric])})
    for name, rows in (("per_seed.csv", results), ("summary.csv", summaries), ("paired_gains.csv", paired),
                       ("paired_summary.csv", paired_summary), ("risk_by_domain.csv", domain_rows)):
        write_csv(output / name, rows)
    lines = ["# Delayed risk rectification: PACS D=100", "", "Photo(source) → Art → Cartoon → Sketch; batch32. "
             f"Seeds={seeds}; mean ± sample SD (ddof=1).", ""]
    tables = (
        ("Online accuracy", (("Art (%)", "Art_accuracy_pct"), ("Cartoon (%)", "Cartoon_accuracy_pct"),
            ("Sketch (%)", "Sketch_accuracy_pct"), ("Macro (%)", "domain_macro_accuracy_pct"),
            ("Stream (%)", "stream_accuracy_pct"), ("Before feedback (%)", "before_feedback_accuracy_pct"),
            ("After feedback (%)", "after_feedback_accuracy_pct"))),
        ("Feedback and costs", (("Queried", "queried"), ("Returned", "returned"), ("Returned (%)", "returned_pct"),
            ("Cross returned", "cross_domain_returned"), ("Cross (%)", "cross_domain_pct"), ("Pending", "pending"),
            ("Formal steps", "main_updates"), ("SP shadow", "candidate_updates"), ("Current aux steps", "current_aux_updates"),
            ("Residual steps", "rectification_updates"), ("Seconds", "stream_seconds"))),
        ("Density and risk diagnostics", (("Eligible batches", "eligible_batches"), ("Mean ESS", "mean_ess"),
            ("ESS / memory", "mean_ess_fraction"), ("Raw ESS", "mean_raw_ess"), ("Clipped fraction", "mean_clip_fraction"),
            ("Holdout density acc", "mean_density_holdout_accuracy"), ("Proxy risk MAE", "proxy_risk_mae"),
            ("Rectified risk MAE", "rectified_risk_mae"), ("Mean training residual", "mean_training_residual"),
            ("Extra bounded truth coeff", "mean_extra_bounded_truth_coefficient"))),
    )
    for title, columns in tables:
        lines += [f"## {title}", "", "| Method | " + " | ".join(n for n, _ in columns) + " |",
                  "| :--- | " + " | ".join("---:" for _ in columns) + " |"]
        for row in summaries:
            lines.append("| " + " | ".join([row["method"]] + [cell(row, m) for _, m in columns]) + " |")
        lines.append("")
    lines += ["## Paired stream gains", "", "| Comparison | Gain (pp) |", "| :--- | ---: |"]
    for row in paired_summary:
        if row["metric"] != "stream_accuracy_pct":
            continue
        value = "N/A" if row["mean"] is None else f"{row['mean']:.2f}" + (f" ± {row['std']:.2f}" if row["std"] is not None else "")
        lines.append(f"| {row['comparison']} | {value} |")
    lines += ["", "All arms use the original query trace, source checkpoint, complete returned memory, truth/history draws and "
        "formal quotas. Both auxiliary arms share activation conditions, current images, soft targets and coefficient. "
        "Only RiskRectify adds the weighted delayed true-minus-soft residual, using the same truth draws. "
        "SP retains its original objective; shadow gates remain SP tests and may make different decisions after trajectories diverge. "
        "Density fitting and pre-update memory-risk diagnostics run in all arms as shared diagnostic overhead. "
        "Density updates are CPU analytic fitting steps, NOT formal backbone updates. Equal formal steps do not imply equal FLOPs. "
        "The source classifier defines soft targets; there is no correction head or external model. "
        "Current truth is used only for offline target/risk/accuracy scoring; never for density fitting, activation or training. "
        "Risk MAE compares estimates of the bounded pre-update batch loss on the same eligible batches; it is not a gradient-direction test. "
        "ESS is calculated AFTER clipping and self-normalization and does not prove overlap. High density classification accuracy "
        "indicates separability, not good calibration. Two-fold feature density fitting is not a full theoretical cross-fitting proof. "
        "Original active queries, feature-space ratios and clipping invalidate any automatic unbiasedness/double-robustness claim. "
        "RiskRectify adds BOTH weighted bounded truth and negative weighted historical soft loss; these three arms "
        "test their combined residual effect, not the negative term separately from stronger real supervision. "
        "No changes before first feedback are expected. Cross percentages exclude pending labels, calculated per seed. "
        "Three seeds are the independent replicates. No performance or significance claim is automatic."]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
