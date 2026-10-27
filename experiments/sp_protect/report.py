"""Seed-paired online effects, plus explicitly non-causal ten-batch windows."""
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
    if mean is None:
        return "N/A"
    precision = 6 if key == "mean_protection_kl" else 2
    return f"{mean:.{precision}f}" + (f" ± {std:.{precision}f}" if std is not None else "")


def summarize(output, results, seeds):
    if len(results) != 6 or {(r["method"], r["seed"]) for r in results} != {(g, s) for g in GROUPS for s in seeds}:
        raise ValueError("Need all six completed paired jobs")
    lookup = {(r["method"], r["seed"]): r for r in results}
    pairs, windows, regions = [], [], []
    for seed in seeds:
        baseline, protected = (lookup[g, seed] for g in GROUPS)
        for key in ("reference_hashes", "formal_draws_sha256", "pseudo_pool_sha256", "selection_sha256",
                    "main_updates", "queried", "returned", "cross_domain_returned", "pending"):
            if baseline[key] != protected[key]:
                raise ValueError(f"Matched protocol differs at {key}, seed={seed}")
        batches = [read_csv(output / g / f"seed_{seed}" / "protection_batches.csv") for g in GROUPS]
        for group, observations in zip(GROUPS, batches):
            for domain in ("ALL", "Art", "Cartoon", "Sketch"):
                rows = [r for r in observations if int(r["updates"]) > 0 and (domain == "ALL" or r["domain"] == domain)]
                count = sum(int(r["selected"]) for r in rows)
                region = {"seed": seed, "method": group, "domain": domain,
                          "trained_batches": len(rows), "selected": count}
                for metric, field in (("pre_accuracy_pct", "selected_pre_correct"),
                        ("post_accuracy_pct", "selected_post_correct"), ("fixed_pct", "selected_fixed"),
                        ("broken_pct", "selected_broken"), ("source_accuracy_pct", "selected_source_correct")):
                    region[metric] = 100 * sum(int(r[field]) for r in rows) / count if count else None
                regions.append(region)
        if len(batches[0]) != len(batches[1]):
            raise ValueError("Paired batch lengths differ")
        for a, b in zip(*batches):
            for key in ("step", "domain", "samples", "selected", "base_candidates", "before_feedback", "released", "updates"):
                if a[key] != b[key]:
                    raise ValueError(f"Paired batch protocol differs: {key}")
            if a["before_feedback"].lower() == "true" and a["online_predictions_sha256"] != b["online_predictions_sha256"]:
                raise ValueError("Waiting-period predictions changed despite no optimizer updates")
        for metric in ("stream_accuracy_pct", "domain_macro_accuracy_pct", "Art_accuracy_pct", "Cartoon_accuracy_pct",
                       "Sketch_accuracy_pct", "before_feedback_accuracy_pct", "after_feedback_accuracy_pct",
                       "selected_fixed_pct", "selected_broken_pct"):
            a, b = baseline[metric], protected[metric]
            pairs.append({"seed": seed, "metric": metric, "protect_minus_sp_pp": b - a if a is not None and b is not None else None})
        # Both deployed models continue training. Overlapping windows are descriptive,
        # not independent interventions and never enter a gate or a training decision.
        for start, row in enumerate(batches[0]):
            if not int(row["released"]) or start + D["future_batches"] >= len(batches[0]):
                continue
            end = start + D["future_batches"] + 1
            for domain in ("ALL", "Art", "Cartoon", "Sketch"):
                matched = [(a, b) for a, b in zip(batches[0][start + 1:end], batches[1][start + 1:end])
                           if domain == "ALL" or a["domain"] == domain]
                total = sum(int(a["samples"]) for a, _ in matched)
                windows.append({"seed": seed, "return_step": int(row["step"]), "future_domain": domain,
                    "future_batches": len(matched), "samples": total,
                    "sp_correct": sum(int(a["pre_correct"]) for a, _ in matched),
                    "protect_correct": sum(int(b["pre_correct"]) for _, b in matched),
                    "gain_pp": 100 * sum(int(b["pre_correct"]) - int(a["pre_correct"]) for a, b in matched) / total if total else None})
    summaries = []
    for group in GROUPS:
        rows = [r for r in results if r["method"] == group]
        summary = {"method": group, "delay": 100, "runs": len(rows)}
        for metric in METRICS:
            for suffix, value in stats([r[metric] for r in rows]).items():
                summary[f"{metric}_{suffix}"] = value
        summaries.append(summary)
    paired_summary = []
    for metric in dict.fromkeys(r["metric"] for r in pairs):
        paired_summary.append({"metric": metric, **stats([r["protect_minus_sp_pp"] for r in pairs if r["metric"] == metric])})
    for name, rows in (("per_seed.csv", results), ("summary.csv", summaries), ("paired_gains.csv", pairs),
                       ("paired_summary.csv", paired_summary), ("trained_region_effects.csv", regions)):
        write_csv(output / name, rows)
    if windows:
        write_csv(output / "future_windows.csv", windows)
    else:
        (output / "future_windows.csv").write_text("seed,return_step,future_domain,future_batches,samples,sp_correct,protect_correct,gain_pp\n", encoding="utf-8")
    lines = ["# SP-Protect: PACS D=100", "", "Photo(source) → Art → Cartoon → Sketch; batch32. "
             f"Seeds={seeds}; mean ± sample SD (ddof=1).", ""]
    tables = (
        ("Online accuracy", (("Art (%)", "Art_accuracy_pct"), ("Cartoon (%)", "Cartoon_accuracy_pct"),
            ("Sketch (%)", "Sketch_accuracy_pct"), ("Macro (%)", "domain_macro_accuracy_pct"),
            ("Stream (%)", "stream_accuracy_pct"), ("Before feedback (%)", "before_feedback_accuracy_pct"),
            ("After feedback (%)", "after_feedback_accuracy_pct"))),
        ("Feedback and costs", (("Queried", "queried"), ("Returned", "returned"), ("Returned (%)", "returned_pct"),
            ("Cross returned", "cross_domain_returned"), ("Cross (%)", "cross_domain_pct"), ("Pending", "pending"),
            ("Formal steps", "main_updates"), ("SP shadow", "candidate_updates"), ("Protected steps", "protection_updates"),
            ("Seconds", "stream_seconds"))),
        ("Current selected region (post-update evaluation only)", (("Selected observations", "selected_observations"),
            ("Pre acc (%)", "selected_accuracy_before_pct"), ("Post acc (%)", "selected_accuracy_after_pct"),
            ("Fixed (%)", "selected_fixed_pct"), ("Broken (%)", "selected_broken_pct"), ("Mean KL", "mean_protection_kl"))),
    )
    for title, columns in tables:
        lines += [f"## {title}", "", "| Method | " + " | ".join(n for n, _ in columns) + " |",
                  "| :--- | " + " | ".join("---:" for _ in columns) + " |"]
        for row in summaries:
            lines.append("| " + " | ".join([row["method"]] + [cell(row, m) for _, m in columns]) + " |")
        lines.append("")
    lines += ["## Paired gains: SP-Protect − SP", "", "| Metric | Gain (pp) |", "| :--- | ---: |"]
    for row in paired_summary:
        mean, std = row["mean"], row["std"]
        value = "N/A" if mean is None else f"{mean:.2f}" + (f" ± {std:.2f}" if std is not None else "")
        lines.append(f"| {row['metric']} | {value} |")
    lines += ["", "Only the SP-Protect formal loss adds current-region KL; the source selector, query/return schedule, "
        "truth/history draws, pseudo pools and formal step quotas are matched. SP gates use the original objective; "
        "their outcomes can differ after the trajectories diverge. No new pretrained model or CLIP dependency. "
        "Targets are detached pre-batch online probabilities on the SAME input view, fixed for the whole batch's training. "
        "Initial KL/gradient is approximately zero; this constrains later steps, not a new first-step teaching signal. "
        "Before-feedback includes the first return batch, whose prediction precedes release. "
        "Post-update selected-region accuracy is evaluated on already-seen images, never online accuracy or independent evidence of generalization. "
        "future_windows.csv compares subsequent ten deployed batches; windows can overlap and both models continue training. "
        "Those windows are descriptive and cannot isolate one return's causal effect. "
        "Extra protection forwards/backwards are reported in summary.csv; equal steps do not imply equal FLOPs. "
        "Cross percentages exclude pending labels and are calculated per seed. "
        "Three seeds, not selected images or windows, are the independent replicates. No gain or significance claim is automatic."]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
