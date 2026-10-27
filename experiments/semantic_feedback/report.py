"""Evaluation-only sample-weighted summaries with seed-level paired effects."""
import statistics

from experiments.delayed_feedback.run import write_csv
from .spec import ARMS


def arm_results(seed, batches):
    rows = []
    for arm in ARMS:
        for phase in ("all", "before_feedback", "after_feedback"):
            for domain in ("ALL", "Art", "Cartoon", "Sketch"):
                selected = [r for r in batches if r["arm"] == arm
                    and (domain == "ALL" or r["domain"] == domain)
                    and (phase == "all" or r["before_feedback"] == (phase == "before_feedback"))]
                total = sum(r["samples"] for r in selected)
                rows.append({"seed": seed, "arm": arm, "phase": phase, "domain": domain,
                    "samples": total, "accuracy_pct": 100 * sum(r["correct"] for r in selected) / total if total else None,
                    "fixed_vs_sp_pct": 100 * sum(r["fixed_vs_sp"] for r in selected) / total if total else None,
                    "broken_vs_sp_pct": 100 * sum(r["broken_vs_sp"] for r in selected) / total if total else None})
    return rows


def aggregate(values):
    values = [v for v in values if v is not None]
    return {"mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None, "n": len(values)}


def display(values):
    summary = aggregate(values)
    if not summary["n"]:
        return "N/A"
    mean = f"{summary['mean']:.2f}"
    return mean + (f" ± {summary['std']:.2f}" if summary["std"] is not None else "")


def summarize(output, results):
    write_csv(output / "per_seed.csv", [{k: v for k, v in r.items() if k != "arms"} for r in results])
    per_arm = [row for result in results for row in result["arms"]]
    write_csv(output / "per_arm.csv", per_arm)
    lookup = {(r["seed"], r["arm"], r["phase"], r["domain"]): r for r in per_arm}
    seeds = [r["seed"] for r in results]
    for result in results:
        seed = result["seed"]
        for domain in ("ALL", "Art", "Cartoon", "Sketch"):
            factual = result["stream_accuracy_pct" if domain == "ALL" else f"{domain}_accuracy_pct"]
            if abs(lookup[seed, ARMS[0], "all", domain]["accuracy_pct"] - factual) > 1e-8:
                raise AssertionError("SP arm and factual parent accuracy disagree")
    summaries = []
    for arm in ARMS:
        for phase in ("all", "before_feedback", "after_feedback"):
            for domain in ("ALL", "Art", "Cartoon", "Sketch"):
                row = {"arm": arm, "phase": phase, "domain": domain}
                for metric in ("accuracy_pct", "fixed_vs_sp_pct", "broken_vs_sp_pct", "samples"):
                    for suffix, value in aggregate([lookup[s, arm, phase, domain][metric] for s in seeds]).items():
                        row[f"{metric}_{suffix}"] = value
                summaries.append(row)
    write_csv(output / "summary.csv", summaries)
    paired = []
    for seed in seeds:
        for other in ARMS[:3]:
            for phase in ("all", "before_feedback", "after_feedback"):
                for domain in ("ALL", "Art", "Cartoon", "Sketch"):
                    a = lookup[seed, ARMS[3], phase, domain]["accuracy_pct"]
                    b = lookup[seed, other, phase, domain]["accuracy_pct"]
                    paired.append({"seed": seed, "comparison": f"Feedback-Fusion − {other}",
                        "phase": phase, "domain": domain, "gain_pp": a - b if a is not None and b is not None else None})
    write_csv(output / "paired_gains.csv", paired)
    lines = ["# Frozen semantic feedback: PACS D=100", "",
        "Photo(source) → Art → Cartoon → Sketch; batch32. Mean ± sample SD across seeds.", "",
        "## Accuracy", "", "| Arm | Art (%) | Cartoon (%) | Sketch (%) | Macro (%) | Stream (%) | Before feedback (%) | After feedback (%) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        cells = [display([lookup[s, arm, "all", d]["accuracy_pct"] for s in seeds]) for d in ("Art", "Cartoon", "Sketch")]
        cells.append(display([statistics.mean(lookup[s, arm, "all", d]["accuracy_pct"] for d in ("Art", "Cartoon", "Sketch")) for s in seeds]))
        cells += [display([lookup[s, arm, p, "ALL"]["accuracy_pct"] for s in seeds]) for p in ("all", "before_feedback", "after_feedback")]
        lines.append("| " + " | ".join([arm] + cells) + " |")
    lines += ["", "## Complementarity on all online images (evaluation only)", "",
        "| Phase | Domain | CLIP fixes SP (%) | CLIP breaks SP (%) | Net (pp) |",
        "| :--- | :--- | ---: | ---: | ---: |"]
    for phase in ("all", "before_feedback", "after_feedback"):
        for domain in ("ALL", "Art", "Cartoon", "Sketch"):
            rows = [lookup[s, ARMS[1], phase, domain] for s in seeds]
            fixed = [r["fixed_vs_sp_pct"] for r in rows]
            broken = [r["broken_vs_sp_pct"] for r in rows]
            net = [a - b if a is not None and b is not None else None for a, b in zip(fixed, broken)]
            lines.append(f"| {phase} | {domain} | {display(fixed)} | {display(broken)} | {display(net)} |")
    lines += ["", "## Paired stream gains", "", "| Comparison | Gain (pp) |", "| :--- | ---: |"]
    for other in ARMS[:3]:
        name = f"Feedback-Fusion − {other}"
        lines.append(f"| {name} | {display([r['gain_pp'] for r in paired if r['comparison'] == name and r['phase'] == 'all' and r['domain'] == 'ALL'])} |")
    metrics = ("queried", "returned", "returned_pct", "cross_domain_returned", "cross_domain_pct", "pending",
               "main_updates", "candidate_updates", "fusion_updates", "clip_image_count", "shared_stream_seconds")
    costs = []
    lines += ["", "## Shared feedback and costs", "", "| Metric | Mean ± SD |", "| :--- | ---: |"]
    for metric in metrics:
        values = [r[metric] for r in results]
        costs.append({"metric": metric, **aggregate(values)})
        lines.append(f"| {metric} | {display(values)} |")
    write_csv(output / "cost_summary.csv", costs)
    lines += ["", "Four prediction arms share one unchanged SP state, query schedule, full supervised budget and stream. "
        "Only Feedback-Fusion uses returned labels to change seven mixture coefficients. "
        "CLIP is an additional pretrained model: comparison with SP is not pretraining- or FLOP-matched. "
        "Shared runtime includes all prediction arms and original SP shadow probes; it is not per-arm latency. "
        "CLIP loading/text initialization time is excluded from stream time. "
        "Predictions precede release/update; the first return batch belongs to before-feedback. "
        "Before-feedback fixed and learned fusion must coincide. "
        "Calibration is learned from selectively queried, historical images; it is not unbiased current-domain calibration. "
        "No true domain boundary enters the algorithm. Cross percentage excludes pending labels and is averaged per seed. "
        "Per-seed paired effects, not probe/image counts, are the independent replicates. "
        "No accuracy or significance claim is automatic."]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
