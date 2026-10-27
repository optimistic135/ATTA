"""Seed-level, domain-separated diagnosis. Windows/events are not seed replicates."""
import statistics
from experiments.delayed_feedback.run import write_csv
from .spec import ARMS, PAIRS


def pct(numerator, denominator):
    return 100 * numerator / denominator if denominator else None


def future_metrics(rows):
    samples = sum(r["samples"] for r in rows)
    result = {"events": len({r["probe_id"] for r in rows}), "samples": samples}
    for arm in ARMS:
        key = arm.lower()
        result[f"{key}_accuracy_pct"] = pct(sum(r[f"{key}_correct"] for r in rows), samples)
    for pair, _, _ in PAIRS:
        fixed, broken = sum(r[f"{pair}_fixed"] for r in rows), sum(r[f"{pair}_broken"] for r in rows)
        result[f"{pair}_fixed"] = fixed
        result[f"{pair}_broken"] = broken
        result[f"{pair}_gain_pp"] = pct(fixed - broken, samples)
    return result


def quality_metrics(rows):
    count = len(rows)
    source = sum(r["source_correct"] for r in rows)
    repair = sum(r["repair_correct"] for r in rows)
    fixed = sum(not r["source_correct"] and r["repair_correct"] for r in rows)
    broken = sum(r["source_correct"] and not r["repair_correct"] for r in rows)
    return {"observations": count, "events": len({r["probe_id"] for r in rows}),
        "source_accuracy_pct": pct(source, count), "repair_accuracy_pct": pct(repair, count),
        "repair_source_gain_pp": pct(repair - source, count), "fixed": fixed, "broken": broken,
        "repair_source_ce_gain": sum(r["source_ce"] - r["repair_ce"] for r in rows) / count if count else None}


def event_diagnostics(probes, future_rows, target_rows):
    """One record per snapshot: join target quality, gate evidence and future effect."""
    results = []
    for probe in probes:
        row = dict(probe)
        future = [r for r in future_rows if r["probe_id"] == probe["probe_id"] and r["complete_window"]]
        row.update({f"future_{key}": value for key, value in future_metrics(future).items()})
        targets = [r for r in target_rows if r["probe_id"] == probe["probe_id"] and r["allowed"]]
        for name in ("all", "source_selected", "repair_selected", "common_selected"):
            selected = [r for r in targets if name == "all"
                or name == "source_selected" and r["source_selected"]
                or name == "repair_selected" and r["repair_selected"]
                or name == "common_selected" and r["source_selected"] and r["repair_selected"]]
            row.update({f"target_{name}_{key}": value for key, value in quality_metrics(selected).items()})
        results.append(row)
    return results


def seed_diagnostics(seed, future_rows, target_rows):
    futures, quality = [], []
    complete = [r for r in future_rows if r["complete_window"]]
    for domain in ("ALL", "Cartoon", "Sketch"):
        for subset in ("all", "source_accepted", "repair_accepted", "both_accepted", "sp_accepted"):
            rows = [r for r in complete if (domain == "ALL" or r["domain"] == domain)
                and (subset == "all" or subset == "source_accepted" and r["source_accept"]
                     or subset == "repair_accepted" and r["repair_accept"]
                     or subset == "both_accepted" and r["source_accept"] and r["repair_accept"]
                     or subset == "sp_accepted" and r["sp_would_accept"])]
            futures.append({"seed": seed, "domain": domain, "subset": subset, **future_metrics(rows)})
        for subset in ("all_allowed", "source_selected", "repair_selected", "common_selected", "disagreement", "sampled"):
            rows = [r for r in target_rows if r["allowed"] and (domain == "ALL" or r["image_domain"] == domain)
                and (subset == "all_allowed" or subset == "source_selected" and r["source_selected"]
                     or subset == "repair_selected" and r["repair_selected"]
                     or subset == "common_selected" and r["source_selected"] and r["repair_selected"]
                     or subset == "disagreement" and r["source_prediction"] != r["repair_prediction"]
                     or subset == "sampled" and r["sampled"])]
            quality.append({"seed": seed, "domain": domain, "subset": subset, **quality_metrics(rows)})
    return futures, quality


def aggregate(rows, keys):
    grouped = {}
    for row in rows:
        group = tuple(row[k] for k in keys)
        grouped.setdefault(group, []).append(row)
    summaries = []
    for group, items in grouped.items():
        result = dict(zip(keys, group))
        for metric in items[0]:
            if metric in keys or metric == "seed":
                continue
            values = [r[metric] for r in items if r[metric] is not None]
            result[f"{metric}_mean"] = statistics.mean(values) if values else None
            result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
            result[f"{metric}_n"] = len(values)
        summaries.append(result)
    return summaries


def cell(row, metric):
    mean, std = row[f"{metric}_mean"], row[f"{metric}_std"]
    return "N/A" if mean is None else f"{mean:.2f} ± {std:.2f}" if std is not None else f"{mean:.2f}"


def summarize(output, results, futures, quality, seeds):
    if len(results) != 3 or {r["seed"] for r in results} != set(seeds):
        raise ValueError("Require three distinct completed seeds")
    write_csv(output / "per_seed.csv", results)
    write_csv(output / "future_per_seed.csv", futures)
    write_csv(output / "quality_per_seed.csv", quality)
    future_summary, quality_summary = aggregate(futures, ("domain", "subset")), aggregate(quality, ("domain", "subset"))
    write_csv(output / "future_summary.csv", future_summary)
    write_csv(output / "quality_summary.csv", quality_summary)
    baseline_keys = ("Art_accuracy_pct", "Cartoon_accuracy_pct", "Sketch_accuracy_pct", "stream_accuracy_pct",
                     "domain_macro_accuracy_pct", "queried", "returned", "cross_domain_returned", "cross_domain_pct",
                     "pending", "main_updates", "candidate_updates", "diagnostic_candidate_updates",
                     "diagnostic_head_updates", "oracle_candidate_updates", "complete_target_probes", "stream_seconds")
    baseline = aggregate([{k: r[k] for k in ("seed",) + baseline_keys} for r in results], ())
    write_csv(output / "summary.csv", baseline)
    text = "# Same-snapshot oracle-target diagnosis: PACS D=100\n\n"
    text += "## Factual SP-ATTA trajectory\n\n| Metric | Mean ± SD |\n| :--- | ---: |\n"
    text += "\n".join(f"| {key} | {cell(baseline[0], key)} |" for key in baseline_keys)
    text += "\n\n## Frozen one-step candidates: next ten batches\n\n"
    text += "| Future domain | Event subset | Events | SOURCE−TRUE (pp) | REPAIR−TRUE (pp) | REPAIR−SOURCE (pp) |\n| :--- | :--- | ---: | ---: | ---: | ---: |\n"
    text += "\n".join(f"| {r['domain']} | {r['subset']} | {cell(r, 'events')} | {cell(r, 'source_true_gain_pp')} | {cell(r, 'repair_true_gain_pp')} | {cell(r, 'repair_source_gain_pp')} |"
                      for r in future_summary)
    text += "\n\n## Oracle contrasts (diagnostic copies only)\n\n"
    text += "| Future domain | Event subset | ORACLE-SOURCE−SOURCE (pp) | ORACLE-REPAIR−REPAIR (pp) | ORACLE-SOURCE−TRUE (pp) | ORACLE-REPAIR−TRUE (pp) |\n| :--- | :--- | ---: | ---: | ---: | ---: |\n"
    text += "\n".join(f"| {r['domain']} | {r['subset']} | {cell(r, 'oracle_source_source_gain_pp')} | {cell(r, 'oracle_repair_repair_gain_pp')} | {cell(r, 'oracle_source_true_gain_pp')} | {cell(r, 'oracle_repair_true_gain_pp')} |"
                      for r in future_summary)
    text += "\n\n## Target quality on current images (evaluation only)\n\n"
    text += "| Image domain | Image subset | Observations | Source acc (%) | Repair acc (%) | Repair−source (pp) |\n| :--- | :--- | ---: | ---: | ---: | ---: |\n"
    text += "\n".join(f"| {r['domain']} | {r['subset']} | {cell(r, 'observations')} | {cell(r, 'source_accuracy_pct')} | {cell(r, 'repair_accuracy_pct')} | {cell(r, 'repair_source_gain_pp')} |"
                      for r in quality_summary)
    text += (f"\n\nPhoto(source) → Art → Cartoon → Sketch; batch32; seeds={seeds}. "
        "Sample-weighted within each seed, then mean ± sample SD across seeds (valid n in CSV). "
        "All five arms share model snapshot and truth/current draws. SOURCE vs ORACLE_SOURCE share the exact mask; "
        "REPAIR vs ORACLE_REPAIR share their exact mask. Oracle targets swap argmax and true-class probabilities, "
        "preserving each target's probability multiset, confidence and entropy. SOURCE and REPAIR masks may differ. "
        "SOURCE/REPAIR accepted subsets use their own fresh-feedback one-step gates; sp_accepted uses the original historical-pool SP gate. "
        "Comparisons across different accepted subsets are selected on different events; use all or both_accepted for direct target comparison. "
        "Only complete ten-batch windows enter future summaries. Future domain is the scored batch domain, not trigger domain. "
        "Current target quality includes all triggered probes; image subsets are window-image observations and can repeat images. "
        "Evaluation truth trains only explicitly nondeployable oracle copies; it never enters factual SP, "
        "ordinary candidate training, gates, head repair or query decisions. Head repair and all candidates are diagnostic overhead; "
        "the unchanged SP trajectory uses its full original formal quota. Equal steps do not imply equal FLOPs. "
        "Oracle is not an achievable method or guaranteed upper bound. Negative oracle gains do not rule out other "
        "loss strengths or update schedules. Local effects do not establish whole-stream gains or significance.\n")
    (output / "summary.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
