"""Seed-weighted local intervention and realized elite precision reports."""
import statistics

from experiments.delayed_feedback.run import write_csv
from .spec import ARMS, PAIRS


def pct(n, d):
    return 100 * n / d if d else None


def future_metrics(rows):
    n = sum(r["samples"] for r in rows)
    result = {"samples": n, "events": len({r["probe_id"] for r in rows})}
    for arm in ARMS:
        result[f"{arm.lower()}_accuracy_pct"] = pct(sum(r[f"{arm.lower()}_correct"] for r in rows), n)
    for name, _, _ in PAIRS:
        fixed, broken = sum(r[f"{name}_fixed"] for r in rows), sum(r[f"{name}_broken"] for r in rows)
        result[f"{name}_fixed"], result[f"{name}_broken"] = fixed, broken
        result[f"{name}_gain_pp"] = pct(fixed - broken, n)
    return result


def target_metrics(rows, batches):
    n = len(rows)
    return {"targets": n, "batches": len(batches), "base_candidates": sum(r["base_candidates"] for r in batches),
        "empty_batch_pct": pct(sum(r["selected"] == 0 for r in batches), len(batches)),
        "selected_of_arrivals_pct": pct(n, sum(r["samples"] for r in batches)),
        "source_accuracy_pct": pct(sum(r["source_correct"] for r in rows), n),
        "model_accuracy_pct": pct(sum(r["model_correct"] for r in rows), n),
        "opportunity_pct": pct(sum(r["correction_opportunity"] for r in rows), n),
        "harm_risk_pct": pct(sum(r["harm_risk"] for r in rows), n),
        "mean_need": statistics.mean(r["need_source_target"] for r in rows) if n else None}


def seed_reports(seed, algorithm):
    future, targets, events = [], [], []
    complete = [r for r in algorithm.future_rows if r["complete_window"]]
    for domain in ("ALL", "Cartoon", "Sketch"):
        for subset in ("all", "active", "sp_accepted"):
            rows = [r for r in complete if (domain == "ALL" or r["domain"] == domain)
                    and (subset == "all" or subset == "active" and r["active"]
                         or subset == "sp_accepted" and r["sp_accept"])]
            future.append({"seed": seed, "domain": domain, "subset": subset, **future_metrics(rows)})
    probe_steps = {r["step"] for r in algorithm.probe_rows}
    first = algorithm.first_return
    for domain in ("ALL", "Art", "Cartoon", "Sketch"):
        for phase in ("all", "before_feedback", "after_feedback", "at_probe"):
            def include(r):
                return ((domain == "ALL" or r["domain"] == domain)
                    and (phase == "all" or phase == "before_feedback" and (first is None or r["step"] <= first)
                         or phase == "after_feedback" and first is not None and r["step"] > first
                         or phase == "at_probe" and r["step"] in probe_steps))
            rows = [r for r in algorithm.target_rows if include(r)]
            batches = [r for r in algorithm.selection_rows if include(r)]
            targets.append({"seed": seed, "domain": domain, "phase": phase, **target_metrics(rows, batches)})
    for probe in algorithm.probe_rows:
        row = {"seed": seed, **probe}
        f = [r for r in complete if r["probe_id"] == probe["probe_id"]]
        t = [r for r in algorithm.target_rows if r["probe_id"] == probe["probe_id"]]
        b = [r for r in algorithm.selection_rows if r["step"] == probe["step"]]
        row.update({f"future_{key}": value for key, value in future_metrics(f).items()})
        row.update({f"target_{key}": value for key, value in target_metrics(t, b).items()})
        events.append(row)
    return {"future": future, "targets": targets, "events": events}


def aggregate(rows, keys):
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in keys), []).append(row)
    results = []
    for group, items in grouped.items():
        result = dict(zip(keys, group))
        for metric in items[0]:
            if metric in keys or metric == "seed":
                continue
            values = [r[metric] for r in items if r[metric] is not None]
            result[f"{metric}_mean"] = statistics.mean(values) if values else None
            result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
            result[f"{metric}_n"] = len(values)
        results.append(result)
    return results


def cell(row, key):
    value, sd = row[f"{key}_mean"], row[f"{key}_std"]
    return "N/A" if value is None else f"{value:.2f} ± {sd:.2f}" if sd is not None else f"{value:.2f}"


def summarize(output, results, diagnostics, seeds):
    if len(results) != 3 or {r["seed"] for r in results} != set(seeds):
        raise ValueError("Require three completed seeds")
    future_rows = [r for seed in seeds for r in diagnostics[seed]["future"]]
    target_rows = [r for seed in seeds for r in diagnostics[seed]["targets"]]
    future = aggregate(future_rows, ("domain", "subset"))
    targets = aggregate(target_rows, ("domain", "phase"))
    metrics = ("Art_accuracy_pct", "Cartoon_accuracy_pct", "Sketch_accuracy_pct", "stream_accuracy_pct",
        "domain_macro_accuracy_pct", "queried", "returned", "cross_domain_returned", "cross_domain_pct", "pending",
        "main_updates", "candidate_updates", "diagnostic_candidate_updates", "complete_elite_probes", "stream_seconds")
    factual = aggregate([{k: r[k] for k in ("seed",) + metrics} for r in results], ())
    for name, rows in (("per_seed", results), ("summary", factual), ("future_per_seed", future_rows),
                       ("future_summary", future), ("targets_per_seed", target_rows), ("targets_summary", targets)):
        write_csv(output / f"{name}.csv", rows)
    text = "# Elite source-target training diagnosis: PACS D=100\n\n## Factual SP trajectory\n\n| Metric | Mean ± SD |\n| :--- | ---: |\n"
    text += "\n".join(f"| {key} | {cell(factual[0], key)} |" for key in metrics)
    text += "\n\n## Frozen one-step future effects\n\n| Future domain | Subset | Events | CURRENT−HISTORY (pp) | CURRENT−REDUCED (pp) | CURRENT−TRUE (pp) | REDUCED−HISTORY (pp) |\n| :--- | :--- | ---: | ---: | ---: | ---: | ---: |\n"
    text += "\n".join(f"| {r['domain']} | {r['subset']} | {cell(r, 'events')} | {cell(r, 'current_history_gain_pp')} | {cell(r, 'current_reduced_gain_pp')} | {cell(r, 'current_true_gain_pp')} | {cell(r, 'reduced_history_gain_pp')} |" for r in future)
    text += "\n\n## Realized elite targets and online-model opportunity\n\n| Domain | Phase | Targets | Empty batches (%) | Source acc (%) | Model acc (%) | Opportunity (%) | Harm risk (%) | Mean need |\n| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    text += "\n".join(f"| {r['domain']} | {r['phase']} | {cell(r, 'targets')} | {cell(r, 'empty_batch_pct')} | {cell(r, 'source_accuracy_pct')} | {cell(r, 'model_accuracy_pct')} | {cell(r, 'opportunity_pct')} | {cell(r, 'harm_risk_pct')} | {cell(r, 'mean_need')} |" for r in targets)
    text += (f"\n\nPACS P(source) → A → C → S; batch32; D100; seeds={seeds}. "
        "Sample-weighted within each seed, then mean ± sample SD; valid n in CSV. "
        "SP uses full original formal quota; all candidates are shadow overhead. No learned head/risk module. "
        "All four arms share pre-update snapshot and truth/history draws. CURRENT uses every selected elite image, "
        "mean over selected K, without a second random subsample. CURRENT and REDUCED share reduced history coefficient. "
        "When K=0 CURRENT falls back to HISTORY; only active isolates current contribution. "
        "Shadow auxiliary coefficients are hypothetical and do not follow the main gate; sp_accepted is descriptive. "
        "Future scoring excludes trigger batch and requires complete ten-batch windows. "
        "Before_feedback includes the first return batch because predictions precede its update. "
        "Elite ties use seeded uniform random order, not offline fractional expectations; realized precision may differ. "
        "Current KL normalizes over selected K, unlike previous masked-batch losses; no claim that selection alone "
        "explains comparisons to earlier experiments. Opportunity/harm counts are diagnostic, not actual fixed/broken. "
        "No oracle labels train models; target precision does not establish whole-stream gains or significance.\n")
    (output / "summary.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
