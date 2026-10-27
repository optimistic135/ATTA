"""Fixed offline usability summaries, never feed scores/labels back into decisions."""
import math
import statistics

from experiments.delayed_feedback.run import write_csv
from .spec import DEFAULTS as D, DOMAINS, SECTIONS


def mean(values):
    return statistics.mean(values) if values else None


def pct(n, d):
    return 100 * n / d if d else None


def quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def tail_accuracy(rows, score_key, correctness_key, high):
    if not rows:
        return None
    k = max(1, math.ceil(D["rank_fraction"] * len(rows)))
    scores = sorted((r[score_key] for r in rows), reverse=high)
    boundary = scores[k - 1]
    strict = [r for r in rows if r[score_key] > boundary] if high else [r for r in rows if r[score_key] < boundary]
    tied = [r for r in rows if r[score_key] == boundary]
    weight = (k - len(strict)) / len(tied)
    correct = sum(r[correctness_key] for r in strict) + weight * sum(r[correctness_key] for r in tied)
    return 100 * correct / k


def seed_reports(seed, algorithm, result):
    sections = {key: [] for key in SECTIONS}
    obs, funnels, queries = algorithm.observation_rows, algorithm.funnel_rows, algorithm.query_rows
    for domain in DOMAINS:
        observations = [r for r in obs if domain == "ALL" or r["image_domain"] == domain]
        for branch in ("source", "repair"):
            score, confidence, correct = f"{branch}_score", f"{branch}_confidence", f"{branch}_correct"
            warm = [r for r in observations if r["warm"] and r["allowed"]]
            base = [r for r in warm if r[confidence] >= D["confidence"] and r[f"{branch}_consistent"]]
            for pool_name, rows in (("warm_all", warm), ("warm_base", base)):
                sections["distribution"].append(distribution(seed, domain, branch, pool_name,
                                                            rows, score, D["reliability_threshold"]))
                if branch == "repair":
                    sections["distribution"].append(distribution(seed, domain, "repair_margin", pool_name,
                                                                rows, "repair_margin", D["correction_margin"]))
            high = tail_accuracy(base, score, correct, True)
            low = tail_accuracy(base, score, correct, False)
            conf_high = tail_accuracy(base, confidence, correct, True)
            sections["ranking"].append({"seed": seed, "domain": domain, "branch": branch,
                "samples": len(base), "tail_weight": max(1, math.ceil(D["rank_fraction"] * len(base))) if base else 0,
                "base_accuracy_pct": pct(sum(r[correct] for r in base), len(base)),
                "risk_high_accuracy_pct": high, "risk_low_accuracy_pct": low,
                "confidence_high_accuracy_pct": conf_high,
                "confidence_low_accuracy_pct": tail_accuracy(base, confidence, correct, False),
                "risk_high_minus_low_pp": high - low if high is not None else None,
                "risk_minus_confidence_high_pp": high - conf_high if high is not None else None})
            edges = D["score_bin_edges"]
            for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
                selected = [r for r in base if lo <= r[score] and (r[score] < hi or i == len(edges) - 2 and r[score] <= hi)]
                sections["score_bins"].append({"seed": seed, "domain": domain, "branch": branch,
                    "bin": f"{lo:.1f}–{hi:.1f}", "samples": len(selected),
                    "mean_score": mean([r[score] for r in selected]),
                    "accuracy_pct": pct(sum(r[correct] for r in selected), len(selected)),
                    "risk_brier": mean([(r[score] - r[correct]) ** 2 for r in selected])})
            query_rows = [r for r in queries if r["branch"] == branch and (domain == "ALL" or r["query_domain"] == domain)]
            for phase in ("all", "before_learning", "after_learning", "after_warmup"):
                selected = [r for r in query_rows if phase == "all" or r["phase"] == phase
                            or phase == "after_warmup" and r["warm_at_query"]]
                sections["query_brier"].append({"seed": seed, "domain": domain, "branch": branch, "phase": phase,
                    "samples": len(selected), "accuracy_pct": pct(sum(r["correct"] for r in selected), len(selected)),
                    "risk_brier": mean([(r["score"] - r["correct"]) ** 2 for r in selected]),
                    "confidence_brier": mean([(r["confidence"] - r["correct"]) ** 2 for r in selected]),
                    "base_rate_brier": mean([(r["base_rate"] - r["correct"]) ** 2 for r in selected]),
                    "mean_score": mean([r["score"] for r in selected]),
                    "mean_updates_at_query": mean([r["risk_updates_at_query"] for r in selected])})
        domain_funnel = [r for r in funnels if r["domain"] == domain]
        for branch in ("source", "repair"):
            stages = ("window", "allowed", "warmup", "confidence", "consistency", "risk")
            if branch == "repair":
                stages += ("margin", "disagreement")
            stages += ("sp_gate",)
            for phase in ("all", "after_warmup"):
                eligible = [r for r in domain_funnel if r["branch"] == branch and (phase == "all" or r["warm"])]
                total = sum(r["passed"] for r in eligible if r["stage"] == "window")
                for stage in stages:
                    rows = [r for r in eligible if r["stage"] == stage]
                    before, passed = sum(r["before"] for r in rows), sum(r["passed"] for r in rows)
                    sections["funnel"].append({"seed": seed, "domain": domain, "branch": branch,
                        "phase": phase, "stage": stage, "before": before, "passed": passed,
                        "dropped": before - passed, "conditional_pass_pct": pct(passed, before),
                        "window_pass_pct": pct(passed, total)})
    policies = algorithm.policy_rows
    sections["warmup"].append({"seed": seed,
        "batches": len(policies), "warm_batches": sum(r["warm"] for r in policies),
        "first_learning_step": algorithm.first_learning_step, "first_warm_step": algorithm.first_warm_step,
        "queried": result["queried"], "returned": result["returned"], "pending": result["pending"],
        "select_observations": sum(r["select_count"] for r in policies),
        "correct_observations": sum(r["correct_total_count"] for r in policies),
        "repair_observations": sum(r["correct_repair_count"] for r in policies),
        "select_sp_allowed_observations": sum(r["select_count"] for r in policies if r["sp_accept"]),
        "correct_sp_allowed_observations": sum(r["correct_total_count"] for r in policies if r["sp_accept"])})
    return sections


def distribution(seed, domain, branch, pool, rows, key, threshold):
    values = [r[key] for r in rows]
    result = {"seed": seed, "domain": domain, "branch": branch, "pool": pool,
        "samples": len(values), "score_mean": mean(values),
        "score_std": statistics.pstdev(values) if values else None,
        "minimum": min(values) if values else None, "maximum": max(values) if values else None,
        "threshold_pass_pct": pct(sum(v >= threshold for v in values), len(values))}
    for name, q in (("p05", .05), ("p25", .25), ("median", .5), ("p75", .75), ("p95", .95), ("p99", .99)):
        result[name] = quantile(values, q)
    return result


KEYS = {"distribution": ("domain", "branch", "pool"), "funnel": ("domain", "branch", "phase", "stage"),
        "ranking": ("domain", "branch"), "query_brier": ("domain", "branch", "phase"),
        "score_bins": ("domain", "branch", "bin"), "warmup": ()}


def aggregate(rows, keys):
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in keys), []).append(row)
    summaries = []
    for group, items in grouped.items():
        summary = dict(zip(keys, group))
        for metric in items[0]:
            if metric in keys or metric == "seed":
                continue
            values = [r[metric] for r in items if r[metric] is not None]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
            summary[f"{metric}_n"] = len(values)
        summaries.append(summary)
    return summaries


def cell(row, key, digits=2):
    value, sd = row[f"{key}_mean"], row[f"{key}_std"]
    if value is None:
        return "N/A"
    return f"{value:.{digits}f} ± {sd:.{digits}f}" if sd is not None else f"{value:.{digits}f}"


def summarize(output, results, diagnostics, seeds):
    if len(results) != 3 or {r["seed"] for r in results} != set(seeds):
        raise ValueError("Require three completed distinct seeds")
    summaries = {}
    for section in SECTIONS:
        rows = [r for seed in seeds for r in diagnostics[seed][section]]
        write_csv(output / f"{section}_per_seed.csv", rows)
        summaries[section] = aggregate(rows, KEYS[section])
        write_csv(output / f"{section}_summary.csv", summaries[section])
    metrics = ("Art_accuracy_pct", "Cartoon_accuracy_pct", "Sketch_accuracy_pct", "stream_accuracy_pct",
        "domain_macro_accuracy_pct", "queried", "returned", "cross_domain_pct", "pending", "main_updates",
        "shadow_head_updates", "shadow_reliability_updates", "stream_seconds")
    factual = aggregate([{k: r[k] for k in ("seed",) + metrics} for r in results], ())
    write_csv(output / "per_seed.csv", results)
    write_csv(output / "summary.csv", factual)
    text = "# Screening usability diagnosis: PACS D=100\n\n## Factual SP trajectory\n\n| Metric | Mean ± SD |\n| :--- | ---: |\n"
    text += "\n".join(f"| {k} | {cell(factual[0], k)} |" for k in metrics)
    text += "\n\n## Warmup and hypothetical policy counts\n\n| Metric | Mean ± SD |\n| :--- | ---: |\n"
    warm = summaries["warmup"][0]
    text += "\n".join(f"| {k[:-5]} | {cell(warm, k[:-5])} |" for k in warm if k.endswith("_mean"))
    text += "\n\n## Warmup score distribution (ALL)\n\n| Branch | Pool | Samples | Mean | Median | P95 | Max | Threshold pass (%) |\n| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    text += "\n".join(f"| {r['branch']} | {r['pool']} | {cell(r, 'samples')} | {cell(r, 'score_mean', 4)} | {cell(r, 'median', 4)} | {cell(r, 'p95', 4)} | {cell(r, 'maximum', 4)} | {cell(r, 'threshold_pass_pct')} |" for r in summaries["distribution"] if r["domain"] == "ALL")
    text += "\n\n## Filter attrition after warmup (ALL)\n\n| Branch | Stage | Before | Passed | Dropped | Conditional pass (%) |\n| :--- | :--- | ---: | ---: | ---: | ---: |\n"
    text += "\n".join(f"| {r['branch']} | {r['stage']} | {cell(r, 'before')} | {cell(r, 'passed')} | {cell(r, 'dropped')} | {cell(r, 'conditional_pass_pct')} |" for r in summaries["funnel"] if r["domain"] == "ALL" and r["phase"] == "after_warmup")
    text += "\n\n## Fixed 10% ranking on warm base-eligible images\n\n| Domain | Branch | Samples | Risk high acc (%) | Risk low acc (%) | Confidence high acc (%) | Risk−confidence high (pp) |\n| :--- | :--- | ---: | ---: | ---: | ---: | ---: |\n"
    text += "\n".join(f"| {r['domain']} | {r['branch']} | {cell(r, 'samples')} | {cell(r, 'risk_high_accuracy_pct')} | {cell(r, 'risk_low_accuracy_pct')} | {cell(r, 'confidence_high_accuracy_pct')} | {cell(r, 'risk_minus_confidence_high_pp')} |" for r in summaries["ranking"])
    text += "\n\n## Query-time Brier by learning phase (ALL)\n\n| Branch | Query phase | Returned samples | Risk Brier | Confidence Brier | Online base-rate Brier |\n| :--- | :--- | ---: | ---: | ---: | ---: |\n"
    text += "\n".join(f"| {r['branch']} | {r['phase']} | {cell(r, 'samples')} | {cell(r, 'risk_brier', 4)} | {cell(r, 'confidence_brier', 4)} | {cell(r, 'base_rate_brier', 4)} |" for r in summaries["query_brier"] if r["domain"] == "ALL")
    text += (f"\n\nPACS P(source) → A → C → S; batch32; D100; seeds={seeds}. "
        "SP retains full formal quota; module training is shadow overhead and does not supply current targets to SP. "
        "All counts describe window-image observations unless explicitly query counts. Repeated images are not independent replicates. "
        "Quantiles computed per seed, then mean ± sample SD across seeds; CSVs retain valid n. "
        "Source/repair pools are separate; top/bottom10% is offline only, with fractional boundary ties and identical coverage for risk/confidence. "
        "After_warmup query rows are a subset of after_learning; phases use module state at query, not return. "
        "Query Brier excludes pending labels and reflects active-query selection and stream-tail censoring. "
        "Test truth scores observations only, never trains modules or selects thresholds. No effectiveness or significance claim is automatic.\n")
    (output / "summary.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
