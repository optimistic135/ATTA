"""Standard-library-only, label-scored offline comparisons; no model fitting."""
import math
import statistics
from collections import defaultdict

DOMAINS = ("ALL", "Art", "Cartoon", "Sketch")
PERIODS = ("all", "early", "middle", "late")
FRACTION = 0.1


def ratio(n, d, scale=100):
    return scale * n / d if d else None


def boolean(value):
    if value in (True, "True", "true", "1", 1):
        return True
    if value in (False, "False", "false", "0", 0):
        return False
    raise ValueError(f"Invalid boolean: {value!r}")


def tail(rows, score_key, correct_key, high):
    """Expected correctness under a score-only rule with uniform boundary ties."""
    if not rows:
        return {"weight": 0, "correct": 0.0, "boundary": None, "tied": 0}
    k = max(1, math.ceil(FRACTION * len(rows)))
    boundary = sorted((r[score_key] for r in rows), reverse=high)[k - 1]
    strict = [r for r in rows if r[score_key] > boundary] if high else [r for r in rows if r[score_key] < boundary]
    tied = [r for r in rows if r[score_key] == boundary]
    weight = (k - len(strict)) / len(tied)
    correct = sum(r[correct_key] for r in strict) + weight * sum(r[correct_key] for r in tied)
    if not 0 <= correct <= k + 1e-9:
        raise AssertionError("Invalid tied-selection weight")
    return {"weight": k, "correct": correct, "boundary": boundary, "tied": len(tied)}


def batch_rankings(seed, observations, batches, confidence):
    by_step = defaultdict(list)
    for row in observations:
        by_step[row["step"]].append(row)
    warm_steps = defaultdict(list)
    for step, batch in enumerate(batches):
        rows = by_step[step]
        if len({r["warm"] for r in rows}) != 1:
            raise ValueError(f"Inconsistent warmup at step {step}")
        if rows[0]["warm"]:
            warm_steps[batch["domain"]].append(step)
    periods = {}
    for steps in warm_steps.values():
        for index, step in enumerate(steps):
            periods[step] = PERIODS[1 + min(2, 3 * index // len(steps))]
    records = []
    for step, period in periods.items():
        batch = batches[step]
        for pool in ("current_batch", "rolling_window"):
            arrivals = [r for r in by_step[step] if batch["lo"] <= r["sample_id"] < batch["hi"]]
            candidates = arrivals if pool == "current_batch" else by_step[step]
            for branch in ("source", "repair"):
                base = [r for r in candidates if r["allowed"] and r[f"{branch}_confidence"] >= confidence
                        and r[f"{branch}_consistent"]]
                ckey = f"{branch}_correct"
                c_hi = tail(base, f"{branch}_confidence", ckey, True)
                c_lo = tail(base, f"{branch}_confidence", ckey, False)
                r_hi = tail(base, f"{branch}_score", ckey, True)
                r_lo = tail(base, f"{branch}_score", ckey, False)
                k, n = c_hi["weight"], len(base)
                base_correct = sum(r[ckey] for r in base)
                records.append({"seed": seed, "step": step, "domain": batch["domain"], "period": period,
                    "pool": pool, "branch": branch, "available_images": len(candidates), "base_candidates": n,
                    "selected_weight": k, "actual_coverage_pct": ratio(k, n),
                    "selected_of_available_pct": ratio(k, len(candidates)),
                    "old_domain_candidates": sum(r["image_domain"] != batch["domain"] for r in base),
                    "confidence_high_correct": c_hi["correct"], "confidence_low_correct": c_lo["correct"],
                    "risk_high_correct": r_hi["correct"], "risk_low_correct": r_lo["correct"],
                    "random_expected_correct": k * base_correct / n if n else 0.0,
                    "confidence_boundary": c_hi["boundary"], "risk_boundary": r_hi["boundary"],
                    "confidence_boundary_ties": c_hi["tied"], "risk_boundary_ties": r_hi["tied"]})
    summaries = []
    for pool in ("current_batch", "rolling_window"):
        for domain in DOMAINS:
            for period in PERIODS:
                for branch in ("source", "repair"):
                    rows = [r for r in records if r["pool"] == pool and r["branch"] == branch
                            and (domain == "ALL" or r["domain"] == domain)
                            and (period == "all" or r["period"] == period)]
                    n_batches = len(rows)
                    k = sum(r["selected_weight"] for r in rows)
                    base = sum(r["base_candidates"] for r in rows)
                    row = {"seed": seed, "pool": pool, "domain": domain, "period": period, "branch": branch,
                        "warm_batches": n_batches, "zero_candidate_batches": sum(r["base_candidates"] == 0 for r in rows),
                        "zero_candidate_pct": ratio(sum(r["base_candidates"] == 0 for r in rows), n_batches),
                        "base_candidates": base, "selected_weight": k,
                        "mean_selected_per_warm_batch": k / n_batches if n_batches else None,
                        "actual_coverage_pct": ratio(k, base),
                        "selected_of_available_pct": ratio(k, sum(r["available_images"] for r in rows))}
                    for name in ("confidence_high", "confidence_low", "risk_high", "risk_low", "random_expected"):
                        row[f"{name}_accuracy_pct"] = ratio(sum(r[f"{name}_correct"] for r in rows), k)
                    for label, left, right in (("risk_minus_confidence_pp", "risk_high", "confidence_high"),
                                               ("confidence_minus_random_pp", "confidence_high", "random_expected"),
                                               ("risk_minus_random_pp", "risk_high", "random_expected"),
                                               ("risk_high_minus_low_pp", "risk_high", "risk_low")):
                        row[label] = (row[f"{left}_accuracy_pct"] - row[f"{right}_accuracy_pct"]) if k else None
                    summaries.append(row)
    return records, summaries, periods


def bias_and_correction(seed, arrivals, feedback, queries, periods, confidence):
    """Time-matched query bias plus paired correctness labels on the exact same images."""
    identities = {r["sample_id"]: r for r in feedback}
    returned_steps = {r["query_step"] for r in feedback if r["return_step"] is not None}
    query_map = {(r["anchor_index"], r["branch"]): r for r in queries}
    bias, correction = [], []
    audit = {"seed": seed, "returned_prediction_pairs": 0, "correctness_mismatches": 0,
             "max_score_abs_difference": 0.0, "max_confidence_abs_difference": 0.0}
    for row in arrivals:
        fb = identities.get(row["sample_id"])
        row["queried"] = fb is not None
        row["returned_query"] = fb is not None and fb["return_step"] is not None
        row["pending_query"] = fb is not None and fb["return_step"] is None
        if fb is not None and fb["query_step"] != row["step"]:
            raise ValueError("Query identity does not belong to its arrival batch")
        if row["returned_query"]:
            for branch in ("source", "repair"):
                q = query_map[(fb["anchor_index"], branch)]
                audit["returned_prediction_pairs"] += 1
                audit["correctness_mismatches"] += int(q["correct"] != row[f"{branch}_correct"])
                audit["max_score_abs_difference"] = max(audit["max_score_abs_difference"], abs(q["score"] - row[f"{branch}_score"]))
                audit["max_confidence_abs_difference"] = max(audit["max_confidence_abs_difference"], abs(q["confidence"] - row[f"{branch}_confidence"]))
    phases = ("all", "before_learning", "after_learning", "after_warmup")
    for support in ("full_stream", "returned_query_steps"):
        supported = [r for r in arrivals if support == "full_stream" or r["step"] in returned_steps]
        for domain in DOMAINS:
            for phase in phases:
                rows = [r for r in supported if (domain == "ALL" or r["image_domain"] == domain)
                    and (phase == "all" or phase == "before_learning" and r["risk_updates"] == 0
                         or phase == "after_learning" and r["risk_updates"] > 0
                         or phase == "after_warmup" and r["warm"])]
                for cohort in ("all_current", "all_queried", "returned_queried", "pending_queried", "unqueried"):
                    chosen = [r for r in rows if cohort == "all_current" or cohort == "all_queried" and r["queried"]
                        or cohort == "returned_queried" and r["returned_query"]
                        or cohort == "pending_queried" and r["pending_query"]
                        or cohort == "unqueried" and not r["queried"]]
                    for branch in ("source", "repair"):
                        bias.append({"seed": seed, "support": support, "domain": domain, "phase": phase,
                            "cohort": cohort, "branch": branch, "samples": len(chosen),
                            "accuracy_pct": ratio(sum(r[f"{branch}_correct"] for r in chosen), len(chosen)),
                            "risk_brier": statistics.mean((r[f"{branch}_score"] - r[f"{branch}_correct"]) ** 2 for r in chosen) if chosen else None,
                            "confidence_brier": statistics.mean((r[f"{branch}_confidence"] - r[f"{branch}_correct"]) ** 2 for r in chosen) if chosen else None,
                            "mean_score": statistics.mean(r[f"{branch}_score"] for r in chosen) if chosen else None})
    # Returned cases use canonical stored query-time correctness, not refitted predictions.
    returned = []
    for fb in feedback:
        if fb["return_step"] is None:
            continue
        s, r = query_map[(fb["anchor_index"], "source")], query_map[(fb["anchor_index"], "repair")]
        returned.append({"image_domain": fb["query_domain"], "warm": s["warm_at_query"],
            "risk_updates": s["risk_updates_at_query"], "source_correct": s["correct"], "repair_correct": r["correct"]})
    for domain in DOMAINS:
        for phase in phases:
            for population, population_rows in (("returned_training_records", returned), ("all_arrivals", arrivals)):
                rows = [r for r in population_rows if (domain == "ALL" or r["image_domain"] == domain)
                    and (phase == "all" or phase == "before_learning" and r["risk_updates"] == 0
                         or phase == "after_learning" and r["risk_updates"] > 0
                         or phase == "after_warmup" and r["warm"])]
                correction.append(case_counts(seed, domain, phase, population, rows))
        for period in PERIODS:
            for pool in ("warm_all", "source_base", "repair_base", "disagreement"):
                rows = [r for r in arrivals if r["warm"] and r["allowed"]
                    and (domain == "ALL" or r["image_domain"] == domain)
                    and (period == "all" or periods[r["step"]] == period)
                    and (pool == "warm_all" or pool == "source_base" and r["source_confidence"] >= confidence and r["source_consistent"]
                         or pool == "repair_base" and r["repair_confidence"] >= confidence and r["repair_consistent"]
                         or pool == "disagreement" and r["source_prediction"] != r["repair_prediction"])]
                correction.append(case_counts(seed, domain, period, pool, rows))
    return bias, correction, audit


def case_counts(seed, domain, phase, population, rows):
    fixes = sum(not r["source_correct"] and r["repair_correct"] for r in rows)
    breaks = sum(r["source_correct"] and not r["repair_correct"] for r in rows)
    return {"seed": seed, "domain": domain, "phase": phase, "population": population,
        "samples": len(rows), "both_correct": sum(r["source_correct"] and r["repair_correct"] for r in rows),
        "both_wrong": sum(not r["source_correct"] and not r["repair_correct"] for r in rows),
        "repair_fixes": fixes, "repair_breaks": breaks, "net_correction_pp": ratio(fixes - breaks, len(rows)),
        "informative_pct": ratio(fixes + breaks, len(rows))}


def matched_query_bias(seed, arrivals):
    """Match each batch's unqueried expectation to its returned-query count."""
    batches = defaultdict(list)
    for row in arrivals:
        batches[row["step"]].append(row)
    results = []
    for domain in DOMAINS:
        for phase in ("all", "before_learning", "after_learning", "after_warmup"):
            eligible = [rows for rows in batches.values() if (domain == "ALL" or rows[0]["image_domain"] == domain)
                and (phase == "all" or phase == "before_learning" and rows[0]["risk_updates"] == 0
                     or phase == "after_learning" and rows[0]["risk_updates"] > 0
                     or phase == "after_warmup" and rows[0]["warm"])]
            for branch in ("source", "repair"):
                n = count = dropped = 0
                actual = control = 0.0
                for rows in eligible:
                    queried = [r for r in rows if r["returned_query"]]
                    unqueried = [r for r in rows if not r["queried"]]
                    if not queried:
                        continue
                    if not unqueried:
                        dropped += 1
                        continue
                    count += 1
                    n += len(queried)
                    actual += sum(r[f"{branch}_correct"] for r in queried)
                    control += len(queried) * sum(r[f"{branch}_correct"] for r in unqueried) / len(unqueried)
                results.append({"seed": seed, "domain": domain, "phase": phase, "branch": branch,
                    "matched_batches": count, "no_control_batches": dropped, "query_weight": n,
                    "queried_accuracy_pct": ratio(actual, n), "time_matched_unqueried_accuracy_pct": ratio(control, n),
                    "query_minus_unqueried_pp": ratio(actual - control, n)})
    return results


def aggregate(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
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
