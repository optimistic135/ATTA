"""Analyze saved screening CSVs offline. Standard library only; no GPU or training."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from .offline_metrics import (BATCH_FIELDS, DOMAINS, FRACTION, aggregate, batch_rankings, bias_and_correction,
                              boolean, matched_query_bias)

SOURCE_METHOD = "SP-selection-usability-diagnosis"
METHOD = "SP-screening-per-batch-analysis"
VERSION = 1
OUTPUTS = ("per_batch.csv", "ranking_per_seed.csv", "ranking_summary.csv", "query_bias_per_seed.csv",
           "query_bias_summary.csv", "correction_cases_per_seed.csv", "correction_cases_summary.csv",
           "matched_bias_per_seed.csv", "matched_bias_summary.csv", "input_audit.csv", "summary.md")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    if not rows and fields is None:
        raise ValueError(f"Empty report schema: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields if fields is not None else list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def optional_int(value):
    return None if value in (None, "", "None") else int(value)


def probability(value):
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"Invalid probability: {value!r}")
    return result


def load_seed(root, seed, manifest):
    folder = root / f"seed_{seed}"
    result = json.loads((folder / "result.json").read_text())
    settings = json.loads((folder / "settings.json").read_text())
    for row in (result, settings):
        if (row.get("method"), row.get("seed"), row.get("delay")) != (SOURCE_METHOD, seed, 100):
            raise ValueError(f"Seed identity mismatch: {folder}")
    if settings.get("defaults") != manifest["defaults"]:
        raise ValueError("Source settings differ from source manifest")
    if settings.get("reference_hashes") != manifest["reference_hashes"][str(seed)] or result.get("reference_hashes") != settings["reference_hashes"]:
        raise ValueError("Reference hashes differ in source outputs")
    batches, offset = [], 0
    domain_order = []
    for step, raw in enumerate(read_csv(folder / "batches.csv")):
        size, domain = int(raw["samples"]), raw["domain"]
        if int(raw["step"]) != step or not 1 <= size <= 32 or domain not in DOMAINS[1:]:
            raise ValueError("Invalid batch clock or size")
        if not domain_order or domain_order[-1] != domain:
            domain_order.append(domain)
        batches.append({"step": step, "domain": domain, "lo": offset, "hi": offset + size})
        offset += size
    if domain_order != ["Art", "Cartoon", "Sketch"]:
        raise ValueError("Require PACS A→C→S stream")
    observations, by_step, counts = [], {}, {}
    image_domains = [b["domain"] for b in batches for _ in range(b["hi"] - b["lo"])]
    for raw in read_csv(folder / "score_observations.csv"):
        row = {key: int(raw[key]) for key in ("step", "sample_id", "source_prediction", "repair_prediction",
                                             "source_correct", "repair_correct", "risk_updates", "evaluation_label")}
        row.update({key: boolean(raw[key]) for key in ("warm", "allowed", "source_consistent", "repair_consistent")})
        row.update({key: probability(raw[key]) for key in ("source_score", "repair_score", "source_confidence", "repair_confidence")})
        row.update(image_domain=raw["image_domain"], current_domain=raw["current_domain"])
        step, sid = row["step"], row["sample_id"]
        if not 0 <= step < len(batches):
            raise ValueError("Observation step out of bounds")
        first = batches[max(0, step - manifest["defaults"]["buffer_batches"] + 1)]["lo"]
        if not first <= sid < batches[step]["hi"] or row["current_domain"] != batches[step]["domain"]:
            raise ValueError("Observation ID outside its rolling window")
        if row["image_domain"] != image_domains[sid] or row["risk_updates"] < 0:
            raise ValueError("Observation domain or update counter mismatch")
        warm = int(raw["feedback_count"]) >= manifest["defaults"]["min_feedback"] and row["risk_updates"] >= manifest["defaults"]["min_reliability_updates"]
        if row["warm"] != warm:
            raise ValueError("Warmup metadata mismatch")
        for branch in ("source", "repair"):
            if row[f"{branch}_correct"] != int(row[f"{branch}_prediction"] == row["evaluation_label"]):
                raise ValueError("Inconsistent saved correctness")
        key = (step, sid)
        if key in by_step:
            raise ValueError("Duplicate observation within a batch")
        by_step[key] = row
        counts[step] = counts.get(step, 0) + 1
        observations.append(row)
    arrivals = []
    for batch in batches:
        first = batches[max(0, batch["step"] - manifest["defaults"]["buffer_batches"] + 1)]["lo"]
        if counts.get(batch["step"]) != batch["hi"] - first:
            raise ValueError("Incomplete rolling-window observations")
        rows = [by_step.get((batch["step"], sid)) for sid in range(batch["lo"], batch["hi"])]
        if any(r is None or r["image_domain"] != batch["domain"] for r in rows):
            raise ValueError("Missing/misaligned newly arrived images")
        arrivals.extend(rows)
    feedback, seen = [], set()
    for index, raw in enumerate(read_csv(folder / "feedback.csv")):
        row = {key: int(raw[key]) for key in ("anchor_index", "sample_id", "query_step")}
        row.update(return_step=optional_int(raw["return_step"]), query_domain=raw["query_domain"])
        step = row["query_step"]
        if not 0 <= step < len(batches):
            raise ValueError("Query step out of bounds")
        expected_return = step + 100 if step + 100 < len(batches) else None
        if (row["anchor_index"] != index or row["sample_id"] in seen or row["return_step"] != expected_return
                or not batches[step]["lo"] <= row["sample_id"] < batches[step]["hi"]
                or row["query_domain"] != batches[step]["domain"]):
            raise ValueError("Feedback identity/clock mismatch")
        feedback.append(row)
        seen.add(row["sample_id"])
    if len(feedback) != result["queried"] or sum(r["return_step"] is not None for r in feedback) != result["returned"]:
        raise ValueError("Source feedback count mismatch")
    queries, keys = [], set()
    for raw in read_csv(folder / "query_scores.csv"):
        row = {key: int(raw[key]) for key in ("anchor_index", "query_step", "return_step", "correct", "risk_updates_at_query")}
        row.update({key: probability(raw[key]) for key in ("score", "confidence")})
        row.update(branch=raw["branch"], warm_at_query=boolean(raw["warm_at_query"]), phase=raw["phase"])
        key = (row["anchor_index"], row["branch"])
        if key in keys or row["branch"] not in ("source", "repair") or row["correct"] not in (0, 1):
            raise ValueError("Invalid query score record")
        keys.add(key)
        if not 0 <= row["anchor_index"] < len(feedback):
            raise ValueError("Query anchor index out of bounds")
        fb = feedback[row["anchor_index"]]
        if row["query_step"] != fb["query_step"] or row["return_step"] != fb["return_step"]:
            raise ValueError("Query score clock mismatch")
        if row["phase"] != ("after_learning" if row["risk_updates_at_query"] else "before_learning"):
            raise ValueError("Query phase mismatch")
        observed = by_step[(fb["query_step"], fb["sample_id"])]
        if row["warm_at_query"] != observed["warm"] or row["risk_updates_at_query"] != observed["risk_updates"]:
            raise ValueError("Query state differs from same-time observation")
        queries.append(row)
    expected = {(r["anchor_index"], branch) for r in feedback if r["return_step"] is not None for branch in ("source", "repair")}
    if keys != expected:
        raise ValueError("Missing or extra returned query scores")
    shadow_updates, steps = [], set()
    for raw in read_csv(folder / "shadow_updates.csv"):
        step, indices = int(raw["step"]), json.loads(raw["risk_anchor_indices"])
        if step in steps or not 0 <= step < len(batches) or not isinstance(indices, list):
            raise ValueError("Invalid shadow update clock")
        for index in indices:
            if type(index) is not int or not 0 <= index < len(feedback) or feedback[index]["return_step"] is None or feedback[index]["return_step"] > step:
                raise ValueError("Risk training draw used a label before its return")
        steps.add(step)
        shadow_updates.append({"step": step, "indices": indices})
    if len(shadow_updates) != result["shadow_reliability_updates"]:
        raise ValueError("Shadow update count mismatch")
    return observations, arrivals, batches, feedback, queries, shadow_updates


def cell(row, key):
    value, sd = row[f"{key}_mean"], row[f"{key}_std"]
    if value is None:
        return "N/A"
    return f"{value:.2f} ± {sd:.2f}" if sd is not None else f"{value:.2f}"


def report(output, per_batch, ranking, bias, correction, matched, audits, seeds):
    rank_summary = aggregate(ranking, ("pool", "domain", "period", "branch"))
    bias_summary = aggregate(bias, ("support", "domain", "phase", "cohort", "branch"))
    correction_summary = aggregate(correction, ("domain", "phase", "population"))
    matched_summary = aggregate(matched, ("domain", "phase", "branch"))
    for name, rows in (("per_batch", per_batch), ("ranking_per_seed", ranking), ("ranking_summary", rank_summary),
                       ("query_bias_per_seed", bias), ("query_bias_summary", bias_summary),
                       ("correction_cases_per_seed", correction), ("correction_cases_summary", correction_summary),
                       ("matched_bias_per_seed", matched), ("matched_bias_summary", matched_summary),
                       ("input_audit", audits)):
        write_csv(output / f"{name}.csv", rows, BATCH_FIELDS if name == "per_batch" else None)
    text = "# Within-batch screening analysis: PACS D=100\n\nNo new training or GPU computation.\n\n"
    text += "## Newly arrived images: fixed within-batch top10%\n\n| Domain | Branch | Selected weight | Empty batches (%) | Confidence acc (%) | Risk acc (%) | Random expectation (%) | Risk−confidence (pp) |\n| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    text += "\n".join(f"| {r['domain']} | {r['branch']} | {cell(r, 'selected_weight')} | {cell(r, 'zero_candidate_pct')} | {cell(r, 'confidence_high_accuracy_pct')} | {cell(r, 'risk_high_accuracy_pct')} | {cell(r, 'random_expected_accuracy_pct')} | {cell(r, 'risk_minus_confidence_pp')} |" for r in rank_summary if r["pool"] == "current_batch" and r["period"] == "all")
    text += "\n\n## Warmup-period thirds: newly arrived images\n\n| Domain | Period | Branch | Warm batches | Confidence acc (%) | Risk acc (%) | Random expectation (%) |\n| :--- | :--- | :--- | ---: | ---: | ---: | ---: |\n"
    text += "\n".join(f"| {r['domain']} | {r['period']} | {r['branch']} | {cell(r, 'warm_batches')} | {cell(r, 'confidence_high_accuracy_pct')} | {cell(r, 'risk_high_accuracy_pct')} | {cell(r, 'random_expected_accuracy_pct')} |" for r in rank_summary if r["pool"] == "current_batch" and r["domain"] in ("Cartoon", "Sketch") and r["period"] != "all")
    text += "\n\n## Query bias: same query-time support, after warmup\n\n| Domain | Branch | Cohort | Samples | Target acc (%) |\n| :--- | :--- | :--- | ---: | ---: |\n"
    text += "\n".join(f"| {r['domain']} | {r['branch']} | {r['cohort']} | {cell(r, 'samples')} | {cell(r, 'accuracy_pct')} |" for r in bias_summary if r["support"] == "returned_query_steps" and r["phase"] == "after_warmup" and r["cohort"] in ("all_current", "returned_queried", "unqueried"))
    text += "\n\n## Query-count-weighted within-batch bias, after warmup\n\n| Domain | Branch | Query weight | Queried acc (%) | Matched unqueried acc (%) | Query−unqueried (pp) |\n| :--- | :--- | ---: | ---: | ---: | ---: |\n"
    text += "\n".join(f"| {r['domain']} | {r['branch']} | {cell(r, 'query_weight')} | {cell(r, 'queried_accuracy_pct')} | {cell(r, 'time_matched_unqueried_accuracy_pct')} | {cell(r, 'query_minus_unqueried_pp')} |" for r in matched_summary if r["phase"] == "after_warmup")
    text += "\n\n## Paired correction labels (ALL)\n\n| Population | Phase | Samples | Repair fixes | Repair breaks | Net correction (pp) | Informative (%) |\n| :--- | :--- | ---: | ---: | ---: | ---: | ---: |\n"
    text += "\n".join(f"| {r['population']} | {r['phase']} | {cell(r, 'samples')} | {cell(r, 'repair_fixes')} | {cell(r, 'repair_breaks')} | {cell(r, 'net_correction_pp')} | {cell(r, 'informative_pct')} |" for r in correction_summary if r["domain"] == "ALL" and r["population"] in ("returned_training_records", "all_arrivals"))
    text += "\n\n## Input alignment audit\n\n| Seed | Returned prediction records | Correctness differences | Max score difference | Max confidence difference |\n| ---: | ---: | ---: | ---: | ---: |\n"
    text += "\n".join(f"| {r['seed']} | {r['returned_prediction_records']} | {r['correctness_mismatches']} | {r['max_score_abs_difference']:.3g} | {r['max_confidence_abs_difference']:.3g} |" for r in audits)
    text += "\n\n## Timing of informative returned labels\n\n| Seed | Different correctness labels | First informative return | First informative training | Informative training draws | All risk draws |\n| ---: | ---: | ---: | ---: | ---: | ---: |\n"
    text += "\n".join(f"| {r['seed']} | {r['informative_returned_queries']} | {r['first_informative_return_step'] if r['first_informative_return_step'] is not None else 'N/A'} | {r['first_informative_training_step'] if r['first_informative_training_step'] is not None else 'N/A'} | {r['informative_training_draws']} | {r['total_risk_training_draws']} |" for r in audits)
    text += (f"\n\nSeeds={seeds}; means ± sample SD across seeds; valid n in CSV. "
        "Each batch uses k=ceil(0.1N), at least1 if nonempty; actual coverage can exceed10% and is reported. "
        "Risk and confidence use identical candidate pool and k. Boundary ties are fractional expectations, "
        "random is uniform-within-batch expectation, not random training. Accuracy weights expected correct by total k. "
        "Periods divide each domain's post-warmup batch sequence into thirds, not the full domain or full stream. "
        "Primary current_batch contains only arriving images once. Rolling-window results are supplementary and repeat images. "
        "Query-bias support matching reduces temporal mixing; within-batch matching additionally weights "
        "unqueried expectation by that batch's query count, but neither establishes causal selection effects; "
        "full-stream and pending cohorts use offline evaluation labels unavailable online. "
        "Returned correction cases use stored query-time correctness; current cases use post-prediction evaluation truth. "
        "Input audit exposes numerical differences between cached query predictions and window rescoring. "
        "No labels fit a model or choose a threshold; these are target-quality diagnostics, not deployed gains or significance evidence.\n")
    (output / "summary.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--input-dir", type=Path, required=True, help="Completed selection_diagnosis result root")
    cli.add_argument("--output-dir", type=Path, help="Default: INPUT/batch_analysis")
    cli.add_argument("--resume", action="store_true", help="Skip verified completed analysis or archive/rebuild incomplete analysis")
    args = cli.parse_args()
    root = args.input_dir.resolve()
    output = args.output_dir.resolve() if args.output_dir else root / "batch_analysis"
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("status") != "complete" or manifest.get("method") != SOURCE_METHOD
            or manifest.get("delays") != [100] or manifest.get("batch_size") != 32):
        raise ValueError("Input must be a completed screening-usability PACS D100 result root")
    seeds = manifest["seeds"]
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Input must contain three distinct seeds")
    files = [root / "manifest.json"] + [root / f"seed_{seed}" / name for seed in seeds for name in
        ("settings.json", "result.json", "batches.csv", "score_observations.csv", "query_scores.csv", "feedback.csv", "shadow_updates.csv")]
    if any(output == p.resolve() or output in p.resolve().parents for p in files):
        raise ValueError("Output may not contain or replace input files")
    hashes = {str(p.relative_to(root)): sha256(p) for p in files}
    confidence = probability(manifest["defaults"]["confidence"])
    config = {"method": METHOD, "version": VERSION, "rank_fraction": FRACTION, "confidence": confidence}
    output.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output.parent / f".{output.name}.analysis.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another analysis is using this output") from None
        if output.exists():
            if not args.resume:
                raise FileExistsError("Output exists; use --resume or a new --output-dir")
            previous = json.loads((output / "manifest.json").read_text())
            if previous.get("config") != config or previous.get("input_hashes") != hashes:
                raise ValueError("Analysis inputs/settings changed; use a new output directory")
            if previous.get("status") == "complete":
                if any(not (output / name).is_file() or sha256(output / name) != previous.get("output_hashes", {}).get(name) for name in OUTPUTS):
                    raise ValueError("Completed report files changed; preserve them and use a new directory")
                print(f"Already complete, skipped: {output}\n{output / 'summary.md'}", flush=True)
                return
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            archive = output.parent / "_batch_analysis_archive" / f"{output.name}_{stamp}"
            archive.parent.mkdir(parents=True, exist_ok=True)
            output.rename(archive)
        output.mkdir(parents=True, exist_ok=False)
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        run = {"status": "running", "config": config, "input_dir": str(root), "input_hashes": hashes,
               "seeds": seeds, "git_commit": commit, "python": sys.version,
               "source_git_commit": manifest.get("git_commit")}
        dump_json(output / "manifest.json", run)
        per_batch, ranking, bias, correction, matched, audits = [], [], [], [], [], []
        try:
            for seed in seeds:
                print(f"Analyzing existing CSVs: seed={seed}", flush=True)
                observations, arrivals, batches, feedback, queries, shadow_updates = load_seed(root, seed, manifest)
                batch_rows, rank_rows, periods = batch_rankings(seed, observations, batches, confidence)
                bias_rows, correction_rows, audit = bias_and_correction(seed, arrivals, feedback, queries, periods, confidence, shadow_updates)
                per_batch.extend(batch_rows)
                ranking.extend(rank_rows)
                bias.extend(bias_rows)
                correction.extend(correction_rows)
                matched.extend(matched_query_bias(seed, arrivals))
                audits.append(audit)
            if any(sha256(p) != hashes[str(p.relative_to(root))] for p in files):
                raise ValueError("Input files changed while analyzing")
            report(output, per_batch, ranking, bias, correction, matched, audits, seeds)
        except BaseException:
            run["status"] = "failed"
            dump_json(output / "manifest.json", run)
            raise
        run.update(status="complete", output_hashes={name: sha256(output / name) for name in OUTPUTS})
        dump_json(output / "manifest.json", run)
        print(f"Complete: {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
