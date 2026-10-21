"""Read completed FCRA-G CSVs on CPU; never load data, models, or checkpoints."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


DELAYS = (0, 20, 50, 100)
DOMAINS = ("Art", "Cartoon", "Sketch")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite CSV value: {value}")
    return result


def pct_count(value, samples):
    # graph.csv stores float32 batch percentages, not exact counts.
    raw = number(value) * samples / 100
    count = round(raw)
    if abs(raw - count) > 0.001 or not 0 <= count <= samples:
        raise ValueError("Batch percentage cannot be reconstructed as a count")
    return count


def rate(numerator, denominator):
    return 100 * numerator / denominator if denominator else None


def optional_step(value):
    return None if value in (None, "", "None") else int(value)


def collect_run(root, reference, delay, seed):
    folder = root / f"D{delay}" / f"seed_{seed}"
    ref_folder = reference / f"D{delay}" / f"seed_{seed}"
    batches = read_csv(folder / "batches.csv")
    graph_rows = read_csv(folder / "graph.csv")
    ref_batches = read_csv(ref_folder / "batches.csv")
    events = read_csv(folder / "expert_feedback.csv")
    feedback = read_csv(folder / "feedback.csv")
    ref_feedback = read_csv(ref_folder / "feedback.csv")
    result = json.loads((folder / "result.json").read_text())
    if (result.get("method") != "FCRA-G" or result["delay"] != delay
            or result["seed"] != seed or not batches or len(batches) != len(ref_batches)):
        raise ValueError(f"Incomplete or mismatched run: {folder}")
    if (delay == 0 and graph_rows) or (delay > 0 and len(graph_rows) != len(batches)):
        raise ValueError(f"Unexpected graph coverage: {folder}")
    schedule_keys = ("sample_id", "anchor_index", "query_step", "query_domain", "due_step",
                     "return_step", "return_domain")
    normalize = lambda value: "" if value in (None, "", "None") else str(value)
    if len(feedback) != len(ref_feedback) or any(
            normalize(actual[key]) != normalize(expected[key])
            for actual, expected in zip(feedback, ref_feedback) for key in schedule_keys):
        raise ValueError("Actual feedback plan differs from the reference")
    returned_ids = {int(row["sample_id"]) for row in feedback
                    if optional_step(row["return_step"]) is not None}
    event_ids = [int(row["sample_id"]) for row in events]
    if (delay == 0 and events) or (delay > 0 and (
            len(event_ids) != len(set(event_ids)) or set(event_ids) != returned_ids)):
        raise ValueError("Expert feedback coverage differs from returned labels")
    first_return = min((optional_step(row["return_step"]) for row in feedback
                        if optional_step(row["return_step"]) is not None), default=None)
    graphs = {int(row["step"]): row for row in graph_rows}
    if len(graphs) != len(graph_rows):
        raise ValueError("Duplicate graph batch")
    audit = []
    for step, (batch, ref) in enumerate(zip(batches, ref_batches)):
        for entry in (batch, ref):
            if (int(entry["step"]) != step or int(entry["seed"]) != seed
                    or int(entry["delay"]) != delay):
                raise ValueError("Batch identity or clock mismatch")
        n = int(batch["samples"])
        if (n <= 0 or batch["domain"] != ref["domain"] or n != int(ref["samples"])
                or int(batch["domain_step"]) != int(ref["domain_step"])):
            raise ValueError("Reference stream structure mismatch")
        final, reference_correct = int(batch["correct"]), int(ref["correct"])
        row = {"delay": delay, "seed": seed, "step": step, "domain": batch["domain"],
               "domain_step": int(batch["domain_step"]), "samples": n,
               "phase": "before_first_feedback" if first_return is None or step <= first_return
               else "after_first_feedback", "reference_correct": reference_correct}
        if delay == 0:
            main = graph1 = graph2 = final
            changed = repaired = harmed = wrong_to_wrong = disagreement1 = disagreement2 = 0
            h = hz = w1 = w2 = beta1 = beta2 = seconds = 0.0
            w0 = 1.0
            fallback = False
        else:
            graph = graphs[step]
            if graph["domain"] != batch["domain"] or int(graph["samples"]) != n:
                raise ValueError("Graph and stream row mismatch")
            main = pct_count(graph["main_accuracy_pct"], n)
            if pct_count(graph["final_accuracy_pct"], n) != final:
                raise ValueError("Recorded final graph accuracy differs from stream scoring")
            graph1 = pct_count(graph["graph1_accuracy_pct"], n)
            graph2 = pct_count(graph["graph2_accuracy_pct"], n)
            changed = int(graph["changed"])
            repaired = int(graph["changed_correct"])
            # changed_incorrect includes both correct->wrong and wrong->wrong.
            harmed = repaired - (final - main)
            wrong_to_wrong = changed - repaired - harmed
            disagreement1 = pct_count(graph["main_graph1_disagreement_pct"], n)
            disagreement2 = pct_count(graph["main_graph2_disagreement_pct"], n)
            if (changed != pct_count(graph["main_final_disagreement_pct"], n)
                    or changed - repaired != int(graph["changed_incorrect"])):
                raise ValueError("Inconsistent prediction change diagnostics")
            h, hz = number(graph["h_mean"]), number(graph["h_zero_pct"])
            w0, w1, w2 = (number(graph[key]) for key in ("w0", "w1", "w2"))
            beta1, beta2 = number(graph["beta1_mean"]), number(graph["beta2_mean"])
            seconds = number(graph["graph_seconds"])
            if graph["fallback"].lower() not in ("true", "false"):
                raise ValueError("Unexpected fallback flag")
            fallback = graph["fallback"].lower() == "true"
            if (not 0 <= h <= 1 or not 0 <= hz <= 100 or min(w0, w1, w2) < 0
                    or abs(w0 + w1 + w2 - 1) > 1e-5
                    or not 0 <= beta1 <= 1 or not 0 <= beta2 <= 1 or seconds < 0):
                raise ValueError("Invalid graph influence or weight diagnostics")
        if (not all(0 <= count <= n for count in (main, final, graph1, graph2,
                                                reference_correct, changed, repaired, harmed,
                                                wrong_to_wrong, disagreement1, disagreement2))
                or repaired > n - main or harmed > main
                or wrong_to_wrong > n - main - repaired):
            raise ValueError("Impossible correction contingency counts")
        row.update(main_correct=main, final_correct=final, graph1_correct=graph1,
                   graph2_correct=graph2, changed=changed, repaired=repaired, harmed=harmed,
                   wrong_to_wrong=wrong_to_wrong, graph1_disagreements=disagreement1,
                   graph2_disagreements=disagreement2, h_mean=h, h_zero_pct=hz,
                   w0=w0, w1=w1, w2=w2, beta1_mean=beta1, beta2_mean=beta2,
                   effective_graph_coefficient=w1 * beta1 + w2 * beta2,
                   graph_seconds=seconds, fallback=int(fallback))
        audit.append(row)
    actual_accuracy = rate(sum(row["final_correct"] for row in audit),
                           sum(row["samples"] for row in audit))
    if abs(actual_accuracy - number(result["stream_accuracy_pct"])) > 1e-5:
        raise ValueError("Run result does not match batch counts")
    return audit, events


def aggregate(rows, scope):
    n = sum(row["samples"] for row in rows)
    total = lambda key: sum(row[key] for row in rows)
    weighted = lambda key: sum(row[key] * row["samples"] for row in rows) / n
    main, final, ref = total("main_correct"), total("final_correct"), total("reference_correct")
    changed, repaired, harmed = total("changed"), total("repaired"), total("harmed")
    return {
        "delay": rows[0]["delay"], "seed": rows[0]["seed"], "scope": scope,
        "samples": n, "batches": len(rows),
        "main_accuracy_pct": rate(main, n), "final_accuracy_pct": rate(final, n),
        "reference_accuracy_pct": rate(ref, n),
        "graph1_accuracy_pct": rate(total("graph1_correct"), n),
        "graph2_accuracy_pct": rate(total("graph2_correct"), n),
        "within_run_gain_pp": rate(final - main, n),
        "paired_reference_gain_pp": rate(final - ref, n),
        "main_reference_gap_pp": rate(main - ref, n),
        "main_reference_mismatched_batches": sum(row["main_correct"] != row["reference_correct"]
                                                 for row in rows),
        "changed": changed, "changed_pct": rate(changed, n),
        "repaired": repaired, "harmed": harmed, "wrong_to_wrong": total("wrong_to_wrong"),
        "net_repairs": repaired - harmed,
        "changed_final_correct_pct": rate(repaired, changed),
        "repair_of_main_errors_pct": rate(repaired, n - main),
        "harm_of_main_correct_pct": rate(harmed, main),
        "graph1_disagreement_pct": rate(total("graph1_disagreements"), n),
        "graph2_disagreement_pct": rate(total("graph2_disagreements"), n),
        **{key: weighted(key) for key in ("h_mean", "h_zero_pct", "w0", "w1", "w2",
                                         "beta1_mean", "beta2_mean", "effective_graph_coefficient")},
        "fallback_batches": total("fallback"), "graph_seconds": total("graph_seconds"),
    }


def feedback_summary(events, delay, seed):
    rows = []
    for scope in ("stream", *DOMAINS):
        subset = [row for row in events if scope == "stream" or row["query_domain"] == scope]
        row = {"delay": delay, "seed": seed, "query_scope": scope, "returned_events": len(subset)}
        for expert in (1, 2):
            values = [number(event[f"gain{expert}"]) for event in subset]
            row[f"expert{expert}_mean_clipped_nll_gain"] = statistics.mean(values) if values else None
            row[f"expert{expert}_positive_gain_pct"] = rate(sum(value > 0 for value in values), len(values))
        rows.append(row)
    return rows


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--results-dir", type=Path, required=True)
    cli.add_argument("--reference-dir", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path,
                     help="New diagnostic directory; defaults to results-dir/diagnostics")
    args = cli.parse_args()
    root, reference = args.results_dir.resolve(), args.reference_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    ref_manifest = json.loads((reference / "manifest.json").read_text())
    seeds = manifest["seeds"]
    if (manifest.get("method") != "FCRA-G" or manifest.get("status") != "complete"
            or tuple(manifest["delays"]) != DELAYS or len(seeds) != 3 or len(set(seeds)) != 3
            or ref_manifest.get("method") != "FCRA" or ref_manifest.get("status") != "complete"
            or set(ref_manifest["seeds"]) != set(seeds)
            or tuple(ref_manifest["delays"]) != DELAYS
            or manifest.get("dataset") != "PACS" or ref_manifest.get("dataset") != "PACS"
            or manifest.get("batch_size") != 32 or ref_manifest.get("batch_size") != 32
            or manifest["checkpoint_sha256"] != ref_manifest["checkpoint_sha256"]):
        raise ValueError("Expected matching complete PACS sweeps with three common seeds")
    if manifest["reference_manifest_sha256"] != fingerprint(reference / "manifest.json"):
        raise ValueError("Reference manifest differs from the recorded experiment")
    batch_audit, per_run, feedback_rows = [], [], []
    hashes = {"results/manifest.json": fingerprint(root / "manifest.json"),
              "reference/manifest.json": fingerprint(reference / "manifest.json")}
    for delay in DELAYS:
        for seed in seeds:
            key = f"D{delay}/seed_{seed}"
            for name, digest in manifest["reference_hashes"][key].items():
                if fingerprint(reference / key / name) != digest:
                    raise ValueError(f"Reference file changed: {key}/{name}")
            rows, events = collect_run(root, reference, delay, seed)
            batch_audit.extend(rows)
            scopes = {"stream": rows}
            for domain in DOMAINS:
                scopes[domain] = [row for row in rows if row["domain"] == domain]
                scopes[f"{domain}_first10"] = [row for row in scopes[domain] if row["domain_step"] < 10]
            for phase in ("before_first_feedback", "after_first_feedback"):
                scopes[phase] = [row for row in rows if row["phase"] == phase]
            per_run.extend(aggregate(subset, scope) for scope, subset in scopes.items() if subset)
            feedback_rows.extend(feedback_summary(events, delay, seed))
            for name in ("batches.csv", "graph.csv", "feedback.csv", "expert_feedback.csv", "result.json"):
                hashes[f"results/{key}/{name}"] = fingerprint(root / key / name)
    summary = []
    metrics = [key for key in per_run[0] if key not in ("delay", "seed", "scope")]
    for delay in DELAYS:
        for scope in dict.fromkeys(row["scope"] for row in per_run):
            group = [row for row in per_run if row["delay"] == delay and row["scope"] == scope]
            entry = {"delay": delay, "scope": scope, "runs": len(group)}
            for metric in metrics:
                values = [row[metric] for row in group if row[metric] is not None]
                entry[f"{metric}_mean"] = statistics.mean(values) if values else None
                entry[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
                entry[f"{metric}_n"] = len(values)
            summary.append(entry)
    output = args.output_dir.resolve() if args.output_dir else root / "diagnostics"
    # A fresh directory is the only write target; never modify the source CSVs.
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "batch_audit.csv", batch_audit)
    write_csv(output / "per_run.csv", per_run)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "feedback_summary.csv", feedback_rows)
    lines = ["# FCRA-G correction audit", "",
             "All rates use sample counts within a seed; ± is the sample SD across seeds.", "",
             "| D | Changed (%) | Repaired | Harmed | Wrong→wrong | Gain vs main (pp) | Gain vs reference (pp) | h | Effective coefficient |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def cell(row, metric):
        mean, std = row[f"{metric}_mean"], row[f"{metric}_std"]
        return "N/A" if mean is None else f"{mean:.4f}" + (f" ± {std:.4f}" if std is not None else "")
    columns = ("changed_pct", "repaired", "harmed", "wrong_to_wrong", "within_run_gain_pp",
               "paired_reference_gain_pp", "h_mean", "effective_graph_coefficient")
    for row in summary:
        if row["scope"] == "stream":
            lines.append(f"| {row['delay']} | " + " | ".join(cell(row, key) for key in columns) + " |")
    lines += ["", "## Interpretation", "",
              "- Check main_reference_gap_pp first: a changed backbone trajectory confounds comparisons to the old run.",
              "- Within-run gain is exactly (repaired - harmed) / samples × 100.",
              "- changed_incorrect in the source CSV combines harmed and wrong-to-wrong; it is not the harm count.",
              "- If experts disagree but final output barely changes, inspect mixture weights and effective coefficient.",
              "- If experts also barely disagree, inspect graph correction strength; old logs do not contain edge mass or Q-P norms.",
              "- Large h_zero_pct suggests missing anchor influence; small h can also arise from weak weights despite connectivity.",
              "- Compare graph1/2 accuracy to main before changing the router. Many changes with zero net repairs suggest cancellation.",
              "- No sample-level joint expert predictions are saved; expert accuracy on disagreement cases cannot be reconstructed.",
              "- Feedback scores cover queried and returned samples only; they are not unbiased current-domain estimates.",
              "- D=0 graph diagnostics are a structural bypass, not evidence for graph effectiveness.",
              "- No significance test or causal explanation is inferred from three rounded means.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "analysis": "FCRA-G existing-log correction audit", "results_dir": str(root),
        "reference_dir": str(reference), "seeds": seeds, "delays": DELAYS,
        "input_sha256": hashes, "inference_or_training": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Analysis complete: {output / 'report.md'}")


if __name__ == "__main__":
    main()
