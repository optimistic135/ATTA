"""Offline domain reports; evaluation CSVs never return to learning."""
from experiments.delayed_feedback.run import write_csv
from experiments.sp_atta.reference import read_csv
from experiments.sp_atta.run import aggregate, cell


def domain_reports(output, results):
    rows = []
    for result in results:
        folder = output / result["method"] / f"seed_{result['seed']}"
        targets = read_csv(folder / "target_quality.csv")
        future = [r for r in read_csv(folder / "selected_future.csv") if r["complete_window"].lower() == "true"]
        probes = read_csv(folder / "selected_probes.csv")
        active_ids = {r["probe_id"] for r in probes if int(r["selected_sampled"]) > 0}
        for domain in ("ALL", "Cartoon", "Sketch"):
            used = [r for r in targets if int(r["training_draws"]) and (domain == "ALL" or r["image_domain"] == domain)]
            draws = sum(int(r["training_draws"]) for r in used)
            target_correct = sum(int(r["selected_correct"]) * int(r["training_draws"]) for r in used)
            source_correct = sum(int(r["source_correct"]) * int(r["training_draws"]) for r in used)
            scored = [r for r in future if domain == "ALL" or r["domain"] == domain]
            active = [r for r in scored if r["probe_id"] in active_ids]
            samples = sum(int(r["samples"]) for r in scored)
            active_samples = sum(int(r["samples"]) for r in active)
            rows.append({"method": result["method"], "seed": result["seed"], "domain": domain,
                "training_draws": draws,
                "selected_target_accuracy_pct": 100 * target_correct / draws if draws else None,
                "source_same_images_accuracy_pct": 100 * source_correct / draws if draws else None,
                "target_net_correction_pp": 100 * (target_correct - source_correct) / draws if draws else None,
                "local_events": len({r["probe_id"] for r in scored}),
                "local_selected_minus_true_pp": 100 * sum(int(r["selected_correct"]) - int(r["true_correct"]) for r in scored) / samples if samples else None,
                "local_selected_minus_history_pp": 100 * sum(int(r["selected_correct"]) - int(r["history_correct"]) for r in scored) / samples if samples else None,
                "active_local_selected_minus_history_pp": 100 * sum(int(r["selected_correct"]) - int(r["history_correct"]) for r in active) / active_samples if active_samples else None})
    write_csv(output / "domain_per_seed.csv", rows)
    metrics = tuple(key for key in rows[0] if key not in ("method", "seed", "domain"))
    summaries = []
    for group, domain in dict.fromkeys((r["method"], r["domain"]) for r in rows):
        summary = aggregate(group, [r for r in rows if r["method"] == group and r["domain"] == domain], metrics)
        summary["domain"] = domain
        summaries.append(summary)
    write_csv(output / "domain_summary.csv", summaries)
    text = "## Domain mechanism\n\n| Method | Domain | Training draws | Target acc (%) | Target net correction (pp) | Active SELECTED−HISTORY (pp) |\n| :--- | :--- | ---: | ---: | ---: | ---: |\n"
    text += "\n".join(f"| {r['method']} | {r['domain']} | {cell(r, 'training_draws')} | {cell(r, 'selected_target_accuracy_pct')} | {cell(r, 'target_net_correction_pp')} | {cell(r, 'active_local_selected_minus_history_pp')} |" for r in summaries)
    return text
