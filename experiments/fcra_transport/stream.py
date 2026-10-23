"""Continuous recurrent PACS stream; segment IDs never enter learning decisions."""

from dataclasses import asdict
from pathlib import Path
import time

import torch

from experiments.delayed_feedback.run import write_csv
from .spec import STAGES


def save_rows(path, rows, fields):
    if rows:
        write_csv(path, rows)
    else:
        Path(path).write_text(",".join(fields) + "\n", encoding="utf-8")


def feedback_rows(algorithm, query_stages, return_stages):
    rows = []
    for record in algorithm.oracle.records:
        returned = record.return_step is not None
        query_stage = query_stages[record.anchor_index]
        return_stage = return_stages.get(record.anchor_index)
        rows.append({**asdict(record), "query_stage": query_stage,
                     "return_stage": return_stage,
                     "cross_domain": record.query_domain != record.return_domain if returned else None,
                     "cross_stage": query_stage != return_stage if returned else None})
    return rows


def counts(rows):
    returned = [row for row in rows if row["return_step"] is not None]
    cross = sum(row["cross_domain"] for row in returned)
    stages = sum(row["cross_stage"] for row in returned)
    return {"queried": len(rows), "returned": len(returned),
            "cross_domain_returned": cross,
            "cross_domain_pct": 100 * cross / len(returned) if returned else None,
            "cross_stage_returned": stages,
            "cross_stage_pct": 100 * stages / len(returned) if returned else None,
            "pending": len(rows) - len(returned)}


@torch.no_grad()
def run_stream(algorithm, output, progress):
    """Photo is checkpoint source only; online visits are A,C,S,A,P.

    A repeat uses the same ImageFolder identities/order; a physical image is
    queried once globally. Labels requested on visit 1 may train visit 2.
    """
    from .policy import DEFAULTS

    output = Path(output)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    algorithm.enable_bn(algorithm.model)
    offsets, total = {}, 0
    for env_id, dataset in enumerate(algorithm.dataset):
        offsets[env_id] = total
        total += len(dataset)
    query_stages, return_stages = {}, {}
    batches, stages = [], []
    global_step = 0
    before_correct = before_total = after_correct = after_total = 0
    first_return = None
    encounter_offset = 0
    for stage, env_id in STAGES:
        domain = algorithm.experiment.domains[env_id]
        loader = algorithm.fast_loader[env_id]
        # This is exactly the physical order consumed by FastDataLoader.
        physical_order = list(loader.sampler)
        cursor = correct_total = sample_total = 0
        progress({"event": "domain", "domain": stage, "total": len(loader)})
        for domain_step, (data, evaluation_labels) in enumerate(loader):
            data = data.to(algorithm.config.device)
            ids = torch.tensor(physical_order[cursor:cursor + len(data)], dtype=torch.long)
            ids += offsets[env_id]
            previous = algorithm.query_anchors.num_elem() if algorithm.query_anchors is not None else 0
            logits = algorithm.select_current(data, ids)
            if algorithm.stream_step != global_step:
                raise AssertionError("Learner clock diverged from stream clock")
            # Truth below is only used for evaluation or oracle encapsulation.
            correct = int((logits.argmax(1).cpu() == evaluation_labels.cpu()).sum())
            correct_total += correct
            sample_total += len(data)
            current = algorithm.query_anchors.num_elem() if algorithm.query_anchors is not None else 0
            label_lookup = {int(sid): int(label) for sid, label in zip(ids, evaluation_labels)}
            for index in range(previous, current):
                sid = int(algorithm.query_anchors.target[index])
                algorithm.oracle.query(sid, index, label_lookup[sid], global_step, domain)
                query_stages[index] = stage
            algorithm.budgets += current - previous
            # Pseudo labels are from the frozen source teacher, never test truth.
            if algorithm.LE:
                _, _, algorithm.source_anchors = algorithm.sample_select(
                    algorithm.teacher, data, ids, algorithm.source_anchors,
                    algorithm.source_n_clusters, 0, use_pseudo_label=True,
                    ent_bound=algorithm.config.atta.SimATTA.el, incremental_cluster=False)
            if not algorithm.target_cluster:
                algorithm.n_clusters = 0
            algorithm.n_clusters += algorithm.nc_increase
            algorithm.source_n_clusters = 101
            descriptions = algorithm.observe(data)
            # In particular for D=0: no new queried label enters branch training.
            audit_cost = algorithm.audit(range(previous, current), ids, descriptions, global_step)
            released = algorithm.oracle.release(global_step, domain)
            if released and first_return is None:
                first_return = global_step
            for index, label in released:
                return_stages[index] = stage
            algorithm.on_labels_released(global_step, domain, released)
            algorithm.received.update(released)
            algorithm.train_batch(audit_cost)
            if algorithm.query_anchors is not None and algorithm.received:
                algorithm.query_anchors = algorithm.update_anchors_feats(algorithm.query_anchors)
            # First-return batch was predicted before its feedback, so it is BEFORE.
            is_before = first_return is None or global_step <= first_return
            if is_before:
                before_correct += correct
                before_total += len(data)
            else:
                after_correct += correct
                after_total += len(data)
            batches.append({
                "seed": algorithm.config.random_seed, "delay": algorithm.oracle.delay,
                "step": global_step, "stage": stage, "domain": domain,
                "domain_step": domain_step, "encounter_offset": encounter_offset,
                "samples": len(data), "correct": correct, "accuracy_pct": 100 * correct / len(data),
                "before_first_feedback": is_before, "new_queries": current - previous,
                "new_returns": len(released), "audit_updates": audit_cost,
                **algorithm.oracle.counts(),
            })
            progress({"event": "batch", "domain": stage, "total": len(loader),
                      "completed": domain_step + 1, "accuracy_pct": 100 * correct_total / sample_total,
                      **algorithm.oracle.counts()})
            cursor += len(data)
            encounter_offset += len(data)
            global_step += 1
        if cursor != len(physical_order):
            raise AssertionError("Dataset identity/order mismatch")
        stages.append({"stage": stage, "domain": domain, "samples": sample_total,
                       "correct": correct_total, "accuracy_pct": 100 * correct_total / sample_total})

    # No queue flush. Cross-domain uses semantic names; cross-stage uses visits.
    feedback = feedback_rows(algorithm, query_stages, return_stages)
    totals = counts(feedback)
    if totals["queried"] != totals["returned"] + totals["pending"]:
        raise AssertionError("Feedback accounting mismatch")
    if algorithm.main_updates + algorithm.audit_updates > global_step * DEFAULTS["steps_per_batch"]:
        raise AssertionError("Stream update cap exceeded")
    domains = []
    for domain in ("Art", "Cartoon", "Sketch", "Photo"):
        group = [row for row in stages if row["domain"] == domain]
        n = sum(row["samples"] for row in group)
        k = sum(row["correct"] for row in group)
        domains.append({"domain": domain, "samples": n, "correct": k,
                        "accuracy_pct": 100 * k / n,
                        **counts([row for row in feedback if row["query_domain"] == domain])})
    for row in stages:
        row.update(counts([item for item in feedback if item["query_stage"] == row["stage"]]))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {"method": "FCRA-Transport", "seed": algorithm.config.random_seed,
              "delay": algorithm.oracle.delay,
              **{f"{row['stage']}_accuracy_pct": row["accuracy_pct"] for row in stages},
              "Art_accuracy_pct": domains[0]["accuracy_pct"],
              "stream_accuracy_pct": 100 * sum(row["correct"] for row in stages) / encounter_offset,
              "stage_macro_accuracy_pct": sum(row["accuracy_pct"] for row in stages) / len(stages),
              "domain_macro_accuracy_pct": sum(row["accuracy_pct"] for row in domains) / len(domains),
              "before_feedback_accuracy_pct": 100 * before_correct / before_total if before_total else None,
              "after_feedback_accuracy_pct": 100 * after_correct / after_total if after_total else None,
              **totals, "main_updates": algorithm.main_updates, "audit_updates": algorithm.audit_updates,
              "transport_updates": algorithm.transport_updates,
              "audited_queries": len(algorithm.audit_predictions) // 3,
              "returned_audits": len(algorithm.reward_rows) // 3,
              "pending_audits": len(algorithm.pending_audits),
              "stream_seconds": elapsed,
              "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 2**20}
    for name, rows, fields in (
        ("batches.csv", batches, ["step", "stage", "domain"]),
        ("stages.csv", stages, ["stage", "domain", "accuracy_pct"]),
        ("domains.csv", domains, ["domain", "accuracy_pct"]),
        ("feedback.csv", feedback, ["sample_id", "query_stage", "return_stage", "cross_domain", "cross_stage"]),
        ("queries.csv", algorithm.query_rows, ["step", "new_queries"]),
        ("audit_predictions.csv", algorithm.audit_predictions,
         ["step", "anchor_index", "action", "probabilities", "context"]),
        ("audit_rewards.csv", algorithm.reward_rows,
         ["return_step", "audit_step", "anchor_index", "action", "reward"]),
        ("decisions.csv", algorithm.decision_rows, ["step", "iteration", "original", "small", "medium"]),
        ("budget.csv", algorithm.budget_rows, ["step", "quota", "main_updates", "audit_updates", "unused"]),
    ):
        save_rows(output / name, rows, fields)
    from experiments.delayed_feedback.run import dump_json
    dump_json(output / "compute.json", {**algorithm.compute, "main_updates": algorithm.main_updates,
                                        "audit_updates": algorithm.audit_updates, "seconds": elapsed})
    return result
