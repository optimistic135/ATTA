"""New-feedback training slots and pool-size-independent supervision on PACS."""

from collections import Counter, OrderedDict
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from experiments.delayed_feedback.run import dump_json
from experiments.delayed_feedback.simatta import write_csv
from experiments.fcra.algorithm import FCRA


DEFAULTS = {
    "feedback_update_fraction": 0.25,
    "fresh_minibatch_fraction": 0.5,
    "true_coefficient": 1.0,
    "pseudo_coefficient": 0.25,
    "fresh_rng_seed_offset": 31817,
    "fresh_policy": "round-robin over an independently shuffled pending-fresh pool",
    "zero_quota_policy": "keep released feedback waiting for the next positive quota",
}


def save_rows(path, rows, header):
    if rows:
        write_csv(path, rows)
    else:
        path.write_text(header + "\n", encoding="utf-8")


class FeedbackFCRA(FCRA):
    def __init__(self, config, delay, trace):
        super().__init__(config, delay)
        self.trace = trace
        if (not 0 < DEFAULTS["feedback_update_fraction"] <= 1
                or not 0 < DEFAULTS["fresh_minibatch_fraction"] < 1
                or DEFAULTS["true_coefficient"] <= 0 or DEFAULTS["pseudo_coefficient"] < 0):
            raise ValueError("Invalid fixed feedback allocation or loss coefficients")
        structure = [(self.experiment.domains[env], min(config.atta.batch_size, len(loader.sampler) - start))
                     for env in config.dataset.test_envs[1:]
                     for loader in [self.fast_loader[env]]
                     for start in range(0, len(loader.sampler), config.atta.batch_size)]
        if structure != [(row["domain"], int(row["samples"])) for row in trace.batches]:
            raise ValueError("Dataset stream differs from the reference batch structure")
        self.waiting_fresh = OrderedDict()
        self.exposure = {}
        self.fresh_rng = torch.Generator().manual_seed(
            (config.random_seed + DEFAULTS["fresh_rng_seed_offset"]) % 2**32)
        self.budget_rows, self.probe_rows, self.online_rows = [], [], []
        self.feedback_steps = self.replay_steps = 0
        self.training_seconds = 0.0
        self.compute = Counter()
        self.model[0].register_forward_hook(self._counter("main"))
        self.teacher[0].register_forward_hook(self._counter("teacher"))

    def _counter(self, kind):
        def hook(module, inputs, output):
            self.compute[f"{kind}_forward_calls"] += 1
            self.compute[f"{kind}_forward_images"] += len(inputs[0])
            if kind == "main" and output.requires_grad:
                def backward(gradient):
                    self.compute["main_backward_calls"] += 1
                    self.compute["main_backward_images"] += len(inputs[0])
                    return gradient
                output.register_hook(backward)
        return hook

    @torch.no_grad()
    def select_queries(self, data, stream_ids):
        """Predict with the original model; query only the sealed reference IDs."""
        self.stream_step += 1
        step = self.stream_step
        self.current_data = data.detach()
        self.model.eval()
        features = self.model[0](data)
        outputs = self.model[1](features)
        if (step >= self.trace.steps or self.trace.bounds[step] !=
                (int(stream_ids[0]), int(stream_ids[-1]) + 1)):
            raise ValueError("Stream clock differs from reference")
        previous = self.query_anchors.num_elem() if self.query_anchors is not None else 0
        records = self.trace.queries[step]
        if [int(row["anchor_index"]) for row in records] != list(range(previous, previous + len(records))):
            raise ValueError("Reference anchor insertion order changed")
        if records:
            positions = torch.tensor([int(row["sample_id"]) - int(stream_ids[0]) for row in records],
                                     dtype=torch.long)
            if int(positions.min()) < 0 or int(positions.max()) >= len(data):
                raise ValueError("Reference query lies outside the current batch")
            device_positions = positions.to(data.device)
            self.query_anchors = self.update_anchors(
                self.query_anchors, data[device_positions].cpu(), stream_ids[positions],
                features[device_positions].cpu(), torch.ones(len(positions)))
            self.teacher.eval()
            teacher_raw = self.teacher[0](data[device_positions])
            teacher_predictions = self.teacher[1](teacher_raw).argmax(1)
            teacher_features = F.normalize(teacher_raw, dim=1).cpu()
            for offset, index in enumerate(range(previous, previous + len(records))):
                self.teacher_predictions[index] = int(teacher_predictions[offset])
                self.teacher_features[index] = teacher_features[offset]
        self.query_rows.append({"step": step, "new_queries": len(records),
                                "pending_before_release": previous - len(self.received)})
        return outputs

    def on_labels_released(self, step, domain, released):
        # Called after oracle.release, before received.update and any optimizer step.
        for index, label in released:
            if index in self.exposure:
                raise ValueError("A feedback event was released twice")
            record = self.oracle.records[index]
            self.exposure[index] = {
                "anchor_index": index, "sample_id": record.sample_id,
                "query_step": record.query_step, "return_step": step,
                "query_domain": record.query_domain, "return_domain": domain,
                "label": int(label), "fresh_draws": 0, "memory_draws": 0,
                "memory_draws_on_return_step": 0,
                "first_use_step": None, "first_use_iteration": None,
                "first_fresh_step": None, "first_fresh_iteration": None,
            }
            if self.oracle.delay > 0:
                self.waiting_fresh[index] = None

    def _record_draws(self, anchor_ids, sampled, role, iteration):
        for position, count in Counter(sampled.tolist()).items():
            event = self.exposure[anchor_ids[position]]
            event[f"{role}_draws"] += count
            if role == "memory" and event["return_step"] == self.stream_step:
                event["memory_draws_on_return_step"] += count
            if event["first_use_step"] is None:
                event["first_use_step"] = self.stream_step
                event["first_use_iteration"] = iteration
            if role == "fresh" and event["first_fresh_step"] is None:
                event["first_fresh_step"] = self.stream_step
                event["first_fresh_iteration"] = iteration

    @torch.enable_grad()
    def cluster_train(self, target_anchors, source_anchors):
        started = time.perf_counter()
        self._record_calibration_events()
        quota = self.trace.old_updates[self.stream_step]
        if quota:
            self._train_quota(quota)
        self.training_seconds += time.perf_counter() - started

    def _train_quota(self, quota):
        indices, data, labels = self._supervised_memory()
        if not indices:
            raise ValueError("Positive supervision quota without any returned label")
        need, relevance, score, probabilities = self._replay_statistics(data, labels)
        lookup = {index: position for position, index in enumerate(indices)}
        fresh_ids = list(self.waiting_fresh) if self.oracle.delay > 0 else []
        fresh_positions = torch.tensor([lookup[index] for index in fresh_ids], dtype=torch.long)
        fresh_set = set(fresh_ids)
        history = torch.tensor([position for position, index in enumerate(indices)
                                if index not in fresh_set], dtype=torch.long)
        # Reserve within the old quota; D=0 never enters this feedback branch.
        fresh_quota = math.ceil(quota * DEFAULTS["feedback_update_fraction"]) if fresh_ids else 0
        order = (torch.randperm(len(fresh_ids), generator=self.fresh_rng)
                 if fresh_ids else torch.empty(0, dtype=torch.long))
        reference = self.source_anchors
        pseudo_count = reference.num_elem() if reference is not None else 0
        reference_alpha = len(indices) / (len(indices) + pseudo_count) if pseudo_count else 1.0
        if pseudo_count and pseudo_count < self.cold_start:
            reference_alpha = min(0.2, reference_alpha)
        size = self.config.train.train_bs
        if size < 2:
            raise ValueError("Feedback minibatch split requires train_bs >= 2")
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.atta.SimATTA.lr, momentum=0.9)
        self.model.train()
        draws = torch.zeros(len(indices))
        fresh_cursor = 0
        for iteration in range(quota):
            feedback_slot = iteration < fresh_quota
            if feedback_slot:
                fresh_size = (max(1, min(size - 1, int(size * DEFAULTS["fresh_minibatch_fraction"])))
                              if len(history) else size)
                slots = (torch.arange(fresh_size) + fresh_cursor) % len(fresh_ids)
                fresh_cursor += fresh_size
                fresh_sampled = fresh_positions[order[slots]]
                if len(history):
                    history_probabilities = probabilities[history]
                    history_probabilities = history_probabilities / history_probabilities.sum()
                    memory_sampled = history[torch.multinomial(
                        history_probabilities, size - fresh_size, replacement=True)]
                else:
                    memory_sampled = torch.empty(0, dtype=torch.long)
                sampled = torch.cat([fresh_sampled, memory_sampled])
                self.feedback_steps += 1
            else:
                fresh_size = 0
                fresh_sampled = torch.empty(0, dtype=torch.long)
                memory_sampled = torch.multinomial(probabilities, size, replacement=True)
                sampled = memory_sampled
                self.replay_steps += 1
            draws.index_add_(0, sampled, torch.ones(size))
            self._record_draws(indices, fresh_sampled, "fresh", iteration)
            self._record_draws(indices, memory_sampled, "memory", iteration)
            losses = F.cross_entropy(self.model(data[sampled].to(self.config.device)),
                                    labels[sampled].to(self.config.device), reduction="none")
            supervised_loss = losses.mean()
            fresh_loss = losses[:fresh_size].mean() if fresh_size else None
            memory_loss = losses[fresh_size:].mean() if fresh_size < size else None
            pseudo_loss, gates = None, None
            if pseudo_count:
                sampled_pseudo = torch.randint(0, pseudo_count, (size,))
                pseudo_labels = reference.target[sampled_pseudo].to(self.config.device).long()
                gates = self._feedback_gates(F.normalize(reference.feats[sampled_pseudo].to(
                    self.config.device), dim=1), pseudo_labels)
                pseudo_loss = (gates * F.cross_entropy(self.model(reference.data[sampled_pseudo].to(
                    self.config.device)), pseudo_labels, reduction="none")).mean()
            if self.oracle.delay == 0:
                true_weight, pseudo_weight = reference_alpha, 1 - reference_alpha
            else:
                pseudo_coefficient = DEFAULTS["pseudo_coefficient"] if pseudo_count else 0.0
                normalizer = DEFAULTS["true_coefficient"] + pseudo_coefficient
                true_weight = DEFAULTS["true_coefficient"] / normalizer
                pseudo_weight = pseudo_coefficient / normalizer
            loss = true_weight * supervised_loss
            if pseudo_count:
                loss = loss + pseudo_weight * pseudo_loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite feedback supervision loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            self.update_rows.append({
                "step": self.stream_step, "iteration": iteration,
                "phase": "new_feedback" if feedback_slot else "full_memory",
                "returned_memory": len(indices), "fresh_pool": len(fresh_ids),
                "source_memory": pseudo_count, "replay_batch": size,
                "fresh_batch": fresh_size, "memory_batch": size - fresh_size,
                "reference_alpha": reference_alpha, "true_weight": true_weight,
                "fresh_weight": true_weight * fresh_size / size,
                "memory_weight": true_weight * (size - fresh_size) / size,
                "pseudo_weight": pseudo_weight,
                "supervised_loss": float(supervised_loss.detach()),
                "fresh_loss": float(fresh_loss.detach()) if fresh_loss is not None else None,
                "memory_loss": float(memory_loss.detach()) if memory_loss is not None else None,
                "pseudo_loss": float(pseudo_loss.detach()) if pseudo_count else None,
                "gate_mean": float(gates.mean()) if pseudo_count else None,
                "loss": float(loss.detach()),
            })
        for index in fresh_ids:
            if self.exposure[index]["fresh_draws"]:
                self.waiting_fresh.pop(index)
        self.replay_rows.append({
            "step": self.stream_step, "returned_memory": len(indices), "updates": quota,
            "feedback_updates": fresh_quota, "full_memory_updates": quota - fresh_quota,
            "need_mean": float(need.mean()), "relevance_mean": float(relevance.mean()),
            "score_mean": float(score.mean()), "coverage": float((draws > 0).sum()) / len(indices),
            "effective_sample_size": float(1 / probabilities.square().sum()),
        })

    @torch.no_grad()
    def on_batch_end(self, step, domain, data, target, outputs, released):
        reference = self.trace.batches[step]
        if domain != reference["domain"] or len(data) != int(reference["samples"]):
            raise ValueError("Current batch differs from the reference")
        truth = target.cpu()
        prediction = outputs.argmax(1).cpu()
        correct = int((prediction == truth).sum())
        previous_returns = int(self.trace.batches[step - 1]["new_returns"]) if step else 0
        self.online_rows.append({
            "step": step, "domain": domain, "samples": len(data), "correct": correct,
            "reference_correct": int(reference["correct"]),
            "paired_net_correct": correct - int(reference["correct"]),
            "previous_batch_new_returns": previous_returns,
            "new_returns": len(released),
            "phase": "before_first_feedback" if self.trace.first_return is None or
            step <= self.trace.first_return else "after_first_feedback",
        })
        updates = self.update_rows[-self.trace.old_updates[step]:] if self.trace.old_updates[step] else []
        if any(row["step"] != step for row in updates):
            raise ValueError("Current batch did not consume its own reference quota")
        self.budget_rows.append({
            "step": step, "reference_updates": self.trace.old_updates[step],
            "actual_updates": len(updates), "new_returns": len(released),
            "feedback_updates": sum(row["phase"] == "new_feedback" for row in updates),
            "full_memory_updates": sum(row["phase"] == "full_memory" for row in updates),
            "waiting_fresh": len(self.waiting_fresh),
        })
        if self.oracle.delay == 0:
            return  # No new forwards, feedback sampling, or modified objective on D=0.
        self.model.eval()
        post = self.model(data).argmax(1).cpu()
        pre_correct, post_correct = prediction == truth, post == truth
        changed = prediction != post
        repaired = int((~pre_correct & post_correct).sum())
        harmed = int((pre_correct & ~post_correct).sum())
        self.probe_rows.append({
            "step": step, "domain": domain, "samples": len(data),
            "new_returns": len(released), "actual_updates": len(updates),
            "pre_correct": correct, "post_correct": int(post_correct.sum()),
            "changed": int(changed.sum()), "repaired": repaired, "harmed": harmed,
            "wrong_to_wrong": int((changed & ~pre_correct & ~post_correct).sum()),
            "net_correct": repaired - harmed,
            "fresh_draws": sum(row["fresh_batch"] for row in updates),
        })

    def run_stream(self, output_dir, progress=None):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.config.device)
        started = time.perf_counter()
        result = super().run_stream(output_dir, progress=progress)
        actual = Counter(row["step"] for row in self.update_rows)
        if (len(self.online_rows) != self.trace.steps or len(self.update_rows) != self.trace.total
                or any(actual[step] != quota for step, quota in enumerate(self.trace.old_updates))):
            raise ValueError("Actual per-batch optimization budget differs from reference")
        self.trace.verify_feedback(self.oracle.audit())
        if self.feedback_steps + self.replay_steps != self.trace.total:
            raise ValueError("Feedback and replay phase accounting differs from total budget")
        total_draws = sum(event["fresh_draws"] + event["memory_draws"] for event in self.exposure.values())
        if (len(self.exposure) != self.oracle.counts()["returned"]
                or total_draws != self.trace.total * self.config.train.train_bs):
            raise ValueError("Returned-label exposure accounting differs from optimizer draws")
        output = Path(output_dir)
        save_rows(output / "budget.csv", self.budget_rows, "step,reference_updates,actual_updates")
        save_rows(output / "feedback_exposure.csv", list(self.exposure.values()),
                  "anchor_index,sample_id,query_step,return_step,query_domain,return_domain,label,"
                  "fresh_draws,memory_draws,memory_draws_on_return_step,first_use_step,"
                  "first_use_iteration,first_fresh_step,first_fresh_iteration")
        save_rows(output / "post_update_probe.csv", self.probe_rows,
                  "step,domain,samples,new_returns,actual_updates,pre_correct,post_correct,"
                  "changed,repaired,harmed,wrong_to_wrong,net_correct,fresh_draws")
        save_rows(output / "online_comparison.csv", self.online_rows,
                  "step,domain,samples,correct,reference_correct,paired_net_correct")
        diagnostics = {
            "reference_updates": self.trace.total,
            "actual_supervised_updates": len(self.update_rows),
            "reference_queries": len(self.trace.records),
            "feedback_updates": self.feedback_steps, "full_memory_updates": self.replay_steps,
            "fresh_draws": sum(event["fresh_draws"] for event in self.exposure.values()),
            "fresh_covered_labels": sum(event["fresh_draws"] > 0 for event in self.exposure.values()),
            "returned_without_fresh_draws": sum(event["fresh_draws"] == 0 for event in self.exposure.values()),
            "true_weight_mean": sum(row["true_weight"] for row in self.update_rows) / self.trace.total
            if self.trace.total else None,
            "pseudo_weight_mean": sum(row["pseudo_weight"] for row in self.update_rows) / self.trace.total
            if self.trace.total else None,
            "paired_stream_gain_pp": 100 * sum(row["paired_net_correct"] for row in self.online_rows)
            / sum(row["samples"] for row in self.online_rows),
        }
        for phase, metric in (("before_first_feedback", "pre_feedback_accuracy_pct"),
                              ("after_first_feedback", "post_feedback_accuracy_pct"),
                              (None, "after_return_online_gain_pp")):
            subset = [row for row in self.online_rows if row["phase"] == phase] if phase else [
                row for row in self.online_rows if row["previous_batch_new_returns"] > 0]
            n = sum(row["samples"] for row in subset)
            key = "correct" if phase else "paired_net_correct"
            diagnostics[metric] = 100 * sum(row[key] for row in subset) / n if n else None
        costs = {
            "training_seconds": self.training_seconds,
            "stream_seconds": time.perf_counter() - started,
            **{key: self.compute[key] for key in (
                "main_forward_calls", "main_forward_images", "main_backward_calls",
                "main_backward_images", "teacher_forward_calls", "teacher_forward_images")},
            "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(self.config.device) / 1024**2
            if torch.cuda.is_available() else None,
        }
        dump_json(output / "costs.json", costs)
        dump_json(output / "budget_result.json", {
            **diagnostics, "per_batch_budget_matched": True, "annotation_schedule_matched": True,
            "d0_batch_correct_match": all(row["correct"] == row["reference_correct"]
                                          for row in self.online_rows) if self.oracle.delay == 0 else None,
        })
        result.update(diagnostics)
        result.update(costs)
        result["method"], result["base_method"] = "FCRA-F", "FCRA"
        return result
