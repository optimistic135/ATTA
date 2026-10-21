"""FCRA-R: supervised-anchor retrieval and local residual propagation."""

from collections import Counter, deque
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from experiments.delayed_feedback.simatta import write_csv
from experiments.fcra.algorithm import DEFAULTS as FCRA_DEFAULTS, FCRA


DEFAULTS = {
    **FCRA_DEFAULTS,
    "retrieval_neighbors": 8,
    "retrieval_similarity": 0.8,
    "residual_temperature": 0.1,
    "residual_prior_count": 4.0,
    "candidate_betas": [0.25, 0.50],
    "feedback_cache_size": 512,
    "feedback_similarity": 0.8,
    "feedback_temperature": 0.1,
    "feedback_age": 50.0,
    "feedback_prior_count": 4.0,
    "feedback_eta": 1.0,
    "feedback_prior": [0.6, 0.3, 0.1],
    "gain_clip": 1.0,
    "score_epsilon": 1e-12,
    "consistency_epsilon": 1e-12,
}


RESIDUAL_HEADER = (
    "step,domain,samples,main_accuracy_pct,final_accuracy_pct,candidate1_accuracy_pct,"
    "candidate2_accuracy_pct,main_candidate1_disagreement_pct,"
    "main_candidate2_disagreement_pct,main_final_disagreement_pct,"
    "candidate1_candidate2_disagreement_pct,changed,changed_correct,changed_incorrect,"
    "support_mean,support_zero_pct,rho_mean,xi_mean,correction_norm_mean,"
    "retrieved_mean,feedback_neighbors_mean,feedback_events,w0,w1,w2,propagation_seconds"
)

FEEDBACK_HEADER = (
    "step,anchor_index,sample_id,query_step,return_step,delay,query_domain,"
    "return_domain,label,p0_true,p1_true,p2_true,output_true,p0_pred,p1_pred,"
    "p2_pred,output_pred,correct0,correct1,correct2,output_correct,"
    "loss0,loss1,loss2,gain0,gain1,gain2,w0,w1,w2"
)


class ResidualFCRA(FCRA):
    """Keep FCRA's supervised path and add supervised-anchor residual correction."""

    def __init__(self, config, delay, trace=None):
        super().__init__(config, delay)
        self.trace = trace
        self.reference_mode = trace is not None
        if self.reference_mode:
            structure = [
                (self.experiment.domains[env],
                 min(config.atta.batch_size, len(loader.sampler) - start))
                for env in config.dataset.test_envs[1:]
                for loader in [self.fast_loader[env]]
                for start in range(0, len(loader.sampler), config.atta.batch_size)
            ]
            expected = [(row["domain"], int(row["samples"])) for row in trace.batches]
            if structure != expected:
                raise ValueError("Current dataset stream and reference batch structure differ")
        self.residual_enabled = delay > 0
        self.feedback_events = deque(maxlen=DEFAULTS["feedback_cache_size"])
        self.scored_queries = set()
        self.query_candidate_predictions = {}
        self.residual_rows, self.residual_event_rows = [], []
        self.propagation_seconds = 0.0
        self.training_seconds = 0.0
        self.residual_batches = 0
        self.anchor_forward_calls = 0
        self.anchor_forward_images = 0
        self.main_forward_calls = 0
        self.teacher_forward_calls = 0
        self.main_model_calls = 0
        self.teacher_model_calls = 0
        self.current_candidate_probs = None
        self.current_output_probs = None
        self.current_weights = None
        self.current_stats = None
        self.model.register_forward_hook(self._count_main_model_forward)
        self.teacher.register_forward_hook(self._count_teacher_model_forward)
        self.model[0].register_forward_hook(self._count_main_forward)
        self.teacher[0].register_forward_hook(self._count_teacher_forward)

    def _count_main_model_forward(self, module, inputs, output):
        self.main_model_calls += 1

    def _count_teacher_model_forward(self, module, inputs, output):
        self.teacher_model_calls += 1

    def _count_main_forward(self, module, inputs, output):
        self.main_forward_calls += 1

    def _count_teacher_forward(self, module, inputs, output):
        self.teacher_forward_calls += 1

    @torch.no_grad()
    def select_queries(self, data, stream_ids):
        if self.reference_mode:
            return self._select_queries_from_trace(data, stream_ids)
        previous = self.query_anchors.num_elem() if self.query_anchors is not None else 0
        outputs = super().select_queries(data, stream_ids)
        if not self.residual_enabled:
            return outputs
        self.model.eval()
        self.teacher.eval()
        source_raw = self.teacher[0](data)
        source_features = F.normalize(source_raw, dim=1)
        corrected = self._residual_correct(data, stream_ids, outputs.softmax(1),
                                          source_features)
        current = self.query_anchors.num_elem() if self.query_anchors is not None else 0
        if current > previous:
            self._save_query_predictions(previous, current, stream_ids, source_features)
        return corrected

    @torch.no_grad()
    def _select_queries_from_trace(self, data, stream_ids):
        """Replay recorded FCRA query IDs and timing exactly."""
        self.stream_step += 1
        self.current_data = data.detach()
        self.model.eval()
        features = self.model[0](data)
        outputs = self.model[1](features)
        previous = self.query_anchors.num_elem() if self.query_anchors is not None else 0
        records = self.trace.queries[self.stream_step]
        expected = list(range(previous, previous + len(records)))
        if [int(row["anchor_index"]) for row in records] != expected:
            raise ValueError("Reference anchor insertion order changed")
        if any(int(row["query_step"]) != self.stream_step for row in records):
            raise ValueError("Reference query step differs from the stream clock")
        first_stream_id = int(stream_ids[0])
        positions = torch.tensor([int(row["sample_id"]) - first_stream_id
                                  for row in records], dtype=torch.long)
        if len(positions) and (int(positions.min()) < 0 or int(positions.max()) >= len(data)):
            raise ValueError("Reference query is outside the current batch")
        if records:
            device_positions = positions.to(data.device)
            self.query_anchors = self.update_anchors(
                self.query_anchors, data[device_positions].cpu(), stream_ids[positions],
                features[device_positions].cpu(), torch.ones(len(positions)))
            self.teacher.eval()
            teacher_raw = self.teacher[0](data[device_positions])
            teacher_features = F.normalize(teacher_raw, dim=1)
            teacher_predictions = self.teacher[1](teacher_raw).argmax(1)
            for offset, index in enumerate(expected):
                self.teacher_predictions[index] = int(teacher_predictions[offset])
                self.teacher_features[index] = teacher_features[offset].cpu()
        current = self.query_anchors.num_elem() if self.query_anchors is not None else 0
        pending = sum(1 for index in range(previous) if index not in self.received)
        self.query_rows.append({
            "step": self.stream_step, "pending_before_release": pending,
            "eligible_before": None, "eligible_after": None, "suppressed": None,
            "new_queries": current - previous, "mean_redundancy": None,
        })
        if not self.residual_enabled:
            return outputs
        self.teacher.eval()
        source_raw = self.teacher[0](data)
        source_features = F.normalize(source_raw, dim=1)
        corrected = self._residual_correct(data, stream_ids, outputs.softmax(1),
                                          source_features)
        if current > previous:
            self._save_query_predictions(previous, current, stream_ids, source_features)
        return corrected

    def _anchor_features(self, indices, feature_dim):
        features = []
        for index in indices:
            feature = self.teacher_features.get(index)
            if feature is None:
                image = self.query_anchors.data[index].unsqueeze(0).to(self.config.device)
                feature = F.normalize(self.teacher[0](image), dim=1)[0].cpu()
                self.teacher_features[index] = feature
            features.append(feature)
        if not features:
            return torch.empty((0, feature_dim), dtype=torch.float32)
        return torch.stack(features)

    @torch.no_grad()
    def _anchor_probabilities(self, images):
        size = self.config.atta.batch_size
        probabilities = []
        for start in range(0, len(images), size):
            batch = images[start:start + size].to(self.config.device)
            probabilities.append(self.model(batch).softmax(1))
            self.anchor_forward_calls += 1
            self.anchor_forward_images += len(batch)
        return torch.cat(probabilities, dim=0)

    @staticmethod
    def _simplex_projection(values):
        rows, classes = values.shape
        ordered, _ = torch.sort(values, dim=1, descending=True)
        cumulative = torch.cumsum(ordered, dim=1) - 1.0
        indices = torch.arange(1, classes + 1, device=values.device, dtype=values.dtype)
        condition = ordered - cumulative / indices > 0
        rho = condition.sum(dim=1) - 1
        theta = cumulative[torch.arange(rows, device=values.device), rho] / (
            rho + 1).to(values.dtype)
        return (values - theta.unsqueeze(1)).clamp_min(0)

    @torch.no_grad()
    def _feedback_weights(self, features, step):
        prior = torch.tensor(DEFAULTS["feedback_prior"], dtype=features.dtype,
                             device=features.device)
        count = len(self.feedback_events)
        if count == 0:
            return prior.unsqueeze(0).expand(len(features), -1), torch.zeros(
                len(features), dtype=features.dtype, device=features.device)
        events = list(self.feedback_events)
        cache_features = torch.stack([event[0] for event in events]).to(features.device)
        return_steps = torch.tensor([event[1] for event in events], dtype=features.dtype,
                                    device=features.device)
        gains = torch.stack([event[2] for event in events]).to(features.device)
        similarity = features @ cache_features.T
        eligible = similarity >= DEFAULTS["feedback_similarity"]
        age = torch.exp(-(step - return_steps).clamp_min(0) / DEFAULTS["feedback_age"])
        weights = (eligible.to(features.dtype)
                   * torch.exp((similarity - 1.0) / DEFAULTS["feedback_temperature"])
                   * age.unsqueeze(0))
        denominator = DEFAULTS["feedback_prior_count"] + weights.sum(1)
        evidence = (weights @ gains) / denominator.unsqueeze(1)
        logits = torch.log(prior).unsqueeze(0) + DEFAULTS["feedback_eta"] * evidence
        return torch.softmax(logits, dim=1), eligible.sum(1).to(features.dtype)

    @torch.no_grad()
    def _residual_correct(self, data, stream_ids, main_probs, source_features):
        started = time.perf_counter()
        self.model.eval()
        self.teacher.eval()
        batch, classes = main_probs.shape
        anchor_indices = sorted(self.received)
        if anchor_indices:
            anchor_labels = torch.tensor(
                [self.received[index] for index in anchor_indices], dtype=torch.long)
            anchor_features = self._anchor_features(
                anchor_indices, source_features.shape[1]).to(self.config.device)
            anchor_images = self.query_anchors.data[anchor_indices]
            anchor_probs = self._anchor_probabilities(anchor_images)
            labels_device = anchor_labels.to(anchor_probs.device)
            one_hot = F.one_hot(labels_device, num_classes=classes).to(anchor_probs.dtype)
            residuals = one_hot - anchor_probs
            similarity = source_features @ anchor_features.T
            neighbours = min(DEFAULTS["retrieval_neighbors"], len(anchor_indices))
            values, positions = similarity.topk(neighbours, dim=1)
            eligible = values >= DEFAULTS["retrieval_similarity"]
            weights = (torch.exp((values - 1.0) / DEFAULTS["residual_temperature"])
                       * eligible.to(values.dtype))
            support = weights.sum(1)
            gathered = residuals[positions]
            denominator = support.clamp_min(DEFAULTS["consistency_epsilon"])
            direction = (weights[:, :, None] * gathered).sum(1) / denominator[:, None]
            magnitude = ((weights * gathered.norm(dim=2)).sum(1)
                         / denominator)
            consistency = direction.norm(dim=1) / (
                DEFAULTS["consistency_epsilon"] + magnitude)
            consistency = consistency.clamp(0, 1)
            rho = support / (DEFAULTS["residual_prior_count"] + support)
            has_support = support > 0
            direction = torch.where(has_support[:, None], direction,
                                    torch.zeros_like(direction))
            consistency = torch.where(has_support, consistency,
                                      torch.zeros_like(consistency))
            correction = rho[:, None] * consistency[:, None] * direction
            retrieved = eligible.sum(1).to(main_probs.dtype)
        else:
            support = torch.zeros(batch, dtype=main_probs.dtype, device=main_probs.device)
            rho = support
            consistency = support
            correction = torch.zeros_like(main_probs)
            retrieved = support
            positions = None
        active = (support > 0) & (correction.norm(dim=1) > DEFAULTS["consistency_epsilon"])
        candidates = [main_probs]
        for beta in DEFAULTS["candidate_betas"]:
            projected = self._simplex_projection(main_probs + beta * correction)
            projected = torch.where(
                torch.isfinite(projected).all(dim=1, keepdim=True),
                projected, main_probs)
            candidates.append(torch.where(active[:, None], projected, main_probs))
        candidates = torch.stack(candidates, dim=1)
        feedback_weights, feedback_neighbours = self._feedback_weights(
            source_features, self.stream_step)
        output = (feedback_weights[:, :, None] * candidates).sum(1)
        stats = {
            "support_mean": float(support.mean()),
            "support_zero_pct": 100.0 * float((support <= 0).float().mean()),
            "rho_mean": float(rho.mean()),
            "xi_mean": float(consistency.mean()),
            "correction_norm_mean": float(correction.norm(dim=1).mean()),
            "retrieved_mean": float(retrieved.mean()),
            "feedback_neighbors_mean": float(feedback_neighbours.mean()),
            "feedback_events": len(self.feedback_events),
            "w0": float(feedback_weights[:, 0].mean()),
            "w1": float(feedback_weights[:, 1].mean()),
            "w2": float(feedback_weights[:, 2].mean()),
            "propagation_seconds": time.perf_counter() - started,
        }
        self.current_candidate_probs = candidates.detach().cpu()
        self.current_output_probs = output.detach().cpu()
        self.current_weights = feedback_weights.detach().cpu()
        self.current_stats = stats
        self.propagation_seconds += stats["propagation_seconds"]
        self.residual_batches += 1
        return output.to(main_probs.device)

    def _save_query_predictions(self, previous, current, stream_ids, source_features):
        if not self.residual_enabled or current <= previous:
            return
        first_stream_id = int(stream_ids[0])
        batch = len(self.current_output_probs)
        for index in range(previous, current):
            sample_id = int(self.query_anchors.target[index])
            local = sample_id - first_stream_id
            if not 0 <= local < batch:
                continue
            self.query_candidate_predictions[index] = {
                "sample_id": sample_id,
                "query_step": self.stream_step,
                "candidates": self.current_candidate_probs[local].clone(),
                "output": self.current_output_probs[local].clone(),
                "weights": self.current_weights[local].clone(),
                "feature": source_features[local].detach().cpu(),
            }

    @torch.no_grad()
    def on_labels_released(self, step, domain, released):
        """Score sealed candidate predictions before labels enter memory."""
        if not self.residual_enabled or not released:
            return
        epsilon = DEFAULTS["score_epsilon"]
        for index, label in released:
            if index in self.scored_queries:
                continue
            self.scored_queries.add(index)
            saved = self.query_candidate_predictions.get(index)
            if saved is None:
                continue
            probabilities = [float(candidate[label]) for candidate in saved["candidates"]]
            losses = [-math.log(max(probability, epsilon)) for probability in probabilities]
            gains = [min(DEFAULTS["gain_clip"],
                         max(-DEFAULTS["gain_clip"], losses[0] - loss))
                     for loss in losses]
            self.feedback_events.append((
                saved["feature"], step,
                torch.tensor(gains, dtype=torch.float32)))
            record = self.oracle.records[index]
            predicted = [int(candidate.argmax()) for candidate in saved["candidates"]]
            output_prediction = int(saved["output"].argmax())
            self.residual_event_rows.append({
                "step": step, "anchor_index": index,
                "sample_id": int(saved["sample_id"]),
                "query_step": record.query_step, "return_step": record.return_step,
                "delay": (record.return_step - record.query_step
                          if record.return_step is not None else self.oracle.delay),
                "query_domain": record.query_domain, "return_domain": record.return_domain,
                "label": int(label),
                "p0_true": probabilities[0], "p1_true": probabilities[1],
                "p2_true": probabilities[2],
                "output_true": float(saved["output"][label]),
                "p0_pred": predicted[0], "p1_pred": predicted[1],
                "p2_pred": predicted[2], "output_pred": output_prediction,
                "correct0": int(predicted[0] == label),
                "correct1": int(predicted[1] == label),
                "correct2": int(predicted[2] == label),
                "output_correct": int(output_prediction == label),
                "loss0": losses[0], "loss1": losses[1], "loss2": losses[2],
                "gain0": gains[0], "gain1": gains[1], "gain2": gains[2],
                "w0": float(saved["weights"][0]),
                "w1": float(saved["weights"][1]),
                "w2": float(saved["weights"][2]),
            })

    @torch.enable_grad()
    def _train_supervised_quota(self, quota):
        """Run exactly the recorded number of FCRA supervised updates."""
        if quota <= 0:
            return
        indices, data, labels = self._supervised_memory()
        if not indices:
            return
        need, relevance, score, probabilities = self._replay_statistics(data, labels)
        reference = self.source_anchors
        has_pseudo = reference is not None and reference.num_elem() > 0
        pseudo_count = reference.num_elem() if has_pseudo else 0
        returned = len(indices)
        alpha = returned / (returned + pseudo_count) if has_pseudo else 1.0
        if has_pseudo and pseudo_count < self.cold_start:
            alpha = min(0.2, alpha)
        size = self.config.train.train_bs
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=self.config.atta.SimATTA.lr, momentum=0.9)
        draws = torch.zeros(returned)
        for iteration in range(quota):
            sampled = torch.multinomial(probabilities, size, replacement=True)
            images = data[sampled].to(self.config.device)
            targets = labels[sampled].to(self.config.device)
            draws.index_add_(0, sampled, torch.ones(size))
            losses = F.cross_entropy(self.model(images), targets, reduction="none")
            supervised_loss = losses.mean()
            pseudo_loss, gates = None, None
            if has_pseudo:
                pseudo_indices = torch.randint(0, pseudo_count, (size,))
                pseudo_labels = reference.target[pseudo_indices].to(
                    self.config.device).long()
                teacher_features = F.normalize(
                    reference.feats[pseudo_indices].to(self.config.device), dim=1)
                gates = self._feedback_gates(teacher_features, pseudo_labels)
                pseudo_loss = (gates * F.cross_entropy(
                    self.model(reference.data[pseudo_indices].to(self.config.device)),
                    pseudo_labels, reduction="none")).mean()
                loss = alpha * supervised_loss + (1 - alpha) * pseudo_loss
            else:
                loss = supervised_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            self.update_rows.append({
                "step": self.stream_step, "iteration": iteration,
                "returned_memory": returned, "replay_batch": int(len(sampled)),
                "source_memory": pseudo_count, "alpha": alpha,
                "has_pseudo": has_pseudo,
                "supervised_loss": float(supervised_loss.detach()),
                "pseudo_loss": float(pseudo_loss.detach()) if has_pseudo else None,
                "gate_mean": float(gates.mean()) if has_pseudo else None,
                "gate_min": float(gates.min()) if has_pseudo else None,
                "gate_max": float(gates.max()) if has_pseudo else None,
                "drawn_probability_mean": float(probabilities[sampled].mean()),
                "calibration_cache": len(self.calibration_events),
            })
        distinct = int((draws > 0).sum())
        self.replay_rows.append({
            "step": self.stream_step, "returned_memory": returned,
            "updates": quota, "source_memory": pseudo_count,
            "need_mean": float(need.mean()), "need_max": float(need.max()),
            "relevance_mean": float(relevance.mean()),
            "score_mean": float(score.mean()), "score_max": float(score.max()),
            "probability_min": float(probabilities.min()),
            "probability_max": float(probabilities.max()),
            "effective_sample_size": float(1 / probabilities.square().sum()),
            "distinct_sampled": distinct,
            "coverage": distinct / returned,
            "max_draws": int(draws.max()),
        })

    def cluster_train(self, target_anchors, source_anchors):
        started = time.perf_counter()
        if self.reference_mode:
            self._record_calibration_events()
            self._train_supervised_quota(self.trace.old_updates[self.stream_step])
            result = None
        else:
            result = super().cluster_train(target_anchors, source_anchors)
        self.training_seconds += time.perf_counter() - started
        return result

    def on_batch_end(self, step, domain, data, target, outputs, released):
        if not self.residual_enabled or self.current_candidate_probs is None:
            return
        target = target.cpu()
        candidates = self.current_candidate_probs.detach().cpu()
        output = self.current_output_probs.detach().cpu()
        if len(target) != len(candidates):
            raise ValueError("Residual diagnostics received a mismatched batch")
        p0, p1, p2 = candidates[:, 0], candidates[:, 1], candidates[:, 2]
        pred0, pred1, pred2, pred_out = (p0.argmax(1), p1.argmax(1),
                                         p2.argmax(1), output.argmax(1))
        correct0, correct1, correct2, correct_out = (
            pred0 == target, pred1 == target, pred2 == target, pred_out == target)
        changed = pred_out != pred0
        stats = self.current_stats
        self.residual_rows.append({
            "step": step, "domain": domain, "samples": len(target),
            "main_accuracy_pct": 100.0 * float(correct0.float().mean()),
            "final_accuracy_pct": 100.0 * float(correct_out.float().mean()),
            "candidate1_accuracy_pct": 100.0 * float(correct1.float().mean()),
            "candidate2_accuracy_pct": 100.0 * float(correct2.float().mean()),
            "main_candidate1_disagreement_pct": 100.0 * float(
                (pred0 != pred1).float().mean()),
            "main_candidate2_disagreement_pct": 100.0 * float(
                (pred0 != pred2).float().mean()),
            "main_final_disagreement_pct": 100.0 * float(changed.float().mean()),
            "candidate1_candidate2_disagreement_pct": 100.0 * float(
                (pred1 != pred2).float().mean()),
            "changed": int(changed.sum()),
            "changed_correct": int((changed & correct_out).sum()),
            "changed_incorrect": int((changed & ~correct_out).sum()),
            "support_mean": stats["support_mean"],
            "support_zero_pct": stats["support_zero_pct"],
            "rho_mean": stats["rho_mean"], "xi_mean": stats["xi_mean"],
            "correction_norm_mean": stats["correction_norm_mean"],
            "retrieved_mean": stats["retrieved_mean"],
            "feedback_neighbors_mean": stats["feedback_neighbors_mean"],
            "feedback_events": stats["feedback_events"],
            "w0": stats["w0"], "w1": stats["w1"], "w2": stats["w2"],
            "propagation_seconds": stats["propagation_seconds"],
        })

    def run_stream(self, output_dir, progress=None):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.config.device)
        started = time.perf_counter()
        result = super().run_stream(output_dir, progress=progress)
        if self.reference_mode:
            self.trace.verify_feedback(self.oracle.audit())
            actual_by_step = Counter(row["step"] for row in self.update_rows)
            mismatches = [
                (step, actual_by_step.get(step, 0), expected)
                for step, expected in enumerate(self.trace.old_updates)
                if actual_by_step.get(step, 0) != expected
            ]
            if mismatches:
                raise ValueError(
                    "Per-step supervised update counts differ from reference: "
                    f"{mismatches[:3]}")
            actual_updates = len(self.update_rows)
            if actual_updates != self.trace.total:
                raise ValueError(
                    f"Actual supervised updates {actual_updates} differ from reference "
                    f"{self.trace.total}")
            result["reference_updates"] = self.trace.total
            result["reference_queries"] = len(self.trace.records)
            result["actual_supervised_updates"] = actual_updates
        output = Path(output_dir)
        if self.residual_rows:
            write_csv(output / "residual.csv", self.residual_rows)
        else:
            (output / "residual.csv").write_text(
                RESIDUAL_HEADER + "\n", encoding="utf-8")
        if self.residual_event_rows:
            write_csv(output / "residual_feedback.csv", self.residual_event_rows)
        else:
            (output / "residual_feedback.csv").write_text(
                FEEDBACK_HEADER + "\n", encoding="utf-8")
        batches = len(self.residual_rows)
        costs = {
            "residual_enabled": self.residual_enabled,
            "total_seconds": time.perf_counter() - started,
            "propagation_seconds": self.propagation_seconds,
            "training_seconds": self.training_seconds,
            "residual_batches": self.residual_batches,
            "mean_retrieved_candidates": (
                sum(row["retrieved_mean"] for row in self.residual_rows) / batches
                if batches else 0.0),
            "mean_feedback_neighbors": (
                sum(row["feedback_neighbors_mean"] for row in self.residual_rows) / batches
                if batches else 0.0),
            "mean_support": (
                sum(row["support_mean"] for row in self.residual_rows) / batches
                if batches else 0.0),
            "mean_correction_norm": (
                sum(row["correction_norm_mean"] for row in self.residual_rows) / batches
                if batches else 0.0),
            "main_encoder_forward_calls": self.main_forward_calls,
            "teacher_encoder_forward_calls": self.teacher_forward_calls,
            "main_model_forward_calls": self.main_model_calls,
            "teacher_model_forward_calls": self.teacher_model_calls,
            "anchor_forward_calls": self.anchor_forward_calls,
            "anchor_forward_images": self.anchor_forward_images,
            "peak_cuda_memory_mb": (torch.cuda.max_memory_allocated(self.config.device)
                                    / (1024 ** 2) if torch.cuda.is_available() else None),
            "peak_cuda_reserved_mb": (torch.cuda.max_memory_reserved(self.config.device)
                                      / (1024 ** 2) if torch.cuda.is_available() else None),
        }
        (output / "costs.json").write_text(
            json.dumps(costs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result.update(costs)
        result["method"] = "FCRA-R"
        result["base_method"] = "FCRA"
        return result
