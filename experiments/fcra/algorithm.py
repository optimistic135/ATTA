"""FCRA: pending coverage, complete-label replay and feedback calibration.

Only the inherited oracle holds unreleased labels. This module receives images,
stream IDs and, after release, the oracle's labeled memory.
"""

from collections import deque
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.delayed_feedback.simatta import DelayedSimATTA, write_csv


DEFAULTS = {
    "pending_penalty": 0.25,
    "replay_mix": 0.25,
    "replay_neighbors": 4,
    "epsilon": 1e-12,
    "teacher_cache_size": 512,
    "gate_neighbors": 8,
    "gate_similarity": 0.8,
    "gate_temperature": 0.1,
    "gate_prior": 4.0,
    "reference": "SimATTA low-entropy frozen-teacher pseudo-label memory",
    "memory": "all returned labels; resampled on every gradient update",
}


class FCRA(DelayedSimATTA):
    """Preserve the delayed stream, with PCFR selection and feedback-calibrated replay."""

    def __init__(self, config, delay):
        super().__init__(config, delay)
        self.stream_step = -1
        self.query_rows, self.replay_rows, self.update_rows = [], [], []
        # Fixed-teacher predictions are stored when a sample is queried. The
        # calibration cache contains only events whose true label has returned.
        self.teacher_predictions = {}
        self.teacher_features = {}
        self.calibrated = set()
        self.calibration_events = deque(maxlen=DEFAULTS["teacher_cache_size"])
        self.calibration_rows = []

    @torch.no_grad()
    def select_queries(self, data, stream_ids):
        self.stream_step += 1
        self.current_data = data.detach()
        self.model.eval()
        features = self.model[0](data)
        outputs = self.model[1](features)
        entropy = self.softmax_entropy(outputs)
        scale = math.log(outputs.shape[1])
        threshold = self.config.atta.SimATTA.eh
        eligible = entropy >= threshold
        redundancy = torch.zeros_like(entropy)
        pending = []
        if self.query_anchors is not None:
            pending = [i for i in range(self.query_anchors.num_elem())
                       if i not in self.received]
        if pending:
            # The inherited stream refreshes this cache after every update.
            # Before the first return, the model has not changed.
            pending_features = self.query_anchors.feats[pending].to(features.device)
            similarities = F.normalize(features, dim=1) @ F.normalize(
                pending_features, dim=1).T
            redundancy = similarities.max(1).values.clamp(0, 1)
        score = entropy / scale - DEFAULTS["pending_penalty"] * redundancy
        keep = eligible & (score >= threshold / scale)
        previous = self.query_anchors.num_elem() if self.query_anchors is not None else 0
        if bool(keep.any()):
            # Original entropy filtering and incremental KMeans now see only
            # candidates not covered too strongly by outstanding requests.
            super().select_queries(data[keep], stream_ids[keep.cpu()])
        current = self.query_anchors.num_elem() if self.query_anchors is not None else 0
        if current > previous:
            # Score only the samples newly admitted by the original selector.
            # This is still before oracle release and before any update.
            self.teacher.eval()
            queried = self.query_anchors.data[previous:current].to(self.config.device)
            teacher_raw = self.teacher[0](queried)
            teacher_features = F.normalize(teacher_raw, dim=1)
            teacher_predictions = self.teacher[1](teacher_raw).argmax(1)
            for offset, index in enumerate(range(previous, current)):
                self.teacher_predictions[index] = int(teacher_predictions[offset])
                self.teacher_features[index] = teacher_features[offset].cpu()
        self.query_rows.append({
            "step": self.stream_step, "pending_before_release": len(pending),
            "eligible_before": int(eligible.sum()), "eligible_after": int(keep.sum()),
            "suppressed": int((eligible & ~keep).sum()), "new_queries": current - previous,
            "mean_redundancy": float(redundancy.mean()),
        })
        # Always return the full batch's prediction, including screened images.
        return outputs

    def _record_calibration_events(self):
        """Join newly returned truth with the teacher prediction saved at query."""
        for index in sorted(self.received):
            if index in self.calibrated:
                continue
            self.calibrated.add(index)
            prediction = self.teacher_predictions.get(index)
            feature = self.teacher_features.get(index)
            if prediction is None or feature is None:
                continue
            label = int(self.received[index])
            correct = int(prediction == label)
            self.calibration_events.append((prediction, feature, correct))
            record = self.oracle.records[index]
            self.calibration_rows.append({
                "step": self.stream_step, "anchor_index": index,
                "sample_id": int(self.query_anchors.target[index]),
                "teacher_prediction": prediction, "label": label, "correct": correct,
                "cache_size": len(self.calibration_events),
                "query_step": record.query_step, "return_step": record.return_step,
                "query_domain": record.query_domain,
                "return_domain": record.return_domain,
            })

    def _supervised_memory(self):
        """Build the complete, never-pruned memory of released labels."""
        indices = sorted(self.received)
        data = self.query_anchors.data[indices].cpu()
        labels = torch.tensor([self.received[i] for i in indices], dtype=torch.long)
        return indices, data, labels

    @torch.no_grad()
    def _replay_statistics(self, data, labels):
        """Score every returned sample at the current, pre-update model state."""
        self.model.eval()
        needs, features = [], []
        size = self.config.atta.batch_size
        for start in range(0, len(data), size):
            images = data[start:start + size].to(self.config.device)
            targets = labels[start:start + size].to(self.config.device)
            encoded = self.model[0](images)
            probabilities = self.model[1](encoded).softmax(1)
            need = 1 - probabilities.gather(1, targets[:, None]).squeeze(1)
            needs.append(need.cpu())
            features.append(F.normalize(encoded, dim=1).cpu())
        need = torch.cat(needs)
        memory_features = torch.cat(features)
        # Relevance uses a local neighbourhood of the current batch instead of
        # one batch centre, so different current classes stay distinguishable.
        current_features = F.normalize(self.model[0](self.current_data), dim=1).cpu()
        affinity = ((1 + memory_features @ current_features.T) / 2).clamp(0, 1)
        neighbours = min(DEFAULTS["replay_neighbors"], current_features.shape[0])
        relevance = affinity.topk(neighbours, dim=1).values.mean(1)
        score = need * relevance
        if not bool(torch.isfinite(score).all()):
            raise FloatingPointError("FCRA encountered a non-finite replay score")
        count = len(score)
        uniform = torch.full((count,), 1.0 / count)
        epsilon = DEFAULTS["epsilon"]
        if float(score.sum()) <= epsilon:
            probabilities = uniform
        else:
            probabilities = (
                (1 - DEFAULTS["replay_mix"]) * uniform
                + DEFAULTS["replay_mix"] * (score + epsilon) / (score + epsilon).sum()
            )
        return need, relevance, score, probabilities

    @torch.no_grad()
    def _feedback_gates(self, features, labels):
        """Local reliability of teacher pseudo labels from returned truth.

        features must be normalized teacher features. Events are checked for
        the same teacher class and cosine similarity >= 0.8; at most the eight
        most similar eligible events contribute, weighted by exp((cos-1)/tau).
        """
        count = len(self.calibration_events)
        if count == 0:
            return torch.ones(len(features), dtype=features.dtype, device=features.device)
        events = list(self.calibration_events)
        cache_features = torch.stack([event[1] for event in events]).to(features.device)
        cache_predictions = torch.tensor(
            [event[0] for event in events], dtype=torch.long, device=features.device)
        cache_correct = torch.tensor(
            [event[2] for event in events], dtype=features.dtype, device=features.device)
        similarity = features @ cache_features.T
        eligible = ((cache_predictions.unsqueeze(0) == labels.unsqueeze(1))
                    & (similarity >= DEFAULTS["gate_similarity"]))
        masked = similarity.masked_fill(~eligible, -1.0)
        neighbours = min(DEFAULTS["gate_neighbors"], count)
        values, positions = masked.topk(neighbours, dim=1)
        valid = values >= DEFAULTS["gate_similarity"]
        weights = (torch.exp((values - 1) / DEFAULTS["gate_temperature"]) * valid)
        evidence = (weights * cache_correct[positions]).sum(1)
        denominator = weights.sum(1)
        gates = ((DEFAULTS["gate_prior"] + evidence)
                 / (DEFAULTS["gate_prior"] + denominator))
        return torch.where(valid.any(1), gates, torch.ones_like(gates))

    @torch.enable_grad()
    def cluster_train(self, target_anchors, source_anchors):
        # Calibration events must exist before the student consumes the labels.
        self._record_calibration_events()
        indices, memory_data, memory_labels = self._supervised_memory()
        if not indices:
            return
        need, relevance, score, probabilities = self._replay_statistics(
            memory_data, memory_labels)
        reference = self.source_anchors
        has_pseudo = reference is not None and reference.num_elem() > 0
        pseudo_count = reference.num_elem() if has_pseudo else 0
        returned = len(indices)
        alpha = returned / (returned + pseudo_count) if has_pseudo else 1.0
        if has_pseudo and pseudo_count < self.cold_start:
            alpha = min(0.2, alpha)
        batch_size = self.config.train.train_bs
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=self.config.atta.SimATTA.lr, momentum=0.9)
        draws = torch.zeros(returned)
        loss_window, tolerance, lowest_loss = [], 0, float("inf")
        iteration = 0
        while True:
            sampled = torch.multinomial(probabilities, batch_size, replacement=True)
            images = memory_data[sampled].to(self.config.device)
            labels = memory_labels[sampled].to(self.config.device)
            draws.index_add_(0, sampled, torch.ones(len(sampled)))
            losses = F.cross_entropy(self.model(images), labels, reduction="none")
            supervised_loss = losses.mean()
            pseudo_loss, gates = None, None
            if has_pseudo:
                pseudo_indices = torch.randint(0, pseudo_count, (batch_size,))
                pseudo_images = reference.data[pseudo_indices].to(self.config.device)
                pseudo_labels = reference.target[pseudo_indices].to(
                    self.config.device).long()
                teacher_features = F.normalize(
                    reference.feats[pseudo_indices].to(self.config.device), dim=1)
                gates = self._feedback_gates(teacher_features, pseudo_labels)
                pseudo_losses = F.cross_entropy(
                    self.model(pseudo_images), pseudo_labels, reduction="none")
                # Fixed denominator: weak gates lower the absolute contribution
                # instead of renormalizing the pseudo-label batch.
                pseudo_loss = (gates * pseudo_losses).mean()
                loss = alpha * supervised_loss + (1 - alpha) * pseudo_loss
            else:
                # SimATTA's empty pseudo memory falls back to truth replay.
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
            # Exact PACS SimATTA stopping structure. Its steps=30 is NOT a
            # fixed-update cap on PACS; the utility experiment uses another loop.
            loss_value = float(losses.mean().detach())
            if len(loss_window) < self.config.atta.SimATTA.stop_tol:
                loss_window.append(loss_value)
            else:
                mean_loss = np.mean(loss_window)
                tolerance += 1
                if mean_loss < lowest_loss:
                    lowest_loss, tolerance = mean_loss, 0
                if tolerance > 5:
                    break
                loss_window = []
            iteration += 1
        distinct = int((draws > 0).sum())
        self.replay_rows.append({
            "step": self.stream_step, "returned_memory": returned,
            "updates": iteration, "source_memory": pseudo_count,
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

    def run_stream(self, output_dir, progress=None):
        result = super().run_stream(output_dir, progress=progress)
        output = Path(output_dir)
        write_csv(output / "query_diagnostics.csv", self.query_rows)
        if self.calibration_rows:
            write_csv(output / "calibration.csv", self.calibration_rows)
        else:
            (output / "calibration.csv").write_text(
                "step,anchor_index,sample_id,teacher_prediction,label,correct,cache_size,"
                "query_step,return_step,query_domain,return_domain\n", encoding="utf-8")
        if self.replay_rows:
            write_csv(output / "replay.csv", self.replay_rows)
        else:
            (output / "replay.csv").write_text(
                "step,returned_memory,updates,source_memory,need_mean,need_max,"
                "relevance_mean,score_mean,score_max,probability_min,probability_max,"
                "effective_sample_size,distinct_sampled,coverage,max_draws\n",
                encoding="utf-8")
        if self.update_rows:
            write_csv(output / "updates.csv", self.update_rows)
        else:
            (output / "updates.csv").write_text(
                "step,iteration,returned_memory,replay_batch,source_memory,alpha,has_pseudo,"
                "supervised_loss,pseudo_loss,gate_mean,gate_min,gate_max,"
                "drawn_probability_mean,calibration_cache\n", encoding="utf-8")
        result["method"] = "FCRA"
        return result
