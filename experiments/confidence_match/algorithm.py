"""Filter existing feedback neighbors by normalized teacher entropy, then apply the unchanged gate."""

import math

import torch
from torch.nn import functional as F

from experiments.fcra.algorithm import DEFAULTS
from experiments.vlcs_calibration.algorithm import ControlledCalibration
from experiments.vlcs_calibration.results import write_csv


class ConfidenceMatch(ControlledCalibration):
    def __init__(self, config, trace, entropy_gap=0.05):
        super().__init__(config, trace, "calibration_only")
        self.entropy_gap = entropy_gap
        self.query_states, self.pseudo_states = {}, []
        self.state_rows, self.match_rows = [], []

    @staticmethod
    def entropy_state(logits):
        return -(logits.softmax(1) * logits.log_softmax(1)).sum(1) / math.log(logits.shape[1])

    @torch.no_grad()
    def select_queries(self, data, stream_ids):
        previous = self.query_anchors.num_elem() if self.query_anchors is not None else 0
        outputs = super().select_queries(data, stream_ids)
        current = self.query_anchors.num_elem() if self.query_anchors is not None else 0
        if current > previous:
            self.teacher.eval()
            images = self.query_anchors.data[previous:current].to(self.config.device)
            states = self.entropy_state(self.teacher(images)).cpu().tolist()
            for index, state in zip(range(previous, current), states):
                self.query_states[index] = state
                self.state_rows.append({"anchor_index": index, "query_step": self.stream_step,
                                        "sample_id": int(self.query_anchors.target[index]),
                                        "normalized_teacher_entropy": state})
        return outputs

    def sample_select(self, model, data, target, anchors, *args, **kwargs):
        result = super().sample_select(model, data, target, anchors, *args, **kwargs)
        if kwargs.get("use_pseudo_label", False):
            states = self.entropy_state(result[0]).detach().cpu()
            self.pseudo_states.extend(float(states[int(position)]) for position in result[1])
            if len(self.pseudo_states) != result[2].num_elem():
                raise RuntimeError("Pseudo entropy-state memory lost alignment")
        return result

    @torch.no_grad()
    def _audit_pseudo_draw(self, indices, baseline_gates, losses):
        reference = self.source_anchors
        features = F.normalize(reference.feats[indices].to(self.config.device), dim=1)
        labels = reference.target[indices].to(self.config.device).long()
        states = features.new_tensor([self.pseudo_states[index] for index in indices.tolist()])
        events = list(self.calibration_events)
        gates = torch.ones_like(baseline_gates)
        neighbors = torch.zeros_like(baseline_gates)
        if events:
            rows = self.calibration_rows[-len(events):]
            cache_states = features.new_tensor([self.query_states[row["anchor_index"]] for row in rows])
            cache_features = torch.stack([event[1] for event in events]).to(features.device)
            predictions = torch.tensor([event[0] for event in events], device=features.device)
            correct = features.new_tensor([event[2] for event in events])
            similarity = features @ cache_features.T
            eligible = ((labels[:, None] == predictions[None, :])
                        & (similarity >= DEFAULTS["gate_similarity"])
                        & ((states[:, None] - cache_states[None, :]).abs() <= self.entropy_gap))
            values, positions = similarity.masked_fill(~eligible, -1).topk(
                min(DEFAULTS["gate_neighbors"], len(events)), dim=1)
            valid = values >= DEFAULTS["gate_similarity"]
            weights = torch.exp((values - 1) / DEFAULTS["gate_temperature"]) * valid
            support = weights.sum(1)
            gates = (DEFAULTS["gate_prior"] + (weights * correct[positions]).sum(1)) / (
                DEFAULTS["gate_prior"] + support)
            neighbors = valid.sum(1).to(gates.dtype)
        self.match_rows.append({
            "step": self.stream_step, "iteration": self.slow_per_step[self.stream_step],
            "original_gate_mean": float(baseline_gates.mean()), "matched_gate_mean": float(gates.mean()),
            "matched_neighbors_mean": float(neighbors.mean()),
            "no_matching_feedback_fraction": float((neighbors == 0).float().mean()),
        })
        return gates

    def run_stream(self, output_dir, progress=None):
        result = super().run_stream(output_dir, progress=progress)
        write_csv(output_dir / "query_teacher_states.csv", self.state_rows)
        write_csv(output_dir / "state_matching.csv", self.match_rows)
        result.update(method="VLCS-confidence-match", variant="confidence_match", entropy_gap=self.entropy_gap)
        return result
