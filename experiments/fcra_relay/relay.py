"""True-label-only semantic memory; no ground-truth domain or pending label input."""

from collections import deque

import torch
from torch.nn import functional as F

from .spec import DEFAULTS


class SemanticRelay:
    def __init__(self, device, classes):
        self.device, self.classes = device, classes
        self.features, self.labels = None, None
        self.events = deque(maxlen=DEFAULTS["calibration_size"])
        self.rows = []

    @torch.no_grad()
    def calibrate(self, context, label, return_step, anchor):
        """Evaluate fixed QUERY-time distributions, then add local reliability evidence."""
        feature, predicted, supported, base, relayed, query_step = context
        if not supported:
            return
        correct = int(predicted == int(label))
        gain = float(relayed[int(label)].clamp_min(1e-12).log()
                     - base[int(label)].clamp_min(1e-12).log())
        self.events.append((feature.cpu(), predicted, correct, query_step))
        self.rows.append({"anchor_index": anchor, "query_step": query_step,
                          "return_step": return_step, "label": int(label),
                          "relay_prediction_at_query": predicted, "correct": correct,
                          "query_log_probability_gain": gain})

    def replace_memory(self, features, labels):
        # Called only after truth release. Complete memory is never pruned.
        self.features = features.detach().to(self.device)
        self.labels = labels.detach().to(self.device).long()

    @torch.no_grad()
    def infer(self, features, base, step):
        count = len(features)
        fallback = {"vote": base.detach(), "target": base.detach(),
                    "valid": torch.zeros(count, device=features.device, dtype=torch.bool),
                    "trust": base.new_zeros(count), "mix": base.new_zeros(count),
                    "support": base.new_zeros(count), "max_similarity": base.new_zeros(count)}
        if self.features is None or len(self.features) == 0:
            return fallback
        similarities = features @ self.features.T
        values, positions = similarities.topk(min(DEFAULTS["relay_neighbors"], len(self.features)), dim=1)
        eligible = values >= DEFAULTS["relay_similarity"]
        weights = torch.exp((values - 1) / DEFAULTS["relay_temperature"]) * eligible
        mass = weights.sum(1, keepdim=True)
        normalized = weights / mass.clamp_min(1e-12)
        vote = base.new_zeros((count, self.classes))
        vote.scatter_add_(1, self.labels[positions], normalized)
        support = normalized.sum(1).square() / normalized.square().sum(1).clamp_min(1e-12)
        top = vote.topk(2, dim=1).values
        predicted = vote.argmax(1)
        valid = ((mass[:, 0] > 0) & (support >= DEFAULTS["relay_support"])
                 & (top[:, 0] >= DEFAULTS["relay_confidence"])
                 & ((top[:, 0] - top[:, 1]) >= DEFAULTS["relay_margin"]))
        trust = base.new_full((count,), DEFAULTS["calibration_prior_correct"] /
                              (DEFAULTS["calibration_prior_correct"] + DEFAULTS["calibration_prior_wrong"]))
        if self.events:
            events = list(self.events)
            keys = torch.stack([item[0] for item in events]).to(features.device)
            classes = torch.tensor([item[1] for item in events], device=features.device)
            outcomes = base.new_tensor([item[2] for item in events])
            ages = base.new_tensor([max(0, step - item[3]) for item in events])
            affinity = features @ keys.T
            same = predicted[:, None] == classes[None, :]
            similarity, nearest = affinity.masked_fill(~same, -2).topk(
                min(DEFAULTS["calibration_neighbors"], len(events)), dim=1)
            evidence = (torch.exp((similarity - 1) / DEFAULTS["relay_temperature"])
                        * (similarity >= DEFAULTS["calibration_similarity"])
                        * torch.exp(-ages[nearest] / DEFAULTS["calibration_age"]))
            good = (evidence * outcomes[nearest]).sum(1)
            trust = ((DEFAULTS["calibration_prior_correct"] + good) /
                     (DEFAULTS["calibration_prior_correct"] + DEFAULTS["calibration_prior_wrong"]
                      + evidence.sum(1)))
        # No agreement condition with the model's predicted class: feedback can
        # replace an incorrect, high-confidence EMA pseudo label.
        valid = valid & (trust >= 0.5)
        mix = DEFAULTS["prediction_mix_cap"] * trust * valid
        prediction = (1 - mix[:, None]) * base + mix[:, None] * vote
        return {"vote": vote, "target": prediction, "valid": valid,
                "trust": trust, "mix": mix, "support": support,
                "max_similarity": values[:, 0]}
