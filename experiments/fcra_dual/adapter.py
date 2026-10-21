"""Small identity-initialized residual adapter and causal expert evidence."""

from collections import deque

import torch
from torch import nn
from torch.nn import functional as F


DEFAULTS = {
    "rank": 16, "residual_ratio": 0.1, "ema": 0.99,
    "fast_lr": 1e-3, "fast_momentum": 0.0,
    "confidence": 0.9, "target_temperature": 0.5, "anchor_weight": 0.1,
    "evidence_size": 512, "evidence_neighbors": 8, "similarity": 0.8,
    "similarity_temperature": 0.1, "age_scale": 50.0, "prior": 4.0,
    "gain_clip": 2.0, "fusion_temperature": 0.2, "fusion_max": 0.5,
    "epsilon": 1e-12,
    "strong_view": "per-image horizontal flip p=0.5; batch ColorJitter(0.2,0.2,0.2,0.05)",
}


class ResidualAdapter(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.norm = nn.LayerNorm(dimension, elementwise_affine=False)
        self.down = nn.Linear(dimension, min(DEFAULTS["rank"], dimension))
        self.up = nn.Linear(min(DEFAULTS["rank"], dimension), dimension)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features):
        residual = self.up(F.gelu(self.down(self.norm(features))))
        bound = DEFAULTS["residual_ratio"] * features.norm(dim=1, keepdim=True)
        scale = (bound / residual.norm(dim=1, keepdim=True).clamp_min(DEFAULTS["epsilon"])).clamp(max=1)
        return features + scale * residual


class ExpertEvidence:
    def __init__(self):
        self.events = deque(maxlen=DEFAULTS["evidence_size"])

    def add(self, feature, gain, step):
        self.events.append((feature.detach().cpu(), float(gain), step))

    @torch.no_grad()
    def weights(self, features, step):
        if not self.events:
            return features.new_full((len(features),), 0.25)
        events = list(self.events)
        keys = torch.stack([item[0] for item in events]).to(features.device)
        gains = features.new_tensor([item[1] for item in events])
        ages = features.new_tensor([step - item[2] for item in events])
        similarity = features @ keys.T
        eligible = similarity >= DEFAULTS["similarity"]
        values, positions = similarity.masked_fill(~eligible, -1).topk(
            min(DEFAULTS["evidence_neighbors"], len(events)), dim=1)
        affinity = (torch.exp((values - 1) / DEFAULTS["similarity_temperature"])
                    * torch.exp(-ages[positions] / DEFAULTS["age_scale"])
                    * (values >= DEFAULTS["similarity"]))
        score = (affinity * gains[positions]).sum(1) / (DEFAULTS["prior"] + affinity.sum(1))
        return DEFAULTS["fusion_max"] * torch.sigmoid(score / DEFAULTS["fusion_temperature"])
