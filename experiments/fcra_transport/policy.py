"""Local decision evidence. Only released audit rewards may enter this module."""

from collections import deque
import math

import numpy as np


ACTIONS = (0.0, 0.1, 0.3)
DEFAULTS = {
    "window_samples": 512, "window_batches": 16, "confidence": 0.9,
    "ema": 0.99, "rho": 0.5, "audit_every": 5, "audit_samples": 4,
    "audit_steps": 1, "steps_per_batch": 30, "reward_clip": 2.0,
    "evidence_size": 2048, "age_scale": 200.0, "context_bandwidth": 0.5,
    "prior": 4.0, "class_shrinkage": 8.0, "minimum_support": 3.0,
    "penalty": 0.5, "margin": 1e-4,
}


class TransportPolicy:
    def __init__(self):
        self.events = deque(maxlen=DEFAULTS["evidence_size"])

    def add(self, context, reward):
        if len(reward) != len(ACTIONS) or not np.isfinite(reward).all():
            raise ValueError("Audit reward must be a finite vector for all actions")
        self.events.append((dict(context), np.asarray(reward, dtype=float)))

    def choose(self, context, step):
        if not self.events:
            return 0, {"support": 0.0, "score": 0.0, "reward": 0.0,
                       "action_rewards": [0.0] * len(ACTIONS),
                       "action_scores": [0.0] * len(ACTIONS)}
        weights, same, rewards = [], [], []
        for old, reward in self.events:
            delta = sum((context[key] - old[key]) ** 2
                        for key in ("distance", "reliability", "uncertainty"))
            weight = math.exp(-delta / (2 * DEFAULTS["context_bandwidth"] ** 2))
            # Age is counted from AUDIT generation, not truth arrival.
            weight *= math.exp(-max(0, step - old["step"]) / DEFAULTS["age_scale"])
            weights.append(weight)
            same.append(old["class"] == context["class"])
            rewards.append(reward)
        weights = np.asarray(weights)
        local = weights * np.asarray(same)
        mass = float(local.sum())
        blend = mass / (mass + DEFAULTS["class_shrinkage"])
        # Explicit global-to-class shrinkage; global evidence remains usable
        # when a predicted class has only a few returned audits.
        weights = blend * local + (1 - blend) * weights
        total = float(weights.sum())
        support = total ** 2 / (float(weights @ weights) + 1e-12)
        rewards = np.stack(rewards)
        mean = (weights[:, None] * rewards).sum(0) / (DEFAULTS["prior"] + total)
        variance = (weights[:, None] * (rewards - mean) ** 2).sum(0) / max(total, 1e-12)
        scores = mean - DEFAULTS["penalty"] * np.sqrt(variance / max(support, 1e-12))
        action = int(scores.argmax())
        if (support < DEFAULTS["minimum_support"] or total < 1.0
                or scores[action] <= DEFAULTS["margin"]):
            action = 0
        return action, {"support": support, "score": float(scores[action]),
                        "reward": float(mean[action]), "action_rewards": mean.tolist(),
                        "action_scores": scores.tolist()}
