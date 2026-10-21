"""A two-factor ablation sharing the no-adapter dual training implementation."""

import torch

from experiments.fcra_dual.algorithm import DualFCRA


VARIANTS = {
    "simatta_control": (False, False),
    "replay_only": (True, False),
    "calibration_only": (False, True),
    "full_fcra": (True, True),
}


class AblationFCRA(DualFCRA):
    def __init__(self, config, trace, variant):
        self.use_replay, self.use_calibration = VARIANTS[variant]
        self.variant = variant
        if trace.fast_enabled or trace.fast_total:
            raise ValueError("Ablations require the D=0 no-adapter reference budget")
        super().__init__(config, 0, trace)

    def _replay_statistics(self, data, labels):
        need, relevance, score, probabilities = super()._replay_statistics(data, labels)
        if not self.use_replay:
            probabilities = torch.full_like(probabilities, 1.0 / len(probabilities))
        return need, relevance, score, probabilities

    def _feedback_gates(self, features, labels):
        if not self.use_calibration:
            return features.new_ones(len(features))
        return super()._feedback_gates(features, labels)

    def run_stream(self, output_dir, progress=None):
        result = super().run_stream(output_dir, progress=progress)
        result.update(method="FCRA-ablation", variant=self.variant,
                      targeted_replay=self.use_replay, teacher_calibration=self.use_calibration)
        if result["fast_updates"] or result["fast_adapter_enabled"]:
            raise RuntimeError("Fast adaptation leaked into the no-adapter ablation")
        return result
