"""A two-factor ablation sharing the no-adapter dual training implementation."""

import torch

from experiments.fcra_dual.algorithm import DualFCRA
from experiments.delayed_feedback.simatta import DelayedSimATTA, write_csv


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


class OriginalSimATTA(DelayedSimATTA):
    """Original selection, loaders, joint loss and early stopping at batch=32.

    The D=0 stream adapter supplies causal scoring, query-ID deduplication and
    tiny-candidate cluster protection, but never replays reference decisions.
    """

    def __init__(self, config):
        super().__init__(config, 0)
        self.update_rows = []

    def one_step_train(self, S_data, S_targets, T_data, T_targets, alpha, optimizer):
        loss = super().one_step_train(S_data, S_targets, T_data, T_targets, alpha, optimizer)
        self.update_rows.append({"iteration": len(self.update_rows),
                                 "supervised_loss": float(loss.detach()), "alpha": alpha})
        return loss

    def run_stream(self, output_dir, progress=None):
        result = super().run_stream(output_dir, progress=progress)
        if self.update_rows:
            write_csv(output_dir / "updates.csv", self.update_rows)
        else:
            (output_dir / "updates.csv").write_text(
                "iteration,supervised_loss,alpha\n", encoding="utf-8")
        result.update(method="FCRA-ablation", variant="simatta_original",
                      targeted_replay=False, teacher_calibration=False,
                      fast_adapter_enabled=False, fast_updates=0,
                      slow_updates=len(self.update_rows), reference_updates=None)
        return result
