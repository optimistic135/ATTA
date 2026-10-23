"""One added direction check, and a donor-matched uniform relaxation control."""

import hashlib

import torch

from experiments.confidence_match.algorithm import ConfidenceMatch
from experiments.vlcs_calibration.results import write_csv
from .math import head_cosines, compatibility_gates, uniform_relaxation


def draw_hash(indices):
    return hashlib.sha256(indices.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class GradientCompatibility(ConfidenceMatch):
    def __init__(self, config, trace, mode, entropy_gap, donor=None):
        super().__init__(config, trace, entropy_gap)
        if mode not in ("direction_check", "uniform_relaxation"):
            raise ValueError("Unknown gradient-compatibility variant")
        if not isinstance(self.fc, torch.nn.Linear):
            raise ValueError("Analytic direction check requires a linear classifier head")
        if (mode == "uniform_relaxation") != (donor is not None):
            raise ValueError("Only uniform_relaxation requires a completed direction-check donor")
        self.mode, self.donor = mode, donor
        self.head_context, self.direction_rows = {}, []
        self.donor_used = set()

    def _audit_supervised_draw(self, indices):
        self.supervised_hash = draw_hash(indices)

    def _slow_forward(self, images, labels, kind):
        if self.mode != "direction_check":
            return super()._slow_forward(images, labels, kind)
        captured = []

        def observe(module, inputs, output):
            captured.append((inputs[0].detach(), output.detach(), labels.detach()))

        handle = self.fc.register_forward_hook(observe)
        try:
            logits = super()._slow_forward(images, labels, kind)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("Expected exactly one classifier call in each training forward")
        self.head_context[kind] = captured[0]
        return logits

    @torch.no_grad()
    def _audit_pseudo_draw(self, indices, gates, losses):
        matched = super()._audit_pseudo_draw(indices, gates, losses)
        key = (self.stream_step, self.slow_per_step[self.stream_step])
        pseudo_hash = draw_hash(indices)
        row = {"step": key[0], "iteration": key[1],
               "supervised_draw_hash": self.supervised_hash, "pseudo_draw_hash": pseudo_hash,
               "pseudo_draws": len(indices), "matched_gate_mean": float(matched.mean())}
        if self.mode == "direction_check":
            cosine, valid = head_cosines(self.head_context["supervised"], self.head_context["pseudo"],
                                         bias=self.fc.bias is not None)
            final = compatibility_gates(matched, cosine, valid)
            row.update(cosine_mean=float(cosine.mean()),
                       conflict_fraction=float(((cosine < 0) & valid).float().mean()),
                       invalid_direction_fraction=float((~valid).float().mean()))
            self.head_context.clear()
            target = float((1 - final).mean())
        else:
            if key not in self.donor:
                raise ValueError(f"Donor lacks training update {key}")
            donor = self.donor[key]
            if (donor["supervised_draw_hash"] != self.supervised_hash
                    or donor["pseudo_draw_hash"] != pseudo_hash
                    or int(donor["pseudo_draws"]) != len(indices)):
                raise ValueError(f"Sampling differs from donor at update {key}")
            target = float(donor["removed_weight_mean"])
            final, scale = uniform_relaxation(matched, target)
            row.update(uniform_scale=scale)
            self.donor_used.add(key)
        removed = float((1 - final).mean())
        if (not bool(torch.isfinite(final).all()) or bool((final < matched - 1e-6).any())
                or bool((final > 1 + 1e-6).any())):
            raise FloatingPointError("Invalid relaxed gate")
        row.update(final_gate_mean=float(final.mean()), removed_weight_mean=removed,
                   target_removed_weight_mean=target, matching_gap=removed - target)
        self.direction_rows.append(row)
        return final

    def run_stream(self, output_dir, progress=None):
        result = super().run_stream(output_dir, progress=progress)
        if not self.direction_rows:
            raise ValueError("No pseudo-supervised updates; direction intervention was never exercised")
        if self.donor is not None and self.donor_used != set(self.donor):
            raise ValueError("Donor and control pseudo-update schedules differ")
        write_csv(output_dir / "direction.csv", self.direction_rows)
        draw_count = sum(row["pseudo_draws"] for row in self.direction_rows)
        weighted = lambda key: sum(row[key] * row["pseudo_draws"] for row in self.direction_rows) / draw_count
        result.update(method="VLCS-gradient-compatibility", variant=self.mode,
                      final_gate_mean=weighted("final_gate_mean"),
                      removed_weight_mean=weighted("removed_weight_mean"),
                      mean_abs_matching_gap=sum(abs(row["matching_gap"]) * row["pseudo_draws"]
                                                for row in self.direction_rows) / draw_count,
                      max_abs_matching_gap=max(abs(row["matching_gap"]) for row in self.direction_rows),
                      unmatched_updates=sum(abs(row["matching_gap"]) > 1e-6 for row in self.direction_rows),
                      head_direction_updates=len(self.direction_rows) if self.mode == "direction_check" else 0)
        return result
