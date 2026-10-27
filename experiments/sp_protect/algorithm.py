"""Only formal SP-Protect training adds a fixed current-batch KL anchor."""
import hashlib
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.sp_atta.algorithm import SPATTA
from experiments.fcra_relay.stream import save_rows
from .spec import DEFAULTS as D, GROUPS


class SPProtect(SPATTA):
    def __init__(self, config, group, trace):
        if group not in GROUPS:
            raise ValueError("Unknown SP-Protect study arm")
        # Keep the SP objective and its original gate in BOTH deployed arms.
        super().__init__(config, "SP-ATTA", trace)
        self.study_group = group
        self.protection_training = False
        self.seen_feedback = False
        self.selection_hash = hashlib.sha256()
        self.selection_rows, self.loss_rows, self.effect_rows = [], [], []
        self.protection_updates = 0
        self.kl_sum = 0.0

    @torch.no_grad()
    def select_queries(self, data, stream_ids):
        output = super().select_queries(data, stream_ids)
        self.teacher.eval()  # Existing frozen source model, not a new checkpoint.
        probability = self.teacher(data).softmax(-1)
        flipped = self.teacher(data.flip(-1)).softmax(-1)
        if not bool(torch.isfinite(probability).all() and torch.isfinite(flipped).all()):
            raise FloatingPointError("Non-finite source selection probabilities")
        confidence = probability.max(-1).values.cpu()
        eligible = ((confidence >= D["confidence"]) &
            probability.argmax(-1).cpu().eq(flipped.argmax(-1).cpu())).nonzero().flatten()
        generator = torch.Generator().manual_seed(self.config.random_seed + 9000011 * self.step + 37)
        shuffled = eligible[torch.randperm(len(eligible), generator=generator)].tolist()
        ranked = sorted(shuffled, key=lambda i: float(confidence[i]), reverse=True)
        k = math.ceil(D["rank_fraction"] * len(ranked))
        self.positions = torch.tensor(ranked[:k], dtype=torch.long)
        device_positions = self.positions.to(data.device)
        self.protect_images = data[device_positions].detach()
        # Same input view; do not sharpen, flip, harden, or refresh these targets.
        self.protect_targets = output[device_positions].float().softmax(-1).detach()
        self.before_predictions = output.argmax(-1).cpu()
        self.before_feedback = not self.seen_feedback
        selected_ids = stream_ids[self.positions]
        self.selection_hash.update(selected_ids.numpy().tobytes())
        for position in self.positions.tolist():
            self.selection_rows.append({"step": self.step, "domain": self.trace.batches[self.step]["domain"],
                "sample_id": int(stream_ids[position]), "source_confidence": float(confidence[position]),
                "source_prediction": int(probability[position].argmax()),
                "online_prediction": int(self.before_predictions[position])})
        self.base_candidates = len(eligible)
        return output

    def on_labels_released(self, step, domain, released):
        super().on_labels_released(step, domain, released)
        if released:
            self.seen_feedback = True

    def cluster_train(self, target_anchors, source_anchors):
        self.protection_training = self.study_group == "SP-Protect" and len(self.positions) > 0
        try:
            return super().cluster_train(target_anchors, source_anchors)
        finally:
            self.protection_training = False

    @torch.enable_grad()
    def _loss(self, model, truth_images, truth_labels, pseudo_images, pseudo_labels, true_coef, pseudo_coef):
        loss, truth, pseudo = super()._loss(model, truth_images, truth_labels,
            pseudo_images, pseudo_labels, true_coef, pseudo_coef)
        # Shadow gates/probes remain the original SP test, even for SP-Protect.
        if model is self.model and self.protection_training:
            log_probability = model(self.protect_images).float().log_softmax(-1)
            kl = F.kl_div(log_probability, self.protect_targets, reduction="batchmean")
            if not bool(torch.isfinite(kl)):
                raise FloatingPointError("Non-finite current protection KL")
            loss = loss + D["protect_beta"] * kl
            self.loss_rows.append({"step": self.step, "iteration": self.formal_by_step[self.step],
                "selected": len(self.positions), "kl": float(kl.detach()), "beta": D["protect_beta"]})
            self.protection_updates += 1
            self.kl_sum += float(kl.detach())
        return loss, truth, pseudo

    @torch.no_grad()
    def on_batch_end(self, step, domain, data, target, outputs, released):
        # This is a diagnostic post-update prediction; it is never scored as online accuracy.
        self.model.eval()
        post = self.model(data).argmax(-1).cpu() if self.formal_by_step[step] else self.before_predictions.clone()
        labels = target.cpu()
        before_correct = self.before_predictions.eq(labels)
        after_correct = post.eq(labels)
        selected = self.positions
        source_lookup = {r["sample_id"]: r["source_prediction"] for r in self.selection_rows[-len(selected):]} if len(selected) else {}
        lo, _ = self.trace.bounds[step]
        self.effect_rows.append({"step": step, "domain": domain, "before_feedback": self.before_feedback,
            "released": len(released), "updates": self.formal_by_step[step], "samples": len(labels),
            "base_candidates": self.base_candidates, "selected": len(selected),
            "pre_correct": int(before_correct.sum()), "post_correct": int(after_correct.sum()),
            "fixed": int((~before_correct & after_correct).sum()),
            "broken": int((before_correct & ~after_correct).sum()),
            "selected_pre_correct": int(before_correct[selected].sum()),
            "selected_post_correct": int(after_correct[selected].sum()),
            "selected_fixed": int((~before_correct[selected] & after_correct[selected]).sum()),
            "selected_broken": int((before_correct[selected] & ~after_correct[selected]).sum()),
            "selected_source_correct": sum(source_lookup[lo + int(p)] == int(labels[p]) for p in selected),
            "online_predictions_sha256": hashlib.sha256(self.before_predictions.numpy().tobytes()).hexdigest()})
        super().on_batch_end(step, domain, data, target, outputs, released)

    def run_stream(self, output_dir, progress=None):
        result = super().run_stream(output_dir, progress)
        output = Path(output_dir)
        for name, rows, fields in (
            ("protection_selection.csv", self.selection_rows, ["step", "sample_id", "source_confidence"]),
            ("protection_losses.csv", self.loss_rows, ["step", "iteration", "selected", "kl", "beta"]),
            ("protection_batches.csv", self.effect_rows, ["step", "domain", "samples", "selected"]),
        ):
            save_rows(output / name, rows, fields)
        total = sum(r["selected"] for r in self.effect_rows)
        result.update(method=self.study_group, selection_sha256=self.selection_hash.hexdigest(),
            protection_updates=self.protection_updates, selected_observations=total,
            mean_protection_kl=self.kl_sum / self.protection_updates if self.protection_updates else None,
            **self.compute)
        for name, field in (("selected_accuracy_before_pct", "selected_pre_correct"),
                ("selected_accuracy_after_pct", "selected_post_correct"),
                ("selected_fixed_pct", "selected_fixed"), ("selected_broken_pct", "selected_broken")):
            result[name] = 100 * sum(r[field] for r in self.effect_rows) / total if total else None
        for phase, flag in (("before_feedback", True), ("after_feedback", False)):
            rows = [r for r in self.effect_rows if r["before_feedback"] == flag]
            samples = sum(r["samples"] for r in rows)
            result[phase + "_accuracy_pct"] = 100 * sum(r["pre_correct"] for r in rows) / samples if samples else None
        return result
