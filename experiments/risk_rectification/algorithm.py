"""Original SP plus matched current soft loss, optionally rectified by delayed truth."""
import hashlib
import math
from pathlib import Path
import statistics

import torch
from torch.nn import functional as F

from experiments.sp_atta.algorithm import SPATTA
from experiments.fcra_relay.stream import save_rows
from .density import estimate_weights
from .spec import GROUPS, DEFAULTS as D


def bounded_log_probability(logits):
    return logits.float().log_softmax(-1).clamp_min(math.log(D["probability_floor"]))


def soft_loss(log_probability, probability):
    return -(probability.detach() * log_probability).sum(-1)


class RiskRectification(SPATTA):
    def __init__(self, config, group, trace):
        if group not in GROUPS:
            raise ValueError("Unknown risk-rectification arm")
        super().__init__(config, "SP-ATTA", trace)
        self.study_group = group
        self.source_cache = {}
        self.formal_training = self.seen_feedback = False
        self.batch_rows, self.loss_rows, self.weight_rows = [], [], []
        self.source_hash, self.planned_hash = hashlib.sha256(), hashlib.sha256()
        self.current_aux_updates = self.rectification_updates = self.density_updates = 0

    @torch.no_grad()
    def select_queries(self, data, stream_ids):
        outputs = super().select_queries(data, stream_ids)
        self.teacher.eval()
        features = self.teacher[0](data)
        probability = self.teacher[1](features).float().softmax(-1)
        normalized = F.normalize(features.float(), dim=-1).cpu()
        if not bool(torch.isfinite(probability).all() and torch.isfinite(normalized).all()):
            raise FloatingPointError("Non-finite frozen-source cache")
        self.current_images = data.detach()
        self.current_targets = probability.detach()
        self.current_features = normalized
        self.pre_logits = outputs.detach().float().cpu()
        self.before_feedback = not self.seen_feedback
        self.source_hash.update(probability.cpu().numpy().tobytes())
        self.source_hash.update(normalized.numpy().tobytes())
        for row in self.trace.queries[self.step]:
            position = int(row["sample_id"]) - int(stream_ids[0])
            self.source_cache[int(row["anchor_index"])] = (normalized[position].clone(), probability[position].cpu().clone())
        self.diagnostic = {"memory": len(self.received), "eligible": False, "reason": "no_formal_training",
            "ess": None, "ess_fraction": None, "clip_fraction": None, "holdout_accuracy": None,
            "proxy_risk": float(soft_loss(bounded_log_probability(outputs), probability).mean()),
            "memory_residual": None, "rectified_risk": None}
        return outputs

    def on_labels_released(self, step, domain, released):
        super().on_labels_released(step, domain, released)
        if released:
            self.seen_feedback = True

    @torch.no_grad()
    def prepare_batch(self):
        ready = sorted(self.received)
        self.diagnostic["memory"] = len(ready)
        self.ready_targets = torch.stack([self.source_cache[i][1] for i in ready]).to(self.config.device)
        features = torch.stack([self.source_cache[i][0] for i in ready])
        density = estimate_weights(features, self.current_features)
        self.diagnostic["reason"] = "insufficient_memory_or_batch" if density is None else "eligible" if density["eligible"] else "low_ess"
        self.eligible = density is not None and density["eligible"]
        self.diagnostic["eligible"] = self.eligible
        if density is not None:
            self.density_updates += 2 * D["density_steps"]
            for key in ("ess", "ess_fraction", "clip_fraction", "holdout_accuracy"):
                self.diagnostic[key] = density[key]
            self.ready_weights = density["weights"].to(self.config.device)
            for index, raw, weight in zip(ready, density["raw"].tolist(), density["weights"].tolist()):
                self.weight_rows.append({"step": self.step, "anchor_index": index,
                    "sample_id": int(self.query_anchors.target[index]), "raw_weight": raw,
                    "normalized_weight": weight, "eligible": self.eligible})
            # Evaluation diagnostic, before formal updates, on AVAILABLE historical truth only.
            self.model.eval()
            residual = 0.0
            for start in range(0, len(ready), D["batch_size"]):
                ids = ready[start:start + D["batch_size"]]
                logits = self.model(self.query_anchors.data[ids].to(self.config.device))
                logp = bounded_log_probability(logits)
                labels = torch.tensor([self.received[i] for i in ids], device=logp.device)
                true = -logp.gather(1, labels[:, None]).squeeze(1)
                pseudo = soft_loss(logp, self.ready_targets[start:start + len(ids)])
                residual += float((self.ready_weights[start:start + len(ids)] * (true - pseudo)).sum())
            self.diagnostic["memory_residual"] = residual / len(ready)
            self.diagnostic["rectified_risk"] = self.diagnostic["proxy_risk"] + residual / len(ready)
        # Mirror the parent's exact truth/history draw order, without using its RNG.
        generator = torch.Generator().manual_seed(self.config.random_seed + 1000003 * self.step + 29)
        count_p = self.source_anchors.num_elem() if self.source_anchors is not None else 0
        self.draw_positions = []
        ready_tensor = torch.tensor(ready)
        for iteration in range(self.trace.quotas[self.step]):
            tpos = torch.randint(len(ready), (self.config.train.train_bs,), generator=generator)
            ppos = torch.randint(count_p, (self.config.train.train_bs,), generator=generator) if count_p else torch.empty(0, dtype=torch.long)
            self.draw_positions.append(tpos.to(self.config.device))
            self.planned_hash.update(f"{self.step}:{iteration}:".encode())
            self.planned_hash.update(ready_tensor[tpos].numpy().tobytes())
            self.planned_hash.update(ppos.numpy().tobytes())

    def cluster_train(self, target_anchors, source_anchors):
        self.prepare_batch()
        self.formal_training = True
        try:
            return super().cluster_train(target_anchors, source_anchors)
        finally:
            self.formal_training = False

    @torch.enable_grad()
    def _loss(self, model, truth_images, truth_labels, pseudo_images, pseudo_labels, true_coef, pseudo_coef):
        if not self.formal_training or model is not self.model:
            return super()._loss(model, truth_images, truth_labels, pseudo_images, pseudo_labels, true_coef, pseudo_coef)
        # Match original SP order: historical pseudo forward, then truth forward.
        pseudo = F.cross_entropy(model(pseudo_images), pseudo_labels) if pseudo_coef > 0 and pseudo_images is not None else None
        truth_logits = model(truth_images)
        truth = F.cross_entropy(truth_logits, truth_labels)
        loss = true_coef * truth + (pseudo_coef * pseudo if pseudo is not None else 0)
        current_loss = residual = auxiliary = None
        if self.eligible and self.study_group != GROUPS[0]:
            current_loss = soft_loss(bounded_log_probability(model(self.current_images)), self.current_targets).mean()
            auxiliary = current_loss
            if self.study_group == GROUPS[2]:
                positions = self.draw_positions[self.formal_by_step[self.step]]
                logp = bounded_log_probability(truth_logits)
                true_bounded = -logp.gather(1, truth_labels[:, None]).squeeze(1)
                historical_soft = soft_loss(logp, self.ready_targets[positions])
                residual = (self.ready_weights[positions] * (true_bounded - historical_soft)).mean()
                auxiliary = current_loss + residual
                self.rectification_updates += 1
            loss = loss + D["eta"] * auxiliary
            self.current_aux_updates += 1
        self.loss_rows.append({"step": self.step, "iteration": self.formal_by_step[self.step],
            "eligible": self.eligible, "true_coefficient": true_coef, "historical_coefficient": pseudo_coef,
            "eta": D["eta"] if auxiliary is not None else 0.0,
            "current_loss": float(current_loss.detach()) if current_loss is not None else None,
            "residual": float(residual.detach()) if residual is not None else None,
            "auxiliary": float(auxiliary.detach()) if auxiliary is not None else None})
        return loss, truth, pseudo

    @torch.no_grad()
    def on_batch_end(self, step, domain, data, target, outputs, released):
        labels = target.cpu()
        before = self.pre_logits.argmax(-1)
        source = self.current_targets.argmax(-1).cpu()
        actual = -bounded_log_probability(self.pre_logits).gather(1, labels[:, None]).squeeze(1).mean()
        self.batch_rows.append({"step": step, "domain": domain, "before_feedback": self.before_feedback,
            "samples": len(labels), "pre_correct": int(before.eq(labels).sum()), "released": len(released),
            "source_correct": int(source.eq(labels).sum()),
            "source_fixes": int((source.eq(labels) & before.ne(labels)).sum()),
            "source_breaks": int((source.ne(labels) & before.eq(labels)).sum()),
            "actual_current_risk": float(actual), **self.diagnostic,
            "online_predictions_sha256": hashlib.sha256(before.numpy().tobytes()).hexdigest()})
        super().on_batch_end(step, domain, data, target, outputs, released)

    def run_stream(self, output_dir, progress=None):
        result = super().run_stream(output_dir, progress)
        if self.planned_hash.hexdigest() != self.formal_hash.hexdigest():
            raise AssertionError("Residual memory positions differ from actual formal draws")
        if len(self.source_cache) != result["queried"]:
            raise AssertionError("Frozen source cache lost query alignment")
        for name, rows, fields in (("risk_batches.csv", self.batch_rows, ["step", "domain", "eligible"]),
                ("risk_losses.csv", self.loss_rows, ["step", "iteration", "auxiliary"]),
                ("density_weights.csv", self.weight_rows, ["step", "anchor_index", "normalized_weight"])):
            save_rows(Path(output_dir) / name, rows, fields)
        result.update(method=self.study_group, source_cache_sha256=self.source_hash.hexdigest(),
            current_aux_updates=self.current_aux_updates, rectification_updates=self.rectification_updates,
            eligible_batches=sum(r["eligible"] for r in self.batch_rows), density_updates=self.density_updates,
            **self.compute)
        for key in ("ess", "ess_fraction", "clip_fraction"):
            values = [r[key] for r in self.batch_rows if r[key] is not None]
            result["mean_" + key] = statistics.mean(values) if values else None
        values = [r["holdout_accuracy"] for r in self.batch_rows if r["holdout_accuracy"] is not None]
        result["mean_density_holdout_accuracy"] = statistics.mean(values) if values else None
        for key in ("current_loss", "residual", "auxiliary"):
            values = [r[key] for r in self.loss_rows if r[key] is not None]
            result["mean_training_" + key] = statistics.mean(values) if values else None
        for key, field in (("proxy_risk_mae", "proxy_risk"), ("rectified_risk_mae", "rectified_risk")):
            rows = [r for r in self.batch_rows if r["eligible"] and r[field] is not None]
            n = sum(r["samples"] for r in rows)
            result[key] = sum(abs(r[field] - r["actual_current_risk"]) * r["samples"] for r in rows) / n if n else None
        n = sum(r["samples"] for r in self.batch_rows)
        for key, field in (("current_target_accuracy_pct", "source_correct"), ("current_target_fixed_pct", "source_fixes"),
                           ("current_target_broken_pct", "source_breaks")):
            result[key] = 100 * sum(r[field] for r in self.batch_rows) / n
        for name, flag in (("before_feedback", True), ("after_feedback", False)):
            rows = [r for r in self.batch_rows if r["before_feedback"] == flag]
            count = sum(r["samples"] for r in rows)
            result[name + "_accuracy_pct"] = 100 * sum(r["pre_correct"] for r in rows) / count if count else None
        return result
