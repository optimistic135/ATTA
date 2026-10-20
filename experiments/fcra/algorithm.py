"""FCRA: pending coverage, current-error retrieval and conflict-aware replay.

Only the inherited oracle holds unreleased labels. This module receives images,
stream IDs and, after release, the oracle's labeled memory.
"""

import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from ATTA.data.loaders.fast_data_loader import InfiniteDataLoader
from experiments.delayed_feedback.simatta import DelayedSimATTA, write_csv


DEFAULTS = {
    "pending_penalty": 0.25,
    "epsilon": 1e-12,
    "retrieval_size": "train.train_bs",
    "reference": "SimATTA low-entropy frozen-teacher pseudo-label memory",
    "projection": "weighted supervised minibatch; all trainable parameters",
}


@torch.no_grad()
def project_supervision(supervised, reference, epsilon=1e-12):
    """Project a gradient onto the reference's non-conflicting half-space.

    None denotes an unused parameter, i.e. a zero vector. The guarantee concerns
    the loss gradients before SGD momentum, not the momentum-adjusted update.
    """
    device = next(g.device for g in supervised if g is not None)
    dot = torch.zeros((), device=device)
    reference_norm = torch.zeros_like(dot)
    supervised_norm = torch.zeros_like(dot)
    for target, source in zip(supervised, reference):
        if target is not None:
            supervised_norm += target.float().square().sum()
        if source is not None:
            reference_norm += source.float().square().sum()
        if target is not None and source is not None:
            dot += (target.float() * source.float()).sum()
    if not bool(torch.isfinite(dot) & torch.isfinite(reference_norm)
                & torch.isfinite(supervised_norm)):
        raise FloatingPointError("FCRA encountered a non-finite loss gradient")
    valid = bool(reference_norm > epsilon)
    conflict = valid and bool(dot < 0)
    corrected = list(supervised)
    if conflict:
        coefficient = dot / reference_norm
        corrected = [
            target if source is None else (
                -coefficient * source if target is None else target - coefficient * source
            )
            for target, source in zip(supervised, reference)
        ]
    dot_after = sum(
        (target.float() * source.float()).sum()
        for target, source in zip(corrected, reference)
        if target is not None and source is not None
    )
    cosine = (float(dot / (reference_norm * supervised_norm).sqrt())
              if valid and bool(supervised_norm > epsilon) else None)
    return corrected, {
        "reference_valid": valid, "conflict": conflict,
        "gradient_cosine": cosine, "dot_before": float(dot),
        "dot_after": float(dot_after),
        "reference_norm_sq": float(reference_norm),
        "supervised_norm_sq": float(supervised_norm),
    }


class FCRA(DelayedSimATTA):
    """Preserve the delayed stream, with PCFR selection and FACA training."""

    def __init__(self, config, delay):
        super().__init__(config, delay)
        self.stream_step = -1
        self.query_rows, self.retrieval_rows, self.update_rows = [], [], []

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
        self.query_rows.append({
            "step": self.stream_step, "pending_before_release": len(pending),
            "eligible_before": int(eligible.sum()), "eligible_after": int(keep.sum()),
            "suppressed": int((eligible & ~keep).sum()), "new_queries": current - previous,
            "mean_redundancy": float(redundancy.mean()),
        })
        # Always return the full batch's prediction, including screened images.
        return outputs

    @torch.no_grad()
    def retrieve(self):
        """Score every returned label at the current, pre-update model state."""
        indices = sorted(self.received)
        if not indices:
            return None
        self.model.eval()
        batch_features = self.model[0](self.current_data)
        reference_feature = F.normalize(batch_features.mean(0, keepdim=True), dim=1)
        needs, relevances = [], []
        size = self.config.atta.batch_size
        for start in range(0, len(indices), size):
            ids = indices[start:start + size]
            images = self.query_anchors.data[ids].to(self.config.device)
            features = self.model[0](images)
            probabilities = self.model[1](features).softmax(1)
            labels = torch.tensor([self.received[i] for i in ids],
                                  device=probabilities.device, dtype=torch.long)
            need = 1 - probabilities.gather(1, labels[:, None]).squeeze(1)
            similarity = (F.normalize(features, dim=1) * reference_feature).sum(1)
            needs.append(need.cpu())
            relevances.append(((1 + similarity) / 2).clamp(0, 1).cpu())
        need, relevance = torch.cat(needs), torch.cat(relevances)
        score = need * relevance
        # Stable tie-breaking uses immutable anchor IDs, never labels/domains.
        positions = sorted(range(len(indices)), key=lambda j: (-float(score[j]), indices[j]))
        positions = positions[:self.config.train.train_bs]
        selected = [indices[j] for j in positions]
        weights = score[positions]
        weights = (weights / weights.mean() if bool(weights.mean() > DEFAULTS["epsilon"])
                   else torch.ones_like(weights))
        for position, index, weight in zip(positions, selected, weights):
            record = self.oracle.records[index]
            self.retrieval_rows.append({
                "step": self.stream_step, "anchor_index": index,
                "sample_id": int(self.query_anchors.target[index]),
                "label": self.received[index], "query_step": record.query_step,
                "return_step": record.return_step, "query_domain": record.query_domain,
                "return_domain": record.return_domain,
                "need": float(need[position]), "relevance": float(relevance[position]),
                "score": float(score[position]), "weight": float(weight),
            })
        return TensorDataset(
            self.query_anchors.data[selected],
            torch.tensor([self.received[i] for i in selected], dtype=torch.long),
            weights,
        )

    @torch.enable_grad()
    def cluster_train(self, target_anchors, source_anchors):
        target_dataset = self.retrieve()
        if target_dataset is None:
            return
        reference = self.source_anchors
        has_reference = reference is not None and reference.num_elem() > 0
        source_count = reference.num_elem() if has_reference else target_anchors.num_elem()
        alpha = target_anchors.num_elem() / (target_anchors.num_elem() + source_count)
        if source_count < self.cold_start:
            alpha = min(0.2, alpha)
        target_loader = iter(InfiniteDataLoader(
            target_dataset, weights=None, batch_size=self.config.train.train_bs,
            num_workers=self.config.num_workers))
        source_loader = (iter(InfiniteDataLoader(
            TensorDataset(reference.data, reference.target), weights=None,
            batch_size=self.config.train.train_bs, num_workers=self.config.num_workers))
            if has_reference else None)
        self.model.train()
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=self.config.atta.SimATTA.lr, momentum=0.9)
        loss_window, tolerance, lowest_loss = [], 0, float("inf")
        iteration = 0
        while True:
            images, labels, weights = next(target_loader)
            images, labels = images.to(self.config.device), labels.to(self.config.device)
            weights = weights.to(self.config.device)
            losses = F.cross_entropy(self.model(images), labels, reduction="none")
            supervised_loss = (weights * losses).mean()
            optimizer.zero_grad(set_to_none=True)
            supervised = torch.autograd.grad(supervised_loss, parameters, allow_unused=True)
            reference_loss = None
            if has_reference:
                source_images, source_labels = next(source_loader)
                reference_loss = self.config.metric.loss_func(
                    self.model(source_images.to(self.config.device)),
                    source_labels.to(self.config.device))
                reference_grad = torch.autograd.grad(reference_loss, parameters, allow_unused=True)
            else:
                reference_grad = [None] * len(parameters)
            corrected, diagnostic = project_supervision(
                supervised, reference_grad, DEFAULTS["epsilon"])
            for parameter, target, source in zip(parameters, corrected, reference_grad):
                if has_reference:
                    gradient = (alpha * target if target is not None else None)
                    if source is not None:
                        term = (1 - alpha) * source
                        gradient = term if gradient is None else gradient + term
                else:
                    # SimATTA's source fallback duplicates truth replay. Use the
                    # calibrated truth loss once, with no artificial reference.
                    gradient = target
                parameter.grad = gradient
            optimizer.step()
            self.update_rows.append({
                "step": self.stream_step, "iteration": iteration,
                "returned_memory": len(self.received), "retrieved": len(target_dataset),
                "source_memory": reference.num_elem() if has_reference else 0,
                "alpha": alpha, "has_reference": has_reference,
                "supervised_loss": float(supervised_loss.detach()),
                "reference_loss": float(reference_loss.detach()) if has_reference else None,
                "first_order_improvement": (self.config.atta.SimATTA.lr * diagnostic["dot_before"]
                                            if diagnostic["reference_valid"] else None),
                **diagnostic,
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

    def run_stream(self, output_dir, progress=None):
        result = super().run_stream(output_dir, progress=progress)
        output = Path(output_dir)
        write_csv(output / "query_diagnostics.csv", self.query_rows)
        if self.retrieval_rows:
            write_csv(output / "retrieval.csv", self.retrieval_rows)
        else:
            (output / "retrieval.csv").write_text(
                "step,anchor_index,sample_id,label,query_step,return_step,query_domain,"
                "return_domain,need,relevance,score,weight\n", encoding="utf-8")
        if self.update_rows:
            write_csv(output / "updates.csv", self.update_rows)
        else:
            (output / "updates.csv").write_text(
                "step,iteration,returned_memory,retrieved,source_memory,alpha,has_reference,"
                "supervised_loss,reference_loss,first_order_improvement,reference_valid,conflict,gradient_cosine,"
                "dot_before,dot_after,reference_norm_sq,supervised_norm_sq\n", encoding="utf-8")
        result["method"] = "FCRA"
        return result
