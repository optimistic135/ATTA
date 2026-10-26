"""Current-structure reconstruction with per-batch EMA coordinate refresh."""

from collections import deque
import copy
import json

import torch
from torch.nn import functional as F

from experiments.fcra.algorithm import FCRA
from experiments.fcra_relay.algorithm import FCRARelay
from .graph import reconstruct
from .spec import DEFAULTS as D


class FCRACSR(FCRA):
    # Stateless utilities only; do not construct or run the old Relay method.
    _count_encoder = FCRARelay._count_encoder
    _strong_view = FCRARelay._strong_view
    _sample = FCRARelay._sample

    def __init__(self, config, delay):
        super().__init__(config, delay)
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        self.ema = copy.deepcopy(self.model).eval()
        for model in (self.teacher, self.ema):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        self.trainable = list(self.model.parameters())
        self.optimizer = torch.optim.SGD(self.trainable, lr=D["lr"], momentum=D["momentum"])
        self.generator = torch.Generator().manual_seed(config.random_seed + 211)
        self.buffer = deque(maxlen=D["buffer_batches"])
        self.memory_cache = None
        self.graph_rows, self.assignment_rows, self.train_rows, self.budget_rows = [], [], [], []
        self.query_predictions = []
        self.main_updates = self.current_updates = self.supervised_updates = 0
        self.reconstructed_draws = self.reconstruction_disagreements = 0
        self.graph_eligible_nodes = self.graph_total_nodes = self.anchor_reencode_images = 0
        self.compute = {f"{name}_encoder_{operation}_{unit}": 0
                        for name in ("main", "source", "ema")
                        for operation in ("forward", "backward") for unit in ("calls", "images")}
        for model, name in ((self.model, "main"), (self.teacher, "source"), (self.ema, "ema")):
            self._count_encoder(model, name)

    @torch.no_grad()
    def select_current(self, data, physical_ids):
        self.model.eval()
        logits = self.model(data)
        previous = self.query_anchors.num_elem() if self.query_anchors is not None else 0
        seen = set() if self.query_anchors is None else set(self.query_anchors.target.tolist())
        keep = torch.tensor([int(sid) not in seen for sid in physical_ids], dtype=torch.bool)
        if bool(keep.any()):
            super().select_queries(data[keep.to(data.device)], physical_ids[keep])
        else:
            self.stream_step += 1
            self.query_rows.append({"step": self.stream_step, "pending_before_release": self.oracle.counts()["pending"],
                                    "eligible_before": 0, "eligible_after": 0, "suppressed": 0,
                                    "new_queries": 0, "mean_redundancy": 0.0})
        self.current_data = data.detach()
        self.current_ids = physical_ids.clone()
        current = self.query_anchors.num_elem() if self.query_anchors is not None else 0
        positions = {int(sid): i for i, sid in enumerate(physical_ids)}
        for index in range(previous, current):
            sid = int(self.query_anchors.target[index])
            self.query_predictions.append({"step": self.stream_step, "anchor_index": index,
                                            "sample_id": sid,
                                            "probabilities": json.dumps(logits[positions[sid]].softmax(0).cpu().tolist())})
        return logits

    @torch.no_grad()
    def _encode(self, images, flip=False):
        features = []
        self.ema.eval()
        for start in range(0, len(images), self.config.atta.batch_size):
            batch = images[start:start + self.config.atta.batch_size].to(self.config.device)
            if flip:
                batch = batch.flip(-1)
            features.append(F.normalize(self.ema[0](batch), dim=1))
        return torch.cat(features)

    @torch.no_grad()
    def _build_graph(self):
        self.buffer.append((self.current_data.cpu(), self.current_ids))
        self.buffer_data = torch.cat([entry[0] for entry in self.buffer])
        self.buffer_ids = torch.cat([entry[1] for entry in self.buffer]).to(self.config.device)
        weak = self._encode(self.buffer_data)
        flipped = self._encode(self.buffer_data, flip=True)
        self.reference_features = F.normalize(weak + flipped, dim=1)
        anchor_features = anchor_ids = labels = None
        if self.received:
            if self.memory_cache is None or len(self.memory_cache[0]) != len(self.received):
                self.memory_cache = self._supervised_memory()
            indices, data, labels_cpu = self.memory_cache
            # ALL historical features use THIS SAME EMA snapshot, even with no
            # newly returned labels; no stale fixed-source feature lookup.
            anchor_features = self._encode(data)
            self.anchor_features = anchor_features
            self.anchor_reencode_images += len(data)
            labels = labels_cpu.to(self.config.device)
            anchor_ids = self.query_anchors.target[indices].to(self.config.device)
        self.graph = reconstruct(weak, flipped, self.buffer_ids, anchor_features, anchor_ids, labels,
                                 self.config.dataset.num_classes)
        count = len(self.buffer_data)
        accepted = int(self.graph["valid"].sum())
        self.graph_eligible_nodes += accepted
        self.graph_total_nodes += count
        self.ema.eval()
        # Head-only diagnostics reuse encoded features; these predictions do not
        # seed the graph and do not affect assignment eligibility.
        prediction = self.ema[1](weak).argmax(1)
        disagreements = self.graph["valid"] & (self.graph["target"].argmax(1) != prediction)
        self.graph_rows.append({"step": self.stream_step, "nodes": count,
                                "returned_anchors": len(self.received), "directed_edges": self.graph["edges"],
                                "accepted": accepted, "coverage_pct": 100 * accepted / count,
                                "disagrees_with_ema": int(disagreements.sum()),
                                "mass_mean": float(self.graph["mass"].mean()),
                                "degree_mean": float(self.graph["degree"].float().mean()),
                                "effective_anchors_mean": float(self.graph["support"].mean()),
                                "agreement_mean": float(self.graph["agreement"].mean())})
        for i in range(count):
            self.assignment_rows.append({"step": self.stream_step, "sample_id": int(self.buffer_ids[i]),
                                          "accepted": bool(self.graph["valid"][i]),
                                          "reconstructed_class": int(self.graph["target"][i].argmax()),
                                          "confidence": float(self.graph["confidence"][i]),
                                          "anchor_count": int(self.graph["class_anchors"][i]),
                                          "effective_anchors": float(self.graph["support"][i]),
                                          "neighbor_agreement": float(self.graph["agreement"][i]),
                                          "mass": float(self.graph["mass"][i]),
                                          "probabilities": json.dumps(self.graph["target"][i].cpu().tolist())})

    @torch.no_grad()
    def _replay(self, size):
        _, data, labels = self.memory_cache
        sampled = self._sample(len(labels), size)
        for i in range(size // 4):
            pos = int(self._sample(len(self.reference_features), 1))
            sampled[i] = (self.anchor_features @ self.reference_features[pos]).argmax().cpu()
        return data[sampled].to(self.config.device), labels[sampled].to(self.config.device)

    def _current_positions(self, size):
        total = len(self.buffer_data)
        size = min(size, total)
        latest = len(self.current_data)
        first = torch.randperm(latest, generator=self.generator)[:min((size + 1) // 2, latest)]
        first += total - latest
        remaining = torch.ones(total, dtype=torch.bool)
        remaining[first] = False
        pool = remaining.nonzero().flatten()
        rest = pool[torch.randperm(len(pool), generator=self.generator)[:size - len(first)]]
        return torch.cat((first, rest))

    @staticmethod
    def _relation_loss(student, reference):
        if len(student) < 2:
            return student.sum() * 0
        diagonal = torch.eye(len(student), device=student.device, dtype=torch.bool)
        student_logits = (student @ student.T / D["relation_temperature"]).masked_fill(diagonal, -1e4)
        reference_logits = (reference @ reference.T / D["relation_temperature"]).masked_fill(diagonal, -1e4)
        target = reference_logits.softmax(1).detach()
        return F.kl_div(student_logits.log_softmax(1), target, reduction="batchmean")

    @torch.no_grad()
    def _ema_update(self):
        for target, source in zip(self.ema.parameters(), self.model.parameters()):
            target.lerp_(source, 1 - D["ema"])
        for target, source in zip(self.ema.buffers(), self.model.buffers()):
            target.copy_(source)
        self.ema.eval()

    @torch.enable_grad()
    def _step(self, with_current):
        self.model.train()
        size = self.config.train.train_bs
        loss = supervised = relation = view = reconstruction = None
        accepted = disagreement = 0
        if self.received:
            images, labels = self._replay(size)
            supervised = F.cross_entropy(self.model(images), labels)
            loss = supervised
        if with_current:
            positions = self._current_positions(size)
            device_positions = positions.to(self.config.device)
            images = self.buffer_data[positions].to(self.config.device)
            reference = self.reference_features[device_positions].detach()
            encoded = self.model[0](self._strong_view(images))
            student = F.normalize(encoded, dim=1)
            relation = self._relation_loss(student, reference)
            view = (1 - (student * reference).sum(1)).mean()
            logits = self.model[1](encoded)
            target = self.graph["target"][device_positions].detach()
            weights = self.graph["weight"][device_positions].detach()
            reconstruction = (F.kl_div(logits.log_softmax(1), target, reduction="none").sum(1) * weights).mean()
            valid = self.graph["valid"][device_positions]
            accepted = int(valid.sum())
            disagreement = int((valid & (target.argmax(1) != logits.detach().argmax(1))).sum())
            current = (D["reconstruction_weight"] * reconstruction
                       + D["relation_weight"] * relation + D["view_weight"] * view)
            loss = current if loss is None else loss + current
        if loss is None:
            return
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite FCRA-CSR loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.trainable, D["gradient_clip"], error_if_nonfinite=True)
        self.optimizer.step()
        self._ema_update()
        self.main_updates += 1
        self.current_updates += int(with_current)
        self.supervised_updates += int(supervised is not None)
        self.reconstructed_draws += accepted
        self.reconstruction_disagreements += disagreement
        self.train_rows.append({"step": self.stream_step, "update": self.main_updates,
                                "with_current": with_current, "has_truth": supervised is not None,
                                "loss": float(loss.detach()),
                                "supervised_loss": float(supervised.detach()) if supervised is not None else None,
                                "relation_loss": float(relation.detach()) if relation is not None else None,
                                "view_loss": float(view.detach()) if view is not None else None,
                                "reconstruction_loss": float(reconstruction.detach()) if reconstruction is not None else None,
                                "reconstructed_draws": accepted, "class_disagreements": disagreement,
                                "gradient_norm_before_clip": float(norm)})

    def train_batch(self):
        self._build_graph()
        before = self.main_updates
        for _ in range(D["current_steps"]):
            self._step(with_current=True)
        if self.received:
            for _ in range(D["replay_steps"]):
                self._step(with_current=False)
        used = self.main_updates - before
        quota = D["current_steps"] + D["replay_steps"]
        if used > quota:
            raise AssertionError("FCRA-CSR exceeded per-batch quota")
        self.budget_rows.append({"step": self.stream_step, "quota": quota, "updates": used,
                                 "truth_available": bool(self.received), "unused": quota - used,
                                 "graph_nodes": len(self.buffer_data), "anchor_reencode_images": self.anchor_reencode_images,
                                 **self.compute})
