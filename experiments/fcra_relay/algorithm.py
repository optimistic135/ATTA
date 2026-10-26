"""Immediate current-image learning plus feedback-driven semantic target repair."""

from collections import deque
import copy
import json

import torch
from torch.nn import functional as F

from experiments.fcra.algorithm import FCRA
from .relay import SemanticRelay
from .spec import DEFAULTS


class FCRARelay(FCRA):
    def __init__(self, config, delay):
        super().__init__(config, delay)
        if not hasattr(self.model[0], "network") or not hasattr(self.model[0].network, "layer4"):
            raise ValueError("FCRA-Relay requires the PACS ResNet backbone")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for module in (self.model[0].network.layer4, self.model[1]):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self.ema = copy.deepcopy(self.model).eval()
        for model in (self.teacher, self.ema):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        self.trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.SGD(self.trainable, lr=DEFAULTS["lr"], momentum=DEFAULTS["momentum"])
        self.relay = SemanticRelay(config.device, config.dataset.num_classes)
        self.buffer = deque(maxlen=DEFAULTS["buffer_batches"])
        self.generator = torch.Generator().manual_seed(config.random_seed + 107)
        self.query_contexts = {}
        self.current_description = None
        self.memory_cache = None
        self.main_updates = self.current_updates = self.supervised_updates = 0
        self.relay_used_samples = self.relay_overrides = 0
        self.train_rows, self.relay_rows, self.budget_rows = [], [], []
        self.query_predictions = []
        self.compute = {f"{name}_encoder_{operation}_{unit}": 0
                        for name in ("main", "source", "ema")
                        for operation in ("forward", "backward") for unit in ("calls", "images")}
        for model, name in ((self.model, "main"), (self.teacher, "source"), (self.ema, "ema")):
            self._count_encoder(model, name)

    def _count_encoder(self, model, name):
        def forward(module, inputs, output):
            for unit, value in (("calls", 1), ("images", len(inputs[0]))):
                self.compute[f"{name}_encoder_forward_{unit}"] += value
            if torch.is_grad_enabled() and output.requires_grad:
                size = len(inputs[0])
                def backward(gradient):
                    self.compute[f"{name}_encoder_backward_calls"] += 1
                    self.compute[f"{name}_encoder_backward_images"] += size
                    return gradient
                output.register_hook(backward)
        model[0].register_forward_hook(forward)

    @torch.no_grad()
    def select_current(self, data, physical_ids):
        """Selection sees only images/IDs; output fusion uses previously returned labels."""
        self.model.eval()
        self.teacher.eval()
        raw = self.model(data).softmax(1)
        semantic = F.normalize(self.teacher[0](data), dim=1)
        seen = set() if self.query_anchors is None else set(self.query_anchors.target.tolist())
        keep = torch.tensor([int(sid) not in seen for sid in physical_ids], dtype=torch.bool)
        previous = self.query_anchors.num_elem() if self.query_anchors is not None else 0
        if bool(keep.any()):
            super().select_queries(data[keep.to(data.device)], physical_ids[keep])
        else:
            self.stream_step += 1
            self.query_rows.append({"step": self.stream_step, "pending_before_release": self.oracle.counts()["pending"],
                                    "eligible_before": 0, "eligible_after": 0, "suppressed": 0,
                                    "new_queries": 0, "mean_redundancy": 0.0})
        self.current_data = data.detach()
        relayed = self.relay.infer(semantic, raw, self.stream_step)
        result = relayed["target"]
        positions = {int(sid): i for i, sid in enumerate(physical_ids)}
        current = self.query_anchors.num_elem() if self.query_anchors is not None else 0
        for index in range(previous, current):
            pos = positions[int(self.query_anchors.target[index])]
            self.query_contexts[index] = (
                semantic[pos].cpu(), int(relayed["vote"][pos].argmax()), bool(relayed["eligible"][pos]),
                raw[pos].cpu(), result[pos].cpu(), self.stream_step)
            self.query_predictions.append({
                "step": self.stream_step, "anchor_index": index,
                "sample_id": int(self.query_anchors.target[index]),
                "eligible": bool(relayed["eligible"][pos]), "adopted": bool(relayed["valid"][pos]),
                "raw_probabilities": json.dumps(raw[pos].cpu().tolist()),
                "vote_probabilities": json.dumps(relayed["vote"][pos].cpu().tolist()),
                "output_probabilities": json.dumps(result[pos].cpu().tolist()),
            })
        self.current_description = {"semantic": semantic.cpu(), "raw": raw.cpu(),
                                    "relayed": result.cpu(), "valid": relayed["valid"].cpu()}
        self.relay_rows.append({"step": self.stream_step, "samples": len(data),
                                "relay_supported": int(relayed["valid"].sum()),
                                "relay_eligible": int(relayed["eligible"].sum()),
                                "relay_changed_predictions": int((raw.argmax(1) != result.argmax(1)).sum()),
                                "mean_trust": float(relayed["trust"].mean()),
                                "mean_mix": float(relayed["mix"].mean()),
                                "mean_support": float(relayed["support"].mean()),
                                "max_similarity_mean": float(relayed["max_similarity"].mean())})
        # Log probabilities maintain the selector API's logits-like contract.
        return result.clamp_min(1e-12).log()

    def on_labels_released(self, step, domain, released):
        # Score all query-time evidence before incorporating ANY new truth into
        # semantic memory. A query's own label was absent at its prediction time.
        for index, label in released:
            context = self.query_contexts.pop(index, None)
            if context is not None:
                self.relay.calibrate(context, label, step, index)

    @torch.no_grad()
    def _sync_memory(self):
        if not self.received:
            return
        if self.memory_cache is None or len(self.memory_cache[0]) != len(self.received):
            indices, data, labels = self._supervised_memory()
            semantic = torch.stack([self.teacher_features[index] for index in indices])
            self.memory_cache = indices, data, labels, semantic
            self.relay.replace_memory(semantic, labels)

    @torch.no_grad()
    def _ema_update(self):
        for target, source in zip(self.ema.parameters(), self.model.parameters()):
            target.lerp_(source, 1 - DEFAULTS["ema"])
        for target, source in zip(self.ema.buffers(), self.model.buffers()):
            target.copy_(source)
        self.ema.eval()

    def _sample(self, count, size):
        return torch.randint(count, (size,), generator=self.generator)

    @torch.no_grad()
    def _replay(self, size):
        _, data, labels, semantic = self.memory_cache
        sampled = self._sample(len(labels), size)
        # A quarter of the batch retrieves across ALL true classes, so a wrong
        # current pseudo class cannot block the needed corrective class.
        current = self.current_description["semantic"]
        for i in range(size // 4):
            pos = int(self._sample(len(current), 1))
            sampled[i] = (semantic @ current[pos]).argmax()
        return data[sampled].to(self.config.device), labels[sampled].to(self.config.device)

    @torch.no_grad()
    def _current_pack(self, size):
        # Half current batch, half recent buffer: old pseudo targets are never
        # cached as permanent truth; relay targets are recomputed after release.
        latest_data, latest_semantic = self.buffer[-1]
        latest = self._sample(len(latest_data), (size + 1) // 2)
        all_data, all_semantic = self.buffer_data, self.buffer_semantic
        recent = self._sample(len(all_data), size // 2)
        images = torch.cat((latest_data[latest], all_data[recent])).to(self.config.device)
        semantic = torch.cat((latest_semantic[latest], all_semantic[recent])).to(self.config.device)
        self.ema.eval()
        self.teacher.eval()
        weak = self.ema(images).softmax(1)
        flip = self.ema(images.flip(-1)).softmax(1)
        reliable = ((weak.max(1).values >= DEFAULTS["pseudo_confidence"])
                    & (weak.argmax(1) == flip.argmax(1)))
        relayed = self.relay.infer(semantic, weak, self.stream_step)
        relay_valid = relayed["valid"]
        sharpened = weak.pow(1 / DEFAULTS["sharpen_temperature"])
        sharpened /= sharpened.sum(1, keepdim=True).clamp_min(1e-12)
        # Supported true-label votes REPLACE the EMA target, even on disagreement.
        target = torch.where(relay_valid[:, None], relayed["vote"], sharpened)
        weight = torch.where(relay_valid, relayed["trust"], reliable.to(weak.dtype))
        source = self.teacher(images).softmax(1)
        return images, target, weight, source, relay_valid, weak

    def _strong_view(self, images):
        # RGB perturbation in pixel space, then restore ImageNet normalization.
        mean = images.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = images.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        pixels = (images * std + mean).clamp(0, 1)
        strength = 0.8 + 0.4 * torch.rand((len(images), 1, 1, 1), generator=self.generator)
        pixels = (pixels * strength.to(images.device)).clamp(0, 1)
        mask = torch.rand((len(images),), generator=self.generator) < 0.5
        pixels[mask.to(images.device)] = pixels[mask.to(images.device)].flip(-1)
        return (pixels - mean) / std

    @torch.enable_grad()
    def _step(self, with_current):
        size = self.config.train.train_bs
        self.model.train()
        supervised = current = source_loss = None
        loss = None
        if self.received:
            images, labels = self._replay(size)
            supervised = F.cross_entropy(self.model(images), labels)
            loss = supervised
        valid_count = overrides = 0
        if with_current:
            images, target, weight, source, relay_valid, weak = self._current_pack(size)
            # _current_pack runs EMA/source in eval; restore student training.
            self.model.train()
            probabilities = self.model(self._strong_view(images)).log_softmax(1)
            current = (F.kl_div(probabilities, target, reduction="none").sum(1) * weight).mean()
            # Source restraint only on unbridged images, so it cannot cancel
            # corrective relay targets by requiring source-class agreement.
            source_loss = (F.kl_div(probabilities, source, reduction="none").sum(1)
                           * (~relay_valid).to(probabilities.dtype)).mean()
            extra = DEFAULTS["current_weight"] * current + DEFAULTS["source_weight"] * source_loss
            loss = extra if loss is None else loss + extra
            valid_count = int(relay_valid.sum())
            overrides = int((relay_valid & (target.argmax(1) != weak.argmax(1))).sum())
        if loss is None:
            return False
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite FCRA-Relay loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.trainable, DEFAULTS["gradient_clip"], error_if_nonfinite=True)
        self.optimizer.step()
        self._ema_update()
        self.main_updates += 1
        self.current_updates += int(with_current)
        self.supervised_updates += int(supervised is not None)
        self.relay_used_samples += valid_count
        self.relay_overrides += overrides
        self.train_rows.append({"step": self.stream_step, "update": self.main_updates,
                                "with_current": with_current, "has_truth": supervised is not None,
                                "supervised_loss": float(supervised.detach()) if supervised is not None else None,
                                "current_loss": float(current.detach()) if current is not None else None,
                                "source_loss": float(source_loss.detach()) if source_loss is not None else None,
                                "relay_samples": valid_count, "relay_overrides": overrides,
                                "gradient_norm_before_clip": float(norm), "loss": float(loss.detach())})
        return True

    def train_batch(self, unused_audit_cost=0):
        if unused_audit_cost:
            raise ValueError("FCRA-Relay has no candidate audit updates")
        self._sync_memory()
        self.buffer.append((self.current_data.detach().cpu(), self.current_description["semantic"]))
        self.buffer_data = torch.cat([entry[0] for entry in self.buffer])
        self.buffer_semantic = torch.cat([entry[1] for entry in self.buffer])
        before = self.main_updates
        for _ in range(DEFAULTS["current_steps"]):
            self._step(with_current=True)
        for _ in range(DEFAULTS["replay_steps"]):
            if self.received:
                self._step(with_current=False)
        quota = DEFAULTS["current_steps"] + DEFAULTS["replay_steps"]
        used = self.main_updates - before
        if used > quota:
            raise AssertionError("Per-batch update quota exceeded")
        self.budget_rows.append({"step": self.stream_step, "quota": quota, "updates": used,
                                 "current_updates": DEFAULTS["current_steps"],
                                 "truth_available": bool(self.received), "unused": quota - used,
                                 **self.compute})
