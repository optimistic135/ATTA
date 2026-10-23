"""Historical transport selected by delayed, isolated short-update audits."""

from collections import deque
import copy
import json
import random

import numpy as np
import torch
from torch.nn import functional as F

from experiments.fcra.algorithm import FCRA
from .features import describe, layer, transport
from .policy import ACTIONS, DEFAULTS, TransportPolicy


class CalibratedTransport(FCRA):
    def __init__(self, config, delay):
        super().__init__(config, delay)
        layer(self.model)  # Fail clearly for unsupported backbones.
        self.ema = copy.deepcopy(self.model).eval()
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=config.atta.SimATTA.lr, momentum=0.9)
        self.policy = TransportPolicy()
        self.styles = deque(maxlen=DEFAULTS["window_samples"])
        self.pending_audits = {}
        self.memory_cache = None
        self.audit_predictions, self.reward_rows, self.decision_rows = [], [], []
        self.budget_rows = []
        self.generator = torch.Generator().manual_seed(config.random_seed + 17)
        self.audit_generator = torch.Generator().manual_seed(config.random_seed + 71)
        self.main_updates = self.audit_updates = 0
        self.transport_updates = 0
        self.compute = {f"{name}_encoder_{operation}_{unit}": 0
                        for name in ("main", "source", "ema", "audit")
                        for operation in ("forward", "backward")
                        for unit in ("calls", "images")}
        # Hooks on the complete encoder include inherited selection, replay
        # scoring and anchor refresh calls. Audit copies get separate hooks.
        self._install_counters(self.model, "main")
        self._install_counters(self.teacher, "source")
        self._install_counters(self.ema, "ema")

    def _install_counters(self, model, name):
        def forward(module, inputs, output):
            key = name + "_encoder_forward_images"
            self.compute[key] = self.compute.get(key, 0) + len(inputs[0])
            key = name + "_encoder_forward_calls"
            self.compute[key] = self.compute.get(key, 0) + 1
            if torch.is_grad_enabled() and output.requires_grad:
                size = len(inputs[0])
                def backward(gradient):
                    key = name + "_encoder_backward_images"
                    self.compute[key] = self.compute.get(key, 0) + size
                    key = name + "_encoder_backward_calls"
                    self.compute[key] = self.compute.get(key, 0) + 1
                    return gradient
                output.register_hook(backward)
        model[0].register_forward_hook(forward)

    @torch.no_grad()
    def select_current(self, data, physical_ids):
        """Predict all encounters, but query each physical image at most once."""
        self.model.eval()
        outputs = self.model(data)
        seen = set() if self.query_anchors is None else set(self.query_anchors.target.tolist())
        keep = torch.tensor([int(sid) not in seen for sid in physical_ids], dtype=torch.bool)
        if bool(keep.any()):
            super().select_queries(data[keep.to(data.device)], physical_ids[keep])
        else:
            self.stream_step += 1
            self.query_rows.append({
                "step": self.stream_step, "pending_before_release": self.oracle.counts()["pending"],
                "eligible_before": 0, "eligible_after": 0, "suppressed": 0,
                "new_queries": 0, "mean_redundancy": 0.0,
            })
        self.current_data = data.detach()
        return outputs

    @torch.no_grad()
    def observe(self, data):
        probabilities, mean, logstd = describe(self.ema, data)
        # Flip is deterministic and consumes no training random-number stream.
        flipped = self.ema(data.flip(-1)).softmax(1)
        self.teacher.eval()
        source = F.normalize(self.teacher[0](data), dim=1)
        confidence, predicted = probabilities.max(1)
        consistent = predicted == flipped.argmax(1)
        for i in range(len(data)):
            self.styles.append({
                "step": self.stream_step, "class": int(predicted[i]),
                "confidence": float(confidence[i]),
                "trusted": bool(consistent[i] and confidence[i] >= DEFAULTS["confidence"]),
                "mean": mean[i].cpu(), "logstd": logstd[i].cpu(),
                "semantic": source[i].cpu(),
                "uncertainty": float(-(probabilities[i] * probabilities[i].clamp_min(1e-12).log()).sum()
                                     / np.log(probabilities.shape[1])),
            })
        while self.styles and self.stream_step - self.styles[0]["step"] >= DEFAULTS["window_batches"]:
            self.styles.popleft()
        # Entire current batch is at the end of the window (batch size <= window).
        return list(self.styles)[-len(data):]

    @staticmethod
    def _context(donor, old_mean, old_logstd, support):
        # Symmetric scale normalization bounds shifts of high-magnitude channels.
        scale = (old_logstd.exp() + donor["logstd"].exp()).clamp_min(1e-6)
        difference = ((old_mean - donor["mean"]) / scale).square().mean()
        difference += (old_logstd - donor["logstd"]).square().mean()
        distance = float(difference.sqrt())
        return {"class": donor["class"], "distance": distance / (1 + distance),
                "reliability": donor["confidence"] * min(1.0, support / 4),
                "uncertainty": donor["uncertainty"], "step": donor["step"]}

    @torch.no_grad()
    def _prepare(self, generator, forced=None):
        # Returned labels are append-only; avoid copying the entire image memory
        # on every optimizer step. A release invalidates the cached length.
        if self.memory_cache is None or len(self.memory_cache[0]) != len(self.received):
            self.memory_cache = self._supervised_memory() if self.received else ([], None, None)
        indices, data, labels = self.memory_cache
        if not indices:
            return None
        donors = [item for item in self.styles if item["trusted"]]
        size = self.config.train.train_bs
        # Uniform full-memory support plus category/semantic retrieval.
        sampled = torch.randint(len(indices), (size,), generator=generator)
        if forced is not None:
            eligible = (labels == forced["class"]).nonzero().flatten()
            if not len(eligible):
                return None
            positions = torch.stack([self.teacher_features[indices[int(j)]] for j in eligible])
            nearest = eligible[(positions @ forced["semantic"]).topk(min(4, len(eligible))).indices]
            sampled = nearest[torch.randint(len(nearest), (size,), generator=generator)]
        elif donors:
            for k in range(size // 4):
                donor = donors[int(torch.randint(len(donors), (1,), generator=generator))]
                eligible = (labels == donor["class"]).nonzero().flatten()
                if len(eligible):
                    positions = torch.stack([self.teacher_features[indices[int(j)]] for j in eligible])
                    sampled[k] = eligible[(positions @ donor["semantic"]).argmax()]
        images = data[sampled].to(self.config.device)
        targets = labels[sampled].to(self.config.device)
        _, old_mean, old_logstd = describe(self.ema, images)
        means, logstds, contexts, matched = [], [], [], []
        for k, j in enumerate(sampled.tolist()):
            pool = ([forced] if forced is not None else
                    [item for item in donors if item["class"] == int(labels[j])])
            if pool:
                semantic = self.teacher_features[indices[j]]
                donor = max(pool, key=lambda item: float(item["semantic"] @ semantic))
                support = sum(item["class"] == donor["class"] for item in donors)
                context = self._context(donor, old_mean[k].cpu(), old_logstd[k].cpu(), support)
                means.append(donor["mean"])
                logstds.append(donor["logstd"])
                matched.append(True)
            else:
                means.append(old_mean[k].cpu())
                logstds.append(old_logstd[k].cpu())
                context = {"class": int(labels[j]), "distance": 0.0, "reliability": 0.0,
                           "uncertainty": 1.0, "step": self.stream_step}
                matched.append(False)
            # Decision context belongs to the CURRENT interaction, not donor age.
            context["step"] = self.stream_step
            contexts.append(context)
        pseudo = None
        reference = self.source_anchors
        pseudo_count = reference.num_elem() if reference is not None else 0
        alpha = len(indices) / (len(indices) + pseudo_count) if pseudo_count else 1.0
        if pseudo_count and pseudo_count < self.cold_start:
            alpha = min(0.2, alpha)
        if pseudo_count:
            pos = torch.randint(pseudo_count, (size,), generator=generator)
            pseudo_labels = reference.target[pos].to(self.config.device).long()
            gates = self._feedback_gates(F.normalize(reference.feats[pos].to(self.config.device), dim=1),
                                         pseudo_labels)
            pseudo = (reference.data[pos].to(self.config.device), pseudo_labels, gates)
        return {"images": images, "labels": targets,
                "mean": torch.stack(means).to(self.config.device),
                "logstd": torch.stack(logstds).to(self.config.device),
                "contexts": contexts, "matched": matched, "alpha": alpha, "pseudo": pseudo}

    @torch.enable_grad()
    def _update(self, model, optimizer, pack, actions):
        model.train()
        images, labels = pack["images"], pack["labels"]
        original = F.cross_entropy(model(images), labels, reduction="none")
        supervised = original.clone()
        for action in (1, 2):
            positions = [i for i, selected in enumerate(actions)
                         if selected == action and pack["matched"][i]]
            if not positions:
                continue
            with transport(model, ACTIONS[action], pack["mean"][positions], pack["logstd"][positions]):
                migrated = F.cross_entropy(model(images[positions]), labels[positions], reduction="none")
            supervised[positions] = ((1 - DEFAULTS["rho"]) * original[positions]
                                     + DEFAULTS["rho"] * migrated)
        supervised = supervised.mean()
        loss = supervised
        if pack["pseudo"] is not None:
            images_p, labels_p, gates = pack["pseudo"]
            pseudo = (F.cross_entropy(model(images_p), labels_p, reduction="none") * gates).mean()
            loss = pack["alpha"] * supervised + (1 - pack["alpha"]) * pseudo
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite transport loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    @torch.no_grad()
    def _ema_update(self):
        for target, source in zip(self.ema.parameters(), self.model.parameters()):
            target.lerp_(source, 1 - DEFAULTS["ema"])
        for target, source in zip(self.ema.buffers(), self.model.buffers()):
            target.copy_(source)
        self.ema.eval()

    def audit(self, new_indices, current_ids, current_descriptions, step):
        """Run before oracle release, including D=0. No queried truth is passed."""
        used = 0
        if step % DEFAULTS["audit_every"] or not self.received:
            return used
        lookup = {int(sid): item for sid, item in zip(current_ids, current_descriptions)}
        python_state, numpy_state = random.getstate(), np.random.get_state()
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        try:
            with torch.random.fork_rng(devices=devices):
                base_cpu = torch.get_rng_state()
                base_cuda = torch.cuda.get_rng_state() if devices else None
                for index in list(new_indices)[:DEFAULTS["audit_samples"]]:
                    sid = int(self.query_anchors.target[index])
                    query = lookup[sid]
                    pool = [item for item in self.styles
                            if item["trusted"] and item["class"] == query["class"]]
                    if not pool:
                        continue
                    donor = max(pool, key=lambda item: float(item["semantic"] @ query["semantic"]))
                    packs = [self._prepare(self.audit_generator, forced=donor)
                             for _ in range(DEFAULTS["audit_steps"])]
                    if any(pack is None for pack in packs):
                        continue
                    cost = len(ACTIONS) * len(packs)
                    if used + cost > DEFAULTS["steps_per_batch"]:
                        break
                    context = dict(packs[0]["contexts"][0])
                    # The candidate update uses a minibatch of history images;
                    # describe the whole intervention instead of its first draw.
                    for key in ("distance", "reliability", "uncertainty"):
                        context[key] = float(np.mean([entry[key] for entry in packs[0]["contexts"]]))
                    chosen, estimate = self.policy.choose(context, step)
                    predictions = []
                    for action in range(len(ACTIONS)):
                        random.setstate(python_state)
                        np.random.set_state(numpy_state)
                        torch.set_rng_state(base_cpu)
                        if devices:
                            torch.cuda.set_rng_state(base_cuda)
                        branch = copy.deepcopy(self.model)
                        # Remove copied encoder counters, then count audit work.
                        branch[0]._forward_hooks.clear()
                        self._install_counters(branch, "audit")
                        optimizer = torch.optim.SGD(branch.parameters(),
                                                    lr=self.config.atta.SimATTA.lr, momentum=0.9)
                        optimizer.load_state_dict(copy.deepcopy(self.optimizer.state_dict()))
                        for pack in packs:
                            self._update(branch, optimizer, pack, [action] * len(pack["labels"]))
                            self.audit_updates += 1
                        branch.eval()
                        with torch.no_grad():
                            image = self.query_anchors.data[index:index + 1].to(self.config.device)
                            probability = branch(image).softmax(1)[0].cpu()
                        predictions.append(probability)
                        self.audit_predictions.append({
                            "step": step, "anchor_index": index, "sample_id": sid,
                            "action": action, "beta": ACTIONS[action],
                            "context": json.dumps(context, sort_keys=True),
                            "probabilities": json.dumps(probability.tolist()),
                            "candidate_predicted_reward": estimate["action_rewards"][action],
                            "candidate_score": estimate["action_scores"][action],
                            "chosen_before_reward": chosen, **estimate,
                        })
                        del optimizer, branch
                    self.pending_audits[index] = (context, torch.stack(predictions), chosen, estimate)
                    used += cost
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        return used

    def on_labels_released(self, step, domain, released):
        for index, label in released:
            event = self.pending_audits.pop(index, None)
            if event is None:
                continue
            context, probabilities, chosen, estimate = event
            logs = probabilities[:, int(label)].clamp_min(1e-12).log()
            rewards = (logs - logs[0]).clamp(-DEFAULTS["reward_clip"], DEFAULTS["reward_clip"])
            # Log pre-event decisions before adding the new reward to policy.
            for action in range(len(ACTIONS)):
                self.reward_rows.append({
                    "return_step": step, "audit_step": context["step"],
                    "anchor_index": index, "label": int(label),
                    "predicted_class_at_audit": context["class"],
                    "action": action, "beta": ACTIONS[action], "reward": float(rewards[action]),
                    "chosen_before_reward": chosen,
                    "predicted_reward": estimate["action_rewards"][action],
                    "predicted_score": estimate["action_scores"][action],
                    "predicted_support": estimate["support"],
                })
            self.policy.add(context, rewards.numpy())

    def train_batch(self, audit_cost):
        quota = DEFAULTS["steps_per_batch"]
        used = 0
        self._record_calibration_events()
        if self.received:
            for iteration in range(quota - audit_cost):
                pack = self._prepare(self.generator)
                actions, supports, scores = [], [], []
                for context, matched in zip(pack["contexts"], pack["matched"]):
                    action, estimate = self.policy.choose(context, self.stream_step)
                    actions.append(action if matched else 0)
                    supports.append(estimate["support"])
                    scores.append(estimate["score"])
                loss = self._update(self.model, self.optimizer, pack, actions)
                transported = sum(action != 0 for action in actions)
                self.transport_updates += int(transported > 0)
                self.main_updates += 1
                used += 1
                self._ema_update()
                self.decision_rows.append({
                    "step": self.stream_step, "iteration": iteration, "loss": loss,
                    "alpha": pack["alpha"], "matched": sum(pack["matched"]),
                    "original": actions.count(0), "small": actions.count(1),
                    "medium": actions.count(2), "evidence": len(self.policy.events),
                    "support_mean": float(np.mean(supports)),
                    "selected_score_mean": float(np.mean(scores)),
                    "returned_memory": len(self.received),
                })
        if used + audit_cost > quota:
            raise AssertionError("Combined update quota exceeded")
        self.budget_rows.append({"step": self.stream_step, "quota": quota,
                                 "main_updates": used, "audit_updates": audit_cost,
                                 "unused": quota - used - audit_cost, **self.compute})
