"""Recorded-query FCRA with a slow supervised core and a fast EMA adapter."""

import copy
from collections import Counter
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from experiments.delayed_feedback.run import dump_json
from experiments.delayed_feedback.simatta import write_csv
from experiments.fcra.algorithm import FCRA
from .adapter import DEFAULTS, ExpertEvidence, ResidualAdapter


@torch.no_grad()
def gradient_norm(parameters):
    norms = [parameter.grad.detach().float().norm() for parameter in parameters
             if parameter.grad is not None]
    norm = torch.stack(norms).norm() if norms else torch.tensor(0.0)
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError("Non-finite optimization gradient")
    return float(norm)


class DualFCRA(FCRA):
    def __init__(self, config, delay, trace):
        super().__init__(config, delay)
        self.trace = trace
        structure = [(self.experiment.domains[env], min(config.atta.batch_size, len(loader.sampler) - start))
                     for env in config.dataset.test_envs[1:]
                     for loader in [self.fast_loader[env]]
                     for start in range(0, len(loader.sampler), config.atta.batch_size)]
        if structure != [(row["domain"], int(row["samples"])) for row in trace.batches]:
            raise ValueError("Current dataset stream and reference batch structure differ")
        self.fast_rng = torch.Generator().manual_seed((config.random_seed + 71933) % 2**32)
        self.adapter, self.adapter_ema, self.fast_optimizer = None, None, None
        if trace.fast_enabled:
            # Initialization must not consume the main model's CPU/CUDA RNG.
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(self.fast_rng.get_state())
                self.adapter = ResidualAdapter(self.fc.in_features)
                self.fast_rng.set_state(torch.get_rng_state())
            self.adapter = self.adapter.to(config.device)
            self.adapter_ema = copy.deepcopy(self.adapter).requires_grad_(False).eval()
            self.fast_optimizer = torch.optim.SGD(self.adapter.parameters(),
                                                 lr=DEFAULTS["fast_lr"], momentum=0)
        self.evidence = ExpertEvidence()
        self.observed, self.query_predictions = set(), {}
        self.expert_rows, self.fast_rows, self.budget_rows = [], [], []
        self.slow_done = self.fast_done = self.refund = 0
        self.slow_per_step = Counter()
        self.compute = Counter()
        self.encoder.register_forward_hook(self._forward_counter("slow"))
        self.teacher[0].register_forward_hook(self._forward_counter("teacher"))

    def _forward_counter(self, kind):
        def hook(module, inputs, output):
            samples = len(inputs[0])
            self.compute[f"{kind}_forward_calls"] += 1
            self.compute[f"{kind}_forward_images"] += samples
            if kind == "slow" and output.requires_grad:
                def count_backward(gradient):
                    self.compute["slow_backward_calls"] += 1
                    self.compute["slow_backward_images"] += samples
                    return gradient
                output.register_hook(count_backward)
        return hook

    def _fast_logits(self, features, adapter):
        # Shared classifier is a constant during the adapter-only backward pass.
        return F.linear(adapter(features), self.fc.weight.detach(), self.fc.bias.detach())

    @torch.no_grad()
    def select_queries(self, data, stream_ids):
        self.stream_step += 1
        self.current_data = data.detach()
        self.model.eval()
        self.teacher.eval()
        features = self.encoder(data)
        slow = self.fc(features).softmax(1)
        stable_raw = None
        if self.adapter_ema is not None:
            fast = self._fast_logits(features, self.adapter_ema).softmax(1)
            stable_raw = self.teacher[0](data)
            gamma = self.evidence.weights(F.normalize(stable_raw, dim=1), self.stream_step)
        else:
            fast, gamma = slow, slow.new_zeros(len(data))
        mixed = (1 - gamma[:, None]) * slow + gamma[:, None] * fast
        if not bool(torch.isfinite(mixed).all()):
            raise FloatingPointError("Non-finite dual-branch prediction")
        self.last_prediction = (slow, fast, mixed, gamma)
        records = self.trace.queries[self.stream_step]
        if records:
            positions = torch.tensor([int(row["sample_id"]) - int(stream_ids[0])
                                      for row in records], dtype=torch.long)
            device_positions = positions.to(data.device)
            previous = self.query_anchors.num_elem() if self.query_anchors is not None else 0
            if [int(row["anchor_index"]) for row in records] != list(range(previous, previous + len(records))):
                raise ValueError("Reference anchor insertion order changed")
            self.query_anchors = self.update_anchors(
                self.query_anchors, data[device_positions].cpu(), stream_ids[positions],
                features[device_positions].cpu(), torch.ones(len(positions)))
            teacher_raw = (stable_raw[device_positions] if stable_raw is not None
                           else self.teacher[0](data[device_positions]))
            teacher_prediction = self.teacher[1](teacher_raw).argmax(1)
            teacher_features = F.normalize(teacher_raw, dim=1).cpu()
            for j, position in enumerate(positions.tolist()):
                index = previous + j
                self.teacher_predictions[index] = int(teacher_prediction[j])
                self.teacher_features[index] = teacher_features[j]
                self.query_predictions[index] = (slow[position].cpu(), fast[position].cpu())
        return mixed

    @torch.no_grad()
    def _observe_returns(self):
        indices = sorted(set(self.received) - self.observed)
        for start in range(0, len(indices), self.config.atta.batch_size):
            ids = indices[start:start + self.config.atta.batch_size]
            self.model.eval()
            features = self.encoder(self.query_anchors.data[ids].to(self.config.device))
            slow = self.fc(features).softmax(1)
            fast = (self._fast_logits(features, self.adapter_ema).softmax(1)
                    if self.adapter_ema is not None else slow)
            for j, index in enumerate(ids):
                label = self.received[index]
                log_slow = slow[j, label].clamp_min(DEFAULTS["epsilon"]).log()
                log_fast = fast[j, label].clamp_min(DEFAULTS["epsilon"]).log()
                gain = float((log_fast - log_slow).clamp(-DEFAULTS["gain_clip"], DEFAULTS["gain_clip"]))
                if self.adapter_ema is not None:
                    self.evidence.add(self.teacher_features[index], gain, self.stream_step)
                old_slow, old_fast = self.query_predictions[index]
                record = self.oracle.records[index]
                self.expert_rows.append({
                    "sample_id": int(self.query_anchors.target[index]), "anchor_index": index,
                    "query_step": record.query_step, "return_step": self.stream_step,
                    "label": label, "slow_probability_at_query": float(old_slow[label]),
                    "fast_probability_at_query": float(old_fast[label]),
                    "slow_probability_at_return": float(slow[j, label]),
                    "fast_probability_at_return": float(fast[j, label]),
                    "gain_at_return": gain,
                })
        self.observed.update(indices)
        self._record_calibration_events()

    @torch.enable_grad()
    def _train_slow(self, quota):
        if not quota:
            return
        if not self.received:
            raise ValueError("Supervised budget was scheduled before the first label")
        indices, data, labels = self._supervised_memory()
        need, relevance, score, probabilities = self._replay_statistics(data, labels)
        reference = self.source_anchors
        pseudo_count = reference.num_elem() if reference is not None else 0
        alpha = len(indices) / (len(indices) + pseudo_count) if pseudo_count else 1.0
        if pseudo_count and pseudo_count < self.cold_start:
            alpha = min(0.2, alpha)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.atta.SimATTA.lr, momentum=0.9)
        draws = torch.zeros(len(indices))
        size = self.config.train.train_bs
        for _ in range(quota):
            sampled = torch.multinomial(probabilities, size, replacement=True)
            self._audit_supervised_draw(sampled)
            draws.index_add_(0, sampled, torch.ones(size))
            supervised_loss = F.cross_entropy(self.model(data[sampled].to(self.config.device)),
                                               labels[sampled].to(self.config.device))
            pseudo_loss, gates = None, None
            if pseudo_count:
                sampled_pseudo = torch.randint(0, pseudo_count, (size,))
                pseudo_labels = reference.target[sampled_pseudo].to(self.config.device).long()
                gates = self._feedback_gates(F.normalize(reference.feats[sampled_pseudo].to(
                    self.config.device), dim=1), pseudo_labels)
                pseudo_losses = F.cross_entropy(self.model(reference.data[sampled_pseudo].to(
                    self.config.device)), pseudo_labels, reduction="none")
                gates = self._audit_pseudo_draw(sampled_pseudo, gates, pseudo_losses)
                pseudo_loss = (gates * pseudo_losses).mean()
                loss = alpha * supervised_loss + (1 - alpha) * pseudo_loss
            else:
                loss = supervised_loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite slow loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = gradient_norm(self.model.parameters())
            optimizer.step()
            iteration = self.slow_per_step[self.stream_step]
            self.slow_per_step[self.stream_step] += 1
            self.slow_done += 1
            self.update_rows.append({
                "step": self.stream_step, "iteration": iteration, "returned_memory": len(indices),
                "replay_batch": size, "source_memory": pseudo_count, "alpha": alpha,
                "supervised_loss": float(supervised_loss.detach()),
                "pseudo_loss": float(pseudo_loss.detach()) if pseudo_count else None,
                "gate_mean": float(gates.mean()) if pseudo_count else None,
                "gradient_norm": norm,
            })
        self.replay_rows.append({
            "step": self.stream_step, "returned_memory": len(indices), "updates": quota,
            "coverage": float((draws > 0).sum()) / len(indices),
            "effective_sample_size": float(1 / probabilities.square().sum()),
            "need_mean": float(need.mean()), "relevance_mean": float(relevance.mean()),
            "score_mean": float(score.mean()),
        })

    def _audit_pseudo_draw(self, indices, gates, losses):
        """Identity hook; diagnostic subclasses may log or intervene explicitly."""
        return gates

    def _audit_supervised_draw(self, indices):
        return None

    def _audit_observe_batch(self, step, domain, ids, truth):
        """Evaluation-only hook. Production training must never consume truth here."""
        return None

    def _strong_view(self, images):
        mean = images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        pixels = (images * std + mean).clamp(0, 1)
        flip = (torch.rand(len(images), generator=self.fast_rng) < 0.5).to(images.device)
        pixels = torch.where(flip[:, None, None, None], pixels.flip(-1), pixels)
        factors = 0.8 + 0.4 * torch.rand(3, generator=self.fast_rng)
        hue = float(-0.05 + 0.1 * torch.rand((), generator=self.fast_rng))
        operations = [(TF.adjust_brightness, float(factors[0])),
                      (TF.adjust_contrast, float(factors[1])),
                      (TF.adjust_saturation, float(factors[2])), (TF.adjust_hue, hue)]
        for index in torch.randperm(4, generator=self.fast_rng).tolist():
            operation, factor = operations[index]
            pixels = operation(pixels, factor)
        return (pixels - mean) / std

    @torch.enable_grad()
    def _train_fast(self):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        with torch.no_grad():
            weak = self.encoder(self.current_data)
            strong = self.encoder(self._strong_view(self.current_data))
            slow_target = self.fc(weak).softmax(1)
            weak_logits = self._fast_logits(weak, self.adapter_ema)
            strong_logits = self._fast_logits(strong, self.adapter_ema)
            unsharpened = weak_logits.softmax(1)
            mask = ((unsharpened.max(1).values >= DEFAULTS["confidence"])
                    & (weak_logits.argmax(1) == strong_logits.argmax(1)))
            target = (weak_logits / DEFAULTS["target_temperature"]).softmax(1)
        if not bool(mask.any()):
            self.fast_rows.append({"step": self.stream_step, "updated": False,
                                   "reliable_samples": 0, "consistency_loss": None,
                                   "anchor_loss": None, "gradient_norm": None})
            return False
        self.adapter.train()
        consistency = (mask * F.kl_div(self._fast_logits(strong, self.adapter).log_softmax(1),
                                      target, reduction="none").sum(1)).mean()
        anchor = F.kl_div(self._fast_logits(weak, self.adapter).log_softmax(1),
                          slow_target, reduction="none").sum(1).mean()
        loss = consistency + DEFAULTS["anchor_weight"] * anchor
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite adapter loss")
        self.fast_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(parameter.grad is not None for parameter in self.model.parameters()):
            raise RuntimeError("Adapter-only update leaked gradients into the core")
        norm = gradient_norm(self.adapter.parameters())
        self.fast_optimizer.step()
        with torch.no_grad():
            for average, parameter in zip(self.adapter_ema.parameters(), self.adapter.parameters()):
                average.mul_(DEFAULTS["ema"]).add_(parameter, alpha=1 - DEFAULTS["ema"])
        self.fast_done += 1
        self.fast_rows.append({"step": self.stream_step, "updated": True,
                               "reliable_samples": int(mask.sum()),
                               "consistency_loss": float(consistency.detach()),
                               "anchor_loss": float(anchor.detach()), "gradient_norm": norm})
        return True

    @torch.no_grad()
    def run_stream(self, output_dir, progress=None):
        output = Path(output_dir)
        self.enable_bn(self.model)
        batch_rows, domain_rows = [], []
        step, offset = 0, 0
        started = time.perf_counter()
        for env in self.config.dataset.test_envs[1:]:
            domain = self.experiment.domains[env]
            correct_total = sample_total = early_correct = early_samples = 0
            loader = self.fast_loader[env]
            if progress:
                progress({"event": "domain", "domain": domain, "total": len(loader)})
            for domain_step, (data, labels) in enumerate(loader):
                before = self.compute.copy()
                batch_started = time.perf_counter()
                data = data.to(self.config.device)
                ids = torch.arange(offset, offset + len(data))
                if (domain != self.trace.batches[step]["domain"]
                        or (offset, offset + len(data)) != self.trace.bounds[step]):
                    raise ValueError("Current stream differs from reference")
                previous = self.query_anchors.num_elem() if self.query_anchors is not None else 0
                probabilities = self.select_queries(data, ids)
                slow, fast, mixed, gamma = self.last_prediction
                truth = labels.cpu()
                correct = int((probabilities.argmax(1).cpu() == truth).sum())
                self._audit_observe_batch(step, domain, ids, truth)
                correct_total += correct
                sample_total += len(data)
                if domain_step < 10:
                    early_correct += correct
                    early_samples += len(data)
                current = self.query_anchors.num_elem() if self.query_anchors is not None else 0
                for index in range(previous, current):
                    sample_id = int(self.query_anchors.target[index])
                    self.oracle.query(sample_id, index, int(truth[sample_id - offset]), step, domain)
                self.budgets += current - previous
                if self.LE:
                    _, _, self.source_anchors = self.sample_select(
                        self.teacher, data, ids, self.source_anchors, self.source_n_clusters, 0,
                        use_pseudo_label=True, ent_bound=self.config.atta.SimATTA.el,
                        incremental_cluster=False)
                if not self.target_cluster:
                    self.n_clusters = 0
                self.n_clusters += self.nc_increase
                self.source_n_clusters = 101
                released = self.oracle.release(step, domain)
                self.received.update(released)
                self._observe_returns()
                self.anchors = self.labeled_anchors()
                slow_quota = self.trace.slow[step]
                if self.received:
                    slow_quota += self.refund
                    self.refund = 0
                self._train_slow(slow_quota)
                skipped = 0
                if self.trace.fast[step] and not self._train_fast():
                    skipped = 1
                    self.refund += 1
                if step == self.trace.steps - 1 and self.refund and self.received:
                    self._train_slow(self.refund)
                    self.refund = 0
                if self.anchors is not None:
                    self.query_anchors = self.update_anchors_feats(self.query_anchors)
                phase = ("before_first_feedback" if self.trace.first_return is None
                         or step <= self.trace.first_return else "after_first_feedback")
                batch_rows.append({
                    "seed": self.config.random_seed, "delay": self.oracle.delay,
                    "step": step, "domain": domain, "domain_step": domain_step,
                    "samples": len(data), "correct": correct, "accuracy_pct": 100 * correct / len(data),
                    "slow_correct": int((slow.argmax(1).cpu() == truth).sum()),
                    "fast_correct": int((fast.argmax(1).cpu() == truth).sum()),
                    "gamma_mean": float(gamma.mean()), "phase": phase,
                    "new_queries": current - previous, "new_returns": len(released), **self.oracle.counts(),
                })
                torch.cuda.synchronize(self.config.device)
                self.budget_rows.append({
                    "step": step, "reference_updates": self.trace.old_updates[step],
                    "slow_quota": self.trace.slow[step], "fast_quota": self.trace.fast[step],
                    "slow_actual": self.slow_per_step[step], "fast_actual": self.trace.fast[step] - skipped,
                    "refund_pending": self.refund, "elapsed_seconds": time.perf_counter() - batch_started,
                    **{key: self.compute[key] - before[key] for key in (
                        "slow_forward_calls", "slow_forward_images", "teacher_forward_calls",
                        "teacher_forward_images", "slow_backward_calls", "slow_backward_images")},
                })
                if progress:
                    progress({"event": "batch", "domain": domain, "total": len(loader),
                              "completed": domain_step + 1, "accuracy_pct": 100 * correct_total / sample_total,
                              **self.oracle.counts()})
                offset += len(data)
                step += 1
            domain_rows.append({"seed": self.config.random_seed, "delay": self.oracle.delay,
                                "domain": domain, "samples": sample_total, "correct": correct_total,
                                "batches": len(loader), "accuracy_pct": 100 * correct_total / sample_total,
                                "first10_accuracy_pct": 100 * early_correct / early_samples})
        if step != self.trace.steps or self.slow_done + self.fast_done != self.trace.total or self.refund:
            raise ValueError("Actual optimization count did not match the recorded total budget")
        audit = self.oracle.audit()
        self.trace.verify_feedback(audit)
        for row in domain_rows:
            row.update(self.oracle.counts(row["domain"]))
        result = {
            "method": "FCRA-dual", "seed": self.config.random_seed, "delay": self.oracle.delay,
            "fast_adapter_enabled": self.adapter is not None,
            **{f"{row['domain']}_accuracy_pct": row["accuracy_pct"] for row in domain_rows},
            **{f"{row['domain']}_first10_accuracy_pct": row["first10_accuracy_pct"] for row in domain_rows},
            "stream_accuracy_pct": 100 * sum(row["correct"] for row in domain_rows) / offset,
            "domain_macro_accuracy_pct": float(np.mean([row["accuracy_pct"] for row in domain_rows])),
            "reference_updates": self.trace.total, "slow_updates": self.slow_done,
            "fast_updates": self.fast_done, "elapsed_seconds": time.perf_counter() - started,
            **{key: self.compute[key] for key in (
                "slow_forward_calls", "slow_forward_images", "teacher_forward_calls",
                "teacher_forward_images", "slow_backward_calls", "slow_backward_images")},
            **self.oracle.counts(),
        }
        if self.oracle.delay == 0 and self.adapter is None:
            result["reference_d0_batch_correct_match"] = all(
                actual["correct"] == int(reference["correct"])
                for actual, reference in zip(batch_rows, self.trace.batches))
        else:
            result["reference_d0_batch_correct_match"] = None
        phases = []
        for phase, metric in (("before_first_feedback", "pre_feedback_accuracy_pct"),
                              ("after_first_feedback", "post_feedback_accuracy_pct")):
            subset = [row for row in batch_rows if row["phase"] == phase]
            samples = sum(row["samples"] for row in subset)
            accuracy = 100 * sum(row["correct"] for row in subset) / samples if samples else None
            result[metric] = accuracy
            phases.append({"phase": phase, "samples": samples, "accuracy_pct": accuracy})
        for name, rows, header in (
            ("batches.csv", batch_rows, "step,samples,correct"),
            ("domains.csv", domain_rows, "domain,samples,correct"),
            ("phases.csv", phases, "phase,samples,accuracy_pct"),
            ("feedback.csv", audit, "sample_id,anchor_index,query_step,query_domain,due_step,return_step,return_domain,cross_domain"),
            ("expert_events.csv", self.expert_rows, "sample_id,anchor_index,query_step,return_step,label,slow_probability_at_query,fast_probability_at_query,slow_probability_at_return,fast_probability_at_return,gain_at_return"),
            ("calibration.csv", self.calibration_rows, "step,anchor_index,sample_id,teacher_prediction,label,correct,cache_size,query_step,return_step,query_domain,return_domain"),
            ("replay.csv", self.replay_rows, "step,returned_memory,updates,coverage,effective_sample_size,need_mean,relevance_mean,score_mean"),
            ("updates.csv", self.update_rows, "step,iteration,returned_memory,replay_batch,source_memory,alpha,supervised_loss,pseudo_loss,gate_mean,gradient_norm"),
            ("adapter_updates.csv", self.fast_rows, "step,updated,reliable_samples,consistency_loss,anchor_loss,gradient_norm"),
            ("budget.csv", self.budget_rows, "step,reference_updates,slow_quota,fast_quota,slow_actual,fast_actual,refund_pending,elapsed_seconds"),
        ):
            if rows:
                write_csv(output / name, rows)
            else:
                (output / name).write_text(header + "\n", encoding="utf-8")
        dump_json(output / "budget_result.json", {
            "reference_total": self.trace.total, "reserved_fast": self.trace.fast_total,
            "actual_slow": self.slow_done, "actual_fast": self.fast_done,
            "annotation_schedule_matched": True, "reference_hashes": self.trace.hashes,
        })
        return result
