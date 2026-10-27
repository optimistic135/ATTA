"""True supervision plus current images, optionally labeled by a feedback-repaired head.

All feedback tests precede assimilation. Current-view targets are cached before
fresh labels train the head; that repair first affects the next stream batch.
"""
import copy
from collections import deque
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.sp_atta.algorithm import SPATTA
from experiments.simatta_rootcause.algorithm import isolated_randomness, restore_random, fingerprint
from experiments.simatta_rootcause.spec import DEFAULTS as PROBE
from experiments.fcra_relay.stream import save_rows
from .spec import DEFAULTS as D, GROUPS


class FeedbackTeacher(SPATTA):
    def __init__(self, config, group, trace):
        if group not in GROUPS:
            raise ValueError("Unknown teacher-study group")
        # Keep the original SP gate and pure-truth fallback.
        super().__init__(config, "SP-ATTA", trace)
        self.study_group = group
        self.repair_head = copy.deepcopy(self.teacher[1]).to(config.device)
        self.repair_head.requires_grad_(True)
        self.origin_head = [p.detach().clone() for p in self.repair_head.parameters()]
        self.window = deque(maxlen=D["buffer_batches"])
        self.query_predictions, self.teacher_rows, self.current_rows = {}, [], []
        self.teacher_evidence = 0.0
        self.teacher_evidence_count = self.teacher_updates = 0
        self.source_correct = self.repair_correct = self.scored_queries = 0
        self.teacher_gain_sum = 0.0
        self.current_seen = self.current_eligible = self.repair_used = 0
        self.current = None
        self.student_updates = 0
        self.compute.update(head_forward_calls=0, head_forward_vectors=0,
                            head_backward_calls=0, head_backward_vectors=0)
        def count_head(module, inputs, output):
            self.compute["head_forward_calls"] += 1
            self.compute["head_forward_vectors"] += len(inputs[0])
            if torch.is_grad_enabled() and output.requires_grad:
                size = len(inputs[0])
                def backward(gradient):
                    self.compute["head_backward_calls"] += 1
                    self.compute["head_backward_vectors"] += size
                    return gradient
                output.register_hook(backward)
        self.repair_head.register_forward_hook(count_head)

    @torch.no_grad()
    def select_queries(self, data, stream_ids):
        output = super().select_queries(data, stream_ids)
        if self.study_group == "SP-ATTA":
            return output
        self.teacher.eval()
        self.repair_head.eval()
        weak = self.teacher[0](data)
        # Deterministic weak/strong pair; normalized RGB remains in normalized space.
        strong_data = torch.flip(data, dims=(-1,))
        strong = self.teacher[0](strong_data)
        source = self.teacher[1](weak).softmax(1)
        repaired = self.repair_head(weak).softmax(1)
        for row in self.trace.queries[self.step]:
            pos = int(row["sample_id"]) - int(stream_ids[0])
            self.query_predictions[int(row["anchor_index"])] = (
                source[pos].cpu(), repaired[pos].cpu())
        self.window.append((data.cpu(), strong_data.cpu(), stream_ids.clone(), weak.cpu(), strong.cpu()))
        return output

    @torch.no_grad()
    def _prepare_current(self, fresh_ids):
        # No true labels are read; exclude fresh validation images from candidates.
        weak = torch.cat([row[0] for row in self.window])
        strong = torch.cat([row[1] for row in self.window])
        ids = torch.cat([row[2] for row in self.window])
        wf = torch.cat([row[3] for row in self.window]).to(self.config.device)
        sf = torch.cat([row[4] for row in self.window]).to(self.config.device)
        source_w, source_s = self.teacher[1](wf).softmax(1), self.teacher[1](sf).softmax(1)
        repaired_w, repaired_s = self.repair_head(wf).softmax(1), self.repair_head(sf).softmax(1)
        use_repair = (self.study_group == "Current-Feedback" and self.teacher_evidence_count > 0
                      and self.teacher_evidence > 0)
        q, qs = (repaired_w, repaired_s) if use_repair else (source_w, source_s)
        mask = (q.max(1).values >= D["confidence"]) & (q.argmax(1) == qs.argmax(1))
        allowed = torch.tensor([int(sid) not in fresh_ids for sid in ids], device=mask.device)
        mask &= allowed
        self.current_seen += len(ids)
        self.current_eligible += int(mask.sum())
        self.repair_used += int(use_repair)
        self.current_pool = (strong, q.detach(), mask.detach())
        self._draw_current(0)
        self.current_rows.append({"step": self.step, "buffer_samples": len(ids),
            "eligible": int(mask.sum()), "training_eligible": int(self.current[2].sum()),
            "repaired_targets": use_repair, "prior_evidence": self.teacher_evidence,
            "excluded_fresh": int((~allowed).sum())})

    def _draw_current(self, iteration):
        # Shared private index schedule; masks/teacher targets may differ by arm.
        images, targets, mask = self.current_pool
        generator = torch.Generator().manual_seed(
            self.config.random_seed + 3000017 * self.step + 7919 * iteration + 83)
        positions = torch.randperm(len(images), generator=generator)[:self.config.train.train_bs]
        device_positions = positions.to(targets.device)
        self.current = (images[positions].to(self.config.device),
                        targets[device_positions], mask[device_positions])

    def on_labels_released(self, step, domain, released):
        if self.study_group == "SP-ATTA":
            return super().on_labels_released(step, domain, released)
        # Prequential teacher diagnosis: saved query predictions precede all use of y.
        with torch.no_grad():
            gains = []
            for index, label in released:
                sq, rq = self.query_predictions.pop(index)
                self.source_correct += int(sq.argmax().item() == label)
                self.repair_correct += int(rq.argmax().item() == label)
                self.scored_queries += 1
                image = self.query_anchors.data[index:index + 1].to(self.config.device)
                feature = self.teacher[0](image)
                source = self.teacher[1](feature).log_softmax(1)[0, label]
                repaired = self.repair_head(feature).log_softmax(1)[0, label]
                gains.append(float(repaired - source))
                self.teacher_rows.append({"step": step, "anchor_index": index,
                    "query_source_correct": int(sq.argmax().item() == label),
                    "query_repair_correct": int(rq.argmax().item() == label),
                    "pre_update_return_ce_gain": gains[-1]})
        fresh_ids = {int(self.query_anchors.target[index]) for index, _ in released}
        self._prepare_current(fresh_ids)
        self.lambda_batch = 0.0
        previous = sorted(self.received)
        eligible = bool(self.current[2].any())
        reason = ("no_fresh" if not released else "no_previous_truth" if not previous
                  else "no_current_targets" if not eligible
                  else "no_formal_quota" if not self.trace.quotas[step] else "eligible")
        row = {"step": step, "domain": domain, "fresh_count": len(released),
               "eligible": reason == "eligible", "reason": reason,
               # CSV columns must also exist before the first eligible gate.
               "true_candidate_ce": None, "aux_candidate_ce": None,
               "gain": None, "would_accept": False, "selected_lambda": 0.0}
        if reason == "eligible":
            with isolated_randomness() as state:
                row.update(self._gate(step, previous, released, None, state))
            self.gate_evaluations += 1
            self.gate_acceptances += int(row["would_accept"])
            self.lambda_batch = D["lambda_aux"] if row["would_accept"] else 0.0
        row["selected_lambda"] = self.lambda_batch
        self.gate_rows.append(row)
        self._fresh_this_batch = bool(released)
        # Fresh evidence changes teacher choice next batch, preserving the current
        # fresh set as a holdout for both candidate training and target selection.
        if gains:
            self.teacher_evidence = (D["teacher_evidence_decay"] * self.teacher_evidence
                + (1 - D["teacher_evidence_decay"]) * sum(gains) / len(gains))
            self.teacher_evidence_count += len(gains)
            self.teacher_gain_sum += sum(gains)

    @torch.enable_grad()
    def _loss(self, model, truth_images, truth_labels, pseudo_images, pseudo_labels, true_coef, pseudo_coef):
        if self.study_group == "SP-ATTA":
            return super()._loss(model, truth_images, truth_labels, pseudo_images, pseudo_labels,
                                 true_coef, pseudo_coef)
        true = F.cross_entropy(model(truth_images), truth_labels)
        pseudo = None
        if pseudo_coef > 0:
            images, targets, mask = self.current
            per_image = F.kl_div(model(images).log_softmax(1), targets, reduction="none").sum(1)
            # Normalize by sampled batch size, not selected count.
            pseudo = (per_image * mask).mean()
        loss = true + pseudo_coef * pseudo if pseudo is not None else true
        return loss, true, pseudo

    def _gate(self, step, previous, released, eligible_pseudo, state):
        if self.study_group == "SP-ATTA":
            return super()._gate(step, previous, released, eligible_pseudo, state)
        generator = torch.Generator().manual_seed(self.config.random_seed + 2000003 * step + 71)
        positions = torch.randint(len(previous), (self.config.train.train_bs,), generator=generator)
        ids = [previous[int(pos)] for pos in positions]
        images = self.query_anchors.data[ids].to(self.config.device)
        labels = torch.tensor([self.received[index] for index in ids], device=self.config.device)
        fresh_data = self.query_anchors.data[[index for index, _ in released]]
        fresh_labels = torch.tensor([label for _, label in released], dtype=torch.long)
        snapshot = copy.deepcopy(self.model).cpu()
        snapshot[0]._forward_hooks.clear()
        candidates, losses = {}, {}
        for name, coefficient in (("TRUE", 0.0), ("AUX", D["lambda_aux"])):
            restore_random(state)
            branch = copy.deepcopy(snapshot).to(self.config.device)
            self._counter(branch, "candidate")
            branch.train()
            optimizer = torch.optim.SGD(branch.parameters(), lr=self.config.atta.SimATTA.lr, momentum=0.9)
            with torch.enable_grad():
                loss, _, _ = self._loss(branch, images, labels, None, None, 1.0, coefficient)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite current candidate loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            self.candidate_updates += 1
            losses[name] = self._validation_loss(branch, fresh_data, fresh_labels)
            optimizer.zero_grad(set_to_none=True)
            candidates[name] = branch.cpu()
        gain = losses["TRUE"] - losses["AUX"]
        if not torch.isfinite(torch.tensor(gain)):
            raise FloatingPointError("Non-finite current gate gain")
        for start in PROBE["probe_starts"]:
            if start not in self.monitored and start <= step < start + PROBE["trigger_window"]:
                self.monitored.add(start)
                self.active.append({"id": start, "step": step, "models": candidates,
                                    "gain": gain, "accepted": gain > D["gain_margin"], "batches": 0})
                break
        if fingerprint(self.model) != fingerprint(snapshot):
            raise AssertionError("Shadow changed formal model")
        return {"true_candidate_ce": losses["TRUE"], "aux_candidate_ce": losses["AUX"],
                "gain": gain, "would_accept": gain > D["gain_margin"]}

    @torch.enable_grad()
    def cluster_train(self, target_anchors, source_anchors):
        if self.study_group == "SP-ATTA":
            before = self.main_updates
            super().cluster_train(target_anchors, source_anchors)
            self.student_updates += self.main_updates - before
            return
        ready = sorted(self.received)
        count_p = self.source_anchors.num_elem() if self.source_anchors is not None else 0
        quota = self.trace.quotas[self.step]
        head_steps = int(quota > 0 and self._fresh_this_batch)
        generator = torch.Generator().manual_seed(self.config.random_seed + 1000003 * self.step + 29)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.atta.SimATTA.lr, momentum=0.9)
        self.model.train()
        for iteration in range(quota):
            positions = torch.randint(len(ready), (self.config.train.train_bs,), generator=generator)
            # Consume the original pseudo draws, even though current targets replace them.
            ppos = (torch.randint(count_p, (self.config.train.train_bs,), generator=generator)
                    if count_p else torch.empty(0, dtype=torch.long))
            indices = torch.tensor(ready)[positions]
            self.formal_hash.update(f"{self.step}:{iteration}:".encode())
            self.formal_hash.update(indices.numpy().tobytes())
            self.formal_hash.update(ppos.numpy().tobytes())
            images = self.query_anchors.data[indices].to(self.config.device)
            labels = torch.tensor([self.received[int(index)] for index in indices], device=self.config.device)
            is_head = bool(head_steps and iteration == quota - 1)
            if is_head:
                # Teacher repair only after this batch's gate/student learning.
                with torch.no_grad():
                    features = self.teacher[0](images)
                self.repair_head.train()
                head_optimizer = torch.optim.SGD(self.repair_head.parameters(), lr=D["teacher_lr"])
                anchor = sum((p - origin).square().mean()
                             for p, origin in zip(self.repair_head.parameters(), self.origin_head))
                loss = F.cross_entropy(self.repair_head(features), labels) + D["teacher_anchor"] * anchor
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite teacher repair loss")
                head_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                head_optimizer.step()
                head_optimizer.zero_grad(set_to_none=True)
                self.repair_head.eval()
                self.teacher_updates += 1
                true, pseudo, coefficient = loss, None, 0.0
            else:
                coefficient = self.lambda_batch
                self._draw_current(iteration)
                loss, true, pseudo = self._loss(self.model, images, labels, None, None, 1.0, coefficient)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite student loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                self.student_updates += 1
                self.aux_updates += int(coefficient > 0)
            self.main_updates += 1  # total formal budget: student + head
            self.formal_by_step[self.step] += 1
            self.true_coefficient_sum += 1.0
            self.pseudo_coefficient_sum += coefficient
            self.update_rows.append({"step": self.step, "iteration": iteration,
                "update_kind": "teacher_head" if is_head else "student",
                "true_coefficient": 1.0, "pseudo_coefficient": coefficient,
                "true_loss": float(true.detach()),
                "pseudo_loss": float(pseudo.detach()) if pseudo is not None else None})

    def run_stream(self, output_dir, progress=None):
        result = super().run_stream(output_dir, progress)
        result.update(method=self.study_group, student_updates=self.student_updates,
            head_updates=self.teacher_updates,
            current_coverage_pct=100 * self.current_eligible / self.current_seen if self.current_seen else None,
            repair_use_pct=100 * self.repair_used / len(self.current_rows) if self.current_rows else None,
            source_query_accuracy_pct=100 * self.source_correct / self.scored_queries if self.scored_queries else None,
            repair_query_accuracy_pct=100 * self.repair_correct / self.scored_queries if self.scored_queries else None,
            repair_minus_source_query_pp=100 * (self.repair_correct - self.source_correct) / self.scored_queries if self.scored_queries else None,
            return_teacher_ce_gain=self.teacher_gain_sum / self.teacher_evidence_count if self.teacher_evidence_count else None,
            teacher_evaluated_returns=self.scored_queries)
        if self.student_updates + self.teacher_updates != sum(self.trace.quotas):
            raise AssertionError("Student + head budget mismatch")
        save_rows(Path(output_dir) / "teacher_feedback.csv", self.teacher_rows, ["step", "anchor_index"])
        save_rows(Path(output_dir) / "current_targets.csv", self.current_rows, ["step", "eligible"])
        return result
