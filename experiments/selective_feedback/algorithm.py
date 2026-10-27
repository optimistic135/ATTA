"""Selective current targets; keep SP's historical path and formal budget cap."""
import copy
import hashlib
import json
from collections import deque
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.sp_atta.algorithm import SPATTA
from experiments.simatta_rootcause.algorithm import isolated_randomness, restore_random, fingerprint
from experiments.fcra_relay.stream import save_rows
from .reliability import descriptors, correctness_loss
from .spec import DEFAULTS as D, GROUPS


class SelectiveFeedback(SPATTA):
    def __init__(self, config, group, trace):
        super().__init__(config, "SP-ATTA", trace)
        if group not in GROUPS:
            raise ValueError("Unknown selective-feedback arm")
        self.study_group = group
        self.head = copy.deepcopy(self.teacher[1]).to(config.device)
        self.head.requires_grad_(True)
        self.origin = [p.detach().clone() for p in self.head.parameters()]
        with isolated_randomness():
            self.risk = torch.nn.Linear(2 * config.dataset.num_classes + 8, 2).to(config.device)
            torch.nn.init.zeros_(self.risk.weight)
            torch.nn.init.zeros_(self.risk.bias)
        self.window = deque(maxlen=D["buffer_batches"])
        self.query_records, self.risk_memory = {}, []
        self.feedback_rows, self.selection_rows, self.target_rows = [], [], []
        self.selected_future, self.selected_probes, self.selected_active = [], [], []
        self.probed = set()
        self.student_updates = self.head_updates = self.reliability_updates = self.current_updates = 0
        self.diagnostic_updates = 0
        self.feedback_hash, self.current_hash = hashlib.sha256(), hashlib.sha256()
        self.source_brier = self.repair_brier = 0.0
        self.source_confidence_brier = self.repair_confidence_brier = 0.0
        self.used_positions = set()
        self.use_counts = {}
        self._fresh = False
        self.compute.update(selected_forward_calls=0, selected_forward_images=0,
                            selected_backward_calls=0, selected_backward_images=0,
                            head_updates=0, reliability_updates=0)

    @torch.no_grad()
    def select_queries(self, data, stream_ids):
        outputs = super().select_queries(data, stream_ids)
        if self.study_group == "SP-ATTA":
            return outputs
        self.teacher.eval()
        self.head.eval()
        wf, sf = self.teacher[0](data), self.teacher[0](data.flip(-1))
        qs, qr = self.teacher[1](wf).softmax(1), self.head(wf).softmax(1)
        features = descriptors(qs, qr)
        scores = self.risk(features).sigmoid()
        for row in self.trace.queries[self.step]:
            pos = int(row["sample_id"]) - int(stream_ids[0])
            self.query_records[int(row["anchor_index"])] = {
                "features": features[pos].cpu(), "source": int(qs[pos].argmax()),
                "repair": int(qr[pos].argmax()), "scores": scores[pos].cpu(),
                "source_confidence": float(qs[pos].max()), "repair_confidence": float(qr[pos].max())}
        self.window.append({"images": data.flip(-1).cpu(), "ids": stream_ids.clone(),
                            "weak_features": wf.cpu(), "strong_features": sf.cpu()})
        return outputs

    @torch.no_grad()
    def _prepare_current(self, released):
        self.used_positions = set()
        self.use_counts = {}
        ids = torch.cat([r["ids"] for r in self.window])
        wf = torch.cat([r["weak_features"] for r in self.window]).to(self.config.device)
        sf = torch.cat([r["strong_features"] for r in self.window]).to(self.config.device)
        qs, qr = self.teacher[1](wf).softmax(1), self.head(wf).softmax(1)
        ss, sr = self.teacher[1](sf).softmax(1), self.head(sf).softmax(1)
        scores = self.risk(descriptors(qs, qr)).sigmoid()
        if not bool(torch.isfinite(torch.cat((qs, qr, ss, sr, scores), dim=1)).all()):
            raise FloatingPointError("Non-finite target or reliability score")
        warm = len(self.risk_memory) >= D["min_feedback"] and self.reliability_updates >= D["min_reliability_updates"]
        fresh_ids = {int(self.query_anchors.target[i]) for i, _ in released}
        allowed = torch.tensor([int(sid) not in fresh_ids for sid in ids], device=self.config.device)
        valid_s = (qs.max(1).values >= D["confidence"]) & (qs.argmax(1) == ss.argmax(1)) & allowed & warm
        valid_r = (qr.max(1).values >= D["confidence"]) & (qr.argmax(1) == sr.argmax(1)) & allowed & warm
        repair = (valid_r & (scores[:, 1] >= D["reliability_threshold"])
                  & (scores[:, 1] - scores[:, 0] >= D["correction_margin"])
                  & (qs.argmax(1) != qr.argmax(1)) & (self.study_group == "SP-Correct"))
        source = valid_s & (scores[:, 0] >= D["reliability_threshold"]) & ~repair
        selected = source | repair
        self.current_pool = {"ids": ids, "images": torch.cat([r["images"] for r in self.window]),
            "targets": torch.where(repair[:, None], qr, qs).detach(), "mask": selected.detach(),
            "source": qs.cpu(), "repair": qr.cpu(), "actions": (source.long() + 2 * repair.long()).cpu(),
            "scores": scores.cpu()}
        self.selection_rows.append({"step": self.step, "warm": warm, "feedback_count": len(self.risk_memory),
            "window_samples": len(ids), "source_selected": int(source.sum()), "repair_selected": int(repair.sum()),
            "selected": int(selected.sum()), "used_distinct": 0})

    def _draw_current(self, iteration):
        pool = self.current_pool
        generator = torch.Generator().manual_seed(self.config.random_seed + 3000017 * self.step + 7919 * iteration + 83)
        positions = torch.randperm(len(pool["ids"]), generator=generator)[:self.config.train.train_bs]
        self.current_hash.update(f"{self.step}:{iteration}:".encode())
        self.current_hash.update(pool["ids"][positions].numpy().tobytes())
        dp = positions.to(self.config.device)
        return (pool["images"][positions].to(self.config.device), pool["targets"][dp], pool["mask"][dp], positions)

    def on_labels_released(self, step, domain, released):
        if self.study_group != "SP-ATTA":
            self._prepare_current(released)  # old models only; no fresh-label learning yet
        super().on_labels_released(step, domain, released)  # historical SP gate unchanged
        if self.study_group == "SP-ATTA":
            return
        self._fresh = bool(released)
        # Small fixed probes also precede fresh-feedback assimilation.
        for start in D["probe_starts"]:
            if start not in self.probed and start <= step < start + D["trigger_window"] and released and self.received and self.trace.quotas[step]:
                with isolated_randomness() as state:
                    self._probe(start, step, state)
                self.probed.add(start)
                break
        for index, label in released:
            record = self.query_records.pop(index)
            correctness = torch.tensor([record["source"] == label, record["repair"] == label], dtype=torch.float32)
            self.risk_memory.append((index, record["features"], correctness))
            self.feedback_hash.update(record["features"].numpy().tobytes())
            self.feedback_hash.update(correctness.numpy().tobytes())
            self.source_brier += float((record["scores"][0] - correctness[0]).square())
            self.repair_brier += float((record["scores"][1] - correctness[1]).square())
            self.source_confidence_brier += (record["source_confidence"] - float(correctness[0])) ** 2
            self.repair_confidence_brier += (record["repair_confidence"] - float(correctness[1])) ** 2
            self.feedback_rows.append({"step": step, "anchor_index": index,
                "source_prediction": record["source"], "repair_prediction": record["repair"],
                "returned_label": label, "query_features": json.dumps(record["features"].tolist()),
                "source_correct": int(correctness[0]), "repair_correct": int(correctness[1]),
                "query_source_confidence": record["source_confidence"], "query_repair_confidence": record["repair_confidence"],
                "query_source_score": float(record["scores"][0]), "query_repair_score": float(record["scores"][1])})

    @torch.enable_grad()
    def _student_loss(self, model, tdata, labels, pdata, plabels, current, coefficient):
        history = F.cross_entropy(model(pdata), plabels) if coefficient > 0 and pdata is not None else None
        true = F.cross_entropy(model(tdata), labels)
        use_current = history is not None and bool(current[2].any())
        current_loss = None
        if use_current:
            per_image = F.kl_div(model(current[0]).log_softmax(1), current[1], reduction="none").sum(1)
            current_loss = (per_image * current[2]).mean()
        loss = true
        if history is not None:
            loss = loss + coefficient * ((1 - D["current_share"]) * history + D["current_share"] * current_loss
                                         if use_current else history)
        return loss, true, history, current_loss, use_current

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
        reserve = min(2, quota) if self._fresh else 0
        generator = torch.Generator().manual_seed(self.config.random_seed + 1000003 * self.step + 29)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.atta.SimATTA.lr, momentum=0.9)
        self.model.train()
        for iteration in range(quota):
            tpos = torch.randint(len(ready), (self.config.train.train_bs,), generator=generator)
            ppos = torch.randint(count_p, (self.config.train.train_bs,), generator=generator) if count_p else torch.empty(0, dtype=torch.long)
            indices = torch.tensor(ready)[tpos]
            self.formal_hash.update(f"{self.step}:{iteration}:".encode())
            self.formal_hash.update(indices.numpy().tobytes())
            self.formal_hash.update(ppos.numpy().tobytes())
            current = self._draw_current(iteration)
            images = self.query_anchors.data[indices].to(self.config.device)
            labels = torch.tensor([self.received[int(i)] for i in indices], device=self.config.device)
            kind = "risk" if reserve and iteration == quota - 1 else "head" if reserve == 2 and iteration == quota - 2 else "student"
            true = history = auxiliary = None
            used = False
            coefficient = self.lambda_batch if count_p and kind == "student" else 0.0
            if kind == "head":
                with torch.no_grad():
                    features = self.teacher[0](images)
                opt = torch.optim.SGD(self.head.parameters(), lr=D["head_lr"])
                anchor = sum((p - origin).square().mean() for p, origin in zip(self.head.parameters(), self.origin))
                loss = F.cross_entropy(self.head(features), labels) + D["head_anchor"] * anchor
            elif kind == "risk":
                gen = torch.Generator().manual_seed(self.config.random_seed + 5000011 * self.step + 97)
                positions = torch.randint(len(self.risk_memory), (self.config.train.train_bs,), generator=gen)
                records = [self.risk_memory[int(i)] for i in positions]
                features = torch.stack([r[1] for r in records]).to(self.config.device)
                correctness = torch.stack([r[2] for r in records]).to(self.config.device)
                opt = torch.optim.SGD(self.risk.parameters(), lr=D["reliability_lr"])
                loss = correctness_loss(self.risk, features, correctness, D["reliability_l2"])
            else:
                pdata = self.source_anchors.data[ppos].to(self.config.device) if count_p else None
                plabels = self.source_anchors.target[ppos].to(self.config.device).long() if count_p else None
                opt = optimizer
                loss, true, history, auxiliary, used = self._student_loss(self.model, images, labels, pdata, plabels, current, coefficient)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite {kind} loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if kind != "student":
                opt.zero_grad(set_to_none=True)
            if kind == "head":
                self.head_updates += 1
                self.compute["head_updates"] += 1
            elif kind == "risk":
                self.reliability_updates += 1
                self.compute["reliability_updates"] += 1
            else:
                self.student_updates += 1
                self.aux_updates += int(coefficient > 0)
                self.current_updates += int(used)
                if used:
                    drawn = current[3][current[2].cpu()].tolist()
                    self.used_positions.update(drawn)
                    for pos in drawn:
                        self.use_counts[pos] = self.use_counts.get(pos, 0) + 1
            self.main_updates += 1
            self.formal_by_step[self.step] += 1
            self.true_coefficient_sum += int(kind == "student")
            self.pseudo_coefficient_sum += coefficient
            self.update_rows.append({"step": self.step, "iteration": iteration, "kind": kind,
                "loss": float(loss.detach()), "true_coefficient": float(kind == "student"),
                "pseudo_coefficient": coefficient, "history_coefficient": coefficient * (1 - D["current_share"]) if used else coefficient,
                "current_coefficient": coefficient * D["current_share"] if used else 0.0,
                "true_loss": float(true.detach()) if true is not None else None,
                "history_loss": float(history.detach()) if history is not None else None,
                "current_loss": float(auxiliary.detach()) if auxiliary is not None else None})

    def _probe(self, probe_id, step, state):
        generator = torch.Generator().manual_seed(self.config.random_seed + 7000003 * step + 53)
        ready = sorted(self.received)
        tpos = torch.randint(len(ready), (self.config.train.train_bs,), generator=generator)
        indices = torch.tensor([ready[int(i)] for i in tpos])
        tdata = self.query_anchors.data[indices].to(self.config.device)
        labels = torch.tensor([self.received[int(i)] for i in indices], device=self.config.device)
        count_p = self.source_anchors.num_elem() if self.source_anchors is not None else 0
        ppos = torch.randint(count_p, (self.config.train.train_bs,), generator=generator) if count_p else None
        pdata = self.source_anchors.data[ppos].to(self.config.device) if count_p else None
        plabels = self.source_anchors.target[ppos].to(self.config.device).long() if count_p else None
        # Private probe draw does not advance the formal current hash schedule.
        saved_hash = self.current_hash.copy()
        current = self._draw_current(0)
        self.current_hash = saved_hash
        snapshot = copy.deepcopy(self.model).cpu()
        snapshot[0]._forward_hooks.clear()
        before = fingerprint(snapshot)
        models = {}
        for arm in ("TRUE", "HISTORY", "SELECTED"):
            restore_random(state)
            model = copy.deepcopy(snapshot).to(self.config.device)
            self._counter(model, "selected")
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=self.config.atta.SimATTA.lr, momentum=0.9)
            with torch.enable_grad():
                if arm == "SELECTED":
                    loss = self._student_loss(model, tdata, labels, pdata, plabels, current, D["lambda_aux"])[0]
                else:
                    loss = self._loss(model, tdata, labels, pdata, plabels, 1.0, D["lambda_aux"] if arm == "HISTORY" else 0.0)[0]
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite local selection probe")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            models[arm] = model.cpu()
            self.diagnostic_updates += 1
        if fingerprint(self.model) != before:
            raise AssertionError("Selection probe changed formal model")
        row = {"probe_id": probe_id, "step": step, "selected_sampled": int(current[2].sum()),
            "sp_accept": self.lambda_batch > 0, "complete_window": False}
        self.selected_probes.append(row)
        self.selected_active.append({"row": row, "models": models, "batches": 0})

    @torch.no_grad()
    def on_batch_end(self, step, domain, data, target, outputs, released):
        super().on_batch_end(step, domain, data, target, outputs, released)
        if self.study_group == "SP-ATTA":
            return
        # Evaluation truth is only read here, after predictions/decisions/training.
        labels = dict(zip(range(*self.trace.bounds[step]), target.cpu().tolist()))
        if not hasattr(self, "evaluation_window"):
            self.evaluation_window = deque(maxlen=D["buffer_batches"])
        self.evaluation_window.append(labels)
        truth = {sid: y for batch in self.evaluation_window for sid, y in batch.items()}
        pool = self.current_pool
        self.selection_rows[-1]["used_distinct"] = len(self.used_positions)
        for pos, sid in enumerate(pool["ids"].tolist()):
            action = int(pool["actions"][pos])
            if not action:
                continue
            label = truth[sid]
            source_pred, repair_pred = int(pool["source"][pos].argmax()), int(pool["repair"][pos].argmax())
            chosen = repair_pred if action == 2 else source_pred
            self.target_rows.append({"step": step, "sample_id": sid, "domain": domain,
                "image_domain": next(b["domain"] for bounds, b in zip(self.trace.bounds, self.trace.batches)
                                     if bounds[0] <= sid < bounds[1]),
                "action": "repair" if action == 2 else "source", "used": pos in self.used_positions,
                "training_draws": self.use_counts.get(pos, 0),
                "source_correct": int(source_pred == label), "selected_correct": int(chosen == label),
                "source_score": float(pool["scores"][pos, 0]), "repair_score": float(pool["scores"][pos, 1])})
        for probe in list(self.selected_active):
            row = probe["row"]
            if step <= row["step"]:
                continue
            correct = {}
            with isolated_randomness():
                for arm, model in probe["models"].items():
                    model.to(self.config.device).eval()
                    correct[arm] = model(data).argmax(1).cpu() == target.cpu()
                    model.cpu()
            result = {"probe_id": row["probe_id"], "step": step, "domain": domain, "samples": len(data),
                "true_correct": int(correct["TRUE"].sum()), "history_correct": int(correct["HISTORY"].sum()),
                "selected_correct": int(correct["SELECTED"].sum()), "sp_accept": row["sp_accept"]}
            for name, other in (("true", "TRUE"), ("history", "HISTORY")):
                result[f"fixed_vs_{name}"] = int((correct["SELECTED"] & ~correct[other]).sum())
                result[f"broken_vs_{name}"] = int((~correct["SELECTED"] & correct[other]).sum())
            self.selected_future.append(result)
            probe["batches"] += 1
            if probe["batches"] == D["future_batches"]:
                row["complete_window"] = True
                self.selected_active = [p for p in self.selected_active if p is not probe]

    def run_stream(self, output_dir, progress=None):
        result = super().run_stream(output_dir, progress)
        complete = {r["probe_id"] for r in self.selected_probes if r["complete_window"]}
        for row in self.selected_future:
            row["complete_window"] = row["probe_id"] in complete
        for name, rows, fields in (
            ("feedback_quality.csv", self.feedback_rows, ["step", "anchor_index"]),
            ("selection.csv", self.selection_rows, ["step", "selected"]),
            ("target_quality.csv", self.target_rows, ["step", "sample_id", "used"]),
            ("selected_probes.csv", self.selected_probes, ["probe_id", "step", "complete_window"]),
            ("selected_future.csv", self.selected_future, ["probe_id", "step", "complete_window"]),
        ):
            save_rows(Path(output_dir) / name, rows, fields)
        used = [r for r in self.target_rows if r["used"]]
        n = sum(r["training_draws"] for r in used)
        windows = sum(r["window_samples"] for r in self.selection_rows)
        selected = sum(r["selected"] for r in self.selection_rows)
        future = [r for r in self.selected_future if r["complete_window"]]
        samples = sum(r["samples"] for r in future)
        active_ids = {r["probe_id"] for r in self.selected_probes if r["selected_sampled"] > 0}
        active = [r for r in future if r["probe_id"] in active_ids]
        active_samples = sum(r["samples"] for r in active)
        feedback = len(self.feedback_rows)
        result.update(method=self.study_group, student_updates=self.student_updates, head_updates=self.head_updates,
            reliability_updates=self.reliability_updates, diagnostic_updates=self.diagnostic_updates,
            complete_selected_probes=len(complete),
            current_update_pct=100 * self.current_updates / self.student_updates if self.student_updates else None,
            selected_coverage_pct=100 * selected / windows if windows else None,
            selected_target_accuracy_pct=100 * sum(r["selected_correct"] * r["training_draws"] for r in used) / n if n else None,
            selected_source_accuracy_pct=100 * sum(r["source_correct"] * r["training_draws"] for r in used) / n if n else None,
            target_net_correction_pp=100 * sum((r["selected_correct"] - r["source_correct"]) * r["training_draws"] for r in used) / n if n else None,
            selected_repair_pct=100 * sum((r["action"] == "repair") * r["training_draws"] for r in used) / n if n else None,
            selected_training_draws=n, selected_window_observations=selected,
            prequential_source_brier=self.source_brier / feedback if feedback else None,
            prequential_repair_brier=self.repair_brier / feedback if feedback else None,
            prequential_source_confidence_brier=self.source_confidence_brier / feedback if feedback else None,
            prequential_repair_confidence_brier=self.repair_confidence_brier / feedback if feedback else None,
            local_selected_minus_true_pp=100 * sum(r["selected_correct"] - r["true_correct"] for r in future) / samples if samples else None,
            local_selected_minus_history_pp=100 * sum(r["selected_correct"] - r["history_correct"] for r in future) / samples if samples else None,
            active_local_selected_minus_history_pp=100 * sum(r["selected_correct"] - r["history_correct"] for r in active) / active_samples if active_samples else None,
            feedback_training_sha256=self.feedback_hash.hexdigest(), current_draws_sha256=self.current_hash.hexdigest())
        # Coefficients and usage rates refer to student steps, excluding module slots.
        result["truth_coefficient_mean"] = 1.0 if self.student_updates else None
        result["pseudo_coefficient_mean"] = self.pseudo_coefficient_sum / self.student_updates if self.student_updates else None
        result["aux_update_pct"] = 100 * self.aux_updates / self.student_updates if self.student_updates else None
        if self.student_updates + self.head_updates + self.reliability_updates != sum(self.trace.quotas):
            raise AssertionError("Formal total budget mismatch")
        return result
