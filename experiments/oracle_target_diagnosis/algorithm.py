"""Five same-snapshot diagnostic arms; evaluation truth is restricted to oracle copies."""
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
from .spec import DEFAULTS as D, METHOD, ARMS, BASE_ARMS, PAIRS


class OracleTargetDiagnosis(SPATTA):
    def __init__(self, config, trace):
        super().__init__(config, "SP-ATTA", trace)
        self.repair_head = copy.deepcopy(self.teacher[1]).to(config.device)
        self.repair_head.requires_grad_(True)
        self.head_origin = [p.detach().clone() for p in self.repair_head.parameters()]
        self.window = deque(maxlen=D["buffer_batches"])
        # Only scoring and explicitly nondeployable oracle copies can read these labels.
        # Ordinary candidates, gates, formal training and head repair never read them.
        self.evaluation_truth = deque(maxlen=D["buffer_batches"])
        self.oracle_packets, self.oracle_rows = [], []
        self.oracle_updates = 0
        self.diagnostic_active, self.pending_target_scores = [], []
        self.probe_rows, self.future_rows, self.target_rows, self.draw_rows = [], [], [], []
        self.head_rows, self.skipped_rows = [], []
        self.triggered = set()
        self.diagnostic_updates = self.head_updates = 0
        self.compute.update(diagnostic_forward_calls=0, diagnostic_forward_images=0,
                            diagnostic_backward_calls=0, diagnostic_backward_images=0,
                            repair_head_updates=0)

    @torch.no_grad()
    def select_queries(self, data, stream_ids):
        outputs = super().select_queries(data, stream_ids)
        # Add only deterministic, no-gradient observations to an independent diagnostic buffer.
        self.teacher.eval()
        weak_features = self.teacher[0](data).cpu()
        flipped = data.flip(-1)
        strong_features = self.teacher[0](flipped).cpu()
        self.window.append({"strong": flipped.cpu(), "ids": stream_ids.clone(),
                            "weak_features": weak_features, "strong_features": strong_features})
        return outputs

    @torch.no_grad()
    def _current_targets(self, released):
        self.repair_head.eval()
        wf = torch.cat([r["weak_features"] for r in self.window]).to(self.config.device)
        sf = torch.cat([r["strong_features"] for r in self.window]).to(self.config.device)
        ids = torch.cat([r["ids"] for r in self.window])
        images = torch.cat([r["strong"] for r in self.window])
        fresh_ids = {int(self.query_anchors.target[index]) for index, _ in released}
        allowed = torch.tensor([int(sid) not in fresh_ids for sid in ids], device=self.config.device)
        targets, masks = {}, {}
        for arm, head in (("SOURCE", self.teacher[1]), ("REPAIR", self.repair_head)):
            weak, strong = head(wf).softmax(1), head(sf).softmax(1)
            if not bool(torch.isfinite(weak).all() and torch.isfinite(strong).all()):
                raise FloatingPointError("Non-finite diagnostic targets")
            targets[arm] = weak.detach()
            masks[arm] = (weak.max(1).values >= D["confidence"]) & (weak.argmax(1) == strong.argmax(1)) & allowed
        return ids, images, targets, masks, allowed

    def on_labels_released(self, step, domain, released):
        # Original SP gate remains unchanged and sees no diagnostic outcomes.
        super().on_labels_released(step, domain, released)
        for start in D["probe_starts"]:
            if start in self.triggered or not start <= step < start + D["trigger_window"]:
                continue
            if not released or not self.received or not self.trace.quotas[step]:
                continue
            with isolated_randomness() as state:
                self._probe(start, step, domain, released, state)
            self.triggered.add(start)
            break

    def _probe(self, probe_id, step, domain, released, state):
        ids, images, targets, masks, allowed = self._current_targets(released)
        previous = sorted(self.received)  # fresh holdout has not entered this memory
        generator = torch.Generator().manual_seed(self.config.random_seed + 3000017 * step + 83)
        tpos = torch.randint(len(previous), (self.config.train.train_bs,), generator=generator)
        truth_indices = torch.tensor([previous[int(i)] for i in tpos])
        positions = torch.randperm(len(ids), generator=generator)[:self.config.train.train_bs]
        device_positions = positions.to(self.config.device)
        truth_images = self.query_anchors.data[truth_indices].to(self.config.device)
        truth_labels = torch.tensor([self.received[int(i)] for i in truth_indices], device=self.config.device)
        current_images = images[positions].to(self.config.device)
        fresh_images = self.query_anchors.data[[i for i, _ in released]]
        fresh_labels = torch.tensor([label for _, label in released], dtype=torch.long)
        draw_record = {"probe_id": probe_id, "step": step, "truth_anchor_indices": truth_indices.tolist(),
                       "current_sample_ids": ids[positions].tolist(),
                       "fresh_anchor_indices": [i for i, _ in released]}
        self.draw_rows.append({**draw_record, "truth_anchor_indices": json.dumps(draw_record["truth_anchor_indices"]),
            "current_sample_ids": json.dumps(draw_record["current_sample_ids"]),
            "fresh_anchor_indices": json.dumps(draw_record["fresh_anchor_indices"])})
        draw_hash = hashlib.sha256(json.dumps(draw_record, sort_keys=True).encode()).hexdigest()
        snapshot = copy.deepcopy(self.model).cpu()
        snapshot[0]._forward_hooks.clear()
        snapshot_hash = fingerprint(snapshot)
        models, losses = {}, {}
        for arm in BASE_ARMS:
            restore_random(state)
            branch = copy.deepcopy(snapshot).to(self.config.device)
            self._counter(branch, "diagnostic")
            branch.train()
            optimizer = torch.optim.SGD(branch.parameters(), lr=self.config.atta.SimATTA.lr, momentum=0.9)
            with torch.enable_grad():
                loss = F.cross_entropy(branch(truth_images), truth_labels)
                if arm != "TRUE":
                    per_image = F.kl_div(branch(current_images).log_softmax(1),
                        targets[arm][device_positions], reduction="none").sum(1)
                    loss = loss + D["lambda_aux"] * (per_image * masks[arm][device_positions]).mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite current-target candidate loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            losses[arm] = self._validation_loss(branch, fresh_images, fresh_labels)
            optimizer.zero_grad(set_to_none=True)
            models[arm] = branch.cpu()
            self.diagnostic_updates += 1
            del optimizer, loss
        if fingerprint(self.model) != snapshot_hash:
            raise AssertionError("Diagnosis changed SP formal model")
        gain_source = losses["TRUE"] - losses["SOURCE"]
        gain_repair = losses["TRUE"] - losses["REPAIR"]
        if not bool(torch.isfinite(torch.tensor([gain_source, gain_repair])).all()):
            raise FloatingPointError("Non-finite diagnostic gate evidence")
        row = {"probe_id": probe_id, "step": step, "domain": domain, "fresh_count": len(released),
            "sp_would_accept": self.gate_rows[-1]["would_accept"], "snapshot_sha256": snapshot_hash,
            "draws_sha256": draw_hash, "truth_count": len(previous), "buffer_samples": len(ids),
            "sampled_current": len(positions), "source_eligible": int(masks["SOURCE"].sum()),
            "repair_eligible": int(masks["REPAIR"].sum()),
            "source_sampled_eligible": int(masks["SOURCE"][device_positions].sum()),
            "repair_sampled_eligible": int(masks["REPAIR"][device_positions].sum()),
            "true_fresh_ce": losses["TRUE"], "source_fresh_ce": losses["SOURCE"],
            "repair_fresh_ce": losses["REPAIR"], "source_gate_gain": gain_source,
            "repair_gate_gain": gain_repair, "source_accept": gain_source > D["gain_margin"],
            "repair_accept": gain_repair > D["gain_margin"], "complete_window": False,
            "oracle_source_changed_selected": None, "oracle_repair_changed_selected": None}
        self.probe_rows.append(row)
        self.diagnostic_active.append({"row": row, "models": models, "batches": 0})
        # Oracle labels are not available at this hook. Preserve the same pre-update
        # snapshot and draws, then fork oracle copies in the evaluation-only hook.
        self.oracle_packets.append({"probe_id": probe_id, "models": models, "row": row,
            "snapshot": snapshot, "state": state, "snapshot_sha256": snapshot_hash,
            "truth_images": truth_images.cpu(), "truth_labels": truth_labels.cpu(),
            "images": images[positions], "ids": ids[positions],
            "targets": {a: targets[a][device_positions].cpu() for a in ("SOURCE", "REPAIR")},
            "masks": {a: masks[a][device_positions].cpu() for a in ("SOURCE", "REPAIR")}})
        # No truth here. These predictions will be scored in the evaluation-only hook.
        self.pending_target_scores.append({"row": row, "ids": ids,
            "targets": {a: t.cpu() for a, t in targets.items()},
            "masks": {a: m.cpu() for a, m in masks.items()}, "allowed": allowed.cpu(),
            "sampled_ids": set(ids[positions].tolist())})

    @torch.no_grad()
    def _score_targets(self, step, target):
        lo, hi = self.trace.bounds[step]
        self.evaluation_truth.append(dict(zip(range(lo, hi), target.cpu().tolist())))
        truth = {sid: label for batch in self.evaluation_truth for sid, label in batch.items()}
        for pending in self.pending_target_scores:
            row = pending["row"]
            source, repair = pending["targets"]["SOURCE"], pending["targets"]["REPAIR"]
            for pos, sid in enumerate(pending["ids"].tolist()):
                label = truth[sid]
                sp, rp = int(source[pos].argmax()), int(repair[pos].argmax())
                self.target_rows.append({"probe_id": row["probe_id"], "step": step, "sample_id": sid,
                    "domain": self.trace.batches[step]["domain"],
                    "image_domain": next(b["domain"] for lohi, b in zip(self.trace.bounds, self.trace.batches)
                                         if lohi[0] <= sid < lohi[1]),
                    "evaluation_label": label, "source_prediction": sp, "repair_prediction": rp,
                    "source_confidence": float(source[pos].max()), "repair_confidence": float(repair[pos].max()),
                    "source_correct": int(sp == label), "repair_correct": int(rp == label),
                    "source_selected": bool(pending["masks"]["SOURCE"][pos]),
                    "repair_selected": bool(pending["masks"]["REPAIR"][pos]),
                    "allowed": bool(pending["allowed"][pos]), "sampled": sid in pending["sampled_ids"],
                    "source_ce": -float(source[pos, label].clamp_min(1e-12).log()),
                    "repair_ce": -float(repair[pos, label].clamp_min(1e-12).log())})
        self.pending_target_scores.clear()
        # Newly visible current labels only enter these independent oracle copies.
        with isolated_randomness():
            self._finish_oracles(truth)

    def _finish_oracles(self, evaluation_truth):
        if not self.oracle_packets:
            return
        main_before, head_before = fingerprint(self.model), fingerprint(self.repair_head)
        for packet in self.oracle_packets:
            if fingerprint(packet["snapshot"]) != packet["snapshot_sha256"]:
                raise AssertionError("Stored pre-update snapshot changed")
            labels = torch.tensor([evaluation_truth[int(sid)] for sid in packet["ids"]], dtype=torch.long)
            for source in ("SOURCE", "REPAIR"):
                q = packet["targets"][source]
                mask = packet["masks"][source]
                row_index = torch.arange(len(q))
                predicted = q.argmax(1)
                corrected = q.clone()
                corrected[row_index, predicted] = q[row_index, labels]
                corrected[row_index, labels] = q[row_index, predicted]
                # A permutation preserves every probability value, hence confidence
                # and entropy. It changes no already-correct target or input mask.
                if not torch.equal(q.sort(1).values, corrected.sort(1).values):
                    raise AssertionError("Oracle changed target probability multiset")
                if mask.any() and not torch.equal(corrected.argmax(1)[mask], labels[mask]):
                    raise AssertionError("Oracle failed to correct a selected target")
                packet["row"][f"oracle_{source.lower()}_changed_selected"] = int(((predicted != labels) & mask).sum())
                arm = f"ORACLE_{source}"
                for pos, sid in enumerate(packet["ids"].tolist()):
                    self.oracle_rows.append({"probe_id": packet["probe_id"], "sample_id": sid,
                        "arm": arm, "selected": bool(mask[pos]), "evaluation_label": int(labels[pos]),
                        "original_prediction": int(predicted[pos]),
                        "oracle_prediction": int(corrected[pos].argmax()),
                        "original_confidence": float(q[pos].max()),
                        "oracle_confidence": float(corrected[pos].max()),
                        "changed": bool(predicted[pos] != labels[pos]),
                        "original_probabilities": json.dumps(q[pos].tolist()),
                        "oracle_probabilities": json.dumps(corrected[pos].tolist())})
                restore_random(packet["state"])
                branch = copy.deepcopy(packet["snapshot"]).to(self.config.device)
                self._counter(branch, "diagnostic")
                branch.train()
                optimizer = torch.optim.SGD(branch.parameters(), lr=self.config.atta.SimATTA.lr, momentum=0.9)
                with torch.enable_grad():
                    loss = F.cross_entropy(branch(packet["truth_images"].to(self.config.device)),
                                           packet["truth_labels"].to(self.config.device))
                    per_image = F.kl_div(branch(packet["images"].to(self.config.device)).log_softmax(1),
                                        corrected.to(self.config.device), reduction="none").sum(1)
                    loss = loss + D["lambda_aux"] * (per_image * mask.to(self.config.device)).mean()
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Non-finite oracle-copy loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                packet["models"][arm] = branch.cpu()
                self.diagnostic_updates += 1
                self.oracle_updates += 1
                del optimizer, loss
        self.oracle_packets.clear()
        if fingerprint(self.model) != main_before or fingerprint(self.repair_head) != head_before:
            raise AssertionError("Oracle diagnosis changed formal model or repair head")

    @torch.no_grad()
    def _score_future(self, step, domain, data, target):
        for probe in list(self.diagnostic_active):
            row = probe["row"]
            if step <= row["step"]:
                continue
            predictions = {}
            with isolated_randomness():
                for arm in ARMS:
                    branch = probe["models"][arm].to(self.config.device).eval()
                    predictions[arm] = branch(data).argmax(1).cpu()
                    branch.cpu()
            truth = target.cpu()
            correct = {a: p == truth for a, p in predictions.items()}
            record = {"probe_id": row["probe_id"], "probe_domain": row["domain"], "step": step,
                "domain": domain, "samples": len(data), "source_accept": row["source_accept"],
                "repair_accept": row["repair_accept"], "sp_would_accept": row["sp_would_accept"],
                **{f"{arm.lower()}_correct": int(correct[arm].sum()) for arm in ARMS}}
            for name, left, right in PAIRS:
                record[f"{name}_fixed"] = int((correct[left] & ~correct[right]).sum())
                record[f"{name}_broken"] = int((~correct[left] & correct[right]).sum())
            self.future_rows.append(record)
            probe["batches"] += 1
            if probe["batches"] == D["future_batches"]:
                row["complete_window"] = True
                self.diagnostic_active = [p for p in self.diagnostic_active if p is not probe]

    @torch.enable_grad()
    def _repair_after_batch(self, step, released):
        if not released or not self.trace.quotas[step]:
            return
        # Mirror the old feedback-study head draw: last formal draw, but head is shadow-only.
        ready = sorted(self.received)
        count_p = self.source_anchors.num_elem() if self.source_anchors is not None else 0
        generator = torch.Generator().manual_seed(self.config.random_seed + 1000003 * step + 29)
        for _ in range(self.trace.quotas[step]):
            tpos = torch.randint(len(ready), (self.config.train.train_bs,), generator=generator)
            if count_p:
                torch.randint(count_p, (self.config.train.train_bs,), generator=generator)
        indices = torch.tensor([ready[int(i)] for i in tpos])
        images = self.query_anchors.data[indices].to(self.config.device)
        labels = torch.tensor([self.received[int(i)] for i in indices], device=self.config.device)
        self.teacher.eval()
        with torch.no_grad():
            features = self.teacher[0](images)
        optimizer = torch.optim.SGD(self.repair_head.parameters(), lr=D["repair_lr"])
        anchor = sum((p - origin).square().mean() for p, origin in zip(self.repair_head.parameters(), self.head_origin))
        loss = F.cross_entropy(self.repair_head(features), labels) + D["repair_anchor"] * anchor
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite diagnostic repair-head loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.head_updates += 1
        self.compute["repair_head_updates"] += 1
        self.head_rows.append({"step": step, "head_loss": float(loss.detach()),
                               "truth_anchor_indices": json.dumps(indices.tolist())})

    def on_batch_end(self, step, domain, data, target, outputs, released):
        super().on_batch_end(step, domain, data, target, outputs, released)
        self._score_targets(step, target)
        self._score_future(step, domain, data, target)
        with isolated_randomness():
            self._repair_after_batch(step, released)
        # Parent appended budget before diagnostic forwards/head training; refresh totals.
        self.budget_rows[-1].update(self.compute)

    def run_stream(self, output_dir, progress=None):
        result = super().run_stream(output_dir, progress)
        complete = {row["probe_id"] for row in self.probe_rows if row["complete_window"]}
        for row in self.future_rows:
            row["complete_window"] = row["probe_id"] in complete
        for start in D["probe_starts"]:
            if start not in self.triggered:
                self.skipped_rows.append({"probe_id": start, "reason": "no fresh feedback with previous truth and formal quota in trigger window"})
        for name, rows, fields in (
            ("target_probes.csv", self.probe_rows, ["probe_id", "step", "complete_window"]),
            ("target_future.csv", self.future_rows, ["probe_id", "step", "complete_window"]),
            ("target_quality.csv", self.target_rows, ["probe_id", "sample_id", "source_correct", "repair_correct"]),
            ("target_draws.csv", self.draw_rows, ["probe_id", "step", "truth_anchor_indices", "current_sample_ids"]),
            ("repair_updates.csv", self.head_rows, ["step", "head_loss"]),
            ("target_skipped.csv", self.skipped_rows, ["probe_id", "reason"]),
            ("oracle_targets.csv", self.oracle_rows, ["probe_id", "sample_id", "arm", "selected"]),
        ):
            save_rows(Path(output_dir) / name, rows, fields)
        result.update(method=METHOD, complete_target_probes=len(complete),
                      diagnostic_candidate_updates=self.diagnostic_updates, diagnostic_head_updates=self.head_updates,
                      oracle_candidate_updates=self.oracle_updates)
        return result
