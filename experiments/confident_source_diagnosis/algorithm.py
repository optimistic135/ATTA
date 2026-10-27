"""Test elite source targets without changing the factual SP trajectory."""
import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.sp_atta.algorithm import SPATTA
from experiments.simatta_rootcause.algorithm import isolated_randomness, restore_random, fingerprint
from experiments.fcra_relay.stream import save_rows
from .spec import ARMS, PAIRS, DEFAULTS as D, METHOD


class ConfidentSourceDiagnosis(SPATTA):
    def __init__(self, config, trace):
        super().__init__(config, "SP-ATTA", trace)
        self.elite_active, self.probe_rows, self.future_rows, self.draw_rows = [], [], [], []
        self.target_rows, self.pending_targets, self.selection_rows, self.skipped_rows = [], [], [], []
        self.triggered = set()
        self.diagnostic_updates = 0
        self.first_return = None
        self.compute.update(elite_forward_calls=0, elite_forward_images=0,
                            elite_backward_calls=0, elite_backward_images=0)

    @torch.no_grad()
    def select_queries(self, data, stream_ids):
        outputs = super().select_queries(data, stream_ids)
        self.teacher.eval()
        q = self.teacher(data).softmax(1)
        flipped = self.teacher(data.flip(-1)).softmax(1)
        if not bool(torch.isfinite(torch.cat((q, flipped), dim=1)).all()):
            raise FloatingPointError("Non-finite frozen-source target")
        confidence = q.max(1).values.cpu()
        eligible = ((confidence >= D["confidence"]) & (q.argmax(1).cpu() == flipped.argmax(1).cpu())).nonzero().flatten()
        generator = torch.Generator().manual_seed(self.config.random_seed + 9000011 * self.step + 37)
        shuffled = eligible[torch.randperm(len(eligible), generator=generator)].tolist()
        # Python stable sort preserves the private random order only for exact ties.
        ranked = sorted(shuffled, key=lambda i: float(confidence[i]), reverse=True)
        k = max(1, math.ceil(D["rank_fraction"] * len(ranked))) if ranked else 0
        positions = torch.tensor(ranked[:k], dtype=torch.long)
        dp = positions.to(self.config.device)
        self.elite = {"positions": positions, "ids": stream_ids[positions],
            "images": data[dp].flip(-1).cpu(), "targets": q[dp].cpu()}
        self.pending_targets = []
        main_prob = outputs.softmax(1).cpu()
        for pos in positions.tolist():
            predicted = int(q[pos].argmax())
            self.pending_targets.append({"step": self.step, "domain": self.trace.batches[self.step]["domain"],
                "sample_id": int(stream_ids[pos]), "source_prediction": predicted,
                "model_prediction": int(outputs[pos].argmax()), "source_confidence": float(confidence[pos]),
                "need_source_target": 1 - float(main_prob[pos, predicted]), "probe_id": None,
                "evaluation_label": None, "source_correct": None, "model_correct": None,
                "correction_opportunity": None, "harm_risk": None, "before_first_feedback": None})
        self.selection_rows.append({"step": self.step, "domain": self.trace.batches[self.step]["domain"],
            "samples": len(data), "base_candidates": len(eligible), "selected": k,
            "actual_candidate_coverage_pct": 100 * k / len(eligible) if len(eligible) else None})
        return outputs

    def on_labels_released(self, step, domain, released):
        super().on_labels_released(step, domain, released)
        if released and self.first_return is None:
            self.first_return = step
        for start in D["probe_starts"]:
            if start in self.triggered or not start <= step < start + D["trigger_window"]:
                continue
            if not released or not self.received or not self.trace.quotas[step] or self.source_anchors is None:
                continue
            fresh = {int(self.query_anchors.target[i]) for i, _ in released}
            history = torch.tensor([i for i, sid in enumerate(self.pseudo_ids) if sid not in fresh], dtype=torch.long)
            if not len(history):
                continue
            with isolated_randomness() as state:
                self._probe(start, step, domain, history, state)
            self.triggered.add(start)
            for row in self.pending_targets:
                row["probe_id"] = start
            break

    def _probe(self, probe_id, step, domain, history, state):
        generator = torch.Generator().manual_seed(self.config.random_seed + 3000017 * step + 83)
        previous = sorted(self.received)
        tpos = torch.randint(len(previous), (self.config.train.train_bs,), generator=generator)
        truth_indices = torch.tensor([previous[int(i)] for i in tpos])
        ppos = history[torch.randint(len(history), (self.config.train.train_bs,), generator=generator)]
        tdata = self.query_anchors.data[truth_indices].to(self.config.device)
        labels = torch.tensor([self.received[int(i)] for i in truth_indices], device=self.config.device)
        pdata = self.source_anchors.data[ppos].to(self.config.device)
        plabels = self.source_anchors.target[ppos].to(self.config.device).long()
        selected = len(self.elite["ids"])
        current_images = self.elite["images"].to(self.config.device)
        current_targets = self.elite["targets"].to(self.config.device)
        snapshot = copy.deepcopy(self.model).cpu()
        snapshot[0]._forward_hooks.clear()
        snapshot_hash = fingerprint(snapshot)
        models, losses = {}, {}
        for arm in ARMS:
            restore_random(state)
            branch = copy.deepcopy(snapshot).to(self.config.device)
            self._counter(branch, "elite")
            branch.train()
            optimizer = torch.optim.SGD(branch.parameters(), lr=self.config.atta.SimATTA.lr, momentum=0.9)
            with torch.enable_grad():
                historical = F.cross_entropy(branch(pdata), plabels) if arm != "TRUE" else None
                true = F.cross_entropy(branch(tdata), labels)
                loss = true
                if arm == "HISTORY" or arm == "CURRENT" and not selected:
                    loss = loss + D["lambda_aux"] * historical
                elif arm in ("REDUCED", "CURRENT"):
                    loss = loss + D["lambda_aux"] * (1 - D["current_share"]) * historical
                if arm == "CURRENT" and selected:
                    current = F.kl_div(branch(current_images).log_softmax(1), current_targets, reduction="batchmean")
                    loss = loss + D["lambda_aux"] * D["current_share"] * current
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite elite candidate loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses[arm] = float(loss.detach())
            models[arm] = branch.cpu()
            self.diagnostic_updates += 1
            del optimizer, loss
        if fingerprint(self.model) != snapshot_hash:
            raise AssertionError("Elite probes changed formal model")
        draw = {"probe_id": probe_id, "step": step, "truth_anchor_indices": truth_indices.tolist(),
                "history_positions": ppos.tolist(), "current_sample_ids": self.elite["ids"].tolist()}
        draw_hash = hashlib.sha256(json.dumps(draw, sort_keys=True).encode()).hexdigest()
        self.draw_rows.append({key: json.dumps(value) if isinstance(value, list) else value for key, value in draw.items()})
        row = {"probe_id": probe_id, "step": step, "domain": domain, "selected": selected,
            "base_candidates": self.selection_rows[-1]["base_candidates"], "sp_accept": self.lambda_batch > 0,
            "snapshot_sha256": snapshot_hash, "draws_sha256": draw_hash, "complete_window": False,
            **{f"{a.lower()}_training_loss": losses[a] for a in ARMS}}
        self.probe_rows.append(row)
        self.elite_active.append({"row": row, "models": models, "batches": 0})

    @torch.no_grad()
    def on_batch_end(self, step, domain, data, target, outputs, released):
        super().on_batch_end(step, domain, data, target, outputs, released)
        lo, _ = self.trace.bounds[step]
        for row in self.pending_targets:
            label = int(target[row["sample_id"] - lo])
            row["evaluation_label"] = label
            row["source_correct"] = int(row["source_prediction"] == label)
            row["model_correct"] = int(row["model_prediction"] == label)
            row["correction_opportunity"] = int(row["source_correct"] and not row["model_correct"])
            row["harm_risk"] = int(not row["source_correct"] and row["model_correct"])
            # Current prediction precedes this batch's first release/update.
            row["before_first_feedback"] = self.first_return is None or step <= self.first_return
            self.target_rows.append(row)
        self.pending_targets.clear()
        for probe in list(self.elite_active):
            row = probe["row"]
            if step <= row["step"]:
                continue
            correct = {}
            with isolated_randomness():
                for arm in ARMS:
                    branch = probe["models"][arm].to(self.config.device).eval()
                    correct[arm] = branch(data).argmax(1).cpu() == target.cpu()
                    branch.cpu()
            scored = {"probe_id": row["probe_id"], "trigger_domain": row["domain"], "step": step,
                "domain": domain, "samples": len(data), "active": row["selected"] > 0, "sp_accept": row["sp_accept"],
                **{f"{arm.lower()}_correct": int(correct[arm].sum()) for arm in ARMS}}
            for name, left, right in PAIRS:
                scored[f"{name}_fixed"] = int((correct[left] & ~correct[right]).sum())
                scored[f"{name}_broken"] = int((~correct[left] & correct[right]).sum())
            self.future_rows.append(scored)
            probe["batches"] += 1
            if probe["batches"] == D["future_batches"]:
                row["complete_window"] = True
                self.elite_active = [p for p in self.elite_active if p is not probe]
        self.budget_rows[-1].update(self.compute)

    def run_stream(self, output_dir, progress=None):
        result = super().run_stream(output_dir, progress)
        complete = {r["probe_id"] for r in self.probe_rows if r["complete_window"]}
        for row in self.future_rows:
            row["complete_window"] = row["probe_id"] in complete
        for start in D["probe_starts"]:
            if start not in self.triggered:
                self.skipped_rows.append({"probe_id": start, "reason": "no fresh feedback, old truth, quota or heldout history in trigger interval"})
        for name, rows, fields in (
            ("elite_probes.csv", self.probe_rows, ["probe_id", "step", "complete_window"]),
            ("elite_future.csv", self.future_rows, ["probe_id", "step", "complete_window"]),
            ("elite_targets.csv", self.target_rows, ["step", "sample_id", "source_correct"]),
            ("elite_selection.csv", self.selection_rows, ["step", "selected"]),
            ("elite_draws.csv", self.draw_rows, ["probe_id", "step", "current_sample_ids"]),
            ("elite_skipped.csv", self.skipped_rows, ["probe_id", "reason"]),
        ):
            save_rows(Path(output_dir) / name, rows, fields)
        result.update(method=METHOD, complete_elite_probes=len(complete), diagnostic_candidate_updates=self.diagnostic_updates,
                      first_feedback_step=self.first_return)
        return result
