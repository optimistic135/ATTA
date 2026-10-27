"""Observe every rejected/accepted target, without changing the SP main trajectory."""
import json
from collections import deque
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.selective_feedback.algorithm import SelectiveFeedback
from experiments.selective_feedback.reliability import descriptors, correctness_loss
from experiments.sp_atta.algorithm import SPATTA
from experiments.simatta_rootcause.algorithm import isolated_randomness, fingerprint
from experiments.fcra_relay.stream import save_rows
from .spec import DEFAULTS as D, METHOD


class SelectionDiagnosis(SelectiveFeedback):
    def __init__(self, config, trace):
        # Reuse precisely the same zero-initialized risk model, source-copy head,
        # query feature extraction and source-feature window as the failed study.
        super().__init__(config, "SP-Select", trace)
        self.observation_rows, self.funnel_rows, self.query_rows, self.policy_rows = [], [], [], []
        self.shadow_rows, self.pending_observations = [], []
        self.evaluation_window = deque(maxlen=D["buffer_batches"])
        self.first_learning_step = None
        self.first_warm_step = None

    def _warm(self):
        return len(self.risk_memory) >= D["min_feedback"] and self.reliability_updates >= D["min_reliability_updates"]

    @torch.no_grad()
    def select_queries(self, data, stream_ids):
        outputs = super().select_queries(data, stream_ids)
        self.window[-1]["domains"] = [self.trace.batches[self.step]["domain"]] * len(stream_ids)
        n = len(self.risk_memory)
        # Online legal base-rate comparator, Beta(1,1) smoothing; no test prevalence.
        rates = [(1 + sum(float(r[2][j]) for r in self.risk_memory)) / (n + 2) for j in (0, 1)]
        for row in self.trace.queries[self.step]:
            record = self.query_records[int(row["anchor_index"])]
            record.update(query_step=self.step, query_domain=self.trace.batches[self.step]["domain"],
                risk_updates_at_query=self.reliability_updates, warm_at_query=self._warm(),
                query_phase="after_learning" if self.reliability_updates else "before_learning",
                source_base_rate=rates[0], repair_base_rate=rates[1])
        return outputs

    @torch.no_grad()
    def _observe_current(self, step, released):
        wf = torch.cat([r["weak_features"] for r in self.window]).to(self.config.device)
        sf = torch.cat([r["strong_features"] for r in self.window]).to(self.config.device)
        qs, qr = self.teacher[1](wf).softmax(1), self.head(wf).softmax(1)
        ss, sr = self.teacher[1](sf).softmax(1), self.head(sf).softmax(1)
        scores = self.risk(descriptors(qs, qr)).sigmoid()
        if not bool(torch.isfinite(torch.cat((qs, qr, ss, sr, scores), dim=1)).all()):
            raise FloatingPointError("Non-finite screening observations")
        # One transfer per tensor; avoid per-observation GPU synchronization.
        qs, qr, ss, sr, scores = [x.cpu() for x in (qs, qr, ss, sr, scores)]
        ids = torch.cat([r["ids"] for r in self.window])
        domains = [name for r in self.window for name in r["domains"]]
        warm = self._warm()
        if warm and self.first_warm_step is None:
            self.first_warm_step = step
        fresh_ids = {int(self.query_anchors.target[i]) for i, _ in released}
        allowed = torch.tensor([int(sid) not in fresh_ids for sid in ids])
        conf_s, conf_r = qs.max(1).values, qr.max(1).values
        consistent_s, consistent_r = qs.argmax(1) == ss.argmax(1), qr.argmax(1) == sr.argmax(1)
        margin = scores[:, 1] - scores[:, 0]
        disagreement = qs.argmax(1) != qr.argmax(1)
        branches = {}
        for branch, confidence, consistency, score in (
            ("source", conf_s, consistent_s, scores[:, 0]),
            ("repair", conf_r, consistent_r, scores[:, 1])):
            cumulative = torch.ones(len(ids), dtype=torch.bool)
            stages = [("window", torch.ones_like(cumulative)), ("allowed", allowed),
                      ("warmup", torch.full_like(cumulative, warm)),
                      ("confidence", confidence >= D["confidence"]),
                      ("consistency", consistency), ("risk", score >= D["reliability_threshold"])]
            if branch == "repair":
                stages += [("margin", margin >= D["correction_margin"]), ("disagreement", disagreement)]
            for order, (stage, condition) in enumerate(stages):
                before = cumulative.clone()
                cumulative &= condition
                for domain in ("ALL",) + tuple(dict.fromkeys(domains)):
                    belongs = torch.tensor([domain == "ALL" or name == domain for name in domains])
                    count_before, passed = int((before & belongs).sum()), int((cumulative & belongs).sum())
                    self.funnel_rows.append({"step": step, "domain": domain, "warm": warm,
                        "branch": branch, "order": order, "stage": stage,
                        "before": count_before, "passed": passed, "dropped": count_before - passed})
            branches[branch] = cumulative
        repair = branches["repair"]
        source_only = branches["source"]
        source_fallback = source_only & ~repair
        self.policy_rows.append({"step": step, "domain": self.trace.batches[step]["domain"],
            "warm": warm, "feedback_count": len(self.risk_memory), "risk_updates": self.reliability_updates,
            "sp_accept": False,
            "samples": len(ids), "select_count": int(source_only.sum()),
            "correct_source_count": int(source_fallback.sum()), "correct_repair_count": int(repair.sum()),
            "correct_total_count": int((source_only | repair).sum())})
        for pos, sid in enumerate(ids.tolist()):
            self.pending_observations.append({"step": step, "sample_id": sid, "image_domain": domains[pos],
                "current_domain": self.trace.batches[step]["domain"], "warm": warm,
                "feedback_count": len(self.risk_memory), "risk_updates": self.reliability_updates,
                "allowed": bool(allowed[pos]), "source_confidence": float(conf_s[pos]),
                "repair_confidence": float(conf_r[pos]), "source_consistent": bool(consistent_s[pos]),
                "repair_consistent": bool(consistent_r[pos]), "source_score": float(scores[pos, 0]),
                "repair_score": float(scores[pos, 1]), "repair_margin": float(margin[pos]),
                "disagreement": bool(disagreement[pos]), "source_prediction": int(qs[pos].argmax()),
                "repair_prediction": int(qr[pos].argmax()), "select_selected": bool(source_only[pos]),
                "correct_source": bool(source_fallback[pos]), "correct_repair": bool(repair[pos]),
                "sp_accept": False, "evaluation_label": None, "source_correct": None, "repair_correct": None})

    def on_labels_released(self, step, domain, released):
        self._observe_current(step, released)  # old modules only
        SPATTA.on_labels_released(self, step, domain, released)  # unchanged historical gate
        for row in self.pending_observations:
            row["sp_accept"] = self.lambda_batch > 0
        self.policy_rows[-1]["sp_accept"] = self.lambda_batch > 0
        domains = tuple(dict.fromkeys(r["image_domain"] for r in self.pending_observations))
        for branch, key, order in (("source", "select_selected", 6), ("repair", "correct_repair", 8)):
            for name in ("ALL",) + domains:
                before = sum(r[key] for r in self.pending_observations if name == "ALL" or r["image_domain"] == name)
                passed = before if self.lambda_batch > 0 else 0
                self.funnel_rows.append({"step": step, "domain": name, "warm": self._warm(),
                    "branch": branch, "order": order, "stage": "sp_gate", "before": before,
                    "passed": passed, "dropped": before - passed})
        for index, label in released:
            record = self.query_records.pop(index)
            correctness = torch.tensor([record["source"] == label, record["repair"] == label], dtype=torch.float32)
            self.risk_memory.append((index, record["features"], correctness))
            for branch, j in (("source", 0), ("repair", 1)):
                self.query_rows.append({"anchor_index": index, "query_step": record["query_step"],
                    "return_step": step, "query_domain": record["query_domain"], "return_domain": domain,
                    "branch": branch, "phase": record["query_phase"], "warm_at_query": record["warm_at_query"],
                    "risk_updates_at_query": record["risk_updates_at_query"],
                    "correct": int(correctness[j]), "score": float(record["scores"][j]),
                    "confidence": record[f"{branch}_confidence"], "base_rate": record[f"{branch}_base_rate"]})

    def cluster_train(self, target_anchors, source_anchors):
        before = self.main_updates
        SPATTA.cluster_train(self, target_anchors, source_anchors)
        self.student_updates += self.main_updates - before

    @torch.enable_grad()
    def _train_shadows(self, step, released):
        quota = self.trace.quotas[step]
        if not released or not quota:
            return
        before = fingerprint(self.model)
        ready = sorted(self.received)
        count_p = self.source_anchors.num_elem() if self.source_anchors is not None else 0
        gen = torch.Generator().manual_seed(self.config.random_seed + 1000003 * step + 29)
        head_indices = None
        for iteration in range(quota):
            positions = torch.randint(len(ready), (self.config.train.train_bs,), generator=gen)
            if count_p:
                torch.randint(count_p, (self.config.train.train_bs,), generator=gen)
            if quota >= 2 and iteration == quota - 2:
                head_indices = torch.tensor([ready[int(i)] for i in positions])
        head_loss = None
        if head_indices is not None:
            images = self.query_anchors.data[head_indices].to(self.config.device)
            labels = torch.tensor([self.received[int(i)] for i in head_indices], device=self.config.device)
            with torch.no_grad():
                features = self.teacher[0](images)
            optimizer = torch.optim.SGD(self.head.parameters(), lr=D["head_lr"])
            anchor = sum((p - origin).square().mean() for p, origin in zip(self.head.parameters(), self.origin))
            loss = F.cross_entropy(self.head(features), labels) + D["head_anchor"] * anchor
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite shadow head loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            head_loss = float(loss.detach())
            self.head_updates += 1
            self.compute["head_updates"] += 1
        gen = torch.Generator().manual_seed(self.config.random_seed + 5000011 * step + 97)
        positions = torch.randint(len(self.risk_memory), (self.config.train.train_bs,), generator=gen)
        records = [self.risk_memory[int(i)] for i in positions]
        features = torch.stack([r[1] for r in records]).to(self.config.device)
        labels = torch.stack([r[2] for r in records]).to(self.config.device)
        optimizer = torch.optim.SGD(self.risk.parameters(), lr=D["reliability_lr"])
        loss = correctness_loss(self.risk, features, labels, D["reliability_l2"])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite shadow reliability loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.reliability_updates += 1
        self.compute["reliability_updates"] += 1
        if self.first_learning_step is None:
            self.first_learning_step = step
        self.shadow_rows.append({"step": step, "head_loss": head_loss, "risk_loss": float(loss.detach()),
            "head_anchor_indices": json.dumps(head_indices.tolist()) if head_indices is not None else "[]",
            "risk_anchor_indices": json.dumps([r[0] for r in records])})
        if fingerprint(self.model) != before:
            raise AssertionError("Shadow modules changed formal SP model")

    @torch.no_grad()
    def on_batch_end(self, step, domain, data, target, outputs, released):
        SPATTA.on_batch_end(self, step, domain, data, target, outputs, released)
        lo, hi = self.trace.bounds[step]
        self.evaluation_window.append(dict(zip(range(lo, hi), target.cpu().tolist())))
        evaluation = {sid: y for batch in self.evaluation_window for sid, y in batch.items()}
        # Only attach labels after all current decisions are fixed; never read these
        # rows from training or threshold selection code.
        for row in self.pending_observations:
            label = evaluation[row["sample_id"]]
            row["evaluation_label"] = label
            row["source_correct"] = int(row["source_prediction"] == label)
            row["repair_correct"] = int(row["repair_prediction"] == label)
            self.observation_rows.append(row)
        self.pending_observations.clear()
        with isolated_randomness():
            self._train_shadows(step, released)
        self.budget_rows[-1].update(self.compute)

    def run_stream(self, output_dir, progress=None):
        # Intentionally bypass SelectiveFeedback.run_stream: head/risk are overhead,
        # not formal quota, and there is no current-target formal training here.
        result = SPATTA.run_stream(self, output_dir, progress)
        for name, rows, fields in (
            ("score_observations.csv", self.observation_rows, ["step", "sample_id", "warm"]),
            ("filter_funnel.csv", self.funnel_rows, ["step", "branch", "stage", "passed"]),
            ("query_scores.csv", self.query_rows, ["anchor_index", "branch", "phase"]),
            ("policy_counts.csv", self.policy_rows, ["step", "warm", "select_count", "correct_total_count"]),
            ("shadow_updates.csv", self.shadow_rows, ["step", "head_loss", "risk_loss"]),
        ):
            save_rows(Path(output_dir) / name, rows, fields)
        result.update(method=METHOD, student_updates=self.student_updates,
            shadow_head_updates=self.head_updates, shadow_reliability_updates=self.reliability_updates,
            first_learning_step=self.first_learning_step, first_warm_step=self.first_warm_step,
            observation_count=len(self.observation_rows), returned_query_count=len(self.query_rows) // 2)
        if self.student_updates != sum(self.trace.quotas):
            raise AssertionError("Shadow diagnosis changed formal quota")
        return result
