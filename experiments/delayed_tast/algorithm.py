"""Frozen source features, one lightweight update per batch, and causal label repair."""
import hashlib
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F

from ATTA.kernel.algorithms.Base import AlgBase
from experiments.delayed_feedback.simatta import DelayedSimATTA
from experiments.delayed_feedback.datasets import EXPERIMENTS
from experiments.delayed_feedback.protocol import DelayedOracle
from experiments.delayed_feedback.run import dump_json
from experiments.fcra_relay.stream import save_rows
from experiments.simatta_rootcause.algorithm import isolated_randomness, fingerprint
from .head import ProjectionEnsemble, prototype_logits, neighbor_targets
from .spec import DEFAULTS as D, GROUPS


class DelayedTAST(AlgBase):
    train_on_env = DelayedSimATTA.train_on_env

    def __init__(self, config, group, trace):
        if group not in GROUPS:
            raise ValueError("Unknown delayed TAST arm")
        super().__init__(config)
        self.group, self.trace = group, trace
        self.experiment = EXPERIMENTS["PACS"]
        self.oracle = DelayedOracle(D["delay"])
        self.model.eval().requires_grad_(False)
        self.source_fingerprint = fingerprint(self.model)
        with isolated_randomness():
            self.head = ProjectionEnsemble(self.encoder.n_outputs).to(config.device)
        self.initial_head_hash = fingerprint(self.head)
        self.optimizer = torch.optim.Adam(self.head.parameters(), lr=D["head_lr"])
        self.classes = self.fc.out_features
        with torch.no_grad():
            warm = self.fc.weight.detach().clone()
            scores = self.fc(warm)
            labels = scores.argmax(-1)
            entropy = -(scores.softmax(-1) * scores.log_softmax(-1)).sum(-1)
        self.bank = {-(i + 1): (warm[i].cpu(), int(labels[i]), float(entropy[i])) for i in range(len(warm))}
        self.query_cache, self.returned = {}, {}
        self.batch_rows, self.update_rows, self.feedback_rows, self.target_rows = [], [], [], []
        self.formal_updates = self.replay_updates = self.replay_draws = 0
        self.replayed = set()
        self.draw_hash, self.support_hash, self.source_hash = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
        self.encoder_calls = self.encoder_images = 0

    def prune(self):
        retained = {}
        for label in range(self.classes):
            candidates = [sid for sid, entry in self.bank.items() if entry[1] == label]
            selected = sorted(candidates, key=lambda sid: (self.bank[sid][2], sid))[:D["support_per_source_class"]]
            retained.update((sid, self.bank[sid]) for sid in selected)
        self.bank = retained

    def support_arrays(self):
        # Returned slots exist in ALL arms, with identical IDs/features. Only
        # Repair uses their true labels and authoritative neighbor votes.
        entries = dict(self.bank)
        for sid in self.returned:
            entries[sid] = self.query_cache[sid]
        ids = sorted(entries)
        features = torch.stack([entries[sid][0] for sid in ids]).to(self.config.device)
        source_labels = torch.tensor([entries[sid][1] for sid in ids], device=self.config.device)
        repaired_labels = torch.tensor([self.returned.get(sid, entries[sid][1]) for sid in ids], device=self.config.device)
        known = torch.tensor([sid in self.returned for sid in ids], device=self.config.device, dtype=torch.bool)
        source = F.one_hot(source_labels, self.classes).float()
        repaired = F.one_hot(repaired_labels, self.classes).float()
        return ids, features, source, repaired, known

    @torch.no_grad()
    def label_only_diagnostic(self, features):
        ids, supports, source, repaired, known = self.support_arrays()
        _, plain, _ = neighbor_targets(self.head, features, supports, source, torch.zeros_like(known))
        _, corrected, exposed = neighbor_targets(self.head, features, supports, repaired, known)
        return plain, corrected, exposed, ids

    @torch.enable_grad()
    def update(self, step, domain, features):
        ids, supports, source, repaired, known = self.support_arrays()
        labels = repaired if self.group == GROUPS[2] else source
        mask = known if self.group == GROUPS[2] else torch.zeros_like(known)
        for sid in ids:
            self.support_hash.update(f"{step}:{sid};".encode())
        targets, _, _ = neighbor_targets(self.head, features, supports, labels, mask)
        ready = sorted(self.returned)
        generator = torch.Generator().manual_seed(self.config.random_seed + 9000011 * step + 37)
        draw = [ready[int(i)] for i in torch.randint(len(ready), (D["truth_batch_size"],), generator=generator)] if ready else []
        self.draw_hash.update(f"{step}:".encode())
        self.draw_hash.update(torch.tensor(draw, dtype=torch.long).numpy().tobytes())
        logits = prototype_logits(self.head, features, supports, labels)
        # Match upstream TAST's sum-over-ensemble, mean-over-batch-and-class KL.
        unsupervised = F.kl_div(logits.log_softmax(-1), targets.detach(), reduction="none").mean((1, 2)).sum()
        supervised = None
        loss = unsupervised
        if draw and self.group != GROUPS[0]:
            truth_features = torch.stack([self.query_cache[sid][0] for sid in draw]).to(features.device)
            truth_labels = torch.tensor([self.returned[sid] for sid in draw], device=features.device)
            prediction = prototype_logits(self.head, truth_features, supports, labels)
            supervised = sum(F.cross_entropy(member, truth_labels) for member in prediction) / self.classes
            loss = loss + D["truth_coefficient"] * supervised
            self.replay_updates += 1
            self.replay_draws += len(draw)
            self.replayed.update(draw)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite delayed TAST objective")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.formal_updates += 1
        self.update_rows.append({"step": step, "domain": domain, "supports": len(ids), "returned_memory": len(ready),
            "unsupervised_loss": float(unsupervised.detach()),
            "supervised_loss": float(supervised.detach()) if supervised is not None else None,
            "replay_count": len(draw) if supervised is not None else 0,
            "sampled_ids": ":".join(map(str, draw)), "formal_updates": 1})

    @torch.no_grad()
    def run_stream(self, output_dir, progress=None):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        offset = step = 0
        domains = []
        for env in self.config.dataset.test_envs[1:]:
            domain = self.experiment.domains[env]
            loader = self.fast_loader[env]
            if progress:
                progress({"event": "domain", "domain": domain, "total": len(loader)})
            correct_total = count_total = 0
            for domain_step, (images, target) in enumerate(loader):
                if step >= self.trace.steps or self.trace.bounds[step] != (offset, offset + len(images)) or self.trace.batches[step]["domain"] != domain:
                    raise ValueError("Stream differs from fixed reference")
                images = images.to(self.config.device)
                self.model.eval()
                features = self.encoder(images).detach()
                source_logits = self.fc(features)
                self.encoder_calls += 1
                self.encoder_images += len(images)
                self.source_hash.update(features.cpu().numpy().tobytes())
                before_feedback = not self.returned
                self.prune()
                _, supports, plain_labels, repaired_labels, known = self.support_arrays()
                labels = repaired_labels if self.group == GROUPS[2] else plain_labels
                mask = known if self.group == GROUPS[2] else torch.zeros_like(known)
                _, probability, _ = neighbor_targets(self.head, features, supports, labels, mask)
                # Same snapshot/IDs: only label and authoritative-vote replacement differs.
                plain, corrected, exposed, _ = self.label_only_diagnostic(features)
                entropy = -(source_logits.softmax(-1) * source_logits.log_softmax(-1)).sum(-1)
                for position in range(len(images)):
                    sid = offset + position
                    self.bank[sid] = (features[position].cpu().clone(), int(source_logits[position].argmax()), float(entropy[position]))
                self.prune()
                for record in self.trace.queries[step]:
                    sid = int(record["sample_id"])
                    position = sid - offset
                    self.query_cache[sid] = (features[position].cpu().clone(), int(source_logits[position].argmax()), float(entropy[position]))
                    self.oracle.query(sid, int(record["anchor_index"]), int(target[position]), step, domain)
                release = self.oracle.release(step, domain)
                for index, label in release:
                    record = self.trace.records[index]
                    sid = int(record["sample_id"])
                    if sid in self.returned:
                        raise ValueError("Duplicate truth return")
                    self.returned[sid] = label
                    self.feedback_rows.append({"step": step, "domain": domain, "sample_id": sid, "label": label,
                        "source_prediction": self.query_cache[sid][1], "source_wrong": self.query_cache[sid][1] != label,
                        "used_for_repair": self.group == GROUPS[2], "in_entropy_bank": sid in self.bank})
                self.update(step, domain, features)
                # Evaluation labels are used only here, after all decisions and training.
                truth = target.cpu()
                predicted = probability.argmax(-1).cpu()
                source_predicted = source_logits.argmax(-1).cpu()
                original = plain.argmax(-1).cpu().eq(truth)
                replacement = corrected.argmax(-1).cpu().eq(truth)
                correct = int(predicted.eq(truth).sum())
                correct_total += correct
                count_total += len(truth)
                self.batch_rows.append({"step": step, "domain": domain, "samples": len(truth), "pre_correct": correct,
                    "source_correct": int(source_predicted.eq(truth).sum()), "before_feedback": before_feedback,
                    "released": len(release), "returned_memory": len(self.returned),
                    "label_repair_plain_correct": int(original.sum()), "label_repair_corrected_correct": int(replacement.sum()),
                    "label_repair_fixed": int((~original & replacement).sum()), "label_repair_broken": int((original & ~replacement).sum()),
                    "repair_exposed": int(exposed.sum()),
                    "online_predictions_sha256": hashlib.sha256(predicted.numpy().tobytes()).hexdigest()})
                for i in range(len(truth)):
                    self.target_rows.append({"step": step, "domain": domain, "sample_id": offset + i,
                        "plain_prediction": int(plain[i].argmax()), "repaired_prediction": int(corrected[i].argmax()),
                        "evaluation_label": int(truth[i]), "repair_exposed": bool(exposed[i]),
                        "online_prediction": int(predicted[i])})
                if progress:
                    progress({"event": "batch", "domain": domain, "total": len(loader), "completed": domain_step + 1,
                        "accuracy_pct": 100 * correct_total / count_total, **self.oracle.counts()})
                offset += len(truth)
                step += 1
            domains.append({"domain": domain, "samples": count_total, "correct": correct_total,
                            "accuracy_pct": 100 * correct_total / count_total})
        torch.cuda.synchronize()
        self.trace.verify_feedback(self.oracle.audit())
        if step != self.trace.steps or self.formal_updates != step * D["steps_per_batch"]:
            raise ValueError("Incomplete stream or head update budget")
        if fingerprint(self.model) != self.source_fingerprint or any(p.grad is not None for p in self.model.parameters()):
            raise AssertionError("Frozen source model changed or received gradients")
        output = Path(output_dir)
        for name, rows, fields in (("batches.csv", self.batch_rows, ["step", "domain", "samples"]),
            ("domains.csv", domains, ["domain", "accuracy_pct"]), ("feedback.csv", self.oracle.audit(), ["sample_id", "return_step"]),
            ("updates.csv", self.update_rows, ["step", "formal_updates"]), ("returned_labels.csv", self.feedback_rows, ["step", "sample_id", "label"]),
            ("target_repair.csv", self.target_rows, ["step", "sample_id", "plain_prediction", "repaired_prediction"])):
            save_rows(output / name, rows, fields)
        samples = sum(r["samples"] for r in self.batch_rows)
        result = {"method": self.group, "seed": self.config.random_seed, "delay": 100,
            **{r["domain"] + "_accuracy_pct": r["accuracy_pct"] for r in domains},
            "stream_accuracy_pct": 100 * sum(r["pre_correct"] for r in self.batch_rows) / samples,
            "domain_macro_accuracy_pct": statistics.mean(r["accuracy_pct"] for r in domains),
            **self.oracle.counts(), "initial_head_sha256": self.initial_head_hash,
            "frozen_source_sha256": self.source_fingerprint, "source_features_sha256": self.source_hash.hexdigest(),
            "replay_draws_sha256": self.draw_hash.hexdigest(), "support_ids_sha256": self.support_hash.hexdigest(),
            "formal_updates": self.formal_updates, "replay_updates": self.replay_updates, "replay_draws": self.replay_draws,
            "replay_unique_labels": len(self.replayed), "repaired_labels": len(self.returned) if self.group == GROUPS[2] else 0,
            "repaired_wrong_source_labels": sum(r["source_wrong"] for r in self.feedback_rows) if self.group == GROUPS[2] else 0,
            "source_stream_accuracy_pct": 100 * sum(r["source_correct"] for r in self.batch_rows) / samples,
            "repair_exposure_pct": 100 * sum(r["repair_exposed"] for r in self.batch_rows) / samples,
            "label_repair_target_gain_pp": 100 * sum(r["label_repair_fixed"] - r["label_repair_broken"] for r in self.batch_rows) / samples,
            "label_repair_fixed_pct": 100 * sum(r["label_repair_fixed"] for r in self.batch_rows) / samples,
            "label_repair_broken_pct": 100 * sum(r["label_repair_broken"] for r in self.batch_rows) / samples,
            "mean_unsupervised_loss": statistics.mean(r["unsupervised_loss"] for r in self.update_rows),
            "mean_supervised_loss": statistics.mean(r["supervised_loss"] for r in self.update_rows if r["supervised_loss"] is not None) if self.replay_updates else None,
            "encoder_forward_calls": self.encoder_calls, "encoder_forward_images": self.encoder_images, "encoder_backward_calls": 0,
            "head_forward_calls": self.head.forward_calls, "head_backward_calls": self.head.backward_calls,
            "stream_seconds": time.perf_counter() - started, "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 2**20}
        result["returned_pct"] = 100 * result["returned"] / result["queried"] if result["queried"] else None
        for phase, flag in (("before_feedback", True), ("after_feedback", False)):
            rows = [r for r in self.batch_rows if r["before_feedback"] == flag]
            count = sum(r["samples"] for r in rows)
            result[phase + "_accuracy_pct"] = 100 * sum(r["pre_correct"] for r in rows) / count if count else None
            if flag:
                result["source_before_feedback_accuracy_pct"] = 100 * sum(r["source_correct"] for r in rows) / count if count else None
        result["waiting_gain_vs_source_pp"] = result["before_feedback_accuracy_pct"] - result["source_before_feedback_accuracy_pct"]
        dump_json(output / "compute.json", {k: result[k] for k in ("encoder_forward_calls", "encoder_forward_images", "encoder_backward_calls", "head_forward_calls", "head_backward_calls")})
        return result
