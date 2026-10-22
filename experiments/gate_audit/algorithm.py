"""Log pseudo-supervision; oracle labels never enter the normal gate formula."""

import csv
import gzip
import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.fcra.algorithm import DEFAULTS
from experiments.fcra_ablation.algorithm import AblationFCRA


MODES = ("baseline", "oracle_correct", "random_count")
FIELDS = ("step", "domain", "iteration", "draw", "sample_id", "source_domain", "label",
          "truth", "correct", "gate", "effective_gate", "raw_ce", "neighbors", "support",
          "similarity_mean", "feedback_error_fraction", "cross_source_fraction",
          "cross_current_fraction", "feedback_age_mean", "support_bin", "neighbor_bin", "neighbor_events")


class GateAudit(AblationFCRA):
    def __init__(self, config, trace, mode, output):
        if mode not in MODES:
            raise ValueError("Unknown diagnostic mode")
        super().__init__(config, trace, "calibration_only")
        self.mode = mode
        self.output = Path(output)
        self.audit_truth = {}  # Kept separate from received/oracle/training memory.
        self.audit_domains = {}
        self.pseudo_ids = []
        self.domain = None
        self.prefix_hash = None
        self.draw_hash = hashlib.sha256()
        self.random_protection = torch.Generator().manual_seed(config.random_seed + 89017)
        self.audit_stream = None
        self.audit_writer = None

    def _audit_observe_batch(self, step, domain, ids, truth):
        self.domain = domain
        for sample, label in zip(ids.tolist(), truth.tolist()):
            self.audit_truth[sample] = int(label)
            self.audit_domains[sample] = domain

    def sample_select(self, model, data, target, anchors, *args, **kwargs):
        # The inherited selector returns current-batch positions for the
        # fixed-teacher low-entropy branch; it appends entries without pruning.
        result = super().sample_select(model, data, target, anchors, *args, **kwargs)
        if kwargs.get("use_pseudo_label", False):
            self.pseudo_ids.extend(int(target[int(position)]) for position in result[1])
            if len(self.pseudo_ids) != result[2].num_elem():
                raise RuntimeError("Pseudo-memory provenance lost alignment")
        return result

    def _state_fingerprint(self):
        """Hash model, memory and RNG state before the first Sketch prediction."""
        digest = hashlib.sha256()

        def tensor(value):
            value = value.detach().cpu().contiguous()
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(value.numpy().tobytes())

        for network in (self.model, self.teacher):
            for key, value in sorted(network.state_dict().items()):
                digest.update(key.encode())
                tensor(value)
        for memory in (self.query_anchors, self.source_anchors):
            if memory is None:
                digest.update(b"none")
            else:
                for name in ("data", "target", "feats", "weight"):
                    tensor(memory[name])
        for prediction, feature, correct in self.calibration_events:
            digest.update(pickle.dumps((prediction, correct)))
            tensor(feature)
        tensor(torch.get_rng_state())
        for state in torch.cuda.get_rng_state_all():
            tensor(state)
        digest.update(pickle.dumps((sorted(self.received.items()), self.pseudo_ids,
                                   self.n_clusters, self.source_n_clusters,
                                   random.getstate(), np.random.get_state())))
        return digest.hexdigest()

    def select_queries(self, data, stream_ids):
        next_step = self.stream_step + 1
        if self.trace.batches[next_step]["domain"] == "Sketch" and self.prefix_hash is None:
            self.prefix_hash = self._state_fingerprint()
            # This is an audit artifact, not a resumable training checkpoint.
            torch.save({"model": self.model.state_dict(), "prefix_sha256": self.prefix_hash},
                       self.output / "sketch_start_model.pt")
        return super().select_queries(data, stream_ids)

    def _audit_supervised_draw(self, indices):
        if self.domain == "Sketch":
            self.draw_hash.update(pickle.dumps(("supervised", self.stream_step,
                                                self.slow_per_step[self.stream_step], indices.tolist())))

    @torch.no_grad()
    def _neighbor_details(self, indices):
        reference = self.source_anchors
        features = F.normalize(reference.feats[indices].to(self.config.device), dim=1)
        labels = reference.target[indices].to(self.config.device).long()
        events = list(self.calibration_events)
        rows = self.calibration_rows[-len(events):] if events else []
        if not events:
            return [[] for _ in indices]
        cache = torch.stack([event[1] for event in events]).to(features.device)
        predictions = torch.tensor([event[0] for event in events], device=features.device)
        similarity = features @ cache.T
        eligible = ((labels[:, None] == predictions[None, :])
                    & (similarity >= DEFAULTS["gate_similarity"]))
        values, positions = similarity.masked_fill(~eligible, -1).topk(
            min(DEFAULTS["gate_neighbors"], len(events)), dim=1)
        details = []
        for sims, ids in zip(values.cpu().tolist(), positions.cpu().tolist()):
            details.append([(sim, events[index][2], rows[index]["query_domain"],
                             rows[index]["return_step"], rows[index]["anchor_index"])
                            for sim, index in zip(sims, ids)
                            if sim >= DEFAULTS["gate_similarity"]])
        return details

    def _audit_pseudo_draw(self, indices, gates, losses):
        ids = indices.tolist()
        if len(self.pseudo_ids) != self.source_anchors.num_elem():
            raise RuntimeError("Incomplete pseudo provenance")
        samples = [self.pseudo_ids[index] for index in ids]
        labels = self.source_anchors.target[indices].long().cpu().tolist()
        # Both flags are diagnostic-only, detached from the gradient graph.
        correct = torch.tensor([label == self.audit_truth[sample]
                                for sample, label in zip(samples, labels)], device=gates.device)
        effective = gates
        if self.domain == "Sketch":
            self.draw_hash.update(pickle.dumps((self.stream_step,
                                                self.slow_per_step[self.stream_step], ids)))
            if self.mode == "oracle_correct":
                effective = torch.where(correct, torch.ones_like(gates), gates)
            elif self.mode == "random_count":
                # Match the number of actually downweighted correct draws, not
                # their identity or total restored weight mass. This control
                # also uses hidden correctness for the count and is diagnostic.
                eligible = torch.where(gates.detach().cpu() < 1)[0]
                count = int((correct & (gates < 1)).sum())
                chosen = eligible[torch.randperm(len(eligible), generator=self.random_protection)[:count]]
                effective = gates.clone()
                effective[chosen.to(gates.device)] = 1
        with torch.no_grad():
            details = self._neighbor_details(indices)
            for draw, (sample, label, ok, gate, actual, ce, neighbors) in enumerate(zip(
                    samples, labels, correct.cpu().tolist(), gates.detach().cpu().tolist(),
                    effective.detach().cpu().tolist(), losses.detach().cpu().tolist(), details)):
                weights = [np.exp((item[0] - 1) / DEFAULTS["gate_temperature"]) for item in neighbors]
                support = float(sum(weights))
                def weighted(position, transform):
                    return (sum(w * transform(item[position]) for w, item in zip(weights, neighbors))
                            / support if support else None)
                self.audit_writer.writerow(dict(zip(FIELDS, (
                    self.stream_step, self.domain, self.slow_per_step[self.stream_step], draw,
                    sample, self.audit_domains[sample], label, self.audit_truth[sample], int(ok),
                    gate, actual, ce, len(neighbors), support,
                    sum(item[0] for item in neighbors) / len(neighbors) if neighbors else None,
                    weighted(1, lambda x: 1 - x),
                    weighted(2, lambda x: int(x != self.audit_domains[sample])),
                    weighted(2, lambda x: int(x != self.domain)),
                    weighted(3, lambda x: self.stream_step - x),
                    "zero" if support == 0 else "weak" if support <= 1 else "medium" if support <= 3 else "strong",
                    "0" if not neighbors else "1-2" if len(neighbors) <= 2 else "3-8",
                    json.dumps([{"similarity": item[0], "correct": item[1], "domain": item[2],
                                 "return_step": item[3], "anchor_index": item[4]} for item in neighbors])))))
        return effective

    def run_stream(self, output_dir, progress=None):
        with gzip.open(self.output / "pseudo_draws.csv.gz", "wt", newline="", encoding="utf-8") as stream:
            self.audit_writer = csv.DictWriter(stream, fieldnames=FIELDS)
            self.audit_writer.writeheader()
            result = super().run_stream(output_dir, progress=progress)
        result.update(method="GateAudit", mode=self.mode, variant=self.mode,
                      diagnostic_only=self.mode != "baseline",
                      prefix_sha256=self.prefix_hash, train_draw_sha256=self.draw_hash.hexdigest())
        return result
