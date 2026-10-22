"""Collect frozen-model evidence. No optimizer, backward pass or adaptation."""

import hashlib

import numpy as np
import torch
from torch.nn import functional as F

from experiments.delayed_feedback.run import dump_json
from experiments.delayed_feedback.simatta import DelayedSimATTA, write_csv
from experiments.fcra.algorithm import DEFAULTS, FCRA


def state_hash(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.dtype, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class FrozenProbe(FCRA):
    """Reuse the original selector and gate, never FCRA's training/stream loops."""

    @torch.no_grad()
    def collect(self, output):
        self.model.requires_grad_(False).eval()
        self.teacher.requires_grad_(False).eval()
        before = {"model": state_hash(self.model), "source": state_hash(self.teacher)}
        if before["model"] != before["source"]:
            raise RuntimeError("Frozen query and source models must start identically")
        arrays = {key: [] for key in (
            "sample_id", "step", "domain", "prediction", "truth", "entropy",
            "gate", "covered", "features")}
        batches, offset, step = [], 0, 0
        for env_id in self.config.dataset.test_envs[1:]:
            domain = self.experiment.domains[env_id]
            for data, truth in self.fast_loader[env_id]:
                self.stream_step = step
                data = data.to(self.config.device)
                ids = torch.arange(offset, offset + len(data))
                # This selector receives stream IDs, never evaluation labels.
                previous = self.query_anchors.num_elem() if self.query_anchors is not None else 0
                DelayedSimATTA.select_queries(self, data, ids)
                current = self.query_anchors.num_elem() if self.query_anchors is not None else 0
                raw = self.teacher[0](data)
                logits = self.teacher[1](raw)
                features = F.normalize(raw, dim=1)
                predictions = logits.argmax(1)
                entropy = self.softmax_entropy(logits)
                if not bool(torch.isfinite(features).all() and torch.isfinite(entropy).all()):
                    raise FloatingPointError("Non-finite frozen-source output")
                queried = torch.zeros(len(data), dtype=torch.bool)
                for index in range(previous, current):
                    sample_id = int(self.query_anchors.target[index])
                    position = sample_id - offset
                    if not 0 <= position < len(data):
                        raise ValueError("New query is not in the current batch")
                    queried[position] = True
                    self.teacher_predictions[index] = int(predictions[position])
                    self.teacher_features[index] = features[position].cpu()
                    # Only the oracle receives query truth, after selection.
                    self.oracle.query(sample_id, index, int(truth[position]), step, domain)
                self.budgets += current - previous
                # All eligible low-entropy, unqueried examples, once per stream.
                positions = torch.where((entropy.cpu() < self.config.atta.SimATTA.el) & ~queried)[0]
                if len(positions):
                    device_positions = positions.to(data.device)
                    held_features = features[device_positions]
                    held_predictions = predictions[device_positions]
                    # Gate reads only previous returns. Even D=0 current labels
                    # are released below, after the entire batch is scored.
                    gates = self._feedback_gates(held_features, held_predictions)
                    if self.calibration_events:
                        events = list(self.calibration_events)
                        cached = torch.stack([event[1] for event in events]).to(data.device)
                        cached_classes = torch.tensor([event[0] for event in events], device=data.device)
                        covered = ((held_features @ cached.T >= DEFAULTS["gate_similarity"])
                                   & (held_predictions[:, None] == cached_classes[None, :])).any(1)
                    else:
                        covered = torch.zeros(len(positions), dtype=torch.bool, device=data.device)
                    values = {
                        "sample_id": ids[positions], "step": torch.full((len(positions),), step),
                        "domain": torch.full((len(positions),), env_id),
                        "prediction": held_predictions.cpu(), "truth": truth[positions].cpu(),
                        "entropy": entropy[device_positions].cpu(), "gate": gates.cpu(),
                        "covered": covered.cpu(), "features": held_features.cpu(),
                    }
                    for key, value in values.items():
                        arrays[key].append(value.numpy())
                released = self.oracle.release(step, domain)
                self.received.update(released)
                self._record_calibration_events()
                if not self.target_cluster:
                    self.n_clusters = 0
                self.n_clusters += self.nc_increase
                batches.append({"step": step, "domain": domain, "samples": len(data),
                                "new_queries": current - previous, "new_returns": len(released),
                                "evaluation_samples": len(positions), "cache_size": len(self.calibration_events)})
                print(f"Frozen probe seed={self.config.random_seed} {domain} step={step} "
                      f"queries={current - previous} evaluated={len(positions)}", flush=True)
                offset += len(data)
                step += 1
        after = {"model": state_hash(self.model), "source": state_hash(self.teacher)}
        if before != after:
            raise RuntimeError("Frozen model parameters or buffers changed")
        packed = {key: np.concatenate(value) if value else np.empty(
            (0, self.fc.in_features) if key == "features" else (0,))
                  for key, value in arrays.items()}
        if len(np.unique(packed["sample_id"])) != len(packed["sample_id"]):
            raise RuntimeError("Evaluation samples repeated")
        queried_ids = set(self.query_anchors.target.tolist()) if self.query_anchors is not None else set()
        if queried_ids.intersection(packed["sample_id"].tolist()):
            raise RuntimeError("A query leaked into the held-out evaluation pool")
        np.savez_compressed(output / "evidence.npz", **packed)
        write_csv(output / "batches.csv", batches)
        feedback = self.oracle.audit()
        if feedback:
            write_csv(output / "feedback.csv", feedback)
        if self.calibration_rows:
            write_csv(output / "calibration.csv", self.calibration_rows)
        result = {"seed": self.config.random_seed, "samples": offset,
                  "evaluation_samples": len(packed["sample_id"]), "batches": step,
                  "before_sha256": before, "after_sha256": after,
                  "updates": 0, **self.oracle.counts()}
        dump_json(output / "collection.json", result)
        return result
