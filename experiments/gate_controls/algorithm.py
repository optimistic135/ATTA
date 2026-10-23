"""Transform gates without exposing diagnostic truth to the transformation."""

import hashlib
from collections import defaultdict

import torch
from torch.nn import functional as F

from experiments.fcra.algorithm import DEFAULTS
from experiments.fcra_ablation.algorithm import AblationFCRA
from experiments.vlcs_calibration.results import write_csv


METHOD = "source-gate-controls"
VARIANTS = ("uniform_gate", "shuffled_gate")


@torch.no_grad()
def transform_gates(gates, mode, generator):
    if mode == "uniform_gate":
        return torch.ones_like(gates) * gates.mean()
    if mode == "shuffled_gate":
        order = torch.randperm(len(gates), generator=generator, device="cpu")
        return gates[order.to(gates.device)]
    raise ValueError(f"Unknown gate control: {mode}")


class GateControl(AblationFCRA):
    def __init__(self, config, trace, mode):
        if mode not in VARIANTS:
            raise ValueError("Unknown gate control")
        super().__init__(config, trace, "calibration_only")
        self.mode = mode
        self.shuffle_rng = torch.Generator(device="cpu").manual_seed(config.random_seed + 93471)
        self.audit_truth, self.audit_domains = {}, {}
        self.pseudo_ids = []
        self.domain = None
        self.gate_rows = []
        self.samples = defaultdict(lambda: [0, 0., 0., 0., 0., 0.])
        self.draw_hash = hashlib.sha256()

    def _audit_observe_batch(self, step, domain, ids, truth):
        self.domain = domain
        # Evaluation-only dictionaries: never read by matching or transformation.
        self.audit_truth.update(zip(ids.tolist(), truth.tolist()))
        self.audit_domains.update((sample, domain) for sample in ids.tolist())

    def sample_select(self, model, data, target, anchors, *args, **kwargs):
        result = super().sample_select(model, data, target, anchors, *args, **kwargs)
        if kwargs.get("use_pseudo_label", False):
            self.pseudo_ids.extend(int(target[int(position)]) for position in result[1])
            if len(self.pseudo_ids) != result[2].num_elem():
                raise RuntimeError("Pseudo-memory provenance lost alignment")
        return result

    def _audit_supervised_draw(self, indices):
        self.draw_hash.update(str(("supervised", self.stream_step,
                                  self.slow_per_step[self.stream_step], indices.tolist())).encode())

    @torch.no_grad()
    def _feedback_gates(self, features, labels):
        gates = super()._feedback_gates(features, labels)
        events = list(self.calibration_events)
        if not events:
            self.covered = torch.zeros(len(features), dtype=torch.bool, device=features.device)
        else:
            cache = torch.stack([event[1] for event in events]).to(features.device)
            predictions = torch.tensor([event[0] for event in events], device=features.device)
            self.covered = ((features @ cache.T >= DEFAULTS["gate_similarity"])
                            & (labels[:, None] == predictions[None, :])).any(1)
        return gates

    @torch.no_grad()
    def _audit_pseudo_draw(self, indices, gates, losses):
        # Compute the training weights before any diagnostic truth lookup.
        effective = transform_gates(gates, self.mode, self.shuffle_rng)
        if not torch.allclose(effective.mean(), gates.mean(), atol=1e-6, rtol=1e-6):
            raise RuntimeError("Gate mass was not preserved")
        self.draw_hash.update(str(("pseudo", self.stream_step,
                                  self.slow_per_step[self.stream_step], indices.tolist())).encode())
        row = {"step": self.stream_step, "domain": self.domain,
               "iteration": self.slow_per_step[self.stream_step], "draws": len(gates),
               "raw_gate_mean": float(gates.mean()), "effective_gate_mean": float(effective.mean()),
               "covered_fraction": float(self.covered.float().mean()),
               "mass_error": float((effective.sum() - gates.sum()).abs())}
        if len(self.pseudo_ids) != self.source_anchors.num_elem():
            raise RuntimeError("Missing pseudo-memory provenance")
        samples = [self.pseudo_ids[index] for index in indices.tolist()]
        labels = self.source_anchors.target[indices].long().cpu().tolist()
        correctness = [int(label == self.audit_truth[sample]) for sample, label in zip(samples, labels)]
        for ok in (0, 1):
            mask = torch.tensor([value == ok for value in correctness], device=gates.device)
            row[f"correct_{ok}_draws"] = int(mask.sum())
            for name, values in (("raw_gate", gates), ("effective_gate", effective),
                                 ("removed_ce", (1 - effective) * losses.detach())):
                row[f"correct_{ok}_{name}_sum"] = float(values[mask].sum())
        self.gate_rows.append(row)
        for sample, ok, raw, actual, ce, covered in zip(
                samples, correctness, gates.cpu().tolist(), effective.cpu().tolist(),
                losses.detach().cpu().tolist(), self.covered.cpu().tolist()):
            key = (self.domain, self.audit_domains[sample], sample, ok)
            acc = self.samples[key]
            for position, value in enumerate((1, raw, actual, (1 - raw) * ce,
                                               (1 - actual) * ce, int(covered))):
                acc[position] += value
        return effective

    def run_stream(self, output_dir, progress=None):
        result = super().run_stream(output_dir, progress=progress)
        write_csv(output_dir / "gate_updates.csv", self.gate_rows or [{"draws": 0}])
        rows = []
        for (domain, source_domain, sample, correct), acc in sorted(self.samples.items()):
            count, raw, actual, raw_ce, actual_ce, covered = acc
            rows.append({"domain": domain, "source_domain": source_domain, "sample_id": sample,
                         "correct": correct, "draws": count, "raw_gate_mean": raw / count,
                         "effective_gate_mean": actual / count,
                         "raw_removed_ce_mean": raw_ce / count,
                         "effective_removed_ce_mean": actual_ce / count,
                         "covered_fraction": covered / count})
        write_csv(output_dir / "gate_samples.csv", rows or [{"draws": 0}])
        result.update(method=METHOD, variant=self.mode, dataset=self.config.dataset.name,
                      source_reference="frozen source model", train_draw_sha256=self.draw_hash.hexdigest(),
                      pseudo_draws=sum(row["draws"] for row in self.gate_rows))
        return result
