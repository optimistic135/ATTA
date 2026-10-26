"""Controlled fixed-trace objectives with fresh-feedback auxiliary validation."""

import copy
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from experiments.delayed_feedback.simatta import DelayedSimATTA
from experiments.fcra_relay.stream import save_rows
from experiments.simatta_rootcause.algorithm import isolated_randomness, restore_random, fingerprint
from experiments.simatta_rootcause.spec import DEFAULTS as PROBE
from .spec import DEFAULTS as D


class SPATTA(DelayedSimATTA):
    def __init__(self, config, group, trace):
        super().__init__(config, D["delay"])
        self.group, self.trace = group, trace
        if dict(config.atta.SimATTA) != trace.simatta:
            raise ValueError("SimATTA configuration differs from reference")
        structure = [(self.experiment.domains[env], min(32, len(loader.sampler) - start))
                     for env in config.dataset.test_envs[1:] for loader in [self.fast_loader[env]]
                     for start in range(0, len(loader.sampler), 32)]
        if structure != [(row["domain"], int(row["samples"])) for row in trace.batches]:
            raise ValueError("Dataset stream shape differs from reference; use the unchanged dataset")
        self.step = -1
        self.pseudo_ids = []
        self.lambda_batch = 0.0
        self.main_updates = self.candidate_updates = self.gate_evaluations = self.gate_acceptances = 0
        self.aux_updates = 0
        self.true_coefficient_sum = self.pseudo_coefficient_sum = 0.0
        self.gate_rows, self.update_rows, self.budget_rows, self.local_rows = [], [], [], []
        self.active, self.monitored = [], set()
        self.formal_hash, self.pseudo_hash = hashlib.sha256(), hashlib.sha256()
        self.compute = {f"{name}_{direction}_{unit}": 0 for name in ("main", "teacher", "candidate")
                        for direction in ("forward", "backward") for unit in ("calls", "images")}
        self._counter(self.model, "main")
        self._counter(self.teacher, "teacher")

    def _counter(self, model, name):
        def hook(module, inputs, output):
            self.compute[f"{name}_forward_calls"] += 1
            self.compute[f"{name}_forward_images"] += len(inputs[0])
            if torch.is_grad_enabled() and output.requires_grad:
                size = len(inputs[0])
                def backward(gradient):
                    self.compute[f"{name}_backward_calls"] += 1
                    self.compute[f"{name}_backward_images"] += size
                    return gradient
                output.register_hook(backward)
        model[0].register_forward_hook(hook)

    @torch.no_grad()
    def select_queries(self, data, stream_ids):
        self.step += 1
        step = self.step
        if step >= self.trace.steps or self.trace.bounds[step] != (int(stream_ids[0]), int(stream_ids[-1]) + 1):
            raise ValueError("Reference stream clock mismatch")
        self.model.eval()
        features = self.model[0](data)
        output = self.model[1](features)
        records = self.trace.queries[step]
        previous = self.query_anchors.num_elem() if self.query_anchors is not None else 0
        if [int(row["anchor_index"]) for row in records] != list(range(previous, previous + len(records))):
            raise ValueError("Reference anchor insertion order mismatch")
        if records:
            positions = torch.tensor([int(row["sample_id"]) - int(stream_ids[0]) for row in records])
            device_positions = positions.to(data.device)
            self.query_anchors = self.update_anchors(
                self.query_anchors, data[device_positions].cpu(), stream_ids[positions],
                features[device_positions].cpu(), torch.ones(len(positions)))
        return output

    def sample_select(self, model, data, target, *args, **kwargs):
        result = super().sample_select(model, data, target, *args, **kwargs)
        if kwargs.get("use_pseudo_label", False):
            positions = torch.as_tensor(result[1], dtype=torch.long)
            self.pseudo_ids.extend(target.detach().cpu()[positions].tolist())
            if len(self.pseudo_ids) != result[2].num_elem():
                raise ValueError("Pseudo-label image identities lost alignment")
        return result

    @torch.no_grad()
    def _validation_loss(self, model, images, labels):
        model.eval()
        loss = 0.0
        for start in range(0, len(images), self.config.atta.batch_size):
            batch = images[start:start + self.config.atta.batch_size].to(self.config.device)
            loss += float(F.cross_entropy(model(batch), labels[start:start + len(batch)].to(self.config.device), reduction="sum"))
        return loss / len(images)

    @torch.enable_grad()
    def _loss(self, model, truth_images, truth_labels, pseudo_images, pseudo_labels, true_coef, pseudo_coef):
        pseudo = None
        if pseudo_coef > 0 and pseudo_images is not None:
            pseudo = F.cross_entropy(model(pseudo_images), pseudo_labels)
        true = F.cross_entropy(model(truth_images), truth_labels)
        loss = true_coef * true
        if pseudo is not None:
            loss = loss + pseudo_coef * pseudo
        return loss, true, pseudo

    def on_labels_released(self, step, domain, released):
        self.lambda_batch = 0.0
        previous = sorted(self.received)
        reference = self.source_anchors
        pseudo_count = reference.num_elem() if reference is not None else 0
        fresh_ids = {int(self.query_anchors.target[index]) for index, _ in released} if released else set()
        eligible_pseudo = torch.tensor([i for i, sid in enumerate(self.pseudo_ids) if sid not in fresh_ids], dtype=torch.long)
        reason = ("no_fresh" if not released else "no_previous_truth" if not previous
                  else "no_pseudo" if not pseudo_count else "no_heldout_pseudo" if not len(eligible_pseudo)
                  else "no_formal_quota" if not self.trace.quotas[step] else "eligible")
        row = {"step": step, "domain": domain, "fresh_count": len(released),
               "previous_truth": len(previous), "pseudo_count": pseudo_count,
               "eligible": reason == "eligible", "reason": reason,
               "true_candidate_ce": None, "aux_candidate_ce": None, "gain": None,
               "would_accept": False, "selected_lambda": 0.0}
        if reason == "eligible":
            with isolated_randomness() as state:
                row.update(self._gate(step, previous, released, eligible_pseudo, state))
            self.gate_evaluations += 1
            self.gate_acceptances += int(row["would_accept"])
            if row["would_accept"]:
                self.lambda_batch = D["lambda_aux"]
        row["selected_lambda"] = (self.lambda_batch if self.group == "SP-ATTA" else
                                  D["lambda_aux"] if self.group == "Fixed-Aux" and pseudo_count else 0.0)
        self.gate_rows.append(row)

    def _gate(self, step, previous, released, eligible_pseudo, state):
        generator = torch.Generator().manual_seed(self.config.random_seed + 2000003 * step + 71)
        size = self.config.train.train_bs
        old_positions = torch.randint(len(previous), (size,), generator=generator)
        old_ids = torch.tensor([previous[int(pos)] for pos in old_positions])
        ppos = eligible_pseudo[torch.randint(len(eligible_pseudo), (size,), generator=generator)]
        tdata = self.query_anchors.data[old_ids].to(self.config.device)
        tlabels = torch.tensor([self.received[int(index)] for index in old_ids], device=self.config.device)
        pdata = self.source_anchors.data[ppos].to(self.config.device)
        plabels = self.source_anchors.target[ppos].to(self.config.device).long()
        fresh_indices = [index for index, _ in released]
        fresh_data = self.query_anchors.data[fresh_indices]
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
                loss, _, _ = self._loss(branch, tdata, tlabels, pdata, plabels, 1.0, coefficient)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite auxiliary candidate loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            self.candidate_updates += 1
            losses[name] = self._validation_loss(branch, fresh_data, fresh_labels)
            optimizer.zero_grad(set_to_none=True)
            candidates[name] = branch.cpu()
            del optimizer, loss
        gain = losses["TRUE"] - losses["AUX"]
        for start in PROBE["probe_starts"]:
            if start not in self.monitored and start <= step < start + PROBE["trigger_window"]:
                self.monitored.add(start)
                self.active.append({"id": start, "step": step, "models": candidates,
                                    "gain": gain, "accepted": gain > D["gain_margin"], "batches": 0})
                break
        if fingerprint(self.model) != fingerprint(snapshot):
            raise AssertionError("Candidate test changed the formal model state")
        return {"true_candidate_ce": losses["TRUE"], "aux_candidate_ce": losses["AUX"],
                "gain": gain, "would_accept": gain > D["gain_margin"]}

    @torch.enable_grad()
    def cluster_train(self, target_anchors, source_anchors):
        ready = sorted(self.received)
        reference = self.source_anchors
        count_p = reference.num_elem() if reference is not None else 0
        alpha = len(ready) / (len(ready) + count_p) if count_p else 1.0
        if count_p and count_p < self.cold_start:
            alpha = min(0.2, alpha)
        if self.group == "Original-MIX":
            true_coef, pseudo_coef = alpha, 1 - alpha if count_p else 0.0
        else:
            true_coef = 1.0
            pseudo_coef = (D["lambda_aux"] if self.group == "Fixed-Aux" else
                           self.lambda_batch if self.group == "SP-ATTA" else 0.0) if count_p else 0.0
        quota = self.trace.quotas[self.step]
        generator = torch.Generator().manual_seed(self.config.random_seed + 1000003 * self.step + 29)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.atta.SimATTA.lr, momentum=0.9)
        truth_data = self.query_anchors.data[ready]
        truth_labels = torch.tensor([self.received[index] for index in ready], dtype=torch.long)
        self.model.train()
        for iteration in range(quota):
            tpos = torch.randint(len(ready), (self.config.train.train_bs,), generator=generator)
            ppos = (torch.randint(count_p, (self.config.train.train_bs,), generator=generator)
                    if count_p else torch.empty(0, dtype=torch.long))
            self.formal_hash.update(f"{self.step}:{iteration}:".encode())
            self.formal_hash.update(torch.tensor(ready)[tpos].numpy().tobytes())
            self.formal_hash.update(ppos.numpy().tobytes())
            tdata, tlabels = truth_data[tpos].to(self.config.device), truth_labels[tpos].to(self.config.device)
            pdata = reference.data[ppos].to(self.config.device) if count_p else None
            plabels = reference.target[ppos].to(self.config.device).long() if count_p else None
            loss, true, pseudo = self._loss(self.model, tdata, tlabels, pdata, plabels, true_coef, pseudo_coef)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite SP-ATTA objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            self.main_updates += 1
            self.aux_updates += int(pseudo_coef > 0)
            self.true_coefficient_sum += true_coef
            self.pseudo_coefficient_sum += pseudo_coef
            self.update_rows.append({"step": self.step, "iteration": iteration, "alpha": alpha,
                                     "true_coefficient": true_coef, "pseudo_coefficient": pseudo_coef,
                                     "true_loss": float(true.detach()),
                                     "pseudo_loss": float(pseudo.detach()) if pseudo is not None else None})

    @torch.no_grad()
    def on_batch_end(self, step, domain, data, target, outputs, released):
        for probe in list(self.active):
            if step <= probe["step"]:
                continue
            predictions = {}
            with isolated_randomness():
                for name in ("TRUE", "AUX"):
                    model = probe["models"][name].to(self.config.device).eval()
                    predictions[name] = model(data).argmax(1).cpu()
                    model.cpu()
            truth = target.cpu()
            correct_t, correct_a = predictions["TRUE"] == truth, predictions["AUX"] == truth
            self.local_rows.append({"probe_id": probe["id"], "gate_step": probe["step"],
                                    "evaluation_step": step, "domain": domain, "samples": len(data),
                                    "true_correct": int(correct_t.sum()), "aux_correct": int(correct_a.sum()),
                                    "fixed": int((~correct_t & correct_a).sum()),
                                    "broken": int((correct_t & ~correct_a).sum()),
                                    "gate_gain": probe["gain"], "would_accept": probe["accepted"]})
            probe["batches"] += 1
            if probe["batches"] == PROBE["future_batches"]:
                self.active = [item for item in self.active if item is not probe]
        pseudo_count = self.source_anchors.num_elem() if self.source_anchors is not None else 0
        self.pseudo_hash.update(torch.tensor(self.pseudo_ids, dtype=torch.long).numpy().tobytes())
        if pseudo_count:
            self.pseudo_hash.update(self.source_anchors.target.cpu().long().numpy().tobytes())
        actual = sum(1 for row in self.update_rows if row["step"] == step)
        if actual != self.trace.quotas[step]:
            raise ValueError("Formal optimizer steps changed from reference")
        self.budget_rows.append({"step": step, "formal_updates": actual, "reference_updates": self.trace.quotas[step],
                                 "pseudo_count": pseudo_count, **self.compute})

    def run_stream(self, output_dir, progress=None):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = super().run_stream(output_dir, progress)
        torch.cuda.synchronize()
        self.trace.verify_feedback(self.oracle.audit())
        if self.step + 1 != self.trace.steps or self.main_updates != sum(self.trace.quotas):
            raise ValueError("Reference stream or update budget changed")
        output = Path(output_dir)
        for name, rows, fields in (
            ("gates.csv", self.gate_rows, ["step", "eligible", "gain"]),
            ("updates.csv", self.update_rows, ["step", "iteration", "true_coefficient", "pseudo_coefficient"]),
            ("budget.csv", self.budget_rows, ["step", "formal_updates", "reference_updates"]),
            ("local_future.csv", self.local_rows, ["probe_id", "evaluation_step", "true_correct", "aux_correct"]),
        ):
            save_rows(output / name, rows, fields)
        result.update({"method": self.group, "main_updates": self.main_updates,
                       "candidate_updates": self.candidate_updates, "gate_evaluations": self.gate_evaluations,
                       "gate_acceptances": self.gate_acceptances,
                       "gate_acceptance_pct": 100 * self.gate_acceptances / self.gate_evaluations if self.gate_evaluations else None,
                       "aux_update_pct": 100 * self.aux_updates / self.main_updates if self.main_updates else None,
                       "truth_coefficient_mean": self.true_coefficient_sum / self.main_updates if self.main_updates else None,
                       "pseudo_coefficient_mean": self.pseudo_coefficient_sum / self.main_updates if self.main_updates else None,
                       "returned_pct": 100 * result["returned"] / result["queried"] if result["queried"] else None,
                       "formal_draws_sha256": self.formal_hash.hexdigest(), "pseudo_pool_sha256": self.pseudo_hash.hexdigest(),
                       "stream_seconds": time.perf_counter() - started,
                       "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 2**20})
        from experiments.delayed_feedback.run import dump_json
        dump_json(output / "compute.json", self.compute)
        return result
