"""Leave the factual SimATTA loop intact; fork a few isolated objective probes."""

from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from experiments.delayed_feedback.simatta import DelayedSimATTA
from experiments.fcra_relay.stream import save_rows
from .spec import ARMS, DEFAULTS as D, METHOD


def random_state():
    return (torch.get_rng_state(), torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            np.random.get_state(), random.getstate())


def restore_random(state):
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1])
    np.random.set_state(state[2])
    random.setstate(state[3])


@contextmanager
def isolated_randomness():
    saved = random_state()
    try:
        yield saved
    finally:
        restore_random(saved)


class RootCauseSimATTA(DelayedSimATTA):
    def __init__(self, config):
        super().__init__(config, D["delay"])
        self.generator = torch.Generator().manual_seed(config.random_seed + 919)
        self.active = []
        self.completed = []
        self.triggered = set()
        self.probe_rows, self.future_rows, self.train_rows, self.draw_rows = [], [], [], []
        self.phase_rows = []
        self.main_updates = 0
        self.last_update_count = 0
        self.first_return = None

    def one_step_train(self, *args, **kwargs):
        # Delegate EXACTLY to the official update. No fixed-step replacement.
        result = super().one_step_train(*args, **kwargs)
        self.main_updates += 1
        return result

    @torch.no_grad()
    def _accuracy(self, model, images, labels):
        model.eval()
        correct = 0
        for start in range(0, len(images), self.config.atta.batch_size):
            batch = images[start:start + self.config.atta.batch_size].to(self.config.device)
            logits = model(batch)
            correct += int((logits.argmax(1).cpu() == labels[start:start + len(batch)].cpu()).sum())
        return 100 * correct / len(labels)

    @torch.enable_grad()
    def _losses(self, model, pool, draw, arm="MIX"):
        true_indices, pseudo_indices = draw
        true = pseudo = None
        # Match the original MIX forward order: source pseudo first, truth next.
        if arm != "TRUE":
            pseudo = F.cross_entropy(model(pool["pseudo_images"][pseudo_indices].to(self.config.device)),
                                     pool["pseudo_labels"][pseudo_indices].to(self.config.device))
        if arm != "PSEUDO":
            true = F.cross_entropy(model(pool["truth_images"][true_indices].to(self.config.device)),
                                   pool["truth_labels"][true_indices].to(self.config.device))
        return true, pseudo

    @torch.enable_grad()
    def _gradients(self, snapshot, pool, draw, alpha):
        model = copy.deepcopy(snapshot).to(self.config.device)
        model.train()
        params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        true, pseudo = self._losses(model, pool, draw)
        gt = torch.autograd.grad(true, params, allow_unused=True)
        gp = torch.autograd.grad(pseudo, params, allow_unused=True)
        dot = true.new_zeros(())
        norm_t, norm_p = dot.clone(), dot.clone()
        for a, b in zip(gt, gp):
            if a is not None:
                norm_t += a.square().sum()
            if b is not None:
                norm_p += b.square().sum()
            if a is not None and b is not None:
                dot += (a * b).sum()
        norm_t, norm_p = float(norm_t.sqrt()), float(norm_p.sqrt())
        cosine = float(dot) / (norm_t * norm_p) if norm_t * norm_p > 1e-12 else None
        ratio = (1 - alpha) * norm_p / (alpha * norm_t) if alpha * norm_t > 1e-12 else None
        result = {"gradient_cosine": cosine, "true_gradient_norm": norm_t,
                  "pseudo_gradient_norm": norm_p, "weighted_pseudo_to_true_norm_ratio": ratio}
        del model, gt, gp, true, pseudo
        return result

    def on_labels_released(self, step, domain, released):
        if released and self.first_return is None:
            self.first_return = step
        for start in D["probe_starts"]:
            if start in self.triggered or not start <= step < start + D["trigger_window"]:
                continue
            pseudo = self.source_anchors
            if not released or pseudo is None or pseudo.num_elem() == 0:
                continue
            # Use all available truth for both TRUE and MIX, rather than changing
            # the truth pool while attributing a difference to pseudo supervision.
            available = dict(self.received)
            available.update(released)
            self.triggered.add(start)
            with isolated_randomness() as saved:
                self._create_probe(step, domain, start, available, released, saved)

    def _create_probe(self, step, domain, start, available, released, saved):
        ready = sorted(available)
        reference = self.source_anchors
        alpha = len(ready) / (len(ready) + reference.num_elem())
        if reference.num_elem() < self.cold_start:
            alpha = min(0.2, alpha)
        pool = {"truth_images": self.query_anchors.data[ready].cpu(),
                "truth_labels": torch.tensor([available[index] for index in ready], dtype=torch.long),
                "pseudo_images": reference.data,
                "pseudo_labels": reference.target.long()}
        size = self.config.train.train_bs
        draws = [(torch.randint(len(ready), (size,), generator=self.generator),
                  torch.randint(reference.num_elem(), (size,), generator=self.generator))
                 for _ in range(D["branch_steps"])]
        fresh = {index for index, _ in released}
        fresh_draws = 0
        for iteration, (truth, pseudo) in enumerate(draws):
            chosen = [ready[int(i)] for i in truth]
            fresh_draws += sum(index in fresh for index in chosen)
            self.draw_rows.append({"probe_id": start, "step": step, "iteration": iteration,
                                   "truth_anchor_indices": json.dumps(chosen),
                                   "pseudo_pool_indices": json.dumps(pseudo.tolist())})
        snapshot = copy.deepcopy(self.model).cpu()
        models = {"SKIP": snapshot}
        gradient = self._gradients(snapshot, pool, draws[0], alpha)
        snapshot.to(self.config.device)
        history_before = self._accuracy(snapshot, pool["truth_images"], pool["truth_labels"])
        snapshot.cpu()
        history_after = {"SKIP": history_before}
        for arm in ARMS[1:]:
            restore_random(saved)
            branch = copy.deepcopy(snapshot).to(self.config.device)
            branch.train()
            optimizer = torch.optim.SGD(branch.parameters(), lr=self.config.atta.SimATTA.lr, momentum=0.9)
            with torch.enable_grad():
                for iteration, draw in enumerate(draws):
                    true, pseudo = self._losses(branch, pool, draw, arm=arm)
                    loss = true if arm == "TRUE" else pseudo if arm == "PSEUDO" else alpha * true + (1 - alpha) * pseudo
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Non-finite root-cause probe loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    self.train_rows.append({"probe_id": start, "step": step, "arm": arm,
                                            "iteration": iteration, "alpha": alpha,
                                            "true_loss": float(true.detach()) if true is not None else None,
                                            "pseudo_loss": float(pseudo.detach()) if pseudo is not None else None,
                                            "objective_loss": float(loss.detach())})
            history_after[arm] = self._accuracy(branch, pool["truth_images"], pool["truth_labels"])
            models[arm] = branch.cpu()
            del optimizer, true, pseudo, loss
        same_domain = sum(self.oracle.records[index].query_domain == domain for index in ready)
        self.active.append({"id": start, "step": step, "domain": domain,
                            "models": models, "alpha": alpha, "truth_count": len(ready),
                            "fresh_count": len(released), "fresh_draws": fresh_draws,
                            "pseudo_count": reference.num_elem(), "history_before": history_before,
                            "history_after": history_after, "gradient": gradient,
                            "truth_same_domain_pct": 100 * same_domain / len(ready),
                            "batches": 0, "samples": 0,
                            "correct": {arm: 0 for arm in ARMS},
                            "fixed": {arm: 0 for arm in ARMS}, "broken": {arm: 0 for arm in ARMS}})

    @torch.no_grad()
    def on_batch_end(self, step, domain, data, target, outputs, released):
        is_before = self.first_return is None or step <= self.first_return
        self.phase_rows.append({"step": step, "domain": domain, "samples": len(data),
                                "correct": int((outputs.argmax(1).cpu() == target.cpu()).sum()),
                                "before_feedback": is_before,
                                "main_updates": self.main_updates - self.last_update_count})
        self.last_update_count = self.main_updates
        for probe in list(self.active):
            if step <= probe["step"]:
                continue
            predictions = {}
            with isolated_randomness():
                for arm in ARMS:
                    branch = probe["models"][arm].to(self.config.device).eval()
                    predictions[arm] = branch(data).argmax(1).cpu()
                    branch.cpu()
            truth = target.cpu()
            skip_correct = predictions["SKIP"] == truth
            probe["samples"] += len(data)
            probe["batches"] += 1
            for arm in ARMS:
                correct = predictions[arm] == truth
                fixed = int((~skip_correct & correct).sum())
                broken = int((skip_correct & ~correct).sum())
                probe["correct"][arm] += int(correct.sum())
                probe["fixed"][arm] += fixed
                probe["broken"][arm] += broken
                self.future_rows.append({"probe_id": probe["id"], "return_step": probe["step"],
                                         "evaluation_step": step, "evaluation_domain": domain,
                                         "arm": arm, "samples": len(data), "correct": int(correct.sum()),
                                         "fixed_vs_skip": fixed, "broken_vs_skip": broken})
            if probe["batches"] == D["future_batches"]:
                self._finish(probe, complete=True)

    def _finish(self, probe, complete):
        for arm in ARMS:
            samples = probe["samples"]
            self.probe_rows.append({
                "seed": self.config.random_seed, "probe_id": probe["id"], "return_step": probe["step"],
                "return_domain": probe["domain"], "arm": arm, "complete_window": complete,
                "future_batches": probe["batches"], "future_samples": samples,
                "future_correct": probe["correct"][arm],
                "future_accuracy_pct": 100 * probe["correct"][arm] / samples if samples else None,
                "gain_vs_skip_pp": 100 * (probe["correct"][arm] - probe["correct"]["SKIP"]) / samples if samples else None,
                "fixed_vs_skip": probe["fixed"][arm], "broken_vs_skip": probe["broken"][arm],
                "history_before_accuracy_pct": probe["history_before"],
                "history_after_accuracy_pct": probe["history_after"][arm],
                "alpha": probe["alpha"], "truth_count": probe["truth_count"],
                "fresh_count": probe["fresh_count"], "fresh_draws": probe["fresh_draws"],
                "pseudo_count": probe["pseudo_count"], "truth_same_domain_pct": probe["truth_same_domain_pct"],
                **probe["gradient"],
            })
        if complete:
            self.completed.append(probe)
        self.active = [item for item in self.active if item is not probe]
        # All metrics needed for summaries are above; do not retain snapshots.
        probe.pop("models")

    def run_stream(self, output_dir, progress=None):
        baseline = super().run_stream(output_dir, progress)
        for probe in list(self.active):
            self._finish(probe, complete=False)
        output = Path(output_dir)
        save_rows(output / "probes.csv", self.probe_rows, ["seed", "probe_id", "arm", "complete_window"])
        save_rows(output / "probe_batches.csv", self.future_rows, ["probe_id", "arm", "evaluation_step", "correct"])
        save_rows(output / "probe_training.csv", self.train_rows, ["probe_id", "arm", "iteration", "objective_loss"])
        save_rows(output / "probe_draws.csv", self.draw_rows, ["probe_id", "iteration", "truth_anchor_indices"])
        save_rows(output / "phases.csv", self.phase_rows, ["step", "samples", "correct", "before_feedback"])
        skipped = [{"probe_id": start, "reason": "no new return with nonempty pseudo pool in trigger window"}
                   for start in D["probe_starts"] if start not in self.triggered]
        save_rows(output / "skipped_probes.csv", skipped, ["probe_id", "reason"])
        result = {"method": METHOD, **baseline, "main_updates": self.main_updates,
                  "returned_pct": 100 * baseline["returned"] / baseline["queried"] if baseline["queried"] else None,
                  "complete_probes": len(self.completed), "triggered_probes": len(self.triggered)}
        for before, key in ((True, "before_feedback_accuracy_pct"), (False, "after_feedback_accuracy_pct")):
            group = [row for row in self.phase_rows if row["before_feedback"] == before]
            total = sum(row["samples"] for row in group)
            result[key] = 100 * sum(row["correct"] for row in group) / total if total else None
        total = sum(probe["samples"] for probe in self.completed)
        for arm in ARMS:
            result[f"{arm}_future_accuracy_pct"] = (100 * sum(probe["correct"][arm] for probe in self.completed) / total
                                                    if total else None)
        for arm in ARMS[1:]:
            result[f"{arm}_minus_SKIP_pp"] = (result[f"{arm}_future_accuracy_pct"] - result["SKIP_future_accuracy_pct"]
                                              if total else None)
        result["TRUE_minus_MIX_pp"] = (result["TRUE_future_accuracy_pct"] - result["MIX_future_accuracy_pct"]
                                       if total else None)
        for metric in ("gradient_cosine", "weighted_pseudo_to_true_norm_ratio"):
            values = [probe["gradient"][metric] for probe in self.completed
                      if probe["gradient"][metric] is not None and math.isfinite(probe["gradient"][metric])]
            result[metric] = sum(values) / len(values) if values else None
        return result
