"""Add one paired alpha*TRUE arm without editing the existing diagnosis."""

import copy
import json

import torch

from experiments.simatta_rootcause.algorithm import RootCauseSimATTA, restore_random, isolated_randomness, fingerprint
from experiments.simatta_rootcause.spec import DEFAULTS as D
from .spec import ARMS, DIAGNOSTIC


class ScaledTruthDiagnosis(RootCauseSimATTA):
    def _create_probe(self, step, domain, start, available, released, saved):
        super()._create_probe(step, domain, start, available, released, saved)
        probe = self.active[-1]
        ready = sorted(available)
        lookup = {index: position for position, index in enumerate(ready)}
        pool = {"truth_images": self.query_anchors.data[ready].cpu(),
                "truth_labels": torch.tensor([available[index] for index in ready], dtype=torch.long),
                "pseudo_images": self.source_anchors.data, "pseudo_labels": self.source_anchors.target.long()}
        draws = [(torch.tensor([lookup[index] for index in json.loads(row["truth_anchor_indices"])]),
                  torch.tensor(json.loads(row["pseudo_pool_indices"]), dtype=torch.long))
                 for row in self.draw_rows[-D["branch_steps"]:]]
        restore_random(saved)
        branch = copy.deepcopy(probe["models"]["SKIP"]).to(self.config.device)
        branch.train()
        optimizer = torch.optim.SGD(branch.parameters(), lr=self.config.atta.SimATTA.lr, momentum=0.9)
        with torch.enable_grad():
            for iteration, draw in enumerate(draws):
                true, _ = self._losses(branch, pool, draw, arm="TRUE")
                loss = probe["alpha"] * true
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite scaled-truth loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                self.train_rows.append({"probe_id": start, "step": step, "arm": "TRUE-scaled",
                                        "iteration": iteration, "alpha": probe["alpha"],
                                        "true_loss": float(true.detach()), "pseudo_loss": None,
                                        "objective_loss": float(loss.detach())})
        probe["history_after"]["TRUE-scaled"] = self._accuracy(branch, pool["truth_images"], pool["truth_labels"])
        optimizer.zero_grad(set_to_none=True)
        probe["models"]["TRUE-scaled"] = branch.cpu()
        for field in ("correct", "fixed", "broken"):
            probe[field]["TRUE-scaled"] = 0
        if fingerprint(self.model) != probe["initial_state_sha256"]:
            raise AssertionError("Scaled-truth diagnosis modified the original main model")

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
                    model = probe["models"][arm].to(self.config.device).eval()
                    predictions[arm] = model(data).argmax(1).cpu()
                    model.cpu()
            truth = target.cpu()
            skip = predictions["SKIP"] == truth
            probe["samples"] += len(data)
            probe["batches"] += 1
            for arm in ARMS:
                correct = predictions[arm] == truth
                fixed, broken = int((~skip & correct).sum()), int((skip & ~correct).sum())
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
        super()._finish(probe, complete)
        row = dict(self.probe_rows[-4])  # SKIP row, shared timing/context.
        size = probe["samples"]
        row.update({"arm": "TRUE-scaled", "future_correct": probe["correct"]["TRUE-scaled"],
                    "future_accuracy_pct": 100 * probe["correct"]["TRUE-scaled"] / size if size else None,
                    "gain_vs_skip_pp": 100 * (probe["correct"]["TRUE-scaled"] - probe["correct"]["SKIP"]) / size if size else None,
                    "fixed_vs_skip": probe["fixed"]["TRUE-scaled"], "broken_vs_skip": probe["broken"]["TRUE-scaled"],
                    "history_after_accuracy_pct": probe["history_after"]["TRUE-scaled"]})
        self.probe_rows.append(row)

    def run_stream(self, output_dir, progress=None):
        result = super().run_stream(output_dir, progress)
        size = sum(probe["samples"] for probe in self.completed)
        result["method"] = DIAGNOSTIC
        result["TRUE-scaled_future_accuracy_pct"] = (100 * sum(probe["correct"]["TRUE-scaled"] for probe in self.completed) / size
                                                     if size else None)
        result["TRUE_minus_TRUE-scaled_pp"] = (result["TRUE_future_accuracy_pct"] - result["TRUE-scaled_future_accuracy_pct"]
                                               if size else None)
        result["TRUE-scaled_minus_MIX_pp"] = (result["TRUE-scaled_future_accuracy_pct"] - result["MIX_future_accuracy_pct"]
                                              if size else None)
        return result
