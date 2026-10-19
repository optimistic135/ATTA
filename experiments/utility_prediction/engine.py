"""Delayed SimATTA trajectory and paired, fixed-schedule label ablations.

Only the factual trajectory selects queries. Counterfactuals replay that common
schedule, so U measures supervision value conditional on it, not a new policy.
"""

import copy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.delayed_feedback.simatta import DelayedSimATTA, write_csv


def cosine(a, b):
    return float(F.cosine_similarity(a.reshape(1, -1).float(),
                                     b.reshape(1, -1).float(), dim=1).item())


def head_gradient(probabilities, features, labels):
    """Analytical mean CE gradient of linear head (weights followed by bias)."""
    residual = probabilities - F.one_hot(labels, probabilities.shape[1])
    weights = residual.T @ features / len(labels)
    return torch.cat((weights.flatten(), residual.mean(0)))


def rng_state():
    return (torch.get_rng_state(), torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None, np.random.get_state())


def restore_rng(state):
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])
    np.random.set_state(state[2])


class UtilitySimATTA(DelayedSimATTA):
    """Reuse the existing delayed adapter, oracle, selection and source loader.

    Fixed update counts replace loss-dependent stopping for this experiment.
    Masking a label keeps replay draws and loss denominators identical in the
    paired branches. Each event is one distinct returned image/label.
    """

    def __init__(self, config, delay, output_dir, steps=30, horizon=20):
        super().__init__(config, delay)
        self.output_dir = Path(output_dir)
        self.snapshot_dir = self.output_dir / "snapshots"
        self.snapshot_dir.mkdir()
        self.steps, self.horizon = steps, horizon
        self.history, self.events = [], []
        self.query_info, self.seen_returns = {}, set()
        # FastDataLoader's sequential subset is the actual per-domain stream.
        self.stream = []
        for env in config.dataset.test_envs[1:]:
            indices = list(self.fast_loader[env].sampler)
            size = config.atta.batch_size
            self.stream.extend((env, indices[i:i + size])
                               for i in range(0, len(indices), size))

    @torch.no_grad()
    def select_queries(self, data, stream_ids):
        previous = self.query_anchors.num_elem() if self.query_anchors is not None else 0
        outputs = super().select_queries(data, stream_ids)
        step = len(self.history)
        self.history.append({"step": step, "ready": [], "source_count": 0})
        self.current_data = data.detach()
        query = self.query_anchors
        current = query.num_elem() if query is not None else 0
        for index in range(previous, current):
            position = int(query.target[index]) - int(stream_ids[0])
            p = outputs[position].softmax(0).cpu()
            feat = query.feats[index].cpu().clone()
            self.query_info[index] = {"query_step": step, "query_prob": p,
                                      "query_feat": feat}
        return outputs

    @torch.no_grad()
    def cluster_train(self, target_anchors, source_anchors):
        """Hook after oracle release and before the factual training update."""
        row = self.history[-1]
        row["ready"] = sorted(self.received)
        row["source_count"] = (self.source_anchors.num_elem()
                               if self.source_anchors is not None else 0)
        returned = sorted(set(self.received) - self.seen_returns)
        if returned:
            self.model.eval()
            batch_feat = self.model[0](self.current_data)
            batch_prob = self.model[1](batch_feat).softmax(1)
            pseudo = self.teacher(self.current_data).argmax(1)
            reference_grad = head_gradient(batch_prob, batch_feat, pseudo).cpu()
            for index in returned:
                x = self.query_anchors.data[index:index + 1].to(self.config.device)
                feat = self.model[0](x)
                p = self.model[1](feat).softmax(1).cpu()[0]
                info = self.query_info[index]
                old = info["query_prob"]
                mid = (old + p) / 2
                js = 0.5 * ((old * (old.clamp_min(1e-12).log()
                                   - mid.clamp_min(1e-12).log())).sum()
                            + (p * (p.clamp_min(1e-12).log()
                                    - mid.clamp_min(1e-12).log())).sum())
                label = torch.tensor([self.received[index]])
                grad = head_gradient(p[None], feat.cpu(), label)
                old_grad = head_gradient(old[None], info["query_feat"][None],
                                         old.argmax().reshape(1))
                current_pseudo_grad = head_gradient(p[None], feat.cpu(),
                                                     p.argmax().reshape(1))
                self.events.append({
                    "anchor_index": index, "sample_id": int(self.query_anchors.target[index]),
                    "delay": self.oracle.delay, "query_step": info["query_step"],
                    "return_step": row["step"],
                    "query_domain": self.experiment.domains[self.stream[info["query_step"]][0]],
                    "return_domain": self.experiment.domains[self.stream[row["step"]][0]],
                    "query_entropy": float(-(old * old.clamp_min(1e-12).log()).sum()),
                    "return_entropy": float(-(p * p.clamp_min(1e-12).log()).sum()),
                    "prediction_changed": int(old.argmax() != p.argmax()),
                    "prediction_js": float(js),
                    "feature_temporal_cosine": cosine(info["query_feat"], feat.cpu()),
                    "feature_batch_cosine": cosine(feat.cpu(), batch_feat.mean(0).cpu()),
                    "gradient_temporal_cosine": cosine(old_grad, current_pseudo_grad),
                    "gradient_reference_cosine": cosine(grad, reference_grad),
                })
            torch.save({key: value.detach().cpu() for key, value in self.model.state_dict().items()},
                       self.snapshot_dir / f"step_{row['step']}.pt")
        self.seen_returns.update(returned)
        saved_rng = rng_state()
        try:
            self.replay_update(self.model, row, self.bank(),
                               self.config.random_seed + row["step"])
        finally:
            restore_rng(saved_rng)

    def bank(self):
        return {"data": self.query_anchors.data,
                "labels": self.received,
                "source_data": self.source_anchors.data if self.source_anchors is not None else None,
                "source_labels": self.source_anchors.target if self.source_anchors is not None else None}

    @torch.enable_grad()
    def replay_update(self, model, row, bank, seed, skip=None, force=None):
        """Identical draws and coefficients; SKIP zeroes only this truth's loss.

        The first target minibatch at return always includes the event, so USE
        really applies that label even with a large replay memory. Other samples
        and all future draws are identical. No replacement label is given to SKIP.
        """
        ready = torch.tensor(row["ready"], dtype=torch.long)
        if not len(ready):
            return
        n_source = row["source_count"]
        alpha = len(ready) / (len(ready) + n_source) if n_source else 0.5
        if (n_source if n_source else len(ready)) < self.cold_start:
            alpha = min(0.2, alpha)
        generator = torch.Generator().manual_seed(seed)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.atta.SimATTA.lr,
                                    momentum=0.9)
        size = self.config.train.train_bs
        device = self.config.device
        for iteration in range(self.steps):
            target_ids = ready[torch.randperm(len(ready), generator=generator)[:size]]
            if iteration == 0 and force is not None:
                # Make a matched draw that includes the event exactly once.
                others = target_ids[target_ids != force][:max(0, len(target_ids) - 1)]
                target_ids = torch.cat((torch.tensor([force]), others))
            targets = torch.tensor([bank["labels"][i] for i in target_ids.tolist()], device=device)
            logits = model(bank["data"][target_ids].to(device))
            losses = F.cross_entropy(logits, targets, reduction="none")
            target_mask = (target_ids != skip).to(device) if skip is not None else torch.ones_like(losses)
            target_loss = (losses * target_mask).mean()
            if n_source:
                source_ids = torch.randperm(n_source, generator=generator)[:size]
                source_loss = F.cross_entropy(model(bank["source_data"][source_ids].to(device)),
                                               bank["source_labels"][source_ids].to(device))
            else:
                # The original adapter falls back to labeled replay as source.
                source_ids = ready[torch.randperm(len(ready), generator=generator)[:size]]
                source_labels = torch.tensor([bank["labels"][i] for i in source_ids.tolist()], device=device)
                source_losses = F.cross_entropy(model(bank["data"][source_ids].to(device)),
                                                source_labels, reduction="none")
                mask = (source_ids != skip).to(device) if skip is not None else torch.ones_like(source_losses)
                source_loss = (source_losses * mask).mean()
            optimizer.zero_grad(set_to_none=True)
            ((1 - alpha) * source_loss + alpha * target_loss).backward()
            optimizer.step()

    def run_factual(self):
        """Record one real delayed stream, without changing existing entrypoints."""
        result = super().run_stream(self.output_dir)
        write_csv(self.output_dir / "return_features.csv", self.events)
        return result

    @torch.no_grad()
    def evaluate_utilities(self):
        if not self.events:
            (self.output_dir / "utilities.csv").write_text(
                "delay,return_step,future_batches,complete_window,accuracy_use,accuracy_skip,utility\n",
                encoding="utf-8")
            return []
        bank = self.bank()
        rows = []
        # Cache at most one 20-batch input window. Models run one branch at a time.
        for step in sorted({event["return_step"] for event in self.events}):
            path = self.snapshot_dir / f"step_{step}.pt"
            state = torch.load(path, map_location="cpu")
            end = min(step + self.horizon + 1, len(self.history))
            future = []
            for future_step in range(step + 1, end):
                env, indices = self.stream[future_step]
                samples = [self.dataset[env][i] for i in indices]
                future.append((torch.stack([x for x, _ in samples]),
                               torch.tensor([int(y) for _, y in samples])))
            for event in (event for event in self.events if event["return_step"] == step):
                accuracies = {}
                saved_rng = rng_state()
                for branch in ("USE", "SKIP"):
                    restore_rng(saved_rng)
                    model = copy.deepcopy(self.model)
                    model.load_state_dict(state, strict=True)
                    skip = event["anchor_index"] if branch == "SKIP" else None
                    # This initialization update occurs AFTER the return batch's
                    # prediction; future accuracy starts at return_step + 1.
                    self.replay_update(model, self.history[step], bank,
                                       self.config.random_seed + step, skip,
                                       force=event["anchor_index"])
                    scores = []
                    for offset, (data, labels) in enumerate(future, 1):
                        model.eval()
                        prediction = model(data.to(self.config.device)).argmax(1).cpu()
                        scores.append(float((prediction == labels).float().mean()))
                        # Truth from this future batch is used only via the
                        # recorded oracle arrivals, after scoring that batch.
                        if offset < len(future):
                            self.replay_update(model, self.history[step + offset], bank,
                                               self.config.random_seed + step + offset, skip)
                    accuracies[branch] = float(np.mean(scores)) if scores else None
                    del model
                restore_rng(saved_rng)
                complete = len(future) == self.horizon
                use, skip_score = accuracies["USE"], accuracies["SKIP"]
                row = {**event, "future_batches": len(future), "complete_window": complete,
                       "accuracy_use": use, "accuracy_skip": skip_score,
                       "utility": use - skip_score if use is not None else None}
                rows.append(row)
                print(f"utility D={self.oracle.delay} step={step} "
                      f"sample={event['sample_id']} U={row['utility']}", flush=True)
            path.unlink()  # Disposable snapshots; keep auditable event results.
            write_csv(self.output_dir / "utilities.csv", rows)
        return rows
