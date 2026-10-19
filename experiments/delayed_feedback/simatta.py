"""Delayed-feedback adapter reusing the official SimATTA implementation."""

import csv
from pathlib import Path

import numpy as np
import torch

from ATTA.kernel.algorithms.SimATTA import SimATTA
from .protocol import DelayedOracle
from .datasets import EXPERIMENTS


DOMAIN_NAMES = {0: "Art", 1: "Cartoon", 2: "Photo", 3: "Sketch"}


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class DelayedSimATTA(SimATTA):
    def __init__(self, config, delay):
        super().__init__(config)
        self.experiment = EXPERIMENTS[config.dataset.name]
        self.oracle = DelayedOracle(delay)
        # Selection memory contains stream IDs, never true target labels. Pending
        # queries still participate in clustering, but not in supervised updates.
        self.query_anchors = None
        self.received = {}

    def train_on_env(self, env_id, train_only_fc=True, train_or_load="load"):
        """Never fall back to source training or write into the checkpoint folder."""
        folder = Path(self.config.ckpt_dir)
        for module, name in ((self.encoder, "encoder"), (self.fc, "fc")):
            path = folder / f"{name}_{env_id}.pth"
            if not path.is_file():
                raise FileNotFoundError(f"Required source checkpoint is missing: {path}")
            state = torch.load(path, map_location=self.config.device)
            module.load_state_dict(state, strict=True)

    @torch.no_grad()
    def select_queries(self, data, stream_ids):
        """Use the original entropy filter, clustering and anchor-weight updates."""
        # Cap clusters only for tiny/empty candidate pools, which the original
        # implementation cannot fit. Ordinary batches use its requested count.
        self.model.eval()
        outputs = self.model(data)
        eligible = int((self.softmax_entropy(outputs) >=
                        self.config.atta.SimATTA.eh).sum().item())
        previous = self.query_anchors.num_elem() if self.query_anchors is not None else 0
        candidates = previous + eligible if self.target_cluster else len(data)
        if candidates == 0:
            return outputs
        clusters = max(1, min(int(self.n_clusters), candidates))
        outputs, _, self.query_anchors = self.sample_select(
            self.model, data, stream_ids, self.query_anchors, clusters, 1,
            ent_bound=self.config.atta.SimATTA.eh,
            incremental_cluster=self.target_cluster,
        )
        return outputs

    def labeled_anchors(self):
        if not self.received:
            return None
        indices = torch.tensor(sorted(self.received), dtype=torch.long)
        labels = torch.tensor([self.received[i] for i in indices.tolist()], dtype=torch.long)
        query = self.query_anchors
        return self.update_anchors(None, query.data[indices], labels,
                                   query.feats[indices], query.weight[indices])

    @torch.no_grad()
    def run_stream(self, output_dir, progress=None):
        output_dir = Path(output_dir)
        self.enable_bn(self.model)
        batch_rows, domain_rows = [], []
        global_step, sample_offset = 0, 0
        for env_id in self.config.dataset.test_envs[1:]:
            domain = self.experiment.domains[env_id]
            correct_total, sample_total = 0, 0
            total_batches = len(self.fast_loader[env_id])
            if progress is not None:
                progress({"event": "domain", "domain": domain, "total": total_batches})
            for domain_step, (data, target) in enumerate(self.fast_loader[env_id]):
                data = data.to(self.config.device)
                stream_ids = torch.arange(sample_offset, sample_offset + len(data))
                previous = (self.query_anchors.num_elem()
                            if self.query_anchors is not None else 0)
                # Selection receives IDs instead of ground truth. Its return value
                # contains predictions made before any feedback/update this step.
                outputs = self.select_queries(data, stream_ids)
                correct = int((outputs.argmax(1).cpu() == target.cpu()).sum().item())
                correct_total += correct
                sample_total += len(data)
                current = (self.query_anchors.num_elem()
                           if self.query_anchors is not None else 0)
                for index in range(previous, current):
                    sample_id = int(self.query_anchors.target[index].item())
                    self.oracle.query(sample_id, index,
                                      target[sample_id - sample_offset].item(),
                                      global_step, domain)
                self.budgets += current - previous

                # Pseudo-label branch is unchanged and never consumes target truth.
                if self.LE:
                    _, _, self.source_anchors = self.sample_select(
                        self.teacher, data, stream_ids, self.source_anchors,
                        self.source_n_clusters, 0, use_pseudo_label=True,
                        ent_bound=self.config.atta.SimATTA.el,
                        incremental_cluster=False,
                    )
                if not self.target_cluster:
                    self.n_clusters = 0
                self.source_n_clusters = 100
                self.n_clusters += self.nc_increase
                self.source_n_clusters += 1

                released = self.oracle.release(global_step, domain)
                self.received.update(released)
                self.anchors = self.labeled_anchors()
                # The original supervised objective needs at least one returned
                # label. Continue replay on every later step, as in SimATTA.
                if self.anchors is not None:
                    source = (self.source_anchors if self.source_anchors is not None
                              and self.source_anchors.num_elem() > 0 else self.anchors)
                    self.cluster_train(self.anchors, source)
                    # Pending samples are unlabeled inputs; refreshing their
                    # features uses no truth and preserves clustering geometry.
                    self.query_anchors = self.update_anchors_feats(self.query_anchors)
                batch_rows.append({
                    "seed": self.config.random_seed, "delay": self.oracle.delay,
                    "step": global_step, "domain": domain, "domain_step": domain_step,
                    "samples": len(data), "correct": correct,
                    "accuracy_pct": 100.0 * correct / len(data),
                    "new_queries": current - previous, "new_returns": len(released),
                    **self.oracle.counts(),
                })
                if progress is not None:
                    progress({
                        "event": "batch", "domain": domain, "total": total_batches,
                        "completed": domain_step + 1,
                        "accuracy_pct": 100.0 * correct_total / sample_total,
                        **self.oracle.counts(),
                    })
                else:
                    print(f"D={self.oracle.delay} seed={self.config.random_seed} "
                          f"step={global_step} {domain} acc={100 * correct / len(data):.2f}%",
                          flush=True)
                sample_offset += len(data)
                global_step += 1
            domain_rows.append({
                "seed": self.config.random_seed, "delay": self.oracle.delay,
                "domain": domain, "batches": len(self.fast_loader[env_id]),
                "samples": sample_total, "correct": correct_total,
                "accuracy_pct": 100.0 * correct_total / sample_total,
            })

        # Counts are attributed to the QUERY domain, after the entire stream.
        for row in domain_rows:
            row.update(self.oracle.counts(row["domain"]))
        total = sum(row["samples"] for row in domain_rows)
        correct = sum(row["correct"] for row in domain_rows)
        result = {
            "seed": self.config.random_seed, "delay": self.oracle.delay,
            **{f"{row['domain']}_accuracy_pct": row["accuracy_pct"] for row in domain_rows},
            "stream_accuracy_pct": 100.0 * correct / total,
            "domain_macro_accuracy_pct": float(np.mean([r["accuracy_pct"] for r in domain_rows])),
            **self.oracle.counts(),
        }
        write_csv(output_dir / "batches.csv", batch_rows)
        write_csv(output_dir / "domains.csv", domain_rows)
        audit = self.oracle.audit()
        if audit:
            write_csv(output_dir / "feedback.csv", audit)
        else:
            (output_dir / "feedback.csv").write_text(
                "sample_id,anchor_index,query_step,query_domain,due_step,"
                "return_step,return_domain,cross_domain\n", encoding="utf-8")
        return result
