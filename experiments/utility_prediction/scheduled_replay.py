"""Replay an existing factual query schedule without selection or feature caching."""

from collections import defaultdict

import torch

from .engine import UtilitySimATTA


class ScheduledNeedReplay(UtilitySimATTA):
    def __init__(self, config, delay, output_dir, feedback, steps=30, horizon=20,
                 event_observer=None):
        super().__init__(config, delay, output_dir, steps, horizon,
                         save_snapshots=False, event_observer=event_observer)
        if feedback["sample_id"].duplicated().any():
            raise ValueError("Original feedback schedule contains repeated sample IDs")
        ordered = feedback.sort_values("anchor_index")
        if ordered["anchor_index"].tolist() != list(range(len(ordered))):
            raise ValueError("Original query anchor indices are not contiguous")
        if not ordered["query_step"].is_monotonic_increasing:
            raise ValueError("Original query steps do not follow anchor insertion order")
        if not (ordered["due_step"] == ordered["query_step"] + delay).all():
            raise ValueError("Original feedback schedule has a different delay")
        self.query_plan = defaultdict(list)
        for record in ordered.to_dict("records"):
            self.query_plan[int(record["query_step"])].append(record)

    @torch.no_grad()
    def select_queries(self, data, stream_ids):
        self.model.eval()
        features = self.model[0](data)
        outputs = self.model[1](features)
        step = len(self.history)
        self.history.append({"step": step, "ready": [], "source_count": 0})
        self.current_data = data.detach()
        records = self.query_plan.get(step, [])
        if not records:
            return outputs
        previous = self.query_anchors.num_elem() if self.query_anchors is not None else 0
        if [int(r["anchor_index"]) for r in records] != list(range(previous, previous + len(records))):
            raise ValueError(f"Query memory differs from original schedule at step {step}")
        positions = [int(r["sample_id"]) - int(stream_ids[0]) for r in records]
        if any(position < 0 or position >= len(data) for position in positions):
            raise ValueError(f"Original query IDs fall outside the current batch at step {step}; "
                             "check seed, batch size and dataset order")
        indices = torch.tensor(positions, dtype=torch.long)
        query_features = features[indices.to(features.device)].cpu().detach()
        self.query_anchors = self.update_anchors(
            self.query_anchors, data[indices.to(data.device)].cpu().detach(),
            stream_ids[indices].cpu(), query_features,
            # Utility replay uses images/labels and uniform replay draws, never
            # clustering weights. No cluster selection is performed in this tool.
            torch.ones(len(records)),
        )
        for offset, position in enumerate(positions):
            self.query_info[previous + offset] = {
                "query_step": step,
                "query_prob": outputs[position].softmax(0).cpu(),
                "query_feat": query_features[offset].clone(),
            }
        return outputs

    def update_anchors_feats(self, anchors):
        # This cache is read by online clustering, which is not run here.
        # The training loss re-encodes raw images, and all return-time metrics
        # also compute fresh features. Avoid re-encoding the entire memory/batch.
        return anchors
