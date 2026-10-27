"""Four online prediction arms on one unchanged SP training trajectory."""
from pathlib import Path
import time

import torch

from experiments.sp_atta.algorithm import SPATTA
from experiments.simatta_rootcause.algorithm import isolated_randomness
from experiments.fcra_relay.stream import save_rows
from experiments.delayed_feedback.run import dump_json
from .semantic import FrozenSemantic, FeedbackFusion
from .spec import ARMS, CLASSES, DEFAULTS as D, METHOD


class SemanticFeedback(SPATTA):
    def __init__(self, config, trace, checkpoint):
        super().__init__(config, "SP-ATTA", trace)
        # Model construction must not consume the formal trajectory's RNG.
        with isolated_randomness():
            self.semantic = FrozenSemantic(checkpoint, config.device)
            self.fusion = FeedbackFusion(config.device)
        self.prediction_rows, self.feedback_rows, self.weight_rows = [], [], []
        self.query_predictions = {}
        self.seen_feedback = False

    @torch.no_grad()
    def select_queries(self, data, stream_ids):
        output = super().select_queries(data, stream_ids)
        sp = output.float().softmax(-1)
        semantic = self.semantic.probabilities(data)
        fixed = (1 - D["initial_weight"]) * sp + D["initial_weight"] * semantic
        self.current = dict(zip(ARMS, (sp.cpu(), semantic.cpu(), fixed.cpu(), self.fusion(sp, semantic).cpu())))
        if self.fusion.updates == 0 and not torch.equal(self.current[ARMS[2]].argmax(-1), self.current[ARMS[3]].argmax(-1)):
            raise AssertionError("Untrained fusion must agree with fixed fusion")
        self.before_feedback = not self.seen_feedback
        for record in self.trace.queries[self.step]:
            sid = int(record["sample_id"])
            position = sid - int(stream_ids[0])
            self.query_predictions[int(record["anchor_index"])] = {
                "sample_id": sid, "query_step": self.step,
                "sp": sp[position].cpu(), "semantic": semantic[position].cpu(),
                "fused": self.current[ARMS[3]][position].clone()}
        return output  # Parent trajectory and progress always report factual SP.

    @torch.no_grad()
    def on_labels_released(self, step, domain, released):
        if released:
            # New truths are available here, but not yet in SP's supervised pool.
            indices = [index for index, _ in released]
            images = self.query_anchors.data[indices].to(self.config.device)
            labels = torch.tensor([label for _, label in released], device=self.config.device)
            self.model.eval()
            sp = self.model(images).float().softmax(-1)
            semantic = torch.stack([self.query_predictions[i]["semantic"] for i in indices]).to(sp.device)
            fused = self.fusion(sp, semantic)
            for position, (index, label) in enumerate(released):
                query = self.query_predictions.pop(index)
                row = {"step": step, "domain": domain, "sample_id": query["sample_id"],
                       "query_step": query["query_step"], "label": int(label),
                       "fusion_updates_before": self.fusion.updates}
                for name, probability in (("query_sp", query["sp"]), ("query_semantic", query["semantic"]),
                        ("query_fused", query["fused"]), ("return_sp", sp[position]),
                        ("return_fused", fused[position])):
                    row[name + "_prediction"] = int(probability.argmax())
                    row[name + "_true_probability"] = float(probability[label])
                self.feedback_rows.append(row)
            before = self.fusion.raw.sigmoid().cpu().tolist()
            nll, anchor = self.fusion.update(sp, semantic, labels)
            after = self.fusion.raw.sigmoid().detach().cpu().tolist()
            for name, old, new in zip(CLASSES, before, after):
                self.weight_rows.append({"step": step, "domain": domain, "class": name,
                    "returned": len(released), "weight_before": old, "weight_after": new,
                    "return_nll_before": nll, "anchor_loss_before": anchor})
            self.seen_feedback = True
        super().on_labels_released(step, domain, released)

    @torch.no_grad()
    def on_batch_end(self, step, domain, data, target, outputs, released):
        truth = target.cpu()
        correct = {name: probability.argmax(-1).eq(truth) for name, probability in self.current.items()}
        if not torch.equal(correct[ARMS[0]], outputs.argmax(-1).cpu().eq(truth)):
            raise AssertionError("Stored SP predictions differ from the factual trajectory")
        for name in ARMS:
            self.prediction_rows.append({"step": step, "domain": domain, "arm": name,
                "before_feedback": self.before_feedback, "samples": len(truth),
                "correct": int(correct[name].sum()),
                "fixed_vs_sp": int((correct[name] & ~correct[ARMS[0]]).sum()),
                "broken_vs_sp": int((~correct[name] & correct[ARMS[0]]).sum()),
                "true_probability_sum": float(self.current[name][torch.arange(len(truth)), truth].sum())})
        super().on_batch_end(step, domain, data, target, outputs, released)

    def run_stream(self, output_dir, progress=None):
        started = time.perf_counter()
        result = super().run_stream(output_dir, progress)
        output = Path(output_dir)
        if len(self.feedback_rows) != result["returned"] or len(self.query_predictions) != result["pending"]:
            raise AssertionError("Fusion feedback/cache lost alignment with the delayed oracle")
        save_rows(output / "arm_batches.csv", self.prediction_rows, ["step", "domain", "arm", "samples", "correct"])
        save_rows(output / "fusion_feedback.csv", self.feedback_rows, ["step", "sample_id", "label"])
        save_rows(output / "fusion_weights.csv", self.weight_rows, ["step", "class", "weight_after"])
        dump_json(output / "fusion_state.json", {"classes": CLASSES,
            "weights": self.fusion.raw.sigmoid().detach().cpu().tolist(), "updates": self.fusion.updates})
        result.update(method=METHOD, fusion_updates=self.fusion.updates,
            clip_image_calls=self.semantic.image_calls, clip_image_count=self.semantic.image_count,
            clip_text_calls=1, shared_stream_seconds=time.perf_counter() - started)
        return result
