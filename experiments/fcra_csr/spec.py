"""Standalone settings; all graph choices are fixed before the experiment."""

STAGES = (("Art", 0), ("Cartoon", 1), ("Sketch", 3))
DELAYS = (0, 20, 50, 100)
DEFAULTS = {
    "buffer_batches": 4, "current_steps": 4, "replay_steps": 26,
    "ema": 0.99, "lr": 0.001, "momentum": 0.9, "gradient_clip": 1.0,
    "graph_neighbors": 8, "graph_similarity": 0.6,
    "seed_neighbors": 8, "seed_similarity": 0.55,
    "kernel_temperature": 0.07, "diffusion": 0.8, "diffusion_iterations": 20,
    "minimum_mass": 0.05, "minimum_degree": 2, "minimum_class_anchors": 2,
    "minimum_anchor_contribution": 0.05, "minimum_effective_anchors": 1.5,
    "assignment_confidence": 0.8, "assignment_margin": 0.2,
    "neighbor_agreement": 0.7, "reconstruction_weight": 0.5,
    "relation_weight": 0.1, "view_weight": 0.05, "relation_temperature": 0.2,
    "trainable": "all student parameters; existing PACS freeze_bn behavior retained",
    "inference": "student only; no neighbor fusion or future correction of scores",
    "pseudo_labels": "no entropy-sharpened class pseudo labels; truth-seeded graph only",
}
METRICS = tuple(f"{stage}_accuracy_pct" for stage, _ in STAGES) + (
    "first_three_macro_accuracy_pct", "first_three_stream_accuracy_pct",
    "stream_accuracy_pct", "domain_macro_accuracy_pct", "stage_macro_accuracy_pct",
    "before_feedback_accuracy_pct", "after_feedback_accuracy_pct",
    "queried", "returned", "returned_pct", "pending", "pending_pct",
    "cross_domain_returned", "cross_domain_pct", "cross_stage_returned", "cross_stage_pct",
    "positive_delay_returned", "positive_delay_returned_pct", "realized_delay_mean",
    "main_updates", "current_updates", "supervised_updates", "reconstructed_draws",
    "reconstruction_disagreements", "graph_eligible_nodes", "graph_total_nodes",
    "graph_coverage_pct", "anchor_reencode_images", "stream_seconds", "peak_cuda_memory_mb",
)
