"""Fixed standalone recurrent-PACS protocol and measured summary fields."""

STAGES = (("Art_1", 0), ("Cartoon", 1), ("Sketch", 3), ("Art_2", 0), ("Photo", 2))
DELAYS = (0, 20, 50, 100)
DEFAULTS = {
    "current_steps": 4, "replay_steps": 26, "ema": 0.99,
    "pseudo_confidence": 0.9, "sharpen_temperature": 0.5,
    "current_weight": 0.5, "source_weight": 0.05,
    "relay_neighbors": 8, "relay_similarity": 0.65, "relay_temperature": 0.07,
    "relay_confidence": 0.7, "relay_margin": 0.15, "relay_support": 1.5,
    "prediction_mix_cap": 0.5, "calibration_prior_correct": 4.0,
    "calibration_prior_wrong": 2.0, "calibration_neighbors": 16,
    "calibration_similarity": 0.7, "calibration_age": 200.0,
    "calibration_size": 512, "buffer_batches": 8,
    "gradient_clip": 1.0, "lr": 0.001, "momentum": 0.9,
    "trainable": "ResNet layer4 and final classifier; other backbone layers frozen",
    "new_consistency": "label relay overrides EMA targets without class-agreement gating",
}
METRICS = tuple(f"{stage}_accuracy_pct" for stage, _ in STAGES) + (
    "Art_accuracy_pct", "first_three_macro_accuracy_pct", "first_three_stream_accuracy_pct",
    "stream_accuracy_pct", "raw_stream_accuracy_pct", "domain_macro_accuracy_pct", "stage_macro_accuracy_pct",
    "before_feedback_accuracy_pct", "after_feedback_accuracy_pct",
    "queried", "returned", "returned_pct", "pending", "pending_pct",
    "cross_domain_returned", "cross_domain_pct", "cross_stage_returned", "cross_stage_pct",
    "positive_delay_returned", "positive_delay_returned_pct", "realized_delay_mean",
    "main_updates", "current_updates", "supervised_updates", "relay_used_samples",
    "relay_overrides", "calibrated_returns", "stream_seconds", "peak_cuda_memory_mb",
)
