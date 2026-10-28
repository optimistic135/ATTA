"""One fixed online three-arm experiment, not an official TAST reproduction."""
METHOD = "Delayed-TAST-study"
GROUPS = ("TAST-Online", "TAST-Replay", "TAST-Repair")
DEFAULTS = {"delay": 100, "batch_size": 32, "ensemble": 5, "projection_divisor": 4,
    "support_per_source_class": 20, "neighbors": 4, "temperature": 10.0,
    "head_lr": 1e-3, "steps_per_batch": 1, "truth_batch_size": 16, "truth_coefficient": 1.0,
    "head_initialization": "Kaiming-normal shared weight and rank-one factors; zero ensemble bias",
    "support_policy": "common source-entropy bank union all returned query features; repair changes labels/votes only",
    "timing": "predict before inserting current images, releasing labels or updating heads",
    "quota": "one Adam step per stream batch including waiting; NOT the old SP per-batch quotas"}
METRICS = ("Art_accuracy_pct", "Cartoon_accuracy_pct", "Sketch_accuracy_pct", "stream_accuracy_pct",
    "domain_macro_accuracy_pct", "before_feedback_accuracy_pct", "after_feedback_accuracy_pct",
    "source_stream_accuracy_pct", "source_before_feedback_accuracy_pct", "waiting_gain_vs_source_pp",
    "queried", "returned", "returned_pct", "cross_domain_returned", "cross_domain_pct", "pending",
    "formal_updates", "replay_updates", "replay_draws", "replay_unique_labels", "repaired_labels",
    "repaired_wrong_source_labels", "repair_exposure_pct", "label_repair_target_gain_pp",
    "label_repair_fixed_pct", "label_repair_broken_pct", "mean_unsupervised_loss", "mean_supervised_loss",
    "encoder_forward_calls", "encoder_forward_images", "encoder_backward_calls", "head_forward_calls",
    "head_backward_calls", "stream_seconds", "peak_cuda_memory_mb")
