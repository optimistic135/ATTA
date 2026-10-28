"""Fixed D100 minimal three-arm experiment; no parameter search."""
METHOD = "SP-risk-rectification-study"
GROUPS = ("SP-ATTA", "SP-CurrentSoft", "SP-RiskRectify")
DEFAULTS = {"delay": 100, "batch_size": 32, "eta": 0.25,
    "density_steps": 50, "density_lr": 0.5, "density_l2": 0.01,
    "density_folds": 2, "weight_min": 0.2, "weight_max": 5.0,
    "min_memory": 16, "min_ess": 8.0, "probability_floor": 1e-4,
    "query_policy": "unchanged original active query trace; no exploration or unbiasedness claim",
    "current_pool": "all new images in current batch; frozen-source soft probabilities",
    "support_policy": "both auxiliary arms use the same min-memory and clipped-weight ESS gate",
    "density": "two-fold out-of-fold balanced logistic fit on normalized frozen-source features, CPU float64"}
METRICS = ("Art_accuracy_pct", "Cartoon_accuracy_pct", "Sketch_accuracy_pct", "stream_accuracy_pct",
    "domain_macro_accuracy_pct", "before_feedback_accuracy_pct", "after_feedback_accuracy_pct",
    "queried", "returned", "returned_pct", "cross_domain_returned", "cross_domain_pct", "pending",
    "main_updates", "candidate_updates", "gate_acceptance_pct", "aux_update_pct",
    "current_aux_updates", "rectification_updates", "eligible_batches", "density_updates",
    "mean_ess", "mean_ess_fraction", "mean_clip_fraction", "mean_density_holdout_accuracy",
    "mean_training_current_loss", "mean_training_residual", "mean_training_auxiliary",
    "proxy_risk_mae", "rectified_risk_mae", "current_target_accuracy_pct", "current_target_fixed_pct",
    "current_target_broken_pct", "stream_seconds", "peak_cuda_memory_mb",
    "main_forward_calls", "main_forward_images", "main_backward_calls", "main_backward_images",
    "teacher_forward_calls", "teacher_forward_images")
