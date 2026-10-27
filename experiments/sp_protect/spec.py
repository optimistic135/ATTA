"""Fixed minimal two-arm PACS D100 study, without extra pretrained models."""
METHOD = "SP-Protect-study"
GROUPS = ("SP-ATTA", "SP-Protect")
DEFAULTS = {"delay": 100, "batch_size": 32, "confidence": 0.9,
            "rank_fraction": 0.1, "protect_beta": 1.0, "future_batches": 10,
            "target": "detached online probabilities before this batch's release and training",
            "selection": "top ceil(0.1 eligible) source confidence >=0.9 and flip agreement",
            "ties": "private seeded random tie break; no global RNG consumption"}
METRICS = ("Art_accuracy_pct", "Cartoon_accuracy_pct", "Sketch_accuracy_pct",
    "stream_accuracy_pct", "domain_macro_accuracy_pct", "before_feedback_accuracy_pct", "after_feedback_accuracy_pct",
    "queried", "returned", "returned_pct", "cross_domain_returned", "cross_domain_pct", "pending",
    "main_updates", "candidate_updates", "gate_acceptance_pct", "aux_update_pct",
    "protection_updates", "selected_observations", "selected_accuracy_before_pct", "selected_accuracy_after_pct",
    "selected_fixed_pct", "selected_broken_pct", "mean_protection_kl", "stream_seconds", "peak_cuda_memory_mb",
    "main_forward_calls", "main_forward_images", "main_backward_calls", "main_backward_images",
    "teacher_forward_calls", "teacher_forward_images")
