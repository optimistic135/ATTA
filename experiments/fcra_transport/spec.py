"""Standard-library-only experiment identity and summary schema."""

STAGES = (("Art_1", 0), ("Cartoon", 1), ("Sketch", 3), ("Art_2", 0), ("Photo", 2))
METRICS = tuple(f"{name}_accuracy_pct" for name, _ in STAGES) + (
    "Art_accuracy_pct", "stream_accuracy_pct", "stage_macro_accuracy_pct",
    "domain_macro_accuracy_pct", "before_feedback_accuracy_pct", "after_feedback_accuracy_pct",
    "queried", "returned", "cross_domain_returned", "cross_domain_pct",
    "cross_stage_returned", "cross_stage_pct", "pending", "main_updates", "audit_updates",
    "transport_updates", "audited_queries", "returned_audits", "pending_audits",
    "stream_seconds", "peak_cuda_memory_mb",
)
