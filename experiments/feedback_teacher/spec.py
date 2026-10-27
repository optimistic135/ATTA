"""Fixed defaults; no parameter sweep."""
from experiments.sp_atta.spec import METRICS as SP_METRICS

GROUPS = ("SP-ATTA", "Current-Source", "Current-Feedback")
DEFAULTS = {
    "delay": 100, "batch_size": 32, "buffer_batches": 4,
    "confidence": 0.9, "lambda_aux": 0.25, "gain_margin": 0.001,
    "teacher_lr": 0.001, "teacher_anchor": 0.01,
    "teacher_evidence_decay": 0.9,
    "teacher_budget": "one head step from each positive feedback-batch quota; both current controls train it",
    "teacher_use": "next batch only; positive pre-update returned-label CE evidence, otherwise source fallback",
}
METRICS = SP_METRICS + ("student_updates", "head_updates", "current_coverage_pct",
    "repair_use_pct", "source_query_accuracy_pct", "repair_query_accuracy_pct",
    "repair_minus_source_query_pp", "return_teacher_ce_gain", "teacher_evaluated_returns")
