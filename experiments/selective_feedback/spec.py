"""One fixed three-arm experiment, no threshold search."""
from experiments.sp_atta.spec import METRICS as SP_METRICS

GROUPS = ("SP-ATTA", "SP-Select", "SP-Correct")
DEFAULTS = {"delay": 100, "batch_size": 32, "buffer_batches": 4,
    "confidence": 0.9, "reliability_threshold": 0.8, "correction_margin": 0.1,
    "min_feedback": 32, "min_reliability_updates": 8, "current_share": 0.25,
    "head_lr": 0.001, "head_anchor": 0.01, "reliability_lr": 0.05,
    "reliability_l2": 0.01, "lambda_aux": 0.25,
    "probe_starts": (100, 130, 170, 210, 240), "trigger_window": 10, "future_batches": 10,
    "budget": "positive-return batch: last two formal slots train source-copy head and risk model; quota1 risk only",
    "fallback": "original SP historical auxiliary when no reliable sampled current targets",
    "gate": "original SP gate only; never use evaluation truth or new risk training before gate",
}
METRICS = SP_METRICS + ("student_updates", "head_updates", "reliability_updates",
    "current_update_pct", "selected_coverage_pct", "selected_target_accuracy_pct",
    "selected_source_accuracy_pct", "target_net_correction_pp", "selected_repair_pct",
    "selected_training_draws", "selected_window_observations",
    "prequential_source_brier", "prequential_repair_brier", "local_selected_minus_true_pp",
    "prequential_source_confidence_brier", "prequential_repair_confidence_brier",
    "local_selected_minus_history_pp", "active_local_selected_minus_history_pp",
    "diagnostic_updates", "complete_selected_probes")
