"""Fixed diagnosis, not an algorithm/parameter search."""

METHOD = "SimATTA-D100-diagnosis"
ARMS = ("SKIP", "TRUE", "PSEUDO", "MIX")
DEFAULTS = {
    "delay": 100, "batch_size": 32, "probe_starts": (100, 130, 160, 190, 220),
    "trigger_window": 10, "branch_steps": 10, "future_batches": 10,
    "truth_pool": "complete returned-label memory including this release",
    "pseudo_pool": "original frozen-teacher low-entropy pseudo-label memory",
    "optimizer": "SGD, original SimATTA lr, momentum 0.9; empty initial state as in each original batch",
    "evaluation": "freeze each branch after the short intervention; score next ten batches",
}
BASELINE_METRICS = (
    "Art_accuracy_pct", "Cartoon_accuracy_pct", "Sketch_accuracy_pct",
    "stream_accuracy_pct", "domain_macro_accuracy_pct", "queried", "returned",
    "cross_domain_returned", "cross_domain_pct", "pending", "returned_pct",
    "before_feedback_accuracy_pct", "after_feedback_accuracy_pct", "main_updates",
)
PROBE_METRICS = tuple(f"{arm}_future_accuracy_pct" for arm in ARMS) + (
    "TRUE_minus_SKIP_pp", "PSEUDO_minus_SKIP_pp", "MIX_minus_SKIP_pp", "TRUE_minus_MIX_pp",
    "gradient_cosine", "weighted_pseudo_to_true_norm_ratio", "complete_probes",
)
