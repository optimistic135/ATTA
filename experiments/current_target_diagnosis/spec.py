"""One fixed diagnosis on the existing PACS D100 trace."""
METHOD = "SP-current-target-diagnosis"
ARMS = ("TRUE", "SOURCE", "REPAIR")
DEFAULTS = {
    "delay": 100, "batch_size": 32, "buffer_batches": 4,
    "probe_starts": (100, 130, 170, 210, 240), "trigger_window": 10,
    "candidate_steps": 1, "future_batches": 10, "lambda_aux": 0.25,
    "confidence": 0.9, "gain_margin": 0.001,
    "repair_lr": 0.001, "repair_anchor": 0.01,
    "main": "unchanged SP-ATTA fixed-trace formal training",
    "head": "one diagnostic-only step at each positive-quota return batch, after formal training",
    "truth_draw": "same previous returned truth excluding fresh holdout, for all three arms",
    "current_draw": "same images and indices; each source has its own confidence/consistency mask",
    "repair_policy": "always probe repaired targets, independently of historical branch-selection evidence",
}
