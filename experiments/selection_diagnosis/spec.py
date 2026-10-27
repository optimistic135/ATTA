"""Fixed diagnostics; thresholds are inspected, never optimized on test truth."""
from experiments.selective_feedback.spec import DEFAULTS as SELECTIVE_DEFAULTS

METHOD = "SP-selection-usability-diagnosis"
DEFAULTS = {key: SELECTIVE_DEFAULTS[key] for key in (
    "delay", "batch_size", "buffer_batches", "confidence", "reliability_threshold", "correction_margin",
    "min_feedback", "min_reliability_updates", "head_lr", "head_anchor", "reliability_lr", "reliability_l2")}
DEFAULTS.update({"rank_fraction": 0.1, "score_bin_edges": (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1),
    "main": "unchanged full-budget SP; copied head and risk updates are shadow overhead",
    "query_phase": "before_learning if risk_updates_at_query=0, otherwise after_learning; after_warmup is a subset",
    "ranking": "offline top/bottom10% on warm base-eligible images; fractional boundary ties",
})
DOMAINS = ("ALL", "Art", "Cartoon", "Sketch")
SECTIONS = ("distribution", "funnel", "ranking", "query_brier", "score_bins", "warmup")
