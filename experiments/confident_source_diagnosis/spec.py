"""A fixed D100 four-arm shadow diagnosis; no parameter sweep."""
METHOD = "SP-confident-source-diagnosis"
ARMS = ("TRUE", "HISTORY", "REDUCED", "CURRENT")
PAIRS = (("current_history", "CURRENT", "HISTORY"),
         ("current_reduced", "CURRENT", "REDUCED"),
         ("current_true", "CURRENT", "TRUE"),
         ("reduced_history", "REDUCED", "HISTORY"))
DEFAULTS = {"delay": 100, "batch_size": 32, "confidence": 0.9, "rank_fraction": 0.1,
    "lambda_aux": 0.25, "current_share": 0.25, "candidate_steps": 1, "future_batches": 10,
    "probe_starts": (100, 130, 170, 210, 240), "trigger_window": 10,
    "selection": "current batch only; confidence>=0.9 and flip agreement; ceil(0.1N), seeded uniform boundary ties",
    "current_loss": "KL on selected images, mean over selected K; no second random subsample or zero mask dilution",
    "main": "unchanged full-budget SP; all four candidates are discarded shadows",
    "truth": "common old returned truth draws; fresh release excluded from probe truth/history draws",
}
