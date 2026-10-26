"""Small fixed PACS D100 study; no search or extra datasets."""

GROUPS = ("Original-MIX", "TRUE-only", "Fixed-Aux", "SP-ATTA")
DIAGNOSTIC = "TRUE-scaled-diagnosis"
ARMS = ("SKIP", "TRUE", "TRUE-scaled", "PSEUDO", "MIX")
DEFAULTS = {"delay": 100, "batch_size": 32, "lambda_aux": 0.25,
            "gain_margin": 0.001, "candidate_steps": 1,
            "no_feedback_policy": "SP-ATTA uses pure truth when a fresh gate cannot be evaluated",
            "validation": "new returned labels excluded from candidate truth and pseudo image draws",
            "candidate_overhead": "same two shadow steps in all four groups, discarded before formal training"}
METRICS = ("Art_accuracy_pct", "Cartoon_accuracy_pct", "Sketch_accuracy_pct",
           "domain_macro_accuracy_pct", "stream_accuracy_pct", "queried", "returned", "returned_pct",
           "cross_domain_returned", "cross_domain_pct", "pending", "main_updates", "candidate_updates",
           "gate_evaluations", "gate_acceptances", "gate_acceptance_pct", "aux_update_pct",
           "truth_coefficient_mean", "pseudo_coefficient_mean", "stream_seconds", "peak_cuda_memory_mb")
