"""Content-correction diagnostic, with masks and probability multisets held fixed."""
from experiments.current_target_diagnosis.spec import DEFAULTS as BASE_DEFAULTS

METHOD = "SP-oracle-target-diagnosis"
BASE_ARMS = ("TRUE", "SOURCE", "REPAIR")
ARMS = BASE_ARMS + ("ORACLE_SOURCE", "ORACLE_REPAIR")
PAIRS = (("source_true", "SOURCE", "TRUE"),
         ("repair_true", "REPAIR", "TRUE"),
         ("repair_source", "REPAIR", "SOURCE"),
         ("oracle_source_true", "ORACLE_SOURCE", "TRUE"),
         ("oracle_repair_true", "ORACLE_REPAIR", "TRUE"),
         ("oracle_source_source", "ORACLE_SOURCE", "SOURCE"),
         ("oracle_repair_repair", "ORACLE_REPAIR", "REPAIR"))
DEFAULTS = {**BASE_DEFAULTS,
    "truth_draw": "same previous returned truth excluding fresh holdout, for all five arms",
    "oracle": "swap q[argmax(q)] and q[y_eval]; same images, mask, confidence, entropy and sampling",
    "oracle_scope": "diagnostic copies only; constructed after current scoring labels become available",
    "oracle_gate": "never used for acceptance, queries, formal training or source-head repair",
}
