"""Deterministic cross-fitted density-ratio proxy, without label input or autograd."""
import torch
from .spec import DEFAULTS as D


@torch.no_grad()
def estimate_weights(memory, current):
    memory, current = memory.double().cpu(), current.double().cpu()
    if len(memory) < D["min_memory"] or len(current) < 4:
        return None
    # Alternating positions in anchor insertion order, not true classes/domains.
    mf = torch.arange(len(memory)) % 2
    cf = torch.arange(len(current)) % 2
    raw = torch.empty(len(memory), dtype=torch.float64)
    heldout_accuracy = []
    for fold in range(2):
        old, new = memory[mf != fold], current[cf != fold]
        coefficient = torch.zeros(memory.shape[1], dtype=torch.float64)
        intercept = torch.zeros((), dtype=torch.float64)
        # Analytic full-batch gradient of .5*mean BCE(old,0)+.5*mean BCE(new,1)+L2/2.
        for _ in range(D["density_steps"]):
            po = (old @ coefficient + intercept).sigmoid()
            pn = (new @ coefficient + intercept).sigmoid()
            gradient = .5 * (old.T @ po / len(old) + new.T @ (pn - 1) / len(new))
            gradient = gradient + D["density_l2"] * coefficient
            bias_gradient = .5 * (po.mean() + pn.mean() - 1)
            coefficient -= D["density_lr"] * gradient
            intercept -= D["density_lr"] * bias_gradient
        old_logits = memory[mf == fold] @ coefficient + intercept
        new_logits = current[cf == fold] @ coefficient + intercept
        # Equal class priors in the fit: p(current|z)/p(memory|z) = exp(log-odds).
        raw[mf == fold] = old_logits.clamp(-30, 30).exp()
        heldout_accuracy.append(.5 * (float((old_logits < 0).double().mean()) + float((new_logits >= 0).double().mean())))
    if not bool(torch.isfinite(raw).all()):
        raise FloatingPointError("Non-finite density ratio")
    clipped = raw.clamp(D["weight_min"], D["weight_max"])
    weights = clipped / clipped.mean()
    ess = float(weights.sum().square() / weights.square().sum())
    return {"weights": weights.float(), "raw": raw.float(),
        "ess": ess, "ess_fraction": ess / len(memory),
        "clip_fraction": float(((raw < D["weight_min"]) | (raw > D["weight_max"])).double().mean()),
        "holdout_accuracy": sum(heldout_accuracy) / len(heldout_accuracy),
        "eligible": ess >= D["min_ess"]}
