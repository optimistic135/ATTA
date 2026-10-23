"""Detached linear-head CE gradients, without per-sample autograd or extra forwards."""

import torch
from torch.nn import functional as F


@torch.no_grad()
def head_cosines(supervised, pseudo, bias=True):
    """Cosine of each pseudo CE head gradient with the mean supervised CE gradient.

    Each tuple contains the ACTUAL student forward's (features, logits, labels).
    For a linear head dCE/dW=(p-one_hot(y)) outer features; include bias if present.
    These gradients describe current raw CE, excluding optimizer momentum/weight decay.
    """
    sx, sz, sy = supervised
    px, pz, py = pseudo
    sx, sz, px, pz = (value.detach().float() for value in (sx, sz, px, pz))
    if not all(bool(torch.isfinite(value).all()) for value in (sx, sz, px, pz)):
        raise FloatingPointError("Non-finite head-gradient inputs")
    sr = sz.softmax(1) - F.one_hot(sy, num_classes=sz.shape[1]).to(sz.dtype)
    pr = pz.softmax(1) - F.one_hot(py, num_classes=pz.shape[1]).to(pz.dtype)
    gradient = sr.T @ sx / len(sx)
    dot = ((pr @ gradient) * px).sum(1)
    norm2 = gradient.square().sum()
    pseudo_norm2 = pr.square().sum(1) * px.square().sum(1)
    if bias:
        mean_residual = sr.mean(0)
        dot = dot + pr @ mean_residual
        norm2 = norm2 + mean_residual.square().sum()
        pseudo_norm2 = pseudo_norm2 + pr.square().sum(1)
    norms = (norm2 * pseudo_norm2).sqrt()
    valid = norms > 1e-12
    cosine = torch.where(valid, dot / norms.clamp_min(1e-12), torch.zeros_like(dot))
    if not bool(torch.isfinite(cosine).all()):
        raise FloatingPointError("Non-finite head-gradient cosine")
    return cosine.clamp(-1, 1), valid


@torch.no_grad()
def compatibility_gates(gates, cosine, valid):
    conflict = (-cosine).clamp(0, 1) * valid.to(cosine.dtype)
    return 1 - conflict * (1 - gates)


@torch.no_grad()
def uniform_relaxation(gates, target_removed_mean):
    """Match donor mean(1-g) when feasible, never strengthen the original gate."""
    available = float((1 - gates).mean())
    scale = min(1.0, target_removed_mean / available) if available > 0 else 0.0
    return 1 - scale * (1 - gates), scale
