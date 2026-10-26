"""Two-view mutual current graph with traceable, true-anchor provenance."""

import torch
from torch.nn import functional as F

from .spec import DEFAULTS as D


@torch.no_grad()
def reconstruct(weak, flipped, node_ids, anchor_features, anchor_ids, labels, classes):
    """Inputs contain current features and ONLY already-returned anchor truth.

    Propagating one column per physical anchor prevents cycles from turning a
    single labeled image into multiple independent pieces of class evidence.
    """
    size = len(weak)
    device = weak.device
    identity = torch.eye(size, device=device, dtype=torch.bool)
    similarity_w = weak @ weak.T
    similarity_f = flipped @ flipped.T
    similarity = (similarity_w + similarity_f) / 2
    positions = similarity.masked_fill(identity, -2).topk(
        min(D["graph_neighbors"], max(1, size - 1)), dim=1).indices
    directed = torch.zeros_like(identity)
    directed.scatter_(1, positions, True)
    edge = (directed & directed.T & ~identity
            & (similarity_w >= D["graph_similarity"])
            & (similarity_f >= D["graph_similarity"]))
    weights = torch.exp((similarity - 1) / D["kernel_temperature"]) * edge
    transition = weights / weights.sum(1, keepdim=True).clamp_min(1e-12)
    degree = edge.sum(1)
    semantic = F.normalize(weak + flipped, dim=1)
    target = weak.new_full((size, classes), 1 / classes)
    result = {"target": target, "valid": torch.zeros(size, dtype=torch.bool, device=device),
              "weight": weak.new_zeros(size), "support": weak.new_zeros(size),
              "mass": weak.new_zeros(size), "degree": degree,
              "agreement": weak.new_zeros(size), "confidence": weak.new_zeros(size),
              "semantic": semantic, "edges": int(edge.sum()), "class_anchors": degree.new_zeros(size)}
    if anchor_features is None or len(anchor_features) == 0:
        return result
    affinity = semantic @ anchor_features.T
    # A newly returned image can receive direct true replay, but it must not
    # seed ITS OWN reconstruction or inflate graph evidence with identity edges.
    affinity = affinity.masked_fill(node_ids[:, None] == anchor_ids[None, :], -2)
    values, nearest = affinity.topk(min(D["seed_neighbors"], len(anchor_features)), dim=1)
    seed_weight = torch.exp((values - 1) / D["kernel_temperature"])
    seed_weight *= values >= D["seed_similarity"]
    seed_weight /= seed_weight.sum(1, keepdim=True).clamp_min(1e-12)
    seed = weak.new_zeros((size, len(anchor_features)))
    seed.scatter_(1, nearest, seed_weight)
    provenance = seed.clone()
    sparse = transition.to_sparse().coalesce()
    for _ in range(D["diffusion_iterations"]):
        provenance = ((1 - D["diffusion"]) * seed
                      + D["diffusion"] * torch.sparse.mm(sparse, provenance))
    # Remove self provenance again: a label can otherwise return to its own
    # current node through a cycle even though the direct seed edge was removed.
    provenance.masked_fill_(node_ids[:, None] == anchor_ids[None, :], 0)
    mass = provenance.sum(1)
    normalized = provenance / mass[:, None].clamp_min(1e-12)
    target = normalized @ F.one_hot(labels, num_classes=classes).to(weak.dtype)
    top = target.topk(2, dim=1).values
    predicted = target.argmax(1)
    winning = normalized * (labels[None, :] == predicted[:, None])
    support = winning.sum(1).square() / winning.square().sum(1).clamp_min(1e-12)
    class_anchors = (winning >= D["minimum_anchor_contribution"]).sum(1)
    neighbor_class = F.one_hot(predicted, num_classes=classes).to(weak.dtype)
    # Unseeded neighbor components do not vote on agreement.
    neighbor_class *= (mass >= D["minimum_mass"])[:, None]
    agreement = (transition @ neighbor_class).gather(1, predicted[:, None])[:, 0]
    valid = ((degree >= D["minimum_degree"]) & (mass >= D["minimum_mass"])
             & (class_anchors >= D["minimum_class_anchors"])
             & (support >= D["minimum_effective_anchors"])
             & (top[:, 0] >= D["assignment_confidence"])
             & ((top[:, 0] - top[:, 1]) >= D["assignment_margin"])
             & (agreement >= D["neighbor_agreement"]))
    # Every CE target must be a normalized distribution, including invalid nodes.
    target = torch.where((mass > 0)[:, None], target, result["target"])
    result.update({"target": target, "valid": valid,
                   "weight": valid * top[:, 0] * agreement * mass.clamp(0, 1),
                   "mass": mass, "support": support, "agreement": agreement,
                   "confidence": top[:, 0], "class_anchors": class_anchors})
    if not bool(torch.isfinite(target).all()):
        raise FloatingPointError("Non-finite graph reconstruction")
    return result
