"""Low-dimensional correctness scores; no target truth at inference time."""
import math
import torch
from torch.nn import functional as F


def descriptors(source, repair):
    """2C+8 bounded features: probabilities, confidences, margins, entropies, agreement, JS."""
    s_top, r_top = source.topk(2, dim=1).values, repair.topk(2, dim=1).values
    log_c = math.log(source.shape[1])
    def entropy(q):
        return -(q * q.clamp_min(1e-12).log()).sum(1) / log_c
    mean = (source + repair) / 2
    js = (0.5 * (source * (source.clamp_min(1e-12).log() - mean.clamp_min(1e-12).log())).sum(1)
          + 0.5 * (repair * (repair.clamp_min(1e-12).log() - mean.clamp_min(1e-12).log())).sum(1)) / math.log(2)
    extra = torch.stack((s_top[:, 0], r_top[:, 0], s_top[:, 0] - s_top[:, 1],
        r_top[:, 0] - r_top[:, 1], entropy(source), entropy(repair),
        (source.argmax(1) == repair.argmax(1)).float(), js), dim=1)
    return torch.cat((source, repair, extra), dim=1).detach()


def correctness_loss(model, features, labels, regularization):
    loss = F.binary_cross_entropy_with_logits(model(features), labels)
    # No class rebalancing: scores should reflect the observed query population.
    return loss + regularization * model.weight.square().mean()
