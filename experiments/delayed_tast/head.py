"""Independent vectorized implementation of the TAST projection/prototype equations.

Algorithm references: Jang et al., ICLR 2023; mingukjang/TAST, adapt_algorithms.py.
No upstream module is imported, downloaded or executed.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .spec import DEFAULTS as D


class ProjectionEnsemble(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        count, hidden = D["ensemble"], max(1, dimension // D["projection_divisor"])
        self.shared = nn.Parameter(torch.empty(hidden, dimension))
        self.input_scale = nn.Parameter(torch.empty(count, dimension))
        self.output_scale = nn.Parameter(torch.empty(count, hidden))
        self.bias = nn.Parameter(torch.zeros(count, hidden))
        for parameter in (self.shared, self.input_scale, self.output_scale):
            nn.init.kaiming_normal_(parameter)
        self.forward_calls = self.backward_calls = 0

    def forward(self, features):
        self.forward_calls += 1
        output = torch.einsum("nd,ed,hd->enh", features, self.input_scale, self.shared)
        output = output * self.output_scale[:, None, :] + self.bias[:, None, :]
        if output.requires_grad and torch.is_grad_enabled():
            def count_backward(gradient):
                self.backward_calls += 1
                return gradient
            output.register_hook(count_backward)
        return output


def prototype_logits(head, features, supports, labels):
    projected = F.normalize(head(features), dim=-1)
    support_projection = head(supports)
    prototypes = torch.einsum("nc,enh->ech", labels, support_projection)
    prototypes = prototypes / labels.sum(0)[None, :, None].clamp_min(1e-12)
    prototypes = F.normalize(prototypes, dim=-1)
    return D["temperature"] * torch.einsum("ebh,ech->ebc", projected, prototypes)


@torch.no_grad()
def neighbor_targets(head, features, supports, labels, known_mask):
    similarity = F.normalize(features, dim=-1) @ F.normalize(supports, dim=-1).T
    nearest = similarity.topk(min(D["neighbors"], len(supports)), dim=-1).indices
    support_logits = prototype_logits(head, supports, supports, labels)
    probabilities = support_logits.softmax(-1)
    votes = F.one_hot(support_logits.argmax(-1), num_classes=labels.shape[1]).float()
    # A genuinely returned label is authoritative ONLY in the repair arm/copy.
    probabilities = torch.where(known_mask[None, :, None], labels[None, :, :], probabilities)
    votes = torch.where(known_mask[None, :, None], labels[None, :, :], votes)
    target = votes[:, nearest, :].mean(2)
    output = probabilities[:, nearest, :].mean(2).mean(0)
    exposure = known_mask[nearest].any(-1)
    if not bool(torch.isfinite(target).all() and torch.isfinite(output).all()):
        raise FloatingPointError("Non-finite neighbor targets")
    return target, output, exposure
