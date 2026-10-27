"""No network access, backbone training, target-label prompts or domain metadata."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .spec import CLASSES, DEFAULTS as D
from experiments.delayed_feedback.run import sha256


class FrozenSemantic:
    def __init__(self, checkpoint, device):
        try:
            import clip
        except ImportError as error:
            raise ImportError("Install the official OpenAI CLIP package; see this experiment's README") from error
        if not checkpoint.is_file() or not hasattr(clip, "load") or not hasattr(clip, "tokenize"):
            raise ValueError("Need a local official CLIP checkpoint and the OpenAI CLIP package")
        self.model, _ = clip.load(str(checkpoint), device=device, jit=False)
        from pathlib import Path
        self.package_sources = {p.name: sha256(p) for p in Path(clip.__file__).parent.glob("*.py")}
        self.model.eval().requires_grad_(False)
        if self.model.visual.input_resolution != 224:
            raise ValueError("This fixed experiment requires ViT-B/32 at 224 pixels")
        with torch.no_grad():
            tokens = clip.tokenize([D["prompt"].format(name) for name in CLASSES]).to(device)
            self.text = F.normalize(self.model.encode_text(tokens).float(), dim=-1)
        self.image_calls = self.image_count = 0

    @torch.no_grad()
    def probabilities(self, images):
        # Reuse identical stream pixels: invert ImageNet normalization, then
        # apply CLIP normalization. The stream has already resized to 224 square.
        if images.shape[1:] != (3, 224, 224):
            raise ValueError("Expected ImageNet-normalized RGB 224-square stream tensors")
        shape = (1, 3, 1, 1)
        mean = images.new_tensor([.485, .456, .406]).view(shape)
        std = images.new_tensor([.229, .224, .225]).view(shape)
        rgb = (images * std + mean).clamp(0, 1)
        clip_mean = images.new_tensor([.48145466, .4578275, .40821073]).view(shape)
        clip_std = images.new_tensor([.26862954, .26130258, .27577711]).view(shape)
        features = F.normalize(self.model.encode_image((rgb - clip_mean) / clip_std).float(), dim=-1)
        self.image_calls += 1
        self.image_count += len(images)
        logits = self.model.logit_scale.exp().float() * features @ self.text.T
        probabilities = logits.softmax(-1)
        if not bool(torch.isfinite(probabilities).all()):
            raise FloatingPointError("Non-finite semantic probabilities")
        return probabilities


class FeedbackFusion(nn.Module):
    def __init__(self, device):
        super().__init__()
        initial = math.log(D["initial_weight"] / (1 - D["initial_weight"]))
        self.raw = nn.Parameter(torch.full((len(CLASSES),), initial, device=device))
        self.optimizer = torch.optim.SGD([self.raw], lr=D["fusion_lr"], momentum=0)
        self.updates = 0

    def forward(self, sp, semantic):
        weight = self.raw.sigmoid()
        mixture = (1 - weight) * sp.detach() + weight * semantic.detach()
        return mixture / mixture.sum(-1, keepdim=True).clamp_min(1e-8)

    @torch.enable_grad()
    def update(self, sp, semantic, labels):
        self.optimizer.zero_grad(set_to_none=True)
        probability = self(sp, semantic).clamp_min(1e-8)
        nll = F.nll_loss(probability.log(), labels)
        anchor = (self.raw.sigmoid() - D["initial_weight"]).square().mean()
        loss = nll + D["anchor_strength"] * anchor
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite feedback-fusion loss")
        loss.backward()
        self.optimizer.step()
        if not bool(torch.isfinite(self.raw).all()):
            raise FloatingPointError("Non-finite fusion parameters")
        self.updates += 1
        return float(nll.detach()), float(anchor.detach())
