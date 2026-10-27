"""One fixed D100 experiment; no test-label parameter selection."""
METHOD = "Semantic-feedback"
ARMS = ("SP-ATTA", "CLIP-only", "Fixed-Fusion", "Feedback-Fusion")
CLASSES = ("dog", "elephant", "giraffe", "guitar", "horse", "house", "person")
CLIP_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
DEFAULTS = {"delay": 100, "batch_size": 32, "clip_model": "ViT-B/32",
            "prompt": "an image of a {}.", "initial_weight": 0.5,
            "fusion_lr": 0.1, "anchor_strength": 0.1,
            "fusion_steps_per_return_batch": 1, "classes": CLASSES}


def validate_classes(root):
    """ImageFolder's alphabetical class order must match text columns everywhere."""
    for domain in (root / "PACS").iterdir():
        if domain.is_dir():
            actual = tuple(sorted(p.name for p in domain.iterdir() if p.is_dir()))
            if actual != CLASSES:
                raise ValueError(f"Unexpected PACS class mapping in {domain}: {actual}")
