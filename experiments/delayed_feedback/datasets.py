"""Dataset-specific defaults for the shared delayed-feedback protocol."""

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COMMON_METRICS = (
    "stream_accuracy_pct", "domain_macro_accuracy_pct", "queried", "returned",
    "cross_domain_returned", "cross_domain_pct", "pending",
)


@dataclass(frozen=True)
class Experiment:
    name: str
    domains: tuple
    environments: tuple
    layouts: tuple

    @property
    def source(self):
        return self.domains[self.environments[0]]

    @property
    def targets(self):
        return tuple(self.domains[i] for i in self.environments[1:])

    @property
    def checkpoint_names(self):
        source_id = self.environments[0]
        return (f"encoder_{source_id}.pth", f"fc_{source_id}.pth")

    @property
    def checkpoint_dir(self):
        return (ROOT / "storage/checkpoints/round1" / self.name / str(list(self.environments))
                / "ResNet18_3l_meanpool_0dp_Truefzbn/0.0001lr_0.0wd/ERM_no_param")

    @property
    def config_path(self):
        return ROOT / "configs/TTA_configs" / self.name / "SimATTA.yaml"

    @property
    def metrics(self):
        return tuple(f"{domain}_accuracy_pct" for domain in self.targets) + COMMON_METRICS


EXPERIMENTS = {
    "PACS": Experiment(
        "PACS", ("Art", "Cartoon", "Photo", "Sketch"), (2, 0, 1, 3),
        (("art_painting", "cartoon", "photo", "sketch"),),
    ),
    "VLCS": Experiment(
        "VLCS", ("Caltech101", "LabelMe", "SUN09", "VOC2007"), (0, 1, 2, 3),
        (("C", "L", "S", "V"), ("Caltech101", "LabelMe", "SUN09", "VOC2007")),
    ),
}
