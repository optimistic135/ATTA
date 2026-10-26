"""Read only IDs, clock, source hashes and formal costs from a completed diagnosis."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from experiments.delayed_feedback.run import sha256


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def integer(value):
    return None if value in (None, "", "None") else int(value)


class SimATTATrace:
    def __init__(self, root, seed, checkpoint_dir):
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        if manifest.get("status") != "complete" or seed not in manifest["seeds"]:
            raise ValueError("Reference must be a completed original SimATTA root-cause run containing this seed")
        expected = {name: sha256(Path(checkpoint_dir) / name) for name in ("encoder_2.pth", "fc_2.pth")}
        if manifest.get("checkpoint_sha256") != expected:
            raise ValueError("Photo checkpoint hashes differ from reference")
        self.folder = self.root / "D100" / f"seed_{seed}"
        names = ("settings.json", "result.json", "batches.csv", "feedback.csv", "phases.csv")
        self.hashes = {name: sha256(self.folder / name) for name in names}
        settings = json.loads((self.folder / "settings.json").read_text())
        result = json.loads((self.folder / "result.json").read_text())
        if any(row.get("seed") != seed or row.get("delay") != 100 for row in (settings, result)):
            raise ValueError("Reference seed or delay mismatch")
        if (settings.get("method") != "SimATTA-D100-diagnosis"
                or result.get("method") != "SimATTA-D100-diagnosis" or settings["batch_size"] != 32):
            raise ValueError("Use the existing original SimATTA D100 diagnosis, not FCRA/Relay/CSR")
        self.simatta = settings["simatta"]
        self.batches = read_csv(self.folder / "batches.csv")
        phases = read_csv(self.folder / "phases.csv")
        if not self.batches or len(phases) != len(self.batches):
            raise ValueError("Reference phases/batches are empty or misaligned")
        self.steps = len(self.batches)
        self.quotas, self.bounds = [], []
        offset, domain_order = 0, []
        for step, (batch, phase) in enumerate(zip(self.batches, phases)):
            size = int(batch["samples"])
            if (int(batch["step"]) != step or int(phase["step"]) != step
                    or int(batch["seed"]) != seed or int(batch["delay"]) != 100
                    or phase["domain"] != batch["domain"] or int(phase["samples"]) != size
                    or not 1 <= size <= 32 or int(phase["main_updates"]) < 0):
                raise ValueError("Invalid reference batch/phase protocol")
            if not domain_order or domain_order[-1] != batch["domain"]:
                domain_order.append(batch["domain"])
            self.bounds.append((offset, offset + size))
            offset += size
            self.quotas.append(int(phase["main_updates"]))
        if domain_order != ["Art", "Cartoon", "Sketch"] or sum(self.quotas) != result["main_updates"]:
            raise ValueError("Reference domains or total formal update count mismatch")
        self.records = read_csv(self.folder / "feedback.csv")
        self.queries = defaultdict(list)
        seen, previous = set(), -1
        for index, row in enumerate(self.records):
            sid, step = int(row["sample_id"]), int(row["query_step"])
            if not 0 <= step < self.steps or step < previous:
                raise ValueError("Invalid query clock")
            lo, hi = self.bounds[step]
            due = step + 100
            returned = due if due < self.steps else None
            if (int(row["anchor_index"]) != index or sid in seen or not lo <= sid < hi
                    or int(row["due_step"]) != due or integer(row["return_step"]) != returned
                    or row["query_domain"] != self.batches[step]["domain"]):
                raise ValueError("Reference feedback identity or due clock mismatch")
            if returned is not None and row["return_domain"] != self.batches[returned]["domain"]:
                raise ValueError("Reference return domain mismatch")
            self.queries[step].append(row)
            seen.add(sid)
            previous = step
        returns = Counter(int(row["return_step"]) for row in self.records if integer(row["return_step"]) is not None)
        accumulated = 0
        for step, batch in enumerate(self.batches):
            accumulated += returns[step]
            if (len(self.queries[step]) != int(batch["new_queries"])
                    or returns[step] != int(batch["new_returns"])
                    or (accumulated == 0 and self.quotas[step] != 0)):
                raise ValueError("Reference query/return/update counts mismatch")
        if len(self.records) != result["queried"] or sum(returns.values()) != result["returned"]:
            raise ValueError("Reference final feedback counts mismatch")

    def verify_feedback(self, actual):
        keys = ("sample_id", "anchor_index", "query_step", "query_domain", "due_step", "return_step", "return_domain")
        if len(actual) != len(self.records):
            raise ValueError("Replay query count changed")
        for row, expected in zip(actual, self.records):
            for key in keys:
                left = "" if row[key] is None else str(row[key])
                right = "" if expected[key] in (None, "", "None") else str(expected[key])
                if left != right:
                    raise ValueError(f"Replay changed {key} for anchor {row['anchor_index']}")
