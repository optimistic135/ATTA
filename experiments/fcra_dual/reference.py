"""Read-only reference protocol and integer-exact budget allocation."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from experiments.delayed_feedback.run import sha256


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def optional_int(value):
    return None if value in (None, "", "None") else int(value)


class ReferenceTrace:
    """Only IDs/times/costs are a protocol; reference accuracy never drives training."""

    def __init__(self, folder, delay, seed):
        self.folder = Path(folder)
        names = ("settings.json", "result.json", "batches.csv", "feedback.csv", "updates.csv")
        self.hashes = {name: sha256(self.folder / name) for name in names}
        self.settings = json.loads((self.folder / "settings.json").read_text())
        result = json.loads((self.folder / "result.json").read_text())
        for record in (self.settings, result):
            if (record.get("method") != "FCRA" or record["seed"] != seed
                    or record["delay"] != delay):
                raise ValueError(f"Wrong reference identity: {self.folder}")
        if self.settings["batch_size"] != 32 or "replay_mix" not in self.settings["fcra"]:
            raise ValueError("Reference must be the completed, full-memory FCRA with batch size 32")
        self.batches = read_csv(self.folder / "batches.csv")
        self.steps = len(self.batches)
        if not self.steps:
            raise ValueError("Empty reference stream")
        offset, domains = 0, []
        self.bounds = []
        for step, row in enumerate(self.batches):
            if (int(row["step"]) != step or int(row["seed"]) != seed
                    or int(row["delay"]) != delay or not 1 <= int(row["samples"]) <= 32):
                raise ValueError(f"Invalid reference batch at step {step}")
            if not domains or domains[-1] != row["domain"]:
                domains.append(row["domain"])
            size = int(row["samples"])
            self.bounds.append((offset, offset + size))
            offset += size
        if domains != ["Art", "Cartoon", "Sketch"]:
            raise ValueError("Reference stream must be Art -> Cartoon -> Sketch")
        self.records = read_csv(self.folder / "feedback.csv")
        self.queries = defaultdict(list)
        seen = set()
        previous_query = -1
        for index, row in enumerate(self.records):
            sample, step = int(row["sample_id"]), int(row["query_step"])
            if not 0 <= step < self.steps or step < previous_query:
                raise ValueError("Reference query steps must follow anchor insertion order")
            start, end = self.bounds[step]
            expected_return = step + delay if step + delay < self.steps else None
            if (int(row["anchor_index"]) != index or sample in seen or not start <= sample < end
                    or int(row["due_step"]) != step + delay
                    or optional_int(row["return_step"]) != expected_return
                    or row["query_domain"] != self.batches[step]["domain"]):
                raise ValueError(f"Invalid reference feedback event {index}")
            if expected_return is not None and row["return_domain"] != self.batches[expected_return]["domain"]:
                raise ValueError("Reference return domain differs from its stream clock")
            seen.add(sample)
            previous_query = step
            self.queries[step].append(row)
        for step, batch in enumerate(self.batches):
            if len(self.queries[step]) != int(batch["new_queries"]):
                raise ValueError("Feedback and batch query counts differ")
        due_counts = Counter(int(row["due_step"]) for row in self.records
                             if int(row["due_step"]) < self.steps)
        for step, batch in enumerate(self.batches):
            if due_counts[step] != int(batch["new_returns"]):
                raise ValueError("Feedback and batch return counts differ")
        self.first_return = min(due_counts, default=None)
        self.old_updates = [0] * self.steps
        # Stream the potentially large update file; only integer costs are retained.
        with (self.folder / "updates.csv").open(newline="", encoding="utf-8") as stream:
            last_step = -1
            for row in csv.DictReader(stream):
                step, iteration = int(row["step"]), int(row["iteration"])
                if (not 0 <= step < self.steps or step < last_step
                        or iteration != self.old_updates[step]
                        or self.first_return is None or step < self.first_return):
                    raise ValueError("Invalid reference update sequence")
                self.old_updates[step] += 1
                last_step = step
        self.total = sum(self.old_updates)
        self.fast_total = min(self.steps, self.total // 10) if delay > 0 else 0
        slow_total = self.total - self.fast_total
        self.slow, self.fast = [], []
        cumulative, allocated = 0, 0
        for step, count in enumerate(self.old_updates):
            cumulative += count
            next_allocated = cumulative * slow_total // self.total if self.total else 0
            self.slow.append(next_allocated - allocated)
            allocated = next_allocated
            # Ceil differences make a waiting-period slot available at step 0.
            self.fast.append(((step + 1) * self.fast_total + self.steps - 1) // self.steps
                             - (step * self.fast_total + self.steps - 1) // self.steps)
        returned = sum(optional_int(row["return_step"]) is not None for row in self.records)
        if len(self.records) != result["queried"] or returned != result["returned"]:
            raise ValueError("Reference result and feedback totals differ")

    def budget_rows(self):
        return [{"step": step, "reference_updates": old, "slow_quota": slow,
                 "fast_quota": fast, "queries": len(self.queries[step])}
                for step, (old, slow, fast) in enumerate(zip(self.old_updates, self.slow, self.fast))]

    def verify_feedback(self, rows):
        keys = ("sample_id", "anchor_index", "query_step", "query_domain", "due_step",
                "return_step", "return_domain")
        if len(rows) != len(self.records):
            raise ValueError("Query schedule changed")
        for actual, expected in zip(rows, self.records):
            for key in keys:
                value = actual[key]
                normalized = "" if value is None else str(value)
                if normalized != (expected[key] or ""):
                    raise ValueError(f"Feedback schedule differs at {key}: {actual['sample_id']}")
