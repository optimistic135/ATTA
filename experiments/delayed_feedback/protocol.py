"""Label-release clock and audit records, independent of the learning framework."""

from collections import deque
from dataclasses import asdict, dataclass


@dataclass
class Feedback:
    sample_id: int
    anchor_index: int
    query_step: int
    query_domain: str
    due_step: int
    return_step: int = None
    return_domain: str = None


class DelayedOracle:
    """Keep truth private until the end of query_step + delay (zero based).

    One instance spans all target domains. The caller must predict before release.
    There is deliberately no end-of-stream flush.
    """

    def __init__(self, delay):
        if not isinstance(delay, int) or delay < 0:
            raise ValueError("delay must be a nonnegative integer")
        self.delay = delay
        self._pending = deque()
        self._seen = set()
        self.records = []

    def query(self, sample_id, anchor_index, label, step, domain):
        if sample_id in self._seen:
            raise ValueError(f"Sample {sample_id} was queried twice")
        self._seen.add(sample_id)
        record = Feedback(sample_id, anchor_index, step, domain, step + self.delay)
        self.records.append(record)
        self._pending.append((record, int(label)))

    def release(self, step, domain):
        released = []
        while self._pending and self._pending[0][0].due_step <= step:
            record, label = self._pending.popleft()
            record.return_step = step
            record.return_domain = domain
            released.append((record.anchor_index, label))
        return released

    def counts(self, query_domain=None):
        records = [r for r in self.records
                   if query_domain is None or r.query_domain == query_domain]
        returned = [r for r in records if r.return_step is not None]
        cross = sum(r.query_domain != r.return_domain for r in returned)
        return {
            "queried": len(records),
            "returned": len(returned),
            "cross_domain_returned": cross,
            "cross_domain_pct": 100.0 * cross / len(returned) if returned else None,
            "pending": len(records) - len(returned),
        }

    def audit(self):
        return [dict(asdict(r), cross_domain=(
            r.query_domain != r.return_domain if r.return_step is not None else None
        )) for r in self.records]
