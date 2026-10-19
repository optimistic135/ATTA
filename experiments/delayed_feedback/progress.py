"""Worker progress messages and a terminal display owned by the coordinator."""

import json
import sys
import threading


PREFIX = "@@ATTA_PROGRESS@@"


def emit_progress(event):
    print(PREFIX + json.dumps(event), flush=True)


def parse_progress(line):
    if not line.startswith(PREFIX):
        return None
    try:
        event = json.loads(line[len(PREFIX):])
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict) or event.get("event") not in ("domain", "batch"):
        return None
    return event


class ProgressDisplay:
    """One sweep bar plus one domain/batch bar per GPU; no worker writes to it."""

    def __init__(self, gpus, total):
        # Keep CLI help and the scheduler's existing standard-library tests light.
        from tqdm import tqdm

        self.lock = threading.RLock()
        self.interactive = sys.stderr.isatty()
        self.overall = tqdm(total=total, desc="Experiments", unit="run", position=0,
                            dynamic_ncols=True, disable=not self.interactive)
        self.bars = {
            gpu: tqdm(total=0, desc=f"GPU {gpu} waiting", unit="batch", position=index + 1,
                      dynamic_ncols=True, leave=True, disable=not self.interactive)
            for index, gpu in enumerate(gpus)
        }
        self.tasks = {}
        self.completed = 0
        self.total = total

    def start(self, gpu, delay, seed):
        with self.lock:
            self.tasks[gpu] = f"GPU {gpu} D={delay} seed={seed}"
            bar = self.bars[gpu]
            bar.reset(total=0)
            bar.set_postfix_str("")
            bar.set_description(f"{self.tasks[gpu]} loading")
            if not self.interactive:
                print(f"Starting {self.tasks[gpu]}", flush=True)

    def update(self, gpu, event):
        with self.lock:
            bar = self.bars[gpu]
            description = f"{self.tasks[gpu]} {event['domain']}"
            if event["event"] == "domain":
                bar.reset(total=event["total"])
                bar.set_postfix_str("")
                bar.set_description(description)
                if not self.interactive:
                    print(f"{description}: 0/{event['total']} batches", flush=True)
            else:
                bar.set_postfix_str(
                    f"acc={event['accuracy_pct']:.2f}% "
                    f"returned={event['returned']} pending={event['pending']}", refresh=False)
                bar.update(event["completed"] - bar.n)
                if not self.interactive:
                    print(f"{description}: {event['completed']}/{event['total']} batches "
                          f"acc={event['accuracy_pct']:.2f}% "
                          f"returned={event['returned']} pending={event['pending']}", flush=True)

    def finish(self, result):
        with self.lock:
            self.completed += 1
            self.overall.update(1)
            if not self.interactive:
                print(f"Completed {self.completed}/{self.total}: "
                      f"D={result['delay']}, seed={result['seed']}", flush=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with self.lock:
            for bar in reversed(list(self.bars.values())):
                bar.close()
            self.overall.close()
