"""Scheduler checks with fake jobs; no CUDA, model or dataset is required."""

import threading
import unittest

from .run import parser
from .scheduler import run_jobs


class SchedulerTests(unittest.TestCase):
    def test_idle_gpu_takes_next_job_without_waiting_for_busy_gpu(self):
        release_first = threading.Event()
        lock = threading.Lock()
        active = set()
        assignments = []
        results = []
        coordinator = threading.get_ident()

        def launch(job, gpu):
            with lock:
                self.assertNotIn(gpu, active)
                active.add(gpu)
                assignments.append((job, gpu))
            try:
                if job == 0:
                    self.assertTrue(release_first.wait(timeout=10))
                return job, gpu
            finally:
                with lock:
                    active.remove(gpu)

        def save_result(result):
            self.assertEqual(threading.get_ident(), coordinator)
            results.append(result)
            if result[0] == 11:
                release_first.set()

        run_jobs(range(12), [0, 1], launch, save_result)
        self.assertEqual(sorted(job for job, _ in results), list(range(12)))
        self.assertEqual(dict(assignments)[0], 0)
        self.assertTrue(all(gpu == 1 for job, gpu in assignments if job != 0))

    def test_single_gpu_and_stop_on_failure(self):
        launched = []

        def launch(job, gpu):
            launched.append((job, gpu))
            if job == 1:
                raise RuntimeError("fake worker failure")
            return job

        results = []
        with self.assertRaisesRegex(RuntimeError, "fake worker failure"):
            run_jobs(range(4), [2], launch, results.append)
        self.assertEqual(launched, [(0, 2), (1, 2)])
        self.assertEqual(results, [0])

    def test_invalid_gpu_lists(self):
        for gpus in ([], [0, 0], [-1]):
            with self.subTest(gpus=gpus), self.assertRaises(ValueError):
                run_jobs([], gpus, lambda job, gpu: job, lambda result: None)

    def test_cli_supports_both_modes(self):
        self.assertEqual(parser().parse_args(["--gpus", "0", "1"]).gpus, [0, 1])
        self.assertEqual(parser().parse_args(["--gpu-idx", "1"]).gpu_idx, 1)
        self.assertIsNone(parser().parse_args([]).gpus)


if __name__ == "__main__":
    unittest.main()
