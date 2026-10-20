"""Recorded-query replay checks; no data/checkpoints or GPU are loaded."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from .scheduled_replay import ScheduledNeedReplay


class ScheduledReplayTests(unittest.TestCase):
    def algorithm(self, records):
        algorithm = ScheduledNeedReplay.__new__(ScheduledNeedReplay)
        algorithm.model = nn.Sequential(nn.Identity(), nn.Identity())
        algorithm.history, algorithm.query_info = [], {}
        algorithm.query_anchors = None
        algorithm.query_plan = {0: records}
        return algorithm

    def test_uses_recorded_order_and_never_calls_clustering(self):
        algorithm = self.algorithm([
            {"anchor_index": 0, "sample_id": 12},
            {"anchor_index": 1, "sample_id": 10},
        ])
        data = torch.tensor([[1., 0.], [0., 1.], [0.5, 0.5]])
        with patch.object(algorithm, "sample_select", side_effect=AssertionError("K-means called")):
            output = algorithm.select_queries(data, torch.tensor([10, 11, 12]))
        self.assertTrue(torch.equal(output, data))
        self.assertEqual(algorithm.query_anchors.target.tolist(), [12, 10])
        self.assertTrue(torch.equal(algorithm.query_info[0]["query_feat"], data[2]))
        self.assertTrue(torch.equal(algorithm.query_info[1]["query_prob"], data[0].softmax(0)))
        with patch.object(algorithm.model[0], "forward", side_effect=AssertionError("cache encoded")):
            self.assertIs(algorithm.update_anchors_feats(algorithm.query_anchors), algorithm.query_anchors)

    def test_rejects_ids_outside_original_batch(self):
        algorithm = self.algorithm([{"anchor_index": 0, "sample_id": 100}])
        with self.assertRaisesRegex(ValueError, "outside the current batch"):
            algorithm.select_queries(torch.zeros((3, 2)), torch.arange(3))

    def test_no_queries_still_records_step(self):
        algorithm = self.algorithm([])
        algorithm.select_queries(torch.zeros((3, 2)), torch.arange(3))
        self.assertIsNone(algorithm.query_anchors)
        self.assertEqual(algorithm.history, [{"step": 0, "ready": [], "source_count": 0}])


if __name__ == "__main__":
    unittest.main()
