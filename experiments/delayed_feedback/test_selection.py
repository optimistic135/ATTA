"""Regression for multiple cluster centres choosing the same query sample.

Run explicitly on the experiment machine; imports the original ATTA dependencies
but does not load a dataset, checkpoint or CUDA model.
"""

import importlib
import unittest
from unittest.mock import patch

from munch import Munch
import numpy as np
import torch
from torch import nn

from .protocol import DelayedOracle
from .simatta import DelayedSimATTA
from .run import DELAYS


class QuerySelectionTests(unittest.TestCase):
    def algorithm(self):
        # Bypass checkpoint/data initialization while exercising the actual
        # select_queries -> original sample_select -> update_anchors call chain.
        algorithm = DelayedSimATTA.__new__(DelayedSimATTA)
        algorithm.model = nn.Sequential(nn.Identity(), nn.Identity())
        algorithm.config = Munch(atta=Munch(
            SimATTA=Munch(eh=0.0), gpu_clustering=False))
        algorithm.query_anchors = None
        algorithm.target_cluster = True
        algorithm.n_clusters = 2
        algorithm.accumulate_weight = True
        return algorithm

    def select(self, algorithm, nearest):
        original = importlib.import_module("ATTA.kernel.algorithms.SimATTA")
        data = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        ids = torch.tensor([5262, 5263, 5264])
        with patch("sklearn.cluster.KMeans") as constructor, patch.object(
                original, "pairwise_distances_argmin_min",
                return_value=(np.array(nearest), np.zeros(2))):
            fitted = constructor.return_value
            fitted.fit.return_value = fitted
            fitted.labels_ = np.array([0, 1, 1])
            fitted.cluster_centers_ = np.zeros((2, 2))
            outputs = algorithm.select_queries(data, ids)
        self.assertTrue(torch.equal(outputs, data))

    def test_shared_nearest_image_is_queried_once_and_weights_are_summed(self):
        algorithm = self.algorithm()
        self.select(algorithm, [0, 0])
        anchors = algorithm.query_anchors
        self.assertEqual(anchors.target.tolist(), [5262])
        self.assertEqual(anchors.num_elem(), 1)
        self.assertEqual(anchors.weight.tolist(), [3])
        for delay in DELAYS:
            oracle = DelayedOracle(delay)
            for index, sample_id in enumerate(anchors.target.tolist()):
                oracle.query(sample_id, index, 4, 0, "Sketch")
            self.assertEqual(oracle.counts()["queried"], 1)
            if delay:
                self.assertEqual(oracle.release(delay - 1, "Sketch"), [])
            self.assertEqual(oracle.release(delay, "Sketch"), [(0, 4)])

    def test_distinct_ids_keep_selection_order_and_original_weights(self):
        algorithm = self.algorithm()
        self.select(algorithm, [2, 0])
        self.assertEqual(algorithm.query_anchors.target.tolist(), [5264, 5262])
        self.assertEqual(algorithm.query_anchors.weight.tolist(), [1, 2])

    def test_repeated_class_labels_in_training_memory_are_not_deduplicated(self):
        algorithm = self.algorithm()
        data = torch.zeros((3, 2))
        anchors = algorithm.update_anchors(None, data, torch.tensor([1, 1, 1]),
                                           data.clone(), torch.ones(3))
        self.assertEqual(anchors.num_elem(), 3)

    def test_selection_flag_is_cleared_after_an_exception(self):
        algorithm = self.algorithm()
        with patch.object(algorithm, "sample_select", side_effect=RuntimeError("selection failed")):
            with self.assertRaisesRegex(RuntimeError, "selection failed"):
                algorithm.select_queries(torch.zeros((3, 2)), torch.arange(3))
        self.assertFalse(algorithm._selecting_queries)


if __name__ == "__main__":
    unittest.main()
