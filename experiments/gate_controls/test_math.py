"""Server-only gate transformation checks; no datasets or CUDA required."""

import unittest

import torch

from .algorithm import transform_gates


class GateTransformationTests(unittest.TestCase):
    def test_uniform_preserves_mean_and_has_no_sample_variation(self):
        gates = torch.tensor([0.4, 0.7, 1.0, 1.0])
        result = transform_gates(gates, "uniform_gate", torch.Generator().manual_seed(3))
        torch.testing.assert_close(result, torch.full_like(gates, gates.mean()))
        torch.testing.assert_close(result.sum(), gates.sum())

    def test_shuffle_preserves_multiset_without_touching_training_rng(self):
        gates = torch.linspace(0.3, 1.0, 32)
        state = torch.get_rng_state().clone()
        first = transform_gates(gates, "shuffled_gate", torch.Generator().manual_seed(19))
        second = transform_gates(gates, "shuffled_gate", torch.Generator().manual_seed(19))
        torch.testing.assert_close(first.sort().values, gates.sort().values)
        torch.testing.assert_close(first, second)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_constant_and_singleton_gates_are_unchanged(self):
        for gates in (torch.ones(8), torch.tensor([0.6])):
            for mode in ("uniform_gate", "shuffled_gate"):
                result = transform_gates(gates, mode, torch.Generator().manual_seed(4))
                torch.testing.assert_close(result, gates)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            transform_gates(torch.ones(3), "unknown", torch.Generator())


if __name__ == "__main__":
    unittest.main()
