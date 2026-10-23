"""Server-side CPU checks; no datasets/checkpoints or CUDA required."""

import unittest

import torch
from torch.nn import functional as F

from .math import head_cosines, compatibility_gates, uniform_relaxation


class HeadGradientTests(unittest.TestCase):
    def test_analytic_cosines_match_autograd(self):
        sx = torch.tensor([[1., 2.], [-2., 1.], [0.5, -1.]])
        px = torch.tensor([[2., -1.], [-0.5, 3.]])
        sy, py = torch.tensor([0, 2, 1]), torch.tensor([1, 0])
        for use_bias in (False, True):
            weight = torch.tensor([[.2, -.1], [.4, .3], [-.2, .5]], requires_grad=True)
            bias = torch.tensor([.1, -.3, .2], requires_grad=True) if use_bias else None
            params = (weight, bias) if use_bias else (weight,)
            sz, pz = F.linear(sx, weight, bias), F.linear(px, weight, bias)
            gradient = torch.autograd.grad(F.cross_entropy(sz, sy), params, retain_graph=True)
            reference = torch.cat([value.flatten() for value in gradient])
            expected = []
            for i in range(len(px)):
                gradient = torch.autograd.grad(F.cross_entropy(pz[i:i+1], py[i:i+1]),
                                               params, retain_graph=True)
                current = torch.cat([value.flatten() for value in gradient])
                expected.append(F.cosine_similarity(reference, current, dim=0))
            actual, valid = head_cosines((sx, sz, sy), (px, pz, py), use_bias)
            torch.testing.assert_close(actual, torch.stack(expected), atol=2e-6, rtol=2e-6)
            self.assertTrue(bool(valid.all()))

    def test_no_gradient_and_nonfinite(self):
        batch = (torch.zeros(2, 3), torch.zeros(2, 2), torch.tensor([0, 1]))
        cosine, valid = head_cosines(batch, batch, bias=False)
        self.assertFalse(bool(valid.any()))
        torch.testing.assert_close(compatibility_gates(torch.tensor([.4, .7]), cosine, valid),
                                   torch.ones(2))
        broken = (torch.full((2, 3), float("nan")), batch[1], batch[2])
        with self.assertRaises(FloatingPointError):
            head_cosines(broken, batch)

    def test_gate_rule_and_strength_matching(self):
        gates = torch.tensor([.4, .6, .8])
        actual = compatibility_gates(gates, torch.tensor([1., -.5, -1.]),
                                    torch.ones(3, dtype=torch.bool))
        torch.testing.assert_close(actual, torch.tensor([1., .8, .8]))
        target = float((1 - actual).mean())
        control, scale = uniform_relaxation(gates, target)
        self.assertTrue(0 <= scale <= 1)
        self.assertAlmostEqual(float((1 - control).mean()), target, places=6)
        control, scale = uniform_relaxation(torch.ones(3), target)
        self.assertEqual(scale, 0)
        torch.testing.assert_close(control, torch.ones(3))
        control, scale = uniform_relaxation(gates, .9)
        self.assertEqual(scale, 1)
        torch.testing.assert_close(control, gates)


if __name__ == "__main__":
    unittest.main()
