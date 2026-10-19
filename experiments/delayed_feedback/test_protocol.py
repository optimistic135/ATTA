"""Protocol checks for execution on the experiment machine, not run automatically."""

import unittest

from .protocol import DelayedOracle


class DelayedOracleTests(unittest.TestCase):
    def test_zero_delay_returns_at_query_step(self):
        oracle = DelayedOracle(0)
        oracle.query(4, 0, 6, 0, "Art")
        self.assertEqual(oracle.release(0, "Art"), [(0, 6)])
        self.assertEqual(oracle.counts()["cross_domain_pct"], 0)
        self.assertEqual(oracle.release(0, "Art"), [])

    def test_boundary_and_pending_denominator(self):
        oracle = DelayedOracle(2)
        oracle.query(1, 0, 3, 0, "Art")
        self.assertEqual(oracle.release(1, "Art"), [])
        oracle.query(2, 1, 5, 2, "Cartoon")
        self.assertEqual(oracle.release(2, "Cartoon"), [(0, 3)])
        self.assertEqual(oracle.counts(), {
            "queried": 2, "returned": 1, "cross_domain_returned": 1,
            "cross_domain_pct": 100.0, "pending": 1,
        })
        self.assertIsNone(oracle.audit()[1]["cross_domain"])
        self.assertEqual(oracle.counts("Art")["returned"], 1)

    def test_no_returns_is_undefined(self):
        oracle = DelayedOracle(30)
        oracle.query(0, 0, 1, 0, "Sketch")
        self.assertEqual(oracle.release(0, "Sketch"), [])
        self.assertIsNone(oracle.counts()["cross_domain_pct"])

    def test_duplicate_and_invalid_delay(self):
        with self.assertRaises(ValueError):
            DelayedOracle(-1)
        oracle = DelayedOracle(1)
        oracle.query(0, 0, 2, 0, "Art")
        with self.assertRaises(ValueError):
            oracle.query(0, 1, 2, 1, "Art")


if __name__ == "__main__":
    unittest.main()
