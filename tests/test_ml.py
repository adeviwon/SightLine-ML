"""
SightLine ML — test suite.
Runs without the trained checkpoint (model-independent tests).
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from synthdata import make_dataset, CONDITIONS, render_document, degrade, to_patch, PRESCRIPTION_LINES
from model import DnCNNLite, PatchDataset


class TestSynthData(unittest.TestCase):
    def test_dataset_builds(self):
        ds = make_dataset(n_per_condition=2, seed=1)
        self.assertEqual(len(ds), 2 * len(CONDITIONS))
        self.assertEqual(len({d["condition"] for d in ds}), len(CONDITIONS))

    def test_degradation_deterministic(self):
        img = render_document(PRESCRIPTION_LINES, seed=42)
        a = degrade(img, seed=42, noise=25.0)
        b = degrade(img, seed=42, noise=25.0)
        self.assertEqual(np.array(a).tobytes(), np.array(b).tobytes())

    def test_degradation_changes_pixels(self):
        img = render_document(PRESCRIPTION_LINES, seed=42)
        deg = degrade(img, seed=42, noise=25.0)
        diff = np.abs(np.asarray(img, dtype=float) - np.asarray(deg, dtype=float)).mean()
        self.assertGreater(diff, 1.0)

    def test_patch_shape(self):
        img = render_document(PRESCRIPTION_LINES, seed=42)
        p = to_patch(img)
        self.assertEqual(p.shape, (64, 256))
        self.assertGreaterEqual(p.min(), 0.0)
        self.assertLessEqual(p.max(), 1.0)


class TestModel(unittest.TestCase):
    def test_forward_shape(self):
        m = DnCNNLite()
        x = torch.rand(2, 1, 64, 256) if (torch := __import__("torch")) else None
        y = m(x)
        self.assertEqual(tuple(y.shape), (2, 1, 64, 256))

    def test_param_count(self):
        m = DnCNNLite()
        n = sum(p.numel() for p in m.parameters())
        self.assertGreater(n, 100_000)
        self.assertLess(n, 1_000_000)

    def test_patch_dataset(self):
        torch = __import__("torch")
        ds = PatchDataset(make_dataset(n_per_condition=2, seed=1))
        self.assertGreater(len(ds), 0)
        deg, clean = ds[0]
        self.assertEqual(tuple(deg.shape), (1, 64, 256))
        self.assertTrue(bool((deg <= 1.0).all()) and bool((deg >= 0.0).all()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
