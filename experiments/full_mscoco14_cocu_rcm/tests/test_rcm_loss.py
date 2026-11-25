import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP))

from models.rcm_loss import multi_positive_infonce


class RCMLossTests(unittest.TestCase):
    def test_finite_and_gradient(self):
        text = torch.randn(4, 768, requires_grad=True)
        pos = torch.randn(4, 2, 768)
        neg = torch.randn(4, 5, 768)
        pm = torch.tensor([[1,1],[1,0],[1,1],[0,0]], dtype=torch.bool)
        nm = torch.ones(4, 5, dtype=torch.bool)
        loss, anchors, npos, nneg = multi_positive_infonce(text, pos, neg, pm, nm)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(anchors, 3)
        self.assertEqual(npos, 5)
        self.assertEqual(nneg, 15)
        loss.backward()
        self.assertIsNotNone(text.grad)
        self.assertTrue(torch.isfinite(text.grad).all())

    def test_empty_batch_has_differentiable_zero(self):
        text = torch.randn(2, 768, requires_grad=True)
        pos = torch.randn(2, 2, 768)
        neg = torch.randn(2, 4, 768)
        pm = torch.zeros(2, 2, dtype=torch.bool)
        nm = torch.ones(2, 4, dtype=torch.bool)
        loss, anchors, _, _ = multi_positive_infonce(text, pos, neg, pm, nm)
        self.assertEqual(anchors, 0)
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(text.grad)


if __name__ == "__main__":
    unittest.main()
