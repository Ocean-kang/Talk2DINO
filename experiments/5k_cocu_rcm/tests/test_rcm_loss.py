import sys
from pathlib import Path
import unittest
import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from models.rcm_loss import multi_positive_infonce


class TestRCMLoss(unittest.TestCase):
    def test_finite_and_gradient(self):
        torch.manual_seed(7)
        text = torch.randn(3, 16, requires_grad=True)
        pos, neg = torch.randn(3, 2, 16), torch.randn(3, 4, 16)
        pm = torch.tensor([[1, 1], [1, 1], [1, 0]], dtype=torch.bool)
        nm = torch.ones(3, 4, dtype=torch.bool)
        loss, n, np_, nn_ = multi_positive_infonce(text, pos, neg, pm, nm, 0.07)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual((n, np_, nn_), (3, 5, 12))
        loss.backward()
        self.assertIsNotNone(text.grad)
        self.assertGreater(float(text.grad.norm()), 0.0)

    def test_empty_batch_returns_differentiable_zero(self):
        text = torch.randn(2, 8, requires_grad=True)
        pos, neg = torch.zeros(2, 2, 8), torch.zeros(2, 3, 8)
        pm, nm = torch.zeros(2, 2, dtype=torch.bool), torch.zeros(2, 3, dtype=torch.bool)
        loss, n, _, _ = multi_positive_infonce(text, pos, neg, pm, nm)
        self.assertEqual(n, 0)
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertIsNotNone(text.grad)


if __name__ == "__main__":
    unittest.main()
