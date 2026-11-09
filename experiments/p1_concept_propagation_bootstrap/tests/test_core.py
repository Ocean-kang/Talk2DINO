import unittest
import torch

from experiments.p1_concept_propagation_bootstrap.core import (
    pool_pseudo_head_features,
)


class TestP1PseudoPooling(unittest.TestCase):
    def test_shape_and_patch_weight_normalization(self):
        torch.manual_seed(11)
        patches = torch.randn(16, 8)
        propagated = torch.randn(3, 16, 4)
        pseudo, weights = pool_pseudo_head_features(
            patches, propagated, temperature=0.2
        )
        self.assertEqual(tuple(pseudo.shape), (4, 3, 8))
        self.assertEqual(tuple(weights.shape), (3, 16, 4))
        self.assertTrue(torch.allclose(
            weights.sum(dim=1), torch.ones(3, 4), atol=1e-6
        ))

    def test_invalid_temperature(self):
        with self.assertRaises(ValueError):
            pool_pseudo_head_features(
                torch.randn(4, 3), torch.randn(2, 4, 1), temperature=0.0
            )


if __name__ == "__main__":
    unittest.main()
