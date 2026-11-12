import unittest
import torch
from experiments.Subset5K_Caption_Head_Comparison.run import batched_caption_head_scores


class IdentityProjector:
    def project_clip_txt(self, x):
        # Test-only deterministic mapping from 512-D text to 768-D visual space.
        return torch.nn.functional.pad(x, (0, 256))


class BatchMatchingTest(unittest.TestCase):
    def test_per_image_caption_head_similarity(self):
        torch.manual_seed(7)
        text = torch.randn(3, 5, 512)
        heads = torch.randn(3, 12, 768)
        result = batched_caption_head_scores(IdentityProjector(), text, heads, "cpu")
        self.assertEqual(tuple(result["similarities"].shape), (3, 5, 12))
        self.assertEqual(tuple(result["best_heads"].shape), (3, 5))
        self.assertTrue(torch.all(result["margins"] >= -1e-7))


if __name__ == "__main__":
    unittest.main()
