import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.Caption_Head_Visualization.core import (
    attention_concentration, compute_matches, describe_matches, head_pool,
)
from experiments.Caption_Head_Visualization.data import resolve_image, select_images
from experiments.Caption_Head_Visualization.visualize import save_visualizations


class IdentityProjector:
    def project_clip_txt(self, x):
        return x


class TestCaptionHeadVisualization(unittest.TestCase):
    def test_head_pool_matches_original_divide_by_num_patches(self):
        patches = torch.tensor([[[2., 0.], [0., 4.]]])
        logits = torch.tensor([[[0., 0.], [5., -5.]]])
        pooled, attn = head_pool(patches, logits)
        expected = (patches.unsqueeze(1) * attn.unsqueeze(-1)).mean(dim=2)
        self.assertTrue(torch.allclose(pooled, expected))
        self.assertTrue(torch.allclose(attn.sum(-1), torch.ones((1, 2))))
        self.assertTrue(torch.allclose(pooled[0, 0], torch.tensor([.5, 1.])))

    def test_caption_to_head_and_json_fields(self):
        heads = torch.tensor([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.]])
        captions = torch.tensor([[1., 0., 0.], [0., 2., 0.], [.4, .8, 0.]])
        scores = compute_matches(IdentityProjector(), captions, heads)
        self.assertEqual(scores['similarities'].shape, (3, 3))
        self.assertEqual(scores['best_heads'].tolist(), [0, 1, 1])
        self.assertAlmostEqual(float(scores['top1_scores'][0]), 1.0)
        desc = describe_matches(scores, ['a', 'b', 'c'], [11, 12, 13], torch.ones(3, 4, 4) / 16)
        self.assertEqual(desc['head_use_counts'], [1, 2, 0])
        self.assertEqual(desc['unique_heads_used'], 2)
        self.assertEqual(desc['unused_head_numbers'], [3])
        self.assertEqual(desc['captions'][1]['head_number'], 2)

    def test_attention_concentration_for_uniform(self):
        a = torch.full((12, 32, 32), 1. / 1024.)
        np.testing.assert_allclose(attention_concentration(a), 1.0, atol=1e-6)

    def test_visual_output_files_and_raw_maps(self):
        with tempfile.TemporaryDirectory() as tmp:
            heads = torch.eye(12)
            text = torch.eye(12)[:5]
            scores = compute_matches(IdentityProjector(), text, heads)
            attention = torch.ones(12, 4, 4) / 16
            attention[0, 0, 0] = 1
            attention[0] /= attention[0].sum()
            d = describe_matches(scores, [f'Caption {i}' for i in range(5)], list(range(5)), attention)
            save_visualizations(Image.new('RGB', (56, 56), 'grey'), attention, scores, d, tmp, all_heads=True)
            for x in ['original_crop.png', 'overview.png', 'similarity_matrix.png', 'all_12_heads.png',
                      'caption_00_head_01.png', 'attention_maps.npz']:
                self.assertTrue((Path(tmp) / x).is_file(), x)
            with np.load(Path(tmp) / 'attention_maps.npz') as z:
                self.assertEqual(z['attention'].shape, (12, 4, 4))

    def test_image_resolver_coco_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'train2014').mkdir()
            p = root / 'train2014' / 'COCO_train2014_00000001.jpg'
            p.write_bytes(b'test')
            self.assertEqual(resolve_image(root, {'id': 1, 'file_name': p.name}), p)

    def test_explicit_ids(self):
        ims = [{'id': 10}, {'id': 20}, {'id': 30}]
        self.assertEqual([x['id'] for x in select_images(ims, ids=[30, 10])], [10, 30])
        self.assertEqual([x['id'] for x in select_images(ims, offset=1, limit=1)], [20])
        with self.assertRaises(ValueError):
            select_images(ims, ids=[40])


if __name__ == '__main__':
    unittest.main()
