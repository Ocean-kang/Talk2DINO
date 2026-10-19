import unittest
from pathlib import Path
import tempfile
import json
import sys
from unittest import mock

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import build_train as bt
import pipeline


class FakeCLIP:
    @staticmethod
    def tokenize(sentences, truncate=True):
        # deterministic vocab independent of available GPU/internet
        return torch.tensor([sum(bytearray(s.encode('utf-8'))) % 256 for s in sentences]).reshape(-1, 1)

    @staticmethod
    def load(model_name, device='cpu', jit=False):
        class Encoder:
            def eval(self):
                return self
            def encode_text(self, batch):
                vals = batch.flatten().float()
                return torch.stack([vals + 1, 257 - vals, torch.ones_like(vals)], dim=1)
        return Encoder(), None


def fake_dataset():
    images = [{'id': 1, 'disentangled_self_attn': torch.ones(12, 768)},
              {'id': 2, 'disentangled_self_attn': torch.ones(12, 768)}]
    annotations = []
    for iid in (1, 2):
        for j, name in enumerate(['a cat', 'a car', 'a person', 'a tree', 'a dog']):
            annotations.append({'id': 10 * iid + j, 'image_id': iid,
                                'caption': name, 'ann_feats': torch.tensor([j+1.0, 1.0, 0.1])})
    return {'images': images, 'annotations': annotations}


class TestPlan(unittest.TestCase):
    def test_novel_filters_duplicates_and_existing(self):
        captions = ['A cat on a mat', 'A dog nearby']
        concepts = [{'concept': 'cats', 'score': 0.99}, {'concept': 'street lamp', 'score': 0.8},
                    {'concept': 'street lamps', 'score': 0.7}]
        chosen = bt.choose_novel(concepts, captions, 3)
        self.assertEqual([x['concept'] for x in chosen], ['street lamp'])

    def test_farthest_uses_argmin_and_original_vectors(self):
        original = torch.tensor([[1., 0], [0., 1], [-1., 0], [0., -1], [1., 1.]])
        idx, sim = bt.farthest_caption(original, torch.tensor([1., 0.]))
        self.assertEqual(idx, 2)
        self.assertEqual(sim[2], -1.0)

    def test_build_five_plus_one_and_five(self):
        original = fake_dataset()
        with tempfile.TemporaryDirectory() as d:
            base, source = Path(d)/'base.pth', Path(d)/'concepts.json'
            torch.save(original, base)
            source.write_text(json.dumps({
                '1': {'image_id': 1, 'raw_captions': [], 'concepts': [
                    {'concept': 'cake', 'score': .8, 'best_head': 3},
                    {'concept': 'clock', 'score': .9, 'best_head': 4}]},
                '2': {'image_id': 2, 'concepts': []}
            }), encoding='utf-8')
            with mock.patch.dict(sys.modules, {'clip': FakeCLIP}):
                for mode, expected in [('5plus1', 11), ('5', 10)]:
                    target = Path(d)/f'{mode}.pth'
                    manifest = Path(d)/f'{mode}.json'
                    result = bt.build(base, source, target, manifest, mode, 3, 'ViT-B/16', 'cpu', 2, 'error')
                    loaded = torch.load(target, map_location='cpu', weights_only=False)
                    self.assertEqual(len(loaded['annotations']), expected)
                    self.assertEqual(result['reencoded_caption_count'], 1 if mode == '5plus1' else result['reencoded_caption_count'])
                    self.assertEqual(len(loaded['images']), 2)
                    audit = json.loads(manifest.read_text())['images']
                    self.assertEqual(len(audit['1']['selected_concepts']), 2)
                    self.assertEqual(len(audit['2']['augmented_captions']), 5)
                    if mode == '5':
                        self.assertEqual(len(audit['1']['augmented_captions']), 5)
                        self.assertEqual(len(audit['1']['assignments']), 2)
                        for rec in audit['1']['assignments']:
                            self.assertEqual(rec['min_similarity'], min(rec['similarities_to_raw_captions']))
                        self.assertTrue(any(x['caption'] != y['caption']
                                            for x,y in zip(loaded['annotations'][:5], original['annotations'][:5])))
                        self.assertTrue(all('ann_feats' in r for r in loaded['annotations']))

    def test_missing_data_fail_closed(self):
        data = fake_dataset()
        with self.assertRaisesRegex(ValueError, 'lack extracted concepts'):
            bt.plan_structure(data, {1: {'concepts': []}}, 3, 'error')
        images, annotations, plans, hist = bt.plan_structure(data, {1: {'concepts': []}}, 3, 'subset')
        self.assertEqual((len(images), len(annotations), len(plans)), (1, 5, 1))

    def test_eval_five_benchmarks_and_weight_naming(self):
        self.assertEqual(set(pipeline.BENCHMARKS), {'voc20', 'voc', 'coco_stuff', 'ade20k', 'context59'})
        with mock.patch.object(pipeline, 'ROOT', Path('/root/Talk2DINO')):
            args = pipeline.parser().parse_args(['--scheme','A','--mode','5plus1'])
            work, target, temporary, name = pipeline.paths(args)
            self.assertTrue(str(target).endswith('/weights/local_cocu_talk2dino_new_train/A_direct_retrieval/5plus1/' + name + '.pth'))
            self.assertEqual(name, 'vitb_mlp_infonce_local_cocu_new_A_5plus1')


if __name__ == '__main__':
    unittest.main()
