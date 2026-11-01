"""Scientific invariants; standard-library matcher and synthetic vectors only."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from common.data import ImageBatches, PromptPoolDataset, reference_budget
from common.nouns import GalleryMatcher, compose_prompt, load_gallery
from common.selection import topk_gallery, caption_candidates, choose_concepts, exact_gallery_rank
from common.cache import validate_bank, text_digest


class Invariants(unittest.TestCase):
    def test_compounds_boundaries_case_and_punctuation(self):
        g = ['car', 'carpet', 'tennis racket', 'traffic light', 'racket']
        r = GalleryMatcher(g).extract(['A CARPET, a tennis-racket, and TRAFFIC LIGHT!', 'a car.'])
        self.assertEqual(set(r['concepts']), set(g))
        r = GalleryMatcher(g).extract(['a carpet'])
        self.assertEqual(r['concepts'], ['carpet'])
        r = GalleryMatcher(['traffic light', 'dog']).extract(['traffic', 'light dog'])
        self.assertEqual(r['concepts'], ['dog'])

    def test_exact_mode_does_not_chop_plural_suffixes(self):
        r = GalleryMatcher(['car', 'news', 'glass', 'bus', 'tennis racket']).extract(
            ['cars news glasses buses tennis rackets'])
        self.assertEqual(r['concepts'], ['news'])
        r = GalleryMatcher(['man', 'mouse', 'bus', 'tennis racket'], 'repo_forms').extract(
            ['men mice buses tennis rackets'])
        # Reuse repository forms unchanged: "man -> men" is not in its table.
        self.assertEqual(set(r['concepts']), {'mouse', 'bus', 'tennis racket'})

    def test_frequency_and_dedup_are_deterministic(self):
        r = GalleryMatcher(['dog', 'traffic light']).extract(['dog dog traffic light', 'dog'])
        self.assertEqual(r['concepts'], ['dog', 'traffic light'])
        self.assertEqual(r['caption_frequency']['dog'], 2)

    def test_existing_gallery_formats_and_order(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'g.json'
            for obj in (['DOG', 'traffic light', 'dog'], {'concept': ['DOG', 'traffic light', 'dog']},
                        [{'concept': 'DOG'}, {'concept': 'traffic light'}, {'concept': 'dog'}],
                        {'DOG': 1, 'traffic light': 2, 'dog': 3}):
                p.write_text(json.dumps(obj))
                g, aliases = load_gallery(p)
                self.assertEqual(g, ['dog', 'traffic light'])
                self.assertTrue(aliases)

    def test_composition_whole_concept_greedy_skip(self):
        text, keep, drop = compose_prompt(['traffic light', 'tennis racket', 'dog'], lambda s: len(s) < 36)
        self.assertEqual(keep, ['traffic light', 'dog'])
        self.assertEqual(drop, ['tennis racket'])
        self.assertEqual(text, 'a photo of traffic light, dog')

    def test_full_gallery_search_matches_dense_and_stable_ties(self):
        torch.manual_seed(42)
        q = F.normalize(torch.randn(12, 32), dim=-1)
        g = F.normalize(torch.randn(53, 32), dim=-1)
        expected, ids = (q @ g.T).topk(20)
        for chunk in (1, 7, 4096):
            actual, indices = topk_gallery(q, g, 20, chunk)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
            self.assertTrue(torch.equal(indices, ids))
        q, g = torch.tensor([[1., 0.]]), torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        _, ids = topk_gallery(q, g, 2, 1)
        self.assertEqual(ids.tolist(), [[0, 1]])
        self.assertEqual(exact_gallery_rank(q[0], g, 1, 1), 2)

    def test_caption_only_ranking_cannot_pass_as_gallery_retrieval(self):
        q = torch.tensor([[1., 0.], [0., 1.]])
        g = F.normalize(torch.tensor([[1., 0.], [.9, .1], [0., 1.], [.1, .9]]), dim=-1)
        vals, ids = topk_gallery(q, g, 2, 1)
        self.assertEqual(caption_candidates(q, g, vals, ids, set()), [])
        cs = caption_candidates(q, g, vals, ids, {1, 3})
        self.assertEqual({c['gallery_rank'] for c in cs}, {2})
        self.assertEqual({c['gallery_id'] for c in choose_concepts(cs, .5)}, {1, 3})
        self.assertEqual(choose_concepts(cs, 1.1), [])
        # A caption concept at full-gallery rank 3 is absent when K=2.
        g = F.normalize(torch.tensor([[1., 0.], [.99, .01], [.98, .02], [0., 1.]]), dim=-1)
        vals, ids = topk_gallery(q[:1], g, 2)
        self.assertEqual(caption_candidates(q[:1], g, vals, ids, {2}), [])
        self.assertEqual(exact_gallery_rank(q[0], g, 2), 3)

    def test_image_sampling_and_exact_previous_budget(self):
        budget = reference_budget(35, 3, 2)
        self.assertEqual(budget, 24)
        used, epoch = 0, 0
        while used < budget:
            batches = ImageBatches(7, 3, 123, epoch, budget - used)
            flat = [i for batch in batches for i in batch]
            self.assertEqual(len(flat), len(set(flat)))
            used += len(batches)
            epoch += 1
        self.assertEqual(used, budget)
        bank = {'texts': ['dog', 'ball'], 'embeddings': torch.eye(2)}
        ds = PromptPoolDataset([{'image_id': 1, 'prompt_pool': bank['texts']}],
            {1: {'disentangled_self_attn': torch.zeros(12, 768)}}, bank)
        vals = []
        for e in range(20):
            ds.set_epoch(e)
            vals.append(int(ds[0]['annotation'].argmax()))
        self.assertEqual(set(vals), {0, 1})
        with self.assertRaises(ValueError):
            PromptPoolDataset([{'image_id': 1, 'prompt_pool': []}], {}, bank)

    def test_cache_rejects_wrong_model_template_order_dimension(self):
        expected = {'clip_model': 'ViT-B/16', 'clip_checkpoint_sha256': 'verified'}
        g = ['dog', 'ball']
        ts = ['a photo of ' + c for c in g]
        meta = dict(expected, prompt_template='a photo of {}', feature_space='clip_text_raw',
                    gallery_concepts=g, texts_sha256=text_digest(ts))
        bank = {'texts': ts, 'embeddings': torch.ones(2, 512), 'metadata': meta}
        validate_bank(bank, expected, g)
        for key, val in [('clip_model', 'wrong'), ('prompt_template', '{}'),
                         ('gallery_concepts', list(reversed(g))), ('texts_sha256', 'wrong'),
                         ('feature_space', 'teacher_projected')]:
            with self.assertRaises(ValueError):
                validate_bank(dict(bank, metadata=dict(meta, **{key: val})), expected, g)
        with self.assertRaises(ValueError):
            validate_bank(dict(bank, embeddings=torch.ones(2, 768)), expected, g)

    def test_remote_network_guard_allows_local_only(self):
        import runpy
        m = runpy.run_path(str(HERE / 'offline_guard/sitecustomize.py'))
        self.assertTrue(m['_local'](('127.0.0.1', 1000)))
        self.assertTrue(m['_local'](('::1', 1000)))
        self.assertFalse(m['_local'](('example.com', 443)))
        with self.assertRaisesRegex(OSError, 'offline'):
            m['_connect'](None, ('8.8.8.8', 443))


if __name__ == '__main__':
    unittest.main()
