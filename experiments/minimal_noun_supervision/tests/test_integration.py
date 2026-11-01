"""CPU pipeline smoke with upstream model/loss/train/validate.
CLIP is a deterministic double. No pretrained model or real COCO is used.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import torch
from common.io import sha256
from common.cache import model_metadata


class OfflinePipeline(unittest.TestCase):
    def test_generation_selection_caches_and_four_training_runs(self):
        fake_clip = types.ModuleType('clip')
        loads = []
        def tokenize(texts, truncate=False):
            if not truncate and any(len(t) > 100 for t in texts):
                raise RuntimeError('test context overflow')
            return torch.tensor([[int.from_bytes(hashlib.sha256(t.encode()).digest()[:4], 'little') % 10000] for t in texts])
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.text_projection = torch.ones(512, 512)
                self.visual = types.SimpleNamespace(conv1=types.SimpleNamespace(weight=torch.ones(1, 3, 16, 16)))
            def encode_text(self, tokens):
                return (tokens.float() + torch.arange(512).unsqueeze(0)).sin()
        def load(path, **kw):
            self.assertTrue(Path(path).is_file())
            loads.append(path)
            return Encoder(), None
        fake_clip.tokenize, fake_clip.load = tokenize, load
        def call(script, args):
            with patch.object(sys, 'argv', [str(HERE / script)] + list(map(str, args))):
                with contextlib.redirect_stdout(io.StringIO()):
                    runpy.run_path(str(HERE / script), run_name='__main__')
        with tempfile.TemporaryDirectory(prefix='mns_gallery_') as tmp, patch.dict(sys.modules, {'clip': fake_clip}):
            tmp = Path(tmp)
            torch.manual_seed(42)
            paths = []
            for split, ids in [('train', range(1, 9)), ('val', range(100, 104))]:
                images = [{'id': i, 'file_name': 'unused.jpg', 'disentangled_self_attn': torch.randn(12, 768),
                           'avg_self_attn_out': torch.randn(768)} for i in ids]
                anns = [{'id': i * 10 + j, 'image_id': i,
                         'caption': 'unmatched caption' if i == 8 else text, 'ann_feats': torch.randn(512)}
                    for i in ids for j, text in enumerate(['dog ball', 'person grass', 'dog', 'tennis racket', 'traffic light'])]
                source = tmp / (split + '.pth')
                torch.save({'images': images, 'annotations': anns}, source)
                paths.append(source)
            clip_p, dino_p = tmp / 'clip.pt', tmp / 'dino.pth'
            clip_p.write_bytes(b'local fixture CLIP')
            dino_p.write_bytes(b'local fixture DINO')
            g = tmp / 'gallery.json'
            g.write_text(json.dumps(['dog', 'ball', 'person', 'grass', 'tennis racket', 'traffic light', 'airplane']))
            meta = model_metadata(clip_p, dino_p)
            meta.update(train_features_sha256=sha256(paths[0]), val_features_sha256=sha256(paths[1]), source_model_attestation=True)
            fm = tmp / 'features.json'
            fm.write_text(json.dumps(meta))
            data = tmp / 'data'
            args = ['--train-features', paths[0], '--val-features', paths[1], '--gallery', g,
                '--clip-checkpoint', clip_p, '--dino-checkpoint', dino_p, '--feature-metadata', fm, '--device', 'cpu']
            call('prepare.py', args + ['--output', data])
            self.assertEqual(len(loads), 1)
            self.assertIn('airplane', json.loads((data / 'shared/gallery.json').read_text()))
            stats = json.loads((data / 'shared/statistics.json').read_text())
            self.assertEqual(stats['no_match_images'], 1)
            self.assertEqual(stats['no_match_ratio'], 1/8)
            # Full bank reuse -> no encoder invocation.
            call('prepare.py', args + ['--output', tmp / 'data_reuse', '--text-cache', data / 'shared/text_bank.pth'])
            self.assertEqual(len(loads), 1)
            from src.model import ProjectionLayer
            teacher = tmp / 'teacher.pth'
            torch.save(ProjectionLayer.from_config({'act': 'tanh', 'hidden_layer': True, 'dino_embed_dim': 768}).state_dict(), teacher)
            cfg = tmp / 'cfg.yaml'
            cfg.write_text('model:\n  act: tanh\n  hidden_layer: true\n  dino_embed_dim: 768\ntrain:\n  lr: 0.0001\n  ltype: infonce\n  num_epochs: 3\n  batch_size: 4\n  save_best_model: false\n')
            select_args = ['--data', data, '--teacher', teacher, '--model-config', cfg, '--device', 'cpu',
                           '--top-k', 2, '--gallery-chunk', 2, '--calibration-images', 4]
            call('select_hvne.py', select_args)
            for method in ('raw', 'cne', 'hvne', 'cnp'):
                m = json.loads((data / method / 'manifest.json').read_text())
                self.assertEqual([r['image_id'] for r in m['records']], list(range(1, 8)))
                self.assertEqual(m['raw_annotation_count'], 40)  # pre-exclusion budget stays fixed
                self.assertTrue(all(r['prompt_pool'] for r in m['records']))
                if method == 'cnp':
                    self.assertTrue(all(len(r['prompt_pool']) == 1 for r in m['records']))
            tag = 'synthetic_' + tmp.name.replace('-', '_')
            try:
                for method in ('raw', 'cne', 'hvne', 'cnp'):
                    original_load, accessed = torch.load, []
                    def tracked_load(path, *a, **kw):
                        accessed.append(str(path))
                        return original_load(path, *a, **kw)
                    with patch.object(torch, 'load', side_effect=tracked_load):
                        call('train.py', ['--method', method, '--data', data, '--config', cfg, '--seed', 11, '--tag', tag,
                                          '--device', 'cpu', '--workers', 0, '--max-steps', 5])
                    self.assertNotIn(str(teacher), accessed)
                    rp = ROOT / 'outputs/minimal_noun_supervision' / tag / method / 'seed_11/result.json'
                    result = json.loads(rp.read_text())
                    self.assertEqual(result['optimizer_updates'], 5)
                    self.assertEqual(result['selected'], 'last')
                    state = torch.load(result['checkpoint'], map_location='cpu')
                    model = ProjectionLayer.from_config({'act': 'tanh', 'hidden_layer': True, 'dino_embed_dim': 768})
                    model.load_state_dict(state, strict=True)
                    self.assertTrue(any(not torch.equal(state[k], torch.load(teacher)[k]) for k in state))
                shutil.rmtree(data / 'hvne')
                call('select_hvne.py', select_args + ['--min-sim', 2])
                stats = json.loads((data / 'hvne/statistics.json').read_text())
                self.assertEqual(stats['fallback_ratio'], 1.)
                self.assertEqual(stats['strict_images'], 0)
                for row in (data / 'hvne/selection_audit.jsonl').read_text().splitlines():
                    r = json.loads(row)
                    self.assertTrue(r['fallback'])
                    self.assertEqual(r['gallery_size_scanned_per_head'], 7)
                    self.assertTrue(all(not x['confidence_pass'] and x['gallery_rank'] is not None for x in r['selected']))
                # CNE/CNP pools share exactly the same pre-composition C_i.
                concepts = json.loads((data / 'shared/concepts_by_image.json').read_text())
                cne = json.loads((data / 'cne/manifest.json').read_text())
                comp = json.loads((data / 'cnp/composition_audit.json').read_text())
                for r, c in zip(cne['records'], comp):
                    cs = concepts[str(r['image_id'])]['concepts']
                    self.assertEqual(r['prompt_pool'], ['a photo of ' + x for x in cs])
                    self.assertEqual(c['input_concepts'], cs)
                self.assertEqual(len(loads), 1)  # no CLIP load during HVNE/training
                # No calibration hit must abstain rather than create empty pools.
                shutil.rmtree(data / 'hvne')
                with patch('common.selection.caption_candidates', return_value=[]):
                    call('select_hvne.py', select_args)
                zero = json.loads((data / 'hvne/statistics.json').read_text())
                self.assertEqual(zero['calibration_status'], 'no_calibration_hits_all_fallback')
                self.assertEqual(zero['fallback_ratio'], 1.)
                self.assertEqual(zero['strict_images'], 0)
                # Entirely unmatched gallery -> tracked exclusion and no train manifest.
                other_g = tmp / 'other_gallery.json'
                other_g.write_text(json.dumps(['zebra', 'ocean']))
                empty_args = list(args)
                empty_args[empty_args.index('--gallery') + 1] = other_g
                with self.assertRaises(ValueError):
                    call('prepare.py', empty_args + ['--output', tmp / 'no_match'])
                self.assertTrue((tmp / 'no_match/shared/matching_audit.json').is_file())
                self.assertFalse((tmp / 'no_match/cne/manifest.json').exists())
            finally:
                shutil.rmtree(ROOT / 'outputs/minimal_noun_supervision' / tag, ignore_errors=True)
                for f in (ROOT / 'weights').glob('vitb_mlp_infonce_mns_%s_*.pth' % tag):
                    f.unlink()


if __name__ == '__main__':
    unittest.main()
