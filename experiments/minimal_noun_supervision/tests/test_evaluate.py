"""Check original task/config/weight routing without running MMSegmentation."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]


class EvalRouting(unittest.TestCase):
    def test_five_original_configs_and_new_weight_stem(self):
        with tempfile.TemporaryDirectory(prefix='mns_eval_') as tmp:
            tag = Path(tmp).name
            stem = 'vitb_mlp_infonce_mns_%s_cne_seed123' % tag
            run = ROOT / 'outputs/minimal_noun_supervision' / tag / 'cne/seed_123'
            checkpoint = ROOT / 'weights' / (stem + '.pth')
            run.mkdir(parents=True)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b'dry-run fixture: never loaded')
            (run / 'result.json').write_text(json.dumps({'checkpoint_stem': stem}))
            try:
                capture = io.StringIO()
                with patch.object(sys, 'argv', ['evaluate.py', '--method', 'cne', '--tag', tag, '--dry-run']), contextlib.redirect_stdout(capture):
                    runpy.run_path(str(HERE / 'evaluate.py'), run_name='__main__')
                lines = capture.getvalue().strip().splitlines()
                self.assertEqual(len(lines), 5)
                self.assertTrue(all('src/open_vocabulary_segmentation/main.py' in l for l in lines))
                self.assertTrue(all('model.proj_name=' + stem in l for l in lines))
                for prefix in ('voc', 'voc_bg', 'context', 'ade', 'stuff'):
                    self.assertTrue(any('/%s/eval_%s_pamr.yml' % (prefix, prefix) in l for l in lines))
                self.assertFalse((run / 'segmentation').exists())
            finally:
                checkpoint.unlink()
                shutil.rmtree(run.parents[1])


if __name__ == '__main__':
    unittest.main()
