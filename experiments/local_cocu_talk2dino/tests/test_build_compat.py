"""Test integration against original novel CoCu build_plan when repo checkout has it."""
import unittest
import importlib.util
from pathlib import Path
import sys
import torch
from experiments.local_cocu_talk2dino.common.data import ROOT

class TestOriginalBuilder(unittest.TestCase):
    def test_train_format(self):
        path=ROOT/'experiments/local_cocu_novel/build_train.py'
        if not path.is_file(): self.skipTest('original repository not installed in test environment')
        folder=str(path.parent)
        sys.path.insert(0,folder)
        try:
            spec=importlib.util.spec_from_file_location('original_build_train_for_test',path)
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            image_id=123
            base={'images':[{'id':image_id,'disentangled_self_attn':torch.ones(12,768)}],
                  'annotations':[{'id':j+1,'image_id':image_id,
                                  'caption':f'This is original caption {j}',
                                  'ann_feats':torch.ones(512)} for j in range(5)]}
            matches={str(image_id):[{'concept':'airplane','score':.33,'best_head':2}]}
            raw,supp,hist,_=mod.plan(base,matches,maximum=3)
            self.assertEqual((len(raw),len(supp),hist[1]),(5,1,1))
            self.assertEqual(supp[0]['caption'],'a photo of a airplane')
        finally:sys.path.remove(folder)
