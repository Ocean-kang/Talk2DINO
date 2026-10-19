"""Checkpoint and command interoperability tests without GPU or COCO."""
import argparse
from pathlib import Path
import tempfile
from unittest import TestCase, mock
from experiments.local_cocu_talk2dino.common import runner

class TestRunner(TestCase):
    def test_name_and_four_evals(self):
        self.assertEqual(runner.proj_name('A'),'vitb_mlp_infonce_local_cocu_talk2dino_A')
        self.assertEqual(runner.proj_name('H',2),'vitb_mlp_infonce_local_cocu_talk2dino_H_r2')
        self.assertEqual(list(runner.EVALS),['voc20','context59','ade150','coco_stuff'])

    def test_weight_archival_and_eval_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'weights').mkdir();(root/'outputs/exp').mkdir(parents=True)
            (root/'outputs/exp/train_augmented.pth').write_bytes(b'train')
            (root/'val.pth').write_bytes(b'val')
            args=argparse.Namespace(val='val.pth',config='configs/vitb_mlp_infonce.yaml',gpu='0',dry_run=False,force_train=False)
            def mock_training(cmd,args):
                self.assertIn('train.py',cmd)
                name=runner.proj_name('A')
                (root/'weights'/f'{name}.pth').write_bytes(b'model-weight')
            with mock.patch.object(runner,'ROOT',root), mock.patch.object(runner,'WEIGHTS',root/'weights/local_cocu_talk2dino'), mock.patch.object(runner,'run',side_effect=mock_training):
                w=runner.do_train('A',args,root/'outputs/exp')
                self.assertEqual(w.read_bytes(),b'model-weight')
                symlink=root/'weights'/f'{runner.proj_name("A")}.pth'
                self.assertTrue(symlink.is_symlink())
                self.assertEqual(symlink.resolve(),w.resolve())
