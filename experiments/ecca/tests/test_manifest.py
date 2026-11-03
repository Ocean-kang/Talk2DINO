"""No model, CUDA, COCO download, or pycocotools required."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
from experiments.ecca.coco_masks import segmentation_mask, semantic_mask, BACKEND


class ManifestTests(unittest.TestCase):
    def test_polygon_coordinates_not_inclusive_pillow_endpoints(self):
        result = segmentation_mask([[1,1, 4,1, 4,3, 1,3]], 5, 6)
        expected = np.zeros((5,6), bool); expected[1:3,1:4] = True
        np.testing.assert_array_equal(result, expected)

    def test_fractional_polygon_and_union(self):
        result = segmentation_mask([[.2,.2, 2.2,.2, 2.2,2.2, .2,2.2],
                                    [4,3, 5,3, 5,4, 4,4]], 5, 6)
        expected = np.zeros((5,6), bool); expected[:2,:2] = True; expected[3,4] = True
        np.testing.assert_array_equal(result, expected)

    def test_compressed_signed_deltas_and_column_major_rle(self):
        expected = np.array([[0,0,1,1], [1,1,0,0], [1,1,0,0]], bool)
        for counts in ([1,2,1,3,2,1,2], '12111N0', b'12111N0'):
            np.testing.assert_array_equal(segmentation_mask({'size':[3,4], 'counts':counts},3,4),expected)

    def test_large_rle_runs(self):
        # COCO signed 5-bit chunks: P3 = 96, T3 = 100.
        result = segmentation_mask({'size':[10,20], 'counts':'P34T3'},10,20)
        self.assertEqual(int(result.sum()),4)
        np.testing.assert_array_equal(np.flatnonzero(result.ravel(order='F')),np.arange(96,100))

    def test_invalid_mask_inputs_fail(self):
        for counts in ([2,-1,11], [2,3], [1.5,10.5], 'P', '~'):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                segmentation_mask({'size':[3,4], 'counts':counts},3,4)
        with self.assertRaises(ValueError):
            segmentation_mask({'size':[4,3], 'counts':[12]},3,4)
        with self.assertRaises(ValueError):
            segmentation_mask([[0,0,1]],3,4)
        with self.assertRaises(ValueError):
            segmentation_mask({'bbox':[0,0,3,4]},3,4)

    def test_overlap_crowd_and_background_labels(self):
        a = {'category_id':3, 'segmentation':[[0,0,3,0,3,2,0,2]]}
        b = {'category_id':9, 'segmentation':[[2,0,4,0,4,2,2,2]]}
        crowd = {'category_id':9, 'iscrowd':1, 'segmentation':{'size':[3,4], 'counts':[2,1,9]}}
        result = semantic_mask([a,b,crowd],3,4,{3:1,9:2})
        np.testing.assert_array_equal(result,[[1,1,255,2],[1,1,255,2],[255,0,0,0]])
        np.testing.assert_array_equal(result,semantic_mask([crowd,b,a],3,4,{3:1,9:2}))

    def test_manifest_cli_without_pycocotools_and_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instances = {'images':[{'id':7,'file_name':'test.jpg','height':3,'width':4}],
                         'categories':[{'id':9,'name':'dog'},{'id':3,'name':'cat'}],
                         'annotations':[{'image_id':7,'category_id':3,'segmentation':[[0,0,2,0,2,2,0,2]]},
                                        {'image_id':7,'category_id':9,'iscrowd':1,
                                         'segmentation':{'size':[3,4],'counts':';1'}}]}
            captions = {'annotations':[{'image_id':7,'caption':'A cat.'}]}
            for name, data in [('instances.json',instances),('captions.json',captions)]:
                (root/name).write_text(json.dumps(data))
            args = ['--captions',str(root/'captions.json'),'--instances',str(root/'instances.json'),
                    '--output',str(root/'manifest.json'),'--limit','200']
            # Importing pycocotools is explicitly blocked even if caller installed it.
            script = "import sys,runpy; sys.modules['pycocotools']=None; sys.argv=['manifest','coco']+sys.argv[1:]; runpy.run_module('experiments.ecca.make_manifest',run_name='__main__')"
            for _ in range(2):
                run = subprocess.run([sys.executable,'-c',script]+args,capture_output=True,text=True)
                self.assertEqual(run.returncode,0,run.stderr)
            manifest = json.loads((root/'manifest.json').read_text())
            self.assertEqual(manifest['mask_backend'],BACKEND)
            self.assertEqual(manifest['classes'],[{'id':1,'name':'cat'},{'id':2,'name':'dog'}])
            with Image.open(root/manifest['images'][0]['mask']) as im:
                expected = np.array([[1,1,0,0],[1,1,0,0],[0,0,0,255]],np.uint8)
                np.testing.assert_array_equal(np.asarray(im),expected)

    def test_missing_manifest_stops_before_loading_models_or_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = ['--manifest',str(root/'absent.json'),'--image-root',tmp,'--output',str(root/'diagnosis')]
            for key in ['--clip-checkpoint','--dino-repo','--dino-checkpoint']:
                args += [key,str(root/'absent.pth')]
            run = subprocess.run([sys.executable,'-m','experiments.ecca.diagnose']+args,capture_output=True,text=True)
            self.assertEqual(run.returncode,2,run.stderr)
            self.assertIn('Run bash experiments/ecca/run.sh manifest',run.stderr)
            self.assertNotIn('Traceback',run.stderr)
            self.assertFalse((root/'diagnosis').exists())


if __name__ == '__main__':
    unittest.main()
