import unittest
import numpy as np
from experiments.ecca.common import normalize,top_mask,iou,pool_region,selection_pass,grouped_bootstrap,corr
from experiments.ecca.prepare import score_candidates,defaults
from experiments.ecca.diagnose import association

class CoreTests(unittest.TestCase):
    def test_spatial_alignment_and_ignore(self):
        valid=np.array([1,1,0,1],bool)
        self.assertEqual(iou([1,0,1,0],[1,0,0,0],valid),1.)
        self.assertIsNone(iou([0,0],[0,0],np.ones(2,bool)))
        self.assertEqual(int(top_mask([.1,.7,.3,.4],.5).sum()),2)
        np.testing.assert_array_equal(top_mask([1,1,1,1],.25),[1,0,0,0])
    def test_evidence_pool_cannot_average_irrelevant_patch(self):
        p=np.array([[1.,0.],[0.,1.]]);a=np.array([[.8,.2],[.1,.9]])
        np.testing.assert_allclose(pool_region(p,a,np.array([1.,0.]),[0,1]),[1,0])
        self.assertIsNone(pool_region(p,a,np.zeros(2),[0]))
    def test_filter_uses_all_three_evidence_conditions(self):
        c=defaults();r={'q':.7,'positive_mass':.8,'flip_iou':.6,'head_score':.4}
        self.assertTrue(selection_pass(r,c))
        for key in r:
            bad=dict(r);bad[key]=0;self.assertFalse(selection_pass(bad,c))
    def test_cluster_resampling(self):
        rows=[{'image_id':1,'x':1.},{'image_id':1,'x':1.},{'image_id':2,'x':3.}]
        result=grouped_bootstrap(rows,'x',repetitions=100)
        self.assertAlmostEqual(result['mean'],5/3);self.assertEqual(result['images'],2)
        self.assertIsNone(corr([1,1,1],[2,3,4]))
    def test_no_gt_no_c_gate_association(self):
        rows=[{'image_id':i,'dense_miss':None,'head_source':'caption'} for i in range(100)]
        self.assertIsNone(association(rows)['partial_correlation'])
        self.assertEqual(association(rows)['caption_grounded_images'],0)
    def test_negative_filter_and_flip_alignment(self):
        class FakeModel:size=28
        p=np.array([[1.,0.],[1.,0.],[0.,1.],[0.,1.]],np.float32)
        att=np.array([[.45,.45,.05,.05],[.05,.05,.45,.45]],np.float32)
        feat={'patches':p,'attention':att,'heads':att@p}
        flip={'patches':p.reshape(2,2,2)[:,::-1,:].reshape(4,2)}
        proj=normalize(np.array([[1.,0.],[0.,1.],[-1.,0.]],np.float32))
        cfg={**defaults(),'gallery':['giraffe','water','car'],'negative_presence_max':2.,'alias_cosine':.99}
        records=score_candidates(FakeModel(),feat,flip,[0,1,2],proj,proj,{1},cfg)
        giraffe=next(r for r in records if r['gallery_id']==0)
        self.assertNotIn(1,giraffe['negative_ids']);self.assertNotIn(0,giraffe['negative_ids'])
        self.assertEqual(giraffe['flip_iou'],1.);self.assertTrue(np.isfinite(giraffe['region']).all())

if __name__=='__main__':unittest.main()
