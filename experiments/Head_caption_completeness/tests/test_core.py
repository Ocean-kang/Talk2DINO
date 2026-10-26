"""CPU-only fast sanity checks; run `python -m unittest discover ...`."""
import unittest
from experiments.Head_caption_completeness.core import coverage,norm,mentioned,lexical_gallery,make_add,make_remove
from experiments.Head_caption_completeness.build import plan


class TestBookkeeping(unittest.TestCase):
    def test_exclusion_unknown(self):
        x=coverage([1,.9,.5],[0,.9,1],[True,False,False])
        self.assertAlmostEqual(x['d_add'],1.)
        self.assertEqual(x['valid_count'],1)
        self.assertEqual(x['d_remove'],0.)
    def test_bidirectional(self):
        self.assertAlmostEqual(coverage([1,0],[0,1])['d_add'],1.)
        self.assertAlmostEqual(coverage([1,0],[0,1])['d_remove'],1.)
    def test_lexical(self):
        words=['traffic light','dogs','table']
        self.assertEqual(set(lexical_gallery('The dog near a traffic light',words)),{'dogs','traffic light'})
        self.assertTrue(mentioned('traffic lights','the traffic light'))
        self.assertFalse(mentioned('cat','cattle'))
    def test_remove_only_safe_clause(self):
        self.assertEqual(make_remove('A woman is standing near a bus and a dog.','dog'),
                         'A woman is standing near a bus.')
        self.assertIsNone(make_remove('A dog runs across the grass.','dog'))
        self.assertEqual(make_add('A person runs.',['dog']), 'A person runs; also visible: dog.')
    def test_build_exact_ids(self):
        image={'id':1,'disentangled_self_attn':None}
        anns=[{'id':100+i,'image_id':1,'caption':f'caption {i}','ann_feats':0} for i in range(5)]
        raw={'images':[image],'annotations':anns}
        rec=[{'annotation_id':100+i,'image_id':1,'raw_caption':f'caption {i}',
              'caption':('caption 0; also visible: dog.' if i==0 else f'caption {i}'),
              'operation':('ADD' if i==0 else 'KEEP'),'concept':('dog' if i==0 else None)} for i in range(5)]
        imgs,edited,changed=plan(raw,rec)
        self.assertEqual([x['id'] for x in edited],[100,101,102,103,104])
        self.assertEqual(len(changed),1)
        self.assertEqual(edited[0]['bsc_cr_raw_caption'],'caption 0')
        with self.assertRaises(ValueError):plan(raw,rec[:-1])


if __name__=='__main__':unittest.main()
