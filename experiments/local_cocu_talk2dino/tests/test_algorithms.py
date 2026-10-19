"""CPU-only deterministic behavioral checks; no COCO or pretrained weights required."""
import unittest
import torch
from experiments.local_cocu_talk2dino.common.algorithms import (
    head_score, top_concepts, uncovered_heads, consistency_score,
    counterfactual_score, view_agreement, bbox_from_map)
from experiments.local_cocu_talk2dino.common.candidates import from_neighbors
from experiments.local_cocu_talk2dino.common.data import already_mentioned,novel

class TestDiscovery(unittest.TestCase):
    def test_novel_phrase_and_plural(self):
        raw=['Two dogs are running with a ball', 'Outdoor road']
        self.assertTrue(already_mentioned('dog',raw))
        self.assertTrue(already_mentioned('road',raw))
        self.assertFalse(already_mentioned('traffic light',raw))
        self.assertEqual(novel(['dog','traffic light','Traffic Light'],raw),['traffic light'])

    def test_head_topk_filters_captions(self):
        heads=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
        vec=torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        rows=top_concepts(['cat','tree','boat'],heads,vec,['a cat standing'],k=3,per_head_k=3)
        self.assertEqual(rows[0]['concept'],'tree')
        self.assertEqual(rows[0]['best_head'],1)
        self.assertNotIn('cat',[x['concept'] for x in rows])
        rows=top_concepts(['cat','tree','boat'],heads,vec,['a cat standing'],k=3,per_head_k=3,permitted_heads=[0])
        self.assertEqual(rows[0]['best_head'],0)

    def test_uncovered_heads(self):
        h=torch.tensor([[1.,0.],[0.,1.]])
        q,cover=uncovered_heads(h,torch.tensor([[1.,0.]]),1)
        self.assertEqual(q,[1])
        self.assertAlmostEqual(cover[0],1.)
        self.assertAlmostEqual(cover[1],0.)

    def test_consistency_rare_and_negative(self):
        score,count,mean,sd=consistency_score(.7,[.3,.4],min_support=2,rare_high=.65)
        self.assertGreater(score,-1e8)
        self.assertEqual(count,0)
        score,*_=consistency_score(.5,[.1,.2],min_support=2,rare_high=.65)
        self.assertLess(score,-1e8)
        self.assertAlmostEqual(counterfactual_score(.7,[.2,.4],.5),.55)

    def test_views(self):
        self.assertIsNone(view_agreement([.05,.1,.4],min_views=3,threshold=.12))
        self.assertIsNotNone(view_agreement([.4,.5,.3],min_views=2,threshold=.2))

    def test_crop_rect(self):
        v=torch.zeros((32,32));v[10:12,18:20]=1
        x0,y0,x1,y1=bbox_from_map(v,.99,expand=0.,pixels=448)
        self.assertTrue(0<=x0<x1<=448 and 0<=y0<y1<=448)

    def test_neighbor_phrase(self):
        out=from_neighbors([2],{2:['A red traffic light near a car']},['traffic light','car','dog'])
        self.assertEqual(set(out),{'traffic light','car'})

if __name__=='__main__': unittest.main()
