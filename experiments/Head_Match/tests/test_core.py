import sys
import unittest
from pathlib import Path
import torch
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from experiments.Head_Match.common.core import words,mentioned,clean_vocabulary,assignments,scored_concepts,consensus,compose,shift
from experiments.Head_Match.common.build_train import plan

class CoreTests(unittest.TestCase):
    def test_lexical_guard(self):
        self.assertTrue(mentioned('bicycles','A man rides bicycles.'))
        self.assertTrue(mentioned('women','Two women laugh'))
        self.assertFalse(mentioned('helmet','A man rides bicycles'))
        self.assertEqual(len(clean_vocabulary(['helmet','Helmet','helmets'])),1)
    def test_soft_assignment(self):
        heads=torch.eye(3)
        p,sim=assignments(heads,heads[0],.08,'soft',2)
        self.assertAlmostEqual(float(p.sum()),1.,places=6)
        self.assertAlmostEqual(float(p[2]),0.,places=6)
        q,_=assignments(heads,heads[0],.08,'argmax',2)
        self.assertEqual(q.tolist(),[1.,0.,0.])
    def test_missing_and_covered(self):
        heads=torch.eye(3);p=torch.tensor([1.,0.,0.])
        concepts=['helmet','bicycle']
        text=torch.tensor([.1,0.,0.])
        feat=torch.tensor([[1.,0.,0.],[.2,.9,0.]])
        clip=torch.tensor([[0.,1.,0.],[0.,0.,1.]])
        result=scored_concepts(heads,p,feat,text,clip,concepts,caption='A man riding a bicycle',
                                min_visual=.2,novelty_max=.7,covered_sim=.99,coverage_threshold=.8)
        self.assertEqual(result['gate'],'missing')
        self.assertEqual(result['selected'][0]['concept'],'helmet')
    def test_consensus(self):
        a=consensus([{'helmet':.7,'dog':.2},{'helmet':.6,'dog':.8}],.5,2)
        self.assertEqual(a,{'helmet':2})
    def test_compose(self):
        self.assertEqual(compose('A rider.',['helmet']),'A rider. Also visible: helmet.')
        self.assertEqual(compose('A rider.',[]),'A rider.')
        self.assertEqual(shift(torch.tensor([1.,0.]),torch.tensor([0.,1.])),1.)
    def test_train_plan_preserves_five_and_changes_only_one(self):
        base={'images':[{'id':2,'disentangled_self_attn':torch.ones((12,768))}],
              'annotations':[{'id':k,'image_id':2,'caption':f'caption {k}',
                              'ann_feats':torch.ones(512)} for k in range(5)]}
        records=[{'annotation_id':k,'raw_caption':f'caption {k}',
                  'augmented_caption':f'caption {k}'+('. Also visible: helmet.' if k==1 else ''),
                  'added':[{'concept':'helmet'}] if k==1 else []} for k in range(5)]
        images,anns,changed,stats=plan(base,{'images':{'2':{'captions':records}}})
        self.assertEqual(len(anns),5)
        self.assertEqual(len(changed),1)
        self.assertIs(anns[0]['ann_feats'],base['annotations'][0]['ann_feats'])
        self.assertNotIn('head_match_original_caption',base['annotations'][1])
    def test_strict_subset(self):
        base={'images':[{'id':1},{'id':2}],
              'annotations':[{'id':i,'image_id':1,'caption':'x'} for i in range(5)]}
        with self.assertRaisesRegex(ValueError,'subset'):
            plan(base,{'images':{'1':{'captions':[]}}})

if __name__=='__main__':unittest.main()

class DecisionSmokeTests(unittest.TestCase):
    def test_four_variants_with_mocked_teacher(self):
        from types import SimpleNamespace
        from experiments.Head_Match.common.discovery import pick_for_caption
        class TeacherStub:
            def project(self, feat, which=0):
                return torch.nn.functional.normalize(torch.as_tensor(feat).float(),dim=-1)
            def embed(self, sentences):
                return torch.tensor([[1.,0.,0.] for _ in sentences])
        args=SimpleNamespace(temperature=.07,assignment='soft',top_heads=2,
            coverage_threshold=.9,covered_sim=.99,min_visual=.1,novelty_max=.9,
            proposal_count=2,max_concepts=1,min_delta=-2.,max_head_shift=1.,min_votes=2)
        heads=torch.eye(3)
        concepts=['helmet','road']
        clip=torch.tensor([[0.,1.,0.],[0.,0.,1.]])
        projected=[torch.tensor([[1.,0.,0.],[.0,1.,0.]])]
        for variant in 'ABCD':
            result=pick_for_caption(variant,TeacherStub(),heads,'A cyclist',torch.tensor([1.,0.,0.]),
                 concepts,clip,projected,args,[0])
            self.assertEqual(result['raw_caption'],'A cyclist')
            self.assertEqual(len(result['head_weights_raw']),3)
            self.assertLessEqual(len(result['added']),1)
