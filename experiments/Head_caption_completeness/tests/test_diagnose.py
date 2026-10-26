import unittest
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from experiments.Head_caption_completeness.diagnose import Context,text_values,ramp
from experiments.Head_caption_completeness.core import norm

class TinyTeacher:
    def __init__(self):self.cache={};self.clip=SimpleNamespace(tokenize=lambda s,truncate=False:None)
    def project(self,x):
        x=x.float()
        return F.normalize(F.pad(x,(0,768-x.shape[-1])),dim=-1)
    def embed(self,texts,template='{}'):
        v=[]
        for s in texts:
            code=sum(map(ord,s))
            x=torch.arange(512,dtype=torch.float32)
            v.append(F.normalize(torch.sin(x*(code%30+1)/40),dim=-1))
        return torch.stack(v)

class TestDiagnostics(unittest.TestCase):
    def test_unknown_is_excluded_by_explicit_mask(self):
        val,u=text_values(torch.tensor([.38,.48,.7]),{0})
        self.assertTrue(u[1]);self.assertFalse(u[0]);self.assertFalse(u[2])
        self.assertEqual(val.tolist(),[1.0,0.0,0.800000011920929])
    def test_per_image_bookkeeping(self):
        x=Context.__new__(Context)
        x.device=torch.device('cpu')
        x.a=SimpleNamespace(head_offset=.0,head_margin=.3,image_offset=.0,image_margin=.2,
                head_weight=.8,head_topk=2,neighbor_topk=2,text_low=.44,text_high=.6,
                relevant_visual=.25,min_visual=.7,min_image=.2,audit_top=2,try_top=1,add_gate=.2,remove_gate=.25,
                remove_visual_ceiling=.25,other_head_ceiling=.45,remove_image_ceiling=.25,
                max_caption_chars=256,min_improvement=.008,max_other_regression=.04,
                min_teacher_delta=-.005,min_wrong_margin=.005)
        x.teacher=TinyTeacher()
        x.vocab=['dog','cat','bus','table'];x.lex={norm(y):y for y in x.vocab};x.idx={v:i for i,v in enumerate(x.vocab)}
        x.clip_words=F.normalize(torch.randn(4,512),dim=-1)
        x.proj_words=F.normalize(torch.randn(4,768),dim=-1)
        x.null_head=torch.zeros(4);x.null_img=torch.zeros(4)
        x.clip_images={1:F.normalize(torch.randn(512),dim=-1)}
        x.neighbors={'1':[]};x.counters={"captions":0,"images":0}
        x.groups={1:[{'id':100+i,'image_id':1,'caption':f'A dog near a table {i}.',
                       'ann_feats':torch.randn(512)} for i in range(5)]}
        im={'id':1,'disentangled_self_attn':torch.randn(12,768)}
        result=x.per_image(im)
        self.assertEqual(len(result),5)
        self.assertEqual([r['annotation_id'] for r in result],list(range(100,105)))
        self.assertTrue(all(0<=r['head_argmax']<12 for r in result))
        self.assertEqual(x.counters['images'],1)
        self.assertEqual(x.counters['captions'],5)

if __name__=='__main__':unittest.main()
