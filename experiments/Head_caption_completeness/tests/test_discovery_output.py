import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from experiments.Head_caption_completeness.diagnose import output_rows

class Fake:
    def __init__(self):
        self.images=[{'id':1}]
        self.counters=Counter(captions=5,images=1)
    def per_image(self,im):
        result=[]
        for i in range(5):
            before={'d_add':.4 if i<2 else .0,'d_remove':.5 if i==2 else .0}
            cand={'ADD':({'op':'ADD','caption':f'A dog {i}; also visible: cat.',
                          'concept':'cat','score':.98,'after_fixed':{'d_add':0,'d_remove':0},
                          'teacher_delta_fixed_head':.03} if i<2 else None),
                  'REMOVE':({'op':'REMOVE','caption':'A dog.',
                             'concept':'table','score':.99,'after_fixed':{'d_add':0,'d_remove':0},
                             'teacher_delta_fixed_head':.02} if i==2 else None)}
            result.append({'image_id':1,'annotation_id':100+i,'raw_caption':f'A dog {i}.',
                           'head_argmax':i,'before':before,'recommendation':'KEEP',
                           'proposals':cand,'candidates':[]})
        return result

class TestOutput(unittest.TestCase):
    def test_variants_and_quota(self):
        with tempfile.TemporaryDirectory() as d:
            a=SimpleNamespace(output=Path(d),audit_images=1000,add_gate=.2,remove_gate=.25,
                    base_train=Path('/mock/train.pth'),gallery=Path('/mock/gallery.json'),
                    teacher=Path('/mock/teacher.pth'),clip_images=Path('/mock/img.pth'),
                    clip_neighbors=Path('/mock/neighbor.json'))
            output_rows(a,Fake())
            docs={e:[json.loads(s) for s in (Path(d)/f'{e}.jsonl').read_text().splitlines()]
                  for e in ['E0','E1','E2','E3','E4']}
            self.assertTrue(all(len(v)==5 for v in docs.values()))
            self.assertEqual([r['annotation_id'] for r in docs['E3']],[100,101,102,103,104])
            self.assertEqual([r['operation'] for r in docs['E0']],['KEEP']*5)
            self.assertEqual(Counter(r['operation'] for r in docs['E3']),
                             Counter(r['operation'] for r in docs['E4']))
            self.assertEqual(Counter(r['operation'] for r in docs['E3'])['ADD'],2)
            self.assertEqual(Counter(r['operation'] for r in docs['E3'])['REMOVE'],1)
if __name__=='__main__':unittest.main()
