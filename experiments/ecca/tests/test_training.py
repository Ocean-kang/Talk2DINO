"""CPU integration test in the EXISTING Talk2DINO environment; no model downloads."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
HAS_TORCH=importlib.util.find_spec('torch') is not None
HAS_CLIP=importlib.util.find_spec('clip') is not None

@unittest.skipUnless(HAS_TORCH and HAS_CLIP,'Requires existing Talk2DINO PyTorch/CLIP environment')
class TrainingTests(unittest.TestCase):
    def test_zero_local_weight_preserves_e2_trajectory_and_budget(self):
        import numpy as np
        import torch
        from experiments.ecca.common import ROOT,digest
        torch.manual_seed(5)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);train=root/'train.pth';val=root/'val.pth'
            images=[{'id':i,'disentangled_self_attn':torch.randn(12,768)} for i in range(4)]
            anns=[{'id':i,'image_id':i//2,'caption':'a cat','ann_feats':torch.randn(512)} for i in range(8)]
            torch.save({'images':images,'annotations':anns},train)
            torch.save({'images':[{'id':20+i,'avg_self_attn_out':torch.randn(768)} for i in range(4)],
                        'annotations':[{'id':20+i,'image_id':20+i,'caption':'a dog','ann_feats':torch.randn(512)} for i in range(4)]},val)
            prepared=root/'prepared';(prepared/'records').mkdir(parents=True)
            signature='synthetic-only';config=ROOT/'configs/vitb_mlp_infonce.yaml'
            (prepared/'source.json').write_text(json.dumps({'signature':signature,'recovery_only':False,
                'captions_sha256':digest(train),'image_ids':list(range(4)),
                'models':{'config_sha256':digest(config)}}))
            (prepared/'feature_checks.json').write_text(json.dumps({'count':1,'checked':[{'synthetic':True}]}))
            raw=np.random.RandomState(3).randn(5,512).astype('float32');torch.save({'raw':raw},prepared/'text_bank.pth')
            for iid in range(4):
                selected=[{'gallery_id':iid,'region':np.ones(768,dtype='float32')/np.sqrt(768),
                    'adaptive_region':np.ones(768,dtype='float32')/np.sqrt(768),'negative_ids':[4]}]
                torch.save({'signature':signature,'selected':selected,'head_control':selected,
                            'known_covered_ids':[]},prepared/'records'/(str(iid)+'.pth'))
            for method in ('E0','E2','E3'):
                cmd=[sys.executable,'-m','experiments.ecca.train','--method',method,'--train',str(train),
                     '--val',str(val),'--prepared',str(prepared),'--model-config',str(config),
                     '--output',str(root/method),'--device','cpu','--epochs','1','--workers','0','--local-weight','0']
                run=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            e2=torch.load(root/'E2/last.pth',map_location='cpu');e3=torch.load(root/'E3/last.pth',map_location='cpu')
            for name in e2:torch.testing.assert_close(e2[name],e3[name],rtol=0,atol=0)
            for method in ('E0','E2','E3'):
                history=json.loads((root/method/'history.jsonl').read_text().strip());self.assertEqual(history['steps'],1)
            self.assertGreater(json.loads((root/'E3/history.jsonl').read_text())['local_examples'],0)

if __name__=='__main__':unittest.main()
