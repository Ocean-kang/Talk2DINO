"""In-process local weight injection; repository evaluator files remain unchanged."""
import os
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.ecca.common import offline
offline()
import torch
import clip
from experiments.ecca.runtime import load_pth, state_dict
original_hub=torch.hub.load
original_clip=clip.load


def local_hub(repo_or_dir,model,*args,**kwargs):
    if model!='dinov2_vitb14_reg':raise ValueError('Only existing DINOv2 ViT-B/14-reg is supported')
    kwargs.pop('source',None);kwargs.pop('pretrained',None)
    model_obj=original_hub(os.environ['ECCA_DINO_REPO'],model,*args,source='local',pretrained=False,**kwargs)
    model_obj.load_state_dict(state_dict(load_pth(os.environ['ECCA_DINO_CHECKPOINT'])),strict=True)
    return model_obj


def local_clip(name,*args,**kwargs):
    if name not in ('ViT-B/16',os.environ['ECCA_CLIP_CHECKPOINT']):raise ValueError('Unexpected CLIP model '+name)
    return original_clip(os.environ['ECCA_CLIP_CHECKPOINT'],*args,**kwargs)


torch.hub.load=local_hub;clip.load=local_clip
# Import the same module namespace used by main.py, then inject ONLY bg-clean off
# when requested. All prompts, masks, resizing and evaluator behavior remain original.
seg=ROOT/'src/open_vocabulary_segmentation';sys.path.insert(0,str(seg))
if os.environ.get('ECCA_BG_CLEAN')=='off':
    from models.dinotext.dinotext import DINOText
    init=DINOText.__init__
    def no_bg(self,*args,**kwargs):
        kwargs['with_bg_clean']=False
        init(self,*args,**kwargs)
    DINOText.__init__=no_bg
runpy.run_path(str(seg/'main.py'),run_name='__main__')
