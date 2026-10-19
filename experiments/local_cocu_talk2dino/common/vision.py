"""Original DINOv2 attention hook and feature aggregation, reused for C/D/G only.

Head pooled vector implements the formula from experiments/local_cocu/extract_head_features.py;
all experiments A/B/E/F/H use the existing cached head_features.pt directly.
"""
from __future__ import annotations
from pathlib import Path
import sys
import torch
from PIL import Image, ImageEnhance, ImageOps
from .data import ROOT

class Vision:
    def __init__(self, device='cuda', size=448):
        sys.path.insert(0,str(ROOT))
        import torchvision.transforms as T
        from src.hooks import get_self_attention,process_self_attention,feats
        self.T = T; self.feats=feats;self.process=process_self_attention
        self.device=device;self.size=size
        self.model=torch.hub.load('facebookresearch/dinov2','dinov2_vitb14_reg').to(device).eval()
        self.hook=self.model.blocks[-1].attn.qkv.register_forward_hook(get_self_attention)
        self.normalize=T.Compose([T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
    def close(self): self.hook.remove()
    def image(self,path):
        with Image.open(path) as img: return img.convert('RGB').copy()
    def center(self,img):
        return self.T.CenterCrop(self.size)(self.T.Resize(self.size,interpolation=self.T.InterpolationMode.BICUBIC)(img))
    @torch.no_grad()
    def get(self, pil):
        raw = self.center(pil)
        t = self.normalize(raw).unsqueeze(0).to(self.device)
        outputs=self.model(t,is_training=True)
        patches=outputs['x_norm_patchtokens']
        _, maps=self.process(self.feats['self_attn'],1,5+(self.size//14)**2,
                            self.model.num_heads,patches.shape[-1],
                            (patches.shape[-1]//self.model.num_heads)**-.5,5,ret_self_attn_maps=True)
        maps=maps.softmax(dim=-1)
        heads=(patches.unsqueeze(1)*maps.unsqueeze(-1)).mean(dim=2)
        # Avoid accidental GPU retention in long COCO processing.
        return heads[0].float().cpu(),maps[0].float().cpu(),raw
    @staticmethod
    def attention_strength(attn):
        a=attn.flatten(start_dim=1)
        n=max(1,int(a.shape[-1]*0.1))
        # Uniform attention has concentration ratio 1; concentrated mass >1.
        ratio=a.topk(n,dim=-1).values.sum(dim=-1)/(n/a.shape[-1])
        return ratio.tolist()
    def views(self, path):
        pil=self.image(path)
        return [self.get(pil)[0],
                self.get(ImageEnhance.Brightness(pil).enhance(.85))[0],
                self.get(ImageOps.mirror(pil))[0]]
