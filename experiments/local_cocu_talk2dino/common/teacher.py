"""Uses precisely the existing CLIP + Talk2DINO text projector mapping."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from .data import ROOT

class Teacher:
    def __init__(self, weights, config, device='cuda', clip_name='ViT-B/16', batch=512):
        # Import only at runtime: pure-algorithm tests do not need CLIP or DINOv2.
        sys.path.insert(0, str(ROOT))
        import clip
        from src.model import ProjectionLayer
        self.device = torch.device(device)
        self.clip = clip
        self.batch = batch
        self.text, _ = clip.load(clip_name, device=device, jit=False)
        self.text.eval().requires_grad_(False)
        self.projector = ProjectionLayer.from_config(str(config))
        ckpt = torch.load(weights, map_location='cpu')
        if 'state_dict' in ckpt: ckpt = ckpt['state_dict']
        elif 'model' in ckpt: ckpt = ckpt['model']
        ckpt = {k.removeprefix('module.').removeprefix('projector.'):v for k,v in ckpt.items()}
        self.projector.load_state_dict(ckpt, strict=True)
        self.projector.to(device).eval().requires_grad_(False)
        self.cache = {}
    @torch.no_grad()
    def encode(self, words, template='a photo of {}'):
        missing = list(dict.fromkeys(x for x in words if (template,x) not in self.cache))
        for start in range(0, len(missing), self.batch):
            part = missing[start:start+self.batch]
            tokens = self.clip.tokenize([template.format(w) for w in part], truncate=True).to(self.device)
            embeddings = self.projector.project_clip_txt(self.text.encode_text(tokens).float())
            embeddings = F.normalize(embeddings.float(), dim=-1).cpu()
            for word,vec in zip(part,embeddings): self.cache[(template,word)] = vec
        return torch.stack([self.cache[(template,w)] for w in words]).to(self.device) if words else torch.empty((0,768),device=self.device)
    @torch.no_grad()
    def encode_captions(self, captions):
        # Keep captions exactly as COCO wrote them, not "a photo of {}" concept prompts.
        return self.encode(captions, template='{}')
    @torch.no_grad()
    def sim(self, heads, embeddings):
        h = F.normalize(torch.as_tensor(heads, device=self.device).float(),dim=-1)
        return h @ embeddings.T
