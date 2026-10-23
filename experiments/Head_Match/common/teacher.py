"""Share one frozen CLIP text encoder among many frozen Talk2DINO projectors."""
from __future__ import annotations
import sys
import torch
import torch.nn.functional as F
from .io import ROOT


class TeacherBank:
    def __init__(self, weights, config, device='cuda', batch_size=256):
        if not weights:raise ValueError('Provide at least one Teacher checkpoint')
        sys.path.insert(0,str(ROOT))
        import clip
        from src.model import ProjectionLayer
        self.clip=clip; self.device=torch.device(device);self.batch_size=batch_size
        self.text,_=clip.load('ViT-B/16',device=device,jit=False)
        self.text.eval().requires_grad_(False)
        self.projectors=[]
        for checkpoint in weights:
            net=ProjectionLayer.from_config(str(config))
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if isinstance(state,dict) and 'state_dict' in state:state=state['state_dict']
            if isinstance(state,dict) and 'model' in state:state=state['model']
            state={k.removeprefix('module.').removeprefix('projector.'):v for k,v in state.items()}
            net.load_state_dict(state,strict=True)
            net.to(self.device).eval().requires_grad_(False)
            self.projectors.append(net)
        self.cache={}
    @torch.no_grad()
    def embed(self, sentences, template='{}'):
        """Returns frozen CLIP embeddings on CPU; caches concept prompts, not sentences."""
        unique=list(dict.fromkeys(str(s) for s in sentences))
        missing=[s for s in unique if (template,s) not in self.cache]
        for start in range(0,len(missing),self.batch_size):
            batch=missing[start:start+self.batch_size]
            tokens=self.clip.tokenize([template.format(t) for t in batch],truncate=True).to(self.device)
            vec=self.text.encode_text(tokens).float().detach().cpu()
            for s,v in zip(batch,vec):self.cache[(template,s)]=v
        if not sentences:return torch.empty((0,512),dtype=torch.float32)
        return torch.stack([self.cache[(template,str(s))] for s in sentences])
    @torch.no_grad()
    def project(self, text_features, teacher_index=0):
        f=torch.as_tensor(text_features,device=self.device,dtype=torch.float32)
        return F.normalize(self.projectors[teacher_index].project_clip_txt(f),dim=-1)
    def clip_unit(self, text_features):
        return F.normalize(torch.as_tensor(text_features,device=self.device,dtype=torch.float32),dim=-1)
    @torch.no_grad()
    def project_all(self,text_features):
        return [self.project(text_features,i) for i in range(len(self.projectors))]
