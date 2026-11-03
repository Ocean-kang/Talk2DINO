"""Local-only model loading; reuse ProjectionLayer and QKV processing from src."""
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from .common import ROOT, offline, digest, normalize


def load_pth(path):
    import torch
    # Trusted user/repository files only; torch 2.1 compatible.
    return torch.load(path, map_location='cpu')


def save_pth(path, obj):
    import os
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, temp)
    os.replace(temp, path)


def model_args(p):
    p.add_argument('--clip-checkpoint', required=True, help='Existing local ViT-B-16.pt')
    p.add_argument('--dino-repo', required=True, help='Existing torch.hub DINOv2 source directory')
    p.add_argument('--dino-checkpoint', required=True, help='Existing ViT-B/14-reg state dict')
    p.add_argument('--teacher', default='weights/vitb_mlp_infonce.pth')
    p.add_argument('--model-config', default='configs/vitb_mlp_infonce.yaml')
    p.add_argument('--device', default='cuda')
    p.add_argument('--size', type=int, default=448)


class Models:
    def __init__(self, args):
        offline()
        import torch
        import clip
        from src.model import ProjectionLayer
        import torchvision.transforms as T
        for path in [args.clip_checkpoint, args.dino_checkpoint, args.teacher, args.model_config]:
            if not Path(path).is_file():
                raise FileNotFoundError('Existing local file required: ' + path)
        if not (Path(args.dino_repo) / 'hubconf.py').is_file():
            raise FileNotFoundError('Existing DINOv2 source with hubconf.py required')
        if args.size <= 0 or args.size % 14:
            raise ValueError('size must be positive and divisible by 14')
        self.torch, self.clip, self.device, self.size = torch, clip, args.device, args.size
        self.text, _ = clip.load(str(Path(args.clip_checkpoint).resolve()), device=args.device, jit=False)
        self.text.eval().requires_grad_(False)
        self.teacher = ProjectionLayer.from_config(args.model_config)
        self.teacher.load_state_dict(state_dict(load_pth(args.teacher)), strict=True)
        self.teacher.to(args.device).eval().requires_grad_(False)
        # pretrained=False prevents hubconf from fetching weights.
        self.dino = torch.hub.load(str(Path(args.dino_repo).resolve()), 'dinov2_vitb14_reg',
                                   source='local', pretrained=False)
        self.dino.load_state_dict(state_dict(load_pth(args.dino_checkpoint)), strict=True)
        self.dino.to(args.device).eval().requires_grad_(False)
        from src.hooks import get_self_attention
        self.hook = self.dino.blocks[-1].attn.qkv.register_forward_hook(get_self_attention)
        self.resize = T.Resize(args.size, interpolation=T.InterpolationMode.BICUBIC)
        self.crop = T.CenterCrop(args.size)
        self.to_tensor = T.Compose([T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])
        self.metadata = {'clip_sha256': digest(args.clip_checkpoint), 'dino_sha256': digest(args.dino_checkpoint),
                         'teacher_sha256': digest(args.teacher), 'config_sha256': digest(args.model_config),
                         'size': args.size, 'transform': 'shorter-side-bicubic-center-crop',
                         'prompt': 'a photo of {}', 'head_pool': 'sum(attention*patch)/P',
                         'attention_scale': .125}

    def encode(self, texts):
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 256):
                tokens = self.clip.tokenize(texts[i:i+256], truncate=True).to(self.device)
                out.append(self.text.encode_text(tokens).float().cpu())
        return torch.cat(out).numpy() if out else np.empty((0,512),np.float32)

    def project(self, raw):
        result=[]
        with self.torch.no_grad():
            for start in range(0,len(raw),256):
                t=self.torch.as_tensor(raw[start:start+256],dtype=self.torch.float32,device=self.device)
                result.append(self.teacher.project_clip_txt(t).float().cpu().numpy())
        return normalize(np.concatenate(result,axis=0)) if result else np.empty((0,768),np.float32)

    def extract(self, image, flip=False):
        from src.hooks import feats, process_self_attention
        torch = self.torch
        crop = self.crop(self.resize(image.convert('RGB')))
        if flip:
            crop = crop.transpose(getattr(Image, 'Transpose', Image).FLIP_LEFT_RIGHT)
        with torch.no_grad():
            feats.pop('self_attn', None)
            out = self.dino(self.to_tensor(crop).unsqueeze(0).to(self.device), is_training=True)
            patches = out['x_norm_patchtokens'].float()
            p = patches.shape[1]
            if patches.shape != (1,(self.size//14)**2,768):
                raise ValueError('Unexpected patch shape')
            _, logits = process_self_attention(feats.pop('self_attn'), 1, p+5, 12, 768, .125, 5, True)
            attention = logits.float().softmax(dim=-1)
            heads = (attention.unsqueeze(-1)*patches.unsqueeze(1)).mean(dim=2)
        return {'patches': patches[0].cpu().numpy(), 'attention': attention[0].cpu().numpy(),
                'heads': heads[0].cpu().numpy(), 'crop': crop}

    def close(self):
        self.hook.remove()


def state_dict(obj):
    for key in ('state_dict', 'model', 'teacher'):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
    result = {}
    for k, v in obj.items():
        for prefix in ('module.', 'projector.', 'backbone.'):
            if k.startswith(prefix):
                k = k[len(prefix):]
        result[k] = v
    return result


def image_path(root, rec):
    filename = rec['file_name']
    paths = [Path(root)/filename]
    split = next((s for s in ('train2014','val2014','test2014') if s in filename), None)
    if split:
        paths.append(Path(root)/split/Path(filename).name)
    for p in paths:
        if p.is_file():
            return p
    raise FileNotFoundError('Image missing: ' + str(paths))


def captions_data(path):
    from .common import read
    data = read(path) if str(path).endswith('.json') else load_pth(path)
    images = {int(r['id']):r for r in data['images']}
    caps = defaultdict(list)
    for ann in data['annotations']:
        if not isinstance(ann.get('caption'), str):
            raise ValueError('Need original caption strings, not just ann_feats')
        caps[int(ann['image_id'])].append(ann)
    if set(caps) - set(images):
        raise ValueError('Annotation image IDs missing from images')
    return images, caps
