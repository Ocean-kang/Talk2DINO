"""DINOv2 ViT-B/14-reg attention extraction, matching ORIGINAL Talk2DINO.

Uses src.hooks and the exact pooling formula in dino_extraction_v2.py.
"""
from __future__ import annotations

import torch
import torchvision.transforms as T
from PIL import Image
from src.hooks import feats, get_self_attention, process_self_attention
from .core import head_pool


class DinoHeadExtractor:
    def __init__(self, device='cuda', size=448, repo='facebookresearch/dinov2', hub_source='github'):
        if size <= 0 or size % 14:
            raise ValueError('DINOv2 crop size must be a positive multiple of 14')
        self.size = int(size)
        self.device = torch.device(device)
        self.model = torch.hub.load(repo, 'dinov2_vitb14_reg', source=hub_source)
        self.model = self.model.to(self.device).eval().requires_grad_(False)
        self.num_heads = int(self.model.num_heads)
        if self.num_heads != 12:
            raise ValueError(f'Expected 12 heads for ViT-B/14-reg, got {self.num_heads}')
        self.hook = self.model.blocks[-1].attn.qkv.register_forward_hook(get_self_attention)
        self.resize = T.Resize(self.size, interpolation=T.InterpolationMode.BICUBIC)
        self.crop = T.CenterCrop(self.size)
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def close(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    @torch.no_grad()
    def extract(self, image: Image.Image):
        # The visualized image MUST be exactly the image seen by DINO.
        cropped = self.crop(self.resize(image.convert('RGB')))
        tensor = self.normalize(cropped).unsqueeze(0).to(self.device)
        feats.pop('self_attn', None)
        output = self.model(tensor, is_training=True)
        patches = output['x_norm_patchtokens']  # B,P,768
        if 'self_attn' not in feats:
            raise RuntimeError('DINOv2 QKV hook did not run; check the installed model version')
        num_patches = (self.size // 14) ** 2
        if patches.shape != (1, num_patches, 768):
            raise RuntimeError(f'Unexpected DINO patch tensor: {tuple(patches.shape)}')
        # CLS(1) + register(4) + P patch tokens. Code uses CLS-to-patch logits.
        _, raw_logits = process_self_attention(
            feats['self_attn'], batch_size=1, num_tokens=5 + num_patches,
            num_attn_heads=self.num_heads, embed_dim=768,
            scale=(768 // self.num_heads) ** -0.5,
            num_global_tokens=5, ret_self_attn_maps=True,
        )
        heads, attention = head_pool(patches.float(), raw_logits.float())
        # No computational graph kept, and no GPU tensors saved to disk.
        feats.pop('self_attn', None)
        return heads[0].cpu(), attention[0].cpu().reshape(self.num_heads, self.size // 14, self.size // 14), cropped
