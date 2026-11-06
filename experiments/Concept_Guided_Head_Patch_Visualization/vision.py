"""DINOv2 ViT-B/14-reg QKV extraction for P0/P1/P2."""
from __future__ import annotations

import torch
import torchvision.transforms as T
from PIL import Image

from src.hooks import feats, get_self_attention
from .core import extract_head_patch_relations


class DinoRelationExtractor:
    def __init__(
        self,
        device: str = "cuda",
        size: int = 448,
        repo: str = "facebookresearch/dinov2",
        hub_source: str = "github",
    ):
        if size != 448:
            raise ValueError(
                "This package fixes size=448 to match the repository's "
                "Talk2DINO ViT-B/14-reg extraction protocol"
            )
        self.size = int(size)
        self.device = torch.device(device)
        self.model = torch.hub.load(repo, "dinov2_vitb14_reg", source=hub_source)
        self.model = self.model.to(self.device).eval().requires_grad_(False)
        self.num_heads = int(self.model.num_heads)
        # The repository's ViT-B/14-reg extraction contract is 12 x 64 = 768.
        # Validate the actual patch tensor against this fixed, audited setup below.
        self.embed_dim = 768
        if self.num_heads != 12:
            raise ValueError(
                "Expected DINOv2 ViT-B/14-reg with 12 heads; "
                f"got heads={self.num_heads}"
            )
        self.num_global_tokens = 5  # CLS + 4 register tokens
        self.hook = self.model.blocks[-1].attn.qkv.register_forward_hook(get_self_attention)
        self.resize = T.Resize(self.size, interpolation=T.InterpolationMode.BICUBIC)
        self.crop = T.CenterCrop(self.size)
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def close(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    @torch.no_grad()
    def extract(self, image: Image.Image, query_index: int | None = None) -> dict:
        # Keep the visualized crop identical to the tensor sent to DINO.
        cropped = self.crop(self.resize(image.convert("RGB")))
        tensor = self.normalize(cropped).unsqueeze(0).to(self.device)
        feats.pop("self_attn", None)
        outputs = self.model(tensor, is_training=True)
        if "self_attn" not in feats:
            raise RuntimeError("Final-block QKV hook did not run")
        patches = outputs["x_norm_patchtokens"]
        grid = self.size // 14
        expected_patches = grid * grid
        if tuple(patches.shape) != (1, expected_patches, self.embed_dim):
            raise RuntimeError(
                f"Unexpected DINO patch shape {tuple(patches.shape)}; "
                f"expected (1,{expected_patches},{self.embed_dim})"
            )
        relations = extract_head_patch_relations(
            feats["self_attn"],
            patches,
            num_heads=self.num_heads,
            num_global_tokens=self.num_global_tokens,
            query_index=query_index,
        )
        # Do not leave a GPU tensor in the repository's shared hook dictionary.
        feats.pop("self_attn", None)
        relations["patch_tokens"] = patches[0].float()
        relations["crop"] = cropped
        relations["grid_size"] = grid
        return relations
