#!/usr/bin/env python3
"""Shared CLIP ViT-B/16 loading and text embedding utilities."""
from pathlib import Path
from urllib.parse import urlparse

import torch
import torch.nn.functional as F


def resolve_device(requested="auto"):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return requested



def load_clip_model(model_name="ViT-B/16", device="auto",
                    cache_dir="~/.cache/clip"):
    try:
        import clip
    except ImportError as exc:
        raise RuntimeError(
            "The existing OpenAI CLIP package is not importable."
        ) from exc

    # 1. 使用 CLIP 公开 API 检查模型
    available_models = clip.available_models()

    if model_name not in available_models:
        raise ValueError(
            "Unknown OpenAI CLIP model: {}. Available: {}".format(
                model_name, available_models
            )
        )

    # 2. 官方 CLIP 模型名称与本地缓存文件名映射
    model_filenames = {
        "RN50": "RN50.pt",
        "RN101": "RN101.pt",
        "RN50x4": "RN50x4.pt",
        "RN50x16": "RN50x16.pt",
        "RN50x64": "RN50x64.pt",
        "ViT-B/32": "ViT-B-32.pt",
        "ViT-B/16": "ViT-B-16.pt",
        "ViT-L/14": "ViT-L-14.pt",
        "ViT-L/14@336px": "ViT-L-14-336px.pt",
    }

    if model_name not in model_filenames:
        raise ValueError(
            "No local filename mapping for {}".format(model_name)
        )

    # 3. 检查本地权重
    from pathlib import Path

    cache = Path(cache_dir).expanduser()
    weight_path = cache / model_filenames[model_name]

    if not weight_path.is_file():
        raise FileNotFoundError(
            "CLIP weights are not cached: {}".format(weight_path)
        )

    print("[CLIP] Model:", model_name)
    print("[CLIP] Weights:", weight_path)

    # 4. 加载现有权重
    device = resolve_device(device)

    model, _ = clip.load(
        model_name,
        device=device,
        jit=False,
        download_root=str(cache)
    )

    model.eval()

    return clip, model, device



@torch.inference_mode()
def encode_texts(texts, clip_module, model, device, batch_size=256):
    """Return L2-normalized CLIP text embeddings on CPU, in input order."""
    if not texts:
        return torch.empty((0, 512), dtype=torch.float32)

    outputs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokens = clip_module.tokenize(batch, truncate=True).to(device)
        features = model.encode_text(tokens).float()
        features = F.normalize(features, dim=-1)
        outputs.append(features.cpu())
        print(
            "[CLIP] encoded {}/{} texts".format(
                min(start + len(batch), len(texts)), len(texts)
            ),
            flush=True,
        )
    return torch.cat(outputs, dim=0)
