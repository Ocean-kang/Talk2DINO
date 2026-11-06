"""Reuse the repository's existing MSCOCO14 gallery and phrase matcher."""
from __future__ import annotations

from pathlib import Path
import torch


def load_gallery_and_matcher(path: str | Path):
    try:
        from experiments.minimal_noun_supervision.common.nouns import (
            load_gallery,
            GalleryMatcher,
        )
    except Exception as exc:
        raise RuntimeError(
            "This experiment reuses experiments/minimal_noun_supervision/common/nouns.py "
            "and its existing local_cocu gallery helpers. Keep the repository's "
            "minimal_noun_supervision and local_cocu folders in place."
        ) from exc
    gallery, aliases = load_gallery(path)
    matcher = GalleryMatcher(gallery, morphology="exact")
    return gallery, aliases, matcher


def encode_gallery_clip_features(
    gallery: list[str],
    *,
    device: str,
    text_bank: str | Path | None = None,
    clip_checkpoint: str | Path | None = None,
    batch_size: int = 256,
) -> torch.Tensor:
    """
    Load verified raw CLIP text embeddings from an existing MNS text bank,
    or encode prompts with an explicitly supplied local CLIP checkpoint file.
    Never pass a CLIP model name, which could trigger a download.
    """
    prompts = ["a photo of " + concept for concept in gallery]
    if text_bank is not None:
        bank_path = Path(text_bank)
        if not bank_path.is_file():
            raise FileNotFoundError(f"Missing text bank: {bank_path}")
        try:
            bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        except TypeError:
            bank = torch.load(bank_path, map_location="cpu")
        if not isinstance(bank, dict) or not {"texts", "embeddings", "metadata"} <= set(bank):
            raise ValueError("Text bank must contain texts, embeddings and metadata")
        meta = bank["metadata"]
        if meta.get("clip_model") != "ViT-B/16":
            raise ValueError("Text bank is not from CLIP ViT-B/16")
        if meta.get("prompt_template") != "a photo of {}":
            raise ValueError("Text bank prompt template must be 'a photo of {}'")
        if meta.get("feature_space") != "clip_text_raw":
            raise ValueError("Text bank must contain raw CLIP text features, not projected vectors")
        if meta.get("gallery_concepts") != gallery:
            raise ValueError("Text bank gallery order/content does not match the supplied gallery")
        texts = bank["texts"]
        embeddings = torch.as_tensor(bank["embeddings"]).float()
        if embeddings.ndim != 2 or embeddings.shape != (len(texts), 512):
            raise ValueError("Text bank embeddings must have shape [num_texts,512]")
        lookup = {text: i for i, text in enumerate(texts)}
        missing = [prompt for prompt in prompts if prompt not in lookup]
        if not missing:
            return torch.stack([embeddings[lookup[p]] for p in prompts], dim=0)
        if clip_checkpoint is None:
            raise ValueError(
                f"Text bank is missing {len(missing)} gallery prompts and no local "
                "--clip-checkpoint was supplied"
            )
        # A partial bank is allowed; fill only the missing prompts locally.
        result = torch.empty((len(prompts), 512), dtype=torch.float32)
        for i, prompt in enumerate(prompts):
            if prompt in lookup:
                result[i] = embeddings[lookup[prompt]]
        encoded = _encode_local_clip(missing, clip_checkpoint, device, batch_size)
        missing_lookup = {p: i for i, p in enumerate(missing)}
        for i, prompt in enumerate(prompts):
            if prompt in missing_lookup:
                result[i] = encoded[missing_lookup[prompt]]
        return result

    if clip_checkpoint is None:
        raise ValueError(
            "Provide either --text-bank (an existing verified MNS text_bank.pth) "
            "or --clip-checkpoint (an existing local ViT-B/16 checkpoint file)."
        )
    return _encode_local_clip(prompts, clip_checkpoint, device, batch_size)


def _encode_local_clip(prompts, checkpoint, device, batch_size):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "CLIP checkpoint must be an existing local file, not a model name: "
            + str(checkpoint)
        )
    import clip

    model, _ = clip.load(str(checkpoint.resolve()), device=device, jit=False)
    if int(model.text_projection.shape[-1]) != 512:
        raise ValueError("Expected a local CLIP ViT-B/16 checkpoint (text dim 512)")
    if int(model.visual.conv1.weight.shape[-1]) != 16:
        raise ValueError("Expected a local CLIP ViT-B/16 checkpoint (patch size 16)")
    model.eval().requires_grad_(False)
    outputs = []
    for start in range(0, len(prompts), int(batch_size)):
        batch = prompts[start:start + int(batch_size)]
        tokens = clip.tokenize(batch, truncate=True).to(device)
        with torch.no_grad():
            values = model.encode_text(tokens).float().cpu()
        outputs.append(values)
    del model
    return torch.cat(outputs, dim=0)
