from __future__ import annotations

import math
import torch


def pool_pseudo_head_features(
    patch_tokens: torch.Tensor,
    propagated_scores: torch.Tensor,
    temperature: float = 0.1,
):
    """
    Convert P1-propagated Caption responses into per-head pseudo features.

    Args:
        patch_tokens: [P, D], DINOv2 x_norm_patchtokens.
        propagated_scores: [H, P, C], A_h @ seed_caption_scores.
        temperature: softmax temperature over patches.

    Returns:
        pseudo_features: [C, H, D], one 12-head feature set per Caption.
        patch_weights: [H, P, C], normalized over P for every (head, caption).

    Pooling deliberately follows Talk2DINO's original convention:
        (weights * patch_tokens).mean(patch_dim)
    rather than changing the feature scale convention to a sum.
    """
    if patch_tokens.ndim != 2:
        raise ValueError(f"patch_tokens must be [P,D], got {tuple(patch_tokens.shape)}")
    if propagated_scores.ndim != 3:
        raise ValueError(
            f"propagated_scores must be [H,P,C], got {tuple(propagated_scores.shape)}"
        )
    if propagated_scores.shape[1] != patch_tokens.shape[0]:
        raise ValueError("P1 response patch count does not match DINO patch tokens")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    patches = patch_tokens.float()
    scores = propagated_scores.float()
    patch_weights = torch.softmax(scores / float(temperature), dim=1)
    pseudo_hcd = torch.einsum("hpc,pd->hcd", patch_weights, patches)
    pseudo_hcd = pseudo_hcd / float(patches.shape[0])
    pseudo_chd = pseudo_hcd.permute(1, 0, 2).contiguous()

    if not torch.isfinite(pseudo_chd).all():
        raise FloatingPointError("Pseudo Head Features contain NaN/Inf")
    if not torch.allclose(
        patch_weights.sum(dim=1),
        torch.ones_like(patch_weights.sum(dim=1)),
        atol=1e-4,
        rtol=1e-4,
    ):
        raise RuntimeError("Patch weights are not normalized over the patch dimension")

    return pseudo_chd, patch_weights


def summarize_caption_propagation(
    seed_scores: torch.Tensor,
    propagated_scores: torch.Tensor,
    patch_weights: torch.Tensor,
    caption_index: int,
):
    """Compact, auditable per-caption/per-head diagnostics."""
    # Move this small per-image diagnostic slice to CPU once. Computing many
    # scalar .item() calls on CUDA would synchronize repeatedly over the full run.
    seed = seed_scores[:, caption_index].detach().float().cpu()
    propagated = propagated_scores[:, :, caption_index].detach().float().cpu()
    weights = patch_weights[:, :, caption_index].detach().float().cpu()

    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1)
    max_weight = weights.max(dim=-1).values
    log_patch_count = math.log(max(2, int(weights.shape[-1])))
    rows = []
    for h in range(propagated.shape[0]):
        rows.append({
            "head_id": int(h + 1),
            "propagated_mean": float(propagated[h].mean().item()),
            "propagated_std": float(propagated[h].std(unbiased=False).item()),
            "weight_entropy": float(entropy[h].item()),
            "normalized_weight_entropy": float(entropy[h].item() / log_patch_count),
            "max_patch_weight": float(max_weight[h].item()),
        })

    return {
        "seed_response": {
            "mean": float(seed.mean().item()),
            "std": float(seed.std(unbiased=False).item()),
            "min": float(seed.min().item()),
            "max": float(seed.max().item()),
        },
        "heads": rows,
    }
