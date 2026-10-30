"""Semi-orthogonal projection utilities for Talk2DINO."""
from __future__ import annotations

import torch


@torch.no_grad()
def project_column_orthonormal_(weight: torch.Tensor) -> None:
    """
    Project a 2-D weight matrix W onto {W : W^T W = I} using its polar factor.

    Talk2DINO's input projector weight has shape [768, 512], so its columns
    should be orthonormal. The function intentionally does not touch bias or
    any other model parameter.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2-D weight matrix, got shape={tuple(weight.shape)}")
    out_dim, in_dim = weight.shape
    if out_dim < in_dim:
        raise ValueError(
            f"Column-orthonormal projection requires out_dim >= in_dim, got {tuple(weight.shape)}"
        )

    # Keep the decomposition in FP32 for numerical stability, even if a caller
    # later decides to use lower precision for model weights.
    w32 = weight.detach().to(dtype=torch.float32)
    u, _, vh = torch.linalg.svd(w32, full_matrices=False)
    projected = u @ vh
    weight.copy_(projected.to(device=weight.device, dtype=weight.dtype))


@torch.no_grad()
def orthogonality_diagnostics(weight: torch.Tensor) -> dict:
    """Return scalar diagnostics for W^T W ~= I and the singular-value spectrum."""
    w = weight.detach().to(dtype=torch.float32)
    if w.ndim != 2:
        raise ValueError(f"Expected a 2-D weight matrix, got shape={tuple(w.shape)}")

    gram = w.T @ w
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    residual = torch.linalg.matrix_norm(gram - identity, ord="fro")
    singular_values = torch.linalg.svdvals(w)

    return {
        "weight_shape": list(w.shape),
        "orthogonality_residual_fro": float(residual.cpu().item()),
        "orthogonality_residual_normalized": float(
            (residual / (gram.shape[0] ** 0.5)).cpu().item()
        ),
        "singular_value_min": float(singular_values.min().cpu().item()),
        "singular_value_max": float(singular_values.max().cpu().item()),
        "singular_value_mean": float(singular_values.mean().cpu().item()),
    }
