"""Tensor operations for frozen Head-Patch relation diagnostics."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def extract_head_patch_relations(
    qkv_output: torch.Tensor,
    patch_tokens: torch.Tensor,
    num_heads: int,
    num_global_tokens: int = 5,
    query_index: int | None = None,
) -> dict:
    """
    Build P0/P1/P2 diagnostics from the final DINOv2 block QKV output.

    Args:
        qkv_output: [B, N, 3*D], output of the final block's attn.qkv layer.
        patch_tokens: [B, P, D], DINOv2 x_norm_patchtokens.
        num_heads: attention head count.
        num_global_tokens: CLS + register tokens (5 for ViT-B/14-reg).
        query_index: patch index used by query-conditioned visualizations.

    P0 reproduces Talk2DINO's disentangled_self_attn pooling:
        softmax(Q_CLS K_patch^T) * patch_tokens, mean over patches.

    P1 is an explicitly patch-only relation:
        softmax(Q_patch K_patch^T) over patch keys.
        It is re-normalized over patch keys, not the original all-token key set.

    P2 is head-specific patch value context:
        Z_h = A_h^P1 V_h^patch
        followed by pairwise cosine affinity between contextualized patches.
    """
    if qkv_output.ndim != 3 or patch_tokens.ndim != 3:
        raise ValueError("Expected qkv_output [B,N,3D] and patch_tokens [B,P,D]")
    batch, token_count, qkv_dim = qkv_output.shape
    if batch != 1 or patch_tokens.shape[0] != 1:
        raise ValueError("This diagnostic currently processes one image at a time")
    if qkv_dim % 3:
        raise ValueError(f"QKV dimension must be divisible by 3, got {qkv_dim}")
    embed_dim = qkv_dim // 3
    if embed_dim % num_heads:
        raise ValueError(f"embed_dim={embed_dim} is not divisible by num_heads={num_heads}")
    head_dim = embed_dim // num_heads
    patch_count = patch_tokens.shape[1]
    if token_count != num_global_tokens + patch_count:
        raise ValueError(
            f"Token mismatch: qkv has {token_count}, expected "
            f"{num_global_tokens}+{patch_count}"
        )
    if patch_tokens.shape[-1] != embed_dim:
        raise ValueError("Patch feature dimension does not match QKV embedding dimension")
    if patch_count < 2:
        raise ValueError("At least two patch tokens are required")

    qkv = qkv_output.reshape(
        batch, token_count, 3, num_heads, head_dim
    ).permute(2, 0, 3, 1, 4)
    q = qkv[0, 0] * (head_dim ** -0.5)  # [H,N,d], scaled query
    k = qkv[1, 0]                       # [H,N,d]
    v = qkv[2, 0]                       # [H,N,d]

    q_patch = q[:, num_global_tokens:, :]
    k_patch = k[:, num_global_tokens:, :]
    v_patch = v[:, num_global_tokens:, :]
    if query_index is None:
        query_index = patch_count // 2
    if not 0 <= int(query_index) < patch_count:
        raise ValueError(f"query_index={query_index} outside [0,{patch_count})")
    query_index = int(query_index)

    # P0: same CLS-to-patch logits and patch-only softmax used by the
    # repository's process_self_attention + disentangled_self_attn extraction.
    p0_logits = torch.matmul(q[:, 0:1, :], k_patch.transpose(-2, -1)).squeeze(1)
    p0 = p0_logits.softmax(dim=-1)  # [H,P]
    patches = patch_tokens[0].float()
    heads = (patches.unsqueeze(0) * p0.unsqueeze(-1)).mean(dim=1)  # [H,D]

    # P1: patch-query to patch-key affinity. Rows sum to one.
    p1_logits = torch.matmul(q_patch, k_patch.transpose(-2, -1))
    p1_affinity = p1_logits.softmax(dim=-1)  # [H,P,P]
    p1_mean_incoming = p1_affinity.mean(dim=1)  # average over query rows -> [H,P]
    p1_query = p1_affinity[:, query_index, :]    # selected query -> all patch keys

    # Measure how much probability the original full-token attention assigns
    # to patch keys. This exposes attention mass spent on CLS/register tokens.
    full_logits = torch.matmul(q_patch, k.transpose(-2, -1))
    full_probs = full_logits.softmax(dim=-1)
    p1_patch_mass = full_probs[:, :, num_global_tokens:].sum(dim=-1).mean(dim=-1)

    # P2: contextualized values and their pairwise feature affinity.
    value_context = torch.matmul(p1_affinity, v_patch)  # [H,P,d]
    context_norm = F.normalize(value_context.float(), p=2, dim=-1)
    p2_affinity = torch.matmul(context_norm, context_norm.transpose(-2, -1))
    diagonal = torch.diagonal(p2_affinity, dim1=-2, dim2=-1)
    p2_mean_nonself = (p2_affinity.sum(dim=-1) - diagonal) / max(1, patch_count - 1)
    p2_query = p2_affinity[:, query_index, :]

    return {
        "heads": heads,
        "p0_cls_to_patch": p0,
        "p1_affinity": p1_affinity,
        "p1_mean_incoming": p1_mean_incoming,
        "p1_query_conditioned": p1_query,
        "p1_patch_mass_in_full_attention": p1_patch_mass,
        "p2_value_context": value_context,
        "p2_mean_feature_affinity": p2_mean_nonself,
        "p2_query_feature_affinity": p2_query,
        "query_index": query_index,
        "num_patches": patch_count,
        "num_heads": num_heads,
        "head_dim": head_dim,
    }


def concept_patch_cosine(
    patch_tokens: torch.Tensor,
    projected_concepts: torch.Tensor,
) -> torch.Tensor:
    """
    Cosine between DINO patch tokens [P,D] and projected CLIP concepts [C,D].
    Both live in Talk2DINO's DINO feature space after W_projector.
    """
    if patch_tokens.ndim != 2 or projected_concepts.ndim != 2:
        raise ValueError("Expected patch_tokens [P,D] and projected_concepts [C,D]")
    if patch_tokens.shape[-1] != projected_concepts.shape[-1]:
        raise ValueError(
            f"Feature-space mismatch: patches={patch_tokens.shape[-1]}, "
            f"concepts={projected_concepts.shape[-1]}"
        )
    patches = F.normalize(patch_tokens.float(), p=2, dim=-1)
    concepts = F.normalize(projected_concepts.float(), p=2, dim=-1)
    return patches @ concepts.transpose(0, 1)  # [P,C]


def propagate_concept_scores(
    patch_affinity: torch.Tensor,
    patch_concept_scores: torch.Tensor,
) -> torch.Tensor:
    """
    Query-to-key propagation.

    A[h,p,q] means query patch p attends to key patch q.
    Therefore (A @ r)[p] aggregates concept evidence from keys q into query p.
    """
    if patch_affinity.ndim != 3 or patch_concept_scores.ndim != 2:
        raise ValueError("Expected affinity [H,P,P] and scores [P,C]")
    if patch_affinity.shape[-1] != patch_concept_scores.shape[0]:
        raise ValueError("Affinity key count and concept score patch count differ")
    return torch.matmul(patch_affinity, patch_concept_scores)  # [H,P,C]


def retrieve_head_concepts(
    head_features: torch.Tensor,
    projected_gallery: torch.Tensor,
    gallery: list[str],
    topk: int = 5,
) -> tuple[torch.Tensor, list[list[dict]]]:
    """Return cosine [H,G] and a readable top-k list for every head."""
    if head_features.ndim != 2 or projected_gallery.ndim != 2:
        raise ValueError("Expected heads [H,D] and projected gallery [G,D]")
    if head_features.shape[-1] != projected_gallery.shape[-1]:
        raise ValueError("Head and projected gallery dimensions differ")
    if projected_gallery.shape[0] != len(gallery):
        raise ValueError("Gallery text/feature count mismatch")
    scores = (
        F.normalize(head_features.float(), dim=-1)
        @ F.normalize(projected_gallery.float(), dim=-1).transpose(0, 1)
    )
    k = min(max(1, int(topk)), len(gallery))
    values, indices = scores.topk(k, dim=-1)
    rows = []
    for h in range(scores.shape[0]):
        rows.append([
            {
                "gallery_index": int(gid),
                "concept": gallery[int(gid)],
                "cosine": float(score),
            }
            for score, gid in zip(values[h].tolist(), indices[h].tolist())
        ])
    return scores, rows
