from __future__ import annotations
import torch
import torch.nn.functional as F


def multi_positive_infonce(
    text_embeddings, positive_features, negative_features,
    positive_mask, negative_mask, temperature=0.07,
):
    """
    Supervised-contrastive multi-positive loss for one concept anchor per row.

    text_embeddings:   [B,D]
    positive_features: [B,K,D]
    negative_features: [B,M,D]
    masks:             [B,K], [B,M]

    Every positive contributes a log-probability term; positives and negatives
    share the denominator. Gradients flow through projected text embeddings.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if text_embeddings.ndim != 2:
        raise ValueError("text_embeddings must be [B,D]")
    if positive_features.ndim != 3 or negative_features.ndim != 3:
        raise ValueError("positive/negative features must be [B,K,D] / [B,M,D]")
    if positive_mask.shape != positive_features.shape[:2]:
        raise ValueError("positive_mask shape mismatch")
    if negative_mask.shape != negative_features.shape[:2]:
        raise ValueError("negative_mask shape mismatch")

    z = F.normalize(text_embeddings.float(), dim=-1)
    pos = F.normalize(positive_features.float(), dim=-1)
    neg = F.normalize(negative_features.float(), dim=-1)
    losses, pos_total, neg_total = [], 0, 0
    for i in range(z.shape[0]):
        p = pos[i][positive_mask[i].bool()]
        n = neg[i][negative_mask[i].bool()]
        if p.shape[0] == 0 or n.shape[0] == 0:
            continue
        pos_logits = torch.matmul(p, z[i]) / temperature
        neg_logits = torch.matmul(n, z[i]) / temperature
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        log_prob = F.log_softmax(logits, dim=0)
        losses.append(-log_prob[:p.shape[0]].mean())
        pos_total += int(p.shape[0])
        neg_total += int(n.shape[0])
    if not losses:
        return text_embeddings.sum() * 0.0, 0, 0, 0
    return torch.stack(losses).mean(), len(losses), pos_total, neg_total
