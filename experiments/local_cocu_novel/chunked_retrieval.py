#!/usr/bin/env python3
"""Memory-bounded exact cosine Top-K, replacing Local-CoCu's N x N allocation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F


def run(feature: Path, out_dir: Path, topk: int, chunk: int, device: str):
    if topk < 1 or chunk < 1:
        raise ValueError("topk and chunk must be positive")
    obj = torch.load(feature, map_location="cpu")
    ids = list(obj)
    if len(ids) < topk + 1:
        raise ValueError(f"Need at least {topk+1} images for topk={topk}")
    x = torch.stack([torch.as_tensor(obj[i]).float().reshape(-1) for i in ids])
    if not torch.isfinite(x).all() or torch.any(x.norm(dim=-1) == 0):
        raise ValueError("CLS vectors contain NaN/Inf/zero norm")
    x = F.normalize(x, dim=-1)
    x = x.to(device)
    result = {}
    with torch.no_grad():
        for a in range(0, len(ids), chunk):
            b = min(a + chunk, len(ids))
            scores = x[a:b] @ x.T
            scores[torch.arange(b-a, device=x.device), torch.arange(a, b, device=x.device)] = -float("inf")
            closest = scores.topk(topk, dim=1).indices.cpu().tolist()
            for src, row in zip(ids[a:b], closest):
                result[str(src)] = [int(ids[j]) for j in row]
            if a == 0 or (a // chunk) % 20 == 0:
                print(f"[RETRIEVAL] {b}/{len(ids)}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "neighbors.json"
    out.write_text(json.dumps(result), encoding="utf-8")
    print("[OUTPUT]", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    run(Path(a.feature), Path(a.output), a.topk, a.chunk_size, a.device)


if __name__ == "__main__":
    main()
