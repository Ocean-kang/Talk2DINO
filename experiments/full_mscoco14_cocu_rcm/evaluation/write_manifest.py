#!/usr/bin/env python3
"""Write a compact, machine-readable record for one completed OVSS invocation."""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path


def file_sha256(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-config", required=True)
    p.add_argument("--eval-config", required=True)
    p.add_argument("--pamr", type=int, choices=(0, 1), required=True)
    p.add_argument("--nproc-per-node", type=int, required=True)
    p.add_argument("--log", required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "command_completed",
        "arm": a.arm,
        "seed": a.seed,
        "benchmark": a.benchmark,
        "checkpoint": a.checkpoint,
        "checkpoint_sha256": file_sha256(a.checkpoint),
        "model_config": a.model_config,
        "model_config_sha256": file_sha256(a.model_config),
        "evaluation_config": a.eval_config,
        "evaluation_config_sha256": file_sha256(a.eval_config),
        "pamr": bool(a.pamr),
        "nproc_per_node": a.nproc_per_node,
        "log": a.log,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
