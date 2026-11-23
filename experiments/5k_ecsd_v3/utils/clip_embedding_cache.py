"""Incremental CLIP text embedding store shared by caption generation and metrics."""
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _normalize_rows(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _weight_signature(model_name, clip_cache_dir):
    if model_name == "ViT-B/16":
        candidates = [clip_cache_dir / "ViT-B-16.pt", clip_cache_dir / "ViT-B-16.pth"]
        for path in candidates:
            if path.is_file():
                stat = path.stat()
                return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return None


def _atomic_npz(path, texts, embeddings):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, texts=np.asarray(texts, dtype=str), embeddings=np.asarray(embeddings, dtype=np.float32))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def encode_texts_cached(texts, model_name, clip_cache_dir, embedding_cache_dir,
                        batch_size=256, device="auto", force_reencode=False):
    """Return normalized {text: vector}; encode only texts missing from the shared model store."""
    if model_name != "ViT-B/16":
        raise ValueError("ECSD v3 is configured for the existing CLIP ViT-B/16 model only; refusing other model names")
    requested = sorted(set(str(x) for x in texts))
    if not requested:
        return {}, {"cache_hit": True, "num_texts": 0, "num_new_texts": 0, "cache_path": None}
    clip_cache_dir = Path(clip_cache_dir).expanduser()
    embedding_cache_dir = Path(embedding_cache_dir).expanduser()
    embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_")
    store_path = embedding_cache_dir / "clip_text_embedding_store_{}.npz".format(slug)
    meta_path = embedding_cache_dir / "clip_text_embedding_store_{}.json".format(slug)

    existing = {}
    old_meta = {}
    if store_path.is_file() and meta_path.is_file() and not force_reencode:
        try:
            old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            with np.load(str(store_path), allow_pickle=False) as store:
                stored_texts = store["texts"].astype(str).tolist()
                stored_vectors = _normalize_rows(store["embeddings"])
            if old_meta.get("model_name") == model_name:
                existing = {text: stored_vectors[i] for i, text in enumerate(stored_texts)}
        except Exception:
            existing = {}
            old_meta = {}

    missing = requested if force_reencode else [x for x in requested if x not in existing]
    resolved_device = old_meta.get("device", "cached")
    weight_sig = _weight_signature(model_name, clip_cache_dir)

    # A changed weight file invalidates the previous store, preventing silent stale embeddings.
    old_sig = old_meta.get("weight_signature")
    if existing and old_sig and weight_sig and old_sig != weight_sig:
        existing = {}
        missing = requested

    if missing:
        candidates = [clip_cache_dir / "ViT-B-16.pt", clip_cache_dir / "ViT-B-16.pth"]
        if model_name == "ViT-B/16" and not any(p.is_file() for p in candidates):
            raise FileNotFoundError(
                "Existing CLIP ViT-B/16 weight not found under {}. ECSD v3 never downloads weights.".format(clip_cache_dir)
            )
        repo_root = Path(__file__).resolve().parents[3]
        helper_dir = repo_root / "experiments" / "5k_ecsd"
        if not helper_dir.is_dir():
            raise FileNotFoundError(
                "Expected original Talk2DINO CLIP helper directory: {}. Place this package inside Talk2DINO.".format(helper_dir)
            )
        sys.path.insert(0, str(helper_dir))
        try:
            from clip_utils import load_clip_model, encode_texts
        except Exception as exc:
            raise RuntimeError("Could not import Talk2DINO experiments/5k_ecsd/clip_utils.py: {}".format(exc))
        clip_module, model, resolved_device = load_clip_model(model_name, device, str(clip_cache_dir))
        encoded = encode_texts(missing, clip_module, model, resolved_device, batch_size=batch_size)
        new_vectors = _normalize_rows(_to_numpy(encoded))
        if new_vectors.ndim != 2 or new_vectors.shape[0] != len(missing):
            raise ValueError("CLIP encoder returned shape {}, expected ({}, D)".format(new_vectors.shape, len(missing)))
        for i, text in enumerate(missing):
            existing[text] = new_vectors[i]
        weight_sig = _weight_signature(model_name, clip_cache_dir)
        all_texts = sorted(existing)
        all_vectors = np.stack([existing[x] for x in all_texts], axis=0)
        _atomic_npz(store_path, all_texts, all_vectors)
        meta = {
            "model_name": model_name,
            "weight_signature": weight_sig,
            "device": str(resolved_device),
            "num_stored_texts": len(all_texts),
            "normalization": "L2 row normalization",
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    else:
        # Cache hit for every requested string; no CLIP import/model load is needed.
        all_texts = sorted(existing)
        all_vectors = np.stack([existing[x] for x in all_texts], axis=0) if all_texts else np.empty((0, 0), dtype=np.float32)

    return {text: existing[text] for text in requested}, {
        "cache_hit": len(missing) == 0,
        "num_texts": len(requested),
        "num_new_texts": len(missing),
        "num_stored_texts": len(existing),
        "cache_path": str(store_path),
        "metadata_path": str(meta_path),
        "device": str(resolved_device),
        "weight_signature": weight_sig,
    }
