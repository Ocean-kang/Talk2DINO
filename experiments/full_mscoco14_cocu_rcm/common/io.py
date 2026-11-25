from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any

import torch


def torch_load(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_jsonl(path: str | Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_torch_save(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


IRREGULAR = {
    "men": "man", "women": "woman", "people": "person",
    "children": "child", "mice": "mouse", "geese": "goose",
    "feet": "foot", "teeth": "tooth",
}
_WORD = re.compile(r"[a-z0-9]+")


def normalize_concept(text: str) -> str:
    # Prefer the exact Local-CoCu normalizer used by the repository.
    try:
        from experiments.local_cocu_talk2dino.common.data import normalize
        return " ".join(normalize(str(text)))
    except Exception:
        words = []
        for word in _WORD.findall(str(text).casefold()):
            word = IRREGULAR.get(word, word)
            if len(word) > 4 and word.endswith("ies"):
                word = word[:-3] + "y"
            elif len(word) > 4 and word.endswith(("sses", "shes", "ches", "xes")):
                word = word[:-2]
            elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is", "ous")):
                word = word[:-1]
            words.append(word)
        return " ".join(words)


def stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}::{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def resolve_image(image_root: str | Path, image_meta: dict) -> Path:
    root = Path(image_root).expanduser()
    file_name = str(image_meta.get("file_name", "")).strip()
    if not file_name:
        raise ValueError(f"Image metadata lacks file_name: {image_meta}")
    p = Path(file_name)
    candidates = [p] if p.is_absolute() else [
        root / file_name,
        root / "train2014" / file_name,
        root / "val2014" / file_name,
        root / "images" / file_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve image {file_name!r} under {root.resolve()}")


def load_concept_archive(archive_dir: str | Path) -> dict[int, dict]:
    archive_dir = Path(archive_dir)
    manifest_path = archive_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing concept archive manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    records = {}
    shard_hashes = manifest.get("shard_sha256", {})
    for shard_name in manifest.get("shards", []):
        if shard_name in shard_hashes and sha256_file(archive_dir / shard_name) != shard_hashes[shard_name]:
            raise RuntimeError(f"Concept archive shard hash mismatch: {shard_name}")
        rows = read_json(archive_dir / shard_name)
        for row in rows:
            image_id = int(row["image_id"])
            if image_id in records:
                raise ValueError(f"Duplicate image_id in concept archive: {image_id}")
            if "concepts" not in row:
                raise ValueError(f"Missing concepts field for image_id={image_id}")
            records[image_id] = row
    if len(records) != int(manifest["num_images"]):
        raise ValueError(
            f"Concept archive count mismatch: loaded={len(records)} manifest={manifest['num_images']}"
        )
    return records


def iter_shard_json(directory: str | Path):
    directory = Path(directory)
    for path in sorted(directory.glob("shard_*.json")):
        yield path, read_json(path)
