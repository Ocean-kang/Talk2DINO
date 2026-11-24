from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def torch_load(path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_concept(text):
    # Reuse Local-CoCu's plural-aware normalizer in the Talk2DINO repository.
    try:
        from experiments.local_cocu_talk2dino.common.data import normalize
        return " ".join(normalize(str(text)))
    except ImportError:
        # Lightweight fallback only for isolated unit tests outside the repository.
        import re
        irregular = {"men": "man", "women": "woman", "people": "person",
                     "children": "child", "mice": "mouse", "geese": "goose"}
        words = []
        for word in re.findall(r"[a-z0-9]+", str(text).casefold()):
            word = irregular.get(word, word)
            if len(word) > 4 and word.endswith("ies"):
                word = word[:-3] + "y"
            elif len(word) > 4 and word.endswith(("sses", "shes", "ches", "xes")):
                word = word[:-2]
            elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is", "ous")):
                word = word[:-1]
            words.append(word)
        return " ".join(words)


def resolve_image(image_root, image_meta):
    root = Path(image_root).expanduser().resolve()
    file_name = str(image_meta.get("file_name", "")).strip()
    image_id = int(image_meta["id"])
    resolved_path = str(image_meta.get("resolved_path", "")).strip()
    if resolved_path and Path(resolved_path).is_file():
        return Path(resolved_path).resolve()
    candidates = []
    if file_name:
        p = Path(file_name)
        if p.is_absolute():
            candidates.append(p)
        else:
            for base in (
                root, root / "images", root / "train2014", root / "val2014",
                root / "images" / "train2014", root / "images" / "val2014",
            ):
                candidates.append(base / file_name)
    names = (f"COCO_train2014_{image_id:012d}.jpg", f"COCO_val2014_{image_id:012d}.jpg")
    for base in (
        root, root / "images", root / "train2014", root / "val2014",
        root / "images" / "train2014", root / "images" / "val2014",
    ):
        candidates.extend(base / n for n in names)
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Cannot resolve image_id={image_id}, file_name={file_name!r} under {root}")


def ensure_clip_cached(model_name="ViT-B/16"):
    import importlib
    from pathlib import Path

    try:
        import clip
        clip_impl = importlib.import_module("clip.clip")
    except ImportError as exc:
        raise RuntimeError("OpenAI CLIP package is required.") from exc

    models = getattr(clip_impl, "_MODELS", None)

    if models is None or not hasattr(clip, "available_models"):
        raise RuntimeError(
            f"Invalid OpenAI CLIP installation: {clip.__file__}"
        )

    if model_name not in clip.available_models() or model_name not in models:
        raise ValueError(f"Unknown CLIP model: {model_name}")

    filename = models[model_name].split("/")[-1]
    weight = Path.home() / ".cache" / "clip" / filename

    if not weight.is_file():
        raise FileNotFoundError(
            f"CLIP weights are not cached: {weight}. "
            "No model download will be attempted."
        )

    return weight



def ensure_dino_hub_cached():
    import torch
    hub = Path(torch.hub.get_dir())
    matches = list(hub.glob("facebookresearch_dinov2_*"))
    if not matches:
        raise FileNotFoundError(f"DINOv2 torch.hub repository was not found under {hub}. Use the existing Talk2DINO cache.")
    return hub
