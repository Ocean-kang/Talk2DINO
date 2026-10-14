#!/usr/bin/env python3
"""Ponytail pipeline: original Local-CoCu -> novel filtering -> original Talk2DINO.

Always launch from the Talk2DINO checkout root (run.sh does this automatically).
No repository core module is modified, and the teacher is never used to
initialize/train the new student.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCAL = Path("experiments/local_cocu")
DEFAULT_OUT = Path("outputs/local_cocu_novel")


def abs_path(path: str | Path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def needs(path, dry):
    if not dry and not abs_path(path).exists():
        raise FileNotFoundError(f"Required input missing: {abs_path(path)}")


def execute(argv, gpu, dry):
    cmd = [sys.executable if str(argv[0]) == "python" else str(argv[0]), *map(str, argv[1:])]
    print("+", shlex.join(cmd), flush=True)
    if not dry:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("PYTHONHASHSEED", "123")
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def source_files(source):
    return {
        "cls": source / "dino_cls" / "dino_cls.pt",
        "neighbors": source / "retrieval" / "neighbors.json",
        "candidate": source / "candidate" / "candidate_concepts.json",
        "heads": source / "head_features" / "head_features.pt",
    }


def extract(args, source):
    files = source_files(source)
    needs(args.coco, args.dry_run)
    needs(args.images, args.dry_run)
    needs(args.gallery, args.dry_run)
    steps = [
        ("cls", ["python", LOCAL / "extract_dino_cls_features.py",
                 "--caption", args.coco, "--image_dir", args.images,
                 "--output", files["cls"].parent]),
        ("neighbors", ["python", HERE / "chunked_retrieval.py", "--feature", files["cls"],
                       "--topk", args.retrieval_topk, "--chunk-size", args.retrieval_chunk,
                       "--output", files["neighbors"].parent, "--device", "cuda"]),
        ("candidate", ["python", LOCAL / "extract_candidate_concepts.py",
                       "--neighbors", files["neighbors"], "--caption", args.coco,
                       "--gallery", args.gallery, "--output", files["candidate"].parent]),
        ("heads", ["python", LOCAL / "extract_head_features.py",
                   "--annotation", args.coco, "--image_dir", args.images,
                   "--model", "dinov2_vitb14_reg", "--resize_dim", 448,
                   "--crop_dim", 448, "--batch_size", args.head_batch,
                   "--output", files["heads"], "--fp16"]),
    ]
    for key, cmd in steps:
        if not args.dry_run and abs_path(files[key]).exists() and not args.force_extract:
            print(f"[CACHE] {files[key]} (use --force-extract to recompute)", flush=True)
        else:
            execute(cmd, args.gpu, args.dry_run)


def filter_stage(args, files, output):
    needs(files["candidate"], args.dry_run)
    needs(args.coco, args.dry_run)
    cmd = ["python", HERE / "filter_novel.py", "--candidates", files["candidate"],
           "--raw-captions", args.coco, "--output", output / "novel_candidates.json",
           "--audit", output / "novel_filter_audit.json"]
    if args.exact_only:
        cmd += ["--exact-only"]
    execute(cmd, args.gpu, args.dry_run)


def match_stage(args, files, output):
    needs(output / "novel_candidates.json", args.dry_run)
    needs(files["heads"], args.dry_run)
    needs(args.config, args.dry_run)
    needs(args.teacher, args.dry_run)
    cmd = ["python", LOCAL / "match_head_concepts.py",
           "--candidate", output / "novel_candidates.json",
           "--head_features", files["heads"],
           "--projector_config", args.config, "--weight", args.teacher,
           "--clip_model", "ViT-B/16", "--topk", max(args.max_novel, 10),
           "--output", output / "matched"]
    if args.threshold is not None:
        cmd.extend(["--threshold", args.threshold])
    execute(cmd, args.gpu, args.dry_run)


def build_stage(args, output):
    detail = output / "matched" / "matched_concepts_detailed.json"
    needs(detail, args.dry_run)
    needs(args.base_train, args.dry_run)
    cmd = ["python", HERE / "build_train.py", "--base-train", args.base_train,
           "--matched", detail, "--output", output / "train_local_cocu_novel_5plus3.pth",
           "--manifest", output / "train_manifest.json", "--max-novel", args.max_novel,
           "--clip-model", "ViT-B/16", "--batch-size", args.clip_batch, "--device", "cuda"]
    if args.strict:
        cmd += ["--strict"]
    if args.exact_only:
        cmd += ["--exact-only"]
    execute(cmd, args.gpu, args.dry_run)


def train_stage(args, output):
    training = output / "train_local_cocu_novel_5plus3.pth"
    needs(training, args.dry_run)
    needs(args.val, args.dry_run)
    needs(args.config, args.dry_run)
    cmd = ["python", "train.py", "--model_config", args.config,
           "--train_dataset", training, "--val_dataset", args.val,
           "--feature_name", "disentangled_self_attn", "--text_features", "ann_feats",
           "--name_pedix", "local_cocu_novel_5plus3"]
    execute(cmd, args.gpu, args.dry_run)
    print("[WEIGHTS] weights/vitb_mlp_infonce_local_cocu_novel_5plus3.pth", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["extract", "filter", "match", "build", "train", "eval", "all"], default="all")
    p.add_argument("--gpu", default="0")
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--reuse-local", action="store_true", help="Use existing outputs/local_cocu/{candidate,head_features}; skip extraction")
    p.add_argument("--source-dir", default="outputs/local_cocu", help="Used only with --reuse-local")
    p.add_argument("--base-train", default="outputs/coco2014_b14/train.pth")
    p.add_argument("--val", default="outputs/coco2014_b14/val.pth")
    p.add_argument("--coco", default="data/coco2014/annotations/captions_train2014.json")
    p.add_argument("--images", default="data/coco2014/train2014")
    p.add_argument("--gallery", default="experiments/local_cocu/MSCOCO14_train_gallery.json")
    p.add_argument("--teacher", default="weights/vitb_mlp_infonce.pth")
    p.add_argument("--config", default="configs/vitb_mlp_infonce.yaml")
    p.add_argument("--retrieval-topk", type=int, default=50)
    p.add_argument("--retrieval-chunk", type=int, default=256)
    p.add_argument("--head-batch", type=int, default=16)
    p.add_argument("--clip-batch", type=int, default=512)
    p.add_argument("--max-novel", type=int, default=3)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--strict", action="store_true", help="Require three novel concepts for every image")
    p.add_argument("--exact-only", action="store_true", help="Do not normalize common English plural nouns")
    p.add_argument("--force-extract", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.max_novel <= 0:
        p.error("--max-novel must be >0")
    output = Path(args.output_dir)
    source = Path(args.source_dir) if args.reuse_local else output / "source"
    files = source_files(source)
    if not args.dry_run and not (ROOT / "src/model.py").exists():
        raise RuntimeError("The project root was not found")
    if args.stage == "eval":
        execute(["bash", HERE / "eval.sh", args.gpu], args.gpu, args.dry_run)
        return
    if args.stage == "extract":
        if args.reuse_local:
            p.error("--stage extract is incompatible with --reuse-local")
        extract(args, source)
        return
    if args.stage == "all" and not args.reuse_local:
        extract(args, source)
    if args.stage in ("all", "filter"):
        filter_stage(args, files, output)
    if args.stage in ("all", "match"):
        match_stage(args, files, output)
    if args.stage in ("all", "build"):
        build_stage(args, output)
    if args.stage in ("all", "train"):
        train_stage(args, output)


if __name__ == "__main__":
    main()
