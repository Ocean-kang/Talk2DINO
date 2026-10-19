#!/usr/bin/env python3
"""Ponytail pipeline: build PTH -> unmodified Talk2DINO train.py -> 5 original segmentation evaluations."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
THIS = Path(__file__).resolve().parent

SCHEMES = {
    "A": "A_direct_retrieval", "B": "B_head_image_retrieval",
    "C": "C_uncovered_heads", "D": "D_region_discovery",
    "E": "E_cross_image_consistency", "F": "F_counterfactual",
    "G": "G_multiview_agreement", "H": "H_iterative_bootstrap",
}
# Explicitly use five different evaluation tasks; voc uses VOC with background.
CFG = "src/open_vocabulary_segmentation/configs"
BENCHMARKS = {
    "voc20": ("voc/dinotext_voc_vitb_mlp_infonce.yml", "voc/eval_voc_pamr.yml", 88.54),
    "voc": ("voc_bg/dinotext_voc_bg_vitb_mlp_infonce.yml", "voc_bg/eval_voc_bg_pamr.yml", 65.77),
    "coco_stuff": ("stuff/dinotext_stuff_vitb_mlp_infonce.yml", "stuff/eval_stuff_pamr.yml", 30.23),
    "ade20k": ("ade/dinotext_ade_vitb_mlp_infonce.yml", "ade/eval_ade_pamr.yml", 22.53),
    "context59": ("context/dinotext_context_vitb_mlp_infonce.yml", "context/eval_context_pamr.yml", 42.42),
}


def run(cmd: list[str], gpu: str, log: Path | None = None) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTHONHASHSEED", "123")
    env.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29501", "RANK": "0",
                "WORLD_SIZE": "1", "LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1"})
    args = [str(x) for x in cmd]
    print("[CMD]", " ".join(args), flush=True)
    if log is None:
        subprocess.run(args, cwd=ROOT, env=env, check=True)
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as output:
            proc = subprocess.Popen(args, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                output.write(line)
                print(line, end="", flush=True)
            code = proc.wait()
            if code:
                raise subprocess.CalledProcessError(code, args)


def relative(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def check(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def source_for(args: argparse.Namespace) -> Path:
    if args.concepts:
        return check(relative(args.concepts))
    base = ROOT / "outputs/local_cocu_talk2dino" / SCHEMES[args.scheme]
    if args.scheme == "H":
        base = base / f"round_{args.round}"
    return check(base / "concepts_by_image.json")


def paths(args: argparse.Namespace) -> tuple[Path, Path, Path, str]:
    round_dir = f"round_{args.round}" if args.scheme == "H" else None
    work = ROOT / "outputs/local_cocu_talk2dino_new_train" / SCHEMES[args.scheme]
    weights_dir = ROOT / "weights/local_cocu_talk2dino_new_train" / SCHEMES[args.scheme]
    if round_dir:
        work = work / round_dir
        weights_dir = weights_dir / round_dir
    work = work / args.mode
    weights_dir = weights_dir / args.mode
    suffix = f"_r{args.round}" if args.scheme == "H" else ""
    proj = f"vitb_mlp_infonce_local_cocu_new_{args.scheme}_{args.mode}{suffix}"
    return work, weights_dir / (proj + ".pth"), ROOT / "weights" / (proj + ".pth"), proj


def link_checkpoint(target: Path, temporary: Path) -> None:
    check(target)
    if temporary.is_symlink():
        if temporary.resolve() != target.resolve():
            raise RuntimeError(f"Conflicting checkpoint symlink: {temporary}")
        return
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite non-symlink checkpoint: {temporary}")
    temporary.symlink_to(os.path.relpath(target, temporary.parent))


def build(args: argparse.Namespace, work: Path) -> None:
    output = work / "train_augmented.pth"
    source = source_for(args)
    if output.exists() and not args.force_build:
        manifest = work / "train_manifest.json"
        if not manifest.is_file():
            raise RuntimeError(f"Training PTH exists without its manifest: {output}; use --force-build")
        meta = json.loads(manifest.read_text(encoding="utf-8"))["meta"]
        if (Path(meta["concepts_source"]).resolve() != source.resolve()
                or meta["max_concepts_per_image"] != args.max_concepts
                or meta["missing_policy"] != args.missing_policy
                or meta["mode"] != args.mode
                or output.stat().st_mtime + 2 < source.stat().st_mtime):
            raise RuntimeError(f"Cached build does not match current inputs: {output}; use --force-build")
        print("[SKIP BUILD]", output)
        return
    check(relative(args.base_train))
    work.mkdir(parents=True, exist_ok=True)
    run([sys.executable, THIS / "build_train.py", "--base-train", relative(args.base_train),
         "--concepts", source, "--output", output, "--manifest", work / "train_manifest.json",
         "--mode", args.mode, "--max-concepts", str(args.max_concepts),
         "--batch-size", str(args.batch_size), "--device", args.device,
         "--missing-policy", args.missing_policy], args.gpu, work / "build.log")
    check(output)


def train(args: argparse.Namespace, work: Path, weight: Path, link: Path, proj: str) -> None:
    if weight.exists() and not args.force_train:
        if (work / "train_augmented.pth").is_file() and (work / "train_augmented.pth").stat().st_mtime > weight.stat().st_mtime + 2:
            raise RuntimeError(f"Train data are newer than cached checkpoint {weight}; use --force-train")
        print("[SKIP TRAIN]", weight)
        link_checkpoint(weight, link)
        return
    check(work / "train_augmented.pth")
    check(relative(args.val))
    check(relative(args.config))
    if link.is_symlink():
        if link.resolve() != weight.resolve():
            raise RuntimeError(f"Refusing to remove foreign symlink: {link}")
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"Unexpected pre-existing weights/{proj}.pth; refusing overwrite")
    if weight.exists():
        if not args.force_train:
            raise FileExistsError(weight)
        weight.unlink()  # explicitly requested replacement, own experiment path only
    weight.parent.mkdir(parents=True, exist_ok=True)
    # Original model config, training function, InfoNCE and original Student initialization.
    run([sys.executable, "train.py", "--model_config", args.config,
         "--train_dataset", work / "train_augmented.pth", "--val_dataset", relative(args.val),
         "--feature_name", "disentangled_self_attn", "--text_features", "ann_feats",
         "--name_pedix", proj[len("vitb_mlp_infonce_"):]], args.gpu, work / "train.log")
    check(link)  # original train.py writes the checkpoint in weights/
    shutil.move(str(link), str(weight))
    link_checkpoint(weight, link)
    print("[SAVED]", weight)


def read_result(source: Path, task: str) -> float:
    obj = json.loads(source.read_text(encoding="utf-8"))
    if task not in obj:
        raise RuntimeError(f"Expected {task!r} in {source}, got keys {list(obj)}")
    val = float(obj[task])
    # The original repository stores mIoU in percentage points (normally 0..100).
    if not (0 <= val <= 100):
        raise RuntimeError(f"Invalid mIoU {val} for {task}")
    return val


def evaluate(args: argparse.Namespace, work: Path, weight: Path, link: Path, proj: str) -> None:
    check(weight)
    link_checkpoint(weight, link)
    evaldir = work / "eval"
    evaldir.mkdir(parents=True, exist_ok=True)
    results_path = evaldir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.is_file() else {}
    for i, task in enumerate(args.tasks.split(",")):
        task = task.strip()
        if task not in BENCHMARKS:
            raise ValueError(f"Unknown task: {task}")
        e, b, baseline = BENCHMARKS[task]
        dest = evaldir / task
        dest.mkdir(parents=True, exist_ok=True)
        copied = dest / "metrics.json"
        if copied.is_file() and not args.force_eval and copied.stat().st_mtime + 2 >= weight.stat().st_mtime:
            print("[SKIP EVAL]", task, copied)
            value = read_result(copied, task)
        else:
            started = time.time()
            run([sys.executable, "src/open_vocabulary_segmentation/main.py",
                 "--eval", "--eval_cfg", f"{CFG}/{e}", "--eval_base_cfg", f"{CFG}/{b}",
                 "--output", dest, "--opts", f"model.proj_name={proj}"],
                args.gpu, dest / "run.log")
            # main.py appends each task into segmentation_results/<proj>.json
            produced = ROOT / "segmentation_results" / f"{proj}.json"
            check(produced)
            if produced.stat().st_mtime < started - 2:
                raise RuntimeError(f"Segmentation result not updated for {task}: {produced}")
            value = read_result(produced, task)
            shutil.copy2(produced, copied)
        results[task] = {"miou": round(value, 4), "baseline": baseline,
                         "delta": round(value - baseline, 4),
                         "eval_cfg": f"{CFG}/{e}", "eval_base_cfg": f"{CFG}/{b}",
                         "checkpoint": str(weight.relative_to(ROOT))}
        results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[RESULT] {task} mIoU={value:.2f} baseline={baseline:.2f} delta={value-baseline:+.2f}")
    print("[SUMMARY]", results_path)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scheme", required=True, choices=list(SCHEMES))
    p.add_argument("--mode", required=True, choices=["5plus1", "5"])
    p.add_argument("--stage", choices=["build", "train", "eval", "all"], default="all")
    p.add_argument("--gpu", default="0", help="Physical CUDA_VISIBLE_DEVICES value, e.g. 1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--concepts", default=None, help="Explicit path to existing concepts_by_image.json")
    p.add_argument("--round", type=int, default=1, help="For H: the pre-extracted concept round")
    p.add_argument("--base-train", default="outputs/coco2014_b14/train.pth")
    p.add_argument("--val", default="outputs/coco2014_b14/val.pth")
    p.add_argument("--config", default="configs/vitb_mlp_infonce.yaml")
    p.add_argument("--max-concepts", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--missing-policy", choices=["error", "keep_raw", "subset"], default="error")
    p.add_argument("--tasks", default=",".join(BENCHMARKS))
    p.add_argument("--force-build", action="store_true")
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--force-eval", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.max_concepts < 1 or args.batch_size < 1 or args.round < 1:
        raise SystemExit("Bad max-concepts/batch-size/round")
    work, weight, link, proj = paths(args)
    if args.dry_run:
        print("[PLAN] mode", args.mode, "scheme", args.scheme, "stage", args.stage)
        print("[PLAN] concepts", args.concepts or str(ROOT / "outputs/local_cocu_talk2dino" / SCHEMES[args.scheme] / (f"round_{args.round}" if args.scheme == "H" else "") / "concepts_by_image.json"))
        print("[PLAN] train pth", work / "train_augmented.pth")
        print("[PLAN] output weight", weight)
        print("[PLAN] benchmarks", args.tasks)
        return
    if args.stage in ("build", "all"):
        build(args, work)
    if args.stage in ("train", "all"):
        train(args, work, weight, link, proj)
    if args.stage in ("eval", "all"):
        evaluate(args, work, weight, link, proj)
    print("[DONE]", args.scheme, args.mode, args.stage)


if __name__ == "__main__":
    main()
