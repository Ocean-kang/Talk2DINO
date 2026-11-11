"""Run from the Talk2DINO checkout using its existing environment."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import uuid

from policies import SEEDS, VARIANTS

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "src/open_vocabulary_segmentation"
TASKS = {
    "voc20": ("voc", "voc"),
    "voc": ("voc_bg", "voc_bg"),
    "coco_stuff": ("stuff", "stuff"),
    "ade20k": ("ade", "ade"),
    "context59": ("context", "context"),
}
BASELINES = ("official", "old_f5")
PAIRS = {"threshold": "f5_control", "head": "threshold", "balanced": "head",
         "cf_l03": "balanced", "cf_l00": "cf_l03", "cf_l01": "cf_l03", "cf_l06": "cf_l03"}


def digest(path):
    with Path(path).open("rb") as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def launch(argv, cwd, log, **env):
    argv = [sys.executable, "-u", *map(str, argv)]
    print(f"[RUN] {log}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        stream.write(repr(argv) + "\n")
        stream.flush()
        subprocess.run(argv, cwd=cwd, env=os.environ | env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def train(args, data, destination, seed):
    destination.mkdir(parents=True, exist_ok=True)
    ckpt = destination / "model.pth"
    data_hash, val_hash, config_hash = args.train_hash, args.val_hash, digest(args.config)
    run = {"seed": seed, "train_sha256": data_hash, "val_sha256": val_hash, "config_sha256": config_hash}
    record = destination / "train.json"
    if ckpt.is_file() and record.is_file():
        if json.loads(record.read_text(encoding="utf-8")) != run | {"checkpoint_sha256": digest(ckpt)}:
            raise ValueError(f"Checkpoint provenance mismatch at {destination}")
        return ckpt
    with tempfile.TemporaryDirectory(dir=destination) as tmp:
        cwd = Path(tmp)
        (cwd / "weights").mkdir()
        cfg = "vitb_mlp_infonce.yaml"
        shutil.copyfile(args.config, cwd / cfg)
        launch([Path(__file__).with_name("seed_train.py"), seed, "--model_config", cfg,
                "--train_dataset", data, "--val_dataset", args.val,
                "--feature_name", "disentangled_self_attn", "--text_features", "ann_feats"],
               cwd, destination / "train.log", CUDA_VISIBLE_DEVICES=args.gpu,
               PYTHONHASHSEED=str(seed), CUBLAS_WORKSPACE_CONFIG=":4096:8")
        (cwd / "weights/vitb_mlp_infonce.pth").replace(ckpt)
    save(record, run | {"checkpoint_sha256": digest(ckpt)})
    return ckpt


def evaluate(args, weight, folder):
    weight = weight.resolve(strict=True)
    checkpoint_hash = digest(weight)
    name = "f5_ponytail_" + uuid.uuid4().hex
    alias = ROOT / "weights" / f"{name}.pth"
    source = ROOT / "segmentation_results" / f"{name}.json"
    try:
        # The original evaluator resolves weights/<proj_name>.pth.
        alias.symlink_to(weight)
        for task, (directory, stem) in TASKS.items():
            dest = folder / task
            dest.mkdir(parents=True, exist_ok=True)
            cfg = EVAL / "configs" / directory / f"dinotext_{stem}_vitb_mlp_infonce.yml"
            base = EVAL / "configs" / directory / f"eval_{stem}_pamr.yml"
            info = {"task": task, "checkpoint": str(weight), "checkpoint_sha256": checkpoint_hash,
                    "eval_sha256": digest(cfg), "base_sha256": digest(base)}
            metrics = dest / "metrics.json"
            if metrics.is_file():
                previous = json.loads(metrics.read_text(encoding="utf-8"))
                if all(previous.get(k) == value for k, value in info.items()) and 0 <= previous["miou"] <= 100:
                    continue
                raise ValueError(f"Evaluation provenance mismatch at {dest}")
            source.unlink(missing_ok=True)
            launch([EVAL / "main.py", "--eval", "--eval_cfg", cfg, "--eval_base_cfg", base,
                    "--output", dest, "--opts", f"model.proj_name={name}"],
                   ROOT, dest / "eval.log", CUDA_VISIBLE_DEVICES=args.gpu,
                   MASTER_ADDR="127.0.0.1", MASTER_PORT=str(args.port), RANK="0",
                   WORLD_SIZE="1", LOCAL_RANK="0", LOCAL_WORLD_SIZE="1",
                   PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
            raw = json.loads(source.read_text(encoding="utf-8"))
            miou = float(raw[task])
            if set(raw) != {task} or not 0 <= miou <= 100:
                raise ValueError(f"Invalid evaluation for {task}: {raw}")
            save(metrics, info | {"miou": miou})
    finally:
        alias.unlink(missing_ok=True)
        source.unlink(missing_ok=True)


def moments(values):
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    half = 2.776445105 * std / math.sqrt(len(SEEDS))
    return {"mean": mean, "std": std, "ci95": [mean - half, mean + half],
            "by_seed": dict(zip(SEEDS, values))}


def summarize(work, variants):
    def score(folder, task):
        path = folder / task / "metrics.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["task"] != task or not 0 <= row["miou"] <= 100:
            raise ValueError(f"Invalid result at {path}")
        return row["miou"]

    report = {}
    table = []
    for task in TASKS:
        fixed = {b: score(work / "fixed" / b, task) for b in BASELINES}
        values = {v: [score(work / v / str(s) / "eval", task) for s in SEEDS] for v in variants}
        results = {}
        for v, series in values.items():
            row = moments(series)
            row["delta_vs_fixed"] = {b: row["mean"] - value for b, value in fixed.items()}
            control = PAIRS.get(v)
            if control in values:
                row["paired"] = {"reference": control, **moments([a-b for a, b in zip(series, values[control])])}
            results[v] = row
            table.append(dict(dataset=task, variant=v, **{k: row[k] for k in ("mean", "std", "ci95")},
                              **fixed, **{f"delta_{b}": d for b, d in row["delta_vs_fixed"].items()},
                              paired_reference=control if control in values else "",
                              paired_mean=row.get("paired", {}).get("mean"),
                              paired_std=row.get("paired", {}).get("std"),
                              paired_ci95=row.get("paired", {}).get("ci95")))
        report[task] = {"fixed": fixed, "variants": results}
    save(work / "summary.json", report)
    with (work / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


def main():
    os.chdir(ROOT)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["build", "train", "eval", "summary", "all"])
    p.add_argument("--work", type=Path, default=Path("outputs/f5_ponytail/run1"))
    p.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    p.add_argument("--gpu", default="0")
    p.add_argument("--device", default="cuda")
    p.add_argument("--port", type=int, default=29601)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--floor", type=float, default=0.1)
    p.add_argument("--margin", type=float, default=0.02)
    p.add_argument("--negative-margin", type=float, default=0.05)
    for key, default in {
        "base-train": "outputs/coco2014_b14/train.pth", "val": "outputs/coco2014_b14/val.pth",
        "heads": "outputs/local_cocu/head_features/head_features.pt",
        "local-neighbors": "outputs/local_cocu_talk2dino/F_counterfactual/neighbors.json",
        "cls-neighbors": "outputs/local_cocu/retrieval/neighbors.json",
        "gallery": "experiments/local_cocu/MSCOCO14_train_gallery.json",
        "official": "weights/vitb_mlp_infonce.pth", "old-f5": None,
    }.items():
        p.add_argument("--" + key, type=Path, default=default)
    args = p.parse_args()
    args.config = ROOT / "configs/vitb_mlp_infonce.yaml"
    work = args.work.resolve()
    variants = list(dict.fromkeys(args.variants))
    if args.limit < 0 or args.batch < 1 or not all(0 <= x <= 1 for x in
                                                  (args.floor, args.margin, args.negative_margin)):
        p.error("Invalid limit, batch or threshold")
    if args.stage in ("eval", "all"):
        if args.old_f5 is None or not args.old_f5.is_file() or not args.official.is_file():
            p.error("Provide readable --old-f5 and --official checkpoint paths")
        if digest(args.old_f5) == digest(args.official):
            p.error("The two fixed baselines must use distinct checkpoints")

    if args.stage in ("build", "all"):
        work.parent.mkdir(parents=True, exist_ok=True)
        if work.exists():
            raise FileExistsError(f"Use a fresh --work for build: {work}")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        from build_data import build
        with tempfile.TemporaryDirectory(dir=work.parent) as tmp:
            staging = Path(tmp) / "build"
            staging.mkdir()
            inputs = {k: {"path": str(getattr(args, k).resolve()), "sha256": digest(getattr(args, k))}
                      for k in ("base_train", "heads", "local_neighbors", "cls_neighbors",
                                "gallery", "official", "config")}
            stats = build(args, staging, variants)
            save(staging / "build.json",
                 {"inputs": inputs, "variants": {v: VARIANTS[v] for v in variants},
                  "seeds": SEEDS, "stats": stats, "limit": args.limit, "device": args.device,
                  "floor": args.floor, "margin": args.margin, "negative_margin": args.negative_margin,
                  "dataset_sha256": {v: digest(staging / v / "train.pth") for v in variants}})
            staging.rename(work)

    if args.stage in ("train", "all"):
        args.val = args.val.resolve(strict=True)
        args.val_hash = digest(args.val)
        metadata = json.loads((work / "build.json").read_text(encoding="utf-8"))
        if digest(args.config) != metadata["inputs"]["config"]["sha256"]:
            raise ValueError("Model config differs from build config")
        for v in variants:
            data = work / v / "train.pth"
            args.train_hash = digest(data)
            if args.train_hash != metadata["dataset_sha256"][v]:
                raise ValueError(f"Training data changed since build: {data}")
            for seed in SEEDS:
                train(args, data, work / v / str(seed), seed)
    if args.stage in ("eval", "all"):
        for b, weight in zip(BASELINES, (args.official, args.old_f5)):
            evaluate(args, weight, work / "fixed" / b)
        for v in variants:
            for seed in SEEDS:
                folder = work / v / str(seed)
                evaluate(args, folder / "model.pth", folder / "eval")
    if args.stage in ("summary", "all"):
        summarize(work, variants)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
