# MSCOCO14 5K Oracle + Teacher-Guided Bootstrap (Original Talk2DINO Training Setting)

**Target repository:** https://github.com/Ocean-kang/Talk2DINO

This is an isolated `experiments/oracle_bootstrap_5k/` Ponytail/side-car experiment.
It reuses existing `experiments/teacher_guided_bootstrap/train_bootstrap.py` and `bootstrap_model.py` directly. No file under `src/`, `configs/` or `experiments/raw_rich_5k/` needs to be modified.

## Goal and 4 arms

Train random-initialized Students on the same 5K Oracle Raw/Rich caption features while using a **Full-COCO-trained Teacher** (`weights/vitb_mlp_infonce.pth`) for training-time head selection at alpha=1.

| Run | Data | alpha | Output |
|---|---|---:|---|
| Raw baseline | Raw 5 captions per image | 0 | `weights/oracle_bootstrap_5k/raw_a0.pth` |
| Rich baseline | Rich 5 captions per image | 0 | `weights/oracle_bootstrap_5k/rich_a0.pth` |
| Raw bootstrap | Raw 5 captions per image | 1 | `weights/oracle_bootstrap_5k/raw_a1.pth` |
| Rich bootstrap | Rich 5 captions per image | 1 | `weights/oracle_bootstrap_5k/rich_a1.pth` |

- 5K images, same 4,500 training and 500 validation image IDs as the existing Oracle experiment.
- ViT-B/14-register DINOv2 448px visual features; CLIP ViT-B/16 text `ann_feats`.
- Train: `disentangled_self_attn`; validation: `avg_self_attn_out`.
- **Original Talk2DINO `configs/vitb_mlp_infonce.yaml`: 100 epochs, batch size 128, lr 1e-4, InfoNCE, save_best_model=false**.
- Adam, `--scheduler linear --warmup 0` = no scheduled decay; seed 123.
- Same original Talk2DINO `DinoClipDataset` and `do_train` caption-level sampling, **not** the 5K `train_pair.py` unique-image mini-batches / 50 epochs / best checkpoint path. Thus the original caption-level sampler may include multiple captions of the same image in one batch, as in the original training protocol.
- Student starts from a fresh random initialization for every arm; Teacher is frozen and **never copied to Student**.
- `alpha=0` is an exact pass-through to the original `ProjectionLayer.forward`, not an approximate rewrite.
- At `alpha=1`, Teacher-only head scores pick `argmax` during **training**. Student alone computes the original InfoNCE matrix; no KD/extra losses. Evaluation is always Student-only.
- Only the final Student `.pth` is saved (not the validation-loss best epoch). Each arm also outputs `.metrics.csv` and `.manifest.json`.

> Because the Teacher was trained on full COCO, this is a **5K Student + Full-COCO Teacher** experiment, **not** strict 5K-only supervision. The alpha=0 baselines do not use Teacher scores but the script still loads its weight.

## 1. Install the ZIP

The ZIP contains a single top-level `oracle_bootstrap_5k/` directory. From your Talk2DINO repo root:

```bash
cd /home/master/code/oymk/Talking2dino/Talk2DINO
# Activate your existing Talk2DINO Python environment first.
unzip oracle_bootstrap_5k.zip -d experiments/
```

Expected paths:

```text
Talk2DINO/
  experiments/
    raw_rich_5k/
      work/raw_train.pth
      work/raw_val.pth
      work/rich_train.pth
      work/rich_val.pth
      work/val_ids.json
    teacher_guided_bootstrap/
      bootstrap_model.py
      train_bootstrap.py
    oracle_bootstrap_5k/
      README.md
      check_inputs.py
      run_train.sh
      run_all.sh
      eval_all.sh
  weights/vitb_mlp_infonce.pth
  configs/vitb_mlp_infonce.yaml
```

The ZIP contains source/scripts only, **not** the 5K data, pretrained model, or checkpoints. It expects the already prepared 5K raw/rich `.pth` with DINO features and CLIP text `ann_feats`. If these are missing, first run the existing `experiments/raw_rich_5k/prepare.py` pipeline plus the existing DINO/CLIP feature extractors using your previous 5K caption JSON.

## 2. Validate existing data and settings

```bash
python experiments/oracle_bootstrap_5k/check_inputs.py
```

Checks:
- 4,500/500 train/val image IDs, identical IDs for Raw/Rich and no overlap;
- exactly five CLIP-feature captions per image;
- required DINO train/val feature keys;
- Teacher and trainer files exist;
- original `configs/vitb_mlp_infonce.yaml` settings match.

An alternative work path can be passed via `--work /path/to/5k/work`.

## 3. Run all four training arms (sequential, GPU 1)

```bash
bash experiments/oracle_bootstrap_5k/run_all.sh 1
```

Or train **one arm**:

```bash
# Rich + Bootstrap alpha=1.0 on GPU 1
bash experiments/oracle_bootstrap_5k/run_train.sh 1 rich 1.0

# Raw + Bootstrap alpha=1.0
bash experiments/oracle_bootstrap_5k/run_train.sh 1 raw 1.0

# Raw original selection alpha=0
bash experiments/oracle_bootstrap_5k/run_train.sh 1 raw 0.0

# Rich original selection alpha=0
bash experiments/oracle_bootstrap_5k/run_train.sh 1 rich 0.0
```

Optional alpha sweep supported, e.g. `run_train.sh 1 rich 0.3` -> `rich_a0p3.pth`.

If your `.pth` data is stored elsewhere:

```bash
bash experiments/oracle_bootstrap_5k/run_train.sh 1 rich 1.0 /path/to/work
```

Safety behavior: existing output weights are not overwritten by default. Set `OVERWRITE=1` if you deliberately want to rerun and replace them:

```bash
OVERWRITE=1 bash experiments/oracle_bootstrap_5k/run_train.sh 1 rich 1.0
```

Dry-run commands without training:

```bash
DRY_RUN=1 bash experiments/oracle_bootstrap_5k/run_all.sh 1
```

## 4. Evaluate open-vocabulary segmentation

First ensure each Student checkpoint exists under `weights/oracle_bootstrap_5k/`.

```bash
# Evaluate all 4 weights x VOC20, Context59, ADE20K, COCO-Stuff on GPU 1
bash experiments/oracle_bootstrap_5k/eval_all.sh 1

# Evaluate Rich Bootstrap only on all four datasets
bash experiments/oracle_bootstrap_5k/eval_all.sh 1 rich_a1

# Evaluate Rich Bootstrap on ADE20K only
bash experiments/oracle_bootstrap_5k/eval_all.sh 1 rich_a1 ade20k
```

The eval script reuses exactly the four dataset config file pairs from the existing `experiments/CoCu_Bootstrap_Talk2dino/eval_all.sh` and always passes `--opts model.proj_name=oracle_bootstrap_5k/<run>` (rather than only `MODEL.WEIGHTS`). It creates `segmentation_results/oracle_bootstrap_5k/` so nested result file writes do not fail. It uses ports 29501--29504; adjust with `PORT_BASE=29600` if necessary.

Checkpoints and result folders:

```text
weights/oracle_bootstrap_5k/raw_a0.pth
weights/oracle_bootstrap_5k/rich_a0.pth
weights/oracle_bootstrap_5k/raw_a1.pth
weights/oracle_bootstrap_5k/rich_a1.pth

output/eval/oracle_bootstrap_5k/<run>/voc20/
output/eval/oracle_bootstrap_5k/<run>/context59/
output/eval/oracle_bootstrap_5k/<run>/ade20k/
output/eval/oracle_bootstrap_5k/<run>/coco_stuff/
segmentation_results/oracle_bootstrap_5k/
```

## 5. Interpretation

Compare Raw vs Rich at the same alpha, and bootstrap gains within each caption arm:

```
raw_bootstrap_gain  = mIoU(raw_a1)  - mIoU(raw_a0)
rich_bootstrap_gain = mIoU(rich_a1) - mIoU(rich_a0)
interaction         = rich_bootstrap_gain - raw_bootstrap_gain
```

Use matching segmentation configs and random seeds. This experiment can detect whether the Full-COCO Teacher helps a 5K Student more on Rich than on Raw captions; it does not establish that any gain comes exclusively from 5K annotations, because Teacher has seen the full COCO training corpus.

## Dependencies and limitations

Run from the repo root with the existing Talk2DINO environment (`torch`, `clip`, `yaml`, repository training dependencies and CUDA). `check_inputs.py` loads the four feature files into CPU memory sequentially. Training and real evaluation require the user's environment and data, and were **not** run when packaging the scripts.
