# Teacher-Guided Caption–Head Selection Bootstrap

Side-car experiment for the current `Ocean-kang/Talk2DINO` repository. The experiment is intentionally isolated under `experiments/teacher_guided_bootstrap/`; no file in `src/`, `configs/`, or the root training pipeline needs to be edited.

## What changes

Original Talk2DINO `ProjectionLayer(alignment_strategy="max_score")` computes, for each caption, a similarity over the DINO attention-head features and takes a hard `argmax` head. This experiment keeps the same mechanism but replaces only the score used by that `argmax` during **Student training**:

```text
student_head_score = softmax(cos(Student(caption), DINO_heads))
teacher_head_score = softmax(cos(Teacher(caption), DINO_heads))
selection_score    = (1 - alpha) * student_head_score
                   + alpha       * teacher_head_score
selected_head      = argmax(selection_score)
```

After `selected_head` is chosen, the original Student batch similarity matrix and the repository's existing InfoNCE loss are used unchanged. Teacher logits are not a target and no KD/KL/bootstrap loss is added.

Important invariants:

- Student is randomly initialized from the original model config. Teacher weights are **not** copied into Student.
- Student is initialized immediately after `set_seed(123)`, before Teacher construction/loading, so `alpha=0` remains a clean baseline.
- Teacher is frozen and used only for training-time head selection.
- `model.eval()` uses the original Student-only forward. Teacher is not needed at validation/inference time.
- Saved `.pth` contains only original `ProjectionLayer` Student keys, so existing segmentation evaluation can load it normally.
- Caption dataset/sampling, DINO features, CLIP `ann_feats`, optimizer, scheduler, batch size, epoch count, validation and InfoNCE all come from the existing repository implementation/config.

## Directory

Copy this folder to:

```text
Talk2DINO/
└── experiments/
    └── teacher_guided_bootstrap/
        ├── bootstrap_model.py
        ├── train_bootstrap.py
        ├── smoke_test.py
        ├── run_ablation.sh
        └── README.md
```

Run all commands from the **Talk2DINO repository root**.

## Existing inputs

This experiment assumes you already have the normal pre-extracted Talk2DINO files, for example:

```text
outputs/coco2014_b14/train.pth
outputs/coco2014_b14/val.pth
```

Training must contain `images[*].disentangled_self_attn` and `annotations[*].ann_feats`. Validation follows the original root `train.py` behavior and reads `avg_self_attn_out` plus `ann_feats`.

You also need the first trained Talk2DINO projector, for example:

```text
weights/vitb_mlp_infonce.pth
```

## 1. Smoke test

```bash
python experiments/teacher_guided_bootstrap/smoke_test.py
```

It checks three experiment invariants:

```text
alpha=0 training forward == original ProjectionLayer forward
Teacher keys do not appear in Student state_dict
eval/inference forward == original Student-only forward
```

## 2. Train one bootstrap Student

Recommended first run (`alpha=0.3`):

```bash
CUDA_VISIBLE_DEVICES=0 \
python experiments/teacher_guided_bootstrap/train_bootstrap.py \
  --teacher-weights weights/vitb_mlp_infonce.pth \
  --train-dataset outputs/coco2014_b14/train.pth \
  --val-dataset outputs/coco2014_b14/val.pth \
  --model-config configs/vitb_mlp_infonce.yaml \
  --alpha 0.3
```

With the current ViT-B config, the training settings remain the repository values: Adam, lr `1e-4`, 100 epochs, batch size 128, `save_best_model: false`.

Default outputs:

```text
weights/vitb_mlp_infonce_teacher_bootstrap_a0p3.pth
weights/vitb_mlp_infonce_teacher_bootstrap_a0p3.metrics.csv
weights/vitb_mlp_infonce_teacher_bootstrap_a0p3.manifest.json
```

The `.pth` is a normal Student `ProjectionLayer` checkpoint. Teacher is not serialized.

## 3. Clean alpha ablation

The proposed first ablation is:

```text
0.0, 0.1, 0.3, 0.5, 0.7, 1.0
```

Run all sequentially on one GPU:

```bash
bash experiments/teacher_guided_bootstrap/run_ablation.sh \
  0 \
  weights/vitb_mlp_infonce.pth \
  outputs/coco2014_b14/train.pth \
  outputs/coco2014_b14/val.pth
```

Expected checkpoints:

```text
weights/vitb_mlp_infonce_teacher_bootstrap_a0.pth
weights/vitb_mlp_infonce_teacher_bootstrap_a0p1.pth
weights/vitb_mlp_infonce_teacher_bootstrap_a0p3.pth
weights/vitb_mlp_infonce_teacher_bootstrap_a0p5.pth
weights/vitb_mlp_infonce_teacher_bootstrap_a0p7.pth
weights/vitb_mlp_infonce_teacher_bootstrap_a1.pth
```

`alpha=0` is implemented as a literal call to the repository's original `ProjectionLayer.forward`, not as an approximate reimplementation.

## 4. Use another Teacher

Pass the path explicitly; no code change is needed:

```bash
CUDA_VISIBLE_DEVICES=1 \
python experiments/teacher_guided_bootstrap/train_bootstrap.py \
  --teacher-weights weights/YOUR_FIRST_TALK2DINO.pth \
  --train-dataset outputs/coco2014_b14/train.pth \
  --val-dataset outputs/coco2014_b14/val.pth \
  --alpha 0.3
```

Teacher and Student use the same architecture config. Student still starts from a fresh random initialization.

## 5. Segmentation evaluation

No special bootstrap model is required for evaluation. Use the repository's existing evaluation command and point `MODEL.WEIGHTS` to the generated Student checkpoint, because its state dict is identical in structure to the original `ProjectionLayer`.

For example, substitute:

```text
MODEL.WEIGHTS=weights/vitb_mlp_infonce_teacher_bootstrap_a0p3.pth
```

into the same VOC20 / Context59 / ADE20K / COCO-Stuff commands you already use.

## Files and responsibilities

`bootstrap_model.py` contains the only algorithmic change: the training-time hard head-selection score. `train_bootstrap.py` mirrors the root training setup and delegates optimizer/scheduler/loss/training loops to `src.train_util.do_train`. `run_ablation.sh` only launches the alpha sweep. `smoke_test.py` guards the clean-baseline and checkpoint-compatibility invariants.
