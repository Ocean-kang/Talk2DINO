# Talk2DINO Semi-Orthogonal Projector (minimal adapter)

## 1. Design principles

- **First principles:** constrain the core linear map from CLIP text embeddings (512-D) to the DINOv2 embedding space (768-D), rather than changing the loss or the whole model.
- **Ponytail / minimal intrusion:** do not edit `train.py`, `src/model.py`, `src/train_util.py`, the original YAML, or the segmentation implementation. The adapter calls the repository's original `train.py` and hooks the original `optimizer.step()` at runtime.
- **Controlled variable:** by default, only `ProjectionLayer.linear_layer.weight` is constrained. Bias, Tanh, hidden layer, InfoNCE, batching, optimizer, scheduler, epoch count, and evaluation code are unchanged.
- **Baseline support:** the same adapter can run with `T2DINO_SO_ENABLE=0`. This disables the orthogonal intervention while retaining the original training loop and permits seed-matched comparisons. For seed 123, the random seed matches the repository's original default.

## 2. Constraint

The standard `configs/vitb_mlp_infonce.yaml` uses a `ProjectionLayer` with a core weight matrix

`W = model.linear_layer.weight`, shape `[768, 512]`.

We enforce column orthonormality:

`W.T @ W = I_512`.

After model initialization (and after any configured starting checkpoint is loaded), and after every original Adam optimizer step, the adapter computes the thin SVD:

`W_raw = U @ diag(S) @ Vh`

and projects the weight to its polar factor:

`W <- U @ Vh`.

The implementation does not alter bias or hidden-layer weights. This is a hard projection, not a soft orthogonality penalty. It adds SVD overhead to each optimizer step and leaves Adam's internal moments unprojected; this is a deliberately minimal projected-Adam baseline, not a full Riemannian optimizer.

## 3. Placement

Unzip the archive into the root of the cloned `Talk2DINO` repository. The final path should be:

```text
Talk2DINO/
└── experiments/
    └── projector_semi_orthogonal/
        ├── README.md
        ├── orthogonal.py
        ├── train_semi_orthogonal.py
        ├── run_seeds.sh
        ├── run_segmentation_eval.sh
        └── run_retrieval_eval.sh
```

From the repository root:

```bash
unzip Talk2DINO_projector_semi_orthogonal.zip -d experiments
chmod +x experiments/projector_semi_orthogonal/*.sh
```

The ZIP contains the folder `projector_semi_orthogonal/`, so extracting with `-d experiments` creates the intended path.

## 4. Single-seed training (recommended smoke test)

Run from the Talk2DINO repository root:

```bash
T2DINO_SO_SEED=123 T2DINO_SO_ENABLE=1 \
python experiments/projector_semi_orthogonal/train_semi_orthogonal.py \
  --model_config configs/vitb_mlp_infonce.yaml \
  --train_dataset ../coco2014_b14/train.pth \
  --val_dataset ../coco2014_b14/val.pth \
  --test_dataset ../coco2014_b14/test.pth \
  --feature_name disentangled_self_attn \
  --text_features ann_feats \
  --name_pedix so_seed123
```

Important:
- Use the exact train/val feature files and feature-name arguments used by your existing original Talk2DINO run. The paths above follow the repository README's `../coco2014_b14/` example; replace only the dataset paths if your files are elsewhere.
- The current repository's CLI argument is `--model_config` (not `--model`).
- `--name_pedix` is required and must be non-empty so this experiment cannot overwrite `weights/vitb_mlp_infonce.pth`.
- Model training parameters are read from `configs/vitb_mlp_infonce.yaml`; this adapter does not override them.
- The original training script saves to `weights/vitb_mlp_infonce_so_seed123.pth`.
- A diagnostic JSON is written to `outputs/projector_semi_orthogonal/so_seed123_orthogonality.json`.

If your original run used another `--feature_name` (for example, a different visual feature field), use that same value. Do not change the feature field solely for this experiment.

## 5. Five-seed experiment

The seeds are `[123, 42, 2024, 2025, 3407]`. The script supports three modes:

```bash
# Train only semi-orthogonal runs
bash experiments/projector_semi_orthogonal/run_seeds.sh so

# Train only seed-matched baselines (no orthogonal projection)
bash experiments/projector_semi_orthogonal/run_seeds.sh baseline

# Train both conditions for each seed (10 runs total; strongest comparison)
bash experiments/projector_semi_orthogonal/run_seeds.sh both
```

To supply non-default data paths, set environment variables before the command:

```bash
TRAIN_DATASET=/path/to/train.pth \
VAL_DATASET=/path/to/val.pth \
TEST_DATASET=/path/to/test.pth \
FEATURE_NAME=disentangled_self_attn \
TEXT_FEATURES=ann_feats \
bash experiments/projector_semi_orthogonal/run_seeds.sh both
```

Each run writes a separate checkpoint under `weights/` and a JSON diagnostic under `outputs/projector_semi_orthogonal/`. Use the same seed for baseline and SO when computing paired differences.

## 6. Segmentation evaluation

The script calls the original Talk2DINO segmentation evaluator and changes only `model.proj_name` so it loads the selected checkpoint. Example for VOC20:

```bash
bash experiments/projector_semi_orthogonal/run_segmentation_eval.sh \
  vitb_mlp_infonce_so_seed123
```

This uses:
- `src/open_vocabulary_segmentation/configs/voc/dinotext_voc_vitb_mlp_infonce.yml`
- `src/open_vocabulary_segmentation/configs/voc/eval_voc_pamr.yml`

To evaluate another benchmark, pass the original repository's matching config pair:

```bash
bash experiments/projector_semi_orthogonal/run_segmentation_eval.sh \
  vitb_mlp_infonce_so_seed123 \
  src/open_vocabulary_segmentation/configs/stuff/dinotext_stuff_vitb_mlp_infonce.yml \
  src/open_vocabulary_segmentation/configs/stuff/eval_stuff_pamr.yml \
  outputs/projector_semi_orthogonal/seg_eval/so_seed123_stuff
```

Check that the selected benchmark's `eval_base_cfg` exists in your checkout. The output directory is isolated per checkpoint/benchmark; the original segmentation code is not modified.

## 7. Retrieval evaluation (optional)

```bash
bash experiments/projector_semi_orthogonal/run_retrieval_eval.sh \
  vitb_mlp_infonce_so_seed123 \
  ../coco2014_b14/test.pth
```

This reuses `src/metrics.py`. Set `IMG_FEATURES` and `TEXT_FEATURES` to the same fields used by your established retrieval evaluation if they differ from the defaults:

```bash
IMG_FEATURES=avg_self_attn_out TEXT_FEATURES=ann_feats \
bash experiments/projector_semi_orthogonal/run_retrieval_eval.sh \
  vitb_mlp_infonce_so_seed123 ../coco2014_b14/test.pth
```

## 8. Correctness checks and interpretation

For SO runs, the diagnostic JSON should show:
- `semi_orthogonal_enabled: true`
- `projected_at_initialization: true`
- `projected_after_every_optimizer_step: true`
- `orthogonality_residual_normalized` close to zero (typically around floating-point numerical error)
- `singular_value_min` and `singular_value_max` close to 1

For baseline runs, the residual is reported but is not expected to be close to zero.

Do not overwrite the official baseline checkpoint. Keep the official weight and each experimental weight under distinct names. Compare the same benchmark, data split, evaluation configs, and seed; report mean ± std over paired seeds rather than selecting a single best seed.

## 9. Scope and limitations

- This implementation targets the repository's standard `ProjectionLayer` and `configs/vitb_mlp_infonce.yaml` (`linear_layer.weight` with shape `[768, 512]`). It is not intended for custom model classes with a different projector.
- SVD is performed after every optimizer step. This adds compute overhead but is the direct hard-projection version of the proposed experiment.
- The projector's linear layer is semi-orthogonal; the entire MLP is **not** distance-preserving because Tanh and the hidden layer remain unconstrained.
- This is an experiment adapter, not a claim that the semi-orthogonal constraint will improve segmentation. Retrieval and segmentation results must establish that empirically.
