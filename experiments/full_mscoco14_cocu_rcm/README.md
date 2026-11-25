# Full MSCOCO14 CoCu-RCM for Talk2DINO

独立 side-car 实验：将 `experiments/5k_cocu_rcm/` 的 CoCu-RCM 训练定义迁移到完整 MSCOCO14 Train2014，不修改 Talk2DINO 原始实现。

> **先看审计结论：** 当前 GitHub `main` 分支的 `experiments/5k_cocu_rcm/run_preprocess.sh` 引用了四个 `experiments/5k_cocu_rcm/data/*.py` 脚本，但仓库树中没有该 `data/` 目录。因此本包不能声称直接复用了这些缺失脚本。本包在独立目录内补齐全量预处理实现，并复用仓库中实际存在的 Local-CoCu、Local-CoCu Novel、Talk2DINO Dataset/Model/Loss/OVSS 代码。

## 1. 实验定义

五个训练 Arm：

| Arm | Global InfoNCE | CoCu Caption | Local supervision |
|---|---|---|---|
| `raw` | ✓ | | |
| `cocu` | ✓ | ✓ | |
| `cocu_same` | ✓ | ✓ | Same-image K-shot |
| `cocu_rcm` | ✓ | ✓ | Cross-image same-concept K-shot |
| `cocu_rcm_shuffled` | ✓ | ✓ | Cross-image wrong-concept control |

核心损失严格沿用 5K 训练代码：

\[
L = L_{\mathrm{global}} + \lambda_{\mathrm{local}} L_{\mathrm{local,scaled}}
\]

- `L_global`：仓库原始 `src.loss.ContrastiveLoss`，InfoNCE 双向交叉熵再除以当前 batch size 的平方。
- `L_local`：原 `experiments/5k_cocu_rcm/models/rcm_loss.py::multi_positive_infonce`，先对有效 Anchor 求平均，再除以 batch size 的平方。
- 默认 `lambda_local=0.1`，Local temperature `0.07`。
- `raw`、`cocu` 不启用 Local Loss。
- 每个 Seed 只创建一份 Projector 初始化，五个 Arm 共用该 Seed 的相同初始化。
- 默认训练参数直接读取 `configs/vitb_mlp_infonce.yaml`，当前仓库配置为 100 epochs、batch size 128、LR 1e-4。

## 2. 复用与新增

**直接复用：**

- `src.dataset.DinoClipDataset`
- `src.model.ProjectionLayer`
- `src.loss.ContrastiveLoss`
- `src.train_util.set_seed / const_lr / cosine_lr`
- `experiments/5k_cocu_rcm/models/rcm_loss.py::multi_positive_infonce`
- `experiments/local_cocu_talk2dino/common/vision.py::Vision`
- `experiments/local_cocu_talk2dino/common/algorithms.py::bbox_from_map`
- `experiments/local_cocu_talk2dino/common/experiment.py::mine('D')`
- `experiments/local_cocu_talk2dino/common/teacher.py::Teacher`
- `experiments/local_cocu_novel/build_train.py`
- 原始 OVSS 配置和 `src/open_vocabulary_segmentation/main.py`

**新增的 Full Scale 适配：**

- Full Train2014 Concept Archive 与分片。
- Chunked、可恢复的 Local-CoCu D 概念发现入口。
- Shard-based DINO Region Gallery 与缓存一致性检查。
- CLIP Region Verification 与概念文本特征缓存。
- Concept Inverted Index 和 paired RCM/Shuffled 采样。
- Lazy shard-loading Dataset，避免训练时把所有 Region 特征搬到 GPU。
- Epoch-boundary resume training。
- Full 专用评估结果收集和规模诊断报告。

## 3. 放置

在 Talk2DINO 仓库根目录解压：

```bash
cd /path/to/Talk2DINO
unzip full_mscoco14_cocu_rcm_experiment.zip -d .
```

必须得到：

```text
Talk2DINO/
└── experiments/
    └── full_mscoco14_cocu_rcm/
```

ZIP 不包含 COCO 数据、模型权重、特征缓存或 DINO/CLIP 权重。所有运行输出在：

```text
outputs/full_mscoco14_cocu_rcm/
```

训练权重使用独立名称：

```text
weights/vitb_mlp_infonce_full_mscoco14_cocu_rcm_<arm>_seed<seed>.pth
```

## 4. 数据与环境

激活 Talk2DINO 原有环境，确认：

```bash
python -c "import torch, torchvision, clip; print(torch.__version__)"
```

默认输入：

```text
outputs/coco2014_b14/train.pth
outputs/coco2014_b14/val.pth
data/coco2014/annotations/captions_train2014.json
data/coco2014/train2014/
experiments/local_cocu/MSCOCO14_train_gallery.json
# Optional only: a separate full head cache can be passed as HEAD_FEATURES
weights/vitb_mlp_infonce.pth
```

`BASE_TRAIN` 必须是完整 Train2014 的 Raw PTH，且每张恰好五条原始 Caption、包含形状严格为 `[12,768]` 的 `disentangled_self_attn` 和 512-D `ann_feats`。`BASE_VAL` 必须是原始 Val2014 PTH，包含形状为 `[768]`（或 `[1,768]`）的 `avg_self_attn_out` 和 512-D `ann_feats`。CoCu discovery 默认直接复用 Train PTH 中的 `disentangled_self_attn`，不需要单独 head cache。

如果你已有 Full ECSD 预处理输出，可直接复用其 Raw PTH：

```bash
BASE_TRAIN=outputs/full_mscoco14_ecsd/work/raw_train.pth \
BASE_VAL=outputs/full_mscoco14_ecsd/work/raw_val.pth \
bash experiments/full_mscoco14_cocu_rcm/scripts/run_preprocess.sh
```

如果还没有完整 Raw PTH，可先运行现有 Full ECSD 预处理管线生成兼容的 Raw 文件：

```bash
COCO_ROOT=data/coco2014 \
RAW_DIR=data/coco2014_full \
bash experiments/full_mscoco14_ecsd/run_preprocess.sh
```

这一步会运行 Full ECSD 的 Caption/特征准备流程，但不会训练 ECSD 模型。之后再用上面的 `BASE_TRAIN`、`BASE_VAL` 路径运行 CoCu-RCM。也可以自行将原始 `dino_extraction_v2.py` 和 `text_features_extraction.py` 用于构建同样 schema 的 Raw PTH。

数据准备脚本会严格检查：
- Train2014 82,783 images、Val2014 40,504 images（可通过环境变量覆盖期望值）。
- 每张图片恰好五条 Raw Caption。
- 训练/验证 Image ID 不重叠。
- Train `disentangled_self_attn`、Val `avg_self_attn_out` 存在。
- CLIP ViT-B/16 `ann_feats` 存在且为 512-D。

不满足条件会直接失败，不会静默截断或复制 Caption。

## 5. CoCu Concept 生成

优先复用现有完整 `matched_concepts_detailed.json`。脚本会对其 Image ID 覆盖做严格检查，5K 文件不能冒充 Full 文件。

默认尝试：

```text
outputs/local_cocu_talk2dino/D_region_discovery/matched/matched_concepts_detailed.json
```

如果它不是完整 Train2014 覆盖，默认使用新入口调用仓库已有 Local-CoCu `mine('D')` 算法生成完整结果。需要：

- 完整 Raw Train2014 PTH，其中 `disentangled_self_attn` 为 `[12,768]`。**默认直接从这个 PTH 复用 Head Features，不额外提取/保存一份约 3GB 的 head cache。**
- COCO Train2014 Caption annotation 和图片。
- `MSCOCO14_train_gallery.json`。
- 现有 Teacher 权重 `weights/vitb_mlp_infonce.pth`。
- 原始 Talk2DINO config。
- CLIP ViT-B/16 缓存、DINOv2 Torch Hub 缓存。

如果你已经有独立的完整 `head_features.pt`（格式 `image_id -> [12,768]`），可通过 `HEAD_FEATURES` 显式指定。只有显式指定的缓存不存在时，才可选择调用原有 Local-CoCu extractor：

```bash
HEAD_FEATURES=/path/to/full_head_features.pt AUTO_EXTRACT_HEADS=1 \
bash experiments/full_mscoco14_cocu_rcm/scripts/run_preprocess.sh
```

自定义路径示例：

```bash
COCU_MATCHES=/path/to/full_matched_concepts_detailed.json \
HEAD_FEATURES=/path/to/full_head_features.pt \
COCO_ANN=/path/to/captions_train2014.json \
IMAGE_ROOT=/path/to/train2014 \
GALLERY=/path/to/MSCOCO14_train_gallery.json \
TEACHER_WEIGHTS=/path/to/vitb_mlp_infonce.pth \
bash experiments/full_mscoco14_cocu_rcm/scripts/run_preprocess.sh
```

生成入口使用 `mine('D')`，按 chunk 写入 checkpoint shards；一次运行期间只加载一个 `Vision`，不会每个 chunk 重新加载 DINO。中断后重新执行会跳过已完成 shard。

## 6. 分阶段预处理

建议先分别运行，不要第一次就直接启动完整训练。

```bash
# 1. Raw Train/Val 数据审计与 staged hardlink
STAGE=data bash experiments/full_mscoco14_cocu_rcm/scripts/run_preprocess.sh

# 2. Full CoCu matches 检查/生成 + Concept Archive
STAGE=concept bash experiments/full_mscoco14_cocu_rcm/scripts/run_preprocess.sh

# 3. 复用 Local-CoCu Novel build_train.py，生成 Raw 5 + up-to-3 CoCu
STAGE=build bash experiments/full_mscoco14_cocu_rcm/scripts/run_preprocess.sh

# 4. DINO Head Region Gallery
STAGE=region bash experiments/full_mscoco14_cocu_rcm/scripts/run_preprocess.sh

# 5. CLIP Region Verification
STAGE=verify bash experiments/full_mscoco14_cocu_rcm/scripts/run_preprocess.sh

# 6. Same / RCM / Shuffled K-shot index
STAGE=sample bash experiments/full_mscoco14_cocu_rcm/scripts/run_preprocess.sh
```

Concept Discovery、Concept Archive、Region Gallery、CLIP Verification 和 K-shot Sampling 都会保存分片与进度清单。Sampling 额外保存 `sampling_progress.json`，中断后重复运行 `STAGE=sample` 会跳过已完成 Anchor shard；如果发现已完成 shard 的 hash 不匹配，会停止并要求 `OVERWRITE=1` 重建，避免静默混合新旧数据。

一次运行所有预处理：

```bash
bash experiments/full_mscoco14_cocu_rcm/scripts/run_preprocess.sh
```

常用配置：

```bash
# 默认设置
K_SHOT=2 NEGATIVES=16 SAMPLING_SEED=123

# 修改分片规模
CONCEPT_SHARD_SIZE=1000 \
REGION_SHARD_SIZE=256 \
CONCEPT_CHUNK_SIZE=256 \
bash experiments/full_mscoco14_cocu_rcm/scripts/run_preprocess.sh

# 修改 CLIP verification 阈值
MIN_CROP_SCORE=0.0 MIN_LOCAL_GAIN=-1.0 MIN_MARGIN=0.0 \
STAGE=verify bash experiments/full_mscoco14_cocu_rcm/scripts/run_preprocess.sh

# 有意重建当前实验输出
OVERWRITE=1 STAGE=region \
bash experiments/full_mscoco14_cocu_rcm/scripts/run_preprocess.sh
```

默认 Region Scope 是有 CoCu Concept 的训练图像。没有 Concept 的图像会被显式统计为 skipped，不进入 RCM Gallery。若要提取所有训练图像：

```bash
OVERWRITE=1 REGION_SCOPE=all STAGE=region \
bash experiments/full_mscoco14_cocu_rcm/scripts/run_preprocess.sh
```

### 主要预处理输出

```text
outputs/full_mscoco14_cocu_rcm/work/
├── raw_train.pth
├── raw_val.pth
├── data_manifest.json
├── data_statistics.json
├── matched_concepts_detailed_train.json
├── train_augmented.pth
├── train_augmented_manifest.json
├── concept_archive/
│   ├── shard_00000.json
│   ├── ...
│   ├── manifest.json
│   ├── concept_statistics.json
│   ├── concept_frequency.json
│   └── concept_coverage_report.json
├── region_gallery/
│   ├── features/shard_00000.pt
│   ├── metadata/shard_00000.json
│   ├── region_feature_index.json
│   └── region_gallery_manifest.json
├── region_verification/
│   ├── clip_text_features.pt
│   ├── clip_text_index.json
│   ├── verified_region_concepts/shard_00000.json
│   ├── verification_manifest.json
│   └── verification_statistics.json
├── sampling_index_same_k2/
├── sampling_index_cross_k2/
├── sampling_index_shuffled_k2/
├── sampling_statistics.json
└── sampling_progress.json
```

## 7. K-shot 采样定义

默认 `K=2, M=16, seed=123`。

- **Same**：从 Anchor 自身图像的同概念已验证 Region 中选 K 个。
- **RCM**：从 K 张不同的训练图像中各选一个同概念 Region。
- **Shuffled**：与 RCM 使用完全相同的 Anchor、K 张来源图像、负样本 ID，但从每张来源图像选择不属于 Anchor 概念的已验证 Region。

为保证 Shuffled 是可解释的机制对照，RCM 来源图像必须同时具有目标概念 Region 和至少一个非目标概念 Region。若某个 Anchor 找不到 K 张满足条件的来源图像，则该 Anchor 在 RCM 和 Shuffled 中同时跳过。

`sampling_statistics.json` 会记录：
- Total / Eligible / Skipped Anchors。
- Same / Cross / Shuffled K-shot Coverage。
- Concept Frequency 和 Frequency Bucket Coverage。
- RCM/Shuffled Anchor Set、Source Image IDs、Negative IDs 是否完全一致。

如果这三个一致性检查不是 `true`，不能将 RCM 与 Shuffled 的 mIoU 差异解释为正确概念对应关系的因果作用。

## 8. Smoke Test

```bash
bash experiments/full_mscoco14_cocu_rcm/scripts/run_smoke_test.sh
```

包含 Python Syntax、Shell Syntax、Synthetic Sampling、Local Loss Gradient 测试。

完成预处理后，可选择运行两步 GPU 训练 Smoke Test：

```bash
RUN_GPU_SMOKE=1 \
bash experiments/full_mscoco14_cocu_rcm/scripts/run_smoke_test.sh
```

它使用 `--max-steps 2 --no-save`，不会保存正式 Arm checkpoint。它会创建或复用 Seed 初始化文件。Local/Global 梯度范数在同一个首个有效局部监督 batch 上计算。

## 9. 五组训练

先只运行 Seed 123：

```bash
SEEDS="123" \
TRAIN_ARMS="raw cocu cocu_same cocu_rcm cocu_rcm_shuffled" \
bash experiments/full_mscoco14_cocu_rcm/scripts/run_train.sh
```

正式多 Seed：

```bash
SEEDS="123 456 789" \
TRAIN_ARMS="raw cocu cocu_same cocu_rcm cocu_rcm_shuffled" \
bash experiments/full_mscoco14_cocu_rcm/scripts/run_train.sh
```

两张或多张 GPU 可以按 Arm 并行（每个 Arm 仍使用原始单卡 Talk2DINO batch/loss 语义，不改变 InfoNCE 的 batch 定义）：

```bash
# 脚本会先创建缺失的 Seed 初始化并完成 Raw，再并行其他四组
GPUS="0 1" SEEDS="123" \
bash experiments/full_mscoco14_cocu_rcm/scripts/run_train_parallel.sh
```

这是跨 Arm 的进程级并行，不是单个模型的 DDP 训练。Region/CLIP 预处理通过 `DEVICE=cuda:0` 或 `cuda:1` 选择执行卡；OVSS 评估可用 `NPROC_PER_NODE=2` 使用多卡分布式评估。

常用控制：

```bash
# 只训练 RCM
TRAIN_ARMS="cocu_rcm" SEEDS="123" \
bash experiments/full_mscoco14_cocu_rcm/scripts/run_train.sh

# 调整 Local Loss 权重（正式主实验建议保持 0.1）
LAMBDA_LOCAL=0.05 \
bash experiments/full_mscoco14_cocu_rcm/scripts/run_train.sh

# 跳过已有正式权重
SKIP_EXISTING=1 \
bash experiments/full_mscoco14_cocu_rcm/scripts/run_train.sh

# 从头重训某些 Arm/Seed：与 OVERWRITE=1 配合
OVERWRITE=1 RESTART=1 TRAIN_ARMS="cocu_rcm" SEEDS="123" \
bash experiments/full_mscoco14_cocu_rcm/scripts/run_train.sh
```

训练每完成一个 Epoch 会保存独立 Resume checkpoint。中断后再次运行相同命令会从最近完成的 Epoch 继续；如果在 Epoch 中间中断，会从该 Epoch 开始重新执行。正式 `.pth` 权重只在全部 Epoch 完成后写入。

训练日志：

```text
outputs/full_mscoco14_cocu_rcm/
├── initializations/
├── checkpoints/
├── logs/
└── summary.csv
```

## 10. OVSS Evaluation

默认 PAMR 关闭，四个 Benchmark：

```bash
SEEDS="123" \
EVAL_ARMS="raw cocu cocu_same cocu_rcm cocu_rcm_shuffled" \
bash experiments/full_mscoco14_cocu_rcm/scripts/run_seg_eval.sh
```

三 Seed：

```bash
SEEDS="123 456 789" \
bash experiments/full_mscoco14_cocu_rcm/scripts/run_seg_eval.sh
```

若要使用 PAMR，必须对所有 Arm 和 Seed 一致设置：

```bash
PAMR=1 bash experiments/full_mscoco14_cocu_rcm/scripts/run_seg_eval.sh
```

## 11. 结果与诊断

评估结束后：

```bash
python experiments/full_mscoco14_cocu_rcm/diagnostics/build_report.py
```

生成：

```text
outputs/full_mscoco14_cocu_rcm/docs/full_mscoco14_cocu_rcm_analysis.md
```

报告包括 5K vs Full 概念/Region/Sampling 对照、按概念频率分桶的 K-shot 覆盖率、五组 OVSS、每个 Seed 的 RCM-Shuffled 配对差值，以及 Loss/Gradient 诊断。报告只读取实际存在的结果；没有运行的实验会标记为“待运行”，不会伪造指标。

## 12. 已知限制与科学注意事项

1. DINO Attention Head Region 是 Attention-conditioned Region Candidate，不是 Ground Truth Mask。
2. CLIP crop-text similarity 是弱监督验证信号，不是检测置信度。
3. `local_gain` 只是 Crop Similarity 相对 Full Image Similarity 的差值。
4. 负样本基于已验证的其他概念 Region；语义相近概念仍可能产生 False Negative。
5. 为保证 RCM/Shuffled 来源图像严格配对，RCM Anchor 覆盖率可能低于不配对的朴素跨图采样，必须报告 Coverage。
6. Full Concept Discovery 和 Region Extraction 都需要完整图像集的 GPU 前向，耗时较长，但每个阶段均按 shard 保存。
7. 本包未在你的 4090D 服务器、真实 COCO 数据、CLIP/DINO 权重上执行端到端实验。静态与合成测试结果会在交付说明中单独列出。
