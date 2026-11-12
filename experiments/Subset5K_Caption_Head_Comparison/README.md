# Subset5K Caption–Head Comparison

**实验目的：** 使用四组已经训练完成的 Talk2DINO Projector，分析 `raw`、`rich`、`rich_weaken`、`rich_diversity` 五条 captions 与同一张图像的 12 个 DINOv2 Attention Head features 之间的对应关系。

本实验只做推理、匹配、统计和可视化，不训练 Projector、不运行 segmentation，也不修改 Talk2DINO 原始源码。

## 1. 实验逻辑

对每张图像、每个 caption 生成一组 12 维余弦相似度：

```text
Caption (CLIP ViT-B/16, 512-D)
          |
          v
对应条件训练好的 Projector
          |
          v
Projected caption (768-D)
          |
          | cosine similarity
          v
12 个 DINOv2 ViT-B/14-reg Attention Head features (12 x 768)
          |
          v
[caption x 12] similarity matrix
          |
          v
Top-1 Head / Top-2 Head / Top-1-Top-2 margin
```

对每个 image、每个条件分别计算。四个条件分别加载自己的 projector checkpoint，但共享同一组冻结 DINO head features。

### 关键实现约束

- **Caption 特征：** 直接读取预处理 PTH 中的 `annotations[].ann_feats`，不重新编码 CLIP。
- **Head 特征：** 直接读取 `images[].disentangled_self_attn`，形状应为 `[12, 768]`。这与 Talk2DINO 训练使用的 head feature 定义一致。
- **映射：** 复用 `src.model.ProjectionLayer` 和原始配置，使用 `projector.project_clip_txt()`，再做 L2 normalization 与 cosine similarity。
- **匹配：** 每条 caption 独立选择相似度最高的 Head，同时保存完整 12-head 相似度、Top-2 和 margin。
- **空间热力图：** 仅对选中的少量图像重新提取 DINO CLS-to-patch attention maps。四个条件复用同一张图的同一组空间 attention maps，只因各自 projector 的匹配结果不同而高亮不同 Head。

> 重要：热力图不是文本条件化的 attention，也不是 segmentation mask。Caption 负责选择 Head，热力图展示的是 DINO 该 Head 原本的 CLS-to-patch attention。必须结合 cosine、Top-1/Top-2 margin 和空间图共同判断，不能把 argmax 本身当成语义匹配成功。

## 2. 放置位置

在 Talk2DINO 仓库根目录解压：

```bash
cd /path/to/Talking2dino/Talk2DINO
unzip Subset5K_Caption_Head_Comparison.zip -d .
```

解压后：

```text
Talk2DINO/
└── experiments/
    └── Subset5K_Caption_Head_Comparison/
        ├── __init__.py
        ├── run.py
        ├── run.sh
        ├── README.md
        └── tests/
            └── test_batch_matching.py
```

依赖复用仓库已有的：

- `experiments/Caption_Head_Visualization/`：DINO head/attention 提取、图片路径解析、热力图绘制。
- `src/model.py`：原版 `ProjectionLayer`。
- `configs/vitb_mlp_infonce.yaml`：原始 Projector 结构。
- 已有 Talk2DINO Python 环境、PyTorch、Pillow、Matplotlib、NumPy。

不安装新包、不下载新语言资源、不修改 `src/`、`train.py`、`configs/` 或已有实验。

## 3. 输入文件

### 3.1 四个 checkpoint

默认从仓库根目录 `weights/` 读取：

```text
weights/
├── vitb_mlp_infonce_subset5k_raw.pth
├── vitb_mlp_infonce_subset5k_rich.pth
├── vitb_mlp_infonce_subset5k_rich_weaken.pth
└── vitb_mlp_infonce_subset5k_rich_diversity.pth
```

你给出的 checkpoint 列表中 `raw` 重复出现，因此本实验按四个不同条件读取四个唯一权重。

### 3.2 四组特征 PTH

默认目录：

```text
outputs/5k_rich_diversity/work/
├── raw_train.pth
├── raw_val.pth
├── rich_train.pth
├── rich_val.pth
├── rich_weaken_train.pth
├── rich_weaken_val.pth
├── rich_diversity_train.pth
└── rich_diversity_val.pth
```

每个 PTH 必须保留：

- `images[]`: `id`、`file_name`、`disentangled_self_attn` `[12,768]`
- `annotations[]`: `image_id`、`caption`、`ann_feats` `[512]`

默认 `--split all` 会合并 train 和 val，因此统计覆盖完整 5k 图片、每个条件 25k captions。程序还会检查四组 image IDs、train/val split、DINO head features 是否一致。不满足时会停止，防止把数据不一致造成的差异误认为 Caption 策略差异。

### 3.3 如果 Raw/Rich/Rich-Weaken 与 Rich-Diversity 的 PTH 在不同目录

例如前三组来自旧的 `raw_rich_weaken_5k`，Rich-Diversity 来自新实验：

```bash
--raw-work-dir outputs/raw_rich_weaken_5k/work \
--rich-work-dir outputs/raw_rich_weaken_5k/work \
--rich-weaken-work-dir outputs/raw_rich_weaken_5k/work \
--rich-diversity-work-dir outputs/5k_rich_diversity/work
```

如果四组 PTH 都已放在 `outputs/5k_rich_diversity/work/`，则不需要这些 override。

## 4. 先做小规模检查

确认当前环境和文件路径后，从仓库根目录运行：

```bash
conda activate talk2dino

CUDA_VISIBLE_DEVICES=0 python -m experiments.Subset5K_Caption_Head_Comparison.run \
  --work-dir outputs/5k_rich_diversity/work \
  --weights-dir weights \
  --config configs/vitb_mlp_infonce.yaml \
  --image-root outputs/5k_rich_diversity/work/images \
  --split all \
  --limit 2 \
  --viz-limit 2 \
  --device cuda
```

如果前三组和第四组 PTH 分开，追加第 3.3 节中的四个 `--*-work-dir` 参数。

先检查：

1. 四个权重都能被原始 `ProjectionLayer` strict load。
2. 每张图像的四组特征均有五条 captions。
3. 四组数据的 image IDs 与 train/val 划分一致。
4. 四组缓存 DINO head features 的逐头 cosine 一致性检查通过。
5. `outputs/Subset5K_Caption_Head_Comparison/index.html` 可以打开。

默认不会覆盖已有输出目录。如果目录非空，换一个 `--output` 路径，或明确添加 `--overwrite`。

## 5. 完整 5k 对应关系分析

小规模检查通过后，执行完整 5k：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.Subset5K_Caption_Head_Comparison.run \
  --work-dir outputs/5k_rich_diversity/work \
  --weights-dir weights \
  --config configs/vitb_mlp_infonce.yaml \
  --image-root outputs/5k_rich_diversity/work/images \
  --split all \
  --limit 0 \
  --batch-images 256 \
  --viz-limit 50 \
  --device cuda
```

若四组特征位于不同目录，追加对应的 `--raw-work-dir` 等参数。

默认行为：

- `--limit 0`：匹配全部 5,000 张图片。
- `--split all`：合并 train 和 val。
- `--batch-images 256`：按图像批量执行 projector 推理，避免逐 caption 频繁启动 GPU kernel。
- `--viz-limit 50`：只为前 50 张图片提取空间 attention 并生成 PNG。匹配和统计仍覆盖全部 5k。
- `--viz-limit 0`：不生成热力图，仅做完整匹配与统计。
- `--no-visualize`：完全跳过图片读取与 DINO 空间 attention 重提取，适合先跑完整 5k 数值分析。

只看验证集：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.Subset5K_Caption_Head_Comparison.run \
  --split val --viz-limit 50 --device cuda
```

指定图片可视化：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.Subset5K_Caption_Head_Comparison.run \
  --viz-image-ids 122688,391895,234567 --device cuda
```

如果服务器已有本地 DINOv2 源码，但 torch.hub GitHub 源不可用：

```bash
--hub-source local --dinov2-repo /path/to/dinov2
```

如果只做数值匹配，不需要重新加载 DINOv2：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.Subset5K_Caption_Head_Comparison.run \
  --no-visualize --viz-limit 0 --device cuda
```

## 6. 输出内容

```text
outputs/Subset5K_Caption_Head_Comparison/
├── run_config.json
├── summary.json
├── summary.csv
├── per_image_metrics.csv
├── cross_arm_comparison.json
├── matches_raw.jsonl
├── matches_rich.jsonl
├── matches_rich_weaken.jsonl
├── matches_rich_diversity.jsonl
├── visualization_errors.jsonl
├── index.html
└── images/
    └── <image_id>/
        ├── raw/
        │   ├── original_crop.png
        │   ├── original_full_view.png
        │   ├── overview.png
        │   ├── similarity_matrix.png
        │   ├── all_12_heads.png
        │   ├── caption_00_head_XX.png
        │   ├── ...
        │   ├── attention_maps.npz
        │   └── diagnostics.json
        ├── rich/
        ├── rich_weaken/
        └── rich_diversity/
```

### 核心结果文件

| 文件 | 作用 |
|---|---|
| `matches_{arm}.jsonl` | 每张图像五条 captions 对 12 个 Heads 的完整 cosine、Top-1、Top-2、margin |
| `summary.json` | 四组整体统计和每个 caption slot 的统计 |
| `summary.csv` | 四组主要指标汇总，方便 Excel 对比 |
| `per_image_metrics.csv` | 每张图像在各条件下的 head 覆盖数、平均相似度和 margin |
| `cross_arm_comparison.json` | 不同条件下每张图像所用 Head 集合的 Jaccard overlap |
| `similarity_matrix.png` | 5 captions × 12 heads 的余弦相似度矩阵，红框为每条 caption 的 Top-1 |
| `overview.png` | 原图和每条 caption 选中 Head 的空间热力图 |
| `all_12_heads.png` | 同一张图全部 12 个 Head 的空间热力图 |

## 7. 如何解读指标

### A. Head Usage / Head Coverage

`head_use_counts_1_to_12` 统计每个 Head 被多少条 captions 选为 Top-1。

- **Head 使用更均衡：** captions 的语义对应可能覆盖更多 heads，但不必然意味着更准确。
- **少数 Head 占据大量 captions：** 可能存在 head collapse，也可能这些 heads 确实承载了更通用的语义。
- **Unique Heads per Image：** 一张图五条 captions 实际使用了多少个不同 heads。值越高说明选择更分散，但不能单独作为性能指标。
- **Normalized Head Usage Entropy：** 统计全数据集 Head 使用分布的均衡程度，范围约为 0 到 1。

### B. Cosine Similarity

- **Top-1 cosine：** 每条 caption 与所选 Head 的投影后余弦相似度。
- **Top-1/Top-2 margin：** 第一名与第二名 Head 的相似度差。margin 越大，选择越明确；margin 接近 0 表示两个 Head 难以区分。
- **注意：** 不同 projector 的分数可用于描述各自的匹配强度，但 projector 可能存在不同的分数校准，不能仅凭平均 cosine 排名断言语义质量高低。

### C. Similarity Matrix 与热力图

- 行是五条 captions，列是 12 个 heads。
- 每行红框表示 Top-1 head。
- 热力图展示被选中 Head 的 DINO CLS-to-patch attention。
- 同一图片、同一个 Head 的热力图在四组条件中应完全相同；不同的是各条件选择了哪个 Head。

### D. Cross-arm Head-set Jaccard

比较两种 Caption 条件在同一张图片上选出的 Head 集合：

\[
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
\]

值高表示两个条件选择的 Head 集合重叠较多，值低表示对应关系发生了变化。它比较的是五条 captions 的 Head 集合，不代表两个条件第 1 条 caption 的语义天然一致。

## 8. 重要的实验边界

1. 不重新训练，不改动任何 checkpoint。
2. 不重新计算 CLIP caption embedding，直接使用各实验 PTH 中缓存的 `ann_feats`。
3. 匹配阶段使用各图片缓存的 `[12,768] disentangled_self_attn`，与 Talk2DINO 训练时的视觉特征定义保持一致。
4. 为了画空间热力图，程序只对 `viz-limit` 指定的图像重跑 DINO 前向；并核验重提取 head features 与缓存特征的 cosine。
5. `CLS-to-patch attention` 是 DINO 自身的空间注意力，不是 projector 产生的文本条件化 mask，也不等价于真实物体区域。
6. `argmax` 总会选出一个 Head，即使所有 cosine 都很低。因此必须联合检查 cosine、margin、head 使用分布和空间热力图。

## 9. 常见问题

- **Missing `{arm}_train.pth` / `{arm}_val.pth`：** 检查 `--work-dir` 或该 arm 的 `--*-work-dir`。
- **Missing `ann_feats`：** 对应 PTH 尚未完成 CLIP ViT-B/16 特征提取。
- **Missing `disentangled_self_attn`：** 对应 PTH 尚未完成 DINOv2 ViT-B/14-reg head feature 提取。
- **Image IDs / split 不一致：** 不建议直接比较。先核对原实验 `val_ids.json`；只有确认接受 split 差异时才使用 `--allow-split-mismatch`。
- **DINO head feature mismatch：** 不应把不同视觉特征当成严格控制变量；检查是否使用相同 DINO 模型、448 resize/center-crop 和预处理。
- **输出目录非空：** 换 `--output`，或在确认需要覆盖后加 `--overwrite`。
- **`ModuleNotFoundError: src`：** 必须从 Talk2DINO 仓库根目录运行 `python -m experiments.Subset5K_Caption_Head_Comparison.run`。

