# Caption–Head Correspondence Visualization (CHCV)

独立的 **原版 Talk2DINO 推理/可视化** 实验，遵循 Ponytail 原则：仅添加 `experiments/Caption_Head_Visualization/`，不修改 `train.py`、`src/`、`configs/`、`weights/`，不重新训练。

## 1. 准备路径

必须从 **Talk2DINO 仓库根目录**运行。ZIP 里的 `experiments/Caption_Head_Visualization/` 放在你的 `experiments/` 下即可。

输入：

- `weights/vitb_mlp_infonce.pth`：原版文本 Projector 权重。
- `configs/vitb_mlp_infonce.yaml`：原版 Projector 结构。
- `outputs/coco2014_b14/train.pth`：包含 `images[]`、`annotations[]`，每张图有 `disentangled_self_attn` `[12,768]`，每条 annotation 有 `caption` 和 `ann_feats` `[512]`。实际路径可通过 `--features` 调整；也可用 `val.pth`。
- 原始 MSCOCO 2014 JPEG 图片，示例目录：`/data/mscoco2014/train2014/COCO_train2014_*.jpg`。传 `--image-root /data/mscoco2014`。若路径本身是 `train2014/` 也可以。
- 已安装原 Talk2DINO 运行环境（`torch`、`torchvision`、`clip`、`Pillow`、`numpy`、`matplotlib`）。需要 `torch.hub` 能找到 DINOv2 `dinov2_vitb14_reg` 权重（本机缓存，或联网首次下载）。

原仓库常见另一种缓存位置是 `../coco2014_b14/train.pth`，不要假定当前仓库存在 `outputs/coco2014_b14/train.pth`，请检查后指定实际位置。

## 2. 运行命令

### 2.1 先跑 1 张指定 COCO 图

```bash
cd /path/to/Talk2DINO
conda activate talk2dino  # 按你的实际环境切换

CUDA_VISIBLE_DEVICES=0 python -m experiments.Caption_Head_Visualization.run \
  --features outputs/coco2014_b14/train.pth \
  --image-root /data/mscoco2014 \
  --image-ids 122688 \
  --device cuda
```

`122688` 仅为示例图片 ID。若这个 ID 不存在于指定 PTH 中，选择其他 image_id 或改用 `--limit 1`。

### 2.2 前 100 张 COCO14 图片

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.Caption_Head_Visualization.run \
  --features outputs/coco2014_b14/train.pth \
  --image-root /data/mscoco2014 \
  --weights weights/vitb_mlp_infonce.pth \
  --limit 100 --device cuda
```

也可用封装脚本：

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/Caption_Head_Visualization/run.sh \
  /data/mscoco2014 \
  --features outputs/coco2014_b14/train.pth --limit 100
```

### 2.3 其他情形

```bash
# 验证集前 50 张，输出到另一目录
CUDA_VISIBLE_DEVICES=0 python -m experiments.Caption_Head_Visualization.run \
  --features outputs/coco2014_b14/val.pth \
  --image-root /data/mscoco2014 \
  --limit 50 --output outputs/Caption_Head_Visualization_val

# 自定义 ID，每条原始 caption 生成一张 PNG
CUDA_VISIBLE_DEVICES=0 python -m experiments.Caption_Head_Visualization.run \
  --features outputs/coco2014_b14/train.pth \
  --image-root /data/mscoco2014 \
  --image-ids 122688,391895,234567

# 原始缓存中没有 ann_feats 时，允许在线用 CLIP ViT-B/16 编码（更慢）
CUDA_VISIBLE_DEVICES=0 python -m experiments.Caption_Head_Visualization.run \
  --features outputs/coco2014_b14/train.pth --image-root /data/mscoco2014 \
  --text-source clip --limit 5

# 使用本地 torch.hub DINOv2 源码：用于不能联网的服务器
CUDA_VISIBLE_DEVICES=0 python -m experiments.Caption_Head_Visualization.run \
  --features outputs/coco2014_b14/train.pth --image-root /data/mscoco2014 \
  --hub-source local --dinov2-repo /path/to/dinov2 --limit 5

# 只有缓存符合原训练协议时，强制使用缓存头表示并严格核验注意力重提取
CUDA_VISIBLE_DEVICES=0 python -m experiments.Caption_Head_Visualization.run \
  --features outputs/coco2014_b14/train.pth --image-root /data/mscoco2014 \
  --feature-source cached --strict-cache --limit 10

# 不创建 12-head 总览以减少 PNG 数量
CUDA_VISIBLE_DEVICES=0 python -m experiments.Caption_Head_Visualization.run \
  --features outputs/coco2014_b14/train.pth --image-root /data/mscoco2014 \
  --limit 100 --no-all-heads
```

默认 `--limit 20`，`--limit 0` 为整个 PTH。`--offset 100` 从第 101 张继续（数据顺序为 PTH 中 images 的原始顺序）。设置了 `--image-ids` 时忽略 `--limit/--offset`。

默认有断点跳过功能：输出内的 `diagnostics.json` 与权重 SHA256 和运行参数吻合，则跳过；强制重算加 `--no-skip-existing`。如果更换 PTH、权重或实验配置，建议同时更换输出目录以便比较。

## 3. 输出文件

```text
outputs/Caption_Head_Visualization/
├── run_config.json
├── matches.jsonl                 # 一图一行，完整 captions 与 5×12 cosine
├── index.html                    # 离线浏览器预览入口（直接打开）
├── summary.json                  # head 占用量、平均 margin、警告数量
├── errors.jsonl                  # 缺文件/异常原因
└── images/
    └── <image_id>/
        ├── original_crop.png     # DINO 真正看到的 448×448 图像
        ├── original_full_view.png # Resize 后的完整视野（未裁剪）
        ├── overview.png          # 原图 + 每条 caption 对应头的热力图
        ├── similarity_matrix.png # C×12 余弦矩阵，标红 Top-1
        ├── all_12_heads.png      # 全部 12 头注意力热力图
        ├── caption_00_head_03.png
        ├── caption_00_head_03_full_view.png # 映射回完整视野，裁剪之外无热区
        ├── caption_01_head_07.png
        ├── ...
        ├── attention_maps.npz    # 原始 12×32×32 attention 概率；未做可视化归一化
        └── diagnostics.json      # 各 head 相似度、margin、使用次数及缓存一致性
```

注意：不同图像/Caption 的 head 编号、相似度取决于真实模型结果；上面的 `head_03`、`head_07` 仅为文件名示例。

## 4. 与原 Talk2DINO 一致的数值逻辑

```text
COCO caption -> CLIP ViT-B/16 (原 ann_feats) -> 原 Projector -> [C,768]
DINOv2 ViT-B/14-reg (448) -> last-block QKV -> CLS-to-patch attention [12,1024]
                         -> x_norm_patchtokens -> 原 disentangled_self_attn [12,768]
cosine(projected_text, heads) -> [C,12] -> 每个 caption argmax head
head attention [32,32] -> resize to 448 -> overlay on exact DINO center crop
```

原代码 `dino_extraction_v2.py` 不是计算 `sum(attn*patch)`，而是计算 `mean(attn*patch)`，本实现通过 `head_pool` 复现 **额外除以 1024** 的行为；余弦匹配虽对正的整体尺度不敏感，但保持相同操作以便数值核验。

默认 `--feature-source auto`：若 PTH 有 `disentangled_self_attn`，**优先用训练时缓存的 heads 做 cosine matching**；空间注意力则重新调用 DINO 前向提取。逐头计算缓存与重提取特征的 cosine，低于 `--min-cache-cosine 0.99` 记录警告；`--strict-cache` 设为严格失败。没有缓存时用重提取 heads。

默认 `--text-source cached`：直接读取 PTH 的原始 `ann_feats`，避免再次计算 CLIP；也支持 `--text-source clip` 强制按 raw captions 再编码。

**重要局限**：

1. 热力图是 DINOv2 最后一层 **CLS→patch 注意力**，并非由文字直接计算出的条件热力图，更不是语义分割真值。文字只决定选中哪个 head；同一张图相同 head 的热力图必然相同。
2. `argmax` 对任何 caption 都会匹配一个 head，低分也会匹配，不表示一定存在相应对象。应结合 score 和 Top-1/Top-2 margin 判断。
3. 主要热力图 overlay 使用 Resize + CenterCrop 后的 448×448 图，与 DINO 实际输入对齐；`*_full_view.png` 额外将热区放回 Resize 后的未裁剪完整视野，裁剪之外没有模型预测，不表示未裁剪区域不存在相应对象。
4. PNG 默认为 **每个 head 各自 5–99 百分位归一化**，便于观察局部结构，但图之间的绝对颜色深浅不可直接比较。`--attention-norm global` 可使用全 12 头共享的缩放范围。`attention_maps.npz` 保留原值，用于定量统计。
5. 默认不对外部图像做 segmentation，也不执行训练；它专用于检查原版 Talk2DINO 的 caption–head 匹配关系。

## 5. 测试与故障排查

```bash
# 从根目录执行，无需 GPU 和 COCO 图片
python -m unittest discover -s experiments/Caption_Head_Visualization/tests -v
python -m compileall -q experiments/Caption_Head_Visualization

# 查看全部命令参数
python -m experiments.Caption_Head_Visualization.run --help
```

- `Missing feature PTH`：先确认实际特征位置；从 `../coco2014_b14/train.pth` 等路径传入 `--features`。
- `Missing cached ann_feats`：重新跑原 `text_features_extraction.py` 或添加 `--text-source clip`。
- `CUDA not available`：确认激活了训练环境，或临时使用 `--device cpu`（非常慢）。
- `torch.hub` 首次无法联网：预先缓存 DINOv2 及其权重，或使用 `--hub-source local --dinov2-repo /path/to/dinov2`；本地 DINOv2 源码仍可能需要已有的权重缓存。
- `Cached/re-extracted heads disagree`：先检查原特征缓存是否确实是 `dinov2_vitb14_reg`、448 crop，及 torch.hub 当前 DINOv2 权重；不能悄悄把明显不匹配的 maps 当成原 heads 的空间解释。
- `ModuleNotFoundError: src`：必须从 Talk2DINO 根目录执行 `python -m ...`，不要直接 `python experiments/Caption_Head_Visualization/run.py`。

### 参考仓库源代码

- `src/hooks.py`：原 QKV forward hook 与 CLS-to-patch attention。
- `dino_extraction_v2.py`：预处理、`disentangled_self_attn`。
- `src/model.py`：`ProjectionLayer.project_clip_txt`、L2 normalize、`max_score`。
- `experiments/local_cocu_talk2dino/common/vision.py`：先前的单图 head feature/attention map 提取惯例。

代码按公开的 `Ocean-kang/Talk2DINO` 主分支相关文件结构编写；未在你的实际 COCO 数据和 GPU 环境运行，建议先用 `--limit 1` 核验视觉结果及 `cached_recomputed_head_cosine`。

**浏览方式**：生成后直接在浏览器打开 `outputs/Caption_Head_Visualization/index.html`，点击各图片可看大图；它是纯本地 HTML，无需开启 HTTP 服务。
