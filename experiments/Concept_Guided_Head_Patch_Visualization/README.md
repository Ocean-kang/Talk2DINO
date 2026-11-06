# Concept-Guided Head-Patch Visualization

冻结 Talk2DINO 诊断实验。它不训练模型、不更新权重、不修改仓库核心文件。目标是对同一张 COCO 图片展示全部 DINOv2 Attention Heads 的 P0/P1/P2 空间关系，并利用原版 Talk2DINO Projector 从已有 MSCOCO14 Gallery 检索 Concept。

## 1. 实验做什么

| 分支 | 计算 | 可视化含义 |
|---|---|---|
| P0 | CLS→Patch attention softmax | 原版 Talk2DINO 的 Head 空间响应 |
| P1 | Patch→Patch QK affinity | 一个 Patch 对其他 Patch 的关系 |
| P1-Concept | `A_h @ cosine(Patch, W_projector(CLIP concept))` | Concept 证据经过指定 Head 关系传播后的空间图 |
| P2 | `Z_h=A_h @ V_h^patch`，再计算 patch-patch cosine | Head 内部 Value Context 的特征一致性 |

每张图片展示全部 12 个 Heads，不预先做 SHA/CAHS 筛选。

## 2. 仓库审计与复用边界

本包放在 Talk2DINO 仓库根目录的 `experiments/` 下，复用当前仓库已有模块：

- `src/hooks.py`：最后一个 DINOv2 block 的 QKV forward hook。
- `src/model.py`：`ProjectionLayer.project_clip_txt()`，把 CLIP 512 维文本映射到 DINO 768 维空间。
- `configs/vitb_mlp_infonce.yaml`：原版 Projector 配置。
- `experiments/Caption_Head_Visualization/data.py`：COCO `.pth` 读取、图片定位、Caption 分组。
- `experiments/Caption_Head_Visualization/run.py`：原版 Projector 权重加载。
- `experiments/Caption_Head_Visualization/visualize.py`：原图热力图叠加与拼图。
- `experiments/minimal_noun_supervision/common/nouns.py`：已有 Gallery 加载、原 Caption 内精确 Gallery 短语匹配。
- `experiments/minimal_noun_supervision/common/cache.py`：已有离线网络保护。
- `experiments/local_cocu/MSCOCO14_train_gallery.json`：已有完整 MSCOCO14 Gallery。

**不修改** `src/`、`train.py`、`configs/`、`weights/` 和现有实验目录。输出只写到 `outputs/Concept_Guided_Head_Patch_Visualization/`。

## 3. 放置 ZIP

在服务器 Talk2DINO 仓库根目录：

```bash
cd /path/to/Talk2DINO
unzip /path/to/Concept_Guided_Head_Patch_Visualization.zip
```

ZIP 内目录应落在：

```text
Talk2DINO/
└── experiments/
    └── Concept_Guided_Head_Patch_Visualization/
        ├── __init__.py
        ├── core.py
        ├── gallery.py
        ├── vision.py
        ├── visualize.py
        ├── run.py
        ├── README.md
        └── tests/
            └── test_core.py
```

本包依赖上述仓库中已有的 `Caption_Head_Visualization` 和 `minimal_noun_supervision` 辅助目录，请不要删除它们。

## 4. 输入文件

默认路径：

```text
weights/vitb_mlp_infonce.pth
configs/vitb_mlp_infonce.yaml
experiments/local_cocu/MSCOCO14_train_gallery.json
outputs/coco2014_b14/train.pth
```

COCO 图像目录通过 `--image-root` 指定，目录内应包含 `train2014/` 和/或 `val2014/`。

Gallery 文件是仓库已有的 JSON，不需要重新构建。原始 Caption 来自 feature PTH 中的 `annotations[].caption`。

### CLIP Gallery 文本特征有两种来源

**优先方式 A：复用已有 text bank**

如果此前的 `minimal_noun_supervision` 已经生成完整 Gallery 的 `shared/text_bank.pth`，直接传入：

```bash
--text-bank outputs/minimal_noun_supervision/full_gallery/shared/text_bank.pth
```

要求 bank 的 metadata 指明：
- `clip_model = ViT-B/16`
- `prompt_template = a photo of {}`
- `feature_space = clip_text_raw`
- `gallery_concepts` 与当前 Gallery 内容和顺序一致

**方式 B：使用已有本地 CLIP checkpoint 编码 Gallery**

```bash
--clip-checkpoint /absolute/path/to/ViT-B-16.pt
```

必须传本地 checkpoint 文件路径，不要传 `ViT-B/16` 模型名称。程序调用现有 `clip.load(local_file_path)`，不会触发模型下载。

本包没有打包 CLIP、DINOv2、Gallery、COCO 图片或任何权重。

## 5. 首次运行：建议只跑 1 张图

先激活你原来的 Talk2DINO 环境，不安装或升级依赖：

```bash
conda activate talk2dino
cd /path/to/Talk2DINO

export COCO_ROOT=/path/to/mscoco2014
export CLIP_CKPT=/path/to/existing/ViT-B-16.pt

CUDA_VISIBLE_DEVICES=0 python -m experiments.Concept_Guided_Head_Patch_Visualization.run \
  --features outputs/coco2014_b14/train.pth \
  --image-root "$COCO_ROOT" \
  --gallery experiments/local_cocu/MSCOCO14_train_gallery.json \
  --weights weights/vitb_mlp_infonce.pth \
  --config configs/vitb_mlp_infonce.yaml \
  --clip-checkpoint "$CLIP_CKPT" \
  --image-ids 122688 \
  --device cuda
```

`122688` 是示例 COCO image ID，必须存在于指定的 PTH 中。若你已经有兼容的 text bank，可将 `--clip-checkpoint "$CLIP_CKPT"` 换成 `--text-bank /path/to/shared/text_bank.pth`。

### 前 10 张图片

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.Concept_Guided_Head_Patch_Visualization.run \
  --features outputs/coco2014_b14/train.pth \
  --image-root "$COCO_ROOT" \
  --gallery experiments/local_cocu/MSCOCO14_train_gallery.json \
  --weights weights/vitb_mlp_infonce.pth \
  --config configs/vitb_mlp_infonce.yaml \
  --clip-checkpoint "$CLIP_CKPT" \
  --limit 10 --device cuda
```

### 继续扩展到 100 张

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.Concept_Guided_Head_Patch_Visualization.run \
  --features outputs/coco2014_b14/train.pth \
  --image-root "$COCO_ROOT" \
  --gallery experiments/local_cocu/MSCOCO14_train_gallery.json \
  --weights weights/vitb_mlp_infonce.pth \
  --config configs/vitb_mlp_infonce.yaml \
  --clip-checkpoint "$CLIP_CKPT" \
  --offset 10 --limit 100 --device cuda \
  --output outputs/Concept_Guided_Head_Patch_Visualization_100
```

### 指定 Query Patch

默认使用 Patch Grid 中心位置。448 输入下 Patch Grid 为 32×32：

```bash
--query-x 16 --query-y 16
```

例如选择左上区域：

```bash
--query-x 5 --query-y 7
```

P1/P2 的 Query-conditioned 图会展示这个 Query Patch 与所有 Key Patches 的关系。

### 严格检查重提取 Head 与 PTH 缓存

```bash
--strict-cache --min-cache-cosine 0.99
```

feature PTH 若有 `images[].disentangled_self_attn`，程序会与当前 QKV 重算的 `[12,768]` Head 逐头计算 cosine。低于阈值时严格模式停止；默认模式记录 warning。关系图和 Concept 检索始终使用同一次 DINO forward 产生的重算 Head，避免拿一份缓存 Head 去解释另一份 QKV 关系图。

## 6. DINOv2 本地缓存与离线要求

运行入口会复用 `minimal_noun_supervision` 的 offline guard，阻止远程 socket/URL 访问。DINOv2 源码和预训练权重必须已经在本机/torch hub cache 中。

默认：

```bash
--hub-source github --dinov2-repo facebookresearch/dinov2
```

这里的 `github` 是 torch.hub 的来源标识，不代表程序允许下载。离线 guard 会阻止远程获取；若本地缓存不存在，会报错而不是下载。

若已有本地 DINOv2 源码目录：

```bash
--hub-source local --dinov2-repo /absolute/path/to/dinov2
```

程序不会自动安装依赖或下载模型/数据。

## 7. 数学定义与张量约定

当前固定 DINOv2 ViT-B/14-reg、448 输入：
- Global tokens：1 CLS + 4 registers = 5。
- Patch tokens：1024。
- Heads：12。
- 每 Head 维度：64。
- Patch tokens：`[1024,768]`。
- Q/K/V：`[12,1029,64]`。

### P0：原版 CLS-to-Patch

\[
A_h^{0}(p)=\operatorname{softmax}_{p}
\left(\frac{q_{h,\mathrm{CLS}}k_{h,p}^{T}}{\sqrt{64}}\right)
\]

\[
H_h=\frac{1}{1024}\sum_p A_h^0(p)X_p
\]

这是仓库 `dino_extraction_v2.py` 的 `disentangled_self_attn` 逻辑。P0 热力图是空间注意力，不是语义分割 Mask。

### P1：Patch-to-Patch QK

\[
A_h^{1}(p,q)=
\operatorname{softmax}_{q}
\left(\frac{q_{h,p}k_{h,q}^{T}}{\sqrt{64}}\right)
\]

本实现只保留 Patch Query 和 Patch Key，并在 Patch Key 上重新归一化。它是**patch-only 诊断关系**，不是原始 DINO 对全部 CLS/register/patch tokens 的完整注意力概率。

为了审计这一区别，`diagnostics.json` 另存每个 Head 的 `p1_patch_mass_in_full_attention`：原始全 Token softmax 中，Patch Keys 实际占据的平均注意力质量。

- Mean incoming：对所有 Query 行求平均，显示被其他 Patch 平均关注的 Key Patch。
- Query-conditioned：选定一个 Query Patch，显示它对全部 Patch Keys 的注意力。

### W_projector-guided Concept–Patch

Gallery concept prompt：

\[
t_c=\mathrm{CLIP}(\text{"a photo of "}+c)
\]

投影到 DINO 空间：

\[
z_c=W_{\mathrm{projector}}(t_c)\in\mathbb R^{768}
\]

原始 Patch–Concept 响应：

\[
r_c(p)=\cos(X_p,z_c)
\]

通过某个 Head 的 P1 关系传播：

\[
r_{h,c}^{P1}(p)=\sum_q A_h^1(p,q)r_c(q)
\]

矩阵行 \(p\) 是 Query，列 \(q\) 是 Key，所以传播方向是 `A @ r`。同一 Concept 在不同 Head 下会得到不同的传播图。

每个 Head 的 Gallery Top-1 用于该 Head 的默认 P1 Concept propagation 图。原始 Caption 中实际出现的 Gallery Concepts 另外计算所有 Head 的 cosine，并在最佳 Head 上保存对应 P1 图。

### P2：Value Context

\[
Z_h=A_h^1V_h^{patch}
\]

\[
S_h(p,q)=\cos(Z_h(p),Z_h(q))
\]

输出：
- Mean feature affinity：每个 Patch 与其他 Patch 的平均非自身 cosine。
- Query feature affinity：选定 Query Patch 与其他 Patch 的 cosine。

P2 表示上下文聚合后特征的相似关系，不是物体概率，也不直接等同于区域 Mask。

## 8. 输出结构

```text
outputs/Concept_Guided_Head_Patch_Visualization/
├── run_config.json
├── gallery.json
├── summary.json
├── matches.jsonl
├── errors.jsonl
├── index.html
└── images/
    └── <image_id>/
        ├── original_crop.png
        ├── all_heads_p0.png
        ├── relations.npz
        ├── diagnostics.json
        ├── caption_concepts/
        │   ├── giraffe_best_head_03.png
        │   └── ...
        └── heads/
            ├── head_00/
            │   ├── p0_cls_to_patch.png
            │   ├── p1_mean_incoming.png
            │   ├── p1_query_conditioned.png
            │   ├── p1_concept_propagation.png
            │   ├── p2_mean_feature_affinity.png
            │   └── p2_query_feature_affinity.png
            ├── head_01/
            └── ...
```

输出目录必须为空或不存在。若需要重新运行或追加不同图片，请使用新的 `--output` 目录，避免覆盖旧结果。

浏览器打开：

```text
outputs/Concept_Guided_Head_Patch_Visualization/index.html
```

HTML 完全离线，图片与 JSON 都是本地相对路径。

## 9. 如何阅读结果

建议按以下顺序：

1. 看 `all_heads_p0.png`：原版 12 个 Head 分别响应图像什么区域。
2. 看每个 Head 的 Gallery Top-K：模型把这个 Head 解释成什么候选 Concept，分数是否有明显 Top-1/Top-2 差距。
3. 看 `p1_mean_incoming.png` 与 `p1_query_conditioned.png`：Head 内部 Patch 关系是否集中、是否连接相同物体的不同部分。
4. 看 `p1_concept_propagation.png`：Gallery Top-1 的 Patch 初始证据经过该 Head 关系传播后，区域是否更连贯，是否发生错误扩散。
5. 看 P2：上下文聚合后 Patch 特征是否形成相对一致的区域，是否过度平滑。
6. 对照 `caption_concepts/`：原 Caption 实际提及的 Concept 在最佳 Head 的 P1 传播图，与 Head 自己检索出的 Gallery Top-1 是否一致。

**不要把 Gallery Top-1 当作物体存在证明。** 它只是 W_projector 与 Head 特征的相似度检索结果。必须结合原图、Caption 和分割 GT 才能判断区域语义是否正确。

热力图采用同一图像、同一关系类型内的共享 5–99 百分位显示范围；不同 P0/P1/P2 之间的颜色值不可直接当作同一量纲比较。`relations.npz` 保存未做可视化归一化的低分辨率原始响应。

## 10. 测试

在 Talk2DINO 根目录：

```bash
python -m unittest discover \
  -s experiments/Concept_Guided_Head_Patch_Visualization/tests -v

python -m compileall -q \
  experiments/Concept_Guided_Head_Patch_Visualization

python -m experiments.Concept_Guided_Head_Patch_Visualization.run --help
```

先跑单图，检查 `cached_recomputed_head_cosine`、Head Top-K Concept 和原图热力图，再扩大样本量。该实验不运行训练或 segmentation evaluation。
