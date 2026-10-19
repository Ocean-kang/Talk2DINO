# Local-CoCu × Talk2DINO — A–H 独立 Bootstrap Concept Discovery 实验

> **Ponytail（旁路开发）**：仅新增 `experiments/local_cocu_talk2dino/`，不改动 `src/`、`train.py`、`configs/`、`src/open_vocabulary_segmentation/`。8 个方案一方案一文件夹，共用实现放在 `common/`。

## 1. 实现依据与不变量

已适配 Ocean-kang/Talk2DINO 主分支：

- `experiments/local_cocu/extract_head_features.py`：原 DINOv2 ViT-B/14-reg、448、12 heads，构建 `[12,768]` 特征；没有缓存时可调用原脚本重新抽取。
- `src.hooks.get_self_attention/process_self_attention`：C/D/G 沿用原 DINOv2 最后一层 QKV 和 attention 加权聚合实现。
- `src.model.ProjectionLayer.project_clip_txt` + `clip.load('ViT-B/16')`：使用 `weights/vitb_mlp_infonce.pth` 作为**冻结概念发现 Teacher**。
- `experiments/local_cocu_novel/build_train.py`：原 `train.pth` 前 5 条 captions 和已提取的原始图像特征完全保留；新发现 concept 以 `a photo of a {concept}` 添加最多 3 条，并用 CLIP ViT-B/16 产生 `ann_feats`。
- **原 `python train.py`**：随机初始化 Student、InfoNCE、训练期 `disentangled_self_attn` 最大 Head 匹配、验证期 `avg_self_attn_out` 不变。这里只改变**预先构造的训练数据**，Teacher 从不参与 Student 的训练前向/反向。
- **原 segmentation main.py**：沿用 VOC20、Context59（`context_bg`）、ADE150、COCO-Stuff 的 `.yml` 和 `model.proj_name`，不替换分割逻辑。

注意：将额外 0–3 条概念独立加入图文训练样本，会改变训练文本数量，这是本文采用的 **5+up-to-3** 设定，不是将 concept 融合进原 5 条句子的设定。若需严格比较同等训练预算，应另加 equal-step 控制组。

## 2. 代码结构

```text
experiments/local_cocu_talk2dino/
├── A_direct_retrieval/           # A: Head -> Global Concept Gallery
├── B_head_image_retrieval/       # B: Head -> Head 跨图检索
├── C_uncovered_heads/            # C: Caption 覆盖不足的 Heads
├── D_region_discovery/          # D: Head Attention Crop 复核
├── E_cross_image_consistency/   # E: 视觉邻居一致性
├── F_counterfactual/            # F: 同场景难负例
├── G_multiview_agreement/       # G: 多视图稳定性
├── H_iterative_bootstrap/       # H: 多轮迭代 Teacher -> Student
├── common/                      # 所有方案共享 IO、Teacher、方法、训练/评估桥接
├── tests/                       # CPU 单测、mock dataset 兼容性测试
├── run.sh                       # 统一入口
└── README.md
```

各方案都包含 `run.py`、`run.sh`、`README.md`，实验目录下没有克隆原始训练代码。

## 3. 环境与准备数据

在仓库根目录执行（不要在 `experiments/` 内运行）：

```bash
cd /home/master/code/oymk/Talking2dino/Talk2DINO
conda activate /home/master/software/oymk/talk2dino

# B/E/F 额外需要 FAISS；CPU 版即可建立近百万 Head 的向量索引
pip install faiss-cpu

# 原仓库已经安装的依赖仍需可用：torch、torchvision、clip、mmcv、mmseg、PIL 等
```

已存在/需要的文件：

```text
weights/vitb_mlp_infonce.pth             # T0 Teacher (用户指定)
configs/vitb_mlp_infonce.yaml
outputs/coco2014_b14/train.pth          # 原始 Talk2DINO train features
outputs/coco2014_b14/val.pth            # 原始 Talk2DINO val features
data/coco2014/annotations/captions_train2014.json
data/coco2014/train2014/
experiments/local_cocu/MSCOCO14_train_gallery.json
outputs/local_cocu/head_features/head_features.pt  # 若已有则直接复用
outputs/local_cocu/retrieval/neighbors.json         # F 的 CLS 难负例来源
```

缺少 Head 缓存时加 `--extract-missing-heads`，自动调用仓库的原始 Local-CoCu 特征提取脚本；F 缺少 CLS 邻居时加 `--extract-missing-cls`。离线服务器须保证 `torch.hub` 中 DINOv2 仓库和权重均已缓存。

## 4. 如何运行

以下所有命令以仓库根目录为当前工作目录。每条命令执行 **concept discovery → 5+up-to-3 数据构建 → 原始 Talk2DINO 训练 → 原始 4 项分割评估**。

```bash
CUDA_VISIBLE_DEVICES=1 bash experiments/local_cocu_talk2dino/run.sh A all 1
CUDA_VISIBLE_DEVICES=1 bash experiments/local_cocu_talk2dino/run.sh B all 1
CUDA_VISIBLE_DEVICES=1 bash experiments/local_cocu_talk2dino/run.sh C all 1
CUDA_VISIBLE_DEVICES=1 bash experiments/local_cocu_talk2dino/run.sh D all 1
CUDA_VISIBLE_DEVICES=1 bash experiments/local_cocu_talk2dino/run.sh E all 1
CUDA_VISIBLE_DEVICES=1 bash experiments/local_cocu_talk2dino/run.sh F all 1
CUDA_VISIBLE_DEVICES=1 bash experiments/local_cocu_talk2dino/run.sh G all 1
CUDA_VISIBLE_DEVICES=1 bash experiments/local_cocu_talk2dino/run.sh H all 1 --rounds 2
```

**分阶段运行**，以上 `all` 可以换成 `mine`、`build`、`train`、`eval`；已存在的阶段产物默认复用：

```bash
bash experiments/local_cocu_talk2dino/run.sh A mine 1
bash experiments/local_cocu_talk2dino/run.sh A build 1
bash experiments/local_cocu_talk2dino/run.sh A train 1
bash experiments/local_cocu_talk2dino/run.sh A eval 1
```

也可直接从各方案文件夹运行：

```bash
python experiments/local_cocu_talk2dino/C_uncovered_heads/run.py \
  --stage mine --gpu 1 --uncovered-heads 4 --attention-min-concentration 1.3
```

**小样本调试**：

```bash
# 只挖掘前 100 张有 Head 缓存的图
bash experiments/local_cocu_talk2dino/run.sh A mine 1 --limit 100 --force-mine

# 如果明确要在这 100 张上训练，需要同样执行 mine 后启用 --subset-train
bash experiments/local_cocu_talk2dino/run.sh A build 1 --subset-train --force-build
bash experiments/local_cocu_talk2dino/run.sh A train 1
```

默认要求 **Head 缓存覆盖 base train.pth 内全部图像**。二者 ID 不一致时不会静默地少训练数据；显式使用 `--subset-train` 才会为已挖掘 ID 创建独立的 `base_train_subset.pth`。

### 方案核心参数

| 方案 | 核心实现与可调参数 |
|---|---|
| A | 每 Head Top-K：`--per-head-k 50`、Gallery 由 `--gallery` 指定 |
| B | 每 Head 搜索 FAISS HNSW：`--neighbor-topk 50 --faiss-index hnsw`；可用 `--faiss-index flat` 精确对照 |
| C | Caption coverage 最低的 `--uncovered-heads 4`；`--attention-min-concentration 1.3` 空间聚焦度指标 |
| D | `--crop-quantile 0.90 --local-threshold 0.1 --global-ratio 0.5`，保存每个 Concept 对应的 bbox |
| E | `--support-neighbors 20 --min-support 2 --support-margin 0.05 --rare-high 0.55` |
| F | `--negative-topk 30 --counterfactual-lambda 0.3`，CLS 邻居中本概念证据较弱的半数作负例 |
| G | `--min-views 2 --view-threshold 0.12 --view-variability 0.2` |
| H | `--rounds 2`，固定 A 检索算子，逐轮更新 Teacher 权重，Student 每轮随机初始化 |

通用概念输出数量 `--max-concepts 3`（默认）、候选池 `--pool-k 30`（默认）、视觉分数过滤 `--threshold`（默认不使用）。为避免读取全部原始图，A/B/E/F 主要使用现有缓存；C/D/G 需要额外遍历图片并执行 DINOv2 GPU forward；B/E/F 的 FAISS 索引和邻居缓存位于**各自方案 outputs 下**。

## 5. 输出结构（每个方案独立）

以 A 为例：

```text
outputs/local_cocu_talk2dino/A_direct_retrieval/
├── concepts_by_image.json                 # 每张图 ID、原始 captions、Concept 和可追溯信息
├── matched/matched_concepts_detailed.json # 兼容原 local_cocu_novel/build_train.py
├── settings.json                          # 参数、Teacher 来源、统计
├── train_augmented.pth                    # 用于原 train.py
├── train_manifest.json                    # 5+up-to-3 构建结果
└── eval/{voc20,context59,ade150,coco_stuff}/

weights/local_cocu_talk2dino/A_direct_retrieval/
└── vitb_mlp_infonce_local_cocu_talk2dino_A.pth
```

`concepts_by_image.json` 格式示意（分数是模型分数；不是人工真值或出现概率）：

```json
{
  "579291": {
    "image_id": 579291,
    "scheme": "A",
    "raw_captions": ["原始 caption 1", "原始 caption 2", "原始 caption 3", "原始 caption 4", "原始 caption 5"],
    "concepts": [
      {"concept": "railway platform", "score": 0.37, "best_head": 4}
    ],
    "num_concepts": 1,
    "diagnostics": {}
  }
}
```

上述 JSON 为 **格式示意**，不是已实际提取的训练结果。B/E 会保留图像近邻；C 保留每个 Head 的 caption coverage；D 保留 `bbox_xyxy_448`；E 留存支持率、支持邻居 ID；F 保留 counterfactual 负例均值；G 保留 `view_scores`。H 放在 `H_iterative_bootstrap/round_1/`、`round_2/` 等独立子目录。

评估工具沿用项目 `model.proj_name=...` 规则，原 `main.py` 从顶层 `weights/` 搜索模型。**训练完权重会移动到用户要求的子目录，并在 `weights/` 下创建指向该权重的符号链接**；顶层只留下兼容性链接，不存在双份 checkpoint。

## 6. 科研设计的局限和消融建议

1. Caption 中未写出的词 **不等于** 图像真的出现该物体；此代码采用大小写/词形归一化连续词匹配排除显式重复，未证明语义真值，仍需对筛选 Concept 抽样做人审。
2. Attention Map 不等于语义 Mask。C 的 attention concentration 只检查是否具有集中的空间响应，不能证明低覆盖 Head 对应真实目标；D 的 bbox 基于 **448×448 归一化预处理图**，不直接对应原图像像素。
3. B 使用跨 Head 最大余弦相似度（不限 Head 编号）。FAISS HNSW 是 ANN 近似结果，对照请运行 `--faiss-index flat`。
4. E 是邻居证据一致性，不是统计独立验证；F 的 CLS 难负例可能包含真实目标；G 三种视图只包括保持所有目标可见的全图扰动，不包括需要额外判断可见性的 RandomCrop。
5. A 与 B 的不同不仅包含图像检索策略，还包含候选集来源；E/F 在 B 上附加筛选，应优先进行各自有/无筛选的 matched candidate 对照。
6. H 每轮都是从头训练 Student，概念 Teacher T0→T1→…；不保证性能或 Concept 质量单调改善。建议在固定验证集合对比 1/2/3 轮，并审查相似但错误的伪标签是否自增强。
7. 当前代码是科研实现，**尚未在你的 GPU/COCO 数据上完成端到端运行**。正式实验前先用小子集核验候选、训练集统计和分割权重加载。

## 7. 检查与测试

```bash
python -m unittest discover -s experiments/local_cocu_talk2dino/tests -v
python -m compileall -q experiments/local_cocu_talk2dino
```

如果修改其中一个发现方案，可独立运行其他方案的构建/训练/评估，无需接触仓库主模块。原训练例程会在 `weights/` 暂时创建 checkpoint；转换为子目录与符号链接后，不影响原始 eval。
