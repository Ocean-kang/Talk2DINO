# BSC-CR / Head-caption completeness (Talk2DINO)

独立实验，不改仓库 `train.py`、`src/model.py`、`src/loss.py`、`src/train_util.py` 或 OVSS 源码。代码放入 `Talk2DINO/experiments/Head_caption_completeness/`。请在 **Talk2DINO 仓库根目录** 执行所有命令。

## 原理/边界

资源限定：CLIP ViT-B/16、冻结 Teacher `weights/vitb_mlp_infonce.pth`、COCO2014 原图/原 5 captions、仓库 DINOv2 12 heads、`MSCOCO14_train_gallery.json`。没有任何 LLM 或额外识别器。FAISS（可选）仅用于向量近邻索引，不是额外模型。

1. **prepare**：冻结 CLIP 提取 COCO14 图像 embedding，CLIP image-image top-k 近邻；如已缓存则复用。需要 COCO train2014 JPEG 原图，`--image-root` 可以指向它或父目录。
2. **diagnose**：冻结 Teacher 逐 Caption 选原 `argmax` Head，Gallery 每个概念算 Head-Teacher 与 CLIP 图像证据；随机抽取其他 COCO 图像构成概念维度错配空分布并以分位数校准。Caption–Concept 结合连续词组识别、CLIP text cosine；相似度处于不确定带时为 `UNKNOWN` 并排除。

   \(D_{add}=\sum_{valid} V(1-T)/\sum_{valid} V\)，\(D_{remove}=\sum_{valid} T(1-V)/\sum_{valid} T\)。这两个值是 Gallery 条件下代理指标，**不是图像的真实语义标注**。

   3 路提候选：Teacher Head-Guided Gallery；CLIP 相似 COCO 图像的 captions（CoCu-Style）；同图其他 captions。仅保留 Gallery 中的候选。为了避免频繁弱编辑，默认每条 Caption 最多接受 **1 个**高置信度增补或删除（满足原方案优先 1–2 个）。保存当前 Head/其他 Head 和五条 Caption 并集的解释信息。
3. **编辑**：KEEP / ADD / REASSIGN_RECOMMENDED / REMOVE / RECHECK。REASSIGN 在当前实验中**只作为诊断记录**，不强制更改原版 Student `argmax`，否则会改变训练算法。ADD 只写 `; also visible: concept.`，不编造关系。REMOVE 只对可安全拆除的 **句尾并列名词短语**执行，且必须在其他四条 Caption 中保留这个概念、其他 heads/整图证据都弱。其余 KEEP。
4. **反事实准入**：冻结**原来的 Head** 重算前后 \(D\)、Teacher 同 Head 匹配、五 Caption 词面语义集合不减；ADD 还需优于一个低证据 Gallery 概念追加（不是已证实的假标签）。不通过就 KEEP。这里不以重新选择 Head 导致的伪得分提升冒充改善。
5. **build**：`E0` 原始 PTH；`E1` ADD；`E2` REMOVE；`E3` 双向编辑；`E4` 无完整性门槛，但仍遵守反事实/语法安全检查，并尽量对齐 `E3` 的 ADD/REMOVE 数目。所有方案仍是每张图 5 条原始 `annotation_id`；只有修改项用冻结 CLIP 重新编码 `ann_feats`；全部图像 Head 不动。
6. **train**：调用原仓库 `train.py`，配置 `configs/vitb_mlp_infonce.yaml`，仍是原始 Student MLP / max_score / InfoNCE / optimizer / 100 epochs，Teacher 不参与 Student 训练。借助 `train_seed.py` 将原始训练器两处固定 `set_seed(123)` 只在内存中重定向为该次 seed；绝不复用 Teacher 参数来初始化 Student。
7. **eval**：调用原仓库 `src/open_vocabulary_segmentation/main.py --eval` 五次，数据集配置与历史方案相同；动态选择 OS 空闲端口 `MASTER_PORT`，仅 EADDRINUSE 时更换端口重试；不同实验的 `proj_name` 唯一，兼容并行。

### 数据集及基线

| key | 原版目标（mIoU） | eval cfg / base cfg |
|---|---:|---|
| voc20 | 88.54 | voc/dinotext_voc_vitb_mlp_infonce.yml / voc/eval_voc_pamr.yml |
| voc | 65.77 | voc_bg/dinotext_voc_bg_vitb_mlp_infonce.yml / voc_bg/eval_voc_bg_pamr.yml |
| coco_stuff | 30.23 | stuff/dinotext_stuff_vitb_mlp_infonce.yml / stuff/eval_stuff_pamr.yml |
| ade20k | 22.53 | ade/dinotext_ade_vitb_mlp_infonce.yml / ade/eval_ade_pamr.yml |
| context59 | 42.42 | context/dinotext_context_vitb_mlp_infonce.yml / context/eval_context_pamr.yml |

`voc20` 和 `voc` **不是相同配置**（后者走 `voc_bg`）；`coco_stuff`、ADE20K、Context59 同理。基线是提供的固定参考数值，不是本包评估得到的新数据。

## 服务器安装

将 ZIP 在 Talk2DINO 根目录解压，zip 根路径已含 `experiments/Head_caption_completeness/`：

```bash
cd /mnt/sda/master/code/oymk/Talking2dino/Talk2DINO
unzip -o /path/to/Head_caption_completeness.zip
# 激活你以前能成功跑 Talk2DINO+mmcv+mmseg 的环境
python -m unittest discover -s experiments/Head_caption_completeness/tests -v
```

无需新增模型。建议 `pip install faiss-cpu`（纯向量检索加速）。如果不装 FAISS，会用分块矩阵乘法实现精确 CLIP KNN。GPU 显存不足可调整 `--batch-size`，数据加载的 `--workers`，或者 `--device`。

项目预期文件，实际存在性请在服务器上核对：

```text
Talk2DINO/
├── configs/vitb_mlp_infonce.yaml
├── weights/vitb_mlp_infonce.pth
├── outputs/coco2014_b14/train.pth
├── outputs/coco2014_b14/val.pth
├── experiments/local_cocu/MSCOCO14_train_gallery.json
├── data/coco2014/train2014/*.jpg   # 或你的实际 JPEG 路径
└── experiments/Head_caption_completeness/
```

上面默认的 `train.pth` 是仓库历史实验路径；**不生成新的 DINO Heads**，因为要复用已预提取的图像 features。

## 1. 冻结 CLIP 图像特征缓存 + CoCu 候选近邻

一次性在全量 COCO14 训练集执行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.Head_caption_completeness.prepare \
  --base-train outputs/coco2014_b14/train.pth \
  --image-root data/coco2014/train2014 \
  --cache outputs/Head_caption_completeness/clip_images.pth \
  --neighbors outputs/Head_caption_completeness/clip_neighbors.json \
  --gpu-device cuda --batch-size 96 --workers 8 --topk 10
```

**不要用 DINO-CLS 检索缓存代替 CLIP 图像近邻**，否则会改变 BSC-CR Step 2 的 CoCu-style 设定。缓存首次生成会较慢；再次运行会跳过同输入的 CLIP 图像编码，但会重新生成 KNN。

## 2. 小样本诊断验证 (800 images)

在正式完整性编辑训练前，先验证原始 Head 固定时的方向性：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.Head_caption_completeness.validate_diagnostic \
  --limit 800 --device cuda \
  --report outputs/Head_caption_completeness/validation/report.json
```

其中 `delete_existing` 期望 \(D_{add}\) 增，`add_wrong` 期望 \(D_{remove}\) 增，`synonym` 期望变化小。输出**实际样本数、平均差值和方向正确比例**；仅作诊断可行性检验，不是监督语义正确率。若趋势不成立，优先调整 `--text-low` / `--text-high` / `--head-margin` / `--image-margin` / `--add-gate` / `--remove-gate`，而不是直接跑五组训练。

## 3. 全量离线诊断 + E0–E4 编辑策略

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.Head_caption_completeness.diagnose \
  --base-train outputs/coco2014_b14/train.pth \
  --gallery experiments/local_cocu/MSCOCO14_train_gallery.json \
  --teacher weights/vitb_mlp_infonce.pth \
  --config configs/vitb_mlp_infonce.yaml \
  --clip-images outputs/Head_caption_completeness/clip_images.pth \
  --clip-neighbors outputs/Head_caption_completeness/clip_neighbors.json \
  --output outputs/Head_caption_completeness/discovery \
  --device cuda
```

可先加 `--limit 800` 做 smoke test。但**不能用只诊断 800 张的 JSONL 构建全量训练数据**：`build.py` 默认检查 annotation_id 完整覆盖并拒绝。小样本训练必须显式 `--subset-train`，不能与全量 baseline 混为一谈。

重要结果：

```text
outputs/Head_caption_completeness/
├── clip_images.pth              # CLIP 图像 vectors
├── clip_neighbors.json          # image_id -> CLIP top-k 邻居
├── validation/report.json       # 删除/错误添加/同义变化
├── discovery/
│   ├── metadata.json            # 各方案修改数及诊断阈值
│   ├── diagnostics.jsonl        # 前 audit-images 图详细覆盖/候选证据，默认 1000
│   └── E0.jsonl ... E4.jsonl    # **全部** annotation_id: 修正结果，fixed-head 分数
└── training/E1/train.pth ... E4/train.pth
```

`--audit-images 999999` 可保存所有图像的详细候选决策（文件非常大）。全量 Caption 最终选择和数值则总在 `E*.jsonl` 中。

## 4. 组装 train.pth / 调用原版训练 / 五项 OVSS

**某一方案单独运行：**

```bash
# --stage build | train | eval | all
CUDA_VISIBLE_DEVICES=0 python -m experiments.Head_caption_completeness.pipeline \
  --variant E3 --seed 123 --stage all --gpu 0
```

**五个消融，每个不同 GPU 并行：**（按你的 GPU 数量修改；同卡并行可能 OOM）

```bash
python -m experiments.Head_caption_completeness.pipeline --variant E0 --seed 123 --stage all --gpu 0 > /tmp/bsc_e0.log 2>&1 &
python -m experiments.Head_caption_completeness.pipeline --variant E1 --seed 123 --stage all --gpu 1 > /tmp/bsc_e1.log 2>&1 &
python -m experiments.Head_caption_completeness.pipeline --variant E2 --seed 123 --stage all --gpu 2 > /tmp/bsc_e2.log 2>&1 &
python -m experiments.Head_caption_completeness.pipeline --variant E3 --seed 123 --stage all --gpu 3 > /tmp/bsc_e3.log 2>&1 &
python -m experiments.Head_caption_completeness.pipeline --variant E4 --seed 123 --stage all --gpu 4 > /tmp/bsc_e4.log 2>&1 &
wait
```

如只想训练而不立即评估，用 `--stage train`。首次对 E1–E4 执行训练前需 `--stage build`，`all` 已包含构建并加文件锁：相同方案不同 seeds 不会同时破坏共同的数据集。**训练数据构建是 seed 不变的，随机种子仅用于 Student 训练。**

支持自选数据集：

```bash
python -m experiments.Head_caption_completeness.pipeline --variant E3 --seed 124 \
  --stage eval --gpu 0 --tasks voc20,voc,coco_stuff,ade20k,context59
```

五个 seeds 独立训练，逐个或在不同 GPU 并行执行：

```bash
for SEED in 123 124 125 126 127; do
  python -m experiments.Head_caption_completeness.pipeline \
     --variant E3 --seed "$SEED" --stage all --gpu 0
done
```

所有 ckpt 在：

```text
weights/Head_caption_completeness/E3/seed_123/vitb_mlp_infonce_bsc_cr_E3_seed123.pth
```

原分割代码只根据 `proj_name` 读取 `weights/`，所以 launcher 自动生成唯一的 `weights/vitb_mlp_infonce_bsc_cr_E3_seed123.pth` 软链接指向上面的 checkpoint。不会覆盖 `weights/vitb_mlp_infonce.pth`。一次只需在当前环境跑一个同名实验/seed；多个**不同**方案/seed 的评估可以并行。

每个实验/seed 存：

```text
outputs/Head_caption_completeness/runs/E3/seed_123/
├── train.log
└── eval/
    ├── results.json
    └── voc20/metrics.json, eval.log, eval_port_attempt_*.log
       # voc, coco_stuff, ade20k, context59 同结构
```

## 5. 汇总五个随机种子

```bash
python -m experiments.Head_caption_completeness.summarize \
 --seeds 123,124,125,126,127 --variants E0,E1,E2,E3,E4
```

汇总输出：`outputs/Head_caption_completeness/summary/mean_std.csv` 和 `.json`，含 mIoU 均值、**样本**标准差、相对你提供的 Baseline 差值；没有运行完所有测试不会填充虚假数据。可以使用 `--allow-missing` 查看阶段性已完成的组合。

## 实验质量检查

- 确保诊断 `metadata.json` 中 `E1_ADD`/`E2_REMOVE`/`E3_*` 修改数有意义；REMOVE 很保守时出现 0 是一个真实结果，不应虚构修改。
- 调参只基于独立验证子集和扰动结果；**不要根据分割测试 mIoU 反复挑选完整性阈值**。
- 比较 E3 与 E4 的实际编辑数：E4 只在候选足够时严格匹配，并在 metadata 中保存不足配额。
- `REASSIGN_RECOMMENDED` 是分析标签，不会改变原来的 `argmax` 训练；若要训练期强制 Head 更换，那应是独立实验。
- 同图其他四 captions 保留原表达不等于真实图像含有该概念，尤其不得把 Teacher/CLIP 分数当作 Ground Truth。
- 编码缓存、随机抽样空分布、Teacher 权重、Gallery、种子、完整命令都应冻结后报告；报告原始对照和五数据集均值/标准差。
- `python -m unittest discover -s experiments/Head_caption_completeness/tests -v` 是静态/CPU 层检查；还需要服务器上执行准备、诊断和一次端到端 OVSS 才算 GPU 集成通过。

### 附：一键多实验 / 多 Seed / 多 GPU 队列

已包含 `launch_matrix.sh`。同批次保证一个 GPU 只执行一个训练或评估进程，批次内并行，失败即停止：

```bash
bash experiments/Head_caption_completeness/launch_matrix.sh \
  E0,E1,E2,E3,E4 123,124,125,126,127 0,1,2,3 all
```

如果只有一张卡，第三参数写 `0`，就会串行运行；如果有两张卡写 `0,1`。注意 **先准备好完整的 CLIP 缓存和离线 discovery/E1-E4.jsonl 文件**，launcher 不负责重复扫描图像，避免每个 seed 都重新运行耗时的语义挖掘。
