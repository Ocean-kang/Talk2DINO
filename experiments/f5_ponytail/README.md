# Talk2DINO F5 Ponytail

本目录对应 Ocean-kang/Talk2DINO 提交 `3777d2e071dd97e6726bd6312f32018221424878`。把 ZIP 解压到现有仓库根目录，得到 `experiments/f5_ponytail/`。继续使用现有的 Talk2DINO 训练环境与 Linux CUDA 分割评估环境。训练沿用 `train.py`、原 projector、InfoNCE、优化器和验证集。分割沿用仓库 MMCV 入口及旧 F5 脚本使用的五组 PAMR 配置。运行器通过进程内 seed 重定向覆盖上游的固定 123。评估临时创建独立的权重别名，结束后清理。

## 输入

从仓库根目录执行命令。以下是默认位置，可用 `python experiments/f5_ponytail/run.py --help` 查看覆盖参数。

| 输入 | 默认位置或准备方法 |
|---|---|
| ViT-B/14 train / val PTH | `outputs/coco2014_b14/{train,val}.pth`。沿用原仓库 README 的 DINO 特征与 CLIP 文本特征提取流程 |
| F Head 缓存 | `outputs/local_cocu/head_features/head_features.pt` |
| F 局部 Head 邻居 | `outputs/local_cocu_talk2dino/F_counterfactual/neighbors.json` |
| F CLS 近邻 | `outputs/local_cocu/retrieval/neighbors.json` |
| F 概念词表 | `experiments/local_cocu/MSCOCO14_train_gallery.json` |
| 官方权重 | `weights/vitb_mlp_infonce.pth` |
| 旧 F5 权重 | 通过 `--old-f5` 指定一份已确定的旧 F5 projector state dict |
| 分割数据 | `data/VOCdevkit/VOC2012`、`data/VOCdevkit/VOC2010`、`data/coco_stuff164k`、`data/ade`，按原仓库准备 split、图像及标注 |

缺少 F 缓存时，在原环境执行下列命令复用原 F 提取和检索。已有缓存可跳过。Head、CLS 邻居和原始 train PTH 应来自同一 COCO train 图像池。若你的 PTH 是子集，请准备同池缓存，避免把其他 split 的 caption 带入检索。

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/local_cocu_talk2dino/F_counterfactual/run.sh \
  --stage mine --gpu 0 --extract-missing-heads --extract-missing-cls
```

若原 F 命令现场抽取了缺失的 Head 或 CLS 缓存，它会将这些文件写到 F 输出的 `source/` 下。构建时按需传入 `--heads outputs/local_cocu_talk2dino/F_counterfactual/source/head_features.pt` 或 `--cls-neighbors outputs/local_cocu_talk2dino/F_counterfactual/source/cls_near/neighbors.json`。如果旧 F 命令因已有概念缓存而跳过检索，追加 `--force-mine` 重新生成。

## 运行

```bash
EXP=experiments/f5_ponytail/run.py
WORK=outputs/f5_ponytail/run1
OLD_F5=/absolute/path/to/your_old_F5.pth

python "$EXP" build --work "$WORK" --gpu 0
python "$EXP" train --work "$WORK" --gpu 0
python "$EXP" eval --work "$WORK" --gpu 0 --old-f5 "$OLD_F5"
python "$EXP" summary --work "$WORK"
```

也可用 `all --work "$WORK" --gpu 0 --old-f5 "$OLD_F5"` 顺序执行。每个工作目录一次只运行一个进程。并行跑不同目录时使用不同 `--port` 和 GPU。构建要求新目录，训练与评估可重跑并复用通过哈希核验的结果。改评估数据或运行环境时使用新工作目录。每个配置保存一个训练 PTH，完整矩阵需预留约八份训练缓存、40 套模型结果和逐图审计的磁盘空间。

小样本构建检查如下。候选与反事实邻居仍从原训练池取，输出训练集只含前 64 个目标图像。

```bash
python "$EXP" build --work outputs/f5_ponytail/smoke --gpu 0 --limit 64 --variants f5_control cf_l03
python -m unittest discover -s experiments/f5_ponytail -v
```

## 消融设计

每张图从 F 的局部 Head 邻居 caption 取词，去除已有概念。保留原 F 的候选预筛，上限 300 词、30 个候选。以冻结的官方权重投影文本。概念目标分数是其与 12 个 Head 的最大余弦相似度，取到最大值的 Head 用于动态门槛和指派。每图最多三个概念，允许零概念。

| 配置 | 通过同 seed 比较的问题 |
|---|---|
| `f5_control` | 重跑 F5 策略。F 的 CLS 邻居取概念得分较低的一半，λ=.3。按 CLIP 空间的概念与原 caption 最小相似度指派 |
| `threshold` | 在上组增加 Head 动态门槛。目标分数至少达到 max(0.10, 同 Head 的最佳原 caption 得分 + 0.02) |
| `head` | 把指派代价改为该 Head 对每条原 caption 的余弦相似度，优先补该 Head 覆盖最弱的 caption |
| `balanced` | 用同一代价求一对一全局最小总成本，减少概念集中到同一条 caption |
| `cf_l03` | 保留前 30 个 CLS 邻居中没有词面提及该概念、且局部证据至少比目标低 0.05 的邻居。得分为目标分数减 λ × 这些邻居的平均最大 Head 相似度。空负例集合的惩罚为零 |
| `cf_l00` / `cf_l01` / `cf_l06` | 固定筛选和指派，分别用 λ=0、0.1、0.6 对比 λ=0.3 |

CLS 近邻提供相似场景，词面过滤与视觉分差提供概念缺席的代理证据。默认参数是预设实验值，五个分割测试集用于最终报告。五个训练 seed 为 123–127。所有组共享 CLIP 长度保护，追加后超出窗口的 caption 保留原文并计数。因此 `f5_control` 是共享输入与保护规则的训练对照，旧 F5 权重单列为固定历史基线。每图保留五条 caption 和原 ID，只重编码实际改写的行。缓存与权重应来自可信来源。

## 产物与解读

- `build.json` 记录输入、配置、数据哈希、概念数分布、指派碰撞和长文本回退数量。`audit.jsonl` 逐图记录原文、Head 覆盖、邻居、候选证据、最终得分、caption 指派与实际文本。
- 每个 seed 的 `train.log`、`train.json`、`model.pth`，以及五个评估的配置、日志和 `metrics.json` 供复核。两套固定权重各跑五个任务一次，结果在 `fixed/`。
- `summary.csv` / `summary.json` 输出五 seed 均值、样本标准差、t 区间、相对两套实测固定基线的百分点差，以及同 seed 消融差值的均值、标准差和区间。固定权重只报告点值。区间反映训练 seed 波动。默认完整矩阵需 40 次训练、210 次分割评估。

