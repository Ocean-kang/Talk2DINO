# Head_Match — Progressive Head-Aware Semantic Bootstrapping (Talk2DINO)

**Ponytail / 旁路实现**：只需将本文件夹加入原仓库的 `experiments/Head_Match/`，原 `train.py`、`configs/`、`src/`、`src/open_vocabulary_segmentation/` 均不修改。适配已核查的 `Ocean-kang/Talk2DINO` 主分支（2026-10-23）。

## 0. 方法不变量

- DINOv2 ViT-B/14-reg 视觉特征**直接复用** `outputs/coco2014_b14/train.pth` 每图的 `disentangled_self_attn` `[12,768]`；不额外前向 DINO。Raw Caption CLIP 向量复用每个原始 annotation 的 `ann_feats` `[512]`。
- 初始教师 `weights/vitb_mlp_infonce.pth`，后续轮次使用上一轮学生。文本编码使用原 `clip` ViT-B/16；Projector **调用仓库 `src.model.ProjectionLayer`、加载 `configs/vitb_mlp_infonce.yaml`**。
- 每图保持原 **5** 条 caption，不新增 5+K 独立 caption。用 Head–Caption–Concept 决策只修改需要补全的 Caption，保留原始 annotation IDs；更改后的 `ann_feats` 通过原 CLIP ViT-B/16 重新编码；不改原有图像特征。
- **每轮 Student 从随机初始化训练**，由仓库原 `python train.py` 与 `src.train_util` 进行 `infonce` 学习。Teacher 仅用于离线构建训练文本，不参与 Student 的反向传播。Student 训练内部仍采用原 `alignment_strategy=max_score` (argmax)。**Soft Assignment 是离线选择机制**，不是修改了原版 Student 损失。
- 每轮从固定 Raw Captions 开始生成新文本（不滚动追加上一轮 Caption）。新 Teacher 重新选择 Head–Caption 权重以及候选 Concept；配置 Gallery 时还会在完整 Gallery 上重新进行每个 Head 的 Top-K 提取。
- 选择出的 Concept **不等于语义真值**。Attention Heads 与 Caption 也不是确定的语义 Mask，必须验证人工审核样本和最终分割效果。

## 1. 四种实验

| 方案 | 目录 | 离线监督构建逻辑 | Round 更新 |
|---|---|---|---|
| **A** Progressive Completion | `A_progressive` | 用 Soft Head 分配，基于候选 Concept 覆盖率门控，按 Head 支持与 CLIP 新颖度补全 | T0 → T1 → T2 |
| **B** Confidence Guided | `B_confidence` | A + 视觉相似度阈值、新颖度与反事实补全文本的匹配增益阈值 | T0 → T1 → T2 |
| **C** Co-Evolution | `C_coevolution` | B + 每接受一个 Concept 就重新计算 Head 软分配，再重新给剩余候选排序；限制 Head 分布偏移 | T0 → T1 → T2 |
| **D** Multi-Teacher Consensus | `D_multi_teacher` | B + 当前与历轮 Teacher 独立投票达标（首轮只有 T0，明确作为 warm-up） | {T0,T1,...} → T2 |

所有方案支持 `--assignment soft`（默认）或 `--assignment argmax` 进行选头消融。

**覆盖率的含义**：固定 Teacher 对当轮候选 Gallery 的 Head 支持分数形成相对权重，计算 Raw Caption 对相对高权重 Concept 的词面 / CLIP 相似度覆盖；这个数值只是候选池内部的代理指标，不是图像中全部真实语义的覆盖率。B/C/D 使用补全文本的 CLIP→Teacher 投影计算 Head 固定权重下的分数变化，也不是人工真实度。默认阈值仅供初试，建议通过抽样人工标注来校准。

**效率提醒**：B/C/D 的反事实筛选对候选完整句子运行 CLIP 文本 Encoder，尤其是 COCO14 全量训练可能较慢。先 `--limit 100 --subset-train` 做闭环冒烟，检查 manifest 与 Head 分配，再运行全量实验。

## 2. 仓库中的真实依赖文件与目录

从 **Talk2DINO 根目录**执行命令。将本 ZIP 解压后的 `experiments/Head_Match/` 置于现有 `experiments` 下：

```text
Talk2DINO/
├── experiments/
│   ├── local_cocu_talk2dino/                   # 旧实验，不修改
│   └── Head_Match/                            # 本实验
│       ├── pipeline.py                        # 一站式 stage/round 控制
│       ├── run.sh
│       ├── common/{core,io,teacher,discovery,build_train}.py
│       ├── A_progressive/run.sh
│       ├── B_confidence/run.sh
│       ├── C_coevolution/run.sh
│       ├── D_multi_teacher/run.sh
│       └── tests/test_core.py
├── weights/vitb_mlp_infonce.pth                # T0 冻结 Teacher
├── configs/vitb_mlp_infonce.yaml               # 原始 Projector + InfoNCE 配置
├── outputs/coco2014_b14/train.pth              # 训练缓存 (包含原始五条 captions)
├── outputs/coco2014_b14/val.pth                # 验证缓存
└── experiments/local_cocu/MSCOCO14_train_gallery.json  # 默认全局候选概念 Gallery
```

**可选候选源**：之前 Local-CoCu 实验的 `concepts_by_image.json`，例如：

```text
outputs/local_cocu_talk2dino/F_counterfactual/concepts_by_image.json
outputs/local_cocu_talk2dino/A_direct_retrieval/concepts_by_image.json
```

`--concepts` 可重复输入，支持旧 JSON 里的 `candidate_concepts`、`curated_concepts`、`concepts`、`selected_concepts`。如果不存在旧 JSON，**默认 Gallery 已足够启动**，由每轮 Teacher 从 Gallery 按 Head 重新提取 Concept。Gallery 默认来自仓库 `experiments/local_cocu/MSCOCO14_train_gallery.json`，可以用 `--gallery /path/to/other_gallery.json` 更换。候选集合质量直接影响结果。

分割评估仍然使用原 Talk2DINO 环境，包括 `data/VOCdevkit/`、`data/ade/`、`data/coco_stuff164k/` 等原评估配置所指向的数据集。若你此前 baseline 的五项分割已成功运行，本项目复用同一环境及分割数据，无需另行转换。

## 3. 配置与环境

```bash
cd /mnt/sda/master/code/oymk/Talking2dino/Talk2DINO
# 按你实际环境启用，例如：
conda activate /home/master/software/oymk/talk2dino

# 验证原依赖（无需安装新的特殊包）
python -c 'import torch, clip, tqdm; from src.model import ProjectionLayer; print("deps ok")'
python -m unittest discover -s experiments/Head_Match/tests -v
python -m compileall -q experiments/Head_Match
```

脚本命令均从根目录执行；原版 `train.py` 硬编码设置 Seed=123，本实验不会假称 `--seed` 能让原训练获得不同随机种子。若要 5 seeds，需先开发独立且正确的 seed 注入适配。`CUDA_VISIBLE_DEVICES` 由 `--gpu` / `run.sh` 的第三个参数指定。

## 4. 一键运行两轮（四方案）

```bash
# 先看计划，不需要 GPU / COCO 数据
bash experiments/Head_Match/run.sh A all 0 2 --dry-run

# A - Progressive Semantic Completion
CUDA_VISIBLE_DEVICES=0 bash experiments/Head_Match/run.sh A all 0 2

# B - Confidence-Guided
CUDA_VISIBLE_DEVICES=0 bash experiments/Head_Match/run.sh B all 0 2

# C - Head–Caption Co-evolution
CUDA_VISIBLE_DEVICES=0 bash experiments/Head_Match/run.sh C all 0 2

# D - Multi-Teacher Consensus
CUDA_VISIBLE_DEVICES=0 bash experiments/Head_Match/run.sh D all 0 3
```

每个 `all` 严格顺序执行：`round_1 mine → build → train → eval → round_2 mine → build → train → eval ...`。D 方案第 1 轮只有 T0，所以是单 Teacher warmup；第 2 轮开始至少需要两名 Teacher 才进行一致性筛选。

**复用旧 F 候选池 + 动态 Gallery 发现**：

```bash
bash experiments/Head_Match/run.sh B all 0 2 \
  --concepts outputs/local_cocu_talk2dino/F_counterfactual/concepts_by_image.json \
  --gallery experiments/local_cocu/MSCOCO14_train_gallery.json
```

**不使用全局 Gallery，仅重评旧候选（这不等于重新发掘新的 Concept）**：

```bash
bash experiments/Head_Match/run.sh A all 0 2 \
  --gallery '' \
  --concepts outputs/local_cocu_talk2dino/A_direct_retrieval/concepts_by_image.json
```

## 5. 分阶段运行 / 小规模调试

```bash
# 只对前 100 张图完整进行 Round 1 冒烟验证（明确训练集人口从 COCO 全量改为 100 张）
bash experiments/Head_Match/run.sh B all 0 1 --limit 100 --subset-train

# 分阶段运行第 1 轮
bash experiments/Head_Match/run.sh A mine 0 2 --round 1
bash experiments/Head_Match/run.sh A build 0 2 --round 1
bash experiments/Head_Match/run.sh A train 0 2 --round 1
bash experiments/Head_Match/run.sh A eval 0 2 --round 1

# 第 1 轮已完成：再构建 / 训练 / 评估第二轮
bash experiments/Head_Match/run.sh A all 0 2 --round 2

# 已训练完成，仅重新执行五项 Segmentation Eval
bash experiments/Head_Match/run.sh A eval 0 2 --round 1 --force-eval
```

注意：小样本训练得到的模型和你在 MSCOCO14 全量上训练的 baseline **训练规模不一致**，不能用其 mIoU 直接判断方法的效果。全量正式运行前应删除/另外隔离小样本输出（比如另建 `outputs/Head_Match` 和 `weights/Head_Match` 的运行副本），或在正式数据上使用 `--force-mine --force-build --force-train --force-eval`。**同一个输出目录只能代表一套固定输入和超参数**；改变阈值时也应使用独立实验副本，以防缓存混淆。

## 6. 五个与 Raw Baseline 一致的 Segmentation Evaluations

| task key | 训练基线 mIoU | eval config | eval_base config |
|---|---:|---|---|
| `voc20` | 88.54 | `voc/dinotext_voc_vitb_mlp_infonce.yml` | `voc/eval_voc_pamr.yml` |
| `voc` | 65.77 | `voc_bg/dinotext_voc_bg_vitb_mlp_infonce.yml` | `voc_bg/eval_voc_bg_pamr.yml` |
| `coco_stuff` | 30.23 | `stuff/dinotext_stuff_vitb_mlp_infonce.yml` | `stuff/eval_stuff_pamr.yml` |
| `ade20k` | 22.53 | `ade/dinotext_ade_vitb_mlp_infonce.yml` | `ade/eval_ade_pamr.yml` |
| `context59` | 42.42 | `context/dinotext_context_vitb_mlp_infonce.yml` | `context/eval_context_pamr.yml` |

分割命令一律转发原 `src/open_vocabulary_segmentation/main.py --eval --eval_cfg ... --eval_base_cfg ... --opts model.proj_name=...`；`weights/` 根目录自动创建指向新权重的**符号链接**，兼容现有 Projector 加载路径。结果从 `segmentation_results/<proj_name>.json` 读取，使用 mIoU 百分点与上述 baseline 比较。

## 7. 输出与审计

```text
outputs/Head_Match/B_confidence/round_1/
├── selection.json                       # image_id、5 个 annotation_id、raw/aug captions、Head 分配及评分
├── selection_stats.json                 # 补全比例、概念数量、Teacher 路径和所有参数
├── train_augmented.pth                  # 原 train.py 的训练特征
├── train_manifest.json                  # 改动的 annotations 和原始图像 ID 集合
├── train.log / build.log
└── eval/
    ├── voc20/{eval.log,metrics.json}
    ├── voc/{eval.log,metrics.json}
    ├── coco_stuff/{eval.log,metrics.json}
    ├── ade20k/{eval.log,metrics.json}
    ├── context59/{eval.log,metrics.json}
    └── summary.json                     # 五套 mIoU、baseline 与 delta
weights/Head_Match/B_confidence/round_1/
└── vitb_mlp_infonce_headmatch_B_r1.pth
weights/vitb_mlp_infonce_headmatch_B_r1.pth  # 符号链接，不额外复制权重
```

每轮数据都会保存 `teacher_weights` 来源，因此可以分析 Teacher 迭代是否真正导致 Concept 更换、Head 重新分配及样本补全比例变化。

## 8. 重要参数与消融

```bash
# 保留离线硬 Argmax 做 head matching 消融
bash experiments/Head_Match/run.sh A all 0 2 --assignment argmax

# Soft Assignment: Top-4 heads / temperature 0.07
bash experiments/Head_Match/run.sh A all 0 2 --assignment soft --top-heads 4 --temperature 0.07

# 更严格的概念选择（需要调整验证阈值）
bash experiments/Head_Match/run.sh B all 0 2 \
  --min-visual 0.3 --coverage-threshold 0.5 --max-concepts 1 --min-delta 0.0

# C：Head 分布变化上限（total variation distance）
bash experiments/Head_Match/run.sh C all 0 2 --max-head-shift 0.45

# D：至少两个 Teacher 一致（从第 2 轮生效）
bash experiments/Head_Match/run.sh D all 0 3 --min-votes 2
```

**避免缓存污染**：`--force-mine` 会重算选择结果，随后务必 `--force-build --force-train --force-eval`；已有权重与结果只在路径一致时可复用。建议每一个消融在独立工作树或复制并更改 pipeline 输出前缀后运行，避免共享目录。

## 9. 验证状态与限制

- 已验证：算法单元测试、Python 编译、pipeline dry-run；已核对 GitHub 原配置文件和原分割任务路径存在。
- **尚未运行**：真实 COCO 2014 数据的完整 Concept 发现、100 epoch Teacher/Student 训练、五项 GPU segmentation eval，所以不存在真实的新 mIoU 结果。
- `B/C/D` 的 Counterfactual 评分为完整补充句的 Teacher 相似度变化，并非人工真实语义检验。真实正确性建议采样标注、人工检查。
- `A/B/C/D` 的新文本使用确定性的 `Also visible: ...`，不推断动作或关系，减少语言生成的幻觉；它是保守语义融合，不等价于人工 GPT Rich Caption。
- Round 1 之后性能**不保证单调提升**。应把选词准确性、Head assignment 变化、各轮 mIoU 与五数据集平均/方差一起报告。`--min-delta`、CLIP covered 阈值需要在训练外的校准集确认。
- **多次 Seed** 未实现，因为仓库原版 `train.py` 固定 seed=123；如需要严格 5 seeds，必须开发保持兼容性的 Seed 实验入口，不要只是重复执行脚本。
