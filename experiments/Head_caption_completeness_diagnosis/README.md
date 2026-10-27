# Head–Caption 语义完整性诊断基准（S0–S6）

**定位**：独立的 BSC-CR 完整性判定审计项目。仅评估 `T_(c,k)`、`V_(h,k)` 与双向覆盖公式的稳定性；**不训练 Student、不生成 edited train.pth、不运行 OVSS，也不修改原始训练/分割代码**。

仓库：<https://github.com/Ocean-kang/Talk2DINO>。新目录：`experiments/Head_caption_completeness_diagnosis/`。复用原代码：

- `experiments.Head_Match.common.teacher.TeacherBank`：CLIP ViT-B/16、冻结的 Talk2DINO Teacher projector、CLIP 文本编码。
- `experiments.Head_caption_completeness.core`：根路径、Gallery 读取、词汇标准化、原来安全的 Concept ADD 格式。
- 使用原始 `train.pth` 中 `[12,768]` Heads、`[512]` Caption embedding，**Teacher 原始 argmax Head 固定不变**。从不修改输入缓存。
- 使用原实验的 `prepare.py` 生成的 CLIP image feature cache；**不强制重算图像特征，也不需要 CLIP image neighbors 文件**。

依赖：原 Talk2DINO 环境能导入 `torch`、`clip`、`src.model`（环境/依赖与旧实验相同），无需新网络模型或 LLM。

## 1. 文件放置

从 Talk2DINO 仓库根目录解压 ZIP，它内置 `experiments/Head_caption_completeness_diagnosis/` 目录：

```bash
cd /mnt/sda/master/code/oymk/Talking2dino/Talk2DINO
unzip -o /path/to/Head_caption_completeness_diagnosis.zip
python -m unittest discover -s experiments/Head_caption_completeness_diagnosis/tests -v
```

需要的已有数据路径（可用 CLI 参数覆盖）：

```text
Talk2DINO/
├── outputs/coco2014_b14/train.pth                 # 原样本 [12,768] + 5 captions 特征
├── weights/vitb_mlp_infonce.pth                   # 冻结的 Teacher
├── configs/vitb_mlp_infonce.yaml
├── experiments/local_cocu/MSCOCO14_train_gallery.json
├── outputs/Head_caption_completeness/clip_images.pth # 原 BSC-CR prepare 结果
└── experiments/Head_caption_completeness_diagnosis/
```

若 `clip_images.pth` 尚不存在，先**复用原实验的数据准备步骤**（需 JPEG）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.Head_caption_completeness.prepare \
  --base-train outputs/coco2014_b14/train.pth \
  --image-root data/coco2014/train2014 \
  --cache outputs/Head_caption_completeness/clip_images.pth \
  --neighbors outputs/Head_caption_completeness/clip_neighbors.json \
  --gpu-device cuda --batch-size 96 --workers 8 --topk 10
```

原 `prepare.py` 默认同时构建 KNN；新诊断只消费其中的 CLIP 图像缓存。如果原缓存由不同图像子集生成，先核对完整图像 ID 覆盖，不要拿不完整缓存假装全量。

## 2. 运行

以下命令都从仓库根目录运行。单 GPU 一次跑完 S0–S6（不重复提图像特征）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.Head_caption_completeness_diagnosis.run \
  --device cuda --limit 800 --captions-per-image 1 --seed 123 \
  --methods S0,S1,S2,S3,S4,S5,S6 \
  --output outputs/Head_caption_completeness_diagnosis/run800
```

CPU 单元/集成冒烟（CLIP Teacher 模型在 CPU 上会慢）：

```bash
python -m unittest discover -s experiments/Head_caption_completeness_diagnosis/tests -v
CUDA_VISIBLE_DEVICES=0 python -m experiments.Head_caption_completeness_diagnosis.run \
  --limit 20 --captions-per-image 1 --max-phrases 24 \
  --output outputs/Head_caption_completeness_diagnosis/smoke20
```

可直接运行脚本：

```bash
bash experiments/Head_caption_completeness_diagnosis/run_matrix.sh 0 800
```

更全面的测试可使用 `--captions-per-image 5`，同时建议先保持 `--limit 800`，避免对整个 COCO 训练集做昂贵的 Phrase-CLIP 编辑重编码。其它 GPU 可并行跑不同 seed，使用不同 `--output`，不要两个进程共用输出目录。

独立改变 Gallery/权重/路径：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.Head_caption_completeness_diagnosis.run \
  --base-train /your/data/train.pth \
  --gallery /your/gallery.json \
  --teacher weights/vitb_mlp_infonce.pth \
  --config configs/vitb_mlp_infonce.yaml \
  --clip-images /your/cache/clip_images.pth \
  --device cuda --limit 800 --output outputs/Head_caption_completeness_diagnosis/custom
```

添加局部别名映射（可选）：`--aliases mapping.json`，JSON 形如 `{"bicycle":["bike"],"person":["human"]}`。不要把含糊的视觉上下位概念一律当成同义词。

## 3. 方案说明与可比性

| 配置 | 文本端 T | 视觉端 V | Mask / 集合 | 作用 |
|---|---|---|---|---|
| S0 | 原 CLIP 全句余弦 + 0/0.8/1 | 原 max-head 背景 + image 混合 | 重现旧 valid 并集 + 强制目标 valid | 对照原异常机制 |
| S1 | 与 S0 相同 | 与 S0 相同 | 固定支持集合 + 双边 UNKNOWN 剔除 | 隔离 UNKNOWN 问题 |
| S2 | 词面 + 手工方向性别名匹配 | 原融合 | 固定双边有效集合 | 显式语义独立性基线 |
| S3 | n-gram Phrase–CLIP + 词面锚点 | 原融合 | 固定双边有效集合 | 局部 CLIP 语义匹配 |
| S4 | 与 S3 相同，增加反事实准入检查 | 原融合 | 固定双边有效集合 | 观察准入比例与准入后方向性；**T 本身不变** |
| S5 | 与 S3 相同 | 单个固定 Head；单 Head 背景分布 | 固定双边有效集合 | 消融 Head/image 融合 |
| S6 | 与 S5 相同 | 与 S5 相同 | 严格同义词成组计算 max coverage | 控制同义 Gallery 重复权重 |

固定 Head 的位置由 Teacher 使用**原始 `ann_feats` 通过未归一化输入投影再 argmax**决定。不同方案不会重选 Head。

S5 同时在记录里输出同图其余 Heads、整图 CLIP 视觉支持，帮助区分当前 Head 负证据和整图缺失。随机 12-head 最大响应和单-head 背景分布分别校准，避免混用。它们是模型分数，不是标注概率。

S3 的 Phrase n-gram 计算是无额外语言模型的近似做法：不保证句法关系、否定语义和自然语言蕴含的完全准确；`--phrase-margin`、`--phrase-offset`、`--unknown-low`、`--unknown-high` 都是需在验证集上预先固定的超参数，不应从测试 mIoU 上挑选。

## 4. 对照干预

每个 Caption 尝试以下操作（不保证所有类型均能构造，因此每组的 n 必须报告）：

- `delete_supported`：选择当前 Head 高证据且已表达的 Concept，删除其字符串表达。删除可能不流畅，视为可解释的合成干预，不可当作图像真值标注。
- `add_high_evidence`：向当前 Caption 添加未表达的高视觉支持 Concept，测试缺失度。不能据此证明其真实存在。
- `add_low_evidence`：添加当前 Head/其余 Heads/整图证据都低的 Concept，测试误添概念识别。**低证据不等于真实不存在**。
- `synonym`：小范围人工确认的同义词替换。
- `neutral_prefix`：添加近中性前缀，测试非目标变化。

只有处于比较双方都已知的 Concept 才会进入 S1–S6 的固定有效集合；S0 特意保留旧实现对 UNKNOWN 的不当处理，以便测量改动的影响。S0–S6 使用同一固定选中 Head、Gallery 与干预字符串。S4 额外统计 `counterfactual_pass`，不将筛选后的准确率与未筛选的直接比较（必须同时报告准入率）。

**特别注意 `D_remove` 的数学方向：** 在其它 T 不变的理想情况下，加入目标 Concept 只有满足 `1 - V_target > D_remove_before` 时，聚合 `D_remove` 才必然上升。因此对 `add_low_evidence` 仅在这个条件成立时计入方向判定分母，条件之外仍输出 raw delta，不能把反向变化简单算成错误。

## 5. 输出目录及指标

```text
outputs/Head_caption_completeness_diagnosis/run800/
├── records.jsonl           # 每个 image+annotation+case+method 的详细逐样本记录
├── metadata.json           # 完整 CLI 超参、缓存/在线一致性检查、样本数量
├── summary.json            # method×case 汇总 + image-level bootstrap CI95
├── summary.csv             # 汇总表，适合 Excel / pandas
└── direction_failures.jsonl # 最多 300 条不满足方向判定的案例
```

主要字段：`target_text_delta`、`target_d_add_contribution`、`off_target_d_add_contribution`、`leakage_ratio`、`delta_add`、`delta_remove`、`direction_success`、`counterfactual_pass`、`target_visual_head`、`target_visual_other_heads`、`target_visual_image`、`raw_embedding_cos_cached_vs_online`、`top_off_target_changes`。

`summary` 同时保留 `n`、`n_comparable`、`target_known_rate`、`direction_n`、`direction_accuracy`、`ci95_delta_add`/`ci95_delta_remove`，不会在 UNKNOWN 和不可构造样本上凑分母。

**推荐先看：** S0→S1 的方向率与有效样本变化；S1→S2→S3 的 off-target drift；S3→S4 的反事实通过率；S3→S5 的视觉支持变化；S5→S6 的词表冗余聚合效果。需结合每组覆盖率解释，避免通过减少有效样本“提升”指标。

## 6. 实验边界 / 已知限制

1. 本包没有运行你的服务器数据，因此报告里**不包含虚构的模型准确率或 mIoU**；需要在服务器实际执行后分析 `summary.csv`。
2. S0 尝试复现先前 `validate_diagnostic.py` 的核心公式和 UNKNOWN 并集，但样本构造采用更谨慎的高/低证据规则，**不保证得到原先 27.35%/31.75% 完全相同的数字**。
3. Caption 已缓存 embedding 与在线重编码的余弦相似度会记录在 metadata 和每条记录中。默认只警告低于 0.995 的样本；可加 `--strict-encoder-consistency` 让其直接报错。
4. S6 只合并少量明确的同义 Gallery 标签，不将上下位概念一律聚合；分组后 `D_add` 的 per-concept 线性分解不再成立，会标为 unavailable。
5. 高/低视觉证据与真实图像语义有差异。建议对小规模样本人工验证目标概念的确存在或不存在后，才能声称真正的语义完整性正确率。
6. 不会生成被“修复”的 training captions；待新的 T/V 判定稳定、可解释后，再单独开展 ADD/REMOVE + 原 Talk2DINO projector 训练和 OVSS。
