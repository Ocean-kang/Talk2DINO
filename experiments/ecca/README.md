# Talk2DINO：A 诊断 → B ECCA → 有条件的 C 实验

本包只新增 `experiments/ecca/`；不修改 `src/`、根目录 `train.py` 或原评估配置，不包含环境安装指令、不下载模型、不向 GitHub 写入内容。

## 1. 放置位置与资源

在 **Talk2DINO 仓库根目录**解压，得到：

```text
Talking2dino/Talk2DINO/
├── src/                            # 复用原实现
├── configs/vitb_mlp_infonce.yaml
├── outputs/coco2014_b14/
│   ├── train.pth                   # 你已有的完整训练特征
│   └── val.pth                     # 你已有的独立验证特征
└── experiments/ecca/
    ├── settings.sh                 # 首先修改这里的本地路径
    ├── run.sh
    ├── diagnose.py / make_manifest.py
    ├── prepare.py / recovery.py
    ├── train.py
    ├── evaluate.py / eval_worker.py
    ├── summarize.py
    ├── runtime.py / common.py
    ├── offline_guard/
    └── tests/
```

在现有 Talk2DINO 环境中操作，不新建环境、不安装/升级/降级任何包。CLIP、DINOv2、Teacher 均从显式本地文件读取。需要现有 DINOv2 源码目录（通常已有 torch.hub 缓存）。不导入 spaCy/NLTK，不下载语言资源。离线保护会阻止隐式远程下载。

先改 `experiments/ecca/settings.sh` 的实际路径：

| 变量 | 用途 |
|---|---|
| `TRAIN_PTH` | 默认 `outputs/coco2014_b14/train.pth`，保留其中全部图像与全部原始 captions |
| `VAL_PTH` | 独立 `val.pth`，含 `avg_self_attn_out` 和 `ann_feats` |
| `IMAGE_ROOT` | 原图根目录，能找到 `train2014/COCO_train2014_*.jpg` |
| `CAPTIONS_JSON` / `INSTANCES_JSON` | A 诊断用的已有 COCO captions/instances 标注 |
| `GALLERY` | 已有 MSCOCO14 gallery，不是新建词库 |
| `TEACHER` | 冻结 Talk2DINO Teacher 权重，默认 `weights/vitb_mlp_infonce.pth` |
| `CLIP_CHECKPOINT` | 已有 `ViT-B-16.pt`，必须是完整本地文件路径 |
| `DINO_REPO` | 已有 DINOv2 源码目录，内含 `hubconf.py` |
| `DINO_CHECKPOINT` | 已有 `dinov2_vitb14_reg4_pretrain.pth` |
| `CONCEPT_ALLOWLIST` | 可选，已有/人工核对的对象与 stuff 概念，每行一个；仅与 gallery 取交集 |
| `SELECTION_CONFIG` | 独立校准得到的冻结阈值文件；首次试跑可留空 |

`train.pth` 需要标准 COCO 字典：`images` 每项含 `id,file_name,disentangled_self_attn:[12,768]`；`annotations` 每项含 `id,image_id,caption,ann_feats:[512]`。不把 `train.pth` 当作原图、Patch 或 GT 缓存。准备阶段从已有 DINOv2 和原图重算两视图 dense 特征，然后只缓存小型区域向量，不保存全量 80k 图片的 dense tensors。

“完整 MSCOCO14”在此表示完整使用你的 `train.pth` 中的训练划分；不会自动合并验证/测试图片，也不假定图像总数恰好为 80k。脚本不截断原始 caption 数量。

```bash
cd /你的路径/Talking2dino/Talk2DINO
CUDA_VISIBLE_DEVICES=0 bash experiments/ecca/run.sh check
bash experiments/ecca/run.sh test
```

检查发现缺依赖、缺模型、缺图像、特征不一致时会停止并报错，不进行安装或下载。不要通过安装新包绕过错误；应检查当前 Talk2DINO 环境是否已正确激活，以及资源路径是否正确。

## 2. 先执行 A：Head 到 Patch 的诊断

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/ecca/run.sh manifest
CUDA_VISIBLE_DEVICES=0 bash experiments/ecca/run.sh diagnose
```

默认从现有 captions/instances 中按 ID 取 200 张。要与之前 200 张热力图严格对应，整理图片 ID 为 `ids200.txt`，改用：

```bash
python -m experiments.ecca.make_manifest coco \
  --captions data/coco2014/annotations/captions_train2014.json \
  --instances data/coco2014/annotations/instances_train2014.json \
  --ids ids200.txt --limit 0 \
  --output outputs/ecca/A/manifest.json
```

然后运行 `run.sh diagnose`。如果 manifest 已存在，可直接编辑/提供自己的 manifest。重复诊断请使用新的输出目录，避免覆盖结果。

A 对每个图像—对象概念输出：

1. 与 DINO 输入完全相同的 448 center crop。
2. 原始 caption 选中 Head 的 attention。多条 caption 含同一对象时，确定性选第一条，记录全部对应 caption 索引。
3. `a photo of {class}` 的原始 dense cosine response。
4. **完整固定类别词表竞争**下的原始 dense argmax，类别集合不会用 GT 的实际出现类别缩小。
5. 可选 GT：仅用于诊断和评价，使用 nearest resize + 相同 crop；crowd/实例重叠像素忽略。

输出位置：

```text
outputs/ecca/A/diagnosis/
├── index.html                      # 浏览器打开，看并排图
├── *.jpg
├── rows.json                       # image-concept 明细
└── report.json                     # 统计与 C 的试跑门槛
```

重点读取：

- `head_gt_recall`：固定 top-20% attention 区域覆盖多少对象像素。
- `dense_iou`、`dense_miss`：原始类别竞争是否真的漏分割对象。
- `attention_dense_top_area_iou`：空间差异代理；**低分不等于匹配错误**。
- `association`：控制 GT 可见面积后的 Head 未覆盖率与 dense 漏分率相关性；以图像为 cluster bootstrap。
- `matched_area_multi_recall_gain` / `matched_area_multi_iou_gain`：相同 top-20% 面积预算下，语义支持的多 Head 平均是否比 caption 单 Head 更合适。

GT 未提供、对象在 crop 外、只有 concept 选头而无原始 caption 对应时，不伪造 caption grounding 真值。VOC manifest 可作为额外的类别诊断，但没有原始 captions 的 VOC 样本不会触发 C 的 caption-based 门槛。

A 是 **center-crop、单 prompt、raw cosine 的机制诊断**；原版 OVSS 使用不同 resize、prompt ensemble、mask 转换和类别背景策略。A 的 `dense_iou` 不能当成官方 mIoU，二者应分别报告。

### C 的预先声明门槛

默认同时要求：至少 50 个有 GT 的 caption 对应图像；面积控制后的相关性 95% CI 下界 > 0；等面积多 Head 覆盖增益 CI 下界 ≥ 0.05；等面积 IoU 增益 CI 下界 > 0。

这些是可修改并记录的**探索性试跑规则**，不是领域公认阈值，也不是“粒度造成错误”的因果证明。需要在看结果前固定，不能事后调到通过。报告为 false 时，训练入口直接拒绝 C。即使通过，仍需 E3/C 严格对照证明方法有效。

## 3. 再执行 B 的前置验证：恢复与独立核验

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/ecca/run.sh recovery
CUDA_VISIBLE_DEVICES=0 bash experiments/ecca/run.sh calibration
```

`recovery` 在 200 张图上，逐图隐藏一个原 caption 明确提及的 gallery concept：从“已覆盖”集合移除该概念及词面别名，然后进行视觉候选检索与筛选。**不会把隐藏概念强制注入候选集**，因此可以测候选召回失败。

它报告 retrieval recovery、Head ranking recovery@K、filtered recovery@K。原始 caption 里出现过的概念只是正例；未提及概念可能真实存在。因此本实验不把非目标词当负例，不输出伪造的真实 Precision 或 False Addition Rate。缓存标为 `recovery_only`，训练入口禁止使用。

`calibration` 使用后面的 100 张图，导出空标签表：

```text
outputs/ecca/calibration/audit_labels.csv
```

打开 `outputs/ecca/calibration/audit.html`，逐项核对原图、所选 Head、概念响应及实际训练区域权重，填写：

```csv
image_id,concept,present,grounded,split
12345,traffic light,1,1,calibration
12345,car,0,0,calibration
23456,bicycle,1,1,test
```

`present=1/0` 表示独立判断真实存在/不存在；`grounded=1/0` 表示该候选所选区域是否确实支持该概念。空值表示未知。**同一图片的所有候选必须属于同一个 split**。校准图片和最终选择精度评估图片分开；GT/人工标签不进入 B 的损失或区域构建。

为避免只核对“好候选”产生偏差，自动阈值搜索要求校准图片的候选完整标注，并至少 100 个有判断的候选。选择阶段初始阈值只是运行起点，不能直接称为已校准。

```bash
mkdir -p outputs/ecca/calibration_report
python -m experiments.ecca.recovery \
  --prepared outputs/ecca/calibration \
  --labels outputs/ecca/calibration/audit_labels.csv \
  --tune --min-precision 0.8 --min-labeled 100 \
  --output outputs/ecca/calibration_report/report.json
```

达到预先指定的 precision/接受量要求才产生 `selection_frozen.json`，否则停止并报告失败。它使用 calibration split 搜索，test split 只报告。0.8 是示例目标，需预先按研究标准决定；样本级观测 precision 不是总体精度保证。

把 `settings.sh` 的 `SELECTION_CONFIG` 设为该文件后再做全量准备。若不准备人工核验，可以用初始阈值做探索性试跑，但不能声称选择精度得到真实验证。局部真实性不能用 Teacher 自己的高相似度充当真值。

若要在已存在的 COCO instances 类别上半自动评估概念真伪，可以先独立建立明确的 concept→category 对应，再填写上述标签表。不要把未映射 gallery 概念直接判为不存在。

## 4. 全量构建 B 的离线概念—区域缓存

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/ecca/run.sh prepare
```

默认 `--limit 0`，完整使用训练划分。该阶段通常比轻量 projector 训练更耗时，因为每图运行原图和水平翻转两次 DINOv2；不在本包中捏造运行时长。

具体流程：

1. 复用仓库 gallery 格式校验与 `filter_novel.py` 的词边界/复数匹配规则。
2. CLIP 本地编码 `a photo of {concept}`，Teacher 投影到 DINO 空间，保存可溯源文本缓存。
3. 每个 Head 在 gallery 中检索 top-8，合并成候选；也支持已有 `concepts_by_image.json`（`image_id -> list[str]`）作为候选池。
4. 候选与 patch 比较；从候选和固定 gallery distractor 子集中，选择文本上易混淆的伪负概念。排除原 caption 已覆盖概念、词面别名、CLIP 高相似别名、视觉分数很高的疑似共现正例。没有合适负例就弃权。
5. 计算相对 patch margin、Head 加权支持度 Q、正 margin attention 质量、翻转还原后的空间稳定性、Head cosine。所有数值都是 Teacher 代理分数，不解释为概率。
6. 只为未被原 captions 词面覆盖、通过证据筛选的概念构建局部区域：`w = attention_best * max(margin,0)`，归一化加权 DINO patch 均值。每图最多 3 个；不够就少选，绝不强行补满。
7. 缓存单 Head 区域及满足近最佳支持度的多 Head 区域。多 Head 只有在 pooled concept cosine 没明显下降时才成为 adaptive 区域。
8. 首先对 3 张图片核对重新提取与原 `train.pth` 的 Head cosine 和原 caption CLIP cosine（≥0.999）。不一致就停止，防止 448/518、模型权重、文字编码来源不一致。

可选候选输入的执行示例（用你自己的本地路径替换）：

```bash
python -m experiments.ecca.prepare \
  --captions outputs/coco2014_b14/train.pth \
  --gallery experiments/cocu_imagelevel_compare/MSCOCO14_train_gallery_exactmatch.json \
  --candidate-file outputs/已有方案/concepts_by_image.json \
  --image-root data/coco2014 \
  --clip-checkpoint /已有路径/ViT-B-16.pt \
  --dino-repo /已有路径/facebookresearch_dinov2_main \
  --dino-checkpoint /已有路径/dinov2_vitb14_reg4_pretrain.pth \
  --teacher weights/vitb_mlp_infonce.pth \
  --output outputs/ecca/prepared_existing_candidates
```

Gallery 本身不是名词/可分割对象标注。本包不安装 POS parser；建议用 `CONCEPT_ALLOWLIST` 限定已核对的对象/stuff 类型，或在审计中单独记录动作/关系/属性。别名排除只覆盖词形和高 CLIP 相似度，不能保证全部语义同义词都被识别。伪负概念仍可能包含图中未描述对象，这个风险必须通过独立审计评估。

缓存：

```text
outputs/ecca/prepared/
├── source.json                     # 源文件/模型哈希、全部图片 ID、参数、代码哈希
├── text_bank.pth                   # 原始 CLIP 和 Teacher 投影特征
├── feature_checks.json
├── concepts_by_image.json           # 可直接交给已有概念可视化工具
├── selection_audit.jsonl            # 便于打包分析的候选与筛选明细
├── records/{image_id}.pth           # 小型区域向量、筛选明细、负概念 ID
└── summary.json                    # 选择量、弃权量、coverage/grounding 代理
```

相同参数和来源重跑会复用完成的逐图记录；来源、阈值或代码改变会拒绝复用旧目录。源文件哈希核验比“文件名相同”更严格。

`coverage_gap_proxy` 是已评分候选中“可靠但原文本词面未覆盖”的比例；`grounding_gap_proxy` 是已评分且词面覆盖候选的筛选失败率。它们只覆盖候选池、仍受 Teacher 和词形匹配误差影响，不能替代真实语义缺失率或真实 grounding 错误率。

## 5. B 的训练实验与公平对照

| 实验 | 全局文本池 | 全局损失 | 额外局部损失 | 主要比较 |
|---|---|---|---|---|
| E0 | 全部原 captions | 原 Talk2DINO | 无 | 原始基线 |
| E1 | 原 captions + Head 排序的未覆盖概念 | 原 Talk2DINO | 无 | 同数量、Head 排序控制 |
| E2 | 原 captions + 证据筛选概念 | 原 Talk2DINO | 无 | E2−E1：筛选效果 |
| E3 | **与 E2 完全一致** | 原 Talk2DINO | 正确概念—区域 | E3−E2：局部目标增益 |
| E4 | **与 E3 完全一致** | 原 Talk2DINO | 同概念、错配其他图片区域 | 语义配对控制 |
| E3raw | 仅原 captions | 原 Talk2DINO | 与 E3 相同局部监督 | 符合“原全局监督 + 局部补充”的直接主线版本 |
| C | **与 E3 完全一致** | 原 Talk2DINO | 证据支持的 adaptive 多 Head 区域 | 只有 A 通过门槛才允许，与 E3 比较 |

E1 在每张图片加入与 E2 **相同数量**的 Head 高分未覆盖概念，候选需具备可计算的负样本证据，但不通过 E2 的证据阈值。这不是重新实现原论文完整 CoCu 检索；若要比较历史 CoCu，需提供已有 CoCu 候选，用相同预算另建 prepared 目录。当前默认候选生成是 Teacher Head→已有 gallery 的最小复用实现。

训练复用 `src.model.ProjectionLayer`、`src.loss.ContrastiveLoss`、原 Adam/AdamW 和 scheduler helper、原 `validate`；模型结构不变，学生随机初始化。Teacher 不加入计算图。默认 lr=1e-4、100 epochs、batch=128、Adam、无 warmup，读取原 YAML；**不重新定义原 InfoNCE**，包括其额外 `/B²` 归一化。

每个方法每 epoch 使用 `ceil(原始caption数 / batch_size)` 次更新。E0 使用与原训练相同的完整 raw DataLoader；扩展文本池使用固定 seed 的均匀随机排列前缀，抽取与原始 captions **相同数量**的样本，所以 E1/E2 的额外文本不会偷偷增加优化步数。原 captions 全部保留在池中，但扩展池不保证每 epoch 每条 raw 都出现；这是固定计算预算下的明确取舍。E2/E3/E4/C 的池、顺序和 batch 大小完全相同。

局部损失采用每区域的“正概念 + 已缓存伪负概念”CE，不把批内重复概念或共现概念直接当成负例：

`L = L_original + lambda * (mean(local_CE) / global_batch_size²)`。

默认 `lambda=0.1`。先对 0.03/0.1/0.3 做独立验证可接受，但不能反复按最终测试集挑参数。记录原损失、归一化局部损失及局部样本数。原损失已除以 B²，因此不能直接加一个未缩放 CE；这里显式对齐了量级。

E4 每个局部样本找另一张图片，排除原 captions 和选中概念已包含该概念的 donor。其本质是经过词面/Teacher 筛查的错配控制，不能保证 donor 图像中绝对不存在该对象；找不到合适 donor 会报错而不会少算局部样本。

先跑单种子：

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/ecca/run.sh train E0 123
CUDA_VISIBLE_DEVICES=0 bash experiments/ecca/run.sh train E1 123
CUDA_VISIBLE_DEVICES=0 bash experiments/ecca/run.sh train E2 123
CUDA_VISIBLE_DEVICES=0 bash experiments/ecca/run.sh train E3 123
CUDA_VISIBLE_DEVICES=0 bash experiments/ecca/run.sh train E4 123
CUDA_VISIBLE_DEVICES=0 bash experiments/ecca/run.sh train E3raw 123
```

E3raw 是直接主线目标，E2/E3 是区分文本补充与局部目标的严格消融。第一轮也可只做 E0/E2/E3/E4，再加 E1/E3raw；不要直接开 C。

断点继续：

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/ecca/run.sh resume E3 123
```

保存 optimizer、模型、epoch、Python/NumPy/CPU/CUDA RNG 状态。相同参数才允许继续；改变训练轮数、λ、数据或模型配置应新开目录。目录中 `last.pth` 是默认主结果，`best.pth` 按相同原始 caption validation loss 选择，另有每 10 epoch 的 `epoch_010.pth` 等。不能用“各方法各自在测试分割上最好的 checkpoint”进行比较。

五种子示例：

```bash
for seed in 123 124 125 126 127; do
  for method in E0 E1 E2 E3 E4 E3raw; do
    CUDA_VISIBLE_DEVICES=0 bash experiments/ecca/run.sh train "$method" "$seed"
    CUDA_VISIBLE_DEVICES=0 bash experiments/ecca/run.sh eval "$method" "$seed"
    CUDA_VISIBLE_DEVICES=0 bash experiments/ecca/run.sh eval_off "$method" "$seed"
  done
done
bash experiments/ecca/run.sh summary
```

## 6. 分割评估：全部复用已有 OVSS

单个权重：

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/ecca/run.sh eval E3 123
CUDA_VISIBLE_DEVICES=0 bash experiments/ecca/run.sh eval_off E3 123
```

五个基准：VOC20、VOC（含背景）、Pascal Context59、ADE20K、COCO-Stuff。`eval` 使用原 PAMR 配置及原模型背景策略；`eval_off` 关闭 PAMR 与 `with_bg_clean`，其他 prompt/预测/背景阈值等仍由原评估器处理。因此 **eval_off 仍是原 OVSS 的消融评估，不等于 A 的原始 cosine argmax**。如果想只关闭 PAMR而保留背景策略，用 `evaluate.py --pamr off --bg-clean config`。

评估在子进程中把原 `torch.hub.load` 和 `clip.load` 绑定到显式本地文件，不改仓库源码。会在 `weights/ecca_{name}.pth` 建立指向实验 checkpoint 的符号链接，原评估器继续按原配置类加载 projector。名称冲突则拒绝覆盖。每次保存原评估日志、mIoU 和配置/权重哈希。

例如评估第 50 轮：

```bash
python -m experiments.ecca.evaluate \
  --checkpoint outputs/ecca/runs/E3/seed_123/epoch_050.pth \
  --name E3_seed123_epoch_050 \
  --output outputs/ecca/runs/E3/seed_123/segmentation_epoch050 \
  --clip-checkpoint /已有路径/ViT-B-16.pt \
  --dino-repo /已有路径/facebookresearch_dinov2_main \
  --dino-checkpoint /已有路径/dinov2_vitb14_reg4_pretrain.pth \
  --pamr on --bg-clean config
```

保留原 evaluator 日志中的每类 IoU，重点看入选概念对应类别及混淆类别。汇总脚本自动统计标准 mIoU；各数据集类别 ID/别名不同，**不自动凭字符串把 gallery 概念映射为 benchmark 类别**。研究中应人工固定映射后分析每类收益。

## 7. C 何时与如何执行

只有 `outputs/ecca/A/diagnosis/report.json` 中 `c_eligible_for_pilot=true` 才能：

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/ecca/run.sh train C 123
CUDA_VISIBLE_DEVICES=0 bash experiments/ecca/run.sh eval C 123
CUDA_VISIBLE_DEVICES=0 bash experiments/ecca/run.sh eval_off C 123
```

C 只改变 **局部概念分支**的区域聚合，保留完整 caption 分支原始的单 Head 机制；这是最小干预版本，不把所有原 caption 都改成多 Head。候选文本、负概念、训练步数、global 输入均与 E3 相同。`settings.json/adaptive_records` 告诉你实际发生了多少区域变化。若为 0，C 实际没有干预，不应把结果当成多粒度方法有效/无效的证据。

## 8. 结果怎么分析、什么时候停止

```text
outputs/ecca/runs/E3/seed_123/
├── settings.json
├── history.jsonl                   # 每轮总损失、global/local、local样本数、val、学习率
├── resume.pth
├── best.pth / last.pth / epoch_*.pth
├── result.json
└── segmentation/{on|off}/{config|off}/{task}/
    ├── eval.log
    └── metrics.json
```

`summary.json/csv` 按任务、后处理、checkpoint 分开，输出 mean±样本标准差、同种子 E0/E2/E3 的配对差、seed bootstrap CI 与精确 sign-flip p。5 seeds 下，两侧精确 sign-flip 的最小 p 是 0.0625；不能因为 5/5 同向就声称 p<0.05。bootstrap CI 也不能当成大样本显著性保证。多个基准的结果应完整报告，不只挑最好一项。

判据：

- A 不支持 Head 未覆盖与 dense 错误的关系：保留 A 的负结果，优先检查概念质量、prompt/背景/后处理；不推进 C。
- E2 优于 E1，而 E3 没优于 E2：主要证据支持更好的概念筛选，尚不支持局部目标增益。
- E3 优于 E2 且优于 E4，并且原始 dense/关闭后处理的表现也改善：支持语义区域对应有价值。
- E3raw 优于 E0：支持保持原始 caption 全局监督、增加局部概念分支的直接路线。
- 局部 loss 降、Teacher proxy 升，但 dense/官方分割无收益：不能称为成功，先排查自我确认和负样本错误。
- C 只在 gate 通过后试跑，并需在相同条件、多种子下超过 E3；面积相关性和可视化本身不能证明粒度是核心因果瓶颈。

本实现是可验证实验框架，尚无新方法训练结果；不能据此预先声称 mIoU 提升、显著性或论文新颖性。
# manifest 缺依赖修复

本版 manifest 无需 pycocotools，不安装新包。更新旧版并保留 settings.sh 的命令见
`FIX_MANIFEST.md`；原错误是 manifest 失败引起 diagnose 缺失输入文件。

