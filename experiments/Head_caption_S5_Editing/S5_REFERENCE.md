# S5：Single-Head Calibrated Phrase–Concept Semantic Completeness

> **用途**：说明在 Talk2DINO / MSCOCO2014 上，如何利用冻结 Teacher、DINOv2 Head Features、CLIP ViT-B/16 和 MSCOCO14 Concept Gallery，对 **固定 Head 与 Caption 之间的语义缺失和潜在冗余**进行双向诊断。
>
> **代码位置**：`experiments/Head_caption_completeness_diagnosis/`，主要实现为 `run.py`、`text_evidence.py`、`core.py`、`perturbations.py`。
>
> **实验性质**：离线指标诊断与编辑扰动验证；**不训练 Student、不自动接纳修改后的 Caption、不执行 OVSS 分割**。所有“支持度”均为模型校准分数，并非标注真值或概率。

## 1. 研究问题与核心思想

原有 BSC-CR 通过整句 CLIP–Concept 相似度衡量 Caption 的语义覆盖，出现两个主要问题：

1. 删除或添加某个 Concept 时，整句重编码可能改变许多无关 Concepts 的支持值（**非目标语义漂移**）。
2. 使用随机图像的 **12 Heads 最大响应**设定视觉背景阈值，却将该阈值用于当前选中的**单个 Head**，两者统计口径不同；原方法还融合整图 CLIP 支持，削弱了当前 Head 的局部解释性。

**S5 = S3 的局部 Phrase–Concept 文本覆盖 + 单 Head 的背景校准视觉支持 + 修复后的双向完整性计算。**

对于一张图像的一个原始 Caption，先由**冻结 Teacher**选定一个 Head，之后始终保持该 Head 不变；再在统一的 Concept Gallery 上构造两组支持向量：

- `V[h,k]`：当前 Head 对 Concept `k` 的视觉支持。
- `T[c,k]`：Caption 是否显式或局部语义地表达 Concept `k`。

根据二者的差异计算：

- `D_add`：Head 支持、Caption 未充分表达的潜在缺失语义。
- `D_remove`：Caption 有表达、Head 支持不足的潜在冗余语义。

最后通过给 Caption **添加/删除指定 Concept**，在固定 Head 与固定可比较 Concept 集合上检查指标如何变化。

## 2. Pipeline 总览

```text
输入：MSCOCO2014 图像与原始 Caption
  │
  ├── DINOv2 图像 12 Heads：[12, 768]（原 train.pth）
  ├── CLIP Caption 特征：[512]（原 train.pth）
  ├── 冻结 Talk2DINO Teacher projector
  └── MSCOCO14 Gallery + CLIP 图像向量缓存
  │
  ▼
① 按原始 Talk2DINO max-score/argmax 选定 Head h*，之后不变
  │
  ├─────────────────────────────────┐
  ▼                                 ▼
② 视觉支路：V[h*,k]             ③ 文本支路：T[c,k]
   CLIP concept prompt             Caption → 局部 n-gram Phrases
   → Teacher projector             → CLIP Phrase–Concept 相似度
   → 单 Head 余弦相似度            → 文本背景校准 + 词面/别名锚定
   → 随机单 Head null 校准         → UNKNOWN
  │                                 │
  └────────────────┬────────────────┘
                   ▼
④ 固定有效 Concept 掩码 + 计算 D_add / D_remove
                   │
                   ▼
⑤ 构造 ADD、DELETE、同义替换、中性前缀对照
                   │
                   ▼
⑥ 原始与修改 Caption 重新计算 T；V 与 h* 不变
                   │
                   ▼
⑦ 指标差值、目标贡献、非目标漂移、方向性判定
                   │
                   ▼
输出：records.jsonl / summary.csv / summary.json / metadata.json
```

## 3. 输入、模型与共享 Concept 空间

**数据与权重（当前代码默认路径）**：

| 文件/组件 | 作用 |
|---|---|
| `outputs/coco2014_b14/train.pth` | `images[*].disentangled_self_attn`（12×768）及 `annotations[*].ann_feats`（512）、原始文本 |
| `weights/vitb_mlp_infonce.pth` | 冻结 Talk2DINO Teacher projector |
| `configs/vitb_mlp_infonce.yaml` | Teacher projector 结构配置 |
| `experiments/local_cocu/MSCOCO14_train_gallery.json` | Concept Gallery |
| `outputs/Head_caption_completeness/clip_images.pth` | 冻结 CLIP ViT-B/16 整图特征缓存；S5 **不将其混入最终 V**，但构造扰动时仍使用它 |

模型编码直接复用 `experiments.Head_Match.common.teacher.TeacherBank`，不单独复制 CLIP 或 Projector 训练实现。

对 Gallery 中的每个 Concept `k`，统一使用 CLIP 提示词：

```text
a photo of a {concept}
```

先提取**未归一化**的原始 CLIP 文本特征 `e_k`（512 维）；构造两种视图：

\[
q_k = \operatorname{Normalize}(e_k) \in \mathbb{R}^{512},
\qquad
 g_k = \operatorname{Normalize}\big(P_{teacher}(e_k)\big)\in\mathbb{R}^{768}.
\]

`q_k` 用于文本 Phrase–Concept 匹配；`g_k` 用于 DINOv2 Head–Concept 匹配。**注意 Projector 输入为原始 CLIP 特征，不应在送入非线性 Projector 前先进行单位化。**

## 4. Step 1：固定原始 Caption 对应的 Head

每张图像已经有 12 个 DINOv2 Head Features：

\[
H_i=\{h_{i,1},\ldots,h_{i,12}\},\quad h_{i,j}\in\mathbb R^{768}.
\]

对原 Caption `c` 直接读取缓存 `ann_feats`，经冻结 Teacher projector 得到：

\[
z_c=\operatorname{Normalize}\big(P_{teacher}(\mathrm{ann\_feats}_c)\big).
\]

按原始 Talk2DINO 的最大相似度规则选择：

\[
h^*=\arg\max_{j\in\{1,\dots,12\}}
\operatorname{cos}(h_{i,j},z_c).
\]

**核心控制变量**：对于该 Caption 的任何 ADD / DELETE / 同义替换 / 中性前缀操作，均使用**原始选中的 `h*`**；不重新执行 Head Assignment，以避免把 Head 更换导致的分数变化当作 Caption 编辑效果。

对应代码：`run.py → FrozenData.evidence()`。

## 5. Step 2：视觉支持 V——单 Head 背景校准（S5 的关键区别）

### 5.1 原始相似度

对每个 Concept `k` 计算固定 Head 与 Teacher 投影 Concept 的余弦相似度：

\[
s^{head}_{h^*,k}=\operatorname{cos}(h^*,g_k).
\]

相似度高只是模型支持高，并不直接等于 Concept 在图像中真实存在。

### 5.2 随机单 Head 的概念条件 null 分布

默认随机选取 **256 张**参与本次运行的图像；每张图像从其 12 Heads 中**随机选一个 Head**，得到背景集合 `B={\tilde h_i}`。

对每个 Gallery Concept 独立计算背景相似度分布的 **90% 分位数**：

\[
b^{single}_k=Q_{0.90}\big(\{\tilde h_i^{\top}g_k\}_{i\in B}\big).
\]

这样校准用的样本统计量和 S5 真正要判断的 **单个 Head** 保持一致。

对照 S3：其背景分布先对每张图像的 12 Heads 取最大值再计算分位数；S3 的最终视觉分数还混合 CLIP 整图分数。

### 5.3 分位数与线性 Ramp 映射

S5 的具体实现为：

\[
\boxed{
V_{h^*,k}=\operatorname{clip}\left(
\frac{s^{head}_{h^*,k}-b^{single}_k-\delta_h}{m_h},0,1
\right)
}
\]

当前默认：`δ_h = 0.01` (`--head-offset`)，`m_h = 0.10` (`--head-margin`)。

- `V=0`：与背景相比没有达到当前阈值下的可检测支持。
- `0<V<1`：相对背景存在中等证据。
- `V=1`：相似度超过背景阈值和 Ramp 上界，**不代表 100% 确信存在**。

S5 **不混合整图 CLIP 分数到最终 V**。不过程序另外输出：

- `target_visual_image`：整图 CLIP 支持；
- `target_visual_other_heads`：除选中 Head 外其他 Heads 的最大支持；
- `target_visual_head`：当前 Head 的单 Head 校准支持。

这些值可辅助判断“只是当前 Head 不支持”与“全图都缺乏证据”的差别，不自动充当真实存在/不存在标签。

对应代码：`run.py → FrozenData.calibrate()`、`FrozenData.evidence()`。

## 6. Step 3：文本支持 T——局部 Phrase–Concept CLIP（继承 S3）

### 6.1 将 Caption 转为局部连续词组

例如：

```text
Caption: A man riding a red bicycle on the road.
```

按 `1~4` 词的连续 n-gram 提取 Phrases，例如：`man`、`bicycle`、`red bicycle`、`riding a red bicycle`、`road` 等。

实际代码限制：

- 默认 `--max-ngram 4`、`--max-phrases 72`；按 n-gram 长度从小到大收集至上限。
- 排除纯停用词片段、首尾为停用词的片段，以及包含简易否定词的片段。
- **不使用额外句法分析器或 LLM**；短语不必都是真实句法成分。

### 6.2 Phrase 对所有 Gallery Concepts 的语义匹配

每个短语也按 `a photo of a {phrase}` 交给冻结 CLIP 文本编码器：

\[
p_r=\operatorname{Normalize}\big(\mathrm{CLIP}_{text}(\texttt{a photo of a \{phrase\}})\big).
\]

然后对每个 Concept `k` 取短语最大相似度：

\[
S(c,k)=\max_{r\in\mathrm{Phrases}(c)}p_r^\top q_k.
\]

这是 S5 对**独立 Concept 语义表达程度**的核心估计，避免直接拿全句特征同时解释所有 Concepts。

### 6.3 文本背景校准、词面锚定与 UNKNOWN

代码先从 Gallery 随机抽取最多 **256** 个 Concepts，计算与各 Concept 的 CLIP 相似度，排除自身完全相同的 Concept–Concept 对，取每个 Concept 的 **95% 分位数**作为文本背景阈值 `b_k^text`。

连续文本支持：

\[
\tilde T_{c,k}=\operatorname{clip}\left(
\frac{S(c,k)-b^{text}_k-\delta_t}{m_t},0,1
\right),
\quad\delta_t=0.015,\quad m_t=0.16.
\]

接着进行三种处理：

1. **显式词面/方向性别名匹配**：例如 `man → person`、`bike ↔ bicycle`；匹配后直接 `T=1`。方向性别名不代表所有上下位词都能双向替换。
2. **简单局部否定检测**：如果词面命中且前两个词内出现 `no/without/not/...`，将该 Concept 设置为 `T=0`；这仅是规则近似。
3. **UNKNOWN**：若未被明确词面/别名决定、且 `0.15 < \tilde T < 0.40`，标记 `UNKNOWN`，不得按缺失（0）直接参与完整性指标。

最终：

\[
T_{c,k}\in[0,1]\quad\text{或}\quad \mathrm{UNKNOWN}.
\]

注意：`T=0` 的意思是**当前文本判定器没有发现足够表达证据**，不是 Concept 在真实图像中不存在。

对应代码：`text_evidence.py → phrases()`、`clip_null()`、`TextScorer.score(method='S3')`。S5 复用这一 `S3` 文本实现，不是另一个文本模型。

## 7. Step 4：固定 Concept 集合，计算双向完整性

添加、删除 Caption 的前后比较，必须使用完全相同的 Concept 维度。

先为第 `k` 个 Concept 判断是否应纳入比较：

\[
B_k = [V_{h^*,k}\ge 0.25] \lor [T_{before,k}\ge0.75]
\lor[T_{after,k}\ge0.75]\lor[k=k_{target}].
\]

再要求该 Concept 在修改前后**都不为 UNKNOWN**：

\[
\mathcal K = \{k:B_k\land \neg U_{before,k}\land \neg U_{after,k}\}.
\]

因此，**目标 Concept 如任意一侧是 UNKNOWN，也不会强制转成有效值**；其方向判断可以是“不可判定”。

在固定集合 `K` 上计算：

**视觉有、文本缺——缺失度**

\[
\boxed{D_{add}(h^*,c)=
\frac{\sum_{k\in\mathcal K}V_{h^*,k}(1-T_{c,k})}
{\sum_{k\in\mathcal K}V_{h^*,k}}}
\]

**文本有、当前 Head 视觉证据弱——潜在冗余度**

\[
\boxed{D_{remove}(h^*,c)=
\frac{\sum_{k\in\mathcal K}T_{c,k}(1-V_{h^*,k})}
{\sum_{k\in\mathcal K}T_{c,k}}}
\]

当分母没有有效支持时，代码返回 `None`（不可计算），并不将无定义的损失视为零。

对应代码：`core.py → comparison_mask()`、`coverage()`。

## 8. Step 5：对指定 Concept 进行文本干预

程序在每条原始 Caption 上尝试构建下面几种**合成扰动**；不保证每条 Caption 都能构造全部类型。

| 扰动 | 候选 Concept 的来源 | 修改文本的方式 | 主要验证目标 |
|---|---|---|---|
| `add_high_evidence` | 当前 Head `V≥0.60`、整图支持 `≥0.30`，且 Caption 未被词面判定表达 | 原 Caption 末尾追加 `; also visible: concept.` | 添加视觉支持语义能否降低 `D_add` |
| `delete_supported` | Caption 已明确表达，且当前 Head `V≥0.60` 的 Concept | 用正则删除**首次**匹配的词面表达及部分冠词 | 删除视觉支持语义能否增加 `D_add` |
| `add_low_evidence` | 当前 Head `V≤0.12`、整图 `≤0.18`、其他 Heads `≤0.30` | 同样在末尾追加 Concept | 低证据 Concept 是否增加潜在冗余 `D_remove` |
| `synonym` | 小规模手工同义对 | 直接替换词面 | 同义语义的支持值是否稳定 |
| `neutral_prefix` | 不指定目标 Concept | Caption 前加入 `Here, ...` | 无明显语义变化时的稳定性 |

> **重要实现边界**：`delete_supported` 当前是字符串级删词，不是完整的自然语言重写：如果目标 Concept 在原句出现多次，删除第一次后其他位置仍可能保留；也不能保证句子通顺。因此“已删除一次字符串”不等于“完全移除了该 Concept 的语义”。高/低视觉证据 Concept 也未经人工真值核验。

对应代码：`perturbations.py → make_cases()`、`remove_mention()`；新增文本使用原项目的 `make_add()`。

## 9. Step 6：在固定 Head 下比较编辑前后分数

对原始 Caption 和修改后 Caption，分别计算：

\[
T_{before},\quad T_{after};\qquad
V_{before}=V_{after}=V_{h^*}.
\]

再用**同一个** `K` 计算两个方向的分数差：

\[
\Delta D_{add}=D_{add}^{after}-D_{add}^{before},\quad
\Delta D_{remove}=D_{remove}^{after}-D_{remove}^{before}.
\]

**方向预期**：

- 添加高视觉证据且此前未表达的 Concept：`ΔD_add < 0`。
- 删除高视觉证据且此前已表达的 Concept：`ΔD_add > 0`。
- 添加低视觉证据的 Concept：在**数学单调条件成立**时，`ΔD_remove > 0`。
- 同义词替换：目标 `T` 变化应尽量小。
- 中性前缀：`ΔD_add` 与非目标漂移都应尽量小。

需要特别强调，`D_remove` 是**归一化加权平均**，因此“添加低证据词”不意味着聚合 `D_remove` 一定上升；在其他概念支持不变时，代码使用的必要方向条件是：

\[
1-V_{h^*,k_{target}}>D_{remove}^{before}.
\]

不满足条件的样本仍然保留数值记录，但**不强行判作方向错误**。

### 9.1 定位编辑究竟影响了谁

固定 `V`、`K` 后，可以将 `ΔD_add` 精确拆为目标与非目标两部分：

\[
\Delta D_{add}=
\underbrace{-\frac{V_j\,\Delta T_j}{\sum_{k\in\mathcal K}V_k}}_{\text{目标 Concept 贡献}}
\; +\;
\underbrace{-\frac{\sum_{k\in\mathcal K,k\ne j}V_k\,\Delta T_k}
{\sum_{k\in\mathcal K}V_k}}_{\text{非目标 Concepts 贡献}}.
\]

代码还保存 `off_target_mean_abs_delta` 和 `leakage_ratio`，用于判断是**特定目标语义起主要作用**，还是其他 Concepts 的漂移主导了聚合指标变化。

### 9.2 额外记录 Teacher 固定 Head 分数变化

虽然 S5 的完整性公式不直接使用 Teacher 整句匹配分数作为损失，它仍记录：

\[
\Delta S_{teacher}
=\operatorname{cos}\big(h^*,P(c_{after})\big)
-\operatorname{cos}\big(h^*,P(c_{before})\big).
\]

实现字段为 `head_teacher_score_delta`。**记录它不等于要求它必须上升，也不代表编辑已经被接受**；S5 当前只做离线诊断。

对应代码：`run.py → evaluate_case()`、`core.py → drift_decomposition()`。

## 10. 输出、评价与 800 图实验的解释

每个 `(image, annotation, perturbation, method)` 输出一条记录。关键字段：

```text
image_id, annotation_id, head_fixed
kind, original_caption, modified_caption, target_concept
target_visual_used, target_visual_image, target_visual_other_heads
target_text_delta, target_known, valid_count
d_add_before, d_add_after, delta_add
d_remove_before, d_remove_after, delta_remove
target_d_add_contribution, off_target_d_add_contribution
leakage_ratio, off_target_mean_abs_delta
head_teacher_score_delta, direction_success
```

结果写入：

```text
outputs/Head_caption_completeness_diagnosis/run800/
├── records.jsonl
├── summary.csv
├── summary.json
├── metadata.json
└── direction_failures.jsonl
```

800 张图的上一轮结果中，S5 对**可判定样本**的方向正确率为：高视觉证据新增 `100%`、删除 `97.29%`、低视觉证据新增 `97.23%`。这些比例必须连同不同组的**有效样本数/可判定覆盖率**报告；测试采用的是构造扰动，不是 Concept 真值标注，因此**不能等价解释为真实语义判断准确率**。

**尚未证明的事项**：

1. 高视觉支持 Concept 是否在图中真实存在，以及 Caption 是否真的遗漏它；
2. 低 Head 支持是 Caption 冗余，还是物体由**其他 Heads**表达；
3. **单 Head 校准仍非完全分布匹配**：校准时随机选一个 Head，而实际 Head 是 Caption 条件下经 Teacher `argmax` 选出的，后者有选择偏差；
4. S5 相比 S3 **同时**改动背景校准分布与是否混合 CLIP 整图证据，结果差异不能只归功于某一个因素；
5. 降低 `D_add/D_remove` 是否能够提高固定 Head–Caption Teacher 相似度、Student 表征质量或分割 mIoU；
6. 对中间置信度 Concepts、复杂否定/关系短语的泛化表现。

## 11. 最短复现实验指令

在 `Talk2DINO/` 根目录执行：

```bash
# 单独运行 S5，前提是 train.pth、Teacher、Gallery、CLIP 图像缓存已存在
CUDA_VISIBLE_DEVICES=0 python -m \
  experiments.Head_caption_completeness_diagnosis.run \
  --device cuda \
  --limit 800 \
  --captions-per-image 1 \
  --seed 123 \
  --methods S5 \
  --output outputs/Head_caption_completeness_diagnosis/S5_only

# 建议同时运行 S3 对照（其他参数保持相同）
CUDA_VISIBLE_DEVICES=0 python -m \
  experiments.Head_caption_completeness_diagnosis.run \
  --device cuda \
  --limit 800 \
  --captions-per-image 1 \
  --seed 123 \
  --methods S3,S5 \
  --output outputs/Head_caption_completeness_diagnosis/S3_vs_S5
```

## 12. 一句话概括

**S5 不以整句 CLIP 相似度直接代表完整性，而是先把固定 DINOv2 Head 和 Caption 分别映射成 Gallery 上的视觉/文本 Concept 支持向量：视觉端采用单 Head 背景校准，文本端采用局部 Phrase–CLIP，再以同一有效 Concept 集合计算双向缺失与冗余，并通过目标 Concept 的 ADD/DELETE 扰动验证指标是否具有局部敏感性。**

---

**代码依据**：`experiments/Head_caption_completeness_diagnosis/run.py`、`text_evidence.py`、`core.py`、`perturbations.py`（已与本次提交的实验代码 ZIP 核对）。
