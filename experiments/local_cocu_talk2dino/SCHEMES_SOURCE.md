## A. Teacher-Guided Direct Concept Retrieval（Teacher 直接概念检索）

**核心思想：** 不再依赖 Image–Image Retrieval，而是使用预训练 Talk2DINO 的 Head-to-Text 对齐能力，从全局 Concept Gallery 中直接发现概念。

**实施步骤：**

1. 从 COCO Captions 或其他词表构建全局 Concept Gallery；统一词形并尽可能包含名词短语。
2. 使用 CLIP Text Encoder 对 Gallery 编码，再经过冻结的 Talk2DINO Teacher Projector，得到所有 Concept 在 DINO 空间中的向量。
3. 为每张目标图像提取 $H=12$ 个 Head Features。
4. 每个 Head 在全部 Concept Embeddings 中直接执行 Top-K 检索，可借助 FAISS 提升搜索效率。
5. 汇总各 Head 的候选，过滤原 Captions 已提及概念，进行相似度阈值、语义去重及置信度筛选。
6. 保留每个 Concept 对应的 `score`、`best_head` 等信息，输出新 Concepts 供下游训练。

**相对当前 Local-CoCu 的变化：** `CLS → 图像邻居 → 邻居 Captions` 被 `Heads → 全局 Concept Gallery` 取代。Teacher 从“候选筛选器”变为“概念发现器”。

**优势：** 流程短、易做对照；不会因为近邻 Captions 的遗漏而直接失去候选概念。

**风险：** Gallery 覆盖范围决定可发现的概念上限；全库检索容易选出高相似但实际不存在的概念。

**建议消融：** `Local-CoCu` vs `Direct Retrieval`；Top-K、阈值、Gallery 大小；全局 Top-K vs 每 Head 独立 Top-K。

---

## B. Teacher-Guided Local Image Retrieval（Head 级图像检索）

**核心思想：** 用 DINOv2 Head Features 做局部 Image–Image Retrieval，替换当前基于 CLS 的全局检索。

**实施步骤：**

1. 为 COCO 图像库提前提取 Head Features，形成局部视觉向量索引。
2. 对目标图像每个 Head，检索其他图像中的相似 Head，不强制匹配相同 Head 编号。
3. 以局部匹配结果形成邻居图片集合；可记录匹配 Head 和相似度，避免只保存图片 ID。
4. 从检索到的图片 Captions 中提取 Candidate Concepts。
5. 利用预训练 Talk2DINO Teacher 检验这些 Concept 与目标图像 Head 的对齐程度。
6. 对候选去重、排除原 Captions 已有概念，选出补充 Concepts。

一种最简单的跨图 Head 相似度：

$$
S(I_i,I_j)=\max_{h,k}\cos(v_{i,h},v_{j,k}).
$$

也可采用若干高分 Head 匹配的均值，减少单个高分异常值造成的错误检索。

**相对当前 Local-CoCu 的变化：** 仅替换图像检索表征/策略：`CLS Retrieval → Head Retrieval`，其余 Candidate Extraction 和 Teacher Matching 可以最大程度复用。

**优势：** 更容易检索到共享局部物体但整体场景不同的图像，适合小目标及多物体图像。

**风险：** Head 特征并非总对应完整物体；Head 编号之间不具有跨图的固定语义身份；Head 数量增加检索成本。

**建议消融：** CLS Top-50 vs Head Top-50；每 Head 检索 vs 跨 Head 最大匹配；仅局部检索 vs CLS+Head 混合检索。

---

## C. Caption-Uncovered Head Discovery（未被 Caption 覆盖的 Head 发现）

**核心思想：** 先检测图像的哪些局部视觉语义未被原始 Captions 充分解释，再只对这些候选区域进行 Concept Discovery。重点是“补充缺失信息”，而不是“添加更多文本”。

**实施步骤：**

1. 提取目标图像的 12 个 Head Features。
2. 用 CLIP+Talk2DINO Teacher 对该图像原有 5 条 Captions 编码。
3. 计算每个 Head 与所有原 Captions 的最大相似度，据此建立 Caption Coverage Score。
4. 找出对现有 Captions 解释程度较低、但具有有效空间注意力响应的 Heads。
5. 仅针对这些 Head 执行方案 A（概念库直接检索）或方案 B（局部相似图像检索）。
6. 对检索到的新 Concept 再执行视觉打分与原 Caption 语义冗余过滤，得到真正具有潜在增量信息的 Concepts。

覆盖度示意：

$$
q_h=\max_{j=1,\ldots,5}\cos\bigl(v_{I,h},P_T(E(t_j))\bigr),\qquad
u_h=1-q_h.
$$

这里 $u_h$ 只是**未覆盖候选分数**，不是真实语义遗漏概率。必须校验低分 Head 是否关注有效物体/区域，否则背景噪声也会被误判为遗漏语义。

**相对当前 Local-CoCu 的变化：** 从“对所有区域寻找相关 Concept”，改为“识别疑似语义缺口 → 定向寻找 Concept”。

**优势：** 与“COCO Captions 语义描述不完备影响 Talk2DINO 性能”的核心假设最一致；有明确科学问题。

**风险：** Attention Head 不是可靠的物体实例分解；低覆盖分数可能来自 Teacher 能力不足或区域本身没有稳定语义。

**建议消融：** 全部 Heads vs 低覆盖 Heads；Coverage 阈值；是否引入注意力区域有效性检查；A/C 和 B/C 组合。

---

## D. Head-Guided Region Concept Discovery（Head 引导局部区域概念发现）

**核心思想：** 利用 Attention Map 定位 Head 关注的局部区域，再对局部图像区域进行 Concept 验证，以降低整图复杂背景的干扰。

**实施步骤：**

1. 提取 DINOv2 每个 Head 的 Attention Map。
2. 对高响应 Patch 区域进行阈值化、连通区域合并或边界框估计，生成多个候选 Crop。
3. 对候选 Crop 使用相同 DINOv2 模型重新提取视觉特征，或对局部 Patch Features 聚合形成区域特征。
4. 使用 Talk2DINO Teacher 将候选 Concept 与区域特征进行比较。
5. 结合整图匹配与局部匹配的分数，过滤局部证据不充分的概念。
6. 保留 `concept`、`crop/region`、`head`、`score`，供可视化和下游补全。

**相对当前 Local-CoCu 的变化：** 不再局限于 Head 全局聚合向量，而是显式引入空间区域对 Concept 的验证。

**优势：** 对小物体、背景复杂和多实例图像有潜在帮助；便于将概念得分与可解释的图像区域关联。

**风险：** Attention Map 并非实例分割 Mask，Crop 可能不准确；额外前向计算更昂贵。

**建议消融：** 不裁剪 vs Head Crop；不同 Attention Threshold；整图+局部双重一致性 vs 纯局部打分。

---

## E. Cross-Image Concept Consistency Bootstrap（跨图概念一致性 Bootstrap）

**核心思想：** 不只验证某个 Concept 在目标图像中是否高分，还检查它是否在局部相似的其他图像中获得一致的视觉证据。

**实施步骤：**

1. 使用方案 B 建立目标 Head 对应的局部相似图像邻居。
2. 对候选 Concept 计算目标 Head 的 Teacher 匹配得分。
3. 在多个相似邻居的对应局部 Head 上计算同一个 Concept 的得分。
4. 统计跨图的平均支持度、支持邻居数量和分数方差。
5. 对“目标图像高分且跨图支持充分”的 Concept 提高置信度。
6. 对低频类别保留单图高置信度通道，避免过强的一致性要求过滤罕见物体。

**相对当前 Local-CoCu 的变化：** 从单图 `max-head` 排序，变为“目标视觉证据 + 多个局部近邻共同验证”。

**优势：** 能一定程度降低随机高分和偶然语义匹配带来的噪声。

**风险：** 近邻图像可能只共享外观或背景，并非共享物体类别；少见 Concept 不一定拥有足够支持样本。

**建议消融：** 不使用一致性 vs 使用；一致性邻居数量；局部邻居 vs CLS 邻居；均值支持度 vs 支持比例。

---

## F. Counterfactual Concept Selection（反事实/难负例概念筛选）

**核心思想：** Concept 不仅需要与目标图像相似，还应能够把目标图像与相似场景的干扰图像区分开，抑制过于笼统的共现场景词。

**实施步骤：**

1. 利用 A、B 或现有 Local-CoCu 得到 Candidate Concepts。
2. 对每个候选建立场景相似但局部语义不同的难负例图像集合。
3. 分别计算该 Concept 在目标图像与负例图像中的 Teacher 匹配分数。
4. 按目标得分减去负例平均得分重新排序。
5. 筛选兼具视觉相关性和区分性的额外概念。

候选分数：

$$
S_{\mathrm{cf}}(I,c)=s_T(I,c)-\lambda\mathbb{E}_{I^-\in\mathcal{N}^{-}(I)}s_T(I^-,c).
$$

**相对当前 Local-CoCu 的变化：** 加入“这个概念是否只是同类场景中普遍高分”的对照证据，不再只看目标得分。

**优势：** 有潜力降低 `road`、`person` 等泛场景概念的过度入选，提高补充概念的辨别性。

**风险：** 难负例可能实际上包含目标概念；负例挖掘质量会直接影响排序可靠性。

**建议消融：** 无反事实重排 vs 有反事实重排；随机负例 vs 同场景难负例；不同 $\lambda$。

---

## G. Multi-View Teacher Agreement（多视图 Teacher 一致性筛选）

**核心思想：** 真实存在的视觉概念通常应在保留该区域的合理增强视图中稳定得到 Teacher 支持，而噪声匹配往往不稳定。

**实施步骤：**

1. 为每张图片生成原图、不同缩放和轻量外观扰动视图；必要时生成局部 Crop。
2. 分别提取 DINOv2 Head Features 和 Attention Maps。
3. 使用同一个冻结 Talk2DINO Teacher 对 Candidate Concepts 打分。
4. 根据区域空间对应关系，确定哪些视图实际保留了候选物体区域。
5. 仅在有效可见视图之间计算 Concept 得分的稳定性和一致性。
6. 将稳定性作为 Concept 质量分数的一部分，而不是要求所有视图都命中。

**相对当前 Local-CoCu 的变化：** 从单次 Teacher 判别变为跨合理图像增强的一致性验证。

**优势：** 可以作为方案 A/B/C/D 的通用概念质量控制模块。

**风险：** 随机裁剪可能删除真实物体；若不处理空间可见性，会错误惩罚小目标。

**建议消融：** 单视图 vs 多视图；缩放一致性 vs 外观扰动一致性；是否加入区域可见性判断。

---

## H. Iterative Concept Bootstrap（迭代式概念自举）

**核心思想：** 训练好的 Teacher 提取新概念，Student 学习补充后的文本；再把训练完成的 Student 作为下一轮 Teacher，反复改进概念发现。

**实施步骤：**

1. 使用最初的已训练 Talk2DINO 作为冻结 Teacher $T_0$。
2. 用任一 Concept Discovery 方案生成高置信度 Novel Concepts，构造增强训练集 $D_1$。
3. 随机初始化 Student $S_1$；采用原始 Talk2DINO InfoNCE 流程训练，不复制 $T_0$ 的参数。
4. 训练完成后，将 $S_1$ 作为新的 Teacher $T_1$。
5. 用 $T_1$ 重新挖掘或重新筛选 Concepts，构造下一轮数据集 $D_2$。
6. 持续少量轮次迭代；依据独立验证集和 Concept 质量审计决定是否停止。

流程：

```text
Pretrained T0
   ↓
Concept Discovery / Filtering
   ↓
Augmented Captions D1
   ↓
Randomly Initialized Student S1 ──(Original Talk2DINO Training)──→ T1
   ↓                                                           │
Next-round Data D2  ←──────────── New Concept Discovery ────────┘
   ↓
Randomly Initialized Student S2 → ...
```

**相对当前 Local-CoCu 的变化：** Teacher 不再固定只执行一次 Concept Selection，而是随着迭代更新发现能力。区别于仓库已有的 `teacher_guided_bootstrap`：后者主要在训练中修改 Head 选择，未必重新挖掘数据集。

**优势：** 允许更好的 Student 反过来提供更高质量的伪标注/补充概念。

**风险：** 确认偏差和错误自增强；每轮分割性能或 Concept 质量并不保证单调提升。

**建议消融：** 1 轮 vs 2/3 轮；每轮全量重挖 vs 仅增量重挖；固定原 Teacher vs 更新 Teacher；高置信度门槛。

---
