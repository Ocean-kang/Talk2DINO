# BSC-CR：基于双向语义完整性的 Talk2DINO Caption 修正方案

## 1. 目标与资源

**目标**：判断训练样本中 Head Feature 与 Caption 的语义覆盖是否匹配；从 Concept Gallery 中选择需要添加或精简的概念；修正训练文本后，保持原版 Talk2DINO 训练与分割评估流程。

**资源限定**：CLIP ViT-B/16、已训练 Teacher Talk2DINO（`weights/vitb_mlp_infonce.pth`）、MSCOCO2014 图像及每图 5 条原始 Captions、MSCOCO14 Concept Gallery、现有 DINOv2 Head Features。**不引入其他模型。**

## 2. 实现流程

### Step 1：双向语义完整性诊断

1. 对每条原始 Caption，使用冻结 Teacher 按原始 `argmax` 策略选出匹配的 Head。
2. **视觉端**：计算当前 Head 与 Gallery 全部 Concepts 的 Teacher 相似度，结合 MSCOCO14 内的错配样本背景分布和 CLIP 整图相似度，得到校准后的 Head–Concept 支持值 $V_{h,k}$。
3. **文本端**：利用词面/同义短语匹配和 CLIP Text–Concept 相似度，得到 Caption 对各 Concept 的覆盖值 $T_{c,k}$；无法可靠判断的概念标为 `UNKNOWN`。
4. 在有效 Concept 上分别计算：

   $$D_{\mathrm{add}}=\frac{\sum_k V_{h,k}(1-T_{c,k})}{\sum_k V_{h,k}+\epsilon}$$

   $$D_{\mathrm{remove}}=\frac{\sum_k T_{c,k}(1-V_{h,k})}{\sum_k T_{c,k}+\epsilon}$$

   - $D_{\mathrm{add}}$ 高：Head 支持、Caption 未表达的语义较多。
   - $D_{\mathrm{remove}}$ 高：Caption 表达、当前 Head 未充分支持的语义较多。
5. 使用**同图 5 条 Captions 的概念并集**和**12 个 Heads 的概念并集**作为对照，区分“单句缺失/整体缺失”与“单 Head 不支持/所有 Heads 均不支持”。阈值通过验证数据校准。

### Step 2：Concept 候选发现与筛选

三路生成候选，最终均限定在 MSCOCO14 Gallery：

1. **Head-Guided**：Teacher 从 Gallery 检索与当前 Head 最相关的 Concepts。
2. **CoCu-Style**：CLIP 检索相似 COCO 图像，从邻居 Captions 中提取 Gallery Concepts。
3. **Same-Image**：从同图其他 4 条 Captions 中提取当前 Caption 尚未表达的 Gallery Concepts。

合并去重后，根据 Head 支持、CLIP 图像证据、Caption 新颖性排序；每条 Caption 优先选 1–2 个可靠候选。对于文本侧超出的 Concepts，必须检查其他 Heads、其他 Captions 与整图证据，**不能将当前 Head 低相似度直接解释为概念错误**。

### Step 3：ADD / REMOVE / KEEP 修正 Caption

| 诊断类型 | 操作 |
|---|---|
| 双向覆盖较好 | **KEEP**：保持原 Caption |
| 视觉语义多、文本少 | **ADD**：加入高置信度且未表达的 Concept，保守措辞，不虚构关系 |
| 文本语义多、当前 Head 少 | **REASSIGN 优先**：先检查其他 Head；仅在可安全编辑且语义由其他 Captions 保留时 **REMOVE** 局部冗余短语 |
| 双向都不匹配或判断不可靠 | **RECHECK / KEEP**：不强制增删 |

**反事实验证**：对修改后的 Caption 重新计算双向覆盖率，并在固定原 Head 下计算 Teacher 匹配分数变化；与错误 Concept 替换对照比较。只有在覆盖损失改善、不损失重要语义、未明显破坏五条 Captions 的整体覆盖时才接受修改；其余保留原文。

### Step 4：原版 Talk2DINO 训练与 OVSS 评估

1. 每张图像**仍保留 5 条 Captions** 和原始 `annotation_id`；只更新被接受修改的 Caption。
2. 使用冻结 CLIP 重新编码修改的文本为 `ann_feats`；图像 Head Features 不变。
3. **Student 从头初始化**，使用原始 `train.py`、`max_score/argmax` Head 选择及 **InfoNCE** 训练 CLIP-to-DINO Projector；Teacher 仅参与离线数据构建。
4. 使用原始 Talk2DINO 流程评估 VOC20、VOC、COCO-Stuff、ADE20K、Context59 的分割 mIoU。

## 3. 实验验证

**先验证诊断**（从 MSCOCO14 抽取约 500–1000 张）：人为删除 Caption 中已有 Concept（检查 $D_{\mathrm{add}}$ 是否升高）；加入明显错误 Concept（检查 $D_{\mathrm{remove}}$ 是否升高）；同义表达改写（检查分数是否稳定）。同时固定原 Head 防止 Head 切换干扰。

**再进行训练消融**：

| 实验 | 设置 |
|---|---|
| E0 | 原始 Raw Talk2DINO Baseline |
| E1 | Completeness-Guided **ADD Only** |
| E2 | Completeness-Guided **REMOVE Only** |
| E3 | **ADD + REMOVE** 联合修正 |
| E4 | ADD + REMOVE，但**不使用完整性门控**（控制修改比例和数量） |

与之前最优 CoCu / Head-Match 对比，记录修改率、添加/删除 Concepts、Head Assignment 变化、双向覆盖变化及五项 mIoU；最终用真实独立随机种子重复实验，报告均值与标准差。

> **方法边界**：上述完整性是 **Gallery 条件下、由 Teacher 与 CLIP 估计的语义覆盖代理**，不是 Head 真实语义的完整标注。研究关键是证明该诊断能指导更有效的训练文本修正，而不是单纯提高 Teacher 相似度。
