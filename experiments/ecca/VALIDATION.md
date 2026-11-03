# 验证范围

**2026-11-01 修订**：移除 manifest 的 pycocotools 依赖。14 项测试通过，
1 项 PyTorch 集成测试跳过；另有 2000 个多边形和 2000 个压缩 RLE 对照官方 C 解码一致。
本次修复和重跑命令见 `FIX_MANIFEST.md`。下述 6 项测试记录是最初交付时的验证范围。

本包针对 Ocean-kang/Talk2DINO commit `0c155aab1ec2ca59914fef49f597c0345b26e1af` 审计：

- src/model.py：ProjectionLayer、project_clip_txt、单 Head 选择。
- src/loss.py：原始 ContrastiveLoss 与额外 /B² 归一化。
- src/dataset.py：train/val 的 COCO PTH schema。
- src/train_util.py：Adam/AdamW、随机种子、scheduler 与 validate。
- src/hooks.py、dino_extraction_v2.py：QKV、5 个 global tokens、scale=.125、head pooling。
- experiments/local_cocu/utils/concept_utils.py：已有 gallery 校验。
- experiments/local_cocu_novel/filter_novel.py：已有词形与词边界匹配。
- experiments/Caption_Head_Visualization/：已有 448 crop 与注意力提取流程。
- src/open_vocabulary_segmentation/：原评估入口、投影权重加载、五基准配置、结果 JSON 写入。
- 既有 minimal_noun_supervision 包：复用离线 socket guard 的实现方式。

已在本次交付环境执行：Python 编译、Shell 语法检查、主要 CLI --help、6 项 NumPy/标准库核心测试、离线保护检查、评估 dry-run 和模拟五种子汇总验证。核心测试覆盖 GT ignore、等面积支持、证据区域聚合、筛选条件、图像级 bootstrap、无 GT 不支持 C、伪负概念排除和翻转坐标。

本次交付环境没有 PyTorch、CLIP、用户的 DINOv2/Teacher 权重与 COCO，因此训练/恢复训练和真实模型提取、五基准 OVSS **未实际运行**。测试套件中的一项 CPU PyTorch 集成测试已显式 skip，不能将其表述为通过。

该集成测试可在原 Talk2DINO 环境运行：构造小型合成特征，不加载预训练模型、不下载；检查 E2 与 E3 在 local_weight=0 时的权重逐值相同、预算相同且局部样本实际存在。

真实全量训练前必须先执行 `run.sh check`、`run.sh test`，再完成 A 与小规模 prepare 的来源一致性核验。未提供任何实际提升数值、性能保证或新颖性结论。
