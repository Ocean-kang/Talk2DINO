# F. 难负例反事实筛选

先从局部 Head 邻居获取候选；使用 CLS 相似的近邻提供相似场景负例，保留局部 concept 证据较低的一半计算负例均值，计算目标分数减去 λ 倍负例分数。

运行（在仓库根目录）：

```bash
CUDA_VISIBLE_DEVICES=1 bash experiments/local_cocu_talk2dino/F_counterfactual/run.sh --stage all --gpu 1
```

按阶段执行：`--stage mine|build|train|eval|all`。

- 概念证据：`outputs/local_cocu_talk2dino/F_counterfactual/concepts_by_image.json`
- 原始 Caption 与 Concept：同一 JSON 条目，`image_id`、`raw_captions`、`concepts`（包含 `score`、`best_head`）。
- 原始训练缓存：`outputs/local_cocu_talk2dino/F_counterfactual/train_augmented.pth`
- 训练产物：`weights/local_cocu_talk2dino/F_counterfactual/`
- 分割评估：`outputs/local_cocu_talk2dino/F_counterfactual/eval/`

详细参数与依赖见上一级 `README.md`。
