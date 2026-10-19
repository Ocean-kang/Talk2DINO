# E. 跨图 Head 概念一致性

在 B 的 Head 邻居中检查同一个 concept 对匹配的相似区域 Head 是否一致，高支持概念优先，保留罕见高置信概念分支。

运行（在仓库根目录）：

```bash
CUDA_VISIBLE_DEVICES=1 bash experiments/local_cocu_talk2dino/E_cross_image_consistency/run.sh --stage all --gpu 1
```

按阶段执行：`--stage mine|build|train|eval|all`。

- 概念证据：`outputs/local_cocu_talk2dino/E_cross_image_consistency/concepts_by_image.json`
- 原始 Caption 与 Concept：同一 JSON 条目，`image_id`、`raw_captions`、`concepts`（包含 `score`、`best_head`）。
- 原始训练缓存：`outputs/local_cocu_talk2dino/E_cross_image_consistency/train_augmented.pth`
- 训练产物：`weights/local_cocu_talk2dino/E_cross_image_consistency/`
- 分割评估：`outputs/local_cocu_talk2dino/E_cross_image_consistency/eval/`

详细参数与依赖见上一级 `README.md`。
