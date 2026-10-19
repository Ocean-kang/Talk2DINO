# C. Caption 未覆盖 Head 发现

用 Teacher 对原始 5 条 Caption 算覆盖分数，优先考虑低覆盖 Head，再用 DINOv2 注意力质量指标滤掉无明显响应 Head，对合格 Head 进行全局概念检索。

运行（在仓库根目录）：

```bash
CUDA_VISIBLE_DEVICES=1 bash experiments/local_cocu_talk2dino/C_uncovered_heads/run.sh --stage all --gpu 1
```

按阶段执行：`--stage mine|build|train|eval|all`。

- 概念证据：`outputs/local_cocu_talk2dino/C_uncovered_heads/concepts_by_image.json`
- 原始 Caption 与 Concept：同一 JSON 条目，`image_id`、`raw_captions`、`concepts`（包含 `score`、`best_head`）。
- 原始训练缓存：`outputs/local_cocu_talk2dino/C_uncovered_heads/train_augmented.pth`
- 训练产物：`weights/local_cocu_talk2dino/C_uncovered_heads/`
- 分割评估：`outputs/local_cocu_talk2dino/C_uncovered_heads/eval/`

详细参数与依赖见上一级 `README.md`。
