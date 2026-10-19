# A. 直接 Head→全局 Gallery 检索

不使用图像邻居。CLIP Text Encoder + 冻结的 Talk2DINO Projector 对全局概念编码；每个 Head 执行 top-k，过滤原文案中出现的概念。

运行（在仓库根目录）：

```bash
CUDA_VISIBLE_DEVICES=1 bash experiments/local_cocu_talk2dino/A_direct_retrieval/run.sh --stage all --gpu 1
```

按阶段执行：`--stage mine|build|train|eval|all`。

- 概念证据：`outputs/local_cocu_talk2dino/A_direct_retrieval/concepts_by_image.json`
- 原始 Caption 与 Concept：同一 JSON 条目，`image_id`、`raw_captions`、`concepts`（包含 `score`、`best_head`）。
- 原始训练缓存：`outputs/local_cocu_talk2dino/A_direct_retrieval/train_augmented.pth`
- 训练产物：`weights/local_cocu_talk2dino/A_direct_retrieval/`
- 分割评估：`outputs/local_cocu_talk2dino/A_direct_retrieval/eval/`

详细参数与依赖见上一级 `README.md`。
