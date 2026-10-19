# H. 多轮概念 Bootstrap

第 1 轮使用初始 Teacher 权重执行 A→补充原 5 条 Caption→原始 train.py 随机初始化 Student；第 2 轮起使用上一轮 Student 作为概念发现 Teacher，所有训练依然执行原始 train.py，默认 2 轮。

运行（在仓库根目录）：

```bash
CUDA_VISIBLE_DEVICES=1 bash experiments/local_cocu_talk2dino/H_iterative_bootstrap/run.sh --stage all --gpu 1
```

按阶段执行：`--stage mine|build|train|eval|all`。

- 概念证据：`outputs/local_cocu_talk2dino/H_iterative_bootstrap/concepts_by_image.json`
- 原始 Caption 与 Concept：同一 JSON 条目，`image_id`、`raw_captions`、`concepts`（包含 `score`、`best_head`）。
- 原始训练缓存：`outputs/local_cocu_talk2dino/H_iterative_bootstrap/train_augmented.pth`
- 训练产物：`weights/local_cocu_talk2dino/H_iterative_bootstrap/`
- 分割评估：`outputs/local_cocu_talk2dino/H_iterative_bootstrap/eval/`

详细参数与依赖见上一级 `README.md`。
