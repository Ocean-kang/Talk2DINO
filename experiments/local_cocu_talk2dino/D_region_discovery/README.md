# D. Head Attention Region 验证

先在原图 Head 得到候选概念，用对应 Head 的 Attention Map 形成 448 像素参考系内的 Crop，经 DINOv2 再编码，结合局部/全局概念匹配分数重排。

运行（在仓库根目录）：

```bash
CUDA_VISIBLE_DEVICES=1 bash experiments/local_cocu_talk2dino/D_region_discovery/run.sh --stage all --gpu 1
```

按阶段执行：`--stage mine|build|train|eval|all`。

- 概念证据：`outputs/local_cocu_talk2dino/D_region_discovery/concepts_by_image.json`
- 原始 Caption 与 Concept：同一 JSON 条目，`image_id`、`raw_captions`、`concepts`（包含 `score`、`best_head`）。
- 原始训练缓存：`outputs/local_cocu_talk2dino/D_region_discovery/train_augmented.pth`
- 训练产物：`weights/local_cocu_talk2dino/D_region_discovery/`
- 分割评估：`outputs/local_cocu_talk2dino/D_region_discovery/eval/`

详细参数与依赖见上一级 `README.md`。
