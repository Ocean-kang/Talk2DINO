# B. 跨图 Head 图像检索

对图像的 12 个 Head 独立执行 FAISS 图像局部匹配，允许任意 head-to-head 匹配。使用邻居原始 5 条 COCO 文案与 Gallery 提取候选，Teacher 匹配。

运行（在仓库根目录）：

```bash
CUDA_VISIBLE_DEVICES=1 bash experiments/local_cocu_talk2dino/B_head_image_retrieval/run.sh --stage all --gpu 1
```

按阶段执行：`--stage mine|build|train|eval|all`。

- 概念证据：`outputs/local_cocu_talk2dino/B_head_image_retrieval/concepts_by_image.json`
- 原始 Caption 与 Concept：同一 JSON 条目，`image_id`、`raw_captions`、`concepts`（包含 `score`、`best_head`）。
- 原始训练缓存：`outputs/local_cocu_talk2dino/B_head_image_retrieval/train_augmented.pth`
- 训练产物：`weights/local_cocu_talk2dino/B_head_image_retrieval/`
- 分割评估：`outputs/local_cocu_talk2dino/B_head_image_retrieval/eval/`

详细参数与依赖见上一级 `README.md`。
