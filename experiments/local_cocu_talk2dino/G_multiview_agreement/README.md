# G. 多视图 Teacher 一致性

对原图、轻度亮度扰动、水平翻转分别重新提取 Head 并匹配；三种全画面视图保留原场景，无随机裁剪导致目标缺失；以多视图均值/标准差评分。

运行（在仓库根目录）：

```bash
CUDA_VISIBLE_DEVICES=1 bash experiments/local_cocu_talk2dino/G_multiview_agreement/run.sh --stage all --gpu 1
```

按阶段执行：`--stage mine|build|train|eval|all`。

- 概念证据：`outputs/local_cocu_talk2dino/G_multiview_agreement/concepts_by_image.json`
- 原始 Caption 与 Concept：同一 JSON 条目，`image_id`、`raw_captions`、`concepts`（包含 `score`、`best_head`）。
- 原始训练缓存：`outputs/local_cocu_talk2dino/G_multiview_agreement/train_augmented.pth`
- 训练产物：`weights/local_cocu_talk2dino/G_multiview_agreement/`
- 分割评估：`outputs/local_cocu_talk2dino/G_multiview_agreement/eval/`

详细参数与依赖见上一级 `README.md`。
