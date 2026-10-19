# Local-CoCu × Talk2DINO — New Caption Training Modes (Ponytail)

本实验只复用 `outputs/local_cocu_talk2dino/` 已经抽取的 concepts，**不会重复运行 CoCu/Teacher/Head/CLS 特征提取**；不修改 `train.py`、`src/`、原 `configs/`、原 5+3 实验，仍调用原始 Talk2DINO `train.py` 训练全新 Student（随机初始化，同一 InfoNCE、head-to-caption matching、原始验证数据）。仅改变训练 PTH 里的 caption 文本与 CLIP `ann_feats`。

## 两种训练模式

- **`5plus1`**：每张图片保留原始 5 条 caption 及其已有 CLIP features。将从该图已有 concept 中选出的最多 3 个真正新增的 concept 合成**一条**新英文 caption，例如 `A photo showing cake, fork, and plate.`；只给这条新 caption 单独调用 CLIP ViT-B/16 编码。某图无可用 concept 时该图仍是 5 条（不会编造第六条），在 manifest 中有计数。这是“5 + 最多 1”，有 concept 的图片恰好 6 条。
- **`5`**：每个候选 concept（使用 `a photo of a {concept}` 编码）与该图**原始 5 条 caption** 的 CLIP ViT-B/16 文本特征计算余弦相似度，将 concept 指派到**相似度最小**（语义最远）的一条原始 caption；同一句分配到多个 concept 时合并追加 `Also visible in the image: ...`。保持每张图严格 **5 条训练文本**，修改后的 caption 重新计算 CLIP `ann_feats`；未修改的原始特征原样复用。所有选择都以**原始** 5 条为相似度基准，不用已经修改的句子滚动匹配。

两方案再次排除五条原始 caption 中已直接出现的词/短语和重复 concept，最多接受 `--max-concepts 3` 个。新文本只是将已有 concept 进行确定性组合，并不额外检查视觉真实性。原图像缓存（包括 `disentangled_self_attn`）、验证数据和训练损失不变。结果以 original Talk2DINO 学习过程为准，概念发现 Teacher **不参与训练**。

## 文件布局

将 ZIP 解压到仓库 `experiments/` 下，使得：

```text
Talk2DINO/
  experiments/
    local_cocu_talk2dino/               # 已有 A-H（不修改）
    local_cocu_talk2dino_new_train/     # 本压缩包根目录
      build_train.py                    # 兼容 base train.pth 的数据构建
      pipeline.py                       # Train/Eval 统一入口
      run.sh                            # A-H × 5plus1|5
      five_plus_one/run.sh              # 5plus1 入口
      five_residual/run.sh              # 5 入口
      tests/test_build_train.py
      README.md
  outputs/coco2014_b14/train.pth
  outputs/coco2014_b14/val.pth
  outputs/local_cocu_talk2dino/A_direct_retrieval/concepts_by_image.json
  configs/vitb_mlp_infonce.yaml
  train.py
```

默认源 JSON 位置由 `--scheme A..H` 自动解析；H 使用 `H_iterative_bootstrap/round_1/concepts_by_image.json`（可用 `--round 2` 替换，构建数据和训练输出也按 `round_2` 独立隔离）。也可以通过 `--concepts /absolute/path/to/concepts_by_image.json` 覆盖。支持 `matched_concepts_detailed.json`（相同的 image-id 字典结构）。

**端口与训练进度条**：子进程使用 `MASTER_PORT=0`，由操作系统/TCPStore 选择临时端口，避免固定的 `29501` 冲突。训练阶段按字节块实时转发 stdout/stderr，保留 tqdm 的回车刷新行为；完整输出仍写入 `train.log`。\n\n**运行前验证**：所有数据相对路径均以仓库根目录为起点。默认检查提取结果至少覆盖训练 PTH 的全部 image_id；没有 concept 的已有图允许退回原始 5 条。若你的 concept 文件本来只来自 5k 子集，请同步指定对应子集 base PTH，或明确使用 `--missing-policy subset`（仅对交集训练，不能直接与全量 baseline 公平对照）。`--missing-policy keep_raw` 可以为缺失 ID 保留五条原始 caption。

## 环境

使用原仓库的环境（需已有 `torch`、`clip`、COCO 的 train/val PTH，及原分割评估依赖）。无需 FAISS、无需 DINO Teacher 权重，**只在 build 阶段用 CLIP 生成文本特征**。

```bash
cd /home/master/code/oymk/Talking2dino/Talk2DINO
conda activate /home/master/software/oymk/talk2dino
python -m unittest discover -s experiments/local_cocu_talk2dino_new_train/tests -v
```

## 直接运行 A 方案（使用物理 GPU 1）

```bash
# 第一种：5 条 raw captions + 1 条 concept 组合 caption
bash experiments/local_cocu_talk2dino_new_train/run.sh A 5plus1 all 1

# 第二种：把 concept 填入语义最远的那条 caption，仍为 5 条
bash experiments/local_cocu_talk2dino_new_train/run.sh A 5 all 1
```

**两种模式可以独立按阶段执行**，例如：

```bash
# 只构造 5+1 训练数据（会从已有 concepts_by_image.json 直接读取）
bash experiments/local_cocu_talk2dino_new_train/run.sh A 5plus1 build 1

# 用原 train.py 训练新初始化 Student（独立权重）
bash experiments/local_cocu_talk2dino_new_train/run.sh A 5plus1 train 1

# 在五个分割数据集上评估
bash experiments/local_cocu_talk2dino_new_train/run.sh A 5plus1 eval 1

# 模式 5 同理
bash experiments/local_cocu_talk2dino_new_train/run.sh A 5 build 1
bash experiments/local_cocu_talk2dino_new_train/run.sh A 5 train 1
bash experiments/local_cocu_talk2dino_new_train/run.sh A 5 eval 1
```

A-H 的所有旧提取结果都能分别喂给两个新模式。以 B 和 H 为例：

```bash
bash experiments/local_cocu_talk2dino_new_train/run.sh B 5plus1 all 1
bash experiments/local_cocu_talk2dino_new_train/run.sh B 5 all 1
bash experiments/local_cocu_talk2dino_new_train/run.sh H 5plus1 all 1 --round 2
```

**自定义 concept 文件**：

```bash
bash experiments/local_cocu_talk2dino_new_train/run.sh A 5 all 1 \
  --concepts outputs/local_cocu_talk2dino/A_direct_retrieval/concepts_by_image.json
```

使用对应实验文件夹独立启动：

```bash
bash experiments/local_cocu_talk2dino_new_train/five_plus_one/run.sh A all 1
bash experiments/local_cocu_talk2dino_new_train/five_residual/run.sh A all 1
```

`--dry-run` 可在不加载任何模型下检查路径规划；`--force-build`、`--force-train`、`--force-eval` 分别明确覆盖**本实验**对应阶段已有产物；权重不会覆盖原版 baseline 或旧的 5+3 权重。

## 输出与权重

以 A 方案为例：

```text
outputs/local_cocu_talk2dino_new_train/A_direct_retrieval/
  5plus1/
    train_augmented.pth
    train_manifest.json             # image_id, raw_captions, selected_concepts, augmented_captions
    build.log
    train.log
    eval/results.json               # 5 项 mIoU / baseline / delta
    eval/<dataset>/run.log
    eval/<dataset>/metrics.json
  5/
    train_augmented.pth
    train_manifest.json             # 附每个 concept 的 5 句相似度和指派编号
    build.log
    train.log
    eval/results.json

weights/local_cocu_talk2dino_new_train/A_direct_retrieval/
  5plus1/vitb_mlp_infonce_local_cocu_new_A_5plus1.pth
  5/vitb_mlp_infonce_local_cocu_new_A_5.pth
```

原 segmentation 代码只搜索 `weights/<proj_name>.pth`，本实验为每个权重创建一个**顶层符号链接**（不重复拷贝权重），并传递 `--opts model.proj_name=<proj_name>`。原评估程序写入 `segmentation_results/<proj_name>.json`，每完成一个 task 本脚本立即快照至各数据集独立路径。

## 五个分割评估（原 PAMR 配置）

| 数据集 | 原配置 | 提供的 baseline mIoU (%) |
|---|---|---:|
| VOC20 | `voc/dinotext_voc_vitb_mlp_infonce.yml` + `voc/eval_voc_pamr.yml` | 88.54 |
| VOC（含背景） | `voc_bg/dinotext_voc_bg_vitb_mlp_infonce.yml` + `voc_bg/eval_voc_bg_pamr.yml` | 65.77 |
| COCO-Stuff | `stuff/dinotext_stuff_vitb_mlp_infonce.yml` + `stuff/eval_stuff_pamr.yml` | 30.23 |
| ADE20K（ade20k） | `ade/dinotext_ade_vitb_mlp_infonce.yml` + `ade/eval_ade_pamr.yml` | 22.53 |
| Context59 | `context/dinotext_context_vitb_mlp_infonce.yml` + `context/eval_context_pamr.yml` | 42.42 |

执行：

```bash
bash experiments/local_cocu_talk2dino_new_train/run.sh A 5plus1 eval 1 \
  --tasks voc20,voc,coco_stuff,ade20k,context59
```

表格是**你提供的基线数值**，本仓库评估配置是对应 task 的原 YAML；并非已完成的新实验结果。确保 baseline 是在**同一套 YAML / PAMR / checkpoint /数据路径**下测得，mIoU delta 才具有可比性。新实验还没有在你的 COCO 数据及 GPU 上跑完训练/分割评估。

## 消融解释

相对于原 5+3，`5plus1` 缩小每图新添加的 annotation 数，可能缓解单词式 caption 与 Head 匹配时的噪声；`5` 完全不增加 annotation 数，测试“低相似性概念分配给描述不足的 caption”是否优于新加文本。注意 **语义最远并不能保证语义上相容**，仍可能给错误 caption 强行追加 concept。建议从 `train_manifest.json` 随机抽查分配结果，并在报告中统计 `images_without_novel_concepts`。
