# Talk2DINO 5K Rich-Diversity-Strengthen

本实验接续上一轮 5K Rich-Diversity。默认只训练新增的 `rich_diversity_strengthen` 组，从 `rich_diversity_strengthen_captions` 读取每张图的 5 条描述。也可以将 `raw`、`rich`、`rich_weaken`、`rich_diversity` 和 `rich_diversity_strengthen` 五组一起重训。

## 1. 放置文件

在 Linux 训练服务器上，进入含有 `train.py` 和 `configs/` 的 **Talk2DINO 仓库根目录**。ZIP 内已包含 `experiments/` 路径。

~~~bash
cd /path/to/Talking2dino/Talk2DINO
unzip /path/to/5k_rich_diversity_strengthen_experiment.zip -d .
~~~

~~~text
Talk2DINO/
├── train.py
├── data/coco2014_subset_50k/
│   ├── 5k_rich_diversity_strengthen.json
│   └── train2014/COCO_train2014_000000000077.jpg  # 其余图片同目录
└── experiments/5k_rich_diversity_strengthen/
    ├── README.md
    ├── prepare.py
    ├── run_preprocess.sh
    ├── run_train.sh
    ├── run_seg_eval.sh
    └── collect_results.py
~~~

ZIP 只含实验脚本和说明。图片可直接放在数据根目录，也可位于 `train2014/`、`val2014/` 或 `images/` 及其 `train2014/`、`val2014/` 子目录。若服务器图片在其他位置，给预处理设置 `DATA_DIR=/path/to/coco`。若 JSON 也在其他位置，再设置 `CAPTIONS=/path/to/5k_rich_diversity_strengthen.json`。

## 2. 跑新增组

激活上一轮的 Talk2DINO 环境，使用训练服务器的 CUDA GPU。分割评估还需要原仓库配置中指向的四个 benchmark 数据集及既有 MMCV/MMSeg 环境。

~~~bash
conda activate talk2dino
export CUDA_VISIBLE_DEVICES=0
EXP=experiments/5k_rich_diversity_strengthen

bash "$EXP/run_preprocess.sh"
bash "$EXP/run_train.sh"
bash "$EXP/run_seg_eval.sh"         # No PAMR
PAMR=1 bash "$EXP/run_seg_eval.sh"  # PAMR
~~~

预处理按上一轮顺序将图像 ID 排序，再用 `random.Random(123)` 打乱并取前 500 个作验证集。训练集有 4,500 张图、22,500 条 caption，验证集有 500 张图、2,500 条 caption。每张图用 DINOv2 `dinov2_vitb14_reg` 提取 448 px 特征，训练用 `disentangled_self_attn`，验证用 `avg_self_attn_out`。文本沿用原版 CLIP ViT-B/16 提取器与 77 token 上限，输出 `ann_feats`。

训练沿用 `train.py` 和 `configs/vitb_mlp_infonce.yaml`，即 100 epochs、batch size 128、Adam、lr `1e-4`、InfoNCE、linear scheduler、warmup 0、seed 123，保存最后一轮投影层权重。分割沿用上一轮的 VOC20、Context59、COCO-Stuff 和 ADE20K 配置。原仓库入口见 [Talk2DINO](https://github.com/Ocean-kang/Talk2DINO)。

预处理显存不足时可用 `DINO_BATCH_SIZE=4 CLIP_BATCH_SIZE=128 bash "$EXP/run_preprocess.sh"`，这两个变量只影响特征提取 batch size。评估可用 `BENCHMARKS="voc20 coco_stuff"` 选择子集。

## 3. 与上一轮对比

本地新旧数据已核对，5,000 个 ID、5,000 张 JPEG 的文件内容以及四组原有 captions 全部一致。要复用上一轮的对照结果，在服务器上核对实际使用的 JSON 和旧 `val_ids.json`，并确认旧训练配置及评估的 PAMR 模式一致。

~~~bash
python "$EXP/prepare.py" check \
  --captions data/coco2014_subset_50k/5k_rich_diversity_strengthen.json \
  --images data/coco2014_subset_50k \
  --previous-captions data/coco2014_subset_50k/5k_rich_diversity.json \
  --previous-val-ids outputs/5k_rich_diversity/work/val_ids.json
~~~

如果已确认旧 `dino.pth` 来自同一批图片、`dinov2_vitb14_reg` 与 448 px 预处理，可以直接复用，省掉视觉特征提取。脚本会核对缓存的 ID 和两种 DINO 特征形状。

~~~bash
DINO_SOURCE=outputs/5k_rich_diversity/work/dino.pth \
PREVIOUS_CAPTIONS=data/coco2014_subset_50k/5k_rich_diversity.json \
PREVIOUS_VAL_IDS=outputs/5k_rich_diversity/work/val_ids.json \
bash "$EXP/run_preprocess.sh"
~~~

优先比较 **Strengthen − Rich-Diversity**。上一轮 Rich-Diversity 的归档 mIoU 如下。

| 评估模式 | VOC20 | Context59 | COCO-Stuff | ADE20K |
| --- | ---: | ---: | ---: | ---: |
| No PAMR | 84.67 | 36.55 | 25.25 | 19.00 |
| PAMR | 86.69 | 38.92 | 27.08 | 20.25 |

若找不到旧验证 ID 或无法确认配置，用同一份新 JSON 重训五组。为独立保存五组结果，下面使用单独的权重标签。

~~~bash
ARMS="raw rich rich_weaken rich_diversity rich_diversity_strengthen"
export WEIGHT_TAG=subset5k_strengthen_all
PREP_ARMS="$ARMS" bash "$EXP/run_preprocess.sh"
TRAIN_ARMS="$ARMS" bash "$EXP/run_train.sh"
EVAL_ARMS="$ARMS" bash "$EXP/run_seg_eval.sh"
PAMR=1 EVAL_ARMS="$ARMS" bash "$EXP/run_seg_eval.sh"
~~~

## 4. 查看输出

- 默认权重为 `weights/vitb_mlp_infonce_subset5k_strengthen_rich_diversity_strengthen.pth`。训练脚本会拒绝覆盖同名权重。重跑用新的 `WEIGHT_TAG`，评估时使用相同标签。
- `outputs/5k_rich_diversity_strengthen/work/` 保存特征、`val_ids.json` 和 `audit_report.json`。每次成功预处理的组可从审计报告查到。更换图片源时用新的 `WORK_DIR`，并在训练时传入同一个目录。同一工作目录内依次执行预处理和训练。
- `outputs/5k_rich_diversity_strengthen/logs/` 保存训练及分割日志。`config_snapshot/` 按权重标签和组保留对应的训练配置、数据审计及验证 ID。
- `segmentation_results/` 下的结果 JSON 按 `_no_pamr` 与 `_pamr` 分开命名。评估时在 `weights/` 下建立相应的权重软链接，两个模式读取同一训练权重。
- `outputs/5k_rich_diversity_strengthen/summary_<WEIGHT_TAG>.csv` 在评估后汇总，两个 PAMR 模式分别记录 mIoU。分批评估不同组时，最后可以显式列出所有要汇总的组。

~~~bash
python "$EXP/collect_results.py" --weight-tag subset5k_strengthen_all \
  --arms raw rich rich_weaken rich_diversity rich_diversity_strengthen
~~~
