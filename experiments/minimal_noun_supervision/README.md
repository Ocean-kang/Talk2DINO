# Minimal Noun Supervision：已有 Gallery 版本

本版把 concept 定义为“已有 MSCOCO14 gallery 中、在该图原始五条 captions 内出现的概念”。不做语言学名词提取，不重建 gallery，不使用外部概念、同义词库或生成模型。所有步骤使用现有环境及本地文件；不需要任何依赖安装或环境调整。

## 1. 与上一版一致的训练协议

| 项目 | 本版行为 |
|---|---|
| CNE | 每个 C_i concept 独立成为 `a photo of {concept}`；完全替换原 captions |
| HVNE | 完整已有 gallery 的每 Head Top-K → 与 C_i 求交 → 原置信度筛选；每个留下的 concept 独立成为 prompt |
| CNP | 与 CNE 完全相同的 C_i，合并成一条 `a photo of c1, c2, c3` |
| Raw | 相同图像、相同 prompt-pool 采样的原五条 captions 对照 |
| 图像采样 | 每个 image pass 每图一次；每批图像不重复；原确定性 seed/epoch/image_id 抽取一个 prompt |
| 总更新次数 | `ceil(A_original / batch_size) * original_num_epochs`；A_original 是 limit 选定后、共同排除前的原 caption 数，四组完全相同 |
| 验证 | 每张原验证图固定第一条原 caption，使用原 `avg_self_attn_out`；四组共用 |
| 优化 | 原配置，原 `ProjectionLayer` / `ContrastiveLoss` / `train` / `validate` / scheduler；原 Adam 参数归属不变 |
| Student | seed 在随机初始化前设定；拒绝 starting_checkpoint；训练入口没有 Teacher 参数，不加载 Teacher 初值，无蒸馏 |
| 分割 | 原 `src/open_vocabulary_segmentation/main.py`、原五任务配置和 PAMR 协议 |

不把 image pass 当作原始 caption epoch。预算仍与上一版参考预算精确一致；最后一个 image pass 可以只执行部分 batch。

InfoNCE 中“不同图像使用相同名词 prompt”的负样本冲突仍存在，与上一版一样，不改损失去消除它。共享 prompt 数和潜在冲突计数写入统计文件。单图一个 batch 的边界情况仍遵循原采样实现。

## 2. 放置、输入和已有环境

在 Talk2DINO 仓库根目录执行。旧目录先备份；新包不覆盖仓库核心代码。

```bash
# 仅当旧实验目录存在时执行；备份名若已存在，请另取名字
mv experiments/minimal_noun_supervision experiments/minimal_noun_supervision_before_gallery
unzip /path/to/Talk2DINO_minimal_noun_supervision.zip

# 在你已经能运行 Talk2DINO 的终端中继续
export MNS_EXP=experiments/minimal_noun_supervision
export CUDA_VISIBLE_DEVICES=0

# 以下均为占位路径，必须替换为你已有的真实文件
export TRAIN_PTH=/your/existing/original_train.pth
export VAL_PTH=/your/existing/original_val.pth
export MNS_GALLERY=/your/existing/MSCOCO14_train_gallery.json
export MNS_CLIP_CHECKPOINT=/your/existing/ViT-B-16.pt
export MNS_DINO_CHECKPOINT=/your/existing/dinov2_vitb14_reg.pth
export MNS_TEACHER=/your/existing/vitb_mlp_infonce.pth
export MNS_CONFIG=configs/vitb_mlp_infonce.yaml
export MNS_FEATURE_METADATA=outputs/minimal_noun_supervision/original_feature_metadata.json
```

仓库曾使用的 gallery 相对路径是 `experiments/local_cocu/MSCOCO14_train_gallery.json`，训练特征相对路径曾为 `outputs/coco2014_b14/{train,val}.pth`。这些只是在源码里看到的默认值，不能证明你的服务器上文件存在；本版脚本不替你假定它们存在。

输入训练 PTH 必须是原版 COCO 字典，含 `images` 和 `annotations`，每张选定训练图恰好五条原 caption，图像里含非零 `disentangled_self_attn: [12,768]`。验证图像须含原 `avg_self_attn_out: [768]`。已有 `ann_feats: [512]` 可复用。独立 Local-CoCu `head_features.pt` 不是包含 captions 的训练 PTH，本入口不把它冒充原训练文件。直接使用原训练 PTH 内已经提好的 Heads，无需再次运行 DINOv2。

### 旧特征的来源核准

模型来源不能从 `[12,768]` / `[512]` 张量形状可靠推断。先核对你原有提取脚本、日志：CLIP ViT-B/16、DINOv2 ViT-B/14 reg、resize/crop 448、对应的本地 checkpoint。若已有符合下方 schema 的来源元数据，直接指定它；若旧缓存没有元数据，核对记录后执行一次：

```bash
python "$MNS_EXP/record_feature_metadata.py" \
  --train-features "$TRAIN_PTH" --val-features "$VAL_PTH" \
  --clip-checkpoint "$MNS_CLIP_CHECKPOINT" \
  --dino-checkpoint "$MNS_DINO_CHECKPOINT" \
  --confirm-source-models --output "$MNS_FEATURE_METADATA"
```

这条命令不提取特征，不加载模型，不联网。它保存两个特征文件与两个本地权重的 SHA256，以及你根据既有记录作出的来源声明；它不能替你证明历史上实际用了哪一个模型。来源不明时不能靠这条命令“认证”未知特征，应先找到原记录。本版不会悄悄重提 DINO 特征或获取资源。

字段：`clip_model`、`clip_checkpoint_sha256`、`dino_model`、`dino_checkpoint_sha256`、`resize=448`、`crop=448`、`head_field=disentangled_self_attn`、`head_shape=[12,768]`、`validation_field=avg_self_attn_out`、`clip_dim=512`、`raw_caption_template={}`、`raw_text_tokenizer_truncate=true`、两个 `*_features_sha256`、`source_model_attestation=true`。生成前实际检查本地文件存在；生成后数据处理会检查各值和哈希。

可选只读环境检查：

```bash
python "$MNS_EXP/check_environment.py"
python "$MNS_EXP/check_environment.py" --segmentation
```

只检查当前已有依赖。失败时会明确报出缺项；程序不会调整环境。

## 3. 数据生成：5k Pilot

```bash
export MNS_DATA=outputs/minimal_noun_supervision/data_pilot5k_gallery
export MNS_TAG=pilot5k_gallery

# CNE / CNP / Raw；同一组图像、同一 C_i，保留完整已有 gallery
bash "$MNS_EXP/scripts/01_prepare.sh" --limit 5000 --subset-seed 123

# 只有此步骤使用 Teacher；默认 K=20，置信度规则延续上一版
bash "$MNS_EXP/scripts/02_select_hvne.sh"
```

不要复用旧版数据输出目录：那是不同的概念定义。新入口拒绝已有非空输出目录，训练也拒绝旧版缺少 provenance 的 manifest。

`prepare.py` 总是处理完整既有 gallery；`--limit` 只选训练图像，不会缩小 HVNE 检索 gallery。`shared/gallery.json` 是已有 gallery 的有序归一化副本，不是从 pilot 或 captions 重新构建的词表。大小写、标点等同的重复表面写法合并并逐项记入 `matching_audit.json`；不按语义合并、不扩充词表。原 gallery 文件路径与哈希保留。

## 4. 匹配规则与限制

先审计并复用：

- `experiments/local_cocu/utils/concept_utils.py:normalize_gallery`：支持已有 gallery JSON 格式；其 `extract_words/match_gallery` 只支持拆词后的单词，故不直接用于本实验复合词。
- `experiments/local_cocu_novel/filter_novel.py:tokenize,in_raw_captions`：复用大小写/标点后的 token 化和完整连续 token 匹配接口，明确启用 `exact_only=True`，关闭原有广泛词尾裁剪。
- 标准库 trie 补充批量复合短语匹配，避免逐图逐条遍历整个 gallery；不是新增 NLP 工具。

默认 `--morphology exact`：ASCII 字母/数字 token，大小写归一；标点、连字符、空白变成 token 分隔；在单条 caption 内匹配完整连续 token 序列；不会跨两条 caption 拼接。`car` 不命中 `carpet`。`tennis-racket` 能命中 `tennis racket`。复合概念整体保留；若 gallery 同时含 `tennis racket` 和 `racket`，二者都可以成为独立 C_i 成员，不擅自删除已有词条。

同一 concept 在一条 caption 出现多次只计一次；概念按“出现于几条 caption”的频率降序、再按概念字典序排列。CNE 和 CNP 完全共用该顺序。

默认不做词形变化，因此 `dogs` 不自动命中 `dog`；不做同义词、词序改写、语义消歧或名词词性判断。gallery 的质量决定概念范围；`hot dog` 中的 `dog` 若也在 gallery 可以被词法命中，这不是物体存在证明。英文以外字母及不连续短语的覆盖有限。

如果确实需要已有的复数规则，可以在重新生成数据时指定：

```bash
bash "$MNS_EXP/scripts/01_prepare.sh" --limit 5000 --morphology repo_forms
```

该模式原样调用 `experiments/Head_caption_S5_Editing/editing.py:forms`：已有有限不规则表、末词 `s/x/z/ch/sh` 加 `es`、辅音+y→ies、其余正向生成复数。它不删除任意词尾 s，也不引入同义词。它不是完整英语形态学，例如原规则不把 `men` 对应到 `man`，且有些本身为复数或不可数的 gallery 词会得到不理想的附加形式。精确模式是默认；任何模式必须让四组共用同一份数据，不把两种模式混合比较。

每图每条命中的原 caption 索引、token span、匹配表面形式和规则都在 `shared/concepts_by_image.json`。可选模式会把对应源码哈希写入 provenance。

### 无匹配和对照可比性

C_i 为空：默认 `--empty-policy exclude`，从 Raw/CNE/CNP/HVNE **共同排除**，不添加 `object` 或其他词。`shared/matching_audit.json` 保存原 captions、图像 ID、排除原因、数量和比例。原先 limit 选中的集合和训练预算不变；保留图片会相应多经历一些 image passes。全部无匹配时停止，不生成可训练空池。

希望只要有空匹配就停止，可用 `--empty-policy error`；仍先写排除审计。验证集不按此规则排除，继续保留上一版共同固定 Raw 验证协议。所有最终训练 manifest 的图像 ID 必须与 provenance 中保留 ID 精确一致。

## 5. 复用已有文本缓存；否则本地编码

默认先复用来源核准后的原 `ann_feats`，再用你指定的**本地 checkpoint 文件路径**调用现有 `clip.load`、`clip.tokenize`、`encode_text`，仅为缺失 prompt 编码。不会把 `ViT-B/16` 模型名交给下载逻辑；DINO 权重只用于来源哈希核对，不执行图像模型。

有兼容文本 bank 时优先传入：

```bash
bash "$MNS_EXP/scripts/01_prepare.sh" --limit 5000 \
  --text-cache /your/existing/verified_text_bank.pth
```

可复用 bank schema：

```python
{
    'texts': [...],                   # 与每一行 embedding 一一对应的精确原文本
    'embeddings': tensor_of_M_by_512,  # 原 CLIP text 向量，不能是 Teacher 投影后 768 维
    'metadata': {
        'clip_model': 'ViT-B/16',
        'clip_checkpoint_sha256': '...',
        'prompt_template': 'a photo of {}',
        'feature_space': 'clip_text_raw',
        'gallery_concepts': [...],     # 与完整归一化 gallery 精确同序
        'texts_sha256': '...'
    }
}
```

缓存可以只覆盖部分 prompts，余下文本在本地补编码。维度、模型哈希、模板、gallery 概念顺序、实际文本行顺序哈希不匹配时直接拒绝，不根据向量形状猜测可复用。

若旧缓存只有裸 tensor 或字典，可在原有缓存生成记录完整的前提下显式封装，不重新编码：

```bash
python "$MNS_EXP/adopt_text_cache.py" \
  --cache /your/existing/raw_clip_gallery_cache.pth \
  --texts /your/existing/exact_encoded_texts_in_row_order.json \
  --gallery "$MNS_GALLERY" --clip-checkpoint "$MNS_CLIP_CHECKPOINT" \
  --source-template 'a photo of {}' --confirm-raw-clip-provenance \
  --output outputs/minimal_noun_supervision/verified_gallery_text_bank.pth
# 如果旧缓存为字典，额外指定其实际 --tensor-key，例如 embeddings
```

该工具不猜 key，不猜行顺序。`--texts` 必须来自已有生成记录，不可临时把 gallery 顺序当作已知缓存顺序。若原特征用裸 concept、不同 prompt、不同 CLIP 或 Teacher 投影，不能通过补元数据改称一致；省略 `--text-cache`，在既有本地 CLIP 上重新编码即可。

运行脚本会设置离线环境标志，并阻止远程 socket/URL 访问；本机通信保留以运行原分布式分割入口。原 evaluator 依赖的 DINO 源码、权重、数据等仍须已经在本地正确缓存，缺失时会报错，不自动获取。

## 6. HVNE 的真实检索范围和异常

对每个 Head 计算它和**全部已有 gallery**的余弦相似度，分块精确合并 Top-K，然后与 C_i 求交集；不会只给 C_i 打分并冒充检索。默认 K=20；相似度阈值来自训练内固定抽样的第 0.75 分位数；head margin 默认 >=0；gallery gap 默认只记录，不作为硬门槛。每 Head 至多一个，跨 Head 去重，每图至多五个，均延续上一版。

`head_margin` 是相对于其它 Heads 的分数差；`gallery_gap` 是相对于该 Head 最强其它 gallery concept 的分数差，因此非 Top1 concept 的 gap 可以是负数。分数、排名和所谓“置信度”是筛选诊断，并非校准后的物体存在概率。

某图没有高置信度结果：保持上一版的显式 fallback——只在其非空 C_i 内取最高 Head-concept 分数，独立 prompt。它的 `fallback=true`、`confidence_pass=false`，原因是 `fallback_caption_best_score_not_high_confidence`；额外计算该 concept 在对应 Head 的**完整 gallery 排名**，不伪造 Top-K 排名。这个局部排序仅用于 fallback；正常候选检索仍是完整 gallery。

训练内校准子集没有任何交集命中：阈值记为 1.0001、校准状态记为 `no_calibration_hits_all_fallback`，全部标记为 fallback，不能称为高置信度筛选成功。可审计后显式给 `--min-sim`，但不要根据分割测试结果调阈值。

`hvne/selection_audit.jsonl`：逐图相似度、完整 gallery rank、零基 Head、候选通过/拒绝原因、gallery 扫描规模和 fallback。`statistics.json` 给数量和比例。`strict_manifest.json` 仅是非 fallback 子集诊断，训练入口拒绝其 method；若未来要训练严格子集，需另建完全同图的 Raw 对照，当前主实验不走这一路径。

## 7. CNP 的长度规则

与 CNE 共用完全相同的 C_i。使用当前已有 OpenAI CLIP tokenizer，`truncate=False` 检查 77 token 上下文（包括特殊 token）。按既定“caption 频率降序、字典序”逐个加入完整 concept：若加入后超长，跳过该整个 concept，继续尝试后续较短 concept。不会切半个复合词，不默默依赖 tokenizer 截断组合文本。

每图输入、保留、移除概念和实际 token 数保存在 `cnp/composition_audit.json`；整体概念保留率在共享统计中。若已有 gallery 中某个独立 prompt 本身就超过限制，停止并报告该词，不偷偷缩小完整检索 gallery。

## 8. 三个独立训练入口与 Raw

```bash
bash "$MNS_EXP/cne/run.sh" 123
bash "$MNS_EXP/hvne/run.sh" 123
bash "$MNS_EXP/cnp/run.sh" 123
bash "$MNS_EXP/scripts/train_one.sh" raw 123
```

每组只读自己的 manifest、共同特征/验证/text bank。权重：`weights/vitb_mlp_infonce_mns_${MNS_TAG}_${method}_seed123.pth`；训练日志、best/last 和预算记录：`outputs/minimal_noun_supervision/${MNS_TAG}/${method}/seed_123/`。不覆盖已有权重或已有运行输出。

只做接口小检查时可用 `--max-steps 5 --workers 0`，这种结果不能当作正式训练精度。正式运行不加预算覆盖参数。原 config 中的 epoch、batch size、学习率和保存 best/last 行为继续生效。

## 9. 原版分割评估

```bash
for method in raw cne hvne cnp; do
  bash "$MNS_EXP/scripts/eval_one.sh" "$method" 123
done
# 只跑 VOC20
bash "$MNS_EXP/scripts/eval_one.sh" cne 123 --tasks voc20
# 只打印原评估命令，不运行
bash "$MNS_EXP/scripts/eval_one.sh" cne 123 --dry-run
```

| CLI task | 原配置目录 | 原结果 key |
|---|---|---|
| voc20 | voc | voc20 |
| voc（含背景） | voc_bg | voc |
| context59 | context | context59 |
| ade20k | ade | ade20k |
| coco_stuff | stuff | coco_stuff |

默认 PAMR=on，与上一版一致。数据集路径仍由你**现有原版 eval YAML**指定；本包不硬编码你的服务器地址。先确认原评估配置能找到已有 VOC/Context/ADE/COCO-Stuff。输出放在各 seed 的 `segmentation/on/<task>/`。

例如 CNE VOC20 的实际原入口命令等价于：

```bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 \
  --master_addr=127.0.0.1 --master_port=29781 \
  src/open_vocabulary_segmentation/main.py --eval \
  --eval_cfg src/open_vocabulary_segmentation/configs/voc/dinotext_voc_vitb_mlp_infonce.yml \
  --eval_base_cfg src/open_vocabulary_segmentation/configs/voc/eval_voc_pamr.yml \
  --output "outputs/minimal_noun_supervision/$MNS_TAG/cne/seed_123/segmentation/on/voc20" \
  --opts "model.proj_name=vitb_mlp_infonce_mns_${MNS_TAG}_cne_seed123"
```

推荐使用 wrapper，它额外检查权重、保存原日志和逐任务指标，并启用离线保护；不改分割计算。

## 10. 全量五 seeds

```bash
export MNS_DATA=outputs/minimal_noun_supervision/data_full_gallery
export MNS_TAG=full_gallery
bash "$MNS_EXP/scripts/01_prepare.sh"
bash "$MNS_EXP/scripts/02_select_hvne.sh"
for method in raw cne hvne cnp; do
  bash "$MNS_EXP/scripts/train_five_seeds.sh" "$method"
  bash "$MNS_EXP/scripts/eval_five_seeds.sh" "$method"
done
python "$MNS_EXP/summarize.py" --tag "$MNS_TAG"
```

种子保持 123/124/125/126/127。完整结果齐全后汇总 mean、sample std 和相同 seed 下 method−Raw 的差值；不静默把缺失 seed 当成完整结果。

## 11. 代码审计、改动和实际验证

阅读 `CHANGES.md`、`UPSTREAM_AUDIT.json` 和 `VALIDATION.md`。仓库 src、原 train.py 和 eval 配置都未修改；本包只修改实验目录。源码读取通过只读渠道完成，没有向 GitHub 写入。
