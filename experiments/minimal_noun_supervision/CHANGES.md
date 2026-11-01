# 本次修改与复用

## 删除

- 删除上一版 `requirements-extra.txt`。
- 删除 `common/nouns.py` 中 spaCy/POS/noun_chunks/语言模型加载逻辑，移除其预检与所有安装指令。
- 删除从当前训练子集重新生成 Gallery、以及无名词时添加通用 object 的行为。

如果服务器已解压旧版，请先备份/改名旧实验目录再解压新包，避免旧的增量依赖文件残留。

## 修改文件

| 文件 | 改动 |
|---|---|
| common/nouns.py | 以已有 G 定义 concept；复用 gallery 格式、token/边界接口，标准库 trie 保留复合概念；可选仅复用原 forms |
| prepare.py | 必须指定已有 G/本地权重/来源元数据；共用 C_i；共同排除；完整 G 的 prompt 缓存；CLIP 长度审计 |
| select_hvne.py | 读取完整已有 G 而非 pilot concept 并集；候选原因；fallback 真正全 gallery 排名；来源/缓存核准 |
| common/selection.py | 完整分块检索保留；平分排序确定性；新增 fallback 的精确全 gallery 排名 |
| train.py | 仅新增数据/缓存来源、共同 ID 和预算一致性检查，以及离线保护；不改变上一版采样、更新预算、模型、损失、核心函数调用 |
| evaluate.py | 原任务路由、命令、结果协议保留，仅启用离线环境与访问保护 |
| audit_heads.py | 头部诊断保留，新增缓存来源检查及离线保护 |
| check_environment.py | 移除新增 NLP 预检，检查已有环境，不执行安装 |
| scripts/env.sh、01_prepare.sh、02_select_hvne.sh | 输入路径显式由参数/环境变量传入，已有环境离线运行 |
| README.md、VALIDATION.md、tests/* | 改写运行说明与验证用例，移除旧 NLP 逻辑；说明旧缓存的来源限制 |

新增 `common/cache.py`、`record_feature_metadata.py`、`adopt_text_cache.py`、`offline_guard/sitecustomize.py` 和本说明。来源核准工具只读取已有文件；缓存封装只接受根据已有生成记录明确知道来源和文本行顺序的特征，不猜测、不假装验证未知来源。

三个 `cne/run.sh`、`hvne/run.sh`、`cnp/run.sh`、训练/评估五 seeds shell、`common/data.py`、`common/io.py`、`summarize.py` 的实验协议保持上一版。三个独立入口不引入单独训练框架。

## 直接复用的仓库代码

| 原有代码 | 复用方式与原因 |
|---|---|
| experiments/local_cocu/utils/concept_utils.py | 直接调用 normalize_gallery 验证原格式；其拆词匹配无法保留短语，故用标准库补充 |
| experiments/local_cocu_novel/filter_novel.py | 直接调用 tokenize 与 in_raw_captions(exact_only=True)；关闭宽泛词尾处理 |
| experiments/Head_caption_S5_Editing/editing.py | 仅可选 repo_forms 模式直接调用 forms，不采用对象白名单/编辑策略 |
| src/model.py | 原 ProjectionLayer，Teacher 离线投影与 Student 随机初始化都使用原类 |
| src/loss.py | 原 ContrastiveLoss/InfoNCE，无新增 loss、网络或蒸馏 |
| src/train_util.py | 原 set_seed/train/validate/const_lr/cosine_lr；原 optimizer 参数归属保持 |
| 原训练 PTH | 直接使用其 disentangled_self_attn、avg_self_attn_out 和经来源核准的 ann_feats |
| text_features_extraction.py | 审计并保持其 clip.tokenize/encode_text 原路径；原 CLI 会以模型名自动获取缺失权重，故不直接调用该 CLI，而以本地文件调用相同已有 CLIP API |
| local_cocu_talk2dino/common/teacher.py | 审计其 template 缓存和相似度定义；其缓存无磁盘来源校验、且默认按模型名加载，因此本次不直接实例化它 |
| src/open_vocabulary_segmentation/main.py 与原 YAML | 原样运行，五任务不改；模型/数据须已在本地 |

不修改 src、原 train.py、原 configs、任何其他实验代码，也没有远程写入。已有 helper 若在你的 checkout 缺失会报出确切路径；不自动下载它们。相对原版 ZIP 的逐文件哈希差异记录见 `FILE_CHANGES.json`。
