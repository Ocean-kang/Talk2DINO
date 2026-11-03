# 2026-11-01：manifest 无新增依赖修复

根因：上一版 make_manifest.py 无条件导入 pycocotools，但原环境未必安装了它。
manifest 阶段退出后，diagnose 找不到 outputs/ecca/A/manifest.json，是同一故障的后续错误。

本版直接读取现有 captions/instances JSON，使用标准库、已有 NumPy/Pillow 生成 PNG。
支持多 Polygon 的并集、非压缩 RLE、压缩 RLE（包括有符号差分），保留 COCO 列优先顺序。
不以 bbox 替代 mask，也不将 crowd 当背景。与上一版一样，crowd 和所有实例重叠像素标为 255，
未标注区域为 0；diagnose 忽略 0 和 255。这些仍然是诊断掩码，不是官方 OVSS 评估结果。
manifest 写入 mask_backend，保留生成方法的来源记录。

另外，diagnose 会在创建输出和加载模型之前检查 manifest；run.sh 也有相同检查。
兼容旧 Pillow 没有 Image.Resampling / Image.Transpose 的命名形式。

## 更新已解压的实验

在 Talk2DINO 仓库根目录执行，替换 ZIP 路径为你的下载位置：

```bash
unzip -o /path/to/Talk2DINO_ECCA_A_B_gated_C.zip 'experiments/ecca/*' -x 'experiments/ecca/settings.sh'
bash experiments/ecca/run.sh test
bash experiments/ecca/run.sh manifest && \
CUDA_VISIBLE_DEVICES=0 bash experiments/ecca/run.sh diagnose
```

上面的解压命令保留你已经填写的 settings.sh。
manifest 不使用 GPU 或模型；成功写出清单后才会运行 diagnose。
确认 settings.sh 的 CAPTIONS_JSON 和 INSTANCES_JSON 指向同一 COCO split 的现有文件。
不需要重新提取 train.pth / val.pth。
原错误留下的空 diagnosis 目录不影响重试；如果目录已有实际诊断产物，程序仍拒绝覆盖。

## 本次验证

- 14 项本地测试通过；1 项 PyTorch 集成测试因交付环境无 PyTorch 而跳过。
- 在明确阻止导入 pycocotools 的子进程中运行真实 manifest CLI，两次运行均成功。
- 测试包含 Polygon、分数坐标、多 Polygon、压缩/非压缩 RLE、非法编码、crowd、
  重叠像素、非连续 category_id 映射，以及缺失 manifest 不加载模型、不创建输出目录。
- 额外对照官方 cocoapi/common/maskApi.c：2000 个随机多边形以及对应 2000 个压缩 RLE，
  输出逐像素一致；包含分数坐标、出界、凹/自交多边形。
  该对照仅在交付机临时编译官方 C 文件，不要求用户安装包、编译或下载任何资源。
- Python 编译和 bash 语法检查通过。
- 未在用户 GPU、模型与真实全量 COCO 上执行 diagnose 或训练；没有任何新的性能结果。

多边形/RLE 转换参考 COCO 官方实现，版权与许可见 COCO_API_LICENSE.txt。
