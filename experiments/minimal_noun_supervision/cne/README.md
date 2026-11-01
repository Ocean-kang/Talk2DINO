# CNE

独立训练入口：在 Talk2DINO 根目录执行 `bash experiments/minimal_noun_supervision/cne/run.sh 123`。

读取 `$MNS_DATA/cne/manifest.json`；使用独立权重名与 `outputs/minimal_noun_supervision/$MNS_TAG/cne/` 输出目录。

数据生成、运行前设置和评估指令见上级 README.md。训练仅依赖当前方法的 manifest 和共享原始图像/文本特征；Teacher 仅在 HVNE 数据生成阶段使用。
