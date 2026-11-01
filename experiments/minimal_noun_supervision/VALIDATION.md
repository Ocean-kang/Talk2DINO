# 实际验证范围

本次修改没有创建环境，没有安装、升级、降级依赖，没有获取模型或语言资源，没有向 GitHub 写入。

使用工作区已经存在的 Python 3.10.22 / PyTorch 2.1.0+cpu / NumPy 1.24.1 运行 12 项 unittest；源文件编译和所有 shell 脚本的 bash 语法检查通过。依赖不需要按这些小版本重新调整，这只是本次验证机器的实际版本。

## 12 项检查

1. 复合词、大小写/标点、完整 token 边界，car 不命中 carpet，不跨两条 captions 拼接词组。
2. 默认不裁剪复数词尾；可选模式调用仓库原 forms 规则并保留其已知限制。
3. concept 去重、caption 频率、确定性排序。
4. 已有四种 gallery JSON 格式、有序归一化、重复写法审计。
5. CNP 以完整 concept 贪心跳过超长项。
6. 分块完整 gallery Top-K 对照密集矩阵；平分时按 gallery ID 排序；fallback 排名可追踪。
7. 对照显示 C_i 内最好但全 gallery 排名在 K 之外的 concept 不能成为正常候选。
8. 每批/每 pass 图像唯一、原始 annotation 预算精确、prompt 采样可重复、空池拒绝。
9. 缓存的模型、模板、概念顺序、文本行顺序、维度和特征空间不符时拒绝。
10. 远程网络守卫拒绝外部地址，保留本机通信。
11. 五个原版分割任务配置、权重名称和命令路由 dry-run；未运行 MMSegmentation。
12. 端到端 CPU 合成数据：已有 gallery -> CNE/CNP/Raw -> HVNE -> 四组各五次 optimizer updates，使用真实上游 ProjectionLayer、ContrastiveLoss、train 和 validate；输出可被原模型 strict=True 加载。也检查了共同排除、排除前预算、无匹配全体停止、全 bank 复用零次重编码、强制全部 fallback、无校准命中全部 abstain、CNE/CNP 共享 C_i、训练过程不读取 Teacher 权重。

集成测试中的 CLIP encoder/tokenizer 是确定性替身，DINO Heads 为合成张量，Teacher 为原 ProjectionLayer 的随机权重；它验证接口和控制变量，不验证预训练模型语义质量。实际匹配代码使用真实仓库 helper，无 NLP 替身。

未运行：真实 MSCOCO14 gallery/原图/权重上的全量数据生成、真实 CLIP prompt 编码、CUDA 训练、任何真实分割评估或五 seeds 精度统计。当前工作区没有你的这些实际资源。不能据此声称分割有效或优于 baseline。

## 在你的已有环境复查

从仓库根目录：

```bash
python -m unittest discover -s experiments/minimal_noun_supervision/tests -v
```

测试生成的数据和运行输出在临时目录及独立随机标签下，测试结束会清理自己生成的内容。测试不会读取你的 Teacher 或改动已有运行。正式数据与训练指令见 README。
