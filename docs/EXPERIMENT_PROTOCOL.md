# EvidenceTrack-GS：实验目的与可证伪协议

本文档描述当前仓库真正要回答的问题，不预设优越性、单调提升或 SOTA 结论。

## 1. 核心问题

扩散模型生成的 pseudo-view 即使视觉上更逼真，也不等于它为 3D Gaussian Splatting 提供了身份正确、可验证的几何证据。实验要区分：

- source-only Track 几何约束是否真的改变了关联 Gaussian 的位置；
- Difix pseudo RGB 相比普通 matched continuation 是否带来可重复的差异；
- 轨迹身份是否比 hard-wrong、random、uniform 和 projection-only 控制更有用；
- 图像或身份指标的差异是否能在独立几何数据上成立。

## 2. 受控分支

首轮协议固定 LLFF Fern、3 个 source views、holdout 8、seed 1，并锁定相同的 A0 起点：

| 分支 | 含义 |
| --- | --- |
| A0 | source-track 修复基线，训练到 10k |
| A1 | 从完整 A0 checkpoint 普通 continuation 到 12k |
| SelfRender | 相同 continuation 预算，使用冻结的 A0 pseudo-view RGB |
| B | 相同 continuation 预算，使用冻结的 Difix pseudo-view RGB |

主比较是 A1 vs B。SelfRender 用于分离一般 pseudo-view supervision 与 Difix target replacement 的影响。A1、SelfRender、B 必须恢复同一个 A0 optimizer/RNG/checkpoint 状态，并使用相同 source-view schedule。

## 3. 预注册输出

每个 scene/seed 都必须保留：

- source-only Track H5、相机投影等价性和 leakage audit；
- geometry smoke、训练环境和 checkpoint provenance；
- pseudo manifest、cache hash、camera pool 和 pair audit；
- A1/B identity diagnostic 的 Correct、Hard-Wrong、Random、Uniform、ProjectionOnly 结果；
- 失败率、failure-aware error、PCK、identity margin 以及必要时的独立几何指标。

匹配失败保留在分母中，不能只汇报成功匹配。第一轮 Fern 是 development run，不能被写成 confirmatory 结论；跨场景聚合必须在所有场景/seed 的 audit 通过后进行。

## 4. 失效与停止规则

任何 P0 门禁失败都停止后续 GPU 计算。启用的 geometry constraint、geometry regularizer 或其验证发生初始化/forward/验证异常时，训练必须立即非零退出，并在输出目录写入 run_status.json，状态为 INVALID；不能把错误替换成零 loss 继续生成结果。

## 5. 当前结论边界

在真实 Fern、Difix、DINOv2、DTU 和跨场景实验完成前，本仓库不能声称：

- 真实场景重建已经改善；
- 改善来自 diffusion-specific 几何机制；
- 身份诊断已经证明因果 attribution；
- 相对于现有方法的性能领先性；
- 运行时间、显存或泛化优势已经成立。

这些是待实验检验的假设，不是实验设计中的预先结论。执行入口见 [docs/FERN_RUNBOOK.md](FERN_RUNBOOK.md)。
