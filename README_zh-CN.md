# EvidenceTrack-GS

**面向扩散辅助稀疏视角 Gaussian Splatting 的轨迹身份控制与几何证据诊断。**

> **核心问题：** Diffusion 让 novel view 看起来更好之后，这种提升是否真的包含与正确 source-track identity 一致的几何收益，还是仅仅来自外观增强、局部投影先验、额外训练或选择性匹配成功？

[English README](README.md)

## 当前状态

本仓库目前处于 **pre-experiment research implementation** 阶段。受控实验协议与 CPU 侧完整性验证已经完成，但决定论文结论的真实场景 GPU 实验尚未执行。

| 模块 | 状态 |
| --- | --- |
| Phase-2.1 CPU smoke | ✅ 11 / 11 |
| Regression tests | ✅ 79 / 79 |
| 三相机合成恢复 | ✅ 通过 |
| Fern CUDA preflight | ⏳ 未运行 |
| A0 / A1 / SelfRender / B | ⏳ 未运行 |
| 真实 Difix 推理 | ⏳ 未运行 |
| DINOv2 identity diagnostic | ⏳ 未运行 |
| DTU 独立几何评估 | ⏳ 未运行 |

当前修复版对应的 CPU 验证日志位于 `code/validation/repair_*`。根目录 `validation/` 中的文件保留为历史快照，不应单独用于证明当前源码已经通过验证。

## 研究动机

稀疏视角 3D Gaussian Splatting 本身具有较强欠约束性。Diffusion 可以生成视觉上更自然、更完整的图像，但“像素更真实”并不等于“三维几何更正确”。

因此，本项目不直接把视觉提升解释为几何提升，而是设计严格的受控实验，区分：

- Appearance improvement
- Extra optimization / pseudo supervision
- Local projection prior
- Identity-specific geometric utility

## 核心实验结构

```text
Source RGB + supplied cameras
          ↓
Source-only feature tracks
          ↓
       A0 @ 10k
      /    |     \
     /     |      \
 A1 @12k  SelfRender @12k  B @12k
 source     A0 pseudo        Difix pseudo
 only       targets           targets
      \     |      /
       \    |     /
      controlled comparison
              ↓
       A1 / B final renders
              ↓
 identity-controlled diagnostic
              ↓
Correct / Hard-Wrong / Random
      / Uniform / ProjectionOnly
              ↓
failure-aware paired geometry evidence
```

最重要的主比较是：

```text
A1 vs B
```

而不是：

```text
A0 vs B
```

因为 A1 与 B 必须从同一个完整 A0 checkpoint 出发，保持相同训练步数、相同 source-view sampling，并通过 pair audit 验证。`SelfRender` 用于控制“额外 pseudo-view supervision 本身”的影响。

## Identity-Controlled Diagnostic

对于同一个 target local window，比较：

- Correct identity
- Local hard-wrong identity
- Random query
- Uniform query
- ProjectionOnly

匹配失败不会被悄悄删除，而是进入 failure-aware denominator。主报告包括：

- Failure rate
- PCK
- Failure-aware capped error
- Geometry-relative gain
- Identity margin
- Hard-negative validity

这一 diagnostic 只用于 evaluation，不会把 held-out RGB 或 held-out correspondence 反馈进训练。因此即使结果为正，也只能说明与 hypothesized mechanism 一致，不能单独证明因果机制。

## 当前 CPU 验证

```bash
cd code
pip install -r requirements-cpu.txt
bash scripts/run_cpu.sh
```

当前 repaired snapshot 已实际通过：

```text
Phase-2.1 smoke:        11 / 11
Pytest regression:      79 / 79
Synthetic recovery:     PASS
```

三相机合成恢复结果：

```text
Mean reprojection error: 9.7511 px → 0.0276 px
3D anchor distance:      0.053852  → 0.000226
```

这些结果只证明 CPU fixture、provenance guard 和局部几何机制按预期工作，**不能代替真实 Fern、CUDA rasterizer、Difix、DINOv2 或 DTU 实验。**

## 第一组真实实验

第一组实验固定为：

```text
LLFF / fern / 3 source views / seed 1
```

严格按照：

```text
docs/FERN_PRECHECK_AND_RUN.md
```

执行：

```text
Environment
→ Track build & audit
→ Camera projection equivalence
→ Live pseudo-camera audit
→ CUDA geometry smoke
→ A0 @ 10k
→ A1 / SelfRender / B
→ Pair audit
→ Reconstruction metrics
→ A1/B identity diagnostic
```

任何 P0 gate 失败都应立即停止，不继续烧后续 GPU 计算。

## 当前不能声称的结论

现在还不能声称：

- B 已经优于 A1；
- Diffusion 已经改善真实三维几何；
- identity margin 已在真实 LLFF 上成立；
- Difix 在真实 GPU 环境中完全可复现；
- DTU geometry 已改善；
- 当前方法达到 SOTA。

这篇工作的价值在于：**让这些结论变成可以被严格证伪和验证的问题，而不是先假定它们成立。**

## 目录

```text
code/          研究源码、协议、工具、测试与脚本
experiments/   预注册实验矩阵与实测结果模板
docs/          实验说明与 Fern server runbook
validation/    历史 CPU 验证快照
audit/         内部 provenance / audit 材料
```

执行与装配说明见 `code/README.md`。
