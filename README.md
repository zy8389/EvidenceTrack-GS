# EvidenceTrack-GS

[English](README_en.md)

**面向扩散辅助稀疏视角 Gaussian Splatting 的轨迹身份控制与几何证据诊断。**

这是可直接开发的完整源码仓库。GeoTrack-GS upstream、累计研究补丁、源码覆盖层和受保护 AST 集成已经展开到仓库根目录；后续修改、测试、提交和推送都直接在这里进行，不再需要先运行 `bootstrap.sh`。

## 当前状态

| 模块 | 状态 |
| --- | --- |
| 完整源码装配 | 通过：两次独立装配，66 个目标文件 0 字节差异 |
| Patch 门禁 | 通过：apply、cached whitespace、diff check |
| Python compileall | 通过 |
| Phase-2.1 CPU smoke | 11 / 11 通过 |
| Regression tests | 79 / 79 通过 |
| 三相机合成恢复 | 通过 |
| Fern CUDA preflight | 未运行 |
| A0 / A1 / SelfRender / B | 未运行 |
| 真实 Difix / DINOv2 / DTU | 未运行 |

验证报告和原始日志位于 [`validation/`](validation/)。当前结果证明完整源码可重建并通过 CPU 完整性门禁，不代表真实 GPU 重建或论文实验已经完成。

## 开发入口

克隆完整仓库：

```bash
git clone https://github.com/zy8389/EvidenceTrack-GS.git
cd EvidenceTrack-GS
```

安装 CPU 检查依赖并运行门禁：

```bash
pip install -r requirements-cpu.txt
export PYTHONPATH="$PWD"
python -m compileall -q diffusion_guidance geometric_constraints tools tests arguments scene gaussian_renderer train.py utils
python -m pytest -q tests
python tools/smoke_test_phase2_1.py
python tools/test_geometry_recovery.py --synthetic-smoke --output synthetic_recovery.json
```

Windows PowerShell 使用：

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m pytest -q tests
python tools/smoke_test_phase2_1.py
```

真实实验从 [`research_docs/FERN_PRECHECK_AND_RUN.md`](research_docs/FERN_PRECHECK_AND_RUN.md) 开始，并使用 `research_scripts/run_stage.sh`。任何 P0 门禁失败都应停止后续 GPU 计算。

## 研究结构

```text
Source RGB + supplied cameras
          |
          v
Source-only feature tracks
          |
          v
       A0 @ 10k
      /    |     \
 A1 @12k  SelfRender @12k  B @12k
 source     A0 pseudo        Difix pseudo
 only       targets           targets
      \      |      /
       controlled comparison
               |
               v
      identity-controlled diagnostic
               |
               v
 Correct / Hard-Wrong / Random / Uniform / ProjectionOnly
```

主比较是 `A1 vs B`。`SelfRender` 控制额外 pseudo-view supervision 本身的影响。匹配失败进入 failure-aware denominator，不会被选择性删除。

## 目录

```text
arguments/                 参数定义
diffusion_guidance/        扩散证据、身份协议与 checkpoint 绑定
geometric_constraints/     严格 Track 几何约束
scene/                     场景与 Gaussian 状态
tools/                     构建、审计、评估和恢复工具
tests/                     CPU 回归测试
configs/                   受控实验配置
research_scripts/          CPU 与真实实验执行脚本
research_docs/             研究说明和 Fern runbook
research_provenance/       累计补丁与来源记录
validation/                当前完整源码验证证据
```

`SHA256SUMS.txt` 记录除清单自身以外的全部跟踪文件字节，可在 POSIX shell 中使用 `sha256sum -c SHA256SUMS.txt` 校验。

## 来源与边界

本仓库基于 `CPy255/GeoTrack-GS` commit `81ada6a32c918591ae7c7a0279dc6ca7a8018e2f`。原 upstream README 保存在 [`README_UPSTREAM.md`](README_UPSTREAM.md)，完整来源记录见 [`research_provenance/README.md`](research_provenance/README.md)。旧的可重建包布局保存在 Git 标签 `reproducible-package-2026-09-17`。

目前不能声称真实 Fern 重建提升、diffusion-specific 几何改善、真实 Difix 可复现、DINOv2 identity attribution、DTU 几何改善或 SOTA。数据集、模型权重、大型 cache 和 checkpoint 不随仓库分发。
