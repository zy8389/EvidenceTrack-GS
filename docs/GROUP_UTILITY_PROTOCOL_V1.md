# Group Utility Protocol v1（冻结记录）

状态：**FROZEN / 2026-09-25**。本文件是 `Pseudo-Supervision_Group_Utility_Protocol_v1.md` 的仓库内冻结记录；旧五组实验目录、旧 summary、旧 archive 均为只读 discovery evidence，不得覆盖。

## Pilot estimand

Flower seed1 是第一个 pilot。冻结的 32 个 pseudo cameras 只做一个预注册的 `4 × 8` partition。每组使用同一段 continuation：从同一个 A0 `chkpnt10000.pth` 分叉，继续 2,000 updates；source camera schedule、optimizer 恢复、硬件、评估程序和 held-out 集合保持一致。

每个 group 建立三个独立 run：

- `A1`：source-only continuation；
- `S_G`：SelfRender target；
- `B_G`：同一 group 的 Difix target。

Group 的 39 个 pseudo calls 固定在 iteration `10050..11950`，间隔 50，按组内 8 个 camera key round-robin；因此每个 view 暴露 4 或 5 次。只允许改变冻结 target，不得重采样 camera、改变预算或复用旧 32-view PASS。

## Utility and leakage boundary

Primary metric 为 PSNR；SSIM、LPIPS 为 secondary。对每个 group：

```text
U_self(G)  = Q(S_G) - Q(A1)
U_difix(G) = Q(B_G) - Q(S_G)   # 主要预测标签
U_total(G) = Q(B_G) - Q(A1)
```

所有 signal 必须在 `S_G/B_G` continuation 前由 A0、source-only 信息和冻结 pseudo input/target cache 计算并保存。不得使用 held-out GT、干预后 checkpoint、最终指标或事后挑阈值。Group utility 不可直接相加为联合 top-k utility；联合 top-k 必须另训并与同 K、同调用及有效权重预算的 random joint subset 比较。

## Go / No-Go

只有在至少两个不同 discovery scene 的留出诊断中，预冻结方向的 `U_difix` 排序优于重复 random/permutation，不依赖单个 group/partition，且真实联合 top-k 在同预算下优于 random joint subset，才允许实现 `Ours`。否则停止 controller 开发，只报告受控分析、配对区间和失败边界，不更换 primary metric 或改写 estimand。

## Required artifacts

每个新 group run 必须保存母 manifest/hash、partition/hash、group manifest/hash、39-call trace/hash、A0 checkpoint/hash、A0 render/input 与 target/sidecar hash、signal provenance、pair audit、metrics、成本和失败日志。任何 hash 不符、held-out 泄漏、non-finite loss、门禁失败或旧 audit 复用都标记为 `INVALID`，并用新 run ID 修复。
