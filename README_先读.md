# GeoTrack-GS × Diffusion Research v2

当前研究定义以 `paper/main_v2.tex`、现行 Markdown、JSON/CSV 协议和 `code/` 源码为准。先阅读 `paper/main_v2.tex` 和 `docs/方案复审与理论说明.md`；开始实验看 `docs/快速开始.md`，查看审查与修订看 `docs/三视角模拟审稿与逐条回复.md`。

`paper/main_v2.pdf`、`docs/研究说明与实验教程.docx` 和 `paper/v1_to_v2.diff` 均为 2026-09-06 生成的历史快照，早于当前 P0/P1/P2 修复。它们可用于回看旧版，但不能代表当前正文、当前运行协议或当前源码，也不得据此执行实验。依用户要求，本轮不重新渲染这些二进制/派生产物。

**当前状态：预实验研究初稿＋修订源码，不是真实数据实验已完成的投稿成稿。**

`validation/` 中的11项CPU smoke、23项pytest和单点合成恢复是修复前快照的历史记录。本轮P0/P1/P2修复后的测试依用户要求尚未运行，不能用旧日志证明当前源码通过。真实PSNR/SSIM/LPIPS、DTU几何、Difix/DINOv2推理、显存和时间也没有运行，均标记为 `NOT VERIFIED`，不得编造。

code提供本轮完整研究源码、累计patch、固定commit装配脚本。未包含上游原仓库未修改文件、数据集、模型权重和CUDA二进制。完整工程用code/scripts/bootstrap.sh恢复；请使用全新目录，旧A0 checkpoint须重跑。

paper内为匿名英文双栏扩展初稿、完整v1文本快照、历史v2阅读PDF、BibTeX和矢量草图。当前 `main_v2.tex` 是正文权威源；`draft_v1.tex` 保持为历史初稿，不随本轮修复改写。当前不是CVPR2027官方模板认证版本。内部研究包包含实名仓库路径，不可直接整体当匿名补充材料提交。

experiments内为完整待实验登记、运行矩阵和实测输入表。JSON/CSV主矩阵包含A0、A1、SelfRender、B四分支，共96条：fern/room的24条为development，其余六场景的72条为confirmatory。Excel工作簿保留此前72条矩阵的历史快照，当前执行以JSON/CSV为准。状态和结果要由真实日志驱动。模拟评审是同一助手的三个专业视角，不是外部专家背书。

## 建议打开顺序

1. `paper/main_v2.tex`：当前英文修订初稿权威源，含理论、方法、实验方案、附录与参考文献；旧 PDF 仅供查看 2026-09-06 版式。
2. `docs/方案复审与理论说明.md` 与 `docs/FERN_PRECHECK_AND_RUN.md`：当前中文逻辑说明和首轮可执行协议；旧 DOCX 仅为 2026-09-06 快照。
3. `experiments/llff_primary_run_matrix.json`：96条A0/A1/SelfRender/B运行条目；Excel工作簿仅作历史快照。
4. `code/README.md`：本轮源码与完整工程装配入口。

第一组Fern实验严格按 `docs/FERN_PRECHECK_AND_RUN.md` 执行。所有门禁JSON必须是精确布尔值 `passed=true`；旧报告输入指纹不一致时视为陈旧并停止。`validation/verification_summary.json` 只汇总其对应源码快照的历史执行结果；最终打包后再重建 `SHA256SUMS.txt`。英文v1完整源码保留用于对照，但不是已完成实验的旧投稿稿。

训练日志的规范路径是各分支目录内的 `train.log`。每个 scene/seed 使用 `run_stage.sh metrics` 生成三份两臂对照 CSV，再用 `run_stage.sh integrity` 生成单次运行的完整性报告；该报告不执行跨场景聚合。A1/B、A1/SelfRender、SelfRender/B 必须在所有目标 scene/seed 完成后分别聚合。
