# 变更与验证边界

## 基础来源

固定仓库：CPy255/GeoTrack-GS；commit：81ada6a32c918591ae7c7a0279dc6ca7a8018e2f。
基础输入：已上传的 GeoTrack_GS_Diffusion_Phase2_1_All_Files (1).zip 中累计 integrity patch 及工具。没有把分支名默认值、其他 commit 或全新项目当作此次基线。

## Research v2 的破坏性兼容变更

旧的不完整 checkpoint 不再作为严格控制实验起点。A0必须重新训练。新增保存格式 geotrack-research-v2，包括训练状态、随机状态和 `controlled_checkpoint_provenance_v2`；最终step以及最终metrics的时点统一为更新后。A0 provenance 锁定 canonical scene/Track/source 图像、校准后相机、seed、严格几何参数和共同训练协议；continuation 在训练前核对该对象。32/50协议保持不变；10k→12k首轮对照冻结拓扑并强制full precision；关闭可选旧regularizer、旧深度路径及GT-DCA，不加入新学习模块。

缓存新增 intrinsics fingerprint、模型revision与尺寸处理，因此旧manifest或旧cache不应混用。DINO从本地固定代码加载，实际权重sha需要记录。指标主入口是paired_evidence_report.py，历史summary中的条件均值只作补充，不直接排名。

## 已执行与未执行

历史记录：根目录 `validation/` 与 `code/validation/readme_update_*` 保留过往源码快照的检查结果，这些记录不是当前修复源码的验证证据。

本轮状态：2026-09-17 从固定 upstream commit `81ada6a32c918591ae7c7a0279dc6ca7a8018e2f` 全新生成累计补丁，没有复用旧补丁。补丁依次通过 `git apply --check`、`git apply --cached --check --whitespace=error-all` 与 `git diff --check`。补丁显式删除 `geometric_constraints/adaptive_weighting.py`，装配顺序固定为“pre-AST cumulative patch → source overlay → guarded AST integration”。

`bootstrap.sh` 在两个独立全新目录中真实执行并成功完成。装配前 64 个保留文件在生成树与独立应用树之间逐字节一致；装配后 66 个目标文件在两次独立装配之间逐字节一致；57 个覆盖层文件与当前包逐字节一致。装配后的 `git diff --check` 通过，unwanted adaptive-weighting 文件在两棵装配树中均不存在。

在全新装配树上，compileall、79/79 pytest、11/11 Phase-2.1 CPU smoke 和三相机 synthetic recovery 全部通过。证据位于 `code/validation/clean_assembly_*_2026-09-17.*`。这证明 clean reproducible assembly 与 CPU 完整性门禁，不证明 CUDA renderer、真实 Fern 重建、真实 Difix/DINOv2 推理或 DTU 几何结果。

`run_stage.sh integrity` 会重新核对当前 Track H5、projection context 全部输入哈希、live pseudo-camera、geometry smoke与三档真实恢复报告、32条pseudo manifest、A1/SelfRender/B配对审计、训练日志指标以及A1/B final checkpoint与Difix/DINO证据绑定。Pseudo manifest 绑定 A0 checkpoint 内嵌 provenance digest。pair-audit v5 强制覆盖每个 pseudo 方法的39/32 usage与严格 manifest 校验，并绑定完整共同训练参数、每个 source 图像的真实字节、校准后的 source training-camera payload，以及各方法精确 final-checkpoint 的路径、文件哈希、角色、render state、父 A0 lineage 和适用的 pseudo-manifest lineage；final checkpoint 本身进入 audited-file inventory。Held-out diagnostic v2 与 paired manifest/metadata/association v4 将 A1/B final provenance digest 传播到最终比较。这些对象或文件在 continuation 间不同、哈希失效、路径不规范，都会 fail closed。它只生成单个dataset/scene/seed的完整性结论并写入`aggregate_performed=false`；跨场景结果必须把A1/B、A1/SelfRender、SelfRender/B三种contrast的CSV分开汇集和聚合。

## 交付代码的完整性说明

代码包提供完整的本轮研究源码、测试、配置、累计补丁、装配器与实验工具；未重新打包上游未修改的大量源文件。bootstrap通过固定提交恢复完整上游并叠加本轮修改。用户需要能访问原仓库。模型权重、数据集、CUDA二进制和字体文件均不在包内。

这不是一个“只有接口、主要功能留pass”的框架，但也不是“包含所有第三方依赖、无需安装即可离线运行”的镜像。相应边界在README与教程中保持一致。

## 学术边界

所有真实性能数字为NR。未写入预期PSNR、虚构显存、假运行时间或模拟审稿人的保证。三份评审为同一助手的内部模拟，多视角不等于真实独立评审。论文匿名工作版不等于完整研究包可直接匿名提交；提交前须剥离个人路径与身份，同时遵守原始代码和模型许可。
