# 三视角模拟审稿与逐条回复

性质说明：以下为同一助手从三个不同专业视角进行的内部模拟评审，不是真实外部审稿人、不是三位独立专家，也不是 CVPR 官方评审。v1 是这一轮形成的初稿快照，v2 是根据下列意见修订的版本。没有用虚构的肯定意见为论文背书。

## Reviewer 1：新颖性与研究定位

总体意见：当前不支持以“新的扩散重建 SOTA 方法”直接投稿，但问题设置可以发展为严谨的分析研究。修复协议很重要，不能把工程 bug 修复全部当作论文方法贡献。

R1-1：Difix、HAD、GIFSplat、Track4DGen 已覆盖生成先验、可靠性过滤或 tracker prior。仅仅“Diffusion＋Track”与已有工作差异不够。回复：相关工作已分别说明设置差异与重叠，不再声称这些组成部分首次出现。核心问题收敛为正确轨迹身份是否在控制 locality 后仍有几何价值。状态：叙事已改，创新性仍待实证。

R1-2：证据诊断没有进入训练 loss，为何用“evidence-driven reconstruction”描述？回复：删除闭环暗示，图中用虚线明确 evaluation-only，标题使用 diagnostics。现阶段不主张身份匹配是 B 收益的已证因果机制。状态：已修订。

R1-3：静态与动态任务不同，不足以证明超过 Track4DGen。回复：正文不把“静态 vs 动态”作为主要创新论据，也不在不同任务间比较绝对分数。状态：已修订。

R1-4：旧工作的继承部分是否存在重复贡献或自引遗漏？回复：内部文档明确固定 upstream commit 与 Phase2.1 来源；投稿前须核对作者已发表论文并第三人称引用、说明新增内容。由于当前未拿到经核对的最终出版条目，不编造作者论文的刊名、页码或 DOI。状态：部分解决，出版信息与重合度核对仍待作者补证。

R1-5：没有真实对照结果，不能评价贡献大小。回复：所有真实指标记为 NR，增加预注册假设、停止条件和失败解释。状态：未解决，必须运行。

复审结论：标题、边界、相关工作和因果叙事明显更可信；是否具有会议级新颖性仍未由当前材料证明。

## Reviewer 2：多视角几何与理论

总体意见：局部几何推导基本可以自洽，但必须严格限定条件。最需要防范的是把局部三角化信息矩阵解释为整个 Gaussian scene 的后验，把给定位姿下的 source-only 写成三图端到端 SfM。

R2-1：quality 中的 sqrt(trace(covariance)) 有长度单位。回复：使用 source camera median baseline 归一化，并将协方差计算本身改为相对阻尼。增加五尺度测试。状态：历史源码快照曾完成 CPU 局部计算验证；当前修复源码与全训练尺度等价性均待验证。

R2-2：所谓“几何恢复保证”没有处理近共线相机、错误匹配及联合非凸目标。回复：给出 Jacobian 的射线零空间分析，只证明局部 Gauss–Newton 信息在非退化条件下正定；明确剩余 Hessian 项、错误对应和关联切换的影响。状态：理论已收窄。

R2-3：源图像缩放与 renderer 投影是否使用同一坐标？回复：使用 half-pixel 变换，明确 principal point 和 CUDA ndc2Pix 的代数关系。新增 patch-center 与 off-center calibration 测试。状态：代数关系已写明，历史源码快照的 CPU 测试曾通过；当前修复源码与真实 CUDA parity 均待跑。

R2-4：源轨迹 identity 是否意味着永久 Gaussian identity？回复：不是。正文明确 piecewise association、重建拓扑后重新关联、many-to-one 碰撞以及有效约束变量数。状态：已改。

R2-5：如何证明生成视图是新增几何观测？回复：不再这样声称。增加条件互信息推论和重复噪声反例，承认生成器提供先验而非新传感器。诊断中的 evidence 指可用性，不是独立似然。状态：已改。

R2-6：SIFT 附加的 held-out 点本身未必是真值。回复：统一称 proxy correspondences，安排人工核验、低纹理/遮挡分层和 DTU 几何。状态：标签性质说明已改，独立验证待跑。

复审结论：支持这些局部命题的表述方式，不支持“全局恢复、校准后验、消除幻觉”等强命题。独立几何数据仍是关键缺口。

## Reviewer 3：实验设计、统计与复现

总体意见：初稿最容易产生偏差的地方不是模型本身，而是样本选择、训练量和坐标约定。必须让审稿人可以从日志复核真正发生了哪些更新。

R3-1：只比较 B12k 与 A0 10k，无法区分扩散、多训练和通用 pseudo-view 监督。回复：强制 A1、SelfRender、B 共享完整 A0 checkpoint、相同 2k source 更新和同一 source 序列；v5 pair audit 还要求完整共同训练参数、source 图像字节清单和校准后训练相机清单逐项一致，并强制验证每个 pseudo 方法的39/32 usage和严格 manifest。它重新读取每个精确 12k checkpoint，核对方法角色、render state、父 A0 lineage、适用的 pseudo-manifest lineage和文件 SHA256。SelfRender/B 共享 pseudo camera、39 次调用、权重与 loss，并都使用 strict frozen-cache 校验。SelfRender−A1估计通用 pseudo-view 监督，B−SelfRender估计 Difix target 替换，B−A1仅表示总分支效应，并分别提供受控 pair audit。状态：代码路径已实现但本轮未运行，服务器记录待产出。

R3-2：原 checkpoint 不恢复全部状态，保存时点还在 optimizer.step 前，最后一步被跳过。回复：引入完整 v2 state，保存与评测移动到更新之后，拒绝旧 tuple；A0 checkpoint 内嵌输入和训练协议 provenance，continuation 在训练前验证，12k checkpoint 再内嵌精确父 A0 与 pseudo lineage；首轮冻结 continuation 拓扑并关闭 AMP。状态：当前 fixture 已按严格协议更新但依用户要求未运行；CUDA 同步恢复待跑。

R3-3：失败预测从误差均值中消失，会奖励更容易失败的方法。回复：PCK 使用全部 eligible 分母；主结果使用同一 paired set、共享 image-diagonal penalty，同时公开 conditional error 和 failure rate。缺失方法、重复结果或 NR 值均触发错误，不被静默丢弃。状态：代码与回归测试已更新；当前修复源码测试依用户要求未运行。

R3-4：Real Target 不是 upper bound。回复：统一改为 reference diagnostic。真实图像遮挡、纹理弱、特征域差异都可能让它更差。状态：已改。

R3-5：GT-inside-window 过滤等于使用 oracle 选择样本。回复：保留已有实现，但降为明确标记的 secondary oracle-conditioned test；非过滤鲁棒性实验单独列入待实验项。状态：说明已改，主鲁棒性扩展待跑。

R3-6：32 个 pseudo cameras 不代表实际都被监督。回复：严格区间 10000<t<12000、interval50，对应39次调用；round-robin 覆盖32个相机，实际usage日志必须一致。状态：历史源码快照的调度 CPU 测试曾通过；当前修复源码与实际日志均待跑。

R3-7：用很多 tracks 进行显著性检验有伪重复问题。回复：先 track→target image→scene，先平均同一 scene 的 seed contrast，再 bootstrap scenes；不把大量同图点当独立样本。开发场景与验证场景分离。状态：实现与预注册已改，样本量依然有限。

R3-8：源码/权重/图像大小不固定，缓存可能沿用旧文件。回复：固定 upstream，Difix 要求模型 revision，DINO 使用本地固定代码；旧缓存跳过功能关闭；记录图像哈希与full-frame resize策略。权重文件哈希和真实环境锁仍需服务器产出。状态：部分解决。

复审结论：v2 的实验可审计性显著提高；历史快照的34个 CPU 检查不能验证当前修复源码，也不能支持真实性能结论。真实模型与硬件结果尚不存在。

## 修改汇总与仍未解决的问题

已落地修改包括：题目和贡献范围收敛；理论假设明确；source-only 条件说明；径向 Huber 与实现一致；half-pixel 与 principal point；失败计分；共同配对；hard-negative validity；Real reference 命名；完整 checkpoint；更新后评测保存；真实视角序列日志；模型缓存版本；场景级统计。

仍未解决的不是文字润色能够替代的事项：真实 A1/SelfRender/B 三项对照（尤其 B−SelfRender 的 Difix-target 替换证据）、非生成增强对照、最终 A1/B GS identity diagnostic、实际 Difix 稳定性、独立几何、速度显存、完整上游装配及 CUDA 兼容性、与已发表工作的最终重合度核对。三份模拟复审都没有给出“可以直接投稿／保证录用”的结论。
