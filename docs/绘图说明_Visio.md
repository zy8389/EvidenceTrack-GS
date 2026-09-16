# 插图草图与 Visio 重绘说明

paper/figures 包含两个与论文正文对应的 TikZ 矢量源文件，以及三个可导入 Visio 的 SVG 草图。它们都是方法结构示意，不是真实实验可视化，没有伪装成已跑出的渲染、重建点云或定量曲线。

## 图1：训练与评估边界

从左到右：Source RGB＋给定相机 → source-source Track identity → anchors／quality → A0 10k。A0右边分为A1、SelfRender、B；A0下方产生32个pseudo camera、冻结A0 render target及冻结Difix cache。A0 render target只连到SelfRender，Difix cache只连到B。图下方为held-out RGB和Projection／GS／Difix／Real reference诊断。训练与评估之间画虚线隔离。禁止画从held-out诊断指回训练的闭环箭头，因为当前未实现。

## 图2：身份诊断与locality反例

画一个局部搜索窗、投影中心十字、proxy标签实心点；右边三个query：正确身份、邻近相似错误身份、random／uniform。三者都指向同一个候选窗口。注释强调“准确不等于使用身份”；下方列出gain over ProjectionOnly、gain over GS、identity margin、failure rate。

## 图3：受控实验时间线

上方共同A0从0到10k；10k处分叉为A1、SelfRender与B，终点均12k。A1为source／track only；SelfRender使用冻结A0 render pseudo target；B使用冻结Difix(A0 render) target。两个pseudo分支都标记10050…11950，每50一步，共39次调用，覆盖相同32个unique views。图右侧列出SelfRender−A1、B−SelfRender、B−A1三种对照，并标注必须检查相同checkpoint hash、source sequence、pseudo camera schedule以及update后保存的结果。

最终投稿图可以保留此结构，换成统一黑白或两色视觉。真实渲染对比图必须等实验完成后从真实输出中选取，不能用生图替代。误差曲线必须从CSV/JSON生成，不能先画理想走势再补数字。
