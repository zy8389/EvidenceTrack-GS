# Geometry constraints

本目录包含两条不同用途的几何路径：

## Strict source-only protocol

正式的 EvidenceTrack-GS 受控实验使用 repaired_geometry.py 中的 StrictGeometryManager，配合 strict_track_store.py 的 source-only Track H5。它要求 source image membership、相机集合、Gaussian association、梯度 coverage 和 checkpoint provenance 都通过门禁。

典型入口：

    python train.py \
      --strict_tracks \
      --strict_source_only_geometry \
      --track_path /path/to/tracks_source_only.h5 \
      -s /path/to/scene \
      -m /path/to/run

真实实验不要直接在当前源码树上重新应用 patch，也不要把 held-out correspondence 混入 source-only Track。

## Legacy compatibility path

其余模块（constraint_engine.py、trajectory_manager.py、reprojection_validator.py 等）保留给旧入口和兼容性检查。它们不是当前 A0/A1/SelfRender/B protocol 的默认实现，也不能被描述成已证明的性能改进。

当前仓库没有 adaptive_weighting.py；文档或脚本不应再引用这个已明确排除的文件。

## Failure semantics

只要显式启用 geometry constraint 或 geometry regularizer，初始化、forward 或验证异常都会立即失败。训练不会把异常转换成零 loss；输出目录会写入 run_status.json，状态为 INVALID。只有通过所有必要门禁的 run 才能进入后续 pair audit 和结果汇总。
