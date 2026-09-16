# 高级几何约束和轨迹跟踪模块

## 概述

高级几何约束和轨迹跟踪模块是 GeoTrack-GS 的核心组件，提供了自适应重投影约束、多尺度几何一致性和智能轨迹质量评估功能，显著提升 3D 场景重建的质量和稳定性。

## 主要特性

- **自适应重投影约束**：根据场景复杂度和特征质量自动调整约束权重
- **多尺度几何一致性**：在多个分辨率尺度上保持几何约束的一致性
- **智能轨迹质量评估**：自动评估和管理特征轨迹质量
- **动态权重调度**：训练过程中动态调整几何约束权重
- **实时几何验证**：实时验证几何约束的有效性

## 快速开始

### 基本使用

```python
from geometric_constraints import ConstraintEngine, TrajectoryManager, AdaptiveWeighting

# 初始化组件
trajectory_manager = TrajectoryManager()
constraint_engine = ConstraintEngine()
adaptive_weighting = AdaptiveWeighting()

# 加载轨迹数据
trajectories = trajectory_manager.load_trajectories("path/to/tracks.h5")

# 计算几何约束
constraint_result = constraint_engine.compute_constraints(
    trajectories=trajectories,
    cameras=cameras,
    gaussians=gaussians
)

# 获取约束损失
loss = constraint_result.loss_value
```

### 集成到训练循环

```python
# 在 train.py 中的使用示例
from geometric_constraints import GeometricConstraintSystem

# 初始化约束系统
constraint_system = GeometricConstraintSystem(config)

# 在训练循环中
for iteration in range(max_iterations):
    # ... 现有的训练代码 ...
    
    # 计算几何约束损失
    constraint_loss = constraint_system.compute_loss(
        gaussians=gaussians,
        cameras=cameras,
        iteration=iteration
    )
    
    # 添加到总损失中
    total_loss = rgb_loss + constraint_loss
    
    # ... 反向传播和优化 ...
```

## 模块组件

### TrajectoryManager（轨迹管理器）

负责管理特征轨迹的生命周期，包括加载、预处理、质量评估和动态更新。

**主要方法**：
- `load_trajectories(track_file)`: 从文件加载轨迹数据
- `preprocess_trajectories(trajectories)`: 预处理轨迹数据
- `get_active_trajectories(min_quality)`: 获取高质量轨迹
- `update_trajectory_quality(trajectory, quality_score)`: 更新轨迹质量

### ConstraintEngine（约束引擎）

核心约束计算引擎，实现多种几何约束的计算和组合。

**支持的约束类型**：
- 重投影约束 (Reprojection Constraints)
- 多尺度约束 (Multi-scale Constraints)
- 时序一致性约束 (Temporal Consistency)
- 极线约束 (Epipolar Constraints)

### AdaptiveWeighting（自适应权重）

根据场景特性和训练状态动态调整约束权重。

**权重调整策略**：
- 纹理感知权重：基于图像纹理复杂度
- 置信度驱动权重：基于重投影误差和特征置信度
- 训练阶段权重：基于训练迭代次数的动态调度

### MultiScaleConstraints（多尺度约束）

在多个分辨率尺度上计算和验证几何约束。

**默认尺度配置**：
- 原始分辨率：权重 0.5
- 1/2 分辨率：权重 0.3
- 1/4 分辨率：权重 0.2

### ReprojectionValidator（重投影验证器）

实时验证几何约束的有效性，提供质量监控和报告。

**验证功能**：
- 约束满足度计算
- 几何质量指标统计
- 异常值检测和处理
- 质量报告生成

## 配置参数

### 基本配置

```python
# 约束系统配置
constraint_config = {
    # 轨迹质量阈值
    "min_trajectory_quality": 0.4,
    "min_trajectory_length": 3,
    "max_reprojection_error": 2.0,
    
    # 权重配置
    "base_constraint_weight": 0.1,
    "texture_weight_range": [0.3, 2.0],
    "confidence_weight_decay": 2.0,
    
    # 多尺度配置
    "scales": [1.0, 0.5, 0.25],
    "scale_weights": [0.5, 0.3, 0.2],
    
    # 训练调度
    "warmup_iterations": 1000,
    "full_weight_iterations": 5000,
    "weight_schedule": "linear"
}
```

### 高级配置

```python
# 高级约束配置
advanced_config = {
    # 异常值检测
    "outlier_detection": {
        "method": "statistical",
        "threshold_multiplier": 2.5,
        "min_inlier_ratio": 0.7
    },
    
    # 轨迹分割
    "trajectory_splitting": {
        "enable": True,
        "drift_threshold": 10.0,
        "ransac_threshold": 1.0,
        "min_segment_length": 3
    },
    
    # 性能优化
    "performance": {
        "batch_size": 1024,
        "use_gpu": True,
        "memory_efficient": True,
        "parallel_processing": True
    }
}
```

## 使用示例

### 示例 1：基本几何约束

```python
import torch
from geometric_constraints import ConstraintEngine, TrajectoryManager

# 初始化组件
trajectory_manager = TrajectoryManager()
constraint_engine = ConstraintEngine()

# 加载数据
trajectories = trajectory_manager.load_trajectories("data/tracks.h5")
cameras = load_cameras("data/cameras.json")
gaussians = load_gaussians("data/gaussians.ply")

# 预处理轨迹
trajectories = trajectory_manager.preprocess_trajectories(trajectories)

# 计算约束
result = constraint_engine.compute_constraints(
    trajectories=trajectories,
    cameras=cameras,
    gaussians=gaussians
)

print(f"约束损失: {result.loss_value.item():.6f}")
print(f"异常值比例: {result.outlier_mask.float().mean().item():.3f}")
```

### 示例 2：自适应权重调整

```python
from geometric_constraints import AdaptiveWeighting

# 初始化自适应权重系统
weighting = AdaptiveWeighting()

# 配置权重策略
weighting.configure({
    "texture_sensitivity": 1.5,
    "confidence_decay": 2.0,
    "temporal_schedule": "cosine"
})

# 在训练循环中使用
for iteration in range(max_iterations):
    # 计算当前权重
    weights = weighting.compute_weights(
        images=current_images,
        reprojection_errors=errors,
        iteration=iteration
    )
    
    # 应用权重到约束计算
    constraint_loss = constraint_engine.compute_weighted_loss(
        constraints=constraints,
        weights=weights
    )
```

### 示例 3：多尺度约束处理

```python
from geometric_constraints import MultiScaleConstraints

# 初始化多尺度约束
multiscale = MultiScaleConstraints(
    scales=[1.0, 0.5, 0.25],
    weights=[0.5, 0.3, 0.2]
)

# 计算多尺度约束
multiscale_loss = multiscale.compute_constraints(
    trajectories=trajectories,
    cameras=cameras,
    gaussians=gaussians
)

# 获取各尺度的贡献
scale_contributions = multiscale.get_scale_contributions()
for scale, contribution in scale_contributions.items():
    print(f"尺度 {scale}: 贡献 {contribution:.4f}")
```

### 示例 4：实时质量监控

```python
from geometric_constraints import ReprojectionValidator

# 初始化验证器
validator = ReprojectionValidator()

# 在训练过程中进行验证
if iteration % 100 == 0:
    # 计算几何质量指标
    quality_metrics = validator.validate_geometry(
        trajectories=trajectories,
        cameras=cameras,
        gaussians=gaussians
    )
    
    print(f"约束满足度: {quality_metrics['satisfaction_rate']:.3f}")
    print(f"平均重投影误差: {quality_metrics['mean_reprojection_error']:.4f}")
    
    # 生成质量报告
    if quality_metrics['satisfaction_rate'] < 0.85:
        report = validator.generate_quality_report()
        print("警告：几何约束满足度较低")
        print(report)
```

## 性能优化建议

### 1. 内存优化

```python
# 使用内存高效模式
constraint_engine.configure({
    "memory_efficient": True,
    "batch_size": 512,  # 根据GPU内存调整
    "gradient_checkpointing": True
})
```

### 2. 计算优化

```python
# 启用并行处理
constraint_engine.configure({
    "parallel_processing": True,
    "num_workers": 4,
    "use_mixed_precision": True
})
```

### 3. 轨迹过滤

```python
# 提前过滤低质量轨迹
trajectory_manager.configure({
    "prefilter_trajectories": True,
    "min_quality_threshold": 0.5,
    "max_trajectories": 10000
})
```

## 故障排除

### 常见问题

1. **内存不足错误**
   - 减少批处理大小
   - 启用内存高效模式
   - 减少同时处理的轨迹数量

2. **收敛速度慢**
   - 调整权重调度策略
   - 检查轨迹质量阈值
   - 优化学习率设置

3. **几何约束失效**
   - 检查相机参数准确性
   - 验证轨迹数据质量
   - 调整异常值检测阈值

### 调试工具

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 使用调试模式
constraint_engine.set_debug_mode(True)

# 保存中间结果
constraint_engine.save_debug_info("debug_output/")
```

## 更多资源

- [API 参考文档](docs/api_reference.md)
- [配置参数详解](docs/configuration.md)
- [性能基准测试](docs/benchmarks.md)
- [常见问题解答](docs/faq.md)