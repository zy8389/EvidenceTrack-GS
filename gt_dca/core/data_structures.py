"""
GT-DCA Data Structures
核心数据结构

Defines the core data structures used throughout the GT-DCA system.
These structures provide type safety and clear data contracts between components.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
import torch
from torch import Tensor


@dataclass
class TrackPoint:
    """
    轨迹点数据结构
    
    Represents a 2D trajectory point extracted from the GeoTrack-GS framework.
    Contains coordinate information, confidence scores, and optional feature descriptors.
    
    Requirements addressed: 1.2 - 2D trajectory points as geometric guidance
    """
    point_id: int                                    # 轨迹点唯一标识符
    coordinates_2d: Tuple[float, float]              # 2D坐标 (x, y)
    confidence: float                                # 置信度分数 [0, 1]
    frame_id: int                                    # 帧ID
    feature_descriptor: Optional[Tensor] = None      # 可选的特征描述符
    
    def __post_init__(self):
        """验证数据完整性"""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"置信度必须在[0,1]范围内，得到: {self.confidence}")
        
        if len(self.coordinates_2d) != 2:
            raise ValueError(f"2D坐标必须是长度为2的元组，得到: {self.coordinates_2d}")
    
    @property
    def x(self) -> float:
        """返回x坐标"""
        return self.coordinates_2d[0]
    
    @property
    def y(self) -> float:
        """返回y坐标"""
        return self.coordinates_2d[1]
    
    def to_tensor(self, device: Optional[torch.device] = None) -> Tensor:
        """转换为张量格式"""
        tensor = torch.tensor(self.coordinates_2d, dtype=torch.float32)
        if device is not None:
            tensor = tensor.to(device)
        return tensor
    
    def is_valid(self) -> bool:
        """检查轨迹点是否有效"""
        tensor = self.to_tensor()
        return (
            self.confidence > 0.0 and
            not torch.isnan(tensor).any() and
            not torch.isinf(tensor).any()
        )


@dataclass
class SamplingPoint:
    """
    采样点数据结构
    
    Represents a sampling point used in deformable sampling.
    Contains base coordinates, predicted offsets, weights, and sampled features.
    
    Requirements addressed: 2.1, 2.2, 2.3 - Deformable sampling with offsets and weights
    """
    base_coord: Tuple[float, float]                  # 基础坐标
    offset: Tuple[float, float]                      # 预测的偏移量
    weight: float                                    # 注意力权重
    sampled_feature: Optional[Tensor] = None         # 采样得到的特征
    
    def __post_init__(self):
        """验证数据完整性"""
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError(f"权重必须在[0,1]范围内，得到: {self.weight}")
    
    @property
    def final_coord(self) -> Tuple[float, float]:
        """返回最终采样坐标（基础坐标 + 偏移量）"""
        return (
            self.base_coord[0] + self.offset[0],
            self.base_coord[1] + self.offset[1]
        )
    
    def to_tensor(self) -> Tensor:
        """转换为张量格式"""
        return torch.tensor(self.final_coord, dtype=torch.float32)
    
    def is_valid(self) -> bool:
        """检查采样点是否有效"""
        coord_tensor = self.to_tensor()
        return (
            self.weight > 0.0 and
            not torch.isnan(coord_tensor).any() and
            not torch.isinf(coord_tensor).any()
        )


@dataclass
class AppearanceFeature:
    """
    外观特征数据结构
    
    Comprehensive structure containing all appearance feature information
    generated through the GT-DCA pipeline.
    
    Requirements addressed: Complete feature representation for enhanced appearance modeling
    """
    base_features: Tensor                            # 基础外观特征 (N, feature_dim)
    geometry_guided_features: Tensor                 # 几何引导后的特征 (N, feature_dim)
    enhanced_features: Tensor                        # 最终增强特征 (N, feature_dim)
    sampling_metadata: List[List[SamplingPoint]]     # 采样元数据 (N个高斯点，每个有多个采样点)
    
    # 可选的调试和分析信息
    attention_weights: Optional[Tensor] = None       # 交叉注意力权重
    geometric_context: Optional[Tensor] = None       # 几何上下文特征
    processing_metadata: Optional[Dict[str, Any]] = None  # 处理元数据
    
    def __post_init__(self):
        """验证数据完整性"""
        n_gaussians = self.base_features.shape[0]
        
        # 验证特征张量形状一致性
        if self.geometry_guided_features.shape[0] != n_gaussians:
            raise ValueError("几何引导特征数量与基础特征不匹配")
        
        if self.enhanced_features.shape[0] != n_gaussians:
            raise ValueError("增强特征数量与基础特征不匹配")
        
        # 验证采样元数据
        if len(self.sampling_metadata) != n_gaussians:
            raise ValueError("采样元数据数量与高斯点数量不匹配")
    
    @property
    def num_gaussians(self) -> int:
        """返回高斯点数量"""
        return self.base_features.shape[0]
    
    @property
    def feature_dim(self) -> int:
        """返回特征维度"""
        return self.base_features.shape[1]
    
    def get_final_features(self) -> Tensor:
        """获取最终的增强特征"""
        return self.enhanced_features
    
    def get_sampling_statistics(self) -> Dict[str, float]:
        """获取采样统计信息"""
        total_samples = sum(len(samples) for samples in self.sampling_metadata)
        valid_samples = sum(
            sum(1 for sample in samples if sample.is_valid())
            for samples in self.sampling_metadata
        )
        
        avg_weight = 0.0
        if total_samples > 0:
            all_weights = [
                sample.weight 
                for samples in self.sampling_metadata 
                for sample in samples
            ]
            avg_weight = sum(all_weights) / len(all_weights)
        
        return {
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "valid_ratio": valid_samples / max(total_samples, 1),
            "average_weight": avg_weight
        }


@dataclass
class GTDCAConfig:
    """
    GT-DCA配置数据结构
    
    Configuration structure for GT-DCA module parameters and settings.
    Provides centralized configuration management.
    """
    # 特征维度配置
    feature_dim: int = 64                           # 基础特征维度
    hidden_dim: int = 32                           # 隐藏层维度
    
    # 采样配置
    num_sample_points: int = 4                      # 采样点数量
    sampling_radius: float = 2.0                    # 采样半径
    
    # 几何引导配置
    use_cross_attention: bool = True                # 是否使用交叉注意力
    attention_heads: int = 4                        # 注意力头数
    
    # 训练配置
    dropout_rate: float = 0.1                       # Dropout率
    use_layer_norm: bool = True                     # 是否使用层归一化
    
    # 错误处理配置
    min_track_points: int = 4                       # 最小轨迹点数量
    confidence_threshold: float = 0.5               # 置信度阈值
    enable_fallback: bool = True                    # 是否启用降级机制
    
    # 性能配置
    enable_caching: bool = True                     # 是否启用缓存
    max_cache_size: int = 1000                      # 最大缓存大小

    # === 混合精度配置 ===
    use_mixed_precision: bool = False               # 是否启用混合精度（AMP）
    amp_dtype: str = "fp16"                        # AMP 数据类型，可选 "fp16" 或 "bf16"
 
    def validate(self) -> None:
        """验证配置参数"""
        if self.feature_dim <= 0:
            raise ValueError("特征维度必须大于0")
        
        if self.num_sample_points <= 0:
            raise ValueError("采样点数量必须大于0")
        
        if not (0.0 <= self.dropout_rate <= 1.0):
            raise ValueError("Dropout率必须在[0,1]范围内")
        
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("置信度阈值必须在[0,1]范围内")

        # 检查 AMP 数据类型
        if self.amp_dtype not in ["fp16", "bf16"]:
            raise ValueError("amp_dtype 必须为 'fp16' 或 'bf16'")


# 类型别名定义
TrackPointList = List[TrackPoint]
SamplingPointList = List[SamplingPoint]
FeatureTensor = Tensor
CoordinateTensor = Tensor