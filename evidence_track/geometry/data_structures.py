"""
核心数据结构定义

包含轨迹、2D点和约束结果的数据类定义
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import torch


@dataclass
class Point2D:
    """2D 特征点数据结构"""
    x: float
    y: float
    frame_id: int
    feature_descriptor: Optional[torch.Tensor] = None
    detection_confidence: float = 1.0
    
    def to_tensor(self) -> torch.Tensor:
        """转换为张量格式"""
        return torch.tensor([self.x, self.y], dtype=torch.float32)
    
    def __post_init__(self):
        """数据验证"""
        if self.detection_confidence < 0.0 or self.detection_confidence > 1.0:
            raise ValueError("detection_confidence must be between 0.0 and 1.0")
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")


@dataclass
class Trajectory:
    """特征轨迹数据结构"""
    id: int
    points_2d: List[Point2D]
    camera_indices: List[int]
    quality_score: float = 0.0
    confidence_scores: List[float] = None
    is_active: bool = True
    last_updated: int = 0
    
    def __post_init__(self):
        """初始化后处理和验证"""
        if self.confidence_scores is None:
            self.confidence_scores = [point.detection_confidence for point in self.points_2d]
        
        # 数据验证
        if len(self.points_2d) != len(self.camera_indices):
            raise ValueError("points_2d and camera_indices must have the same length")
        
        if len(self.confidence_scores) != len(self.points_2d):
            raise ValueError("confidence_scores must match the number of points")
        
        if self.quality_score < 0.0 or self.quality_score > 1.0:
            raise ValueError("quality_score must be between 0.0 and 1.0")
    
    @property
    def length(self) -> int:
        """轨迹长度（点的数量）"""
        return len(self.points_2d)
    
    @property
    def average_confidence(self) -> float:
        """平均置信度"""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)
    
    def get_points_tensor(self) -> torch.Tensor:
        """获取所有点的张量表示"""
        points = torch.stack([point.to_tensor() for point in self.points_2d])
        return points
    
    def get_frame_ids(self) -> List[int]:
        """获取所有帧ID"""
        return [point.frame_id for point in self.points_2d]
    
    def is_valid(self, min_length: int = 3, min_quality: float = 0.4) -> bool:
        """检查轨迹是否有效"""
        return (self.length >= min_length and 
                self.quality_score >= min_quality and 
                self.is_active)
    
    def update_quality_score(self, new_score: float):
        """更新质量分数"""
        if new_score < 0.0 or new_score > 1.0:
            raise ValueError("Quality score must be between 0.0 and 1.0")
        self.quality_score = new_score


@dataclass
class ConstraintResult:
    """约束计算结果数据结构"""
    loss_value: torch.Tensor
    individual_errors: List[float]
    outlier_mask: torch.Tensor
    quality_metrics: Dict[str, float]
    scale_contributions: Dict[float, float] = None
    weight_distribution: Dict[str, float] = None
    
    def __post_init__(self):
        """初始化默认值"""
        if self.scale_contributions is None:
            self.scale_contributions = {}
        if self.weight_distribution is None:
            self.weight_distribution = {}
    
    @property
    def total_loss(self) -> float:
        """总损失值"""
        return float(self.loss_value.item())
    
    @property
    def outlier_ratio(self) -> float:
        """异常值比例"""
        if self.outlier_mask.numel() == 0:
            return 0.0
        return float(self.outlier_mask.sum()) / self.outlier_mask.numel()
    
    @property
    def mean_error(self) -> float:
        """平均误差"""
        if not self.individual_errors:
            return 0.0
        return sum(self.individual_errors) / len(self.individual_errors)