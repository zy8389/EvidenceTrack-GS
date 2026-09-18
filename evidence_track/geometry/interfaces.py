"""
核心接口定义

定义约束引擎和质量评估器的抽象接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import torch
from .data_structures import Trajectory, ConstraintResult


class QualityAssessor(ABC):
    """质量评估器抽象接口"""
    
    @abstractmethod
    def assess_trajectory_quality(self, trajectory: Trajectory) -> float:
        """
        评估轨迹质量
        
        Args:
            trajectory: 待评估的轨迹
            
        Returns:
            质量分数 (0.0-1.0)
        """
        pass
    
    @abstractmethod
    def compute_quality_metrics(self, trajectory: Trajectory) -> Dict[str, float]:
        """
        计算详细的质量指标
        
        Args:
            trajectory: 待评估的轨迹
            
        Returns:
            包含各项质量指标的字典
        """
        pass
    
    @abstractmethod
    def is_trajectory_reliable(self, trajectory: Trajectory, threshold: float = 0.4) -> bool:
        """
        判断轨迹是否可靠
        
        Args:
            trajectory: 待判断的轨迹
            threshold: 质量阈值
            
        Returns:
            是否可靠
        """
        pass
    
    @abstractmethod
    def detect_outliers(self, trajectory: Trajectory) -> torch.Tensor:
        """
        检测轨迹中的异常值
        
        Args:
            trajectory: 待检测的轨迹
            
        Returns:
            异常值掩码张量
        """
        pass


class ConstraintEngine(ABC):
    """约束引擎抽象接口"""
    
    @abstractmethod
    def compute_reprojection_constraints(self, 
                                       trajectories: List[Trajectory],
                                       cameras: List,  # Camera objects
                                       gaussian_points: torch.Tensor) -> ConstraintResult:
        """
        计算重投影约束
        
        Args:
            trajectories: 特征轨迹列表
            cameras: 相机参数列表
            gaussian_points: 高斯点云
            
        Returns:
            约束计算结果
        """
        pass
    
    @abstractmethod
    def compute_multiscale_constraints(self,
                                     trajectories: List[Trajectory],
                                     cameras: List,
                                     scales: List[float]) -> ConstraintResult:
        """
        计算多尺度约束
        
        Args:
            trajectories: 特征轨迹列表
            cameras: 相机参数列表
            scales: 尺度列表
            
        Returns:
            多尺度约束结果
        """
        pass
    
    @abstractmethod
    def compute_adaptive_weights(self,
                               trajectories: List[Trajectory],
                               image_regions: Optional[torch.Tensor] = None,
                               iteration: int = 0) -> torch.Tensor:
        """
        计算自适应权重
        
        Args:
            trajectories: 特征轨迹列表
            image_regions: 图像区域张量
            iteration: 当前训练迭代次数
            
        Returns:
            权重张量
        """
        pass
    
    @abstractmethod
    def validate_constraints(self, 
                           constraint_result: ConstraintResult,
                           tolerance: float = 0.85) -> Tuple[bool, Dict[str, float]]:
        """
        验证约束有效性
        
        Args:
            constraint_result: 约束计算结果
            tolerance: 容忍度阈值
            
        Returns:
            (是否有效, 验证指标字典)
        """
        pass
    
    @abstractmethod
    def update_constraint_parameters(self, 
                                   performance_metrics: Dict[str, float]):
        """
        更新约束参数
        
        Args:
            performance_metrics: 性能指标字典
        """
        pass


class TrajectoryManager(ABC):
    """轨迹管理器抽象接口"""
    
    @abstractmethod
    def load_trajectories(self, track_file: str) -> List[Trajectory]:
        """
        从文件加载轨迹
        
        Args:
            track_file: 轨迹文件路径
            
        Returns:
            轨迹列表
        """
        pass
    
    @abstractmethod
    def preprocess_trajectories(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        """
        预处理轨迹
        
        Args:
            trajectories: 原始轨迹列表
            
        Returns:
            预处理后的轨迹列表
        """
        pass
    
    @abstractmethod
    def update_trajectory_quality(self, trajectory: Trajectory, quality_score: float):
        """
        更新轨迹质量分数
        
        Args:
            trajectory: 目标轨迹
            quality_score: 新的质量分数
        """
        pass
    
    @abstractmethod
    def get_active_trajectories(self, min_quality: float = 0.4) -> List[Trajectory]:
        """
        获取活跃轨迹
        
        Args:
            min_quality: 最小质量阈值
            
        Returns:
            活跃轨迹列表
        """
        pass
    
    @abstractmethod
    def split_trajectory(self, trajectory: Trajectory, split_points: List[int]) -> List[Trajectory]:
        """
        分割轨迹
        
        Args:
            trajectory: 待分割的轨迹
            split_points: 分割点索引列表
            
        Returns:
            分割后的轨迹列表
        """
        pass