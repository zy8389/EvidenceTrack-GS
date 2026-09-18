"""
GT-DCA Core Interfaces
核心接口定义

Defines the core interfaces and abstract base classes for the GT-DCA system.
These interfaces establish the contracts for different components and enable
modular design and testing.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
import torch
from torch import Tensor

from .data_structures import TrackPoint, SamplingPoint, AppearanceFeature


class AppearanceFeatureGenerator(ABC):
    """
    基础外观特征生成器接口
    
    Abstract interface for generating base appearance features from 3D Gaussian primitives.
    This serves as the query vector for the subsequent geometry-guided processing.
    
    Requirements addressed: 1.1 - Generate learnable feature vectors for 3D Gaussian primitives
    """
    
    @abstractmethod
    def generate_query_features(self, gaussian_primitives: Tensor) -> Tensor:
        """
        为3D高斯基元生成可学习的基础外观特征向量
        
        Args:
            gaussian_primitives: 3D高斯基元参数 (N, primitive_dim)
            
        Returns:
            基础外观特征向量 (N, feature_dim)
        """
        pass
    
    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """返回特征向量维度"""
        pass


class GeometryGuidedProcessor(ABC):
    """
    几何引导处理器接口
    
    Abstract interface for processing geometry-guided appearance features.
    Uses cross-attention mechanism to inject geometric context into query vectors.
    
    Requirements addressed: 1.2, 1.3 - Geometric guidance using 2D trajectory points
    """
    
    @abstractmethod
    def process_geometry_guidance(
        self, 
        query_features: Tensor, 
        track_points_2d: List[TrackPoint]
    ) -> Tensor:
        """
        通过几何引导处理查询特征
        
        Args:
            query_features: 基础查询特征 (N, feature_dim)
            track_points_2d: 2D特征轨迹点列表
            
        Returns:
            几何引导后的特征 (N, feature_dim)
        """
        pass
    
    @abstractmethod
    def extract_geometric_context(self, track_points_2d: List[TrackPoint]) -> Tensor:
        """
        从2D轨迹点提取几何上下文
        
        Args:
            track_points_2d: 2D特征轨迹点列表
            
        Returns:
            几何上下文特征 (M, context_dim)
        """
        pass


class DeformableSampler(ABC):
    """
    可变形采样器接口
    
    Abstract interface for deformable sampling from 2D feature maps.
    Predicts sampling offsets and weights for enhanced feature extraction.
    
    Requirements addressed: 2.1, 2.2, 2.3 - Deformable sampling with offset and weight prediction
    """
    
    @abstractmethod
    def predict_sampling_offsets(
        self, 
        guided_queries: Tensor, 
        projection_coords: Tensor
    ) -> Tensor:
        """
        预测采样偏移量
        
        Args:
            guided_queries: 几何引导后的查询向量 (N, feature_dim)
            projection_coords: 高斯点2D投影坐标 (N, 2)
            
        Returns:
            采样偏移量 (N, num_sample_points, 2)
        """
        pass
    
    @abstractmethod
    def predict_sampling_weights(
        self, 
        guided_queries: Tensor, 
        projection_coords: Tensor
    ) -> Tensor:
        """
        预测采样权重
        
        Args:
            guided_queries: 几何引导后的查询向量 (N, feature_dim)
            projection_coords: 高斯点2D投影坐标 (N, 2)
            
        Returns:
            采样权重 (N, num_sample_points)，已归一化
        """
        pass
    
    @abstractmethod
    def sample_features(
        self, 
        feature_map_2d: Tensor, 
        sampling_coords: Tensor, 
        sampling_weights: Tensor
    ) -> Tensor:
        """
        从2D特征图采样特征
        
        Args:
            feature_map_2d: 2D图像特征图 (H, W, C)
            sampling_coords: 采样坐标 (N, num_sample_points, 2)
            sampling_weights: 采样权重 (N, num_sample_points)
            
        Returns:
            加权采样特征 (N, C)
        """
        pass
    
    @property
    @abstractmethod
    def num_sample_points(self) -> int:
        """返回采样点数量"""
        pass


class GTDCAInterface(ABC):
    """
    GT-DCA主模块接口
    
    Main interface for the complete GT-DCA module that integrates all components.
    Provides the end-to-end processing pipeline for enhanced appearance modeling.
    
    Requirements addressed: All core requirements (1.1-2.3, 3.1-3.3, 4.1-4.2)
    """
    
    @abstractmethod
    def forward(
        self,
        gaussian_primitives: Tensor,
        track_points_2d: List[TrackPoint],
        feature_map_2d: Tensor,
        projection_coords: Tensor,
        viewpoint_camera: Optional[object] = None
    ) -> AppearanceFeature:
        """
        GT-DCA完整前向传播
        
        Args:
            gaussian_primitives: 3D高斯基元参数 (N, primitive_dim)
            track_points_2d: 2D特征轨迹点列表
            feature_map_2d: 当前视角2D特征图 (H, W, C)
            projection_coords: 高斯点2D投影坐标 (N, 2)
            viewpoint_camera: 可选的视点相机参数
            
        Returns:
            增强的外观特征对象
        """
        pass
    
    @abstractmethod
    def get_enhanced_features(self, *args, **kwargs) -> Tensor:
        """
        获取增强的外观特征张量
        
        Returns:
            增强外观特征 (N, feature_dim)
        """
        pass
    
    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        """返回GT-DCA模块是否启用"""
        pass
    
    @abstractmethod
    def enable(self) -> None:
        """启用GT-DCA模块"""
        pass
    
    @abstractmethod
    def disable(self) -> None:
        """禁用GT-DCA模块"""
        pass


class GaussianModelIntegration(ABC):
    """
    3DGS系统集成接口
    
    Interface for integrating GT-DCA with existing 3D Gaussian Splatting systems.
    Provides compatibility layer and fallback mechanisms.
    
    Requirements addressed: 3.1, 3.2, 3.3 - Seamless integration with 3DGS pipeline
    """
    
    @abstractmethod
    def get_appearance_features(
        self, 
        viewpoint_camera: object,
        use_gt_dca: bool = True
    ) -> Tensor:
        """
        获取外观特征（GT-DCA增强或标准SH）
        
        Args:
            viewpoint_camera: 视点相机参数
            use_gt_dca: 是否使用GT-DCA增强
            
        Returns:
            外观特征张量
        """
        pass
    
    @abstractmethod
    def get_sh_features(self, viewpoint_camera: object) -> Tensor:
        """
        获取标准球谐函数特征（降级方案）
        
        Args:
            viewpoint_camera: 视点相机参数
            
        Returns:
            SH特征张量
        """
        pass
    
    @abstractmethod
    def setup_gt_dca_integration(self, gt_dca_module: GTDCAInterface) -> None:
        """
        设置GT-DCA集成
        
        Args:
            gt_dca_module: GT-DCA模块实例
        """
        pass