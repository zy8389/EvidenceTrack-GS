"""
GT-DCA Main Module
GT-DCA主模块

Integrates all GT-DCA sub-modules to provide the complete two-stage
"guidance-sampling" pipeline for enhanced appearance modeling.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional, Dict, Any, Tuple
import logging
import contextlib

from ..core.interfaces import GTDCAInterface, GaussianModelIntegration
from ..core.data_structures import (
    TrackPoint, SamplingPoint, AppearanceFeature, GTDCAConfig
)
from .base_appearance_generator import BaseAppearanceFeatureGenerator
from .geometry_guided_module import GeometryGuidedModule
from .deformable_sampling_module import DeformableSamplingModule


logger = logging.getLogger(__name__)


class GTDCAModule(nn.Module, GTDCAInterface):
    """
    GT-DCA主模块实现
    
    Integrates all sub-modules to provide the complete GT-DCA pipeline:
    1. Base appearance feature generation (query vectors)
    2. Geometry-guided processing using cross-attention
    3. Deformable sampling from 2D feature maps
    
    Requirements addressed: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 4.1, 4.2 - Complete GT-DCA integration
    """
    
    def __init__(self, config: Optional[GTDCAConfig] = None):
        super().__init__()
        
        # Use default config if none provided
        if config is None:
            config = GTDCAConfig()
        
        self.config = config
        config.validate()  # Validate configuration parameters
        
        # Module state
        self._enabled = True
        self._device = None
        
        # Task: 创建GTDCAModule类整合所有子模块
        
        # Sub-module 1: Base appearance feature generator
        self.base_generator = BaseAppearanceFeatureGenerator(config)
        
        # Sub-module 2: Geometry guided module  
        self.geometry_guided = GeometryGuidedModule(config)
        
        # Sub-module 3: Deformable sampling module
        self.deformable_sampling = DeformableSamplingModule(config)
        
        # Task: 添加模块配置和参数管理功能
        
        # Configuration management
        self._config_cache = {}
        self._performance_stats = {
            "forward_calls": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0
        }
        
        # Optional caching for performance optimization
        if config.enable_caching:
            self._feature_cache = {}
            self._max_cache_size = config.max_cache_size
        else:
            self._feature_cache = None
        
        logger.info(f"GT-DCA模块初始化完成，配置: {config}")
    
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
        
        Task: 实现完整的两阶段前向传播流程
        
        Stage 1: Geometry Guidance
        - Generate base appearance features (queries)
        - Apply geometry guidance using cross-attention
        
        Stage 2: Deformable Sampling  
        - Predict sampling offsets and weights
        - Sample features from 2D feature map
        - Aggregate weighted features
        
        Args:
            gaussian_primitives: 3D高斯基元参数 (N, primitive_dim)
            track_points_2d: 2D特征轨迹点列表
            feature_map_2d: 当前视角2D特征图 (H, W, C)
            projection_coords: 高斯点2D投影坐标 (N, 2)
            viewpoint_camera: 可选的视点相机参数
            
        Returns:
            增强的外观特征对象
        """
        if not self._enabled:
            raise RuntimeError("GT-DCA模块已禁用，请先启用模块")
        
        # Update performance statistics
        self._performance_stats["forward_calls"] += 1
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if start_time:
            start_time.record()
        
        # Validate inputs
        self._validate_inputs(gaussian_primitives, track_points_2d, feature_map_2d, projection_coords)
        
        # Set device and ensure all modules are on the correct device
        if self._device is None:
            self._device = gaussian_primitives.device
            # Move all sub-modules to the correct device
            self.to(self._device)
        
        # Ensure all input tensors are on the correct device
        gaussian_primitives = gaussian_primitives.to(self._device)
        feature_map_2d = feature_map_2d.to(self._device)
        projection_coords = projection_coords.to(self._device)
        
        try:
            # === Optional: 混合精度上下文 ===
            amp_context = (
                torch.cuda.amp.autocast(
                    dtype=torch.float16 if self.config.amp_dtype == "fp16" else torch.bfloat16,
                    enabled=self.config.use_mixed_precision and torch.cuda.is_available()
                )
                if torch.cuda.is_available() else contextlib.nullcontext()
            )

            with amp_context:
                # === Stage 1: Geometry Guidance ===

                # Step 1.1: Generate base appearance features (query vectors)
                base_features = self.base_generator.generate_query_features(gaussian_primitives)

                # Step 1.2: Apply geometry guidance using cross-attention
                geometry_guided_features = self.geometry_guided.process_geometry_guidance(
                    query_features=base_features,
                    track_points_2d=track_points_2d
                )

                # === Stage 2: Deformable Sampling ===

                # Step 2.1: Perform deformable sampling from 2D feature map
                enhanced_features = self.deformable_sampling.forward(
                    guided_queries=geometry_guided_features,
                    feature_map_2d=feature_map_2d,
                    projection_coords=projection_coords
                )
            
            # === Generate Sampling Metadata ===
            
            # Get detailed sampling information for analysis
            sampling_offsets, sampling_weights, sampling_coords = self.deformable_sampling.get_sampling_metadata(
                guided_queries=geometry_guided_features,
                projection_coords=projection_coords
            )
            
            # Convert to SamplingPoint objects
            sampling_metadata = self._create_sampling_metadata(
                projection_coords, sampling_offsets, sampling_weights, sampling_coords
            )
            
            # === Create AppearanceFeature Result ===
            
            appearance_feature = AppearanceFeature(
                base_features=base_features,
                geometry_guided_features=geometry_guided_features,
                enhanced_features=enhanced_features,
                sampling_metadata=sampling_metadata,
                # Optional debug information
                geometric_context=self.geometry_guided.extract_geometric_context(track_points_2d),
                processing_metadata={
                    "num_track_points": len(track_points_2d),
                    "valid_track_points": len([p for p in track_points_2d if p.is_valid()]),
                    "feature_map_shape": feature_map_2d.shape,
                    "num_gaussians": gaussian_primitives.shape[0],
                    "config": self.config
                }
            )
            
            # Update performance statistics
            if start_time and torch.cuda.is_available():
                end_time = torch.cuda.Event(enable_timing=True)
                end_time.record()
                torch.cuda.synchronize()
                processing_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
                self._performance_stats["total_processing_time"] += processing_time
                self._performance_stats["average_processing_time"] = (
                    self._performance_stats["total_processing_time"] / 
                    self._performance_stats["forward_calls"]
                )
            
            return appearance_feature
            
        except Exception as e:
            # More detailed error reporting for device issues
            if "device" in str(e).lower():
                print(f"GT-DCA前向传播失败: {e}")
                print(f"设备信息: GT-DCA设备={self._device}")
                print(f"输入张量设备: gaussian_primitives={gaussian_primitives.device}, feature_map_2d={feature_map_2d.device}, projection_coords={projection_coords.device}")
                if 'base_features' in locals():
                    print(f"中间张量设备: base_features={base_features.device}")
                if 'geometry_guided_features' in locals():
                    print(f"中间张量设备: geometry_guided_features={geometry_guided_features.device}")
            else:
                logger.error(f"GT-DCA前向传播失败: {e}")
            
            # Return fallback result with base features only
            return self._create_fallback_result(gaussian_primitives, base_features if 'base_features' in locals() else None)
    
    def get_enhanced_features(
        self,
        gaussian_primitives: Tensor,
        track_points_2d: List[TrackPoint],
        feature_map_2d: Tensor,
        projection_coords: Tensor,
        viewpoint_camera: Optional[object] = None
    ) -> Tensor:
        """
        获取增强的外观特征张量
        
        Convenience method that returns only the final enhanced features tensor.
        
        Args:
            Same as forward()
            
        Returns:
            增强外观特征 (N, feature_dim)
        """
        appearance_feature = self.forward(
            gaussian_primitives, track_points_2d, feature_map_2d, 
            projection_coords, viewpoint_camera
        )
        return appearance_feature.get_final_features()
    
    def _validate_inputs(
        self, 
        gaussian_primitives: Tensor, 
        track_points_2d: List[TrackPoint],
        feature_map_2d: Tensor, 
        projection_coords: Tensor
    ) -> None:
        """验证输入参数"""
        # Validate gaussian_primitives
        if gaussian_primitives.dim() != 2:
            raise ValueError(f"gaussian_primitives必须是2D张量，得到形状: {gaussian_primitives.shape}")
        
        # Validate projection_coords
        if projection_coords.dim() != 2 or projection_coords.shape[1] != 2:
            raise ValueError(f"projection_coords必须是(N, 2)形状，得到: {projection_coords.shape}")
        
        if gaussian_primitives.shape[0] != projection_coords.shape[0]:
            raise ValueError("高斯基元数量与投影坐标数量不匹配")
        
        # Validate feature_map_2d
        if feature_map_2d.dim() not in [3, 4]:
            raise ValueError(f"feature_map_2d必须是3D或4D张量，得到形状: {feature_map_2d.shape}")
        
        # Validate track_points_2d
        if not isinstance(track_points_2d, list):
            raise ValueError("track_points_2d必须是TrackPoint列表")
        
        # Check for minimum track points if required
        valid_points = [p for p in track_points_2d if p.is_valid()]
        if len(valid_points) < self.config.min_track_points:
            raise RuntimeError(
                f"❌ 有效轨迹点数量({len(valid_points)})少于最小要求({self.config.min_track_points})！\n"
                f"GT-DCA需要至少 {self.config.min_track_points} 个有效轨迹点才能正常工作。\n"
                "请确保:\n"
                "1. 轨迹文件包含足够的有效轨迹点\n"
                "2. 轨迹点的置信度满足要求\n"
                "3. 使用正确的轨迹文件格式"
            )
    
    def _create_sampling_metadata(
        self,
        projection_coords: Tensor,
        sampling_offsets: Tensor,
        sampling_weights: Tensor,
        sampling_coords: Tensor
    ) -> List[List[SamplingPoint]]:
        """创建采样元数据"""
        metadata = []
        
        for i in range(projection_coords.shape[0]):
            gaussian_samples = []
            base_coord = (projection_coords[i, 0].item(), projection_coords[i, 1].item())
            
            for j in range(sampling_offsets.shape[1]):
                offset = (sampling_offsets[i, j, 0].item(), sampling_offsets[i, j, 1].item())
                weight = sampling_weights[i, j].item()
                
                sample_point = SamplingPoint(
                    base_coord=base_coord,
                    offset=offset,
                    weight=weight,
                    sampled_feature=None  # Feature tensor not stored in metadata for memory efficiency
                )
                gaussian_samples.append(sample_point)
            
            metadata.append(gaussian_samples)
        
        return metadata
    
    def _create_fallback_result(
        self, 
        gaussian_primitives: Tensor, 
        base_features: Optional[Tensor] = None
    ) -> AppearanceFeature:
        """创建降级结果"""
        if base_features is None:
            # Generate minimal base features
            base_features = torch.zeros(
                gaussian_primitives.shape[0], 
                self.config.feature_dim,
                device=gaussian_primitives.device
            )
        
        # Create minimal AppearanceFeature with base features only
        return AppearanceFeature(
            base_features=base_features,
            geometry_guided_features=base_features.clone(),
            enhanced_features=base_features.clone(),
            sampling_metadata=[[] for _ in range(base_features.shape[0])],
            processing_metadata={"fallback": True}
        )
    
    # === Configuration and State Management ===
    
    @property
    def is_enabled(self) -> bool:
        """返回GT-DCA模块是否启用"""
        return self._enabled
    
    def enable(self) -> None:
        """启用GT-DCA模块"""
        self._enabled = True
        logger.info("GT-DCA模块已启用")
    
    def disable(self) -> None:
        """禁用GT-DCA模块"""
        self._enabled = False
        logger.info("GT-DCA模块已禁用")
    
    def update_config(self, new_config: GTDCAConfig) -> None:
        """更新模块配置"""
        new_config.validate()
        old_config = self.config
        self.config = new_config
        
        # Cache old config for potential rollback
        self._config_cache["previous"] = old_config
        
        logger.info(f"GT-DCA配置已更新: {new_config}")
    
    def rollback_config(self) -> None:
        """回滚到之前的配置"""
        if "previous" in self._config_cache:
            self.config = self._config_cache["previous"]
            logger.info("GT-DCA配置已回滚到之前版本")
        else:
            logger.warning("没有可回滚的配置版本")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return self._performance_stats.copy()
    
    def reset_performance_stats(self) -> None:
        """重置性能统计信息"""
        self._performance_stats = {
            "forward_calls": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0
        }
        logger.info("性能统计信息已重置")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况"""
        if not torch.cuda.is_available():
            return {"error": "CUDA不可用"}
        
        return {
            "allocated": torch.cuda.memory_allocated(self._device) / 1024**2,  # MB
            "cached": torch.cuda.memory_reserved(self._device) / 1024**2,      # MB
            "max_allocated": torch.cuda.max_memory_allocated(self._device) / 1024**2  # MB
        }
    
    def clear_cache(self) -> None:
        """清空缓存"""
        if self._feature_cache is not None:
            self._feature_cache.clear()
            logger.info("特征缓存已清空")
    
    # === Debugging and Analysis Methods ===
    
    def analyze_geometric_context(self, track_points_2d: List[TrackPoint]) -> Dict[str, Any]:
        """分析几何上下文质量"""
        valid_points = [p for p in track_points_2d if p.is_valid()]
        high_conf_points = [p for p in valid_points if p.confidence >= self.config.confidence_threshold]
        
        if not valid_points:
            return {"error": "没有有效的轨迹点"}
        
        confidences = [p.confidence for p in valid_points]
        
        return {
            "total_points": len(track_points_2d),
            "valid_points": len(valid_points),
            "high_confidence_points": len(high_conf_points),
            "average_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "meets_minimum": len(valid_points) >= self.config.min_track_points
        }
    
    def get_module_info(self) -> Dict[str, Any]:
        """获取模块信息"""
        return {
            "enabled": self._enabled,
            "config": self.config,
            "device": str(self._device) if self._device else None,
            "performance_stats": self._performance_stats,
            "sub_modules": {
                "base_generator": type(self.base_generator).__name__,
                "geometry_guided": type(self.geometry_guided).__name__,
                "deformable_sampling": type(self.deformable_sampling).__name__
            }
        }
    
    def __repr__(self) -> str:
        return f"GTDCAModule(enabled={self._enabled}, feature_dim={self.config.feature_dim})"