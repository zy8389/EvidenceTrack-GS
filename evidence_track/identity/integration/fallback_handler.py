"""
Fallback Handler
降级处理器

Handles fallback scenarios when GT-DCA processing fails or is unavailable.
Provides graceful degradation to standard SH-based appearance modeling.
"""

import torch
from torch import Tensor
from typing import Optional, Dict, Any, List, Callable
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class FallbackReason(Enum):
    """降级原因枚举"""
    GEOMETRY_FAILURE = "geometry_failure"
    SAMPLING_FAILURE = "sampling_failure"
    INSUFFICIENT_TRACKS = "insufficient_tracks"
    INVALID_FEATURE_MAP = "invalid_feature_map"
    BOUNDARY_ISSUES = "boundary_issues"
    COMPUTATION_ERROR = "computation_error"
    MEMORY_ERROR = "memory_error"
    TIMEOUT_ERROR = "timeout_error"


class FallbackLevel(Enum):
    """降级级别枚举"""
    NONE = 0           # 无降级，正常处理
    PARTIAL = 1        # 部分降级，使用简化处理
    STANDARD_SH = 2    # 降级到标准SH模型
    MINIMAL = 3        # 最小降级，使用基础特征


class FallbackHandler:
    """
    降级处理器
    
    Handles various fallback scenarios for GT-DCA processing failures.
    Ensures system robustness by providing alternative processing paths.
    
    Requirements addressed: 6.3 - Fallback to standard SH model when geometry info is missing
    """
    
    def __init__(self, gaussian_model, config: Optional[Dict[str, Any]] = None):
        """
        初始化降级处理器
        
        Args:
            gaussian_model: 高斯模型实例，用于访问标准SH特征
            config: 降级处理配置
        """
        self.gaussian_model = gaussian_model
        self.config = config or {}
        
        # 统计信息
        self.fallback_count = 0
        self.last_fallback_reason = None
        self.fallback_history: List[Dict[str, Any]] = []
        self.reason_counts: Dict[FallbackReason, int] = {reason: 0 for reason in FallbackReason}
        
        # 配置参数
        self.max_fallback_attempts = self.config.get('max_fallback_attempts', 3)
        self.enable_progressive_fallback = self.config.get('enable_progressive_fallback', True)
        self.fallback_timeout = self.config.get('fallback_timeout', 5.0)  # 秒
        
        # 降级策略映射
        self.fallback_strategies: Dict[FallbackReason, Callable] = {
            FallbackReason.GEOMETRY_FAILURE: self._handle_geometry_failure,
            FallbackReason.SAMPLING_FAILURE: self._handle_sampling_failure,
            FallbackReason.INSUFFICIENT_TRACKS: self._handle_insufficient_tracks,
            FallbackReason.INVALID_FEATURE_MAP: self._handle_invalid_feature_map,
            FallbackReason.BOUNDARY_ISSUES: self._handle_boundary_issues,
            FallbackReason.COMPUTATION_ERROR: self._handle_computation_error,
            FallbackReason.MEMORY_ERROR: self._handle_memory_error,
            FallbackReason.TIMEOUT_ERROR: self._handle_timeout_error,
        }
        
        # 当前降级级别
        self.current_fallback_level = FallbackLevel.NONE
    
    def get_standard_sh_features(self, viewpoint_camera: object) -> Tensor:
        """
        获取标准球谐函数特征
        
        Args:
            viewpoint_camera: 视点相机参数
            
        Returns:
            标准SH特征张量
        """
        try:
            # 获取标准的SH特征
            features = self.gaussian_model.get_features
            return features
            
        except Exception as e:
            logger.error(f"标准SH特征获取失败: {e}")
            # 如果连标准特征都获取失败，返回零特征
            return self._get_zero_features()
    
    def handle_geometry_failure(
        self, 
        gaussian_primitives: Tensor,
        reason: str = "几何信息不可用"
    ) -> Tensor:
        """
        处理几何信息失效的情况
        
        Args:
            gaussian_primitives: 3D高斯基元参数
            reason: 失效原因
            
        Returns:
            降级后的外观特征
        """
        self._log_fallback("geometry_failure", reason)
        
        # 回退到基于高斯基元位置的简单特征
        xyz = gaussian_primitives[:, :3]  # 假设前3维是位置信息
        
        # 基于位置生成简单的外观特征
        # 这是一个非常基础的实现，实际中可能需要更复杂的逻辑
        feature_dim = getattr(self.gaussian_model, 'feature_dim', 256)
        features = torch.zeros(xyz.shape[0], feature_dim, device=xyz.device)
        
        # 使用位置信息生成一些基础特征
        features[:, :3] = torch.tanh(xyz * 0.1)  # 归一化位置特征
        
        return features
    
    def handle_sampling_failure(
        self, 
        guided_queries: Tensor,
        reason: str = "采样失败"
    ) -> Tensor:
        """
        处理可变形采样失败的情况
        
        Args:
            guided_queries: 几何引导后的查询向量
            reason: 失效原因
            
        Returns:
            降级后的特征（直接返回引导查询）
        """
        self._log_fallback("sampling_failure", reason)
        
        # 直接返回几何引导的查询向量，跳过采样阶段
        return guided_queries
    
    def handle_track_points_insufficient(
        self, 
        available_points: int,
        minimum_required: int = 4
    ) -> bool:
        """
        处理轨迹点数量不足的情况
        
        Args:
            available_points: 可用轨迹点数量
            minimum_required: 最小需求数量
            
        Returns:
            是否应该启用降级模式
        """
        if available_points < minimum_required:
            reason = f"轨迹点不足: {available_points} < {minimum_required}"
            self._log_fallback("insufficient_tracks", reason)
            return True
        
        return False
    
    def handle_feature_map_invalid(
        self, 
        feature_map: Optional[Tensor],
        reason: str = "特征图无效"
    ) -> Tensor:
        """
        处理特征图无效的情况
        
        Args:
            feature_map: 可能无效的特征图
            reason: 无效原因
            
        Returns:
            有效的特征图（可能是占位符）
        """
        if feature_map is None or feature_map.numel() == 0:
            self._log_fallback("invalid_feature_map", reason)
            
            # 返回默认尺寸的零特征图
            height, width, channels = 480, 640, 256
            device = getattr(self.gaussian_model, '_xyz', torch.tensor([])).device
            return torch.zeros(height, width, channels, device=device)
        
        # 检查特征图是否包含NaN或Inf
        if torch.isnan(feature_map).any() or torch.isinf(feature_map).any():
            self._log_fallback("corrupted_feature_map", "特征图包含NaN或Inf值")
            return torch.zeros_like(feature_map)
        
        return feature_map
    
    def handle_boundary_sampling(
        self, 
        coords: Tensor, 
        feature_map_shape: tuple
    ) -> Tensor:
        """
        处理边界采样问题
        
        Args:
            coords: 采样坐标
            feature_map_shape: 特征图形状 (H, W, C)
            
        Returns:
            边界安全的采样坐标
        """
        from ..utils.boundary_handling import BoundaryHandler, BoundaryStrategy
        
        h, w = feature_map_shape[:2]
        
        # 使用专门的边界处理器
        boundary_handler = BoundaryHandler(BoundaryStrategy.CLAMP)
        processed_coords, boundary_mask = boundary_handler._handle_boundary_coordinates(
            coords, (w, h)
        )
        
        # 记录边界处理统计
        if boundary_mask is not None:
            clipped_count = boundary_mask.sum().item()
            self._log_fallback("boundary_clipping", f"处理了{clipped_count}个越界采样点")
        
        return processed_coords
    
    def execute_fallback(
        self, 
        reason: FallbackReason, 
        context: Dict[str, Any],
        **kwargs
    ) -> Any:
        """
        执行降级处理
        
        Args:
            reason: 降级原因
            context: 上下文信息
            **kwargs: 额外参数
            
        Returns:
            降级处理结果
            
        Requirements addressed: 6.3 - Comprehensive fallback execution system
        """
        import time
        
        start_time = time.time()
        
        try:
            # 记录降级事件
            self._record_fallback_event(reason, context)
            
            # 选择合适的降级策略
            if reason in self.fallback_strategies:
                strategy = self.fallback_strategies[reason]
                result = strategy(context, **kwargs)
            else:
                logger.warning(f"未知的降级原因: {reason}, 使用默认策略")
                result = self._handle_default_fallback(context, **kwargs)
            
            # 更新降级级别
            self._update_fallback_level(reason)
            
            processing_time = time.time() - start_time
            logger.info(f"降级处理完成: {reason.value}, 耗时: {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"降级处理失败: {reason.value}, 错误: {str(e)}")
            # 最后的降级方案
            return self._handle_critical_fallback(context)
    
    def _handle_geometry_failure(self, context: Dict[str, Any], **kwargs) -> Tensor:
        """处理几何信息失效"""
        gaussian_primitives = context.get('gaussian_primitives')
        if gaussian_primitives is None:
            return self._get_zero_features()
        
        return self.handle_geometry_failure(gaussian_primitives, context.get('reason', ''))
    
    def _handle_sampling_failure(self, context: Dict[str, Any], **kwargs) -> Tensor:
        """处理采样失效"""
        guided_queries = context.get('guided_queries')
        if guided_queries is None:
            return self._get_zero_features()
        
        return self.handle_sampling_failure(guided_queries, context.get('reason', ''))
    
    def _handle_insufficient_tracks(self, context: Dict[str, Any], **kwargs) -> bool:
        """处理轨迹点不足"""
        available_points = context.get('available_points', 0)
        minimum_required = context.get('minimum_required', 4)
        
        return self.handle_track_points_insufficient(available_points, minimum_required)
    
    def _handle_invalid_feature_map(self, context: Dict[str, Any], **kwargs) -> Tensor:
        """处理特征图无效"""
        feature_map = context.get('feature_map')
        reason = context.get('reason', '特征图无效')
        
        return self.handle_feature_map_invalid(feature_map, reason)
    
    def _handle_boundary_issues(self, context: Dict[str, Any], **kwargs) -> Tensor:
        """处理边界问题"""
        coords = context.get('coords')
        feature_map_shape = context.get('feature_map_shape')
        
        if coords is None or feature_map_shape is None:
            logger.error("边界处理缺少必要参数")
            return coords if coords is not None else torch.tensor([])
        
        return self.handle_boundary_sampling(coords, feature_map_shape)
    
    def _handle_computation_error(self, context: Dict[str, Any], **kwargs) -> Tensor:
        """处理计算错误"""
        logger.warning("计算错误，回退到标准SH模型")
        self.current_fallback_level = FallbackLevel.STANDARD_SH
        
        try:
            viewpoint_camera = context.get('viewpoint_camera')
            if viewpoint_camera:
                return self.get_standard_sh_features(viewpoint_camera)
        except Exception:
            pass
        
        return self._get_zero_features()
    
    def _handle_memory_error(self, context: Dict[str, Any], **kwargs) -> Tensor:
        """处理内存错误"""
        logger.warning("内存不足，使用最小降级模式")
        self.current_fallback_level = FallbackLevel.MINIMAL
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return self._get_minimal_features()
    
    def _handle_timeout_error(self, context: Dict[str, Any], **kwargs) -> Tensor:
        """处理超时错误"""
        logger.warning("处理超时，使用快速降级模式")
        self.current_fallback_level = FallbackLevel.PARTIAL
        
        return self._get_fast_fallback_features(context)
    
    def _handle_default_fallback(self, context: Dict[str, Any], **kwargs) -> Tensor:
        """默认降级处理"""
        logger.info("使用默认降级策略")
        return self._get_zero_features()
    
    def _handle_critical_fallback(self, context: Dict[str, Any]) -> Tensor:
        """关键降级处理 - 最后的保障"""
        logger.critical("执行关键降级处理")
        self.current_fallback_level = FallbackLevel.MINIMAL
        
        try:
            return torch.zeros(1, 256, device='cpu')
        except Exception:
            # 如果连这个都失败了，返回None让上层处理
            return None
    
    def _get_minimal_features(self) -> Tensor:
        """获取最小特征"""
        try:
            # 尝试获取最基本的特征
            device = getattr(self.gaussian_model, '_xyz', torch.tensor([])).device
            return torch.zeros(1, 64, device=device)  # 使用更小的特征维度
        except Exception:
            return torch.zeros(1, 64, device='cpu')
    
    def _get_fast_fallback_features(self, context: Dict[str, Any]) -> Tensor:
        """获取快速降级特征"""
        try:
            # 使用简化的特征生成
            gaussian_primitives = context.get('gaussian_primitives')
            if gaussian_primitives is not None:
                # 只使用位置信息生成简单特征
                xyz = gaussian_primitives[:, :3]
                return torch.tanh(xyz * 0.1)  # 简单的位置编码
            else:
                return self._get_minimal_features()
        except Exception:
            return self._get_minimal_features()
    
    def _record_fallback_event(self, reason: FallbackReason, context: Dict[str, Any]):
        """记录降级事件"""
        import time
        
        event = {
            'timestamp': time.time(),
            'reason': reason.value,
            'context_keys': list(context.keys()),
            'fallback_level': self.current_fallback_level.value,
            'attempt_count': self.reason_counts[reason] + 1
        }
        
        self.fallback_history.append(event)
        self.reason_counts[reason] += 1
        self.fallback_count += 1
        self.last_fallback_reason = reason.value
        
        # 限制历史记录长度
        if len(self.fallback_history) > 100:
            self.fallback_history = self.fallback_history[-50:]
        
        logger.warning(f"GT-DCA降级处理 [{reason.value}]: 第{event['attempt_count']}次")
    
    def _update_fallback_level(self, reason: FallbackReason):
        """更新降级级别"""
        if self.enable_progressive_fallback:
            # 根据降级原因和频率调整级别
            if self.reason_counts[reason] > 3:
                if self.current_fallback_level.value < FallbackLevel.STANDARD_SH.value:
                    self.current_fallback_level = FallbackLevel.STANDARD_SH
                    logger.info("升级到标准SH降级级别")
            
            if self.fallback_count > 10:
                if self.current_fallback_level.value < FallbackLevel.MINIMAL.value:
                    self.current_fallback_level = FallbackLevel.MINIMAL
                    logger.info("升级到最小降级级别")
    
    def should_use_fallback(self, reason: FallbackReason) -> bool:
        """
        判断是否应该使用降级处理
        
        Args:
            reason: 降级原因
            
        Returns:
            是否应该降级
        """
        # 检查降级次数限制
        if self.reason_counts[reason] >= self.max_fallback_attempts:
            logger.warning(f"降级次数达到上限: {reason.value}")
            return False
        
        # 检查当前降级级别
        if self.current_fallback_level == FallbackLevel.MINIMAL:
            return True  # 最小级别总是允许降级
        
        return True
    
    def get_current_fallback_level(self) -> FallbackLevel:
        """获取当前降级级别"""
        return self.current_fallback_level
    
    def set_fallback_level(self, level: FallbackLevel):
        """设置降级级别"""
        self.current_fallback_level = level
        logger.info(f"设置降级级别为: {level.name}")
    
    def _get_zero_features(self) -> Tensor:
        """获取零特征作为最后的降级方案"""
        try:
            num_gaussians = self.gaussian_model.get_xyz.shape[0]
            feature_dim = getattr(self.gaussian_model, 'feature_dim', 256)
            device = self.gaussian_model._xyz.device
            
            return torch.zeros(num_gaussians, feature_dim, device=device)
            
        except Exception:
            # 如果连基本信息都获取不到，返回最小的占位符
            return torch.zeros(1, 256)
    
    def _log_fallback(self, fallback_type: str, reason: str) -> None:
        """记录降级事件（兼容旧接口）"""
        self.fallback_count += 1
        self.last_fallback_reason = reason
        
        logger.warning(f"GT-DCA降级处理 [{fallback_type}]: {reason} (总计: {self.fallback_count})")
    
    def get_fallback_statistics(self) -> dict:
        """获取降级统计信息"""
        return {
            "total_fallbacks": self.fallback_count,
            "last_reason": self.last_fallback_reason,
            "current_level": self.current_fallback_level.name,
            "reason_counts": {reason.value: count for reason, count in self.reason_counts.items()},
            "recent_events": self.fallback_history[-10:] if self.fallback_history else []
        }
    
    def reset_statistics(self) -> None:
        """重置降级统计"""
        self.fallback_count = 0
        self.last_fallback_reason = None
        self.fallback_history.clear()
        self.reason_counts = {reason: 0 for reason in FallbackReason}
        self.current_fallback_level = FallbackLevel.NONE
        logger.info("降级统计已重置")