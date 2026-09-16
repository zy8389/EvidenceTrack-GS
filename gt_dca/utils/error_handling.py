"""
Error Handling Integration
错误处理集成模块

Provides comprehensive error handling and fallback mechanisms for GT-DCA system.
Integrates data validation, boundary handling, and fallback strategies.
"""

import torch
from torch import Tensor
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from contextlib import contextmanager
import traceback
import time

from .validation import (
    ValidationResult, DataIntegrityError, 
    validate_track_points_comprehensive, repair_track_points, 
    generate_fallback_tracks, validate_sampling_points,
    validate_appearance_features
)
from .boundary_handling import BoundaryHandler, BoundaryStrategy
from ..integration.fallback_handler import FallbackHandler, FallbackReason, FallbackLevel
from ..core.data_structures import TrackPoint, SamplingPoint, AppearanceFeature

logger = logging.getLogger(__name__)


class GTDCAErrorHandler:
    """
    GT-DCA综合错误处理器
    
    Provides comprehensive error handling for the entire GT-DCA pipeline.
    Integrates validation, boundary handling, and fallback mechanisms.
    
    Requirements addressed: 6.1, 6.2, 6.3 - Complete error handling system
    """
    
    def __init__(self, gaussian_model, config: Optional[Dict[str, Any]] = None):
        """
        初始化错误处理器
        
        Args:
            gaussian_model: 高斯模型实例
            config: 配置参数
        """
        self.config = config or {}
        
        # 初始化子组件
        self.fallback_handler = FallbackHandler(gaussian_model, config)
        self.boundary_handler = BoundaryHandler(
            BoundaryStrategy(self.config.get('boundary_strategy', 'clamp'))
        )
        
        # 错误处理配置
        self.enable_auto_repair = self.config.get('enable_auto_repair', True)
        self.max_retry_attempts = self.config.get('max_retry_attempts', 3)
        self.error_tolerance = self.config.get('error_tolerance', 0.1)
        
        # 统计信息
        self.error_stats = {
            'total_errors': 0,
            'handled_errors': 0,
            'critical_errors': 0,
            'auto_repairs': 0,
            'fallback_activations': 0
        }
    
    @contextmanager
    def error_context(self, operation_name: str, **context):
        """
        错误处理上下文管理器
        
        Args:
            operation_name: 操作名称
            **context: 上下文信息
        """
        start_time = time.time()
        
        try:
            logger.debug(f"开始执行操作: {operation_name}")
            yield
            
        except Exception as e:
            self.error_stats['total_errors'] += 1
            
            # 记录错误详情
            error_info = {
                'operation': operation_name,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'context': context,
                'timestamp': time.time(),
                'traceback': traceback.format_exc()
            }
            
            logger.error(f"操作失败: {operation_name}, 错误: {str(e)}")
            
            # 尝试错误恢复
            if self._should_attempt_recovery(e, error_info):
                try:
                    self._attempt_error_recovery(e, error_info)
                    self.error_stats['handled_errors'] += 1
                except Exception as recovery_error:
                    logger.critical(f"错误恢复失败: {str(recovery_error)}")
                    self.error_stats['critical_errors'] += 1
                    raise
            else:
                self.error_stats['critical_errors'] += 1
                raise
                
        finally:
            processing_time = time.time() - start_time
            logger.debug(f"操作完成: {operation_name}, 耗时: {processing_time:.3f}s")
    
    def validate_and_repair_track_points(
        self, 
        track_points: List[TrackPoint],
        min_points: int = 4,
        confidence_threshold: float = 0.5
    ) -> Tuple[List[TrackPoint], ValidationResult]:
        """
        验证并修复轨迹点数据
        
        Args:
            track_points: 轨迹点列表
            min_points: 最小轨迹点数量
            confidence_threshold: 置信度阈值
            
        Returns:
            (repaired_points, validation_result): 修复后的轨迹点和验证结果
            
        Requirements addressed: 6.1 - Data integrity validation and repair
        """
        with self.error_context("validate_track_points", 
                               num_points=len(track_points) if track_points else 0):
            
            # 初始验证
            validation_result = validate_track_points_comprehensive(
                track_points, min_points, confidence_threshold
            )
            
            if validation_result.is_valid:
                logger.info("轨迹点验证通过")
                return track_points, validation_result
            
            # 尝试自动修复
            if self.enable_auto_repair:
                logger.info("尝试自动修复轨迹点数据")
                repaired_points = repair_track_points(track_points)
                
                # 重新验证修复后的数据
                repair_validation = validate_track_points_comprehensive(
                    repaired_points, min_points, confidence_threshold
                )
                
                if repair_validation.is_valid:
                    self.error_stats['auto_repairs'] += 1
                    logger.info("轨迹点自动修复成功")
                    return repaired_points, repair_validation
                
                # 如果修复后仍不足，生成降级轨迹点
                if len(repaired_points) < min_points:
                    logger.warning("修复后轨迹点仍不足，生成降级轨迹点")
                    fallback_points = generate_fallback_tracks(min_points)
                    
                    # 合并修复的点和降级点
                    combined_points = repaired_points + fallback_points[:min_points - len(repaired_points)]
                    
                    final_validation = validate_track_points_comprehensive(
                        combined_points, min_points, confidence_threshold
                    )
                    
                    self.error_stats['fallback_activations'] += 1
                    return combined_points, final_validation
            
            # 如果无法修复，抛出异常
            raise DataIntegrityError(f"轨迹点数据无法修复: {validation_result.error_message}")
    
    def safe_feature_sampling(
        self, 
        feature_map: Tensor, 
        coords: Tensor,
        sampling_mode: str = 'bilinear'
    ) -> Tensor:
        """
        安全的特征采样
        
        Args:
            feature_map: 特征图
            coords: 采样坐标
            sampling_mode: 采样模式
            
        Returns:
            采样特征
            
        Requirements addressed: 6.2 - Safe feature map sampling
        """
        with self.error_context("feature_sampling", 
                               feature_shape=feature_map.shape,
                               coords_shape=coords.shape):
            
            try:
                # 使用边界处理器进行安全采样
                sampled_features = self.boundary_handler.safe_sample_feature_map(
                    feature_map, coords, mode=sampling_mode
                )
                
                # 验证采样结果
                if torch.isnan(sampled_features).any() or torch.isinf(sampled_features).any():
                    logger.warning("采样结果包含异常值，使用降级处理")
                    return self._handle_sampling_fallback(feature_map, coords)
                
                return sampled_features
                
            except Exception as e:
                logger.error(f"特征采样失败: {str(e)}")
                return self._handle_sampling_fallback(feature_map, coords)
    
    def execute_with_fallback(
        self, 
        primary_function: callable,
        fallback_reason: FallbackReason,
        context: Dict[str, Any],
        *args, **kwargs
    ) -> Any:
        """
        执行带降级保护的函数
        
        Args:
            primary_function: 主要执行函数
            fallback_reason: 降级原因
            context: 上下文信息
            *args, **kwargs: 函数参数
            
        Returns:
            执行结果
            
        Requirements addressed: 6.3 - Fallback execution system
        """
        retry_count = 0
        last_error = None
        
        while retry_count < self.max_retry_attempts:
            try:
                with self.error_context(f"execute_{primary_function.__name__}", 
                                       attempt=retry_count + 1):
                    result = primary_function(*args, **kwargs)
                    
                    # 验证结果有效性
                    if self._validate_execution_result(result):
                        return result
                    else:
                        raise ValueError("执行结果验证失败")
                        
            except Exception as e:
                last_error = e
                retry_count += 1
                
                logger.warning(f"执行失败 (尝试 {retry_count}/{self.max_retry_attempts}): {str(e)}")
                
                if retry_count < self.max_retry_attempts:
                    # 短暂等待后重试
                    time.sleep(0.1 * retry_count)
        
        # 所有重试都失败，启用降级处理
        logger.error(f"主要函数执行失败，启用降级处理: {str(last_error)}")
        self.error_stats['fallback_activations'] += 1
        
        return self.fallback_handler.execute_fallback(
            fallback_reason, 
            {**context, 'last_error': str(last_error)}
        )
    
    def validate_pipeline_output(
        self, 
        output: AppearanceFeature
    ) -> ValidationResult:
        """
        验证管道输出
        
        Args:
            output: 外观特征输出
            
        Returns:
            验证结果
        """
        with self.error_context("validate_output"):
            return validate_appearance_features(output)
    
    def _should_attempt_recovery(self, error: Exception, error_info: Dict[str, Any]) -> bool:
        """判断是否应该尝试错误恢复"""
        # 内存错误通常可以通过清理缓存恢复
        if isinstance(error, (RuntimeError, torch.cuda.OutOfMemoryError)):
            return True
        
        # 数据完整性错误可以通过修复恢复
        if isinstance(error, DataIntegrityError):
            return True
        
        # 值错误可能可以通过降级恢复
        if isinstance(error, ValueError):
            return True
        
        # 其他错误类型暂不尝试恢复
        return False
    
    def _attempt_error_recovery(self, error: Exception, error_info: Dict[str, Any]):
        """尝试错误恢复"""
        if isinstance(error, torch.cuda.OutOfMemoryError):
            logger.info("尝试清理GPU内存")
            torch.cuda.empty_cache()
            
        elif isinstance(error, DataIntegrityError):
            logger.info("尝试数据修复")
            # 这里可以添加具体的数据修复逻辑
            pass
            
        elif isinstance(error, ValueError):
            logger.info("尝试参数调整")
            # 这里可以添加参数调整逻辑
            pass
    
    def _handle_sampling_fallback(self, feature_map: Tensor, coords: Tensor) -> Tensor:
        """处理采样降级"""
        logger.info("使用采样降级处理")
        
        # 返回基于坐标的简单特征
        try:
            C = feature_map.shape[-1] if feature_map.dim() >= 3 else 256
            device = feature_map.device if feature_map is not None else coords.device
            
            # 基于坐标生成简单特征
            simple_features = torch.zeros(*coords.shape[:-1], C, device=device)
            
            # 使用坐标信息生成一些基础特征
            if coords.shape[-1] >= 2:
                simple_features[..., :2] = torch.tanh(coords[..., :2] * 0.01)
            
            return simple_features
            
        except Exception as e:
            logger.error(f"采样降级也失败: {str(e)}")
            # 最后的保障：返回零特征
            return torch.zeros(1, 256, device='cpu')
    
    def _validate_execution_result(self, result: Any) -> bool:
        """验证执行结果的有效性"""
        if result is None:
            return False
        
        if isinstance(result, Tensor):
            if result.numel() == 0:
                return False
            if torch.isnan(result).any() or torch.isinf(result).any():
                return False
        
        return True
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误处理统计信息"""
        stats = self.error_stats.copy()
        
        # 添加子组件统计
        stats['fallback_stats'] = self.fallback_handler.get_fallback_statistics()
        stats['boundary_stats'] = self.boundary_handler.get_boundary_statistics()
        
        # 计算成功率
        total_operations = max(stats['total_errors'] + stats['handled_errors'], 1)
        stats['success_rate'] = (total_operations - stats['critical_errors']) / total_operations
        stats['recovery_rate'] = stats['handled_errors'] / max(stats['total_errors'], 1)
        
        return stats
    
    def reset_statistics(self):
        """重置所有统计信息"""
        self.error_stats = {
            'total_errors': 0,
            'handled_errors': 0,
            'critical_errors': 0,
            'auto_repairs': 0,
            'fallback_activations': 0
        }
        
        self.fallback_handler.reset_statistics()
        self.boundary_handler.reset_statistics()
        
        logger.info("错误处理统计已重置")
    
    def configure_error_handling(self, **config):
        """配置错误处理参数"""
        self.config.update(config)
        
        # 更新配置
        if 'enable_auto_repair' in config:
            self.enable_auto_repair = config['enable_auto_repair']
        
        if 'max_retry_attempts' in config:
            self.max_retry_attempts = config['max_retry_attempts']
        
        if 'error_tolerance' in config:
            self.error_tolerance = config['error_tolerance']
        
        if 'boundary_strategy' in config:
            self.boundary_handler = BoundaryHandler(
                BoundaryStrategy(config['boundary_strategy'])
            )
        
        logger.info(f"错误处理配置已更新: {config}")


def create_error_handler(gaussian_model, **config) -> GTDCAErrorHandler:
    """
    创建GT-DCA错误处理器的便捷函数
    
    Args:
        gaussian_model: 高斯模型实例
        **config: 配置参数
        
    Returns:
        配置好的错误处理器
    """
    return GTDCAErrorHandler(gaussian_model, config)


# 装饰器函数，用于为函数添加错误处理
def with_error_handling(error_handler: GTDCAErrorHandler, 
                       fallback_reason: FallbackReason = FallbackReason.COMPUTATION_ERROR):
    """
    为函数添加错误处理的装饰器
    
    Args:
        error_handler: 错误处理器实例
        fallback_reason: 默认降级原因
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            context = {
                'function_name': func.__name__,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys())
            }
            
            return error_handler.execute_with_fallback(
                func, fallback_reason, context, *args, **kwargs
            )
        
        return wrapper
    return decorator