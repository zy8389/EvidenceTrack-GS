"""
Validation Utilities
验证工具函数

Provides validation functions for GT-DCA data structures and inputs.
Ensures data integrity and helps with error handling.
"""

from typing import List, Tuple, Optional, Dict, Any
import torch
from torch import Tensor
import logging

from ..core.data_structures import TrackPoint, SamplingPoint, AppearanceFeature

logger = logging.getLogger(__name__)


class DataIntegrityError(Exception):
    """轨迹点数据不完整或损坏"""
    pass


class ValidationResult:
    """验证结果类"""
    def __init__(self, is_valid: bool, error_message: str = "", warnings: List[str] = None):
        self.is_valid = is_valid
        self.error_message = error_message
        self.warnings = warnings or []
        
    def add_warning(self, warning: str):
        """添加警告信息"""
        self.warnings.append(warning)
        
    def __bool__(self):
        return self.is_valid


def validate_track_points(track_points: List[TrackPoint]) -> Tuple[bool, str]:
    """
    验证轨迹点数据的完整性
    
    Args:
        track_points: 轨迹点列表
        
    Returns:
        (is_valid, error_message): 验证结果和错误信息
    """
    if not track_points:
        return False, "轨迹点列表为空"
    
    for i, point in enumerate(track_points):
        if not point.is_valid():
            return False, f"轨迹点 {i} 无效: ID={point.point_id}"
        
        # 检查坐标范围合理性
        if abs(point.x) > 10000 or abs(point.y) > 10000:
            return False, f"轨迹点 {i} 坐标超出合理范围: ({point.x}, {point.y})"
    
    return True, ""


def validate_feature_map(feature_map: Tensor) -> Tuple[bool, str]:
    """
    验证2D特征图的有效性
    
    Args:
        feature_map: 特征图张量 (H, W, C)
        
    Returns:
        (is_valid, error_message): 验证结果和错误信息
    """
    if feature_map is None:
        return False, "特征图为None"
    
    if feature_map.dim() != 3:
        return False, f"特征图维度错误，期望3维，得到{feature_map.dim()}维"
    
    if feature_map.numel() == 0:
        return False, "特征图为空"
    
    if torch.isnan(feature_map).any():
        return False, "特征图包含NaN值"
    
    if torch.isinf(feature_map).any():
        return False, "特征图包含Inf值"
    
    return True, ""


def validate_coordinates(coords: Tensor, bounds: Optional[Tuple[int, int]] = None) -> Tuple[bool, str]:
    """
    验证坐标张量的有效性
    
    Args:
        coords: 坐标张量 (..., 2)
        bounds: 可选的边界限制 (width, height)
        
    Returns:
        (is_valid, error_message): 验证结果和错误信息
    """
    if coords is None:
        return False, "坐标张量为None"
    
    if coords.shape[-1] != 2:
        return False, f"坐标张量最后一维应为2，得到{coords.shape[-1]}"
    
    if torch.isnan(coords).any():
        return False, "坐标包含NaN值"
    
    if torch.isinf(coords).any():
        return False, "坐标包含Inf值"
    
    if bounds is not None:
        width, height = bounds
        if (coords[..., 0] < 0).any() or (coords[..., 0] >= width).any():
            return False, f"x坐标超出边界 [0, {width})"
        
        if (coords[..., 1] < 0).any() or (coords[..., 1] >= height).any():
            return False, f"y坐标超出边界 [0, {height})"
    
    return True, ""


def validate_track_points_comprehensive(
    track_points: List[TrackPoint], 
    min_points: int = 4,
    confidence_threshold: float = 0.5
) -> ValidationResult:
    """
    全面验证轨迹点数据完整性
    
    Args:
        track_points: 轨迹点列表
        min_points: 最小轨迹点数量
        confidence_threshold: 置信度阈值
        
    Returns:
        ValidationResult: 详细的验证结果
        
    Requirements addressed: 6.1 - Track point data integrity validation
    """
    result = ValidationResult(True)
    
    # 基本完整性检查
    if not track_points:
        return ValidationResult(False, "轨迹点列表为空")
    
    if len(track_points) < min_points:
        return ValidationResult(
            False, 
            f"轨迹点数量不足: {len(track_points)} < {min_points}"
        )
    
    # 详细验证每个轨迹点
    valid_points = 0
    low_confidence_count = 0
    duplicate_ids = set()
    seen_ids = set()
    
    for i, point in enumerate(track_points):
        try:
            # 检查基本有效性
            if not point.is_valid():
                result.add_warning(f"轨迹点 {i} (ID={point.point_id}) 基本验证失败")
                continue
            
            # 检查置信度
            if point.confidence < confidence_threshold:
                low_confidence_count += 1
                result.add_warning(
                    f"轨迹点 {i} (ID={point.point_id}) 置信度过低: {point.confidence:.3f}"
                )
            
            # 检查ID重复
            if point.point_id in seen_ids:
                duplicate_ids.add(point.point_id)
            else:
                seen_ids.add(point.point_id)
            
            # 检查坐标合理性
            if abs(point.x) > 10000 or abs(point.y) > 10000:
                result.add_warning(
                    f"轨迹点 {i} (ID={point.point_id}) 坐标可能异常: ({point.x:.2f}, {point.y:.2f})"
                )
            
            valid_points += 1
            
        except Exception as e:
            result.add_warning(f"轨迹点 {i} 验证时发生异常: {str(e)}")
    
    # 汇总检查结果
    if duplicate_ids:
        result.add_warning(f"发现重复ID: {list(duplicate_ids)}")
    
    if low_confidence_count > len(track_points) * 0.5:
        result.add_warning(f"超过50%的轨迹点置信度过低 ({low_confidence_count}/{len(track_points)})")
    
    # 最终有效性判断
    if valid_points < min_points:
        result.is_valid = False
        result.error_message = f"有效轨迹点数量不足: {valid_points} < {min_points}"
    
    logger.info(f"轨迹点验证完成: {valid_points}/{len(track_points)} 有效, {len(result.warnings)} 个警告")
    
    return result


def repair_track_points(track_points: List[TrackPoint]) -> List[TrackPoint]:
    """
    修复不完整的轨迹点数据
    
    Args:
        track_points: 原始轨迹点列表
        
    Returns:
        修复后的轨迹点列表
        
    Requirements addressed: 6.1 - Incomplete data handling and repair
    """
    if not track_points:
        logger.warning("轨迹点列表为空，无法修复")
        return []
    
    repaired_points = []
    removed_count = 0
    repaired_count = 0
    
    for i, point in enumerate(track_points):
        try:
            # 尝试修复坐标异常
            x, y = point.coordinates_2d
            
            # 修复极端坐标值
            if abs(x) > 10000:
                x = max(-10000, min(10000, x))
                repaired_count += 1
                logger.debug(f"修复轨迹点 {i} 的x坐标: {point.x} -> {x}")
            
            if abs(y) > 10000:
                y = max(-10000, min(10000, y))
                repaired_count += 1
                logger.debug(f"修复轨迹点 {i} 的y坐标: {point.y} -> {y}")
            
            # 修复置信度异常
            confidence = point.confidence
            if confidence < 0:
                confidence = 0.0
                repaired_count += 1
            elif confidence > 1:
                confidence = 1.0
                repaired_count += 1
            
            # 创建修复后的轨迹点
            repaired_point = TrackPoint(
                point_id=point.point_id,
                coordinates_2d=(x, y),
                confidence=confidence,
                frame_id=point.frame_id,
                feature_descriptor=point.feature_descriptor
            )
            
            # 最终验证修复后的点
            if repaired_point.is_valid():
                repaired_points.append(repaired_point)
            else:
                removed_count += 1
                logger.debug(f"移除无法修复的轨迹点 {i} (ID={point.point_id})")
                
        except Exception as e:
            removed_count += 1
            logger.warning(f"修复轨迹点 {i} 时发生异常，已移除: {str(e)}")
    
    logger.info(f"轨迹点修复完成: 修复 {repaired_count} 个, 移除 {removed_count} 个, 保留 {len(repaired_points)} 个")
    
    return repaired_points


def generate_fallback_tracks(
    num_points: int = 8, 
    image_size: Tuple[int, int] = (640, 480)
) -> List[TrackPoint]:
    """
    生成降级用的轨迹点
    
    Args:
        num_points: 生成的轨迹点数量
        image_size: 图像尺寸 (width, height)
        
    Returns:
        生成的轨迹点列表
        
    Requirements addressed: 6.1 - Fallback track generation for incomplete data
    """
    width, height = image_size
    fallback_points = []
    
    # 在图像中心区域生成规则分布的轨迹点
    margin_x = width * 0.2
    margin_y = height * 0.2
    
    effective_width = width - 2 * margin_x
    effective_height = height - 2 * margin_y
    
    # 计算网格布局
    cols = int(torch.sqrt(torch.tensor(num_points)).item())
    rows = (num_points + cols - 1) // cols
    
    point_id = 0
    for row in range(rows):
        for col in range(cols):
            if point_id >= num_points:
                break
                
            # 计算坐标
            x = margin_x + (col + 0.5) * effective_width / cols
            y = margin_y + (row + 0.5) * effective_height / rows
            
            # 添加轻微随机扰动
            x += torch.randn(1).item() * 5.0
            y += torch.randn(1).item() * 5.0
            
            # 确保坐标在有效范围内
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            
            fallback_point = TrackPoint(
                point_id=point_id,
                coordinates_2d=(x, y),
                confidence=0.8,  # 中等置信度
                frame_id=0
            )
            
            fallback_points.append(fallback_point)
            point_id += 1
    
    logger.info(f"生成了 {len(fallback_points)} 个降级轨迹点")
    return fallback_points


def validate_sampling_points(sampling_points: List[SamplingPoint]) -> ValidationResult:
    """
    验证采样点数据完整性
    
    Args:
        sampling_points: 采样点列表
        
    Returns:
        ValidationResult: 验证结果
    """
    result = ValidationResult(True)
    
    if not sampling_points:
        return ValidationResult(False, "采样点列表为空")
    
    valid_count = 0
    total_weight = 0.0
    
    for i, point in enumerate(sampling_points):
        if point.is_valid():
            valid_count += 1
            total_weight += point.weight
        else:
            result.add_warning(f"采样点 {i} 无效")
    
    # 检查权重归一化
    if abs(total_weight - 1.0) > 1e-6:
        result.add_warning(f"采样点权重和不为1: {total_weight:.6f}")
    
    if valid_count == 0:
        result.is_valid = False
        result.error_message = "没有有效的采样点"
    
    return result


def validate_appearance_features(features: AppearanceFeature) -> ValidationResult:
    """
    验证外观特征数据完整性
    
    Args:
        features: 外观特征对象
        
    Returns:
        ValidationResult: 验证结果
    """
    result = ValidationResult(True)
    
    try:
        # 检查特征张量
        for name, tensor in [
            ("base_features", features.base_features),
            ("geometry_guided_features", features.geometry_guided_features),
            ("enhanced_features", features.enhanced_features)
        ]:
            if tensor is None:
                result.is_valid = False
                result.error_message = f"{name} 为 None"
                return result
            
            if torch.isnan(tensor).any():
                result.add_warning(f"{name} 包含 NaN 值")
            
            if torch.isinf(tensor).any():
                result.add_warning(f"{name} 包含 Inf 值")
        
        # 检查采样元数据
        for i, samples in enumerate(features.sampling_metadata):
            sample_result = validate_sampling_points(samples)
            if not sample_result.is_valid:
                result.add_warning(f"高斯点 {i} 的采样点无效: {sample_result.error_message}")
    
    except Exception as e:
        result.is_valid = False
        result.error_message = f"验证外观特征时发生异常: {str(e)}"
    
    return result