"""
Boundary Handling Utilities
边界处理工具

Provides safe sampling functions and boundary handling strategies for feature maps.
Ensures robust operation when sampling coordinates are near or outside image boundaries.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, Union, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BoundaryStrategy(Enum):
    """边界处理策略枚举"""
    CLAMP = "clamp"              # 裁剪到边界
    REFLECT = "reflect"          # 反射填充
    REPLICATE = "replicate"      # 复制边界值
    ZERO = "zero"               # 零填充
    CIRCULAR = "circular"        # 循环填充


class BoundaryHandler:
    """
    边界处理器
    
    Handles various boundary conditions when sampling from 2D feature maps.
    Provides multiple strategies for dealing with out-of-bounds coordinates.
    
    Requirements addressed: 6.2 - Safe feature map sampling and boundary handling
    """
    
    def __init__(self, strategy: BoundaryStrategy = BoundaryStrategy.CLAMP):
        """
        初始化边界处理器
        
        Args:
            strategy: 边界处理策略
        """
        self.strategy = strategy
        self.stats = {
            "total_samples": 0,
            "boundary_violations": 0,
            "clipped_coordinates": 0
        }
    
    def safe_sample_feature_map(
        self, 
        feature_map: Tensor, 
        coords: Tensor,
        mode: str = 'bilinear',
        align_corners: bool = True
    ) -> Tensor:
        """
        安全的特征图采样函数
        
        Args:
            feature_map: 特征图张量 (B, C, H, W) 或 (C, H, W)
            coords: 采样坐标 (..., 2) 格式为 (x, y)
            mode: 插值模式 ('bilinear', 'nearest')
            align_corners: 是否对齐角点
            
        Returns:
            采样得到的特征 (..., C)
            
        Requirements addressed: 6.2 - Safe feature map sampling
        """
        self.stats["total_samples"] += coords.numel() // 2
        
        # 确保特征图是4D格式 (B, C, H, W)
        if feature_map.dim() == 3:
            feature_map = feature_map.unsqueeze(0)  # (1, C, H, W)
            batch_added = True
        else:
            batch_added = False
        
        B, C, H, W = feature_map.shape
        
        # 处理边界情况
        processed_coords, boundary_mask = self._handle_boundary_coordinates(
            coords, (W, H)
        )
        
        # 转换坐标到grid_sample格式 [-1, 1]
        grid_coords = self._normalize_coordinates(processed_coords, (W, H))
        
        # 确保grid坐标格式正确
        original_shape = grid_coords.shape[:-1]
        grid_coords = grid_coords.view(1, -1, 1, 2)  # (1, N, 1, 2)
        
        try:
            # 执行采样
            sampled_features = F.grid_sample(
                feature_map, 
                grid_coords, 
                mode=mode, 
                padding_mode='zeros',
                align_corners=align_corners
            )  # (B, C, N, 1)
            
            # 重塑输出
            sampled_features = sampled_features.squeeze(-1).transpose(1, 2)  # (B, N, C)
            sampled_features = sampled_features.view(*original_shape, C)
            
            # 如果原来是3D特征图，移除batch维度
            if batch_added:
                sampled_features = sampled_features.squeeze(0)
            
            # 处理边界违规的采样点
            if boundary_mask is not None:
                sampled_features = self._apply_boundary_mask(
                    sampled_features, boundary_mask
                )
            
            return sampled_features
            
        except Exception as e:
            logger.error(f"特征图采样失败: {str(e)}")
            # 返回零特征作为降级方案
            output_shape = list(coords.shape[:-1]) + [C]
            return torch.zeros(output_shape, device=feature_map.device, dtype=feature_map.dtype)
    
    def _handle_boundary_coordinates(
        self, 
        coords: Tensor, 
        image_size: Tuple[int, int]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        处理边界坐标
        
        Args:
            coords: 原始坐标 (..., 2)
            image_size: 图像尺寸 (width, height)
            
        Returns:
            (processed_coords, boundary_mask): 处理后的坐标和边界掩码
        """
        width, height = image_size
        processed_coords = coords.clone()
        boundary_mask = None
        
        # 检测边界违规
        x_out = (coords[..., 0] < 0) | (coords[..., 0] >= width)
        y_out = (coords[..., 1] < 0) | (coords[..., 1] >= height)
        boundary_violations = x_out | y_out
        
        if boundary_violations.any():
            self.stats["boundary_violations"] += boundary_violations.sum().item()
            boundary_mask = boundary_violations
            
            if self.strategy == BoundaryStrategy.CLAMP:
                processed_coords[..., 0] = torch.clamp(coords[..., 0], 0, width - 1)
                processed_coords[..., 1] = torch.clamp(coords[..., 1], 0, height - 1)
                self.stats["clipped_coordinates"] += boundary_violations.sum().item()
                
            elif self.strategy == BoundaryStrategy.REFLECT:
                # 反射边界处理
                processed_coords[..., 0] = self._reflect_coordinate(coords[..., 0], width)
                processed_coords[..., 1] = self._reflect_coordinate(coords[..., 1], height)
                
            elif self.strategy == BoundaryStrategy.CIRCULAR:
                # 循环边界处理
                processed_coords[..., 0] = coords[..., 0] % width
                processed_coords[..., 1] = coords[..., 1] % height
                
            # REPLICATE和ZERO策略在grid_sample中处理
        
        return processed_coords, boundary_mask
    
    def _reflect_coordinate(self, coord: Tensor, size: int) -> Tensor:
        """
        反射坐标处理
        
        Args:
            coord: 坐标值
            size: 维度大小
            
        Returns:
            反射后的坐标
        """
        # 将坐标映射到 [0, 2*size) 范围
        coord_mod = coord % (2 * size)
        
        # 反射处理
        reflected = torch.where(
            coord_mod < size,
            coord_mod,
            2 * size - 1 - coord_mod
        )
        
        return torch.clamp(reflected, 0, size - 1)
    
    def _normalize_coordinates(
        self, 
        coords: Tensor, 
        image_size: Tuple[int, int]
    ) -> Tensor:
        """
        将像素坐标归一化到[-1, 1]范围
        
        Args:
            coords: 像素坐标 (..., 2)
            image_size: 图像尺寸 (width, height)
            
        Returns:
            归一化后的坐标 (..., 2)
        """
        width, height = image_size
        
        # 归一化到[-1, 1]
        normalized_coords = coords.clone()
        normalized_coords[..., 0] = 2.0 * coords[..., 0] / (width - 1) - 1.0
        normalized_coords[..., 1] = 2.0 * coords[..., 1] / (height - 1) - 1.0
        
        return normalized_coords
    
    def _apply_boundary_mask(
        self, 
        features: Tensor, 
        boundary_mask: Tensor
    ) -> Tensor:
        """
        应用边界掩码处理
        
        Args:
            features: 采样特征
            boundary_mask: 边界违规掩码
            
        Returns:
            处理后的特征
        """
        if self.strategy == BoundaryStrategy.ZERO:
            # 将边界违规的采样点设为零
            features = features.clone()
            features[boundary_mask] = 0.0
        
        return features
    
    def create_padded_feature_map(
        self, 
        feature_map: Tensor, 
        padding: Union[int, Tuple[int, int, int, int]] = 1,
        mode: str = 'replicate'
    ) -> Tensor:
        """
        创建填充的特征图以避免边界问题
        
        Args:
            feature_map: 原始特征图 (C, H, W) 或 (B, C, H, W)
            padding: 填充大小，可以是单个值或(left, right, top, bottom)
            mode: 填充模式 ('constant', 'reflect', 'replicate', 'circular')
            
        Returns:
            填充后的特征图
            
        Requirements addressed: 6.2 - Boundary padding strategies
        """
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        
        # 确保特征图是4D格式
        if feature_map.dim() == 3:
            feature_map = feature_map.unsqueeze(0)
            batch_added = True
        else:
            batch_added = False
        
        # 应用填充
        padded_map = F.pad(feature_map, padding, mode=mode)
        
        # 如果原来是3D，移除batch维度
        if batch_added:
            padded_map = padded_map.squeeze(0)
        
        return padded_map
    
    def adjust_coordinates_for_padding(
        self, 
        coords: Tensor, 
        padding: Union[int, Tuple[int, int, int, int]]
    ) -> Tensor:
        """
        调整坐标以适应填充后的特征图
        
        Args:
            coords: 原始坐标 (..., 2)
            padding: 填充大小
            
        Returns:
            调整后的坐标
        """
        if isinstance(padding, int):
            left = top = padding
        else:
            left, right, top, bottom = padding
        
        adjusted_coords = coords.clone()
        adjusted_coords[..., 0] += left   # x坐标偏移
        adjusted_coords[..., 1] += top    # y坐标偏移
        
        return adjusted_coords
    
    def get_boundary_statistics(self) -> dict:
        """
        获取边界处理统计信息
        
        Returns:
            统计信息字典
        """
        total_samples = max(self.stats["total_samples"], 1)
        
        return {
            "total_samples": self.stats["total_samples"],
            "boundary_violations": self.stats["boundary_violations"],
            "violation_rate": self.stats["boundary_violations"] / total_samples,
            "clipped_coordinates": self.stats["clipped_coordinates"],
            "strategy": self.strategy.value
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            "total_samples": 0,
            "boundary_violations": 0,
            "clipped_coordinates": 0
        }


def safe_bilinear_sample(
    feature_map: Tensor, 
    coords: Tensor,
    boundary_strategy: BoundaryStrategy = BoundaryStrategy.CLAMP
) -> Tensor:
    """
    安全的双线性插值采样函数
    
    Args:
        feature_map: 特征图 (C, H, W)
        coords: 采样坐标 (..., 2)
        boundary_strategy: 边界处理策略
        
    Returns:
        采样特征 (..., C)
        
    Requirements addressed: 6.2 - Safe bilinear sampling with boundary handling
    """
    handler = BoundaryHandler(boundary_strategy)
    return handler.safe_sample_feature_map(feature_map, coords, mode='bilinear')


def safe_nearest_sample(
    feature_map: Tensor, 
    coords: Tensor,
    boundary_strategy: BoundaryStrategy = BoundaryStrategy.CLAMP
) -> Tensor:
    """
    安全的最近邻采样函数
    
    Args:
        feature_map: 特征图 (C, H, W)
        coords: 采样坐标 (..., 2)
        boundary_strategy: 边界处理策略
        
    Returns:
        采样特征 (..., C)
    """
    handler = BoundaryHandler(boundary_strategy)
    return handler.safe_sample_feature_map(feature_map, coords, mode='nearest')


def create_safe_sampling_grid(
    coords: Tensor, 
    image_size: Tuple[int, int],
    margin: float = 0.1
) -> Tensor:
    """
    创建安全的采样网格，确保坐标在安全边界内
    
    Args:
        coords: 原始坐标 (..., 2)
        image_size: 图像尺寸 (width, height)
        margin: 安全边界比例 (0.0-0.5)
        
    Returns:
        安全的采样坐标
    """
    width, height = image_size
    
    # 计算安全边界
    safe_margin_x = width * margin
    safe_margin_y = height * margin
    
    safe_coords = coords.clone()
    safe_coords[..., 0] = torch.clamp(
        coords[..., 0], 
        safe_margin_x, 
        width - safe_margin_x - 1
    )
    safe_coords[..., 1] = torch.clamp(
        coords[..., 1], 
        safe_margin_y, 
        height - safe_margin_y - 1
    )
    
    return safe_coords


def validate_sampling_coordinates(
    coords: Tensor, 
    image_size: Tuple[int, int],
    tolerance: float = 0.1
) -> Tuple[bool, str, Tensor]:
    """
    验证采样坐标的有效性
    
    Args:
        coords: 采样坐标 (..., 2)
        image_size: 图像尺寸 (width, height)
        tolerance: 容忍的边界外距离
        
    Returns:
        (is_valid, message, valid_mask): 验证结果、消息和有效性掩码
    """
    width, height = image_size
    
    # 检查坐标范围
    x_valid = (coords[..., 0] >= -tolerance) & (coords[..., 0] < width + tolerance)
    y_valid = (coords[..., 1] >= -tolerance) & (coords[..., 1] < height + tolerance)
    valid_mask = x_valid & y_valid
    
    # 检查NaN和Inf
    finite_mask = torch.isfinite(coords).all(dim=-1)
    valid_mask = valid_mask & finite_mask
    
    valid_ratio = valid_mask.float().mean().item()
    
    if valid_ratio < 0.5:
        return False, f"有效坐标比例过低: {valid_ratio:.2%}", valid_mask
    elif valid_ratio < 1.0:
        invalid_count = (~valid_mask).sum().item()
        return True, f"部分坐标无效: {invalid_count} 个", valid_mask
    else:
        return True, "所有坐标有效", valid_mask