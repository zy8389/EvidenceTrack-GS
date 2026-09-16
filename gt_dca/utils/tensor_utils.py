"""
Tensor Utilities
张量工具函数

Common tensor operations and utilities for GT-DCA processing.
"""

import torch
from torch import Tensor
from typing import List, Callable, Any
import torch.nn.functional as F


def safe_normalize(tensor: Tensor, dim: int = -1, eps: float = 1e-8) -> Tensor:
    """
    安全的张量归一化，避免除零错误
    
    Args:
        tensor: 输入张量
        dim: 归一化维度
        eps: 防止除零的小值
        
    Returns:
        归一化后的张量
    """
    norm = torch.norm(tensor, dim=dim, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return tensor / norm


def safe_sampling(
    feature_map: Tensor, 
    coords: Tensor, 
    mode: str = 'bilinear',
    padding_mode: str = 'border'
) -> Tensor:
    """
    安全的特征图采样，处理边界情况
    
    Args:
        feature_map: 特征图 (B, C, H, W) 或 (H, W, C)
        coords: 采样坐标 (..., 2)，范围应在[-1, 1]
        mode: 插值模式
        padding_mode: 边界填充模式
        
    Returns:
        采样后的特征
    """
    # 确保特征图格式为 (B, C, H, W)
    if feature_map.dim() == 3:  # (H, W, C)
        feature_map = feature_map.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        added_batch = True
    else:
        added_batch = False
    
    # 确保坐标格式正确
    if coords.dim() == 2:  # (N, 2)
        coords = coords.unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)
        added_dims = True
    else:
        added_dims = False
    
    # 执行采样
    try:
        sampled = F.grid_sample(
            feature_map, 
            coords, 
            mode=mode, 
            padding_mode=padding_mode,
            align_corners=False
        )
        
        # 恢复原始维度
        if added_batch:
            sampled = sampled.squeeze(0)  # 移除批次维度
        
        if added_dims:
            sampled = sampled.squeeze(0).squeeze(0)  # 移除添加的维度
            
        return sampled
        
    except Exception as e:
        # 如果采样失败，返回零张量
        if added_batch:
            return torch.zeros(feature_map.shape[1], coords.shape[-2], device=feature_map.device)
        else:
            return torch.zeros(feature_map.shape[1], coords.shape[-2], device=feature_map.device)


def batch_process(
    data_list: List[Any], 
    process_fn: Callable, 
    batch_size: int = 32,
    **kwargs
) -> List[Any]:
    """
    批量处理数据列表
    
    Args:
        data_list: 待处理的数据列表
        process_fn: 处理函数
        batch_size: 批次大小
        **kwargs: 传递给处理函数的额外参数
        
    Returns:
        处理后的结果列表
    """
    results = []
    
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i + batch_size]
        batch_results = process_fn(batch, **kwargs)
        
        if isinstance(batch_results, list):
            results.extend(batch_results)
        else:
            results.append(batch_results)
    
    return results


def clamp_coordinates(coords: Tensor, bounds: tuple) -> Tensor:
    """
    将坐标限制在指定边界内
    
    Args:
        coords: 坐标张量 (..., 2)
        bounds: 边界 (width, height)
        
    Returns:
        限制后的坐标
    """
    width, height = bounds
    coords_clamped = coords.clone()
    coords_clamped[..., 0] = torch.clamp(coords_clamped[..., 0], 0, width - 1)
    coords_clamped[..., 1] = torch.clamp(coords_clamped[..., 1], 0, height - 1)
    return coords_clamped


def normalize_coordinates(coords: Tensor, bounds: tuple) -> Tensor:
    """
    将坐标归一化到[-1, 1]范围（用于grid_sample）
    
    Args:
        coords: 坐标张量 (..., 2)，范围在[0, width-1] x [0, height-1]
        bounds: 边界 (width, height)
        
    Returns:
        归一化后的坐标，范围在[-1, 1]
    """
    width, height = bounds
    normalized = coords.clone().float()
    
    # 归一化到[0, 1]
    normalized[..., 0] = normalized[..., 0] / (width - 1)
    normalized[..., 1] = normalized[..., 1] / (height - 1)
    
    # 转换到[-1, 1]
    normalized = normalized * 2.0 - 1.0
    
    return normalized