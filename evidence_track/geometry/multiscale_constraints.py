"""
多尺度几何约束实现

实现多尺度图像金字塔处理和几何约束计算
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np
from .data_structures import Trajectory, ConstraintResult, Point2D
from .interfaces import ConstraintEngine


class MultiScaleImagePyramid:
    """多尺度图像金字塔处理器"""
    
    def __init__(self, scales: List[float] = None, blur_sigma: float = 1.0):
        """
        初始化多尺度金字塔处理器
        
        Args:
            scales: 尺度列表，默认为 [1.0, 0.5, 0.25]
            blur_sigma: 高斯模糊标准差
        """
        self.scales = scales if scales is not None else [1.0, 0.5, 0.25]
        self.blur_sigma = blur_sigma
        self.pyramid_cache = {}
    
    def build_pyramid(self, image: torch.Tensor, cache_key: Optional[str] = None) -> Dict[float, torch.Tensor]:
        """
        构建图像金字塔
        
        Args:
            image: 输入图像张量 [C, H, W]
            cache_key: 缓存键，用于避免重复计算
            
        Returns:
            尺度到图像的映射字典
        """
        if cache_key and cache_key in self.pyramid_cache:
            return self.pyramid_cache[cache_key]
        
        pyramid = {}
        
        for scale in self.scales:
            if scale == 1.0:
                pyramid[scale] = image
            else:
                # 计算目标尺寸
                _, h, w = image.shape
                target_h = int(h * scale)
                target_w = int(w * scale)
                
                # 先进行高斯模糊以避免混叠
                if self.blur_sigma > 0:
                    kernel_size = int(2 * np.ceil(2 * self.blur_sigma) + 1)
                    blurred = self._gaussian_blur(image, kernel_size, self.blur_sigma)
                else:
                    blurred = image
                
                # 下采样
                scaled_image = F.interpolate(
                    blurred.unsqueeze(0), 
                    size=(target_h, target_w), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                
                pyramid[scale] = scaled_image
        
        if cache_key:
            self.pyramid_cache[cache_key] = pyramid
        
        return pyramid
    
    def _gaussian_blur(self, image: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        """
        应用高斯模糊
        
        Args:
            image: 输入图像
            kernel_size: 核大小
            sigma: 标准差
            
        Returns:
            模糊后的图像
        """
        # 创建高斯核
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        
        # 应用分离的高斯核
        kernel_1d = g.view(1, 1, -1).to(image.device)
        
        # 水平方向
        blurred = F.conv1d(
            image.view(-1, 1, image.size(-1)), 
            kernel_1d, 
            padding=kernel_size // 2
        ).view(image.shape)
        
        # 垂直方向
        blurred = F.conv1d(
            blurred.transpose(-1, -2).contiguous().view(-1, 1, image.size(-2)), 
            kernel_1d, 
            padding=kernel_size // 2
        ).view(*blurred.transpose(-1, -2).shape).transpose(-1, -2)
        
        return blurred
    
    def scale_coordinates(self, points: torch.Tensor, scale: float) -> torch.Tensor:
        """
        缩放坐标点
        
        Args:
            points: 原始坐标点 [N, 2]
            scale: 缩放因子
            
        Returns:
            缩放后的坐标点
        """
        return points * scale
    
    def clear_cache(self):
        """清空金字塔缓存"""
        self.pyramid_cache.clear()


class MultiScaleConstraints:
    """多尺度几何约束计算器"""
    
    def __init__(self, 
                 scales: List[float] = None,
                 scale_weights: Dict[float, float] = None,
                 reprojection_threshold: float = 2.0):
        """
        初始化多尺度约束计算器
        
        Args:
            scales: 尺度列表
            scale_weights: 各尺度的权重
            reprojection_threshold: 重投影误差阈值
        """
        self.scales = scales if scales is not None else [1.0, 0.5, 0.25]
        self.scale_weights = scale_weights if scale_weights is not None else {
            1.0: 1.0, 0.5: 0.7, 0.25: 0.5
        }
        self.reprojection_threshold = reprojection_threshold
        self.pyramid_processor = MultiScaleImagePyramid(scales)
    
    def compute_multiscale_reprojection_loss(self,
                                           trajectories: List[Trajectory],
                                           cameras: List,
                                           gaussian_points: torch.Tensor,
                                           images: Optional[List[torch.Tensor]] = None) -> ConstraintResult:
        """
        计算多尺度重投影损失
        
        Args:
            trajectories: 特征轨迹列表
            cameras: 相机参数列表
            gaussian_points: 高斯点云
            images: 图像列表（可选，用于构建金字塔）
            
        Returns:
            多尺度约束结果
        """
        total_loss = torch.tensor(0.0, requires_grad=True)
        individual_errors = []
        outlier_masks = []
        scale_contributions = {}
        
        for scale in self.scales:
            scale_loss, scale_errors, scale_outliers = self._compute_scale_loss(
                trajectories, cameras, gaussian_points, scale
            )
            
            # 应用尺度权重
            weight = self.scale_weights.get(scale, 1.0)
            weighted_loss = scale_loss * weight
            total_loss = total_loss + weighted_loss
            
            individual_errors.extend(scale_errors)
            outlier_masks.append(scale_outliers)
            scale_contributions[scale] = float(weighted_loss.item())
        
        # 合并异常值掩码
        combined_outlier_mask = torch.cat(outlier_masks) if outlier_masks else torch.tensor([])
        
        # 计算质量指标
        quality_metrics = self._compute_quality_metrics(
            individual_errors, combined_outlier_mask, scale_contributions
        )
        
        return ConstraintResult(
            loss_value=total_loss,
            individual_errors=individual_errors,
            outlier_mask=combined_outlier_mask,
            quality_metrics=quality_metrics,
            scale_contributions=scale_contributions
        )
    
    def _compute_scale_loss(self,
                          trajectories: List[Trajectory],
                          cameras: List,
                          gaussian_points: torch.Tensor,
                          scale: float) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        """
        计算特定尺度的损失
        
        Args:
            trajectories: 轨迹列表
            cameras: 相机列表
            gaussian_points: 高斯点云
            scale: 当前尺度
            
        Returns:
            (损失值, 个体误差列表, 异常值掩码)
        """
        scale_loss = torch.tensor(0.0, requires_grad=True)
        errors = []
        outlier_flags = []
        
        for traj in trajectories:
            if not traj.is_valid():
                continue
            
            # 缩放轨迹点坐标
            scaled_points = []
            for point in traj.points_2d:
                scaled_x = point.x * scale
                scaled_y = point.y * scale
                scaled_points.append(Point2D(
                    x=scaled_x, y=scaled_y, 
                    frame_id=point.frame_id,
                    detection_confidence=point.detection_confidence
                ))
            
            # 计算重投影误差
            traj_loss, traj_errors = self._compute_trajectory_reprojection_loss(
                scaled_points, traj.camera_indices, cameras, gaussian_points, scale
            )
            
            scale_loss = scale_loss + traj_loss
            errors.extend(traj_errors)
            
            # 检测异常值
            outliers = [error > self.reprojection_threshold for error in traj_errors]
            outlier_flags.extend(outliers)
        
        outlier_mask = torch.tensor(outlier_flags, dtype=torch.bool)
        return scale_loss, errors, outlier_mask  
  
    def _compute_trajectory_reprojection_loss(self,
                                            scaled_points: List[Point2D],
                                            camera_indices: List[int],
                                            cameras: List,
                                            gaussian_points: torch.Tensor,
                                            scale: float) -> Tuple[torch.Tensor, List[float]]:
        """
        计算单个轨迹的重投影损失
        
        Args:
            scaled_points: 缩放后的2D点列表
            camera_indices: 相机索引列表
            cameras: 相机参数列表
            gaussian_points: 高斯点云
            scale: 当前尺度
            
        Returns:
            (轨迹损失, 误差列表)
        """
        traj_loss = torch.tensor(0.0, requires_grad=True)
        errors = []
        
        # 简化的重投影计算（实际实现需要根据具体的相机模型和高斯点云结构调整）
        for i, (point, cam_idx) in enumerate(zip(scaled_points, camera_indices)):
            if cam_idx >= len(cameras):
                continue
            
            # 模拟重投影计算
            # 实际实现需要使用真实的相机参数和3D点
            observed = torch.tensor([point.x, point.y])
            
            # 这里使用简化的投影模型
            # 实际应该使用相机的内参和外参进行完整的3D到2D投影
            if len(gaussian_points) > i:
                projected = gaussian_points[i][:2]  # 假设前两维是投影坐标
            else:
                projected = torch.zeros(2)
            
            # 计算重投影误差
            error = torch.norm(observed - projected)
            traj_loss = traj_loss + error
            errors.append(float(error.item()))
        
        return traj_loss, errors
    
    def _compute_quality_metrics(self,
                               individual_errors: List[float],
                               outlier_mask: torch.Tensor,
                               scale_contributions: Dict[float, float]) -> Dict[str, float]:
        """
        计算质量指标
        
        Args:
            individual_errors: 个体误差列表
            outlier_mask: 异常值掩码
            scale_contributions: 各尺度贡献
            
        Returns:
            质量指标字典
        """
        metrics = {}
        
        if individual_errors:
            metrics['mean_error'] = np.mean(individual_errors)
            metrics['std_error'] = np.std(individual_errors)
            metrics['max_error'] = np.max(individual_errors)
            metrics['min_error'] = np.min(individual_errors)
        else:
            metrics.update({
                'mean_error': 0.0, 'std_error': 0.0,
                'max_error': 0.0, 'min_error': 0.0
            })
        
        if outlier_mask.numel() > 0:
            metrics['outlier_ratio'] = float(outlier_mask.sum()) / outlier_mask.numel()
        else:
            metrics['outlier_ratio'] = 0.0
        
        # 尺度贡献统计
        if scale_contributions:
            total_contribution = sum(scale_contributions.values())
            for scale, contribution in scale_contributions.items():
                metrics[f'scale_{scale}_contribution'] = contribution / total_contribution if total_contribution > 0 else 0.0
        
        return metrics
    
    def compute_scale_adaptive_weights(self,
                                     trajectories: List[Trajectory],
                                     error_history: Dict[float, List[float]] = None) -> Dict[float, float]:
        """
        计算尺度自适应权重
        
        Args:
            trajectories: 轨迹列表
            error_history: 各尺度的历史误差
            
        Returns:
            更新后的尺度权重
        """
        if not error_history:
            return self.scale_weights.copy()
        
        adaptive_weights = {}
        
        for scale in self.scales:
            if scale in error_history and error_history[scale]:
                # 基于历史误差调整权重
                recent_errors = error_history[scale][-10:]  # 最近10次的误差
                avg_error = np.mean(recent_errors)
                
                # 误差越小，权重越大
                base_weight = self.scale_weights.get(scale, 1.0)
                error_factor = 1.0 / (1.0 + avg_error)
                adaptive_weights[scale] = base_weight * error_factor
            else:
                adaptive_weights[scale] = self.scale_weights.get(scale, 1.0)
        
        # 归一化权重
        total_weight = sum(adaptive_weights.values())
        if total_weight > 0:
            for scale in adaptive_weights:
                adaptive_weights[scale] /= total_weight
        
        return adaptive_weights
    
    def update_scale_weights(self, new_weights: Dict[float, float]):
        """
        更新尺度权重
        
        Args:
            new_weights: 新的权重字典
        """
        self.scale_weights.update(new_weights)


class MultiScaleConstraintEngine(ConstraintEngine):
    """多尺度约束引擎实现"""
    
    def __init__(self, 
                 scales: List[float] = None,
                 reprojection_threshold: float = 2.0,
                 adaptive_weighting: bool = True):
        """
        初始化多尺度约束引擎
        
        Args:
            scales: 尺度列表
            reprojection_threshold: 重投影误差阈值
            adaptive_weighting: 是否启用自适应权重
        """
        self.multiscale_processor = MultiScaleConstraints(
            scales=scales, 
            reprojection_threshold=reprojection_threshold
        )
        self.adaptive_weighting = adaptive_weighting
        self.error_history = {scale: [] for scale in (scales or [1.0, 0.5, 0.25])}
    
    def compute_reprojection_constraints(self,
                                       trajectories: List[Trajectory],
                                       cameras: List,
                                       gaussian_points: torch.Tensor) -> ConstraintResult:
        """
        计算重投影约束（单尺度版本）
        
        Args:
            trajectories: 特征轨迹列表
            cameras: 相机参数列表
            gaussian_points: 高斯点云
            
        Returns:
            约束计算结果
        """
        # 使用单一尺度（1.0）进行计算
        return self.multiscale_processor.compute_multiscale_reprojection_loss(
            trajectories, cameras, gaussian_points
        )
    
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
        # 更新处理器的尺度设置
        self.multiscale_processor.scales = scales
        
        # 如果启用自适应权重，更新权重
        if self.adaptive_weighting:
            adaptive_weights = self.multiscale_processor.compute_scale_adaptive_weights(
                trajectories, self.error_history
            )
            self.multiscale_processor.update_scale_weights(adaptive_weights)
        
        # 计算多尺度约束
        result = self.multiscale_processor.compute_multiscale_reprojection_loss(
            trajectories, cameras, gaussian_points=torch.tensor([])  # 占位符
        )
        
        # 更新误差历史
        for scale, contribution in result.scale_contributions.items():
            if scale in self.error_history:
                self.error_history[scale].append(contribution)
                # 保持历史记录在合理范围内
                if len(self.error_history[scale]) > 100:
                    self.error_history[scale] = self.error_history[scale][-50:]
        
        return result
    
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
        # 基于轨迹质量和迭代次数计算权重
        weights = []
        
        for traj in trajectories:
            if not traj.is_valid():
                weights.append(0.0)
                continue
            
            # 基础权重基于轨迹质量
            base_weight = traj.quality_score
            
            # 迭代衰减因子
            decay_factor = 1.0 / (1.0 + iteration * 0.001)
            
            # 长度奖励
            length_bonus = min(1.0, traj.length / 10.0)
            
            final_weight = base_weight * decay_factor * length_bonus
            weights.append(final_weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
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
        validation_metrics = {}
        
        # 检查总体误差
        mean_error = constraint_result.mean_error
        validation_metrics['mean_error'] = mean_error
        
        # 检查异常值比例
        outlier_ratio = constraint_result.outlier_ratio
        validation_metrics['outlier_ratio'] = outlier_ratio
        
        # 检查尺度一致性
        scale_consistency = self._check_scale_consistency(constraint_result.scale_contributions)
        validation_metrics['scale_consistency'] = scale_consistency
        
        # 综合评估
        is_valid = (
            mean_error < (1.0 - tolerance) * 10.0 and  # 误差阈值
            outlier_ratio < (1.0 - tolerance) and      # 异常值比例阈值
            scale_consistency > tolerance               # 尺度一致性阈值
        )
        
        validation_metrics['overall_validity'] = float(is_valid)
        
        return is_valid, validation_metrics
    
    def _check_scale_consistency(self, scale_contributions: Dict[float, float]) -> float:
        """
        检查尺度间的一致性
        
        Args:
            scale_contributions: 各尺度贡献
            
        Returns:
            一致性分数 (0.0-1.0)
        """
        if not scale_contributions or len(scale_contributions) < 2:
            return 1.0
        
        contributions = list(scale_contributions.values())
        mean_contribution = np.mean(contributions)
        
        if mean_contribution == 0:
            return 1.0
        
        # 计算变异系数
        cv = np.std(contributions) / mean_contribution
        
        # 转换为一致性分数（变异系数越小，一致性越高）
        consistency = 1.0 / (1.0 + cv)
        
        return consistency
    
    def update_constraint_parameters(self, performance_metrics: Dict[str, float]):
        """
        更新约束参数
        
        Args:
            performance_metrics: 性能指标字典
        """
        # 根据性能指标调整重投影阈值
        if 'mean_error' in performance_metrics:
            mean_error = performance_metrics['mean_error']
            if mean_error > 3.0:
                self.multiscale_processor.reprojection_threshold *= 1.1
            elif mean_error < 1.0:
                self.multiscale_processor.reprojection_threshold *= 0.9
        
        # 根据异常值比例调整自适应权重策略
        if 'outlier_ratio' in performance_metrics:
            outlier_ratio = performance_metrics['outlier_ratio']
            if outlier_ratio > 0.3:
                self.adaptive_weighting = True
            elif outlier_ratio < 0.1:
                self.adaptive_weighting = False


class ScaleConsistencyConstraint:
    """尺度一致性约束实现"""
    
    def __init__(self, 
                 consistency_threshold: float = 0.8,
                 scale_tolerance: float = 0.1,
                 regularization_weight: float = 0.1):
        """
        初始化尺度一致性约束
        
        Args:
            consistency_threshold: 一致性阈值
            scale_tolerance: 尺度容忍度
            regularization_weight: 正则化权重
        """
        self.consistency_threshold = consistency_threshold
        self.scale_tolerance = scale_tolerance
        self.regularization_weight = regularization_weight
    
    def compute_scale_consistency_loss(self,
                                     scale_results: Dict[float, ConstraintResult]) -> torch.Tensor:
        """
        计算尺度一致性损失
        
        Args:
            scale_results: 各尺度的约束结果
            
        Returns:
            一致性损失张量
        """
        if len(scale_results) < 2:
            return torch.tensor(0.0, requires_grad=True)
        
        consistency_loss = torch.tensor(0.0, requires_grad=True)
        scales = sorted(scale_results.keys())
        
        # 计算相邻尺度间的一致性
        for i in range(len(scales) - 1):
            scale1, scale2 = scales[i], scales[i + 1]
            result1, result2 = scale_results[scale1], scale_results[scale2]
            
            # 误差一致性
            error_consistency = self._compute_error_consistency(result1, result2)
            
            # 异常值一致性
            outlier_consistency = self._compute_outlier_consistency(result1, result2)
            
            # 综合一致性损失
            scale_consistency_loss = (1.0 - error_consistency) + (1.0 - outlier_consistency)
            consistency_loss = consistency_loss + scale_consistency_loss
        
        return consistency_loss * self.regularization_weight
    
    def _compute_error_consistency(self, 
                                 result1: ConstraintResult, 
                                 result2: ConstraintResult) -> float:
        """
        计算误差一致性
        
        Args:
            result1: 第一个尺度的结果
            result2: 第二个尺度的结果
            
        Returns:
            误差一致性分数
        """
        if not result1.individual_errors or not result2.individual_errors:
            return 1.0
        
        # 计算误差分布的相似性
        errors1 = np.array(result1.individual_errors)
        errors2 = np.array(result2.individual_errors)
        
        # 归一化误差
        if len(errors1) != len(errors2):
            min_len = min(len(errors1), len(errors2))
            errors1 = errors1[:min_len]
            errors2 = errors2[:min_len]
        
        if len(errors1) == 0:
            return 1.0
        
        # 计算相关系数
        if np.std(errors1) > 0 and np.std(errors2) > 0:
            correlation = np.corrcoef(errors1, errors2)[0, 1]
            consistency = (correlation + 1.0) / 2.0  # 转换到 [0, 1] 范围
        else:
            consistency = 1.0 if np.allclose(errors1, errors2) else 0.0
        
        return max(0.0, min(1.0, consistency))
    
    def _compute_outlier_consistency(self, 
                                   result1: ConstraintResult, 
                                   result2: ConstraintResult) -> float:
        """
        计算异常值一致性
        
        Args:
            result1: 第一个尺度的结果
            result2: 第二个尺度的结果
            
        Returns:
            异常值一致性分数
        """
        if result1.outlier_mask.numel() == 0 or result2.outlier_mask.numel() == 0:
            return 1.0
        
        # 确保掩码长度一致
        mask1 = result1.outlier_mask
        mask2 = result2.outlier_mask
        
        min_len = min(mask1.numel(), mask2.numel())
        if min_len == 0:
            return 1.0
        
        mask1 = mask1[:min_len]
        mask2 = mask2[:min_len]
        
        # 计算异常值检测的一致性
        agreement = (mask1 == mask2).float().mean()
        return float(agreement.item())
    
    def validate_scale_consistency(self, 
                                 scale_results: Dict[float, ConstraintResult]) -> Tuple[bool, Dict[str, float]]:
        """
        验证尺度一致性
        
        Args:
            scale_results: 各尺度的约束结果
            
        Returns:
            (是否一致, 一致性指标)
        """
        if len(scale_results) < 2:
            return True, {'consistency_score': 1.0}
        
        consistency_scores = []
        scales = sorted(scale_results.keys())
        
        for i in range(len(scales) - 1):
            scale1, scale2 = scales[i], scales[i + 1]
            result1, result2 = scale_results[scale1], scale_results[scale2]
            
            error_consistency = self._compute_error_consistency(result1, result2)
            outlier_consistency = self._compute_outlier_consistency(result1, result2)
            
            overall_consistency = (error_consistency + outlier_consistency) / 2.0
            consistency_scores.append(overall_consistency)
        
        avg_consistency = np.mean(consistency_scores)
        is_consistent = avg_consistency >= self.consistency_threshold
        
        metrics = {
            'consistency_score': avg_consistency,
            'min_consistency': np.min(consistency_scores),
            'max_consistency': np.max(consistency_scores),
            'consistency_std': np.std(consistency_scores)
        }
        
        return is_consistent, metrics
    
    def create_fallback_strategy(self, 
                               scale_results: Dict[float, ConstraintResult],
                               failed_scales: List[float]) -> Dict[float, float]:
        """
        创建尺度失效时的降级处理策略
        
        Args:
            scale_results: 各尺度的约束结果
            failed_scales: 失效的尺度列表
            
        Returns:
            调整后的尺度权重
        """
        # 获取有效尺度
        valid_scales = [scale for scale in scale_results.keys() if scale not in failed_scales]
        
        if not valid_scales:
            # 如果所有尺度都失效，使用默认权重
            return {1.0: 1.0}
        
        # 基于质量重新分配权重
        scale_weights = {}
        total_quality = 0.0
        
        for scale in valid_scales:
            result = scale_results[scale]
            # 基于平均误差计算质量分数
            quality = 1.0 / (1.0 + result.mean_error) if result.mean_error > 0 else 1.0
            scale_weights[scale] = quality
            total_quality += quality
        
        # 归一化权重
        if total_quality > 0:
            for scale in scale_weights:
                scale_weights[scale] /= total_quality
        
        return scale_weights


class EnhancedMultiScaleConstraintEngine(MultiScaleConstraintEngine):
    """增强的多尺度约束引擎，包含一致性约束"""
    
    def __init__(self, 
                 scales: List[float] = None,
                 reprojection_threshold: float = 2.0,
                 adaptive_weighting: bool = True,
                 enable_consistency_constraint: bool = True):
        """
        初始化增强的多尺度约束引擎
        
        Args:
            scales: 尺度列表
            reprojection_threshold: 重投影误差阈值
            adaptive_weighting: 是否启用自适应权重
            enable_consistency_constraint: 是否启用一致性约束
        """
        super().__init__(scales, reprojection_threshold, adaptive_weighting)
        self.enable_consistency_constraint = enable_consistency_constraint
        self.consistency_constraint = ScaleConsistencyConstraint()
        self.scale_failure_history = {}
    
    def compute_multiscale_constraints(self,
                                     trajectories: List[Trajectory],
                                     cameras: List,
                                     scales: List[float]) -> ConstraintResult:
        """
        计算增强的多尺度约束（包含一致性约束）
        
        Args:
            trajectories: 特征轨迹列表
            cameras: 相机参数列表
            scales: 尺度列表
            
        Returns:
            增强的多尺度约束结果
        """
        # 计算各尺度的基础约束
        scale_results = {}
        total_loss = torch.tensor(0.0, requires_grad=True)
        all_errors = []
        all_outliers = []
        
        for scale in scales:
            try:
                # 为每个尺度单独计算约束
                scale_trajectories = self._scale_trajectories(trajectories, scale)
                scale_result = self._compute_single_scale_constraint(
                    scale_trajectories, cameras, scale
                )
                scale_results[scale] = scale_result
                total_loss = total_loss + scale_result.loss_value
                all_errors.extend(scale_result.individual_errors)
                all_outliers.append(scale_result.outlier_mask)
                
            except Exception as e:
                # 记录失效的尺度
                if scale not in self.scale_failure_history:
                    self.scale_failure_history[scale] = 0
                self.scale_failure_history[scale] += 1
                print(f"Scale {scale} failed: {e}")
                continue
        
        # 添加尺度一致性约束
        if self.enable_consistency_constraint and len(scale_results) > 1:
            consistency_loss = self.consistency_constraint.compute_scale_consistency_loss(scale_results)
            total_loss = total_loss + consistency_loss
        
        # 验证一致性并处理失效尺度
        if scale_results:
            is_consistent, consistency_metrics = self.consistency_constraint.validate_scale_consistency(scale_results)
            
            if not is_consistent:
                # 识别失效尺度并应用降级策略
                failed_scales = self._identify_failed_scales(scale_results)
                if failed_scales:
                    fallback_weights = self.consistency_constraint.create_fallback_strategy(
                        scale_results, failed_scales
                    )
                    self.multiscale_processor.update_scale_weights(fallback_weights)
        
        # 合并结果
        combined_outlier_mask = torch.cat(all_outliers) if all_outliers else torch.tensor([])
        
        # 计算综合质量指标
        quality_metrics = self._compute_enhanced_quality_metrics(
            all_errors, combined_outlier_mask, scale_results
        )
        
        # 添加一致性指标
        if 'consistency_score' in locals():
            quality_metrics.update(consistency_metrics)
        
        return ConstraintResult(
            loss_value=total_loss,
            individual_errors=all_errors,
            outlier_mask=combined_outlier_mask,
            quality_metrics=quality_metrics,
            scale_contributions={scale: float(result.loss_value.item()) 
                               for scale, result in scale_results.items()}
        )
    
    def _scale_trajectories(self, trajectories: List[Trajectory], scale: float) -> List[Trajectory]:
        """
        缩放轨迹坐标
        
        Args:
            trajectories: 原始轨迹列表
            scale: 缩放因子
            
        Returns:
            缩放后的轨迹列表
        """
        scaled_trajectories = []
        
        for traj in trajectories:
            scaled_points = []
            for point in traj.points_2d:
                scaled_point = Point2D(
                    x=point.x * scale,
                    y=point.y * scale,
                    frame_id=point.frame_id,
                    feature_descriptor=point.feature_descriptor,
                    detection_confidence=point.detection_confidence
                )
                scaled_points.append(scaled_point)
            
            scaled_traj = Trajectory(
                id=traj.id,
                points_2d=scaled_points,
                camera_indices=traj.camera_indices.copy(),
                quality_score=traj.quality_score,
                confidence_scores=traj.confidence_scores.copy(),
                is_active=traj.is_active,
                last_updated=traj.last_updated
            )
            scaled_trajectories.append(scaled_traj)
        
        return scaled_trajectories
    
    def _compute_single_scale_constraint(self,
                                       trajectories: List[Trajectory],
                                       cameras: List,
                                       scale: float) -> ConstraintResult:
        """
        计算单个尺度的约束
        
        Args:
            trajectories: 轨迹列表
            cameras: 相机列表
            scale: 当前尺度
            
        Returns:
            单尺度约束结果
        """
        # 简化的单尺度约束计算
        loss = torch.tensor(0.0, requires_grad=True)
        errors = []
        outliers = []
        
        for traj in trajectories:
            if not traj.is_valid():
                continue
            
            # 模拟重投影误差计算
            traj_errors = []
            for i, point in enumerate(traj.points_2d):
                # 简化的误差计算
                error = abs(point.x - point.y) * scale  # 占位符计算
                traj_errors.append(error)
                loss = loss + torch.tensor(error, requires_grad=True)
            
            errors.extend(traj_errors)
            
            # 异常值检测
            if traj_errors:
                threshold = np.percentile(traj_errors, 75) + 1.5 * (
                    np.percentile(traj_errors, 75) - np.percentile(traj_errors, 25)
                )
                traj_outliers = [error > threshold for error in traj_errors]
                outliers.extend(traj_outliers)
        
        outlier_mask = torch.tensor(outliers, dtype=torch.bool)
        
        quality_metrics = {
            'mean_error': np.mean(errors) if errors else 0.0,
            'outlier_ratio': float(outlier_mask.sum()) / len(outliers) if outliers else 0.0
        }
        
        return ConstraintResult(
            loss_value=loss,
            individual_errors=errors,
            outlier_mask=outlier_mask,
            quality_metrics=quality_metrics
        )
    
    def _identify_failed_scales(self, scale_results: Dict[float, ConstraintResult]) -> List[float]:
        """
        识别失效的尺度
        
        Args:
            scale_results: 各尺度结果
            
        Returns:
            失效尺度列表
        """
        failed_scales = []
        
        for scale, result in scale_results.items():
            # 基于误差和异常值比例判断失效
            if (result.mean_error > 5.0 or  # 误差过大
                result.outlier_ratio > 0.5 or  # 异常值过多
                scale in self.scale_failure_history and self.scale_failure_history[scale] > 3):  # 历史失效次数过多
                failed_scales.append(scale)
        
        return failed_scales
    
    def _compute_enhanced_quality_metrics(self,
                                        all_errors: List[float],
                                        combined_outlier_mask: torch.Tensor,
                                        scale_results: Dict[float, ConstraintResult]) -> Dict[str, float]:
        """
        计算增强的质量指标
        
        Args:
            all_errors: 所有误差
            combined_outlier_mask: 合并的异常值掩码
            scale_results: 各尺度结果
            
        Returns:
            增强的质量指标字典
        """
        metrics = {}
        
        # 基础指标
        if all_errors:
            metrics.update({
                'mean_error': np.mean(all_errors),
                'std_error': np.std(all_errors),
                'max_error': np.max(all_errors),
                'min_error': np.min(all_errors)
            })
        
        if combined_outlier_mask.numel() > 0:
            metrics['outlier_ratio'] = float(combined_outlier_mask.sum()) / combined_outlier_mask.numel()
        
        # 尺度特定指标
        if scale_results:
            scale_errors = [result.mean_error for result in scale_results.values()]
            metrics['scale_error_variance'] = np.var(scale_errors)
            metrics['scale_error_range'] = np.max(scale_errors) - np.min(scale_errors)
        
        return metrics