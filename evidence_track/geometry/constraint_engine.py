"""
核心约束引擎实现

实现增强的重投影约束计算、约束融合和异常值检测
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass

from .interfaces import ConstraintEngine
from .data_structures import Trajectory, ConstraintResult, Point2D
from .config import ConstraintConfig


@dataclass
class ReprojectionStats:
    """重投影统计信息"""
    mean_error: float
    std_error: float
    median_error: float
    max_error: float
    outlier_count: int
    total_count: int
    
    @property
    def outlier_ratio(self) -> float:
        return self.outlier_count / max(self.total_count, 1)


class HuberLoss(torch.nn.Module):
    """Huber损失函数实现"""
    
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算Huber损失
        
        Args:
            input: 预测值
            target: 目标值
            
        Returns:
            Huber损失值
        """
        residual = torch.abs(input - target)
        condition = residual < self.delta
        
        # 对于小误差使用L2损失，大误差使用L1损失
        loss = torch.where(
            condition,
            0.5 * residual ** 2,
            self.delta * residual - 0.5 * self.delta ** 2
        )
        
        return loss


class RobustEstimator:
    """鲁棒估计器"""
    
    @staticmethod
    def compute_mad_threshold(errors: torch.Tensor, factor: float = 2.5) -> float:
        """
        计算基于中位数绝对偏差(MAD)的异常值阈值
        
        Args:
            errors: 误差张量
            factor: MAD倍数因子
            
        Returns:
            异常值阈值
        """
        median = torch.median(errors)
        mad = torch.median(torch.abs(errors - median))
        threshold = median + factor * mad * 1.4826  # 1.4826是正态分布的MAD缩放因子
        return float(threshold)
    
    @staticmethod
    def detect_outliers_iqr(errors: torch.Tensor, factor: float = 1.5) -> torch.Tensor:
        """
        使用四分位距(IQR)方法检测异常值
        
        Args:
            errors: 误差张量
            factor: IQR倍数因子
            
        Returns:
            异常值掩码
        """
        q1 = torch.quantile(errors, 0.25)
        q3 = torch.quantile(errors, 0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        
        outliers = (errors < lower_bound) | (errors > upper_bound)
        return outliers
    
    @staticmethod
    def detect_outliers_zscore(errors: torch.Tensor, threshold: float = 3.0) -> torch.Tensor:
        """
        使用Z-score方法检测异常值
        
        Args:
            errors: 误差张量
            threshold: Z-score阈值
            
        Returns:
            异常值掩码
        """
        mean = torch.mean(errors)
        std = torch.std(errors)
        
        if std < 1e-8:  # 避免除零
            return torch.zeros_like(errors, dtype=torch.bool)
        
        z_scores = torch.abs((errors - mean) / std)
        outliers = z_scores > threshold
        return outliers


class EnhancedReprojectionConstraint:
    """增强的重投影约束计算器"""
    
    def __init__(self, config: ConstraintConfig):
        self.config = config
        self.device = config.get_device()
        self.dtype = config.get_torch_dtype()
        
        # 初始化损失函数
        self.huber_loss = HuberLoss(delta=1.0)
        self.robust_estimator = RobustEstimator()
        
        # 统计信息
        self.stats_history = []
        
        self.logger = logging.getLogger(__name__)
    
    def compute_batch_reprojection(self, 
                                 trajectories: List[Trajectory],
                                 cameras: List,
                                 gaussian_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        高效的批量重投影计算
        
        Args:
            trajectories: 特征轨迹列表
            cameras: 相机参数列表
            gaussian_points: 高斯点云 [N, 3]
            
        Returns:
            (重投影误差, 有效掩码)
        """
        if not trajectories:
            return torch.tensor([], device=self.device), torch.tensor([], dtype=torch.bool, device=self.device)
        
        # 收集所有有效的轨迹点
        all_points_2d = []
        all_camera_indices = []
        all_point_indices = []
        
        for traj_idx, trajectory in enumerate(trajectories):
            if not trajectory.is_active or trajectory.quality_score < self.config.quality.min_quality_score:
                continue
                
            points_tensor = trajectory.get_points_tensor().to(self.device, dtype=self.dtype)
            camera_indices = torch.tensor(trajectory.camera_indices, device=self.device)
            
            all_points_2d.append(points_tensor)
            all_camera_indices.append(camera_indices)
            all_point_indices.extend([traj_idx] * len(trajectory.points_2d))
        
        if not all_points_2d:
            return torch.tensor([], device=self.device), torch.tensor([], dtype=torch.bool, device=self.device)
        
        # 合并所有点
        batch_points_2d = torch.cat(all_points_2d, dim=0)  # [M, 2]
        batch_camera_indices = torch.cat(all_camera_indices, dim=0)  # [M]
        
        # 批量重投影计算
        projected_points = self._batch_project_points(gaussian_points, cameras, batch_camera_indices)
        
        # 计算重投影误差
        reprojection_errors = torch.norm(batch_points_2d - projected_points, dim=1)  # [M]
        
        # 生成有效掩码
        valid_mask = torch.isfinite(reprojection_errors) & (reprojection_errors < 100.0)  # 排除极大误差
        
        return reprojection_errors, valid_mask
    
    def _batch_project_points(self, 
                            gaussian_points: torch.Tensor,
                            cameras: List,
                            camera_indices: torch.Tensor) -> torch.Tensor:
        """
        批量投影3D点到2D
        
        Args:
            gaussian_points: 3D点 [N, 3]
            cameras: 相机列表
            camera_indices: 相机索引 [M]
            
        Returns:
            投影的2D点 [M, 2]
        """
        projected_points = []
        
        # 按相机分组处理以提高效率
        unique_cameras = torch.unique(camera_indices)
        
        for cam_idx in unique_cameras:
            mask = camera_indices == cam_idx
            if not mask.any():
                continue
            
            # 获取该相机对应的3D点
            point_indices = torch.where(mask)[0]
            points_3d = gaussian_points[point_indices]  # [K, 3]
            
            # 投影到2D
            camera = cameras[int(cam_idx)]
            projected_2d = self._project_camera(points_3d, camera)
            
            projected_points.append((point_indices, projected_2d))
        
        # 重新排序结果
        result = torch.zeros((len(camera_indices), 2), device=self.device, dtype=self.dtype)
        for indices, points in projected_points:
            result[indices] = points
        
        return result
    
    def _project_camera(self, points_3d: torch.Tensor, camera) -> torch.Tensor:
        """
        使用相机参数投影3D点
        
        Args:
            points_3d: 3D点 [N, 3]
            camera: 相机对象
            
        Returns:
            2D投影点 [N, 2]
        """
        # 这里需要根据实际的相机类实现
        # 假设相机有world_to_camera和project方法
        if hasattr(camera, 'world_view_transform') and hasattr(camera, 'projection_matrix'):
            # 世界坐标到相机坐标
            points_homo = torch.cat([points_3d, torch.ones(points_3d.shape[0], 1, device=self.device)], dim=1)
            points_cam = torch.matmul(points_homo, camera.world_view_transform.T)
            
            # 相机坐标到屏幕坐标
            points_proj = torch.matmul(points_cam, camera.projection_matrix.T)
            
            # 透视除法
            points_2d = points_proj[:, :2] / (points_proj[:, 3:4] + 1e-8)
            
            return points_2d
        else:
            # 简化的投影模型
            self.logger.warning("Using simplified projection model")
            return points_3d[:, :2]  # 简单的正交投影
    
    def compute_robust_loss(self, 
                          reprojection_errors: torch.Tensor,
                          use_huber: bool = True) -> torch.Tensor:
        """
        计算鲁棒损失
        
        Args:
            reprojection_errors: 重投影误差
            use_huber: 是否使用Huber损失
            
        Returns:
            鲁棒损失值
        """
        if use_huber:
            # 使用Huber损失
            zero_target = torch.zeros_like(reprojection_errors)
            loss = self.huber_loss(reprojection_errors, zero_target)
        else:
            # 使用L2损失
            loss = 0.5 * reprojection_errors ** 2
        
        return loss
    
    def compute_statistics(self, reprojection_errors: torch.Tensor, valid_mask: torch.Tensor) -> ReprojectionStats:
        """
        计算重投影误差统计信息
        
        Args:
            reprojection_errors: 重投影误差
            valid_mask: 有效掩码
            
        Returns:
            统计信息
        """
        if not valid_mask.any():
            return ReprojectionStats(0.0, 0.0, 0.0, 0.0, 0, 0)
        
        valid_errors = reprojection_errors[valid_mask]
        
        # 检测异常值
        outlier_threshold = self.config.quality.outlier_threshold_pixels
        outlier_mask = valid_errors > outlier_threshold
        
        stats = ReprojectionStats(
            mean_error=float(torch.mean(valid_errors)),
            std_error=float(torch.std(valid_errors)),
            median_error=float(torch.median(valid_errors)),
            max_error=float(torch.max(valid_errors)),
            outlier_count=int(torch.sum(outlier_mask)),
            total_count=int(torch.sum(valid_mask))
        )
        
        return stats
    
    def detect_outliers(self, 
                       reprojection_errors: torch.Tensor,
                       method: str = "mad") -> torch.Tensor:
        """
        检测异常值
        
        Args:
            reprojection_errors: 重投影误差
            method: 检测方法 ("mad", "iqr", "zscore", "threshold")
            
        Returns:
            异常值掩码
        """
        if len(reprojection_errors) == 0:
            return torch.tensor([], dtype=torch.bool, device=self.device)
        
        if method == "mad":
            threshold = self.robust_estimator.compute_mad_threshold(reprojection_errors)
            outliers = reprojection_errors > threshold
        elif method == "iqr":
            outliers = self.robust_estimator.detect_outliers_iqr(reprojection_errors)
        elif method == "zscore":
            outliers = self.robust_estimator.detect_outliers_zscore(reprojection_errors)
        elif method == "threshold":
            threshold = self.config.quality.outlier_threshold_pixels
            outliers = reprojection_errors > threshold
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return outliers
    
    def compute_enhanced_reprojection_constraints(self,
                                                trajectories: List[Trajectory],
                                                cameras: List,
                                                gaussian_points: torch.Tensor) -> ConstraintResult:
        """
        计算增强的重投影约束
        
        Args:
            trajectories: 特征轨迹列表
            cameras: 相机参数列表
            gaussian_points: 高斯点云
            
        Returns:
            约束计算结果
        """
        # 批量重投影计算
        reprojection_errors, valid_mask = self.compute_batch_reprojection(
            trajectories, cameras, gaussian_points
        )
        
        if len(reprojection_errors) == 0:
            return ConstraintResult(
                loss_value=torch.tensor(0.0, device=self.device),
                individual_errors=[],
                outlier_mask=torch.tensor([], dtype=torch.bool, device=self.device),
                quality_metrics={"mean_error": 0.0, "outlier_ratio": 0.0}
            )
        
        # 异常值检测
        outlier_mask = self.detect_outliers(reprojection_errors[valid_mask])
        
        # 计算统计信息
        stats = self.compute_statistics(reprojection_errors, valid_mask)
        
        # 决定是否使用鲁棒损失
        use_huber = stats.outlier_ratio > 0.1 or stats.mean_error > 1.0
        
        # 计算鲁棒损失
        valid_errors = reprojection_errors[valid_mask]
        inlier_mask = ~outlier_mask
        
        if inlier_mask.any():
            inlier_errors = valid_errors[inlier_mask]
            loss_values = self.compute_robust_loss(inlier_errors, use_huber)
            total_loss = torch.mean(loss_values)
        else:
            total_loss = torch.tensor(0.0, device=self.device)
        
        # 构建完整的异常值掩码
        full_outlier_mask = torch.zeros_like(reprojection_errors, dtype=torch.bool)
        if valid_mask.any():
            full_outlier_mask[valid_mask] = outlier_mask
        
        # 质量指标
        quality_metrics = {
            "mean_error": stats.mean_error,
            "std_error": stats.std_error,
            "median_error": stats.median_error,
            "max_error": stats.max_error,
            "outlier_ratio": stats.outlier_ratio,
            "total_points": stats.total_count,
            "use_huber_loss": use_huber
        }
        
        # 记录统计信息
        self.stats_history.append(stats)
        if len(self.stats_history) > 1000:  # 限制历史记录长度
            self.stats_history.pop(0)
        
        return ConstraintResult(
            loss_value=total_loss,
            individual_errors=reprojection_errors[valid_mask].cpu().tolist(),
            outlier_mask=full_outlier_mask,
            quality_metrics=quality_metrics
        )


class ConstraintFusion:
    """约束融合器"""
    
    def __init__(self, config: ConstraintConfig):
        self.config = config
        self.device = config.get_device()
        self.dtype = config.get_torch_dtype()
        
        # 融合权重历史
        self.weight_history = []
        
    def compute_constraint_weights(self, 
                                 constraint_results: Dict[str, ConstraintResult],
                                 iteration: int = 0) -> Dict[str, float]:
        """
        计算约束权重
        
        Args:
            constraint_results: 各类约束结果字典
            iteration: 当前迭代次数
            
        Returns:
            约束权重字典
        """
        weights = {}
        
        # 基于质量的权重调整
        for constraint_type, result in constraint_results.items():
            base_weight = self._get_base_weight(constraint_type)
            
            # 质量调整因子
            quality_factor = self._compute_quality_factor(result)
            
            # 时序调整因子
            temporal_factor = self._compute_temporal_factor(iteration)
            
            # 最终权重
            final_weight = base_weight * quality_factor * temporal_factor
            weights[constraint_type] = max(final_weight, 0.001)  # 最小权重阈值
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # 记录权重历史
        self.weight_history.append(weights.copy())
        if len(self.weight_history) > 100:
            self.weight_history.pop(0)
        
        return weights
    
    def _get_base_weight(self, constraint_type: str) -> float:
        """获取约束类型的基础权重"""
        base_weights = {
            "reprojection": 1.0,
            "multiscale": 0.5,
            "temporal": 0.3,
            "geometric": 0.7
        }
        return base_weights.get(constraint_type, 0.5)
    
    def _compute_quality_factor(self, result: ConstraintResult) -> float:
        """计算基于质量的调整因子"""
        outlier_ratio = result.outlier_ratio
        mean_error = result.quality_metrics.get("mean_error", 1.0)
        
        # 异常值比例越高，权重越低
        outlier_penalty = max(0.1, 1.0 - 2.0 * outlier_ratio)
        
        # 误差越大，权重越低
        error_penalty = max(0.1, 1.0 / (1.0 + mean_error))
        
        return outlier_penalty * error_penalty
    
    def _compute_temporal_factor(self, iteration: int) -> float:
        """计算时序调整因子"""
        config = self.config.weighting
        
        if iteration < config.early_stage_iterations:
            return config.early_stage_weight
        elif iteration < config.mid_stage_iterations:
            return config.mid_stage_weight
        else:
            return config.final_stage_weight
    
    def fuse_constraints(self, 
                        constraint_results: Dict[str, ConstraintResult],
                        weights: Dict[str, float]) -> ConstraintResult:
        """
        融合多个约束结果
        
        Args:
            constraint_results: 约束结果字典
            weights: 权重字典
            
        Returns:
            融合后的约束结果
        """
        if not constraint_results:
            return ConstraintResult(
                loss_value=torch.tensor(0.0, device=self.device),
                individual_errors=[],
                outlier_mask=torch.tensor([], dtype=torch.bool, device=self.device),
                quality_metrics={}
            )
        
        # 加权损失融合
        total_loss = torch.tensor(0.0, device=self.device)
        total_weight = 0.0
        
        all_errors = []
        all_outlier_masks = []
        combined_metrics = {}
        
        for constraint_type, result in constraint_results.items():
            weight = weights.get(constraint_type, 0.0)
            if weight > 0:
                total_loss += weight * result.loss_value
                total_weight += weight
                
                # 收集误差和异常值掩码
                all_errors.extend(result.individual_errors)
                if result.outlier_mask.numel() > 0:
                    all_outlier_masks.append(result.outlier_mask)
                
                # 合并质量指标
                for key, value in result.quality_metrics.items():
                    metric_key = f"{constraint_type}_{key}"
                    combined_metrics[metric_key] = value
        
        # 归一化损失
        if total_weight > 0:
            total_loss = total_loss / total_weight
        
        # 合并异常值掩码
        if all_outlier_masks:
            combined_outlier_mask = torch.cat(all_outlier_masks, dim=0)
        else:
            combined_outlier_mask = torch.tensor([], dtype=torch.bool, device=self.device)
        
        # 添加融合权重信息
        combined_metrics["weight_distribution"] = weights
        combined_metrics["total_constraints"] = len(constraint_results)
        
        return ConstraintResult(
            loss_value=total_loss,
            individual_errors=all_errors,
            outlier_mask=combined_outlier_mask,
            quality_metrics=combined_metrics,
            weight_distribution=weights
        )


class GeometricConsistencyChecker:
    """几何一致性检查器"""
    
    def __init__(self, config: ConstraintConfig):
        self.config = config
        self.device = config.get_device()
        
    def check_epipolar_consistency(self, 
                                 trajectories: List[Trajectory],
                                 cameras: List) -> Dict[str, float]:
        """
        检查对极几何一致性
        
        Args:
            trajectories: 轨迹列表
            cameras: 相机列表
            
        Returns:
            一致性指标字典
        """
        if len(trajectories) < 2 or len(cameras) < 2:
            return {"epipolar_error": 0.0, "consistency_ratio": 1.0}
        
        total_error = 0.0
        valid_pairs = 0
        
        # 检查轨迹对之间的对极约束
        for i, traj1 in enumerate(trajectories):
            for j, traj2 in enumerate(trajectories[i+1:], i+1):
                if not (traj1.is_active and traj2.is_active):
                    continue
                
                # 找到共同的相机视图
                common_cameras = set(traj1.camera_indices) & set(traj2.camera_indices)
                if len(common_cameras) < 2:
                    continue
                
                # 计算对极误差
                epipolar_error = self._compute_epipolar_error(traj1, traj2, cameras, common_cameras)
                if epipolar_error is not None:
                    total_error += epipolar_error
                    valid_pairs += 1
        
        if valid_pairs == 0:
            return {"epipolar_error": 0.0, "consistency_ratio": 1.0}
        
        mean_error = total_error / valid_pairs
        consistency_ratio = max(0.0, 1.0 - mean_error / 10.0)  # 假设10像素为完全不一致
        
        return {
            "epipolar_error": mean_error,
            "consistency_ratio": consistency_ratio,
            "valid_pairs": valid_pairs
        }
    
    def _compute_epipolar_error(self, 
                              traj1: Trajectory, 
                              traj2: Trajectory,
                              cameras: List,
                              common_cameras: set) -> Optional[float]:
        """计算两条轨迹间的对极误差"""
        # 简化的对极误差计算
        # 在实际实现中，这里需要使用基础矩阵和对极几何
        
        errors = []
        for cam_idx in common_cameras:
            # 找到对应的点
            try:
                idx1 = traj1.camera_indices.index(cam_idx)
                idx2 = traj2.camera_indices.index(cam_idx)
                
                point1 = traj1.points_2d[idx1]
                point2 = traj2.points_2d[idx2]
                
                # 简化的距离误差（实际应该是点到对极线的距离）
                error = ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5
                errors.append(error)
                
            except (ValueError, IndexError):
                continue
        
        return sum(errors) / len(errors) if errors else None
    
    def validate_geometric_constraints(self, 
                                     constraint_result: ConstraintResult,
                                     tolerance: float = 0.85) -> Tuple[bool, Dict[str, float]]:
        """
        验证几何约束的有效性
        
        Args:
            constraint_result: 约束结果
            tolerance: 容忍度阈值
            
        Returns:
            (是否有效, 验证指标字典)
        """
        metrics = constraint_result.quality_metrics
        
        # 检查各项指标
        checks = {
            "outlier_ratio_check": metrics.get("outlier_ratio", 1.0) < self.config.quality.max_outlier_ratio,
            "mean_error_check": metrics.get("mean_error", float('inf')) < self.config.quality.outlier_threshold_pixels,
            "consistency_check": metrics.get("consistency_ratio", 0.0) > tolerance
        }
        
        # 计算总体有效性
        valid_checks = sum(checks.values())
        total_checks = len(checks)
        overall_validity = valid_checks / total_checks >= tolerance
        
        validation_metrics = {
            "overall_valid": overall_validity,
            "validity_ratio": valid_checks / total_checks,
            **checks
        }
        
        return overall_validity, validation_metrics


class ConstraintEngineImpl(ConstraintEngine):
    """约束引擎具体实现"""
    
    def __init__(self, config: ConstraintConfig):
        self.config = config
        self.device = config.get_device()
        self.dtype = config.get_torch_dtype()
        
        # 初始化组件
        self.reprojection_constraint = EnhancedReprojectionConstraint(config)
        self.constraint_fusion = ConstraintFusion(config)
        self.geometry_checker = GeometricConsistencyChecker(config)
        
        # 性能统计
        self.performance_history = []
        
        self.logger = logging.getLogger(__name__)
    
    def compute_reprojection_constraints(self, 
                                       trajectories: List[Trajectory],
                                       cameras: List,
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
        return self.reprojection_constraint.compute_enhanced_reprojection_constraints(
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
        # 这里需要与多尺度约束模块集成
        # 暂时返回基础实现
        self.logger.warning("Multiscale constraints not fully implemented yet")
        
        return ConstraintResult(
            loss_value=torch.tensor(0.0, device=self.device),
            individual_errors=[],
            outlier_mask=torch.tensor([], dtype=torch.bool, device=self.device),
            quality_metrics={"multiscale_placeholder": True},
            scale_contributions={scale: 1.0/len(scales) for scale in scales}
        )
    
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
        if not trajectories:
            return torch.tensor([], device=self.device)
        
        weights = []
        
        for trajectory in trajectories:
            # 基于质量的权重
            quality_weight = max(0.1, trajectory.quality_score)
            
            # 基于置信度的权重
            confidence_weight = max(0.1, trajectory.average_confidence)
            
            # 基于轨迹长度的权重
            length_weight = min(1.0, trajectory.length / 10.0)
            
            # 时序权重
            temporal_weight = self.constraint_fusion._compute_temporal_factor(iteration)
            
            # 组合权重
            combined_weight = quality_weight * confidence_weight * length_weight * temporal_weight
            weights.append(combined_weight)
        
        return torch.tensor(weights, device=self.device, dtype=self.dtype)
    
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
        return self.geometry_checker.validate_geometric_constraints(constraint_result, tolerance)
    
    def update_constraint_parameters(self, 
                                   performance_metrics: Dict[str, float]):
        """
        更新约束参数
        
        Args:
            performance_metrics: 性能指标字典
        """
        # 记录性能历史
        self.performance_history.append(performance_metrics.copy())
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        
        # 基于性能调整参数
        mean_error = performance_metrics.get("mean_error", 1.0)
        outlier_ratio = performance_metrics.get("outlier_ratio", 0.0)
        
        # 动态调整异常值阈值
        if outlier_ratio > 0.2:  # 异常值过多
            current_threshold = self.config.quality.outlier_threshold_pixels
            new_threshold = min(current_threshold * 1.1, 5.0)  # 适当放宽阈值
            self.config.quality.outlier_threshold_pixels = new_threshold
            self.logger.info(f"Adjusted outlier threshold to {new_threshold}")
        
        # 动态调整质量阈值
        if mean_error > 2.0:  # 误差过大
            current_quality = self.config.quality.min_quality_score
            new_quality = max(current_quality * 0.9, 0.2)  # 适当降低质量要求
            self.config.quality.min_quality_score = new_quality
            self.logger.info(f"Adjusted quality threshold to {new_quality}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.performance_history:
            return {}
        
        recent_metrics = self.performance_history[-10:]  # 最近10次的指标
        
        summary = {
            "recent_mean_error": sum(m.get("mean_error", 0) for m in recent_metrics) / len(recent_metrics),
            "recent_outlier_ratio": sum(m.get("outlier_ratio", 0) for m in recent_metrics) / len(recent_metrics),
            "total_evaluations": len(self.performance_history),
            "weight_stability": self._compute_weight_stability(),
            "constraint_config": {
                "outlier_threshold": self.config.quality.outlier_threshold_pixels,
                "quality_threshold": self.config.quality.min_quality_score,
                "max_outlier_ratio": self.config.quality.max_outlier_ratio
            }
        }
        
        return summary
    
    def _compute_weight_stability(self) -> float:
        """计算权重稳定性指标"""
        if len(self.constraint_fusion.weight_history) < 2:
            return 1.0
        
        recent_weights = self.constraint_fusion.weight_history[-10:]
        if len(recent_weights) < 2:
            return 1.0
        
        # 计算权重变化的标准差
        weight_changes = []
        for i in range(1, len(recent_weights)):
            prev_weights = recent_weights[i-1]
            curr_weights = recent_weights[i]
            
            # 计算权重向量的L2距离
            common_keys = set(prev_weights.keys()) & set(curr_weights.keys())
            if common_keys:
                change = sum((prev_weights.get(k, 0) - curr_weights.get(k, 0))**2 for k in common_keys)**0.5
                weight_changes.append(change)
        
        if not weight_changes:
            return 1.0
        
        # 稳定性 = 1 - 归一化的变化幅度
        mean_change = sum(weight_changes) / len(weight_changes)
        stability = max(0.0, 1.0 - mean_change)
        
        return stability


class AdvancedOutlierDetector:
    """高级异常值检测器"""
    
    def __init__(self, config: ConstraintConfig):
        self.config = config
        self.device = config.get_device()
        self.dtype = config.get_torch_dtype()
        
        # 异常值检测历史
        self.detection_history = []
        
        self.logger = logging.getLogger(__name__)
    
    def detect_statistical_outliers(self, 
                                  errors: torch.Tensor,
                                  method: str = "ensemble") -> torch.Tensor:
        """
        基于统计的异常值检测
        
        Args:
            errors: 误差张量
            method: 检测方法 ("mad", "iqr", "zscore", "ensemble")
            
        Returns:
            异常值掩码
        """
        if len(errors) == 0:
            return torch.tensor([], dtype=torch.bool, device=self.device)
        
        if method == "ensemble":
            # 集成多种方法
            mad_outliers = self._detect_mad_outliers(errors)
            iqr_outliers = self._detect_iqr_outliers(errors)
            zscore_outliers = self._detect_zscore_outliers(errors)
            
            # 投票机制：至少两种方法认为是异常值
            vote_count = mad_outliers.int() + iqr_outliers.int() + zscore_outliers.int()
            ensemble_outliers = vote_count >= 2
            
            return ensemble_outliers
        
        elif method == "mad":
            return self._detect_mad_outliers(errors)
        elif method == "iqr":
            return self._detect_iqr_outliers(errors)
        elif method == "zscore":
            return self._detect_zscore_outliers(errors)
        else:
            raise ValueError(f"Unknown detection method: {method}")
    
    def _detect_mad_outliers(self, errors: torch.Tensor, factor: float = 2.5) -> torch.Tensor:
        """基于MAD的异常值检测"""
        median = torch.median(errors)
        mad = torch.median(torch.abs(errors - median))
        
        if mad < 1e-8:  # 避免除零
            return torch.zeros_like(errors, dtype=torch.bool)
        
        threshold = median + factor * mad * 1.4826
        return errors > threshold
    
    def _detect_iqr_outliers(self, errors: torch.Tensor, factor: float = 1.5) -> torch.Tensor:
        """基于IQR的异常值检测"""
        q1 = torch.quantile(errors, 0.25)
        q3 = torch.quantile(errors, 0.75)
        iqr = q3 - q1
        
        if iqr < 1e-8:
            return torch.zeros_like(errors, dtype=torch.bool)
        
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        
        return (errors < lower_bound) | (errors > upper_bound)
    
    def _detect_zscore_outliers(self, errors: torch.Tensor, threshold: float = 3.0) -> torch.Tensor:
        """基于Z-score的异常值检测"""
        mean = torch.mean(errors)
        std = torch.std(errors)
        
        if std < 1e-8:
            return torch.zeros_like(errors, dtype=torch.bool)
        
        z_scores = torch.abs((errors - mean) / std)
        return z_scores > threshold
    
    def detect_geometric_outliers(self, 
                                trajectories: List[Trajectory],
                                cameras: List) -> Dict[int, torch.Tensor]:
        """
        基于几何一致性的异常值检测
        
        Args:
            trajectories: 轨迹列表
            cameras: 相机列表
            
        Returns:
            轨迹ID到异常值掩码的映射
        """
        outlier_masks = {}
        
        for trajectory in trajectories:
            if not trajectory.is_active or trajectory.length < 3:
                continue
            
            # 检测轨迹内的几何不一致性
            geometric_outliers = self._detect_trajectory_geometric_outliers(trajectory, cameras)
            
            if geometric_outliers.any():
                outlier_masks[trajectory.id] = geometric_outliers
        
        return outlier_masks
    
    def _detect_trajectory_geometric_outliers(self, 
                                            trajectory: Trajectory,
                                            cameras: List) -> torch.Tensor:
        """检测单条轨迹内的几何异常值"""
        points = trajectory.get_points_tensor()
        
        if len(points) < 3:
            return torch.zeros(len(points), dtype=torch.bool, device=self.device)
        
        # 计算轨迹的运动一致性
        motion_outliers = self._detect_motion_outliers(points)
        
        # 计算重投影一致性（如果有足够的相机信息）
        if len(cameras) > 1:
            reprojection_outliers = self._detect_reprojection_outliers(trajectory, cameras)
            # 合并两种检测结果
            combined_outliers = motion_outliers | reprojection_outliers
        else:
            combined_outliers = motion_outliers
        
        return combined_outliers
    
    def _detect_motion_outliers(self, points: torch.Tensor) -> torch.Tensor:
        """检测运动异常值"""
        if len(points) < 3:
            return torch.zeros(len(points), dtype=torch.bool, device=self.device)
        
        # 计算相邻点之间的位移
        displacements = torch.diff(points, dim=0)  # [N-1, 2]
        displacement_norms = torch.norm(displacements, dim=1)  # [N-1]
        
        # 检测位移的异常值
        if len(displacement_norms) < 2:
            return torch.zeros(len(points), dtype=torch.bool, device=self.device)
        
        # 使用MAD检测异常位移
        median_disp = torch.median(displacement_norms)
        mad_disp = torch.median(torch.abs(displacement_norms - median_disp))
        
        if mad_disp < 1e-8:
            return torch.zeros(len(points), dtype=torch.bool, device=self.device)
        
        threshold = median_disp + 3.0 * mad_disp * 1.4826
        outlier_displacements = displacement_norms > threshold
        
        # 将位移异常值映射回点
        outlier_points = torch.zeros(len(points), dtype=torch.bool, device=self.device)
        
        # 异常位移的两个端点都标记为异常值
        for i, is_outlier in enumerate(outlier_displacements):
            if is_outlier:
                outlier_points[i] = True      # 起始点
                outlier_points[i + 1] = True  # 结束点
        
        return outlier_points
    
    def _detect_reprojection_outliers(self, 
                                    trajectory: Trajectory,
                                    cameras: List) -> torch.Tensor:
        """检测重投影异常值"""
        # 简化的重投影一致性检查
        outliers = torch.zeros(trajectory.length, dtype=torch.bool, device=self.device)
        
        # 这里可以实现更复杂的重投影一致性检查
        # 暂时使用基于置信度的简单检测
        confidence_threshold = 0.3
        for i, confidence in enumerate(trajectory.confidence_scores):
            if confidence < confidence_threshold:
                outliers[i] = True
        
        return outliers
    
    def process_outliers(self, 
                        trajectories: List[Trajectory],
                        outlier_masks: Dict[int, torch.Tensor],
                        action: str = "suppress") -> List[Trajectory]:
        """
        处理检测到的异常值
        
        Args:
            trajectories: 轨迹列表
            outlier_masks: 异常值掩码字典
            action: 处理动作 ("suppress", "remove", "split")
            
        Returns:
            处理后的轨迹列表
        """
        processed_trajectories = []
        
        for trajectory in trajectories:
            if trajectory.id not in outlier_masks:
                processed_trajectories.append(trajectory)
                continue
            
            outlier_mask = outlier_masks[trajectory.id]
            
            if action == "suppress":
                # 抑制异常值的权重
                processed_traj = self._suppress_outliers(trajectory, outlier_mask)
                processed_trajectories.append(processed_traj)
                
            elif action == "remove":
                # 移除异常值点
                processed_traj = self._remove_outliers(trajectory, outlier_mask)
                if processed_traj.length >= self.config.quality.min_trajectory_length:
                    processed_trajectories.append(processed_traj)
                    
            elif action == "split":
                # 在异常值处分割轨迹
                split_trajectories = self._split_at_outliers(trajectory, outlier_mask)
                processed_trajectories.extend(split_trajectories)
                
            else:
                raise ValueError(f"Unknown outlier action: {action}")
        
        return processed_trajectories
    
    def _suppress_outliers(self, trajectory: Trajectory, outlier_mask: torch.Tensor) -> Trajectory:
        """抑制异常值的权重"""
        new_confidence_scores = trajectory.confidence_scores.copy()
        
        for i, is_outlier in enumerate(outlier_mask):
            if is_outlier and i < len(new_confidence_scores):
                new_confidence_scores[i] *= 0.1  # 大幅降低置信度
        
        # 创建新的轨迹对象
        new_trajectory = Trajectory(
            id=trajectory.id,
            points_2d=trajectory.points_2d.copy(),
            camera_indices=trajectory.camera_indices.copy(),
            quality_score=trajectory.quality_score * 0.8,  # 降低质量分数
            confidence_scores=new_confidence_scores,
            is_active=trajectory.is_active,
            last_updated=trajectory.last_updated
        )
        
        return new_trajectory
    
    def _remove_outliers(self, trajectory: Trajectory, outlier_mask: torch.Tensor) -> Trajectory:
        """移除异常值点"""
        inlier_indices = [i for i, is_outlier in enumerate(outlier_mask) if not is_outlier]
        
        if not inlier_indices:
            # 如果所有点都是异常值，返回空轨迹
            return Trajectory(
                id=trajectory.id,
                points_2d=[],
                camera_indices=[],
                quality_score=0.0,
                confidence_scores=[],
                is_active=False,
                last_updated=trajectory.last_updated
            )
        
        # 保留内点
        new_points_2d = [trajectory.points_2d[i] for i in inlier_indices]
        new_camera_indices = [trajectory.camera_indices[i] for i in inlier_indices]
        new_confidence_scores = [trajectory.confidence_scores[i] for i in inlier_indices]
        
        new_trajectory = Trajectory(
            id=trajectory.id,
            points_2d=new_points_2d,
            camera_indices=new_camera_indices,
            quality_score=trajectory.quality_score,
            confidence_scores=new_confidence_scores,
            is_active=trajectory.is_active,
            last_updated=trajectory.last_updated
        )
        
        return new_trajectory
    
    def _split_at_outliers(self, trajectory: Trajectory, outlier_mask: torch.Tensor) -> List[Trajectory]:
        """在异常值处分割轨迹"""
        if not outlier_mask.any():
            return [trajectory]
        
        # 找到连续的内点段
        segments = []
        current_segment = []
        
        for i, is_outlier in enumerate(outlier_mask):
            if not is_outlier:
                current_segment.append(i)
            else:
                if len(current_segment) >= self.config.quality.min_trajectory_length:
                    segments.append(current_segment)
                current_segment = []
        
        # 处理最后一个段
        if len(current_segment) >= self.config.quality.min_trajectory_length:
            segments.append(current_segment)
        
        # 创建分割后的轨迹
        split_trajectories = []
        for seg_idx, segment in enumerate(segments):
            new_points_2d = [trajectory.points_2d[i] for i in segment]
            new_camera_indices = [trajectory.camera_indices[i] for i in segment]
            new_confidence_scores = [trajectory.confidence_scores[i] for i in segment]
            
            new_trajectory = Trajectory(
                id=trajectory.id * 1000 + seg_idx,  # 生成新的ID
                points_2d=new_points_2d,
                camera_indices=new_camera_indices,
                quality_score=trajectory.quality_score,
                confidence_scores=new_confidence_scores,
                is_active=trajectory.is_active,
                last_updated=trajectory.last_updated
            )
            
            split_trajectories.append(new_trajectory)
        
        return split_trajectories
    
    def generate_outlier_report(self, 
                              outlier_masks: Dict[int, torch.Tensor],
                              trajectories: List[Trajectory]) -> Dict[str, Any]:
        """
        生成异常值检测报告
        
        Args:
            outlier_masks: 异常值掩码字典
            trajectories: 轨迹列表
            
        Returns:
            异常值报告
        """
        total_points = sum(traj.length for traj in trajectories)
        total_outliers = sum(mask.sum().item() for mask in outlier_masks.values())
        
        trajectory_stats = []
        for trajectory in trajectories:
            if trajectory.id in outlier_masks:
                outlier_count = outlier_masks[trajectory.id].sum().item()
                outlier_ratio = outlier_count / trajectory.length if trajectory.length > 0 else 0.0
            else:
                outlier_count = 0
                outlier_ratio = 0.0
            
            trajectory_stats.append({
                "trajectory_id": trajectory.id,
                "total_points": trajectory.length,
                "outlier_count": outlier_count,
                "outlier_ratio": outlier_ratio,
                "quality_score": trajectory.quality_score
            })
        
        report = {
            "summary": {
                "total_trajectories": len(trajectories),
                "total_points": total_points,
                "total_outliers": total_outliers,
                "overall_outlier_ratio": total_outliers / max(total_points, 1),
                "affected_trajectories": len(outlier_masks)
            },
            "trajectory_details": trajectory_stats,
            "detection_settings": {
                "outlier_threshold_pixels": self.config.quality.outlier_threshold_pixels,
                "max_outlier_ratio": self.config.quality.max_outlier_ratio,
                "min_trajectory_length": self.config.quality.min_trajectory_length
            }
        }
        
        # 记录检测历史
        self.detection_history.append(report["summary"].copy())
        if len(self.detection_history) > 100:
            self.detection_history.pop(0)
        
        return report


# 更新约束引擎实现，集成异常值检测
class EnhancedConstraintEngine(ConstraintEngineImpl):
    """增强的约束引擎，集成高级异常值检测"""
    
    def __init__(self, config: ConstraintConfig):
        super().__init__(config)
        
        # 添加异常值检测器
        self.outlier_detector = AdvancedOutlierDetector(config)
        
        # 异常值处理策略
        self.outlier_action = "suppress"  # "suppress", "remove", "split"
        
    def compute_reprojection_constraints(self, 
                                       trajectories: List[Trajectory],
                                       cameras: List,
                                       gaussian_points: torch.Tensor) -> ConstraintResult:
        """
        计算带异常值检测的重投影约束
        
        Args:
            trajectories: 特征轨迹列表
            cameras: 相机参数列表
            gaussian_points: 高斯点云
            
        Returns:
            约束计算结果
        """
        # 首先进行几何异常值检测
        geometric_outliers = self.outlier_detector.detect_geometric_outliers(trajectories, cameras)
        
        # 处理几何异常值
        if geometric_outliers:
            processed_trajectories = self.outlier_detector.process_outliers(
                trajectories, geometric_outliers, self.outlier_action
            )
        else:
            processed_trajectories = trajectories
        
        # 计算重投影约束
        constraint_result = super().compute_reprojection_constraints(
            processed_trajectories, cameras, gaussian_points
        )
        
        # 对重投影误差进行统计异常值检测
        if len(constraint_result.individual_errors) > 0:
            errors_tensor = torch.tensor(constraint_result.individual_errors, device=self.device)
            statistical_outliers = self.outlier_detector.detect_statistical_outliers(errors_tensor)
            
            # 合并异常值掩码
            if constraint_result.outlier_mask.numel() > 0:
                combined_outliers = constraint_result.outlier_mask | statistical_outliers
            else:
                combined_outliers = statistical_outliers
            
            # 更新约束结果
            constraint_result.outlier_mask = combined_outliers
            constraint_result.quality_metrics["statistical_outlier_ratio"] = float(statistical_outliers.sum()) / len(statistical_outliers)
        
        # 添加异常值检测信息
        constraint_result.quality_metrics["geometric_outlier_count"] = len(geometric_outliers)
        constraint_result.quality_metrics["outlier_action"] = self.outlier_action
        
        return constraint_result
    
    def set_outlier_action(self, action: str):
        """设置异常值处理动作"""
        if action not in ["suppress", "remove", "split"]:
            raise ValueError(f"Invalid outlier action: {action}")
        self.outlier_action = action
        self.logger.info(f"Outlier action set to: {action}")
    
    def get_outlier_report(self, trajectories: List[Trajectory], cameras: List) -> Dict[str, Any]:
        """获取异常值检测报告"""
        geometric_outliers = self.outlier_detector.detect_geometric_outliers(trajectories, cameras)
        return self.outlier_detector.generate_outlier_report(geometric_outliers, trajectories)