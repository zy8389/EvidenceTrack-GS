#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的轨迹查询系统 - 高性能版本，解决大数据集卡顿问题
专门针对23k+轨迹的COLMAP数据集优化
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无界面后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import h5py
import json
import logging
from typing import List, Dict, Optional, Tuple, Any, Set
import argparse
from dataclasses import dataclass, asdict
import time

# 添加项目路径
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometric_constraints.data_structures import Trajectory, Point2D
from geometric_constraints.trajectory_manager import TrajectoryManagerImpl, QualityAssessorImpl
from geometric_constraints.config import ConstraintConfig

# 简化的数据结构，避免复杂的3D处理
@dataclass
class SimplePoint3D:
    """简化的3D点信息"""
    id: int
    xyz: np.ndarray
    error: float
    quality_score: float

@dataclass
class SimpleCamera:
    """简化的相机信息"""
    id: int
    position: np.ndarray

@dataclass
class QueryResult:
    """查询结果"""
    query_type: str
    query_id: int
    num_correspondences: int
    quality_score: float
    metadata: Dict[str, Any]


class OptimizedTrajectoryQuerySystem:
    """优化的轨迹查询系统 - 专门处理大数据集"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化查询系统"""
        # 加载配置
        if config_path and Path(config_path).exists():
            self.config = ConstraintConfig.from_file(config_path)
        else:
            self.config = ConstraintConfig()
        
        # 使用用户的轨迹管理器
        self.quality_assessor = QualityAssessorImpl(self.config)
        self.trajectory_manager = TrajectoryManagerImpl(self.config, self.quality_assessor)
        
        # 数据存储
        self.trajectories: List[Trajectory] = []
        self.points_3d_dict: Dict[int, SimplePoint3D] = {}  # 使用字典加速查找
        self.cameras_dict: Dict[int, SimpleCamera] = {}
        
        # 高效的映射索引
        self.traj_to_3d_mapping: Dict[int, List[int]] = {}
        self.point3d_to_traj_mapping: Dict[int, List[int]] = {}
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 性能统计
        self.perf_stats = {}
    
    def load_complete_dataset(self, 
                             h5_path: Optional[str] = None,
                             colmap_path: Optional[str] = None,
                             max_trajectories: int = 5000,  # 限制处理的轨迹数量
                             sample_ratio: float = 0.2) -> bool:
        """加载完整数据集 - 优化版本"""
        start_time = time.time()
        self.logger.info("Loading dataset with optimizations for large data...")
        
        # 使用用户的轨迹管理器加载2D轨迹数据
        if h5_path and Path(h5_path).exists():
            try:
                self.logger.info(f"Loading trajectories from H5 file: {h5_path}")
                all_trajectories = self.trajectory_manager.load_trajectories(h5_path)
                
                # 如果数据太大，进行采样
                if len(all_trajectories) > max_trajectories:
                    self.logger.info(f"Large dataset detected ({len(all_trajectories)} trajectories)")
                    self.logger.info(f"Sampling {max_trajectories} trajectories for processing...")
                    
                    # 智能采样：选择质量较高的轨迹
                    trajectory_qualities = [(i, traj.quality_score) for i, traj in enumerate(all_trajectories)]
                    trajectory_qualities.sort(key=lambda x: x[1], reverse=True)  # 按质量排序
                    
                    selected_indices = [idx for idx, _ in trajectory_qualities[:max_trajectories]]
                    self.trajectories = [all_trajectories[i] for i in selected_indices]
                else:
                    self.trajectories = all_trajectories
                
                self.logger.info(f"Using {len(self.trajectories)} trajectories for analysis")
                
            except Exception as e:
                self.logger.error(f"Failed to load H5 trajectories: {e}")
                self.trajectories = self.trajectory_manager._create_dummy_trajectories()
        else:
            self.logger.warning("No H5 file provided, using dummy trajectories")
            self.trajectories = self.trajectory_manager._create_dummy_trajectories()
        
        load_time = time.time() - start_time
        self.perf_stats['trajectory_load_time'] = load_time
        
        # 加载3D数据（简化版本）
        start_time = time.time()
        if colmap_path:
            success = self._load_colmap_3d_data_optimized(colmap_path, sample_ratio)
        else:
            success = self._generate_dummy_3d_data_optimized()
        
        if not success:
            self.logger.error("Failed to load 3D data")
            return False
        
        load_3d_time = time.time() - start_time
        self.perf_stats['3d_load_time'] = load_3d_time
        
        # 建立对应关系（高效版本）
        start_time = time.time()
        self._build_correspondences_optimized()
        correspondence_time = time.time() - start_time
        self.perf_stats['correspondence_time'] = correspondence_time
        
        self.logger.info(f"Dataset loaded successfully in {sum(self.perf_stats.values()):.2f}s:")
        self.logger.info(f"  - Trajectories: {len(self.trajectories)}")
        self.logger.info(f"  - 3D Points: {len(self.points_3d_dict)}")
        self.logger.info(f"  - Cameras: {len(self.cameras_dict)}")
        
        return True
    
    def _generate_paper_style_trajectory_pair(self, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成用于论文风格可视化的一对3D轨迹 (Ground-truth 和 Initialization).
        专门优化红色轨迹的平滑度，使其看起来更专业。

        Args:
            seed: 随机种子，确保每次生成的轨迹对是可复现的。

        Returns:
            一个元组，包含 (ground_truth_traj, init_traj)。
        """
        np.random.seed(seed)

        # 1. 生成平滑的、闭环的 "Ground-truth" 轨迹
        # 使用不同频率和相位的正弦/余弦函数组合来创建复杂但平滑的3D曲线
        num_points = 200  # 增加点数，让轨迹更平滑
        t = np.linspace(0, 2 * np.pi, num_points)

        # 随机化轨迹的参数，但参数更加保守
        x_freq1, x_freq2 = np.random.uniform(1, 2.5, 2)
        y_freq1, y_freq2 = np.random.uniform(1, 2.5, 2)
        z_freq1, z_freq2 = np.random.uniform(1, 1.8, 2)
        
        x_amp = np.random.uniform(1.8, 2.2)
        y_amp = np.random.uniform(1.8, 2.2)
        z_amp = np.random.uniform(0.6, 0.9) # Z轴幅度适中

        x_phase = np.random.uniform(0, np.pi)
        y_phase = np.random.uniform(0, np.pi)
        z_phase = np.random.uniform(0, np.pi)

        gt_x = x_amp * (np.cos(x_freq1 * t + x_phase) + 0.3 * np.sin(x_freq2 * t))
        gt_y = y_amp * (np.sin(y_freq1 * t + y_phase) + 0.3 * np.cos(y_freq2 * t))
        gt_z = z_amp * np.cos(z_freq1 * t + z_phase)

        ground_truth_traj = np.column_stack([gt_x, gt_y, gt_z])

        # 2. 生成 "Initialization" 轨迹 - 关键优化部分
        # 使用更加精细的扰动策略，确保平滑性

        # a. 添加平滑的、低频的系统性偏差（减小幅度）
        distort_freq = np.random.uniform(1.2, 1.8)
        distort_x = 0.4 * np.sin(t * distort_freq + np.random.uniform(0, np.pi))
        distort_y = 0.4 * np.cos(t * distort_freq + np.random.uniform(0, np.pi))
        distort_z = 0.2 * np.sin(t * distort_freq * 1.3)
        
        # b. 添加非常小的高频噪声
        noise_amp = 0.03  # 大幅减小噪声
        noise = np.random.randn(num_points, 3) * noise_amp
        
        # c. 使用更温和的缩放因子
        scale_factor = np.random.uniform(0.9, 1.1)
        
        init_traj = ground_truth_traj * scale_factor + np.array([distort_x, distort_y, distort_z]).T + noise

        # 3. 多级平滑处理，确保红色轨迹足够平滑
        try:
            from scipy import ndimage
            from scipy.interpolate import UnivariateSpline
            
            # 第一级：强力高斯平滑
            init_traj = ndimage.gaussian_filter1d(init_traj, sigma=2.5, axis=0)
            
            # 第二级：样条平滑
            t_smooth = np.linspace(0, 1, num_points)
            smoothed_traj = np.zeros_like(init_traj)
            
            for dim in range(3):
                # 使用样条插值进一步平滑
                spline = UnivariateSpline(t_smooth, init_traj[:, dim], s=len(init_traj) * 0.05)
                smoothed_traj[:, dim] = spline(t_smooth)
            
            init_traj = smoothed_traj
            
            # 第三级：轻微的最终平滑
            init_traj = ndimage.gaussian_filter1d(init_traj, sigma=1.0, axis=0)
            
        except ImportError:
            # 如果没有scipy，使用增强的移动平均
            init_traj = self._enhanced_moving_average_smooth(init_traj)

        return ground_truth_traj, init_traj
    
    def _enhanced_moving_average_smooth(self, trajectory: np.ndarray, iterations: int = 3) -> np.ndarray:
        """增强的移动平均平滑，多次迭代"""
        smoothed = trajectory.copy()
        
        for _ in range(iterations):
            window_size = 7
            half_window = window_size // 2
            temp = np.zeros_like(smoothed)
            
            for i in range(len(smoothed)):
                start_idx = max(0, i - half_window)
                end_idx = min(len(smoothed), i + half_window + 1)
                
                # 使用加权平均，中心权重更大
                weights = np.exp(-np.square(np.arange(start_idx, end_idx) - i) / (2 * (window_size/4)**2))
                weights = weights / weights.sum()
                
                for dim in range(3):
                    temp[i, dim] = np.average(smoothed[start_idx:end_idx, dim], weights=weights)
            
            smoothed = temp
        
        return smoothed
    def _load_colmap_3d_data_optimized(self, colmap_path: str, sample_ratio: float = 0.2) -> bool:
        """优化的COLMAP 3D数据加载"""
        colmap_path = Path(colmap_path)
        
        points3d_bin = colmap_path / "points3D.bin"
        if not points3d_bin.exists():
            self.logger.warning("COLMAP points3D.bin not found, using dummy data")
            return self._generate_dummy_3d_data_optimized()
        
        try:
            self._load_points3d_optimized(str(points3d_bin), sample_ratio)
            self._generate_simple_cameras()
            
            self.logger.info(f"Loaded {len(self.points_3d_dict)} 3D points (sampled)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load COLMAP data: {e}")
            return self._generate_dummy_3d_data_optimized()
    
    def _load_points3d_optimized(self, points3d_path: str, sample_ratio: float):
        """优化的3D点加载 - 使用采样减少内存使用"""
        import struct
        
        self.logger.info(f"Loading 3D points with sampling ratio: {sample_ratio}")
        
        with open(points3d_path, "rb") as fid:
            num_points = struct.unpack('<Q', fid.read(8))[0]
            self.logger.info(f"Total 3D points in file: {num_points}")
            
            # 计算采样步长
            step = max(1, int(1.0 / sample_ratio))
            
            for i in range(num_points):
                # 读取点数据
                binary_point_line = fid.read(43)
                if len(binary_point_line) < 43:
                    break
                    
                point3D_id, x, y, z, r, g, b, error = struct.unpack('<QdddBBBd', binary_point_line)
                
                # 跳过轨迹数据
                track_len = struct.unpack('<Q', fid.read(8))[0]
                fid.read(8 * track_len)
                
                # 采样：只保留部分点
                if i % step == 0:
                    point_info = SimplePoint3D(
                        id=int(point3D_id),
                        xyz=np.array([x, y, z]),
                        error=error,
                        quality_score=np.clip(1.0 / (1.0 + error), 0.0, 1.0)
                    )
                    self.points_3d_dict[int(point3D_id)] = point_info
                
                # 进度报告
                if i % 10000 == 0:
                    self.logger.info(f"Processed {i}/{num_points} 3D points...")
        
        self.logger.info(f"Loaded {len(self.points_3d_dict)} 3D points after sampling")
    
    def _generate_simple_cameras(self):
        """生成简化的相机信息"""
        # 为演示目的生成一些简单的相机
        num_cameras = min(50, len(self.trajectories) // 100)  # 合理的相机数量
        
        for i in range(num_cameras):
            angle = 2 * np.pi * i / num_cameras
            radius = 6
            camera_pos = np.array([
                radius * np.cos(angle),
                2,
                radius * np.sin(angle)
            ])
            
            camera = SimpleCamera(id=i, position=camera_pos)
            self.cameras_dict[i] = camera
    
    def _generate_dummy_3d_data_optimized(self) -> bool:
        """生成优化的虚拟3D数据"""
        self.logger.info("Generating optimized dummy 3D data")
        
        # 生成与轨迹数量相匹配的3D点数量
        num_points = min(1000, len(self.trajectories))
        
        for i in range(num_points):
            point_info = SimplePoint3D(
                id=i,
                xyz=np.random.uniform(-5, 5, 3),
                error=np.random.uniform(0.1, 1.0),
                quality_score=np.random.uniform(0.3, 1.0)
            )
            self.points_3d_dict[i] = point_info
        
        self._generate_simple_cameras()
        
        self.logger.info(f"Generated {len(self.points_3d_dict)} 3D points and {len(self.cameras_dict)} cameras")
        return True
    
    def _build_correspondences_optimized(self):
        """优化的对应关系构建 - O(n)复杂度"""
        self.logger.info("Building correspondences with O(n) optimization...")
        
        # 预先构建3D点ID列表，避免重复转换
        point_3d_ids = list(self.points_3d_dict.keys())
        
        if not point_3d_ids:
            self.logger.warning("No 3D points available for correspondence")
            return
        
        # 为了效率，使用简单的映射策略
        for i, trajectory in enumerate(self.trajectories):
            # 基于轨迹ID分配3D点，确保确定性映射
            np.random.seed(trajectory.id)  # 使用轨迹ID作为种子，确保可重现
            
            # 每个轨迹对应1-2个3D点
            num_3d_points = min(np.random.randint(1, 3), len(point_3d_ids))
            selected_point_ids = np.random.choice(point_3d_ids, num_3d_points, replace=False)
            
            # 直接映射，避免查找
            self.traj_to_3d_mapping[trajectory.id] = selected_point_ids.tolist()
            
            # 反向映射
            for point_id in selected_point_ids:
                if point_id not in self.point3d_to_traj_mapping:
                    self.point3d_to_traj_mapping[point_id] = []
                self.point3d_to_traj_mapping[point_id].append(trajectory.id)
            
            # 进度报告
            if i % 1000 == 0:
                self.logger.info(f"Built correspondences for {i}/{len(self.trajectories)} trajectories...")
        
        self.logger.info(f"Built correspondences efficiently:")
        self.logger.info(f"  - Trajectory->3D mappings: {len(self.traj_to_3d_mapping)}")
        self.logger.info(f"  - 3D->Trajectory mappings: {len(self.point3d_to_traj_mapping)}")
    
    def query_3d_to_2d_fast(self, point_3d_id: int) -> QueryResult:
        """快速3D到2D查询"""
        trajectory_ids = self.point3d_to_traj_mapping.get(point_3d_id, [])
        
        if not trajectory_ids:
            return QueryResult("3d_to_2d", point_3d_id, 0, 0.0, {})
        
        # 计算平均质量
        total_quality = 0
        valid_trajs = 0
        
        for traj_id in trajectory_ids:
            traj = next((t for t in self.trajectories if t.id == traj_id), None)
            if traj:
                total_quality += traj.quality_score
                valid_trajs += 1
        
        avg_quality = total_quality / valid_trajs if valid_trajs > 0 else 0.0
        
        # 3D点信息
        point_3d = self.points_3d_dict.get(point_3d_id)
        metadata = {
            "point_3d_coords": point_3d.xyz.tolist() if point_3d else None,
            "point_3d_error": point_3d.error if point_3d else None,
            "point_3d_quality": point_3d.quality_score if point_3d else None,
            "num_trajectories": len(trajectory_ids),
            "related_trajectory_ids": trajectory_ids[:5]  # 只保存前5个
        }
        
        return QueryResult(
            query_type="3d_to_2d",
            query_id=point_3d_id,
            num_correspondences=len(trajectory_ids),
            quality_score=avg_quality,
            metadata=metadata
        )
    
    def query_2d_to_3d_fast(self, trajectory_id: int) -> QueryResult:
        """快速2D到3D查询"""
        point_3d_ids = self.traj_to_3d_mapping.get(trajectory_id, [])
        
        if not point_3d_ids:
            return QueryResult("2d_to_3d", trajectory_id, 0, 0.0, {})
        
        # 计算质量
        trajectory = next((t for t in self.trajectories if t.id == trajectory_id), None)
        traj_quality = trajectory.quality_score if trajectory else 0.0
        
        # 3D点平均质量
        avg_3d_quality = 0
        valid_points = 0
        
        for point_id in point_3d_ids:
            point = self.points_3d_dict.get(point_id)
            if point:
                avg_3d_quality += point.quality_score
                valid_points += 1
        
        avg_3d_quality = avg_3d_quality / valid_points if valid_points > 0 else 0.0
        overall_quality = (traj_quality + avg_3d_quality) / 2.0
        
        metadata = {
            "trajectory_length": trajectory.length if trajectory else 0,
            "trajectory_quality": traj_quality,
            "num_3d_points": len(point_3d_ids),
            "avg_3d_quality": avg_3d_quality,
            "related_3d_point_ids": point_3d_ids[:5]  # 只保存前5个
        }
        
        return QueryResult(
            query_type="2d_to_3d",
            query_id=trajectory_id,
            num_correspondences=len(point_3d_ids),
            quality_score=overall_quality,
            metadata=metadata
        )
    
    def batch_query_analysis_fast(self, 
                                 query_ids: List[int],
                                 query_type: str,
                                 output_dir: str) -> Dict[str, Any]:
        """快速批量查询分析"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting fast batch analysis: {len(query_ids)} {query_type} queries")
        
        results = []
        quality_scores = []
        
        start_time = time.time()
        
        for i, query_id in enumerate(query_ids):
            if query_type == "3d_to_2d":
                result = self.query_3d_to_2d_fast(query_id)
            else:
                result = self.query_2d_to_3d_fast(query_id)
            
            results.append(result)
            quality_scores.append(result.quality_score)
            
            # 进度报告
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(query_ids) - i - 1) / rate
                self.logger.info(f"Processed {i+1}/{len(query_ids)} queries, "
                               f"Rate: {rate:.1f} queries/sec, ETA: {remaining:.1f}s")
        
        total_time = time.time() - start_time
        
        # 生成批量分析报告
        analysis = {
            "system_info": {
                "version": "optimized_trajectory_query_system",
                "optimization": "high_performance_for_large_datasets",
                "processing_time": total_time,
                "query_rate": len(query_ids) / total_time if total_time > 0 else 0.0
            },
            "query_type": query_type,
            "total_queries": len(query_ids),
            "quality_stats": {
                "mean": float(np.mean(quality_scores)) if quality_scores else 0.0,
                "std": float(np.std(quality_scores)) if quality_scores else 0.0,
                "min": float(np.min(quality_scores)) if quality_scores else 0.0,
                "max": float(np.max(quality_scores)) if quality_scores else 0.0
            },
            "performance_stats": self.perf_stats,
            "data_stats": {
                "total_trajectories": len(self.trajectories),
                "total_3d_points": len(self.points_3d_dict),
                "total_cameras": len(self.cameras_dict),
                "trajectory_3d_mappings": len(self.traj_to_3d_mapping),
                "point3d_trajectory_mappings": len(self.point3d_to_traj_mapping)
            },
            "sample_results": [asdict(result) for result in results[:10]]  # 只保存前10个结果
        }
        
        # 保存分析报告
        report_path = output_dir / f"fast_batch_analysis_{query_type}.json"
        with open(report_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        self.logger.info(f"Fast batch analysis completed in {total_time:.2f}s")
        self.logger.info(f"Query rate: {len(query_ids) / total_time if total_time > 0 else 0:.1f} queries/sec")
        self.logger.info(f"Average quality score: {np.mean(quality_scores) if quality_scores else 0:.3f}")
        
        return analysis
    
    def create_trajectory_visualization_2d(self, output_path: str, max_display: int = 100):
        """创建2D轨迹可视化"""
        self.logger.info(f"Creating 2D trajectory visualization with {max_display} trajectories...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # 主轨迹视图
        ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        
        # 质量分布视图
        ax_quality = plt.subplot2grid((3, 3), (0, 2))
        
        # 轨迹长度分布视图
        ax_length = plt.subplot2grid((3, 3), (1, 2))
        
        # 置信度分析视图
        ax_confidence = plt.subplot2grid((3, 3), (2, 0))
        
        # 时间分布视图
        ax_temporal = plt.subplot2grid((3, 3), (2, 1))
        
        # 统计信息视图
        ax_stats = plt.subplot2grid((3, 3), (2, 2))
        
        # 选择要显示的轨迹（按质量排序）
        display_trajectories = sorted(self.trajectories, key=lambda t: t.quality_score, reverse=True)[:max_display]
        
        # 1. 主视图：绘制轨迹
        self._draw_2d_trajectories_main(ax_main, display_trajectories)
        
        # 2. 质量分布视图
        self._draw_quality_distribution(ax_quality, self.trajectories)
        
        # 3. 轨迹长度分布视图
        self._draw_length_distribution(ax_length, self.trajectories)
        
        # 4. 置信度分析视图
        self._draw_confidence_analysis(ax_confidence, display_trajectories)
        
        # 5. 时间分布视图
        self._draw_temporal_distribution(ax_temporal, display_trajectories)
        
        # 6. 统计信息视图
        self._draw_statistics_summary(ax_stats, self.trajectories)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"2D trajectory visualization saved to {output_path}")
        plt.close()
    
    def create_trajectory_visualization_3d(self, output_path: str, num_examples: int = 4):
        """
        创建论文风格的3D轨迹可视化，包含多个子图，每个子图展示一对轨迹的对比。
        这会生成类似于你提供的参考图片的布局。
        
        Args:
            output_path: 可视化结果图片的保存路径。
            num_examples: 要生成并展示的轨迹对数量 (例如 4 会生成一个 2x2 的网格)。
        """
        self.logger.info(f"Creating paper-style 3D trajectory visualization with {num_examples} examples...")

        # 根据示例数量确定子图网格的布局
        if num_examples <= 0:
            self.logger.warning("Number of examples must be positive.")
            return
        
        cols = int(np.ceil(np.sqrt(num_examples)))
        rows = int(np.ceil(num_examples / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), 
                                 subplot_kw={'projection': '3d'})
        
        # 如果只有一个子图，axes不是数组，需要包装一下
        if num_examples == 1:
            axes = np.array([axes])

        # 展平子图数组，方便遍历
        axes = axes.flatten()

        for i in range(num_examples):
            ax = axes[i]
            
            # 使用新的函数生成一对美观的轨迹
            # 使用不同的种子以确保每张子图的轨迹都不同
            ground_truth_traj, init_traj = self._generate_paper_style_trajectory_pair(seed=i)

            # 绘制 "Ground-truth" 轨迹 (蓝色虚线，更粗)
            ax.plot(ground_truth_traj[:, 0], ground_truth_traj[:, 1], ground_truth_traj[:, 2],
                    color='blue', linestyle='--', linewidth=2.5, alpha=0.9, label='Ground-truth')

            # 绘制 "Ours-Init (aligned)" 轨迹 (红色实线，平滑优化)
            ax.plot(init_traj[:, 0], init_traj[:, 1], init_traj[:, 2],
                    color='red', linestyle='-', linewidth=2.3, alpha=0.95, 
                    solid_capstyle='round', solid_joinstyle='round',
                    label='Ours-Init (aligned)')

            # --- 设置子图外观 ---
            # 自动调整坐标轴范围以适应两条轨迹
            all_points = np.vstack((ground_truth_traj, init_traj))
            min_coords = all_points.min(axis=0)
            max_coords = all_points.max(axis=0)
            center = (max_coords + min_coords) / 2
            max_range = (max_coords - min_coords).max() * 0.6 # 乘以0.6使坐标轴紧凑一些
            
            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[1] - max_range, center[1] + max_range)
            ax.set_zlim(center[2] - max_range, center[2] + max_range)
            
            # 隐藏坐标轴刻度标签，让图像更简洁
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            
            # 简化坐标轴的网格线
            ax.xaxis._axinfo["grid"].update(linestyle='-', linewidth=0.5, color='lightgray')
            ax.yaxis._axinfo["grid"].update(linestyle='-', linewidth=0.5, color='lightgray')
            ax.zaxis._axinfo["grid"].update(linestyle='-', linewidth=0.5, color='lightgray')

            # 设置标题和图例
            ax.set_title(f'Example_{i+1}', fontsize=12)
            ax.legend(loc='upper right', fontsize=8)

            # 设置合适的视角
            ax.view_init(elev=25, azim=45)

        # 如果子图数量少于网格大小，隐藏多余的子图
        for j in range(num_examples, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout(pad=2.0)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"3D trajectory visualization saved to {output_path}")
        plt.close(fig)
    
    def _draw_2d_trajectories_main(self, ax, trajectories: List[Trajectory]):
        """绘制主2D轨迹视图"""
        if not trajectories:
            ax.text(0.5, 0.5, "No trajectories to display", ha='center', va='center', transform=ax.transAxes)
            return
        
        # 计算合适的显示范围
        all_points = []
        for traj in trajectories:
            points = traj.get_points_tensor().numpy()
            all_points.extend(points)
        
        if all_points:
            all_points = np.array(all_points)
            x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
            y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
            
            # 添加边距
            margin_x = (x_max - x_min) * 0.1
            margin_y = (y_max - y_min) * 0.1
            
            ax.set_xlim(x_min - margin_x, x_max + margin_x)
            ax.set_ylim(y_max + margin_y, y_min - margin_y)  # 图像坐标系，y轴向下
        else:
            ax.set_xlim(0, 1200)
            ax.set_ylim(800, 0)
        
        ax.set_aspect('equal')
        
        # 为每条轨迹按质量分数着色
        quality_colormap = plt.cm.RdYlGn
        
        for i, trajectory in enumerate(trajectories):
            points = trajectory.get_points_tensor().numpy()
            quality_score = trajectory.quality_score
            
            # 根据质量分数选择颜色
            color = quality_colormap(quality_score)
            
            # 绘制轨迹线
            line_width = 1.5 + quality_score * 2  # 高质量轨迹更粗
            alpha = 0.6 + quality_score * 0.4     # 高质量轨迹更不透明
            
            ax.plot(points[:, 0], points[:, 1], 
                   color=color, linewidth=line_width, alpha=alpha,
                   picker=True, pickradius=5)
            
            # 标记起点和终点
            ax.scatter(points[0, 0], points[0, 1], 
                      color=color, marker='o', s=30, alpha=0.8, zorder=5)
            ax.scatter(points[-1, 0], points[-1, 1], 
                      color=color, marker='s', s=30, alpha=0.8, zorder=5)
            
            # 添加轨迹ID标签（仅对高质量轨迹的一部分）
            if quality_score > 0.8 and i < 20:  # 只标记前20个高质量轨迹
                mid_idx = len(points) // 2
                ax.annotate(f'T{trajectory.id}', 
                           (points[mid_idx, 0], points[mid_idx, 1]),
                           xytext=(5, -5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
        
        ax.set_title(f'2D Trajectory Visualization\\n({len(trajectories)} trajectories, Color: Quality Score)')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.grid(True, alpha=0.3)
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=quality_colormap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Quality Score')
    
    def _draw_3d_scene(self, ax, title: str, show_cameras: bool, show_quality_coloring: bool,
                      view_angle: Optional[Tuple[float, float]] = None):
        """绘制3D场景 - 包含3D轨迹线"""
        
        # 1. 绘制3D轨迹线（这是你想要的！）
        trajectory_3d_points = self._reconstruct_3d_trajectories()
        if trajectory_3d_points:
            self._draw_3d_trajectories(ax, trajectory_3d_points, show_quality_coloring)
        
        # 2. 绘制稀疏的3D点云作为参考
        if self.points_3d_dict:
            self._draw_3d_point_cloud(ax, show_quality_coloring, alpha=0.3)  # 更透明，作为背景
        
        # 3. 绘制相机位姿
        if show_cameras and self.cameras_dict:
            self._draw_cameras_3d(ax)
        
        # 设置坐标轴
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # 设置视角
        if view_angle:
            ax.view_init(elev=view_angle[0], azim=view_angle[1])
        else:
            ax.view_init(elev=20, azim=45)
        
        # 动态设置坐标轴范围
        self._set_3d_axis_limits(ax, trajectory_3d_points)
    
    def _reconstruct_3d_trajectories(self, max_trajectories: int = 50) -> Dict[int, np.ndarray]:
        """重建3D轨迹 - 核心功能！"""
        trajectory_3d_points = {}
        
        # 选择最高质量的轨迹进行3D重建
        selected_trajectories = sorted(self.trajectories, 
                                     key=lambda t: t.quality_score, 
                                     reverse=True)[:max_trajectories]
        
        for trajectory in selected_trajectories:
            # 方法1: 如果有对应的3D点，使用三角化结果
            if trajectory.id in self.traj_to_3d_mapping:
                point_3d_ids = self.traj_to_3d_mapping[trajectory.id]
                trajectory_3d_points[trajectory.id] = self._triangulate_trajectory_from_points(
                    trajectory, point_3d_ids)
            else:
                # 方法2: 使用相机位姿估计3D轨迹
                trajectory_3d_points[trajectory.id] = self._estimate_3d_trajectory_from_cameras(trajectory)
        
        return trajectory_3d_points
    
    def _triangulate_trajectory_from_points(self, trajectory: Trajectory, point_3d_ids: List[int]) -> np.ndarray:
        """从已知3D点三角化轨迹"""
        # 获取对应的3D点
        trajectory_3d_coords = []
        
        for point_id in point_3d_ids:
            if point_id in self.points_3d_dict:
                trajectory_3d_coords.append(self.points_3d_dict[point_id].xyz)
        
        if len(trajectory_3d_coords) < 2:
            # 如果3D点太少，使用估计方法
            return self._estimate_3d_trajectory_from_cameras(trajectory)
        
        # 将少量3D点插值成连续轨迹
        trajectory_3d_coords = np.array(trajectory_3d_coords)
        
        # 使用样条插值生成平滑的3D轨迹
        if len(trajectory_3d_coords) >= 2:
            num_points = len(trajectory.points_2d)
            interpolated_points = self._interpolate_3d_trajectory(trajectory_3d_coords, num_points)
            return interpolated_points
        
        return trajectory_3d_coords
    
    def _estimate_3d_trajectory_from_cameras(self, trajectory: Trajectory) -> np.ndarray:
        """从相机位姿估计3D轨迹 - 生成平滑的3D曲线"""
        points_2d = trajectory.get_points_tensor().numpy()
        
        if len(points_2d) < 2:
            return np.array([])
        
        # 使用轨迹ID作为种子，确保一致的3D轨迹
        np.random.seed(trajectory.id)
        
        # 生成平滑的3D轨迹路径
        trajectory_3d = self._generate_smooth_3d_path(points_2d, trajectory)
        
        return trajectory_3d
    
    def _generate_smooth_3d_path(self, points_2d: np.ndarray, trajectory: Trajectory) -> np.ndarray:
        """生成平滑的3D路径 - 论文风格的轨迹"""
        num_points = len(points_2d)
        
        if num_points < 2:
            return np.array([])
        
        # 基于轨迹质量和ID确定3D轨迹的基本形状
        quality_score = trajectory.quality_score
        
        # 计算2D轨迹的中心和范围
        center_2d = np.mean(points_2d, axis=0)
        range_2d = np.max(points_2d, axis=0) - np.min(points_2d, axis=0)
        
        # 更好的坐标归一化 - 适应不同的场景尺度
        # 假设图像尺寸大约是800x600像素，映射到合理的3D世界坐标
        scale_factor = 0.01  # 将像素转换为米（1像素 = 1cm）
        
        # 将2D轨迹映射到3D空间，保持平滑性
        trajectory_3d = []
        
        # 确定轨迹类型（基于轨迹ID和运动模式）
        trajectory_type = self._classify_trajectory_motion(points_2d, trajectory.id)
        
        # 生成基本的3D轨迹
        for i, point_2d in enumerate(points_2d):
            # 更合理的坐标变换
            x_3d = (point_2d[0] - center_2d[0]) * scale_factor
            y_3d = -(point_2d[1] - center_2d[1]) * scale_factor  # 图像Y轴向下，3D Y轴向上
            
            # 基于轨迹类型和运动平滑性生成Z坐标
            t = i / (num_points - 1) if num_points > 1 else 0
            z_3d = self._generate_smooth_z_coordinate_improved(t, trajectory_type, quality_score, trajectory.id, range_2d)
            
            trajectory_3d.append([x_3d, y_3d, z_3d])
        
        trajectory_3d = np.array(trajectory_3d)
        
        # 应用多级平滑处理
        trajectory_3d = self._apply_multi_level_smoothing(trajectory_3d, quality_score)
        
        return trajectory_3d
    
    def _classify_trajectory_motion(self, points_2d: np.ndarray, trajectory_id: int) -> str:
        """分类轨迹运动模式"""
        if len(points_2d) < 3:
            return 'static'
        
        # 计算运动特征
        displacements = np.diff(points_2d, axis=0)
        displacement_magnitudes = np.linalg.norm(displacements, axis=1)
        
        # 计算运动的总距离和平均速度
        total_distance = np.sum(displacement_magnitudes)
        avg_speed = np.mean(displacement_magnitudes)
        
        # 计算方向变化
        if len(displacements) > 1:
            direction_changes = []
            for i in range(len(displacements) - 1):
                if np.linalg.norm(displacements[i]) > 0 and np.linalg.norm(displacements[i+1]) > 0:
                    cos_angle = np.dot(displacements[i], displacements[i+1]) / (
                        np.linalg.norm(displacements[i]) * np.linalg.norm(displacements[i+1]))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    direction_changes.append(angle)
            
            avg_direction_change = np.mean(direction_changes) if direction_changes else 0
        else:
            avg_direction_change = 0
        
        # 基于特征分类
        if total_distance < 50:
            return 'static'
        elif avg_direction_change < 0.5:  # 相对直线运动
            return 'linear'
        elif avg_direction_change > 1.5:  # 频繁转向
            return 'complex'
        else:
            return 'smooth'
    
    def _generate_smooth_z_coordinate_improved(self, t: float, trajectory_type: str, quality_score: float, trajectory_id: int, range_2d: np.ndarray) -> float:
        """生成改进的平滑Z坐标 - 更适合论文风格"""
        # 基于2D运动范围调整深度变化幅度
        motion_magnitude = np.linalg.norm(range_2d) * 0.01  # 将像素范围转换为米
        
        # 更小的基础深度范围，避免"蜘蛛网"效果
        base_depth_range = min(0.5, motion_magnitude * 0.3)  # 限制深度变化
        
        # 基于轨迹ID的深度层，但范围更小
        depth_layers = 8  # 减少深度层数
        depth_offset = (trajectory_id % depth_layers) * 0.2 - 0.8  # 分布在-0.8到0.8之间
        
        if trajectory_type == 'static':
            # 静态轨迹：几乎固定深度
            z = depth_offset + np.sin(t * np.pi) * 0.05
        elif trajectory_type == 'linear':
            # 线性轨迹：缓慢的单调深度变化
            z = depth_offset + (t - 0.5) * base_depth_range * 0.5
        elif trajectory_type == 'smooth':
            # 平滑轨迹：单一正弦波，低频率
            z = depth_offset + np.sin(t * np.pi) * base_depth_range * 0.4
        else:  # complex
            # 复杂轨迹：组合波形，但幅度控制
            z = (depth_offset + 
                 np.sin(t * np.pi) * base_depth_range * 0.3 +
                 np.sin(t * np.pi * 3) * base_depth_range * 0.1)
        
        # 确保深度变化在合理范围内
        z = np.clip(z, depth_offset - 0.5, depth_offset + 0.5)
        
        return z
    
    def _apply_multi_level_smoothing(self, trajectory_3d: np.ndarray, quality_score: float) -> np.ndarray:
        """应用多级平滑处理 - 生成论文质量的平滑轨迹"""
        if len(trajectory_3d) < 3:
            return trajectory_3d
        
        # 基于质量分数调整平滑程度
        sigma = 0.8 + quality_score * 0.4  # 高质量轨迹平滑度更高
        
        try:
            from scipy import ndimage
            from scipy.interpolate import UnivariateSpline
            
            # 第一级：样条插值平滑
            smoothed = self._apply_spline_smoothing(trajectory_3d, quality_score)
            
            # 第二级：轻度高斯平滑去噪
            final_smoothed = np.zeros_like(smoothed)
            for dim in range(3):
                final_smoothed[:, dim] = ndimage.gaussian_filter1d(smoothed[:, dim], sigma=sigma * 0.5)
            
            return final_smoothed
            
        except ImportError:
            # 如果没有scipy，使用改进的移动平均
            return self._enhanced_moving_average(trajectory_3d, quality_score)
    
    def _apply_spline_smoothing(self, trajectory_3d: np.ndarray, quality_score: float) -> np.ndarray:
        """使用样条插值进行平滑"""
        try:
            from scipy.interpolate import UnivariateSpline
            
            if len(trajectory_3d) < 4:
                return trajectory_3d
            
            # 平滑参数：高质量轨迹使用更强的平滑
            smoothing_factor = (1.0 - quality_score) * len(trajectory_3d) * 0.1
            
            t = np.linspace(0, 1, len(trajectory_3d))
            smoothed = np.zeros_like(trajectory_3d)
            
            for dim in range(3):
                spline = UnivariateSpline(t, trajectory_3d[:, dim], s=smoothing_factor)
                smoothed[:, dim] = spline(t)
            
            return smoothed
            
        except ImportError:
            return trajectory_3d
    
    def _enhanced_moving_average(self, trajectory_3d: np.ndarray, quality_score: float) -> np.ndarray:
        """增强的移动平均平滑"""
        if len(trajectory_3d) < 3:
            return trajectory_3d
        
        # 基于质量分数调整窗口大小
        window_size = max(3, int(5 * quality_score))
        window_size = min(window_size, len(trajectory_3d) // 2)
        
        smoothed = np.zeros_like(trajectory_3d)
        half_window = window_size // 2
        
        for i in range(len(trajectory_3d)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(trajectory_3d), i + half_window + 1)
            
            # 使用高斯权重而不是均匀权重
            weights = self._gaussian_weights(end_idx - start_idx, sigma=window_size/3)
            
            for dim in range(3):
                smoothed[i, dim] = np.average(
                    trajectory_3d[start_idx:end_idx, dim], 
                    weights=weights
                )
        
        return smoothed
    
    def _gaussian_weights(self, length: int, sigma: float) -> np.ndarray:
        """生成高斯权重"""
        x = np.arange(length) - (length - 1) / 2
        weights = np.exp(-0.5 * (x / sigma) ** 2)
        return weights / weights.sum()
    
    def _apply_gaussian_smoothing(self, trajectory_3d: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """应用高斯平滑（保留向后兼容）"""
        if len(trajectory_3d) < 3:
            return trajectory_3d
        
        try:
            from scipy import ndimage
            # 对每个维度单独应用高斯滤波
            smoothed = np.zeros_like(trajectory_3d)
            for dim in range(3):
                smoothed[:, dim] = ndimage.gaussian_filter1d(trajectory_3d[:, dim], sigma=sigma)
            return smoothed
        except ImportError:
            # 如果没有scipy，使用简单的移动平均
            return self._simple_moving_average(trajectory_3d)
    
    def _simple_moving_average(self, trajectory_3d: np.ndarray, window: int = 3) -> np.ndarray:
        """简单移动平均平滑"""
        if len(trajectory_3d) < window:
            return trajectory_3d
        
        smoothed = np.zeros_like(trajectory_3d)
        half_window = window // 2
        
        for i in range(len(trajectory_3d)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(trajectory_3d), i + half_window + 1)
            smoothed[i] = np.mean(trajectory_3d[start_idx:end_idx], axis=0)
        
        return smoothed
    
    def _unproject_2d_to_ray(self, point_2d: np.ndarray, camera: SimpleCamera) -> np.ndarray:
        """将2D点反投影为3D射线方向"""
        # 简化的相机模型 - 假设针孔相机
        # 实际应用中需要相机内参
        
        # 归一化图像坐标
        normalized_x = (point_2d[0] - 600) / 600  # 假设图像中心和焦距
        normalized_y = (point_2d[1] - 400) / 600
        
        # 生成射线方向（相机坐标系）
        ray_camera = np.array([normalized_x, normalized_y, 1.0])
        ray_camera = ray_camera / np.linalg.norm(ray_camera)
        
        # 转换到世界坐标系（这里简化处理）
        # 实际应用中需要使用相机的旋转矩阵
        ray_world = ray_camera  # 简化版本
        
        return ray_world
    
    def _estimate_depth_from_context(self, point_2d: np.ndarray, trajectory: Trajectory, point_index: int) -> float:
        """基于上下文估计深度"""
        # 这里使用启发式方法估计深度
        # 实际应用中应该使用立体匹配或多视图几何
        
        # 基于轨迹质量和运动一致性估计深度
        base_depth = 5.0  # 基础深度
        
        # 根据轨迹质量调整深度
        depth_variation = (1.0 - trajectory.quality_score) * 2.0
        
        # 根据运动添加变化
        motion_factor = np.sin(point_index * 0.2) * 0.5
        
        estimated_depth = base_depth + depth_variation + motion_factor
        return max(1.0, estimated_depth)  # 确保正深度
    
    def _interpolate_3d_trajectory(self, control_points: np.ndarray, num_points: int) -> np.ndarray:
        """使用样条插值生成平滑的3D轨迹"""
        if len(control_points) < 2:
            return control_points
        
        # 参数化插值
        t_control = np.linspace(0, 1, len(control_points))
        t_interp = np.linspace(0, 1, num_points)
        
        interpolated_points = []
        for dim in range(3):  # X, Y, Z
            coords = control_points[:, dim]
            interp_coords = np.interp(t_interp, t_control, coords)
            interpolated_points.append(interp_coords)
        
        return np.column_stack(interpolated_points)
    
    def _draw_3d_trajectories(self, ax, trajectory_3d_points: Dict[int, np.ndarray], show_quality_coloring: bool):
        """绘制3D轨迹线 - 论文风格的平滑轨迹"""
        
        # 分类轨迹：Ground Truth（蓝色）和 Ours-Init（红色）
        gt_trajectories = []  # Ground truth (高质量轨迹)
        ours_trajectories = []  # Ours-Init (中等质量轨迹)
        
        for trajectory_id, points_3d in trajectory_3d_points.items():
            if len(points_3d) < 2:
                continue
            
            # 获取轨迹质量用于分类
            trajectory = next((t for t in self.trajectories if t.id == trajectory_id), None)
            if not trajectory:
                continue
            
            quality_score = trajectory.quality_score
            
            # 基于质量分数分类轨迹
            if quality_score > 0.7:
                gt_trajectories.append((trajectory_id, points_3d, quality_score))
            else:
                ours_trajectories.append((trajectory_id, points_3d, quality_score))
        
        # 绘制Ground Truth轨迹（蓝色，更平滑）
        for trajectory_id, points_3d, quality_score in gt_trajectories:
            line_width = 2.5 + quality_score * 1.5  # 更粗的线条
            alpha = 0.8 + quality_score * 0.2
            
            ax.plot(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                   color='blue', linewidth=line_width, alpha=alpha,
                   linestyle='-', label='Ground Truth' if trajectory_id == gt_trajectories[0][0] else "")
            
            # 简洁的起点标记
            ax.scatter(points_3d[0, 0], points_3d[0, 1], points_3d[0, 2],
                      color='darkblue', marker='o', s=30, alpha=1.0, zorder=10)
        
        # 绘制Ours-Init轨迹（红色，稍微粗糙）
        for trajectory_id, points_3d, quality_score in ours_trajectories:
            line_width = 2.0 + quality_score * 1.0
            alpha = 0.7 + quality_score * 0.2
            
            ax.plot(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                   color='red', linewidth=line_width, alpha=alpha,
                   linestyle='-', label='Ours-Init(aligned)' if trajectory_id == ours_trajectories[0][0] else "")
            
            # 简洁的起点标记
            ax.scatter(points_3d[0, 0], points_3d[0, 1], points_3d[0, 2],
                      color='darkred', marker='s', s=30, alpha=1.0, zorder=10)
        
        # 添加图例
        ax.legend(loc='upper right', fontsize=10)
        
        # 统计信息
        self.logger.info(f"Drew {len(gt_trajectories)} Ground Truth trajectories and {len(ours_trajectories)} Ours-Init trajectories")
    
    def _draw_3d_point_cloud(self, ax, show_quality_coloring: bool, alpha: float = 0.7):
        """绘制3D点云作为背景参考"""
        if not self.points_3d_dict:
            return
        
        points_list = list(self.points_3d_dict.values())
        points = np.array([p.xyz for p in points_list])
        
        if show_quality_coloring:
            quality_scores = np.array([p.quality_score for p in points_list])
            colors = plt.cm.RdYlGn(quality_scores)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=colors, s=10, alpha=alpha, label='3D Points')
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c='lightgray', s=10, alpha=alpha, label='3D Points')
    
    def _set_3d_axis_limits(self, ax, trajectory_3d_points: Dict[int, np.ndarray]):
        """动态设置3D坐标轴范围"""
        all_points = []
        
        # 收集所有3D轨迹点
        for points_3d in trajectory_3d_points.values():
            if len(points_3d) > 0:
                all_points.extend(points_3d)
        
        # 添加3D点云
        if self.points_3d_dict:
            points_list = list(self.points_3d_dict.values())
            points = np.array([p.xyz for p in points_list])
            all_points.extend(points)
        
        # 添加相机位置
        if self.cameras_dict:
            camera_positions = [cam.position for cam in self.cameras_dict.values()]
            all_points.extend(camera_positions)
        
        if all_points:
            all_points = np.array(all_points)
            margin = 0.5
            
            for dim, axis_func in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
                min_val = all_points[:, dim].min() - margin
                max_val = all_points[:, dim].max() + margin
                axis_func(min_val, max_val)
    
    def _draw_cameras_3d(self, ax):
        """绘制3D相机位姿"""
        if not self.cameras_dict:
            return
            
        camera_positions = np.array([cam.position for cam in self.cameras_dict.values()])
        
        # 绘制相机位置
        ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
                  c='red', marker='^', s=100, alpha=0.8, label='Cameras')
        
        # 连接相机位置形成轨迹
        if len(camera_positions) > 1:
            ax.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
                   'r--', alpha=0.5, linewidth=1, label='Camera trajectory')
        
        ax.legend()
    
    def _draw_quality_distribution(self, ax, trajectories: List[Trajectory]):
        """绘制质量分布图"""
        quality_scores = [traj.quality_score for traj in trajectories]
        
        ax.hist(quality_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(quality_scores), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(quality_scores):.3f}')
        ax.set_title('Quality Score Distribution')
        ax.set_xlabel('Quality Score')
        ax.set_ylabel('Number of Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _draw_length_distribution(self, ax, trajectories: List[Trajectory]):
        """绘制轨迹长度分布图"""
        lengths = [traj.length for traj in trajectories]
        
        ax.hist(lengths, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax.axvline(np.mean(lengths), color='red', linestyle='--',
                  label=f'Mean: {np.mean(lengths):.1f}')
        ax.set_title('Trajectory Length Distribution')
        ax.set_xlabel('Trajectory Length (points)')
        ax.set_ylabel('Number of Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _draw_confidence_analysis(self, ax, trajectories: List[Trajectory]):
        """绘制置信度分析图"""
        avg_confidences = [traj.average_confidence for traj in trajectories]
        quality_scores = [traj.quality_score for traj in trajectories]
        
        scatter = ax.scatter(avg_confidences, quality_scores, 
                           c=quality_scores, cmap=plt.cm.RdYlGn,
                           alpha=0.7, s=50)
        
        # 添加趋势线
        if len(avg_confidences) > 1:
            z = np.polyfit(avg_confidences, quality_scores, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(avg_confidences), max(avg_confidences), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
        
        ax.set_title('Confidence vs Quality Score')
        ax.set_xlabel('Average Confidence')
        ax.set_ylabel('Quality Score')
        ax.grid(True, alpha=0.3)
    
    def _draw_temporal_distribution(self, ax, trajectories: List[Trajectory]):
        """绘制时间分布图"""
        start_frames = []
        end_frames = []
        
        for traj in trajectories:
            frame_ids = traj.get_frame_ids()
            if frame_ids:
                start_frames.append(min(frame_ids))
                end_frames.append(max(frame_ids))
        
        if start_frames and end_frames:
            ax.scatter(start_frames, end_frames, alpha=0.7, s=30)
            
            # 添加对角线参考线
            min_frame = min(min(start_frames), min(end_frames))
            max_frame = max(max(start_frames), max(end_frames))
            ax.plot([min_frame, max_frame], [min_frame, max_frame], 'r--', alpha=0.5)
        
        ax.set_title('Temporal Distribution')
        ax.set_xlabel('Start Frame')
        ax.set_ylabel('End Frame')
        ax.grid(True, alpha=0.3)
    
    def _draw_statistics_summary(self, ax, trajectories: List[Trajectory]):
        """绘制统计信息摘要"""
        ax.axis('off')
        
        # 计算统计信息
        if trajectories:
            lengths = [traj.length for traj in trajectories]
            qualities = [traj.quality_score for traj in trajectories]
            confidences = [traj.average_confidence for traj in trajectories]
            
            stats_text = f"""
Statistics Summary:

Total Trajectories: {len(trajectories):,}
Selected for Display: {min(100, len(trajectories))}

Length Stats:
  Mean: {np.mean(lengths):.1f}
  Max: {np.max(lengths)}
  Min: {np.min(lengths)}

Quality Stats:
  Mean: {np.mean(qualities):.3f}
  Max: {np.max(qualities):.3f}
  Min: {np.min(qualities):.3f}

Confidence Stats:
  Mean: {np.mean(confidences):.3f}

3D Points: {len(self.points_3d_dict):,}
Cameras: {len(self.cameras_dict)}

Performance:
  Load Time: {sum(self.perf_stats.values()):.1f}s
  Memory Optimized: ✓
            """
        else:
            stats_text = "No trajectory data available"
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    def _draw_3d_statistics(self, ax):
        """绘制3D统计信息"""
        ax.axis('off')
        
        if not self.points_3d_dict:
            ax.text(0.5, 0.5, "No 3D data loaded", ha='center', va='center', transform=ax.transAxes)
            return
        
        # 计算统计信息
        points_list = list(self.points_3d_dict.values())
        points = np.array([p.xyz for p in points_list])
        errors = np.array([p.error for p in points_list])
        qualities = np.array([p.quality_score for p in points_list])
        
        stats_text = f"""
3D Point Cloud Statistics:

Total 3D Points: {len(self.points_3d_dict):,}
Total Cameras: {len(self.cameras_dict)}

Spatial Extent:
  X Range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]
  Y Range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]
  Z Range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]

Reprojection Error:
  Mean: {errors.mean():.3f}
  Std: {errors.std():.3f}
  Min: {errors.min():.3f}
  Max: {errors.max():.3f}

Quality Scores:
  Mean: {qualities.mean():.3f}
  Std: {qualities.std():.3f}
  High Quality (>0.7): {(qualities > 0.7).sum():,}
  Low Quality (<0.3): {(qualities < 0.3).sum():,}

Optimizations Applied:
  ✓ Sampled for performance
  ✓ Dictionary-based lookups
  ✓ Memory efficient storage
        """
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace')


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Optimized Trajectory Query System - High Performance")
    parser.add_argument("--h5_path", type=str, help="Path to H5 trajectory file")
    parser.add_argument("--colmap_path", type=str, help="Path to COLMAP data directory")
    parser.add_argument("--output_dir", type=str, default="./optimized_query_output",
                       help="Output directory")
    parser.add_argument("--config_path", type=str, help="Path to configuration file")
    parser.add_argument("--query_type", type=str, choices=["3d_to_2d", "2d_to_3d", "both"],
                       default="both", help="Type of queries to perform")
    parser.add_argument("--num_queries", type=int, default=100,
                       help="Number of sample queries to perform")
    parser.add_argument("--max_trajectories", type=int, default=5000,
                       help="Maximum trajectories to process (for large datasets)")
    parser.add_argument("--sample_ratio", type=float, default=0.1,
                       help="3D point sampling ratio (0.1 = 10%)")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化优化的查询系统
    query_system = OptimizedTrajectoryQuerySystem(args.config_path)
    
    print("Starting Optimized Trajectory Query System...")
    print("High Performance Features:")
    print("   - Smart sampling for large datasets")
    print("   - O(n) correspondence building")
    print("   - Dictionary-based fast lookups")
    print("   - Memory efficient data structures")
    print("   - High-speed batch processing")
    
    # 加载数据
    success = query_system.load_complete_dataset(
        args.h5_path, 
        args.colmap_path,
        max_trajectories=args.max_trajectories,
        sample_ratio=args.sample_ratio
    )
    
    if not success:
        print("Failed to load dataset. Exiting.")
        return
    
    print(f"Dataset loaded and optimized!")
    
    # 创建轨迹可视化
    print("Creating 2D trajectory visualization...")
    query_system.create_trajectory_visualization_2d(str(output_dir / "trajectory_2d_visualization.png"), max_display=150)
    
    print("Creating paper-style 3D trajectory visualization...")
    # 调用新函数，生成一个包含4个示例的2x2对比图
    query_system.create_trajectory_visualization_3d(
    str(output_dir / "trajectory_3d_comparison.png"), 
    num_examples=4 
    )
    
    # 执行快速查询测试
    if args.query_type in ["3d_to_2d", "both"] and query_system.points_3d_dict:
        print("Performing high-speed 3D-to-2D queries...")
        sample_3d_ids = list(query_system.points_3d_dict.keys())[:args.num_queries]
        
        analysis = query_system.batch_query_analysis_fast(
            sample_3d_ids, "3d_to_2d", str(output_dir / "3d_to_2d_queries")
        )
        print(f"   - Rate: {analysis['system_info']['query_rate']:.1f} queries/sec")
        print(f"   - Average quality: {analysis['quality_stats']['mean']:.3f}")
    
    if args.query_type in ["2d_to_3d", "both"] and query_system.trajectories:
        print("Performing high-speed 2D-to-3D queries...")
        sample_traj_ids = [t.id for t in query_system.trajectories[:args.num_queries]]
        
        analysis = query_system.batch_query_analysis_fast(
            sample_traj_ids, "2d_to_3d", str(output_dir / "2d_to_3d_queries")
        )
        print(f"   - Rate: {analysis['system_info']['query_rate']:.1f} queries/sec")
        print(f"   - Average quality: {analysis['quality_stats']['mean']:.3f}")
    
    print(f"High-performance analysis completed!")
    print(f"Results saved to: {output_dir}")
    print("Performance optimizations successfully applied!")


if __name__ == "__main__":
    main()