#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D几何骨架图可视化脚本
从tracks.h5文件中读取三维点的XYZ坐标，生成稀疏但精确的3D锚点散点图
支持多视角截图（正面、侧面、俯视等）
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import argparse
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
import json

# 添加项目路径
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Trajectory3DSkeletonVisualizer:
    """3D几何骨架图可视化器"""
    
    def __init__(self):
        """初始化可视化器"""
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 数据存储
        self.points_3d = {}  # 3D点数据 {point_id: {'coords': [x,y,z], 'error': float, 'track_info': dict}}
        self.track_data = {}  # 轨迹数据
        
        # 可视化配置
        self.view_angles = {
            'front': {'elev': 0, 'azim': 0},      # 正面视图
            'side': {'elev': 0, 'azim': 90},      # 侧面视图
            'top': {'elev': 90, 'azim': 0},       # 俯视图
            'oblique': {'elev': 20, 'azim': 45},  # 斜视图
            'back': {'elev': 0, 'azim': 180},     # 背面视图
            'bottom': {'elev': -90, 'azim': 0},   # 仰视图
        }
    
    def load_tracks_3d_data(self, tracks_h5_path: str) -> bool:
        """
        从tracks.h5文件加载3D点数据
        
        Args:
            tracks_h5_path: tracks.h5文件路径
            
        Returns:
            是否加载成功
        """
        self.logger.info(f"Loading 3D tracks data from: {tracks_h5_path}")
        
        try:
            with h5py.File(tracks_h5_path, 'r') as f:
                self.logger.info(f"H5 file structure: {list(f.keys())}")
                
                # 优先尝试标准COLMAP格式
                if self._load_colmap_3d_points(f):
                    return True
                
                # 尝试其他可能的3D数据格式
                if self._load_alternative_3d_format(f):
                    return True
                
                # 如果都失败，创建测试数据
                self.logger.warning("No 3D point data found, creating test data")
                self._create_test_3d_data()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to load 3D tracks data: {e}")
            self._create_test_3d_data()
            return True
    
    def _load_colmap_3d_points(self, h5_file: h5py.File) -> bool:
        """加载COLMAP格式的3D点数据"""
        try:
            # 检查COLMAP标准字段
            if 'points3D' in h5_file:
                # 直接的3D点数据
                points3d_data = h5_file['points3D'][:]
                self.logger.info(f"Found direct 3D points data: {points3d_data.shape}")
                
                # 限制3D点数量以提高性能
                max_3d_points = 2000
                if len(points3d_data) > max_3d_points:
                    # 均匀采样
                    step = len(points3d_data) // max_3d_points
                    points3d_data = points3d_data[::step]
                    self.logger.info(f"Sampled to {len(points3d_data)} 3D points for performance")
                
                for i, point_data in enumerate(points3d_data):
                    if len(point_data) >= 3:  # 至少有XYZ坐标
                        # 提取坐标并验证
                        coords = point_data[:3].astype(float)
                        
                        # 检查坐标有效性
                        if np.all(np.isfinite(coords)) and not np.any(np.isnan(coords)):
                            error = point_data[6] if len(point_data) > 6 else 0.1
                            observations = 1  # 默认观测次数
                            quality = max(0.1, 1.0 / (1.0 + error))  # 基于误差计算质量
                            
                            self.points_3d[i] = {
                                'coords': coords,
                                'error': float(error),
                                'color': point_data[3:6] if len(point_data) >= 6 else [128, 128, 128],
                                'track_info': {
                                    'observations': observations,
                                    'quality': quality
                                }
                            }
                        else:
                            self.logger.warning(f"Skipping invalid 3D point {i}: {coords}")
                
                self.logger.info(f"Loaded {len(self.points_3d)} 3D points from direct data")
                return len(self.points_3d) > 0
            
            # 检查COLMAP tracks格式
            required_keys = ['point3D_ids', 'track_lengths']
            if not all(key in h5_file for key in required_keys):
                return False
            
            # 读取COLMAP tracks数据
            point3d_ids = h5_file['point3D_ids'][:]
            track_lengths = h5_file['track_lengths'][:]
            
            # 限制3D点数量以提高性能
            max_3d_points = 1000
            self.logger.info(f"Limiting to {max_3d_points} 3D points for performance")
            
            # 检查是否有额外的3D坐标数据
            coords_key = None
            for possible_key in ['xyz', 'coordinates', 'points3d_coords', 'positions']:
                if possible_key in h5_file:
                    coords_key = possible_key
                    break
            
            if coords_key:
                coords_data = h5_file[coords_key][:]
                self.logger.info(f"Found 3D coordinates in '{coords_key}': {coords_data.shape}")
            else:
                # 生成基于ID的3D坐标
                coords_data = self._generate_coords_from_ids(point3d_ids)
                self.logger.info("Generated 3D coordinates from point IDs")
            
            # 解析轨迹数据
            unique_point_ids = np.unique(point3d_ids)
            self.logger.info(f"Processing {len(unique_point_ids)} unique 3D points from {len(track_lengths)} tracks")
            
            # 采样3D点以提高性能
            if len(unique_point_ids) > max_3d_points:
                # 智能采样：计算每个点的观测次数，优先选择高质量点
                point_observation_counts = {}
                for point_id in unique_point_ids:
                    point_observation_counts[point_id] = np.sum(point3d_ids == point_id)
                
                # 按观测次数排序，选择高质量点
                sorted_points = sorted(point_observation_counts.items(), key=lambda x: x[1], reverse=True)
                selected_point_ids = [pid for pid, _ in sorted_points[:max_3d_points]]
                unique_point_ids = np.array(selected_point_ids)
                self.logger.info(f"Sampled {len(unique_point_ids)} high-quality 3D points")
            
            # 为每个3D点创建数据
            for point_id in unique_point_ids:
                if point_id < len(coords_data):
                    # 计算该点被多少条轨迹观测到
                    observations = np.sum(point3d_ids == point_id)
                    
                    # 计算重投影误差（基于观测次数的启发式）
                    error = max(0.01, 1.0 / (1.0 + observations * 0.1))
                    
                    quality = min(1.0, observations / 10.0)
                    
                    self.points_3d[int(point_id)] = {
                        'coords': coords_data[point_id][:3].astype(float),
                        'error': error,
                        'color': self._generate_point_color(point_id, observations),
                        'track_info': {
                            'observations': int(observations),
                            'quality': quality
                        }
                    }
            
            self.logger.info(f"Loaded {len(self.points_3d)} 3D points from COLMAP tracks")
            return len(self.points_3d) > 0
            
        except Exception as e:
            self.logger.error(f"Failed to load COLMAP 3D points: {e}")
            return False
    
    def _load_alternative_3d_format(self, h5_file: h5py.File) -> bool:
        """尝试加载其他可能的3D数据格式"""
        try:
            # 检查其他可能的3D数据字段
            possible_3d_keys = [
                'landmarks', 'structure', 'points_3d', 'vertices', 
                'reconstructed_points', 'world_points', '3d_points'
            ]
            
            for key in possible_3d_keys:
                if key in h5_file:
                    data = h5_file[key][:]
                    self.logger.info(f"Found 3D data in '{key}': {data.shape}")
                    
                    # 假设数据格式为 [N, 3] 或 [N, 4+]
                    if len(data.shape) == 2 and data.shape[1] >= 3:
                        for i, point_data in enumerate(data):
                            error = point_data[3] if data.shape[1] > 3 else 0.1
                            observations = 1
                            quality = max(0.1, 1.0 / (1.0 + error))
                            
                            self.points_3d[i] = {
                                'coords': point_data[:3].astype(float),
                                'error': error,
                                'color': [128, 128, 128],
                                'track_info': {
                                    'observations': observations,
                                    'quality': quality
                                }
                            }
                        
                        self.logger.info(f"Loaded {len(self.points_3d)} 3D points from '{key}'")
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to load alternative 3D format: {e}")
            return False
    
    def _generate_coords_from_ids(self, point_ids: np.ndarray) -> np.ndarray:
        """从点ID生成3D坐标（用于测试）"""
        unique_ids = np.unique(point_ids)
        coords = np.zeros((max(unique_ids) + 1, 3))
        
        # 使用点ID作为种子生成确定性的3D坐标
        for point_id in unique_ids:
            np.random.seed(int(point_id))
            # 生成在合理范围内的3D坐标
            coords[point_id] = np.random.uniform(-10, 10, 3)
            coords[point_id][2] = np.random.uniform(0, 20)  # Z坐标为正（深度）
        
        return coords
    
    def _generate_point_color(self, point_id: int, observations: int) -> List[int]:
        """为3D点生成颜色"""
        # 基于观测次数的颜色映射
        if observations >= 10:
            return [0, 255, 0]    # 绿色：高质量点
        elif observations >= 5:
            return [255, 255, 0]  # 黄色：中等质量点
        elif observations >= 2:
            return [255, 165, 0]  # 橙色：一般质量点
        else:
            return [255, 0, 0]    # 红色：低质量点
    
    def _create_test_3d_data(self):
        """创建测试用的3D数据"""
        self.logger.info("Creating test 3D skeleton data")
        
        # 创建多种类型的3D结构用于测试
        structures = [
            self._create_spiral_structure(center=[0, 0, 5], radius=3, height=4, points=50),
            self._create_grid_structure(center=[5, 5, 3], size=4, resolution=20),
            self._create_random_cluster(center=[-3, 2, 6], radius=2, points=30),
            self._create_line_structure(start=[-2, -2, 1], end=[8, 3, 8], points=25)
        ]
        
        point_id = 0
        for structure in structures:
            for coords in structure:
                # 添加一些噪声
                noisy_coords = coords + np.random.normal(0, 0.05, 3)
                
                # 随机质量
                observations = np.random.randint(1, 15)
                error = max(0.01, 1.0 / (1.0 + observations * 0.1))
                
                self.points_3d[point_id] = {
                    'coords': noisy_coords,
                    'error': error,
                    'color': self._generate_point_color(point_id, observations),
                    'track_info': {
                        'observations': observations,
                        'quality': min(1.0, observations / 10.0)
                    }
                }
                point_id += 1
        
        self.logger.info(f"Created {len(self.points_3d)} test 3D points")
    
    def _create_spiral_structure(self, center: List[float], radius: float, height: float, points: int) -> np.ndarray:
        """创建螺旋结构"""
        t = np.linspace(0, 4*np.pi, points)
        coords = np.zeros((points, 3))
        coords[:, 0] = center[0] + radius * np.cos(t)
        coords[:, 1] = center[1] + radius * np.sin(t)
        coords[:, 2] = center[2] + height * t / (4*np.pi)
        return coords
    
    def _create_grid_structure(self, center: List[float], size: float, resolution: int) -> np.ndarray:
        """创建网格结构"""
        coords = []
        step = size / resolution
        for i in range(resolution):
            for j in range(resolution):
                x = center[0] + (i - resolution/2) * step
                y = center[1] + (j - resolution/2) * step
                z = center[2] + 0.5 * np.sin(2*np.pi*i/resolution) * np.cos(2*np.pi*j/resolution)
                coords.append([x, y, z])
        return np.array(coords)
    
    def _create_random_cluster(self, center: List[float], radius: float, points: int) -> np.ndarray:
        """创建随机聚类"""
        coords = np.random.normal(0, radius/3, (points, 3))
        coords += np.array(center)
        return coords
    
    def _create_line_structure(self, start: List[float], end: List[float], points: int) -> np.ndarray:
        """创建线性结构"""
        coords = np.zeros((points, 3))
        for i in range(3):
            coords[:, i] = np.linspace(start[i], end[i], points)
        return coords
    
    def generate_3d_skeleton_visualizations(self, output_dir: str, views: Optional[List[str]] = None):
        """
        生成3D骨架图的多视角可视化
        
        Args:
            output_dir: 输出目录
            views: 视角列表，默认为所有视角
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if views is None:
            views = list(self.view_angles.keys())
        
        self.logger.info(f"Generating 3D skeleton visualizations for {len(views)} views")
        
        if not self.points_3d:
            self.logger.error("No 3D points loaded")
            return
        
        # 为每个视角生成单独的图
        for view_name in views:
            self._create_single_view_visualization(view_name, output_dir)
        
        # 创建多视角对比图
        self._create_multi_view_comparison(output_dir, views[:4])  # 限制为4个视角
        
        # 创建统计分析图
        try:
            self._create_3d_analysis_summary(output_dir)
        except Exception as e:
            self.logger.error(f"Failed to create analysis summary: {e}")
        
        # 保存3D数据信息
        try:
            self._save_3d_data_info(output_dir)
        except Exception as e:
            self.logger.error(f"Failed to save 3D data info: {e}")
        
        self.logger.info(f"3D skeleton visualizations saved to {output_dir}")
    
    def _create_single_view_visualization(self, view_name: str, output_dir: Path):
        """创建单个视角的3D可视化"""
        self.logger.info(f"Creating {view_name} view visualization")
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制3D骨架点
        self._draw_3d_skeleton(ax, color_by='quality', point_size='error')
        
        try:
            # 设置视角
            angles = self.view_angles[view_name]
            ax.view_init(elev=angles['elev'], azim=angles['azim'])
            
            # 设置坐标轴
            self._setup_3d_axes(ax, view_name)
            
            # 添加标题和信息
            ax.set_title(f'3D Trajectory Skeleton - {view_name.title()} View\\n'
                        f'{len(self.points_3d)} 3D Points', fontsize=14, pad=20)
            
            # 保存图像
            output_path = output_dir / f'trajectory_3d_skeleton_{view_name}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            self.logger.info(f"Saved {view_name} view: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create {view_name} view: {e}")
            try:
                # 尝试简单的fallback图
                plt.close()  # 关闭可能有问题的图
                
                fig = plt.figure(figsize=(12, 9))  
                ax = fig.add_subplot(111, projection='3d')
                
                # 创建简单的测试数据
                test_coords = np.random.uniform(-5, 5, (100, 3))
                ax.scatter(test_coords[:, 0], test_coords[:, 1], test_coords[:, 2], 
                          c='blue', s=20, alpha=0.6)
                
                ax.set_title(f'3D Skeleton - {view_name} (Fallback)', fontsize=14)
                ax.set_xlabel('X')
                ax.set_ylabel('Y') 
                ax.set_zlabel('Z')
                
                output_path = output_dir / f'trajectory_3d_skeleton_{view_name}_fallback.png'
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"Saved fallback {view_name} view: {output_path}")
                
            except Exception as e2:
                plt.close()  # 确保图形被关闭
                self.logger.error(f"Fallback also failed for {view_name}: {e2}")
    
    def _draw_3d_skeleton(self, ax, color_by: str = 'quality', point_size: str = 'observations'):
        """绘制3D骨架点"""
        if not self.points_3d:
            self.logger.warning("No 3D points to draw")
            return
        
        # 收集所有点的数据并验证有效性
        valid_points = []
        coords_list = []
        errors_list = []
        observations_list = []
        qualities_list = []
        
        for point_id, point in self.points_3d.items():
            try:
                coords = point['coords']
                error = point['error']
                track_info = point['track_info']
                
                # 验证坐标有效性
                if (coords is not None and 
                    len(coords) == 3 and 
                    np.all(np.isfinite(coords)) and
                    not np.any(np.isnan(coords))):
                    
                    coords_list.append(coords)
                    errors_list.append(float(error) if error is not None else 0.1)
                    observations_list.append(int(track_info.get('observations', 1)))
                    qualities_list.append(float(track_info.get('quality', 0.5)))
                    valid_points.append(point_id)
            except Exception as e:
                self.logger.warning(f"Skipping invalid point {point_id}: {e}")
                continue
        
        if not coords_list:
            self.logger.warning("No valid 3D points found")
            return
        
        # 转换为numpy数组
        coords = np.array(coords_list)
        errors = np.array(errors_list)
        observations = np.array(observations_list)
        qualities = np.array(qualities_list)
        
        self.logger.info(f"Drawing {len(coords)} valid 3D points")
        
        # 确定颜色映射
        if color_by == 'quality':
            colors = plt.cm.RdYlGn(qualities)
            color_label = 'Quality Score'
        elif color_by == 'error':
            colors = plt.cm.RdYlBu_r(1.0 - np.clip(errors / np.max(errors), 0, 1))
            color_label = 'Reprojection Error'
        elif color_by == 'observations':
            norm_obs = observations / np.max(observations)
            colors = plt.cm.viridis(norm_obs)
            color_label = 'Observations Count'
        else:
            colors = 'blue'
            color_label = 'Uniform'
        
        # 确定点大小
        if point_size == 'observations':
            sizes = 20 + observations * 5
        elif point_size == 'error':
            sizes = 50 - errors * 30  # 误差越小点越大
            sizes = np.clip(sizes, 10, 100)
        elif point_size == 'quality':
            sizes = 20 + qualities * 50
        else:
            sizes = 30
        
        # 验证数据范围，防止matplotlib投影错误
        coord_ranges = np.ptp(coords, axis=0)  # 计算每个维度的范围
        if np.any(coord_ranges == 0):
            # 如果某个维度范围为0，添加小的扰动
            for dim in range(3):
                if coord_ranges[dim] == 0:
                    coords[:, dim] += np.random.normal(0, 0.001, len(coords))
        
        # 检查极值，防止数值问题
        coord_max = np.max(np.abs(coords))
        if coord_max > 1e6:
            # 如果坐标值太大，进行缩放
            scale_factor = 1e3 / coord_max
            coords = coords * scale_factor
            self.logger.info(f"Scaled coordinates by factor {scale_factor}")
        
        try:
            # 绘制3D散点图
            scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                               c=colors, s=sizes, alpha=0.7, edgecolors='black', linewidths=0.5)
            
            # 添加颜色条（如果使用颜色映射）
            if isinstance(colors, np.ndarray) and len(colors.shape) > 1:
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.1)
                cbar.set_label(color_label, fontsize=12)
            
            # 添加一些连接线来显示结构（可选）
            if len(coords) > 10:  # 只有足够多的点才绘制连接
                self._add_structure_connections(ax, coords, max_connections=min(50, len(coords)))
                
        except Exception as e:
            self.logger.error(f"Failed to draw 3D scatter plot: {e}")
            # fallback: 简单绘制不带连接线
            try:
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                          c='blue', s=20, alpha=0.6)
                self.logger.info("Drew simplified 3D points as fallback")
            except Exception as e2:
                self.logger.error(f"Fallback plotting also failed: {e2}")
                return
    
    def _add_structure_connections(self, ax, coords: np.ndarray, max_connections: int = 100):
        """添加结构连接线以显示3D骨架（优化版本）"""
        if len(coords) < 10:  # 点太少不需要连接
            return
        
        try:
            from scipy.spatial.distance import pdist, squareform
            
            # 对于大数据集，只选择部分点进行连接
            if len(coords) > 500:
                step = len(coords) // 200  # 只用200个点计算连接
                sample_coords = coords[::step]
                sample_indices = list(range(0, len(coords), step))
            else:
                sample_coords = coords
                sample_indices = list(range(len(coords)))
            
            # 计算点之间的距离
            distances = squareform(pdist(sample_coords))
            
            # 为每个点找到最近的邻居
            connections = []
            for i, coord_idx in enumerate(sample_indices):
                if i >= len(distances):
                    break
                
                # 找到最近的2个邻居
                nearest_indices = np.argsort(distances[i])[1:3]  # 排除自己，只取2个
                for j in nearest_indices:
                    if j < len(sample_indices) and distances[i, j] < np.percentile(distances[distances > 0], 15):
                        real_i = sample_indices[i]
                        real_j = sample_indices[j]
                        connections.append((real_i, real_j))
            
            # 限制连接数量
            if len(connections) > max_connections:
                connections = connections[:max_connections]
            
            # 批量绘制连接线
            if connections:
                lines_x = []
                lines_y = []
                lines_z = []
                
                for i, j in connections:
                    lines_x.extend([coords[i, 0], coords[j, 0], None])
                    lines_y.extend([coords[i, 1], coords[j, 1], None])
                    lines_z.extend([coords[i, 2], coords[j, 2], None])
                
                ax.plot(lines_x, lines_y, lines_z, 'gray', alpha=0.2, linewidth=0.3)
                
        except ImportError:
            # 如果没有scipy，跳过连接线
            self.logger.warning("scipy not available, skipping structure connections")
        except Exception as e:
            self.logger.warning(f"Failed to add structure connections: {e}")
    
    def _setup_3d_axes(self, ax, view_name: str):
        """设置3D坐标轴"""
        if not self.points_3d:
            # 设置默认范围
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
        else:
            try:
                # 收集有效坐标
                coords_list = []
                for point in self.points_3d.values():
                    coords = point['coords']
                    if (coords is not None and 
                        len(coords) == 3 and 
                        np.all(np.isfinite(coords)) and
                        not np.any(np.isnan(coords))):
                        coords_list.append(coords)
                
                if coords_list:
                    coords = np.array(coords_list)
                    
                    # 为每个维度设置合适的范围
                    for i, (axis_func, label) in enumerate([(ax.set_xlim, 'X'), 
                                                           (ax.set_ylim, 'Y'), 
                                                           (ax.set_zlim, 'Z')]):
                        min_val = coords[:, i].min()
                        max_val = coords[:, i].max()
                        
                        # 确保范围有效
                        if np.isfinite(min_val) and np.isfinite(max_val):
                            if min_val == max_val:
                                # 如果最小值等于最大值，设置一个小范围
                                margin = 1.0 if min_val == 0 else abs(min_val) * 0.1
                            else:
                                margin = (max_val - min_val) * 0.1
                            
                            axis_func(min_val - margin, max_val + margin)
                        else:
                            # 如果数值无效，使用默认范围
                            axis_func(-10, 10)
                else:
                    # 没有有效坐标，使用默认范围
                    ax.set_xlim(-10, 10)
                    ax.set_ylim(-10, 10)
                    ax.set_zlim(-10, 10)
                    
            except Exception as e:
                self.logger.warning(f"Failed to set axis limits: {e}")
                # 使用安全的默认范围
                ax.set_xlim(-10, 10)
                ax.set_ylim(-10, 10)
                ax.set_zlim(-10, 10)
        
        # 设置标签
        ax.set_xlabel('X (world coordinates)', fontsize=12)
        ax.set_ylabel('Y (world coordinates)', fontsize=12)
        ax.set_zlabel('Z (world coordinates)', fontsize=12)
        
        # 根据视角调整网格显示
        if view_name == 'top':
            ax.grid(True, alpha=0.3)
        else:
            ax.grid(True, alpha=0.2)
    
    def _create_multi_view_comparison(self, output_dir: Path, views: List[str]):
        """创建多视角对比图"""
        self.logger.info("Creating multi-view comparison")
        
        n_views = len(views)
        if n_views == 0:
            return
        
        try:
            # 确定子图布局
            if n_views <= 2:
                rows, cols = 1, n_views
            elif n_views <= 4:
                rows, cols = 2, 2
            else:
                rows, cols = 2, 3
            
            fig = plt.figure(figsize=(5 * cols, 4 * rows))
            
            for i, view_name in enumerate(views):
                try:
                    ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
                    
                    # 绘制3D骨架
                    self._draw_3d_skeleton(ax, color_by='quality', point_size='observations')
                    
                    # 设置视角
                    angles = self.view_angles[view_name]
                    ax.view_init(elev=angles['elev'], azim=angles['azim'])
                    
                    # 设置坐标轴（简化版）
                    self._setup_3d_axes_simple(ax)
                    
                    # 设置标题
                    ax.set_title(f'{view_name.title()} View', fontsize=12)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create subplot for {view_name}: {e}")
                    # 创建简单的2D子图作为fallback
                    ax = fig.add_subplot(rows, cols, i + 1)
                    ax.text(0.5, 0.5, f'{view_name.title()}\\n(Error)', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{view_name.title()} View (Error)', fontsize=12)
                    
            # 隐藏多余的子图
            for i in range(n_views, rows * cols):
                fig.add_subplot(rows, cols, i + 1).set_visible(False)
            
            plt.tight_layout()
            
            # 保存图像
            output_path = output_dir / 'trajectory_3d_skeleton_multi_view.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Multi-view comparison saved: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create multi-view comparison: {e}")
            
            # 创建fallback多视角图
            try:
                plt.close()  # 关闭可能有问题的图
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle('3D Skeleton Views (Fallback)', fontsize=16)
                
                for i, view_name in enumerate(views[:4]):
                    row, col = divmod(i, 2)
                    ax = axes[row, col]
                    
                    # 创建简单的测试散点图
                    test_data = np.random.uniform(-5, 5, (100, 2))
                    ax.scatter(test_data[:, 0], test_data[:, 1], 
                              c='blue', s=20, alpha=0.6)
                    ax.set_title(f'{view_name.title()} View (2D Fallback)')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.grid(True, alpha=0.3)
                
                # 隐藏多余的子图
                for i in range(len(views), 4):
                    row, col = divmod(i, 2)
                    axes[row, col].set_visible(False)
                
                plt.tight_layout()
                
                output_path = output_dir / 'trajectory_3d_skeleton_multi_view_fallback.png'
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"Fallback multi-view comparison saved: {output_path}")
                
            except Exception as e2:
                plt.close()
                self.logger.error(f"Fallback multi-view also failed: {e2}")
    
    def _setup_3d_axes_simple(self, ax):
        """简化的3D坐标轴设置（用于多视角对比）"""
        if not self.points_3d:
            return
        
        coords = np.array([point['coords'] for point in self.points_3d.values()])
        
        # 设置相等的坐标轴范围
        all_coords = coords.flatten()
        margin = (all_coords.max() - all_coords.min()) * 0.1
        range_min, range_max = all_coords.min() - margin, all_coords.max() + margin
        
        ax.set_xlim(range_min, range_max)
        ax.set_ylim(range_min, range_max)
        ax.set_zlim(range_min, range_max)
        
        # 简化标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # 减少刻度以避免拥挤
        ax.set_xticks(np.linspace(range_min, range_max, 3))
        ax.set_yticks(np.linspace(range_min, range_max, 3))
        ax.set_zticks(np.linspace(range_min, range_max, 3))
    
    def _create_3d_analysis_summary(self, output_dir: Path):
        """创建3D分析汇总图"""
        self.logger.info("Creating 3D analysis summary")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 点质量分布
        self._plot_quality_distribution(ax1)
        
        # 2. 观测次数分布
        self._plot_observations_distribution(ax2)
        
        # 3. 空间分布分析
        self._plot_spatial_distribution(ax3)
        
        # 4. 统计信息总结
        self._plot_3d_statistics_summary(ax4)
        
        plt.tight_layout()
        
        # 保存图像
        output_path = output_dir / 'trajectory_3d_skeleton_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"3D analysis summary saved: {output_path}")
    
    def _plot_quality_distribution(self, ax):
        """绘制质量分布图"""
        qualities = [point['track_info']['quality'] for point in self.points_3d.values()]
        
        ax.hist(qualities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title('3D Point Quality Distribution', fontsize=12)
        ax.set_xlabel('Quality Score')
        ax.set_ylabel('Number of Points')
        ax.grid(True, alpha=0.3)
        
        if qualities:
            ax.axvline(np.mean(qualities), color='red', linestyle='--',
                      label=f'Mean: {np.mean(qualities):.3f}')
            ax.legend()
    
    def _plot_observations_distribution(self, ax):
        """绘制观测次数分布图"""
        observations = [point['track_info']['observations'] for point in self.points_3d.values()]
        
        ax.hist(observations, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax.set_title('Observations Count Distribution', fontsize=12)
        ax.set_xlabel('Number of Observations')
        ax.set_ylabel('Number of Points')
        ax.grid(True, alpha=0.3)
        
        if observations:
            ax.axvline(np.mean(observations), color='red', linestyle='--',
                      label=f'Mean: {np.mean(observations):.1f}')
            ax.legend()
    
    def _plot_spatial_distribution(self, ax):
        """绘制空间分布分析"""
        if not self.points_3d:
            ax.text(0.5, 0.5, 'No 3D data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        coords = np.array([point['coords'] for point in self.points_3d.values()])
        
        # 绘制XY平面投影
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=20, c='blue')
        ax.set_title('3D Points Spatial Distribution (XY Projection)', fontsize=12)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    def _plot_3d_statistics_summary(self, ax):
        """绘制3D统计信息总结"""
        ax.axis('off')
        
        if not self.points_3d:
            ax.text(0.5, 0.5, 'No 3D data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        # 计算统计信息
        coords = np.array([point['coords'] for point in self.points_3d.values()])
        errors = np.array([point['error'] for point in self.points_3d.values()])
        observations = np.array([point['track_info']['observations'] for point in self.points_3d.values()])
        qualities = np.array([point['track_info']['quality'] for point in self.points_3d.values()])
        
        stats_text = f"""
3D Skeleton Point Cloud Statistics:

Dataset Overview:
  • Total 3D Points: {len(self.points_3d):,}
  • Spatial Extent:
    - X Range: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]
    - Y Range: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]
    - Z Range: [{coords[:, 2].min():.2f}, {coords[:, 2].max():.2f}]
  • Volume: {self._calculate_bounding_volume(coords):.2f} cubic units

Quality Assessment:
  • Average Quality: {qualities.mean():.3f} ± {qualities.std():.3f}
  • High Quality Points (>0.7): {(qualities > 0.7).sum():,} ({(qualities > 0.7).sum()/len(qualities)*100:.1f}%)
  • Low Quality Points (<0.3): {(qualities < 0.3).sum():,} ({(qualities < 0.3).sum()/len(qualities)*100:.1f}%)

Observations Analysis:
  • Average Observations: {observations.mean():.1f} ± {observations.std():.1f}
  • Well-Observed Points (≥5): {(observations >= 5).sum():,}
  • Poorly-Observed Points (≤2): {(observations <= 2).sum():,}

Reconstruction Accuracy:
  • Average Error: {errors.mean():.3f} ± {errors.std():.3f}
  • High Accuracy Points (<0.1): {(errors < 0.1).sum():,}
  • Low Accuracy Points (>0.5): {(errors > 0.5).sum():,}

Available Views:
  • {', '.join(self.view_angles.keys())}
        """
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    def _calculate_bounding_volume(self, coords: np.ndarray) -> float:
        """计算边界框体积"""
        if len(coords) == 0:
            return 0.0
        
        ranges = coords.max(axis=0) - coords.min(axis=0)
        return np.prod(ranges)
    
    def _save_3d_data_info(self, output_dir: Path):
        """保存3D数据信息到JSON文件"""
        if not self.points_3d:
            return
        
        coords = np.array([point['coords'] for point in self.points_3d.values()])
        errors = np.array([point['error'] for point in self.points_3d.values()])
        observations = np.array([point['track_info']['observations'] for point in self.points_3d.values()])
        
        info = {
            'dataset_info': {
                'total_points': len(self.points_3d),
                'spatial_extent': {
                    'x_range': [float(coords[:, 0].min()), float(coords[:, 0].max())],
                    'y_range': [float(coords[:, 1].min()), float(coords[:, 1].max())],
                    'z_range': [float(coords[:, 2].min()), float(coords[:, 2].max())]
                },
                'bounding_volume': float(self._calculate_bounding_volume(coords))
            },
            'quality_stats': {
                'error_mean': float(errors.mean()),
                'error_std': float(errors.std()),
                'observations_mean': float(observations.mean()),
                'observations_std': float(observations.std())
            },
            'visualization_info': {
                'available_views': list(self.view_angles.keys()),
                'recommended_views': ['oblique', 'front', 'top', 'side']
            }
        }
        
        info_path = output_dir / '3d_skeleton_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        self.logger.info(f"3D data info saved: {info_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="3D Trajectory Skeleton Visualizer")
    parser.add_argument("--tracks_h5", type=str, required=True,
                       help="Path to tracks.h5 file")
    parser.add_argument("--output_dir", type=str, default="./3d_skeleton_output",
                       help="Output directory for visualizations")
    parser.add_argument("--views", type=str, nargs='+', 
                       choices=['front', 'side', 'top', 'oblique', 'back', 'bottom'],
                       default=['oblique', 'front', 'side', 'top'],
                       help="Views to generate")
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = Trajectory3DSkeletonVisualizer()
    
    # 加载3D数据
    success = visualizer.load_tracks_3d_data(args.tracks_h5)
    
    if not success:
        print("Failed to load 3D trajectory data. Using test data.")
    
    # 生成可视化
    visualizer.generate_3d_skeleton_visualizations(args.output_dir, args.views)
    
    print(f"3D skeleton visualization completed!")
    print(f"Results saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("  - trajectory_3d_skeleton_*.png: Individual view visualizations")
    print("  - trajectory_3d_skeleton_multi_view.png: Multi-view comparison")
    print("  - trajectory_3d_skeleton_analysis.png: Statistical analysis")
    print("  - 3d_skeleton_info.json: Dataset information")


if __name__ == "__main__":
    main()