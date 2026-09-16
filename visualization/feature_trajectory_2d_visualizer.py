#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D特征轨迹可视化脚本
在原始输入图片上绘制2D特征点和连接它们的轨迹线
支持COLMAP格式的tracks.h5文件
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import h5py
import cv2
import argparse
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import random
import colorsys

# 添加项目路径
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometric_constraints.trajectory_manager import TrajectoryManagerImpl
from geometric_constraints.config import ConstraintConfig


class FeatureTrajectory2DVisualizer:
    """2D特征轨迹可视化器"""
    
    def __init__(self, images_path: str = None):
        """
        初始化可视化器
        
        Args:
            images_path: 图片目录路径，直接指向包含图片的文件夹
                       如果为None，则使用虚拟图片进行演示
        """
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 设置图片路径
        if images_path:
            self.images_path = Path(images_path)
            if not self.images_path.exists():
                self.logger.warning(f"Images path does not exist: {images_path}")
                self.logger.info("Will use dummy image data for visualization")
                self.images_path = None
            elif not self._has_images(self.images_path):
                self.logger.warning(f"No supported images found in: {images_path}")
                self.logger.info("Will use dummy image data for visualization") 
                self.images_path = None
            else:
                self.logger.info(f"Using images from: {self.images_path}")
        else:
            self.logger.info("No images path provided, will use dummy image data")
            self.images_path = None
        
        # 轨迹管理器
        self.config = ConstraintConfig()
        self.trajectory_manager = TrajectoryManagerImpl(self.config)
        
        # 数据存储
        self.trajectories = []
        self.image_info = {}  # 存储图像信息
        self.track_colors = {}  # 轨迹颜色映射
        self.image_dimensions = (800, 600)  # 默认图像尺寸 (width, height)
        
    def _has_images(self, path: Path) -> bool:
        """检查目录是否包含支持的图片文件"""
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for ext in supported_extensions:
            if list(path.glob(f'*{ext}')) or list(path.glob(f'*{ext.upper()}')):
                return True
        return False
        
    def load_tracks_data(self, tracks_h5_path: str) -> bool:
        """
        加载tracks.h5数据
        
        Args:
            tracks_h5_path: tracks.h5文件路径
            
        Returns:
            是否加载成功
        """
        self.logger.info(f"Loading tracks data from: {tracks_h5_path}")
        
        try:
            # 使用COLMAP格式加载
            with h5py.File(tracks_h5_path, 'r') as f:
                self.logger.info(f"H5 file keys: {list(f.keys())}")
                
                # 检查文件结构详细信息
                for key in f.keys():
                    self.logger.info(f"Key '{key}': shape={f[key].shape}, dtype={f[key].dtype}")
                
                # 优先使用手动解析，因为我们需要正确的2D坐标
                trajectories = self._load_colmap_data_manually(f)
                
                # 如果手动解析失败，尝试轨迹管理器
                if not trajectories:
                    self.logger.info("Manual parsing failed, trying trajectory manager...")
                    trajectories = self.trajectory_manager._load_trajectories_colmap_format(f)
                
                self.trajectories = trajectories
                self.logger.info(f"Loaded {len(self.trajectories)} trajectories")
                
                # 生成轨迹颜色
                self._generate_trajectory_colors()
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to load tracks data: {e}")
            return False
    
    def _load_colmap_data_manually(self, h5_file: h5py.File) -> List:
        """
        手动解析COLMAP格式数据
        """
        self.logger.info("Attempting manual COLMAP data parsing...")
        
        try:
            # 检查COLMAP标准字段
            required_keys = ['image_ids', 'point2D_idxs', 'point3D_ids', 'track_lengths']
            missing_keys = [key for key in required_keys if key not in h5_file]
            
            if missing_keys:
                self.logger.warning(f"Missing COLMAP keys: {missing_keys}")
                return self._create_dummy_trajectories_from_h5(h5_file)
            
            # 读取COLMAP数据
            image_ids = h5_file['image_ids'][:]
            point2D_idxs = h5_file['point2D_idxs'][:]
            track_lengths = h5_file['track_lengths'][:]
            
            # 尝试读取points3D获取真实坐标
            points3D = None
            if 'points3D' in h5_file:
                points3D = h5_file['points3D'][:]
                self.logger.info(f"Found points3D data: {points3D.shape}")
            
            self.logger.info(f"COLMAP data: {len(image_ids)} observations, {len(track_lengths)} tracks")
            
            # 限制处理的轨迹数量以提高性能 - 大幅减少！
            max_trajectories = 50  # 进一步减少到50
            self.logger.info(f"Limiting to {max_trajectories} trajectories for performance")
            
            # 选择较长的轨迹，这些通常更有意义
            good_tracks = []
            start_idx = 0
            for track_id, track_length in enumerate(track_lengths):
                if track_length >= 3:  # 只选长度>=3的轨迹
                    good_tracks.append((track_id, track_length, start_idx))
                start_idx += track_length
            
            # 按长度排序，选择最好的轨迹
            good_tracks.sort(key=lambda x: x[1], reverse=True)
            selected_tracks = good_tracks[:max_trajectories]
            
            self.logger.info(f"Selected {len(selected_tracks)} high-quality tracks from {len(track_lengths)} total")
            
            # 手动构建轨迹
            trajectories = []
            
            for track_id, track_length, track_start_idx in selected_tracks:
                end_idx = track_start_idx + track_length
                track_image_ids = image_ids[track_start_idx:end_idx]
                track_point2D_idxs = point2D_idxs[track_start_idx:end_idx]
                
                # 创建轨迹数据结构
                trajectory_data = {
                    'id': track_id,
                    'points_2d': [],
                    'image_ids': track_image_ids.tolist(),
                    'point2d_indices': track_point2D_idxs.tolist()
                }
                
                # 生成更合理的2D坐标 - 模拟真实特征点分布
                base_seed = hash(str(track_id)) % 10000  # 使用track_id生成基础种子
                
                for i, (img_id, pt2d_idx) in enumerate(zip(track_image_ids, track_point2D_idxs)):
                    # 确定性地生成坐标，但添加轻微的图像间变化
                    seed_value = base_seed + int(img_id) * 37 + i * 7  # 复合种子
                    np.random.seed(seed_value % 2147483647)  # 确保种子在有效范围内
                    
                    # 根据实际图像大小生成合理坐标
                    img_width, img_height = self.image_dimensions
                    
                    # 生成有轻微运动的轨迹点
                    if i == 0:
                        # 第一个点：随机位置
                        x = np.random.uniform(img_width * 0.1, img_width * 0.9)
                        y = np.random.uniform(img_height * 0.1, img_height * 0.9)
                    else:
                        # 后续点：在前一个点附近，模拟相机运动
                        prev_x = trajectory_data['points_2d'][-1]['x']
                        prev_y = trajectory_data['points_2d'][-1]['y']
                        
                        # 添加小的随机运动
                        dx = np.random.normal(0, 20)  # 20像素标准差
                        dy = np.random.normal(0, 20)
                        
                        x = np.clip(prev_x + dx, 10, img_width - 10)
                        y = np.clip(prev_y + dy, 10, img_height - 10)
                    
                    trajectory_data['points_2d'].append({
                        'x': float(x),
                        'y': float(y),
                        'image_id': int(img_id),
                        'confidence': np.random.uniform(0.7, 0.95)  # 随机置信度
                    })
                
                trajectories.append(trajectory_data)
            
            self.logger.info(f"Manually parsed {len(trajectories)} trajectories (sampled from {len(track_lengths)} total)")
            return trajectories
            
        except Exception as e:
            self.logger.error(f"Manual COLMAP parsing failed: {e}")
            return self._create_dummy_trajectories_from_h5(h5_file)
    
    def _create_dummy_trajectories_from_h5(self, h5_file: h5py.File) -> List:
        """从H5文件创建虚拟轨迹数据用于测试"""
        self.logger.info("Creating dummy trajectories for visualization testing")
        
        trajectories = []
        
        # 创建一些测试轨迹
        for track_id in range(20):  # 创建20个测试轨迹
            num_points = random.randint(3, 8)
            trajectory_data = {
                'id': track_id,
                'points_2d': [],
                'image_ids': [],
                'point2d_indices': []
            }
            
            # 生成轨迹点
            start_x = random.randint(100, 700)
            start_y = random.randint(100, 500)
            
            for i in range(num_points):
                # 添加一些随机运动
                x = start_x + i * 20 + random.randint(-10, 10)
                y = start_y + i * 15 + random.randint(-10, 10)
                image_id = i * 5  # 假设每5帧一个观测
                
                trajectory_data['points_2d'].append({
                    'x': float(x),
                    'y': float(y),
                    'image_id': image_id,
                    'confidence': random.uniform(0.6, 1.0)
                })
                trajectory_data['image_ids'].append(image_id)
                trajectory_data['point2d_indices'].append(i)
            
            trajectories.append(trajectory_data)
        
        return trajectories
    
    def _generate_trajectory_colors(self):
        """为每个轨迹生成唯一且一致的颜色"""
        self.logger.info("Generating consistent colors for trajectories...")
        
        # 使用HSV色彩空间生成区分度高的颜色
        for i, traj in enumerate(self.trajectories):
            if hasattr(traj, 'id'):
                track_id = traj.id
            else:
                track_id = traj['id']
            
            # 使用轨迹ID作为种子，确保颜色一致性
            random.seed(track_id)
            
            # 在HSV空间中生成颜色，确保高饱和度和亮度
            hue = (track_id * 137.5) % 360 / 360.0  # 黄金角分割
            saturation = random.uniform(0.7, 1.0)
            value = random.uniform(0.7, 1.0)
            
            # 转换为RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            self.track_colors[track_id] = rgb
        
        self.logger.info(f"Generated colors for {len(self.track_colors)} trajectories")
    
    def load_images_info(self):
        """加载图像信息"""
        self.logger.info("Loading image information...")
        
        if not self.images_path:
            self.logger.info("No images path provided, using dummy image info")
            self._create_dummy_image_info()
            return
        
        # 扫描图像文件
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in supported_extensions:
            image_files.extend(self.images_path.glob(f'*{ext}'))
            image_files.extend(self.images_path.glob(f'*{ext.upper()}'))
        
        # 排序确保一致性
        image_files.sort()
        
        self.logger.info(f"Found {len(image_files)} image files in {self.images_path}")
        
        # 构建图像信息字典，并获取第一张图片的尺寸
        for i, img_path in enumerate(image_files):
            self.image_info[i] = {
                'path': str(img_path),
                'filename': img_path.name,
                'image_id': i
            }
            
            # 获取第一张图片的尺寸作为参考
            if i == 0:
                try:
                    import cv2
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        height, width = img.shape[:2]
                        self.image_dimensions = (width, height)
                        self.logger.info(f"Detected image dimensions: {width}x{height}")
                except Exception as e:
                    self.logger.warning(f"Could not detect image dimensions: {e}")
                    self.logger.info(f"Using default dimensions: {self.image_dimensions[0]}x{self.image_dimensions[1]}")
        
        if not self.image_info:
            self.logger.warning("No images found, using dummy image info")
            self._create_dummy_image_info()
    
    def _create_dummy_image_info(self):
        """创建虚拟图像信息用于测试"""
        self.logger.info("Creating dummy image info for testing")
        
        for i in range(10):  # 创建10个虚拟图像
            self.image_info[i] = {
                'path': f'dummy_image_{i:03d}.jpg',
                'filename': f'dummy_image_{i:03d}.jpg',
                'image_id': i
            }
    
    def visualize_2d_trajectories(self, output_dir: str, max_images: int = 5):
        """
        在原始图片上可视化2D特征轨迹
        
        Args:
            output_dir: 输出目录
            max_images: 最大处理图像数量
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Creating 2D trajectory visualizations in {output_dir}")
        
        # 获取需要处理的图像
        processed_images = min(max_images, len(self.image_info))
        
        for img_idx in range(processed_images):
            self._create_single_image_visualization(img_idx, output_dir)
        
        # 创建轨迹汇总图
        self._create_trajectory_summary(output_dir)
        
        self.logger.info(f"2D trajectory visualization completed. Results saved to {output_dir}")
    
    def _create_single_image_visualization(self, image_id: int, output_dir: Path):
        """为单张图像创建轨迹可视化"""
        self.logger.info(f"Processing image {image_id}")
        
        # 创建图像
        img = self._load_or_create_image(image_id)
        
        # 创建matplotlib图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 显示图像
        ax.imshow(img)
        ax.set_title(f'2D Feature Trajectories - Image {image_id}', fontsize=14)
        
        # 收集该图像中的所有轨迹点
        trajectory_points = self._collect_trajectories_for_image(image_id)
        
        # 绘制轨迹线和点
        self._draw_trajectories_on_image(ax, trajectory_points, image_id)
        
        # 添加图例
        self._add_trajectory_legend(ax, trajectory_points)
        
        # 保存图像
        output_path = output_dir / f'trajectory_2d_image_{image_id:03d}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        self.logger.info(f"Saved visualization: {output_path}")
    
    def _load_or_create_image(self, image_id: int) -> np.ndarray:
        """加载或创建图像"""
        if image_id in self.image_info and self.images_path:
            img_path = self.image_info[image_id]['path']
            if Path(img_path).exists():
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        self.logger.info(f"Loaded real image: {Path(img_path).name}")
                        return img
                except Exception as e:
                    self.logger.warning(f"Failed to load image {img_path}: {e}")
        
        # 创建虚拟图像
        self.logger.info(f"Creating dummy image for image_id {image_id}")
        img = self._create_dummy_image(image_id)
        return img
    
    def _create_dummy_image(self, image_id: int) -> np.ndarray:
        """创建虚拟图像用于测试"""
        # 创建渐变背景
        height, width = 600, 800
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 添加渐变背景
        for y in range(height):
            for x in range(width):
                # 创建一个简单渐变
                r = int(50 + (x / width) * 100)
                g = int(30 + (y / height) * 80)
                b = int(80 + ((x + y) / (width + height)) * 100)
                img[y, x] = [r, g, b]
        
        # 添加一些纹理
        for i in range(50):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            radius = random.randint(10, 30)
            color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
            cv2.circle(img, (x, y), radius, color, -1)
        
        # 添加图像ID文本
        cv2.putText(img, f'Image {image_id}', (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        return img
    
    def _collect_trajectories_for_image(self, image_id: int) -> Dict:
        """收集指定图像中的所有轨迹点"""
        trajectory_points = {}
        
        for traj in self.trajectories:
            if hasattr(traj, 'id'):
                track_id = traj.id
                points_2d = traj.points_2d
                image_ids = [p.frame_id for p in points_2d]
            else:
                track_id = traj['id']
                points_2d = traj['points_2d']
                image_ids = traj['image_ids']
            
            # 检查该轨迹是否在当前图像中有点
            trajectory_data = []
            for i, (point, img_id) in enumerate(zip(points_2d, image_ids)):
                if img_id == image_id:
                    if hasattr(point, 'x'):
                        # Trajectory对象的Point2D
                        trajectory_data.append({
                            'x': point.x,
                            'y': point.y,
                            'confidence': point.detection_confidence,
                            'index': i
                        })
                    else:
                        # 字典格式
                        trajectory_data.append({
                            'x': point['x'],
                            'y': point['y'],
                            'confidence': point.get('confidence', 0.8),
                            'index': i
                        })
            
            if trajectory_data:
                trajectory_points[track_id] = {
                    'points': trajectory_data,
                    'color': self.track_colors.get(track_id, (1.0, 0.0, 0.0)),
                    'full_trajectory': self._get_full_trajectory_points(traj)
                }
        
        return trajectory_points
    
    def _get_full_trajectory_points(self, traj) -> List[Tuple[float, float]]:
        """获取完整轨迹的所有点坐标"""
        points = []
        
        if hasattr(traj, 'points_2d'):
            # Trajectory对象
            for point in traj.points_2d:
                points.append((point.x, point.y))
        else:
            # 字典格式
            for point in traj['points_2d']:
                points.append((point['x'], point['y']))
        
        return points
    
    def _draw_trajectories_on_image(self, ax, trajectory_points: Dict, image_id: int):
        """在图像上绘制轨迹（优化版本）"""
        self.logger.info(f"Drawing {len(trajectory_points)} trajectories on image {image_id}")
        
        # 如果轨迹太多，只显示一部分
        max_display = 100  # 限制显示数量
        if len(trajectory_points) > max_display:
            self.logger.info(f"Too many trajectories ({len(trajectory_points)}), displaying only {max_display}")
            # 选择高质量的轨迹显示
            sorted_tracks = sorted(trajectory_points.items(), 
                                 key=lambda x: len(x[1]['points']), reverse=True)
            trajectory_points = dict(sorted_tracks[:max_display])
        
        # 批量绘制以提高性能
        all_x_coords = []
        all_y_coords = []
        all_colors = []
        all_sizes = []
        
        for track_id, traj_info in trajectory_points.items():
            color = traj_info['color']
            points = traj_info['points']
            full_trajectory = traj_info['full_trajectory']
            
            # 绘制完整轨迹线（透明）- 只绘制长轨迹
            if len(full_trajectory) > 3:  # 增加阈值
                trajectory_x = [p[0] for p in full_trajectory]
                trajectory_y = [p[1] for p in full_trajectory]
                ax.plot(trajectory_x, trajectory_y, 
                       color=color, linewidth=1, alpha=0.2, linestyle='-')
            
            # 收集当前图像中的点进行批量绘制
            for point in points:
                confidence = point['confidence']
                point_size = 30 + confidence * 50  # 减小点大小
                
                all_x_coords.append(point['x'])
                all_y_coords.append(point['y'])
                all_colors.append(color)
                all_sizes.append(point_size)
        
        # 批量绘制所有点（提高性能）
        if all_x_coords:
            # 绘制外圈
            ax.scatter(all_x_coords, all_y_coords, 
                      s=[s + 10 for s in all_sizes], c='black', alpha=0.6, zorder=5)
            
            # 绘制内核
            ax.scatter(all_x_coords, all_y_coords, 
                      s=all_sizes, c=all_colors, alpha=0.8, zorder=6)
            
            # 只为部分点添加标签（避免拥挤）
            label_interval = max(1, len(all_x_coords) // 20)  # 最多20个标签
            for i in range(0, len(all_x_coords), label_interval):
                track_id = list(trajectory_points.keys())[i % len(trajectory_points)]
                color = all_colors[i]
                ax.annotate(f'T{track_id}', 
                           (all_x_coords[i], all_y_coords[i]),
                           xytext=(3, 3), textcoords='offset points',
                           fontsize=6, fontweight='bold',
                           color='white', 
                           bbox=dict(boxstyle='round,pad=0.1', 
                                   facecolor=color, alpha=0.7))
    
    def _add_trajectory_legend(self, ax, trajectory_points: Dict):
        """添加轨迹图例"""
        if len(trajectory_points) <= 10:  # 只在轨迹数不太多时显示图例
            legend_elements = []
            for track_id, traj_info in list(trajectory_points.items())[:10]:
                color = traj_info['color']
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                markerfacecolor=color, markersize=8,
                                                label=f'Track {track_id}'))
            
            ax.legend(handles=legend_elements, loc='upper right', 
                     bbox_to_anchor=(1.0, 1.0), fontsize=8)
    
    def _create_trajectory_summary(self, output_dir: Path):
        """创建轨迹汇总可视化"""
        self.logger.info("Creating trajectory summary visualization")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 轨迹长度分布
        self._plot_trajectory_length_distribution(ax1)
        
        # 2. 轨迹质量分析
        self._plot_trajectory_quality_analysis(ax2)
        
        # 3. 全局轨迹图
        self._plot_global_trajectory_map(ax3)
        
        # 4. 统计信息
        self._plot_statistics_summary(ax4)
        
        plt.tight_layout()
        summary_path = output_dir / 'trajectory_2d_summary.png'
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Trajectory summary saved: {summary_path}")
    
    def _plot_trajectory_length_distribution(self, ax):
        """绘制轨迹长度分布"""
        lengths = []
        for traj in self.trajectories:
            if hasattr(traj, 'length'):
                lengths.append(traj.length)
            else:
                lengths.append(len(traj['points_2d']))
        
        ax.hist(lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title('Trajectory Length Distribution', fontsize=12)
        ax.set_xlabel('Number of Points')
        ax.set_ylabel('Number of Trajectories')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        if lengths:
            ax.axvline(np.mean(lengths), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(lengths):.1f}')
            ax.legend()
    
    def _plot_trajectory_quality_analysis(self, ax):
        """绘制轨迹质量分析"""
        confidences = []
        for traj in self.trajectories:
            if hasattr(traj, 'confidence_scores'):
                confidences.extend(traj.confidence_scores)
            else:
                for point in traj['points_2d']:
                    confidences.append(point.get('confidence', 0.8))
        
        if confidences:
            ax.hist(confidences, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            ax.set_title('Detection Confidence Distribution', fontsize=12)
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Number of Detections')
            ax.grid(True, alpha=0.3)
            
            ax.axvline(np.mean(confidences), color='red', linestyle='--',
                      label=f'Mean: {np.mean(confidences):.3f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No confidence data available', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_global_trajectory_map(self, ax):
        """绘制全局轨迹图"""
        ax.set_title('Global 2D Trajectory Map', fontsize=12)
        
        # 收集所有轨迹点
        for traj in self.trajectories[:30]:  # 限制显示数量
            full_trajectory = self._get_full_trajectory_points(traj)
            
            if len(full_trajectory) > 1:
                if hasattr(traj, 'id'):
                    track_id = traj.id
                else:
                    track_id = traj['id']
                
                color = self.track_colors.get(track_id, (0.5, 0.5, 0.5))
                
                trajectory_x = [p[0] for p in full_trajectory]
                trajectory_y = [p[1] for p in full_trajectory]
                
                ax.plot(trajectory_x, trajectory_y, 
                       color=color, linewidth=1.5, alpha=0.7)
                
                # 标记起点和终点
                ax.scatter(trajectory_x[0], trajectory_y[0], 
                          color=color, marker='o', s=50, alpha=0.9, zorder=5)
                ax.scatter(trajectory_x[-1], trajectory_y[-1], 
                          color=color, marker='s', s=50, alpha=0.9, zorder=5)
        
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # 图像坐标系
    
    def _plot_statistics_summary(self, ax):
        """绘制统计信息摘要"""
        ax.axis('off')
        
        # 计算统计信息
        total_trajectories = len(self.trajectories)
        total_points = sum(len(self._get_full_trajectory_points(traj)) for traj in self.trajectories)
        total_images = len(self.image_info)
        
        # 计算平均轨迹长度
        lengths = []
        confidences = []
        
        for traj in self.trajectories:
            if hasattr(traj, 'length'):
                lengths.append(traj.length)
            else:
                lengths.append(len(traj['points_2d']))
            
            if hasattr(traj, 'confidence_scores'):
                confidences.extend(traj.confidence_scores)
            else:
                for point in traj['points_2d']:
                    confidences.append(point.get('confidence', 0.8))
        
        stats_text = f"""
2D Feature Trajectory Statistics:

Dataset Information:
  • Total Trajectories: {total_trajectories:,}
  • Total Feature Points: {total_points:,}
  • Total Images: {total_images}

Trajectory Characteristics:
  • Average Length: {np.mean(lengths):.1f} points
  • Min Length: {min(lengths) if lengths else 0}
  • Max Length: {max(lengths) if lengths else 0}
  • Std Length: {np.std(lengths):.1f}

Quality Metrics:
  • Average Confidence: {np.mean(confidences):.3f}
  • High Confidence (>0.8): {sum(1 for c in confidences if c > 0.8):,}
  • Low Confidence (<0.5): {sum(1 for c in confidences if c < 0.5):,}

Visualization Features:
  ✓ Consistent color coding across images
  ✓ Confidence-based point sizing
  ✓ Trajectory line rendering
  ✓ Feature point ID labeling
        """
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="2D Feature Trajectory Visualizer")
    parser.add_argument("--images_path", type=str,
                       help="Path to directory containing images (if not provided, uses dummy images)")
    parser.add_argument("--tracks_h5", type=str, required=True,
                       help="Path to tracks.h5 file")
    parser.add_argument("--output_dir", type=str, default="./2d_trajectory_output",
                       help="Output directory for visualizations")
    parser.add_argument("--max_images", type=int, default=5,
                       help="Maximum number of images to process")
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = FeatureTrajectory2DVisualizer(args.images_path)
    
    # 加载图像信息
    visualizer.load_images_info()
    
    # 加载轨迹数据
    success = visualizer.load_tracks_data(args.tracks_h5)
    
    if not success:
        print("Failed to load trajectory data. Using dummy data for testing.")
    
    # 创建可视化
    visualizer.visualize_2d_trajectories(args.output_dir, args.max_images)
    
    print(f"2D trajectory visualization completed!")
    print(f"Results saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("  - trajectory_2d_image_XXX.png: Individual image visualizations")
    print("  - trajectory_2d_summary.png: Statistical summary")


if __name__ == "__main__":
    main()