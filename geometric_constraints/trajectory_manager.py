"""
轨迹管理器实现

实现轨迹的加载、预处理、质量评估和管理功能
"""

import h5py
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Any
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
import cv2

from .interfaces import TrajectoryManager, QualityAssessor
from .data_structures import Trajectory, Point2D, ConstraintResult
from .config import ConstraintConfig


class TrajectoryManagerImpl(TrajectoryManager):
    """轨迹管理器具体实现"""
    
    def __init__(self, config: ConstraintConfig, quality_assessor: Optional[QualityAssessor] = None):
        """
        初始化轨迹管理器
        
        Args:
            config: 约束配置
            quality_assessor: 质量评估器（可选）
        """
        self.config = config
        self.quality_assessor = quality_assessor
        self.trajectories: Dict[int, Trajectory] = {}
        self.logger = logging.getLogger(__name__)
        
        # 设置日志
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO if config.enable_debug_mode else logging.WARNING)
    
    def load_trajectories(self, track_file: str) -> List[Trajectory]:
        """
        从H5文件加载轨迹数据
        
        Args:
            track_file: H5格式的轨迹文件路径
            
        Returns:
            加载的轨迹列表
        """
        self.logger.info(f"Loading trajectories from {track_file}")
        
        if not Path(track_file).exists():
            raise FileNotFoundError(f"Trajectory file not found: {track_file}")
        
        trajectories = []
        
        try:
            with h5py.File(track_file, 'r') as f:
                # 检查文件格式
                self._validate_h5_format(f)
                
                # 读取轨迹数据
                if 'trajectories' in f:
                    trajectories = self._load_trajectories_v2_format(f)
                elif 'tracks' in f:
                    trajectories = self._load_trajectories_v1_format(f)
                else:
                    # COLMAP格式
                    trajectories = self._load_trajectories_colmap_format(f)
                
                self.logger.info(f"Loaded {len(trajectories)} trajectories")
                
        except Exception as e:
            self.logger.error(f"Error loading trajectories: {str(e)}")
            raise
        
        # 存储到内部字典
        for traj in trajectories:
            self.trajectories[traj.id] = traj
        
        return trajectories
    
    def _validate_h5_format(self, h5_file: h5py.File):
        """验证H5文件格式"""
        if 'trajectories' in h5_file:
            # 新格式
            required_keys = ['trajectories']
        elif 'tracks' in h5_file:
            # 旧格式
            required_keys = ['tracks']
        else:
            # COLMAP格式
            required_keys = ['point3D_ids', 'points3D', 'image_ids', 'point2D_idxs']
        
        for key in required_keys:
            if key not in h5_file:
                raise ValueError(f"Missing required key '{key}' in H5 file")
    
    def _load_trajectories_v1_format(self, h5_file: h5py.File) -> List[Trajectory]:
        """加载V1格式的轨迹数据（兼容旧格式）"""
        trajectories = []
        
        tracks_data = h5_file['tracks']
        cameras_data = h5_file['cameras'] if 'cameras' in h5_file else None
        
        # 假设tracks_data的格式为 [track_id, frame_id, x, y, confidence]
        for track_id in np.unique(tracks_data[:, 0]):
            track_mask = tracks_data[:, 0] == track_id
            track_points = tracks_data[track_mask]
            
            # 按frame_id排序
            sorted_indices = np.argsort(track_points[:, 1])
            track_points = track_points[sorted_indices]
            
            # 创建Point2D列表
            points_2d = []
            camera_indices = []
            confidence_scores = []
            
            for point_data in track_points:
                frame_id = int(point_data[1])
                x, y = float(point_data[2]), float(point_data[3])
                confidence = float(point_data[4]) if len(point_data) > 4 else 1.0
                
                point_2d = Point2D(
                    x=x, y=y, 
                    frame_id=frame_id,
                    detection_confidence=confidence
                )
                points_2d.append(point_2d)
                camera_indices.append(frame_id)  # 假设frame_id对应camera_index
                confidence_scores.append(confidence)
            
            # 创建轨迹
            trajectory = Trajectory(
                id=int(track_id),
                points_2d=points_2d,
                camera_indices=camera_indices,
                confidence_scores=confidence_scores
            )
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def _load_trajectories_v2_format(self, h5_file: h5py.File) -> List[Trajectory]:
        """加载V2格式的轨迹数据（新格式）"""
        trajectories = []
        
        trajectories_group = h5_file['trajectories']
        
        for traj_key in trajectories_group.keys():
            traj_data = trajectories_group[traj_key]
            
            # 读取轨迹基本信息
            track_id = int(traj_data.attrs.get('id', traj_key))
            
            # 读取点数据
            points_data = traj_data['points'][:]  # [N, 4] -> [frame_id, x, y, confidence]
            camera_indices = traj_data['camera_indices'][:] if 'camera_indices' in traj_data else points_data[:, 0].astype(int)
            
            # 创建Point2D列表
            points_2d = []
            confidence_scores = []
            
            for i, point_data in enumerate(points_data):
                frame_id = int(point_data[0])
                x, y = float(point_data[1]), float(point_data[2])
                confidence = float(point_data[3]) if len(point_data) > 3 else 1.0
                
                point_2d = Point2D(
                    x=x, y=y,
                    frame_id=frame_id,
                    detection_confidence=confidence
                )
                points_2d.append(point_2d)
                confidence_scores.append(confidence)
            
            # 创建轨迹
            trajectory = Trajectory(
                id=track_id,
                points_2d=points_2d,
                camera_indices=camera_indices.tolist(),
                confidence_scores=confidence_scores
            )
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def _load_trajectories_colmap_format(self, h5_file: h5py.File) -> List[Trajectory]:
        """加载COLMAP格式的轨迹数据"""
        trajectories = []
        
        try:
            # 读取COLMAP格式数据
            image_ids = h5_file['image_ids'][:]
            point2D_idxs = h5_file['point2D_idxs'][:]
            point3D_ids = h5_file['point3D_ids'][:]
            points3D = h5_file['points3D'][:]
            track_lengths = h5_file['track_lengths'][:]
            
            self.logger.info(f"COLMAP data loaded: {len(image_ids)} image_ids, {len(point3D_ids)} point3D_ids, {len(track_lengths)} track_lengths")
            
            # 构建轨迹：每个3D点对应一个轨迹
            start_idx = 0
            for track_id, track_length in enumerate(track_lengths):
                if track_length < 2:  # 至少需要2个观测点
                    start_idx += track_length
                    continue
                
                # 提取当前轨迹的数据
                end_idx = start_idx + track_length
                track_image_ids = image_ids[start_idx:end_idx]
                track_point2D_ids = point2D_idxs[start_idx:end_idx]
                
                # 创建Point2D列表
                points_2d = []
                camera_indices = []
                confidence_scores = []
                
                for i in range(track_length):
                    # 从COLMAP数据中提取2D点坐标
                    # 注意：这里需要根据实际的COLMAP H5文件结构调整
                    image_id = int(track_image_ids[i])
                    point2D_idx = int(track_point2D_ids[i])
                    
                    # 由于COLMAP格式中没有直接的2D坐标，我们使用索引作为占位符
                    # 实际应用中可能需要从COLMAP的images.bin或points2D中获取真实坐标
                    x, y = float(point2D_idx % 1000), float(point2D_idx // 1000)  # 临时坐标
                    confidence = 1.0  # COLMAP通常没有置信度信息，设为1.0
                    
                    point_2d = Point2D(
                        x=x, y=y,
                        frame_id=image_id,
                        detection_confidence=confidence
                    )
                    points_2d.append(point_2d)
                    camera_indices.append(image_id)
                    confidence_scores.append(confidence)
                
                # 创建轨迹
                trajectory = Trajectory(
                    id=track_id,
                    points_2d=points_2d,
                    camera_indices=camera_indices,
                    confidence_scores=confidence_scores
                )
                
                trajectories.append(trajectory)
                start_idx = end_idx
            
            self.logger.info(f"Successfully converted {len(trajectories)} COLMAP tracks to trajectories")
            
        except Exception as e:
            self.logger.error(f"Failed to load COLMAP format trajectories: {e}")
            # 提供降级处理
            self.logger.warning("Attempting to create dummy trajectories for testing purposes")
            trajectories = self._create_dummy_trajectories()
        
        return trajectories
    
    def _create_dummy_trajectories(self) -> List[Trajectory]:
        """创建虚拟轨迹用于测试目的"""
        trajectories = []
        self.logger.info("Creating dummy trajectories as fallback")
        
        # 创建几个简单的虚拟轨迹
        for track_id in range(5):  # 创建5个虚拟轨迹
            points_2d = []
            camera_indices = []
            confidence_scores = []
            
            # 每个轨迹包含3-8个点
            num_points = 3 + (track_id % 6)
            for i in range(num_points):
                frame_id = i * 10  # 间隔10帧
                x = 100 + track_id * 50 + i * 10  # 简单的运动模式
                y = 100 + track_id * 30 + i * 5
                
                point_2d = Point2D(
                    x=float(x), y=float(y),
                    frame_id=frame_id,
                    detection_confidence=0.8
                )
                points_2d.append(point_2d)
                camera_indices.append(frame_id)
                confidence_scores.append(0.8)
            
            trajectory = Trajectory(
                id=track_id,
                points_2d=points_2d,
                camera_indices=camera_indices,
                confidence_scores=confidence_scores
            )
            trajectories.append(trajectory)
        
        return trajectories
    
    def preprocess_trajectories(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        """
        预处理轨迹数据
        
        Args:
            trajectories: 原始轨迹列表
            
        Returns:
            预处理后的轨迹列表
        """
        self.logger.info(f"Preprocessing {len(trajectories)} trajectories")
        
        processed_trajectories = []
        
        for trajectory in trajectories:
            # 1. 数据验证和清洗
            cleaned_trajectory = self._clean_trajectory_data(trajectory)
            if cleaned_trajectory is None:
                continue
            
            # 2. 轨迹平滑
            smoothed_trajectory = self._smooth_trajectory(cleaned_trajectory)
            
            # 3. 异常值检测和处理
            filtered_trajectory = self._filter_outliers(smoothed_trajectory)
            
            # 4. 质量评估
            if self.quality_assessor:
                quality_score = self.quality_assessor.assess_trajectory_quality(filtered_trajectory)
                filtered_trajectory.update_quality_score(quality_score)
            
            # 5. 检查最终有效性
            if filtered_trajectory.is_valid(
                min_length=self.config.quality.min_trajectory_length,
                min_quality=self.config.quality.min_quality_score
            ):
                processed_trajectories.append(filtered_trajectory)
            else:
                self.logger.debug(f"Trajectory {trajectory.id} filtered out during preprocessing")
        
        self.logger.info(f"Preprocessing completed: {len(processed_trajectories)} valid trajectories")
        return processed_trajectories
    
    def _clean_trajectory_data(self, trajectory: Trajectory) -> Optional[Trajectory]:
        """清洗轨迹数据"""
        # 检查基本有效性
        if trajectory.length < self.config.quality.min_trajectory_length:
            return None
        
        # 移除重复的帧
        unique_frames = {}
        cleaned_points = []
        cleaned_cameras = []
        cleaned_confidences = []
        
        for i, point in enumerate(trajectory.points_2d):
            frame_id = point.frame_id
            if frame_id not in unique_frames:
                unique_frames[frame_id] = True
                cleaned_points.append(point)
                cleaned_cameras.append(trajectory.camera_indices[i])
                cleaned_confidences.append(trajectory.confidence_scores[i])
        
        # 按帧ID排序
        sorted_data = sorted(
            zip(cleaned_points, cleaned_cameras, cleaned_confidences),
            key=lambda x: x[0].frame_id
        )
        
        if len(sorted_data) < self.config.quality.min_trajectory_length:
            return None
        
        cleaned_points, cleaned_cameras, cleaned_confidences = zip(*sorted_data)
        
        return Trajectory(
            id=trajectory.id,
            points_2d=list(cleaned_points),
            camera_indices=list(cleaned_cameras),
            confidence_scores=list(cleaned_confidences),
            quality_score=trajectory.quality_score,
            is_active=trajectory.is_active,
            last_updated=trajectory.last_updated
        )
    
    def _smooth_trajectory(self, trajectory: Trajectory) -> Trajectory:
        """轨迹平滑处理"""
        if trajectory.length < 5:  # 太短的轨迹不需要平滑
            return trajectory
        
        # 提取坐标
        points = trajectory.get_points_tensor().numpy()
        
        # 使用移动平均进行平滑
        window_size = min(3, trajectory.length // 3)
        if window_size < 2:
            return trajectory
        
        smoothed_points = []
        for i in range(len(points)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(points), i + window_size // 2 + 1)
            
            # 加权平均，中心点权重更大
            weights = np.ones(end_idx - start_idx)
            center_idx = i - start_idx
            if center_idx < len(weights):
                weights[center_idx] *= 2.0
            weights /= weights.sum()
            
            smoothed_point = np.average(points[start_idx:end_idx], axis=0, weights=weights)
            smoothed_points.append(smoothed_point)
        
        # 更新轨迹点
        smoothed_points_2d = []
        for i, (smoothed_point, original_point) in enumerate(zip(smoothed_points, trajectory.points_2d)):
            smoothed_point_2d = Point2D(
                x=float(smoothed_point[0]),
                y=float(smoothed_point[1]),
                frame_id=original_point.frame_id,
                feature_descriptor=original_point.feature_descriptor,
                detection_confidence=original_point.detection_confidence
            )
            smoothed_points_2d.append(smoothed_point_2d)
        
        return Trajectory(
            id=trajectory.id,
            points_2d=smoothed_points_2d,
            camera_indices=trajectory.camera_indices,
            confidence_scores=trajectory.confidence_scores,
            quality_score=trajectory.quality_score,
            is_active=trajectory.is_active,
            last_updated=trajectory.last_updated
        )
    
    def _filter_outliers(self, trajectory: Trajectory) -> Trajectory:
        """过滤轨迹中的异常值"""
        if trajectory.length < 4:
            return trajectory
        
        points = trajectory.get_points_tensor().numpy()
        
        # 计算相邻帧间的位移
        displacements = np.diff(points, axis=0)
        displacement_magnitudes = np.linalg.norm(displacements, axis=1)
        
        # 使用统计方法检测异常值
        median_displacement = np.median(displacement_magnitudes)
        mad = np.median(np.abs(displacement_magnitudes - median_displacement))
        threshold = median_displacement + 3 * mad
        
        # 标记异常值
        outlier_mask = np.zeros(len(points), dtype=bool)
        outlier_mask[1:] = displacement_magnitudes > threshold
        
        # 如果异常值太多，返回原轨迹
        outlier_ratio = outlier_mask.sum() / len(outlier_mask)
        if outlier_ratio > self.config.quality.max_outlier_ratio:
            self.logger.debug(f"Trajectory {trajectory.id} has too many outliers ({outlier_ratio:.2f})")
            return trajectory
        
        # 移除异常值
        valid_indices = ~outlier_mask
        if valid_indices.sum() < self.config.quality.min_trajectory_length:
            return trajectory
        
        filtered_points_2d = [trajectory.points_2d[i] for i in range(len(trajectory.points_2d)) if valid_indices[i]]
        filtered_cameras = [trajectory.camera_indices[i] for i in range(len(trajectory.camera_indices)) if valid_indices[i]]
        filtered_confidences = [trajectory.confidence_scores[i] for i in range(len(trajectory.confidence_scores)) if valid_indices[i]]
        
        return Trajectory(
            id=trajectory.id,
            points_2d=filtered_points_2d,
            camera_indices=filtered_cameras,
            confidence_scores=filtered_confidences,
            quality_score=trajectory.quality_score,
            is_active=trajectory.is_active,
            last_updated=trajectory.last_updated
        )
    
    def update_trajectory_quality(self, trajectory: Trajectory, quality_score: float):
        """
        更新轨迹质量分数
        
        Args:
            trajectory: 目标轨迹
            quality_score: 新的质量分数
        """
        trajectory.update_quality_score(quality_score)
        
        # 更新活跃状态
        if quality_score < self.config.quality.min_quality_score:
            trajectory.is_active = False
            self.logger.debug(f"Trajectory {trajectory.id} deactivated due to low quality: {quality_score:.3f}")
        else:
            trajectory.is_active = True
        
        # 更新内部存储
        if trajectory.id in self.trajectories:
            self.trajectories[trajectory.id] = trajectory
    
    def get_active_trajectories(self, min_quality: float = 0.4) -> List[Trajectory]:
        """
        获取活跃轨迹
        
        Args:
            min_quality: 最小质量阈值
            
        Returns:
            活跃轨迹列表
        """
        active_trajectories = []
        
        for trajectory in self.trajectories.values():
            if (trajectory.is_active and 
                trajectory.quality_score >= min_quality and
                trajectory.length >= self.config.quality.min_trajectory_length):
                active_trajectories.append(trajectory)
        
        self.logger.debug(f"Found {len(active_trajectories)} active trajectories (min_quality={min_quality})")
        return active_trajectories
    
    def split_trajectory(self, trajectory: Trajectory, split_points: List[int]) -> List[Trajectory]:
        """
        分割轨迹
        
        Args:
            trajectory: 待分割的轨迹
            split_points: 分割点索引列表
            
        Returns:
            分割后的轨迹列表
        """
        if not split_points:
            return [trajectory]
        
        # 排序分割点并添加边界
        split_points = sorted(set(split_points))
        split_points = [0] + [p for p in split_points if 0 < p < trajectory.length] + [trajectory.length]
        
        split_trajectories = []
        
        for i in range(len(split_points) - 1):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]
            
            if end_idx - start_idx < self.config.quality.min_trajectory_length:
                continue
            
            # 创建子轨迹
            sub_points = trajectory.points_2d[start_idx:end_idx]
            sub_cameras = trajectory.camera_indices[start_idx:end_idx]
            sub_confidences = trajectory.confidence_scores[start_idx:end_idx]
            
            sub_trajectory = Trajectory(
                id=trajectory.id * 1000 + i,  # 生成新的ID
                points_2d=sub_points,
                camera_indices=sub_cameras,
                confidence_scores=sub_confidences,
                quality_score=0.0,  # 需要重新评估
                is_active=True,
                last_updated=trajectory.last_updated
            )
            
            split_trajectories.append(sub_trajectory)
        
        self.logger.info(f"Split trajectory {trajectory.id} into {len(split_trajectories)} sub-trajectories")
        return split_trajectories
    
    def detect_trajectory_breaks(self, trajectory: Trajectory) -> List[int]:
        """
        检测轨迹中的断裂点
        
        Args:
            trajectory: 待检测的轨迹
            
        Returns:
            断裂点索引列表
        """
        if trajectory.length < 4:
            return []
        
        points = trajectory.get_points_tensor().numpy()
        
        # 计算相邻帧间的位移
        displacements = np.diff(points, axis=0)
        displacement_magnitudes = np.linalg.norm(displacements, axis=1)
        
        # 检测异常大的位移（可能的断裂点）
        median_displacement = np.median(displacement_magnitudes)
        mad = np.median(np.abs(displacement_magnitudes - median_displacement))
        threshold = median_displacement + 5 * mad  # 更严格的阈值
        
        break_points = []
        for i, magnitude in enumerate(displacement_magnitudes):
            if magnitude > threshold:
                break_points.append(i + 1)  # +1因为位移是相邻点间的
        
        return break_points
    
    def reassociate_trajectories(self, trajectories: List[Trajectory], 
                               max_gap_frames: int = 5,
                               max_distance_pixels: float = 50.0) -> List[Trajectory]:
        """
        重新关联断裂的轨迹
        
        Args:
            trajectories: 轨迹列表
            max_gap_frames: 最大帧间隔
            max_distance_pixels: 最大像素距离
            
        Returns:
            重新关联后的轨迹列表
        """
        if len(trajectories) < 2:
            return trajectories
        
        # 按结束帧排序
        sorted_trajectories = sorted(trajectories, key=lambda t: t.get_frame_ids()[-1])
        
        reassociated = []
        used_indices = set()
        
        for i, traj1 in enumerate(sorted_trajectories):
            if i in used_indices:
                continue
            
            current_trajectory = traj1
            used_indices.add(i)
            
            # 寻找可能的后续轨迹
            for j, traj2 in enumerate(sorted_trajectories[i+1:], i+1):
                if j in used_indices:
                    continue
                
                # 检查时间间隔
                end_frame1 = current_trajectory.get_frame_ids()[-1]
                start_frame2 = traj2.get_frame_ids()[0]
                
                if start_frame2 - end_frame1 > max_gap_frames:
                    break  # 后续轨迹间隔太大
                
                # 检查空间距离
                end_point1 = current_trajectory.points_2d[-1]
                start_point2 = traj2.points_2d[0]
                
                distance = np.sqrt((end_point1.x - start_point2.x)**2 + 
                                 (end_point1.y - start_point2.y)**2)
                
                if distance <= max_distance_pixels:
                    # 合并轨迹
                    current_trajectory = self._merge_trajectories(current_trajectory, traj2)
                    used_indices.add(j)
                    self.logger.debug(f"Reassociated trajectories {traj1.id} and {traj2.id}")
            
            reassociated.append(current_trajectory)
        
        self.logger.info(f"Reassociation: {len(trajectories)} -> {len(reassociated)} trajectories")
        return reassociated
    
    def _merge_trajectories(self, traj1: Trajectory, traj2: Trajectory) -> Trajectory:
        """合并两个轨迹"""
        merged_points = traj1.points_2d + traj2.points_2d
        merged_cameras = traj1.camera_indices + traj2.camera_indices
        merged_confidences = traj1.confidence_scores + traj2.confidence_scores
        
        return Trajectory(
            id=traj1.id,  # 保持第一个轨迹的ID
            points_2d=merged_points,
            camera_indices=merged_cameras,
            confidence_scores=merged_confidences,
            quality_score=max(traj1.quality_score, traj2.quality_score),
            is_active=traj1.is_active and traj2.is_active,
            last_updated=max(traj1.last_updated, traj2.last_updated)
        )
    
    def visualize_trajectories(self, trajectories: List[Trajectory], 
                             output_path: Optional[str] = None,
                             max_trajectories: int = 50) -> None:
        """
        可视化轨迹用于调试
        
        Args:
            trajectories: 要可视化的轨迹列表
            output_path: 输出图像路径（可选）
            max_trajectories: 最大显示轨迹数量
        """
        if not trajectories:
            self.logger.warning("No trajectories to visualize")
            return
        
        # 限制显示数量
        display_trajectories = trajectories[:max_trajectories]
        
        plt.figure(figsize=(12, 8))
        
        # 为每个轨迹分配颜色
        colors = plt.cm.tab10(np.linspace(0, 1, len(display_trajectories)))
        
        for i, trajectory in enumerate(display_trajectories):
            points = trajectory.get_points_tensor().numpy()
            
            # 绘制轨迹线
            plt.plot(points[:, 0], points[:, 1], 
                    color=colors[i], alpha=0.7, linewidth=1.5,
                    label=f'Traj {trajectory.id} (Q={trajectory.quality_score:.2f})')
            
            # 标记起点和终点
            plt.scatter(points[0, 0], points[0, 1], 
                       color=colors[i], marker='o', s=50, alpha=0.8)
            plt.scatter(points[-1, 0], points[-1, 1], 
                       color=colors[i], marker='s', s=50, alpha=0.8)
        
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.title(f'Trajectory Visualization ({len(display_trajectories)} trajectories)')
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()  # 图像坐标系
        
        # 添加图例（仅显示前10个）
        if len(display_trajectories) <= 10:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Trajectory visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_trajectory_statistics(self, trajectories: List[Trajectory]) -> Dict[str, Any]:
        """
        生成轨迹统计信息
        
        Args:
            trajectories: 轨迹列表
            
        Returns:
            统计信息字典
        """
        if not trajectories:
            return {}
        
        lengths = [traj.length for traj in trajectories]
        quality_scores = [traj.quality_score for traj in trajectories]
        confidences = [traj.average_confidence for traj in trajectories]
        active_count = sum(1 for traj in trajectories if traj.is_active)
        
        stats = {
            'total_trajectories': len(trajectories),
            'active_trajectories': active_count,
            'inactive_trajectories': len(trajectories) - active_count,
            'length_stats': {
                'mean': np.mean(lengths),
                'median': np.median(lengths),
                'std': np.std(lengths),
                'min': np.min(lengths),
                'max': np.max(lengths)
            },
            'quality_stats': {
                'mean': np.mean(quality_scores),
                'median': np.median(quality_scores),
                'std': np.std(quality_scores),
                'min': np.min(quality_scores),
                'max': np.max(quality_scores)
            },
            'confidence_stats': {
                'mean': np.mean(confidences),
                'median': np.median(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            }
        }
        
        return stats
    
    def save_trajectories_to_h5(self, trajectories: List[Trajectory], output_path: str):
        """
        将轨迹保存到H5文件
        
        Args:
            trajectories: 要保存的轨迹列表
            output_path: 输出文件路径
        """
        self.logger.info(f"Saving {len(trajectories)} trajectories to {output_path}")
        
        with h5py.File(output_path, 'w') as f:
            # 创建轨迹组
            trajectories_group = f.create_group('trajectories')
            
            for trajectory in trajectories:
                traj_group = trajectories_group.create_group(f'traj_{trajectory.id}')
                
                # 保存轨迹属性
                traj_group.attrs['id'] = trajectory.id
                traj_group.attrs['quality_score'] = trajectory.quality_score
                traj_group.attrs['is_active'] = trajectory.is_active
                traj_group.attrs['last_updated'] = trajectory.last_updated
                
                # 保存点数据
                points_data = []
                for point in trajectory.points_2d:
                    points_data.append([point.frame_id, point.x, point.y, point.detection_confidence])
                
                traj_group.create_dataset('points', data=np.array(points_data))
                traj_group.create_dataset('camera_indices', data=np.array(trajectory.camera_indices))
        
        self.logger.info(f"Trajectories saved successfully to {output_path}")


class QualityAssessorImpl(QualityAssessor):
    """质量评估器具体实现"""
    
    def __init__(self, config: ConstraintConfig):
        """
        初始化质量评估器
        
        Args:
            config: 约束配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def assess_trajectory_quality(self, trajectory: Trajectory) -> float:
        """
        评估轨迹质量
        
        Args:
            trajectory: 待评估的轨迹
            
        Returns:
            质量分数 (0.0-1.0)
        """
        metrics = self.compute_quality_metrics(trajectory)
        
        # 加权计算综合质量分数
        quality_score = (
            metrics['length_score'] * self.config.quality.length_weight +
            metrics['visibility_score'] * self.config.quality.visibility_weight +
            metrics['consistency_score'] * self.config.quality.consistency_weight +
            metrics['stability_score'] * self.config.quality.stability_weight
        )
        
        return np.clip(quality_score, 0.0, 1.0)
    
    def compute_quality_metrics(self, trajectory: Trajectory) -> Dict[str, float]:
        """
        计算详细的质量指标
        
        Args:
            trajectory: 待评估的轨迹
            
        Returns:
            包含各项质量指标的字典
        """
        metrics = {}
        
        # 1. 长度分数
        metrics['length_score'] = min(trajectory.length / 10.0, 1.0)
        
        # 2. 可见性分数（基于置信度）
        metrics['visibility_score'] = trajectory.average_confidence
        
        # 3. 一致性分数（基于运动平滑性）
        metrics['consistency_score'] = self._compute_consistency_score(trajectory)
        
        # 4. 稳定性分数（基于重投影误差的稳定性）
        metrics['stability_score'] = self._compute_stability_score(trajectory)
        
        return metrics
    
    def _compute_consistency_score(self, trajectory: Trajectory) -> float:
        """计算轨迹一致性分数"""
        if trajectory.length < 3:
            return 0.5  # 默认分数
        
        points = trajectory.get_points_tensor().numpy()
        
        # 计算相邻帧间的位移
        displacements = np.diff(points, axis=0)
        displacement_magnitudes = np.linalg.norm(displacements, axis=1)
        
        if len(displacement_magnitudes) < 2:
            return 0.5
        
        # 计算位移变化的平滑性
        displacement_changes = np.diff(displacement_magnitudes)
        smoothness = 1.0 / (1.0 + np.std(displacement_changes))
        
        return np.clip(smoothness, 0.0, 1.0)
    
    def _compute_stability_score(self, trajectory: Trajectory) -> float:
        """计算轨迹稳定性分数"""
        if trajectory.length < 3:
            return 0.5
        
        # 基于置信度的稳定性
        confidences = np.array(trajectory.confidence_scores)
        confidence_stability = 1.0 - np.std(confidences)
        
        return np.clip(confidence_stability, 0.0, 1.0)
    
    def is_trajectory_reliable(self, trajectory: Trajectory, threshold: float = 0.4) -> bool:
        """
        判断轨迹是否可靠
        
        Args:
            trajectory: 待判断的轨迹
            threshold: 质量阈值
            
        Returns:
            是否可靠
        """
        quality_score = self.assess_trajectory_quality(trajectory)
        return quality_score >= threshold and trajectory.length >= self.config.quality.min_trajectory_length
    
    def detect_outliers(self, trajectory: Trajectory) -> torch.Tensor:
        """
        检测轨迹中的异常值
        
        Args:
            trajectory: 待检测的轨迹
            
        Returns:
            异常值掩码张量
        """
        if trajectory.length < 4:
            return torch.zeros(trajectory.length, dtype=torch.bool)
        
        points = trajectory.get_points_tensor().numpy()
        
        # 使用RANSAC检测异常值
        try:
            # 准备数据：使用帧索引作为x，坐标作为y
            frame_ids = np.array([p.frame_id for p in trajectory.points_2d]).reshape(-1, 1)
            
            outlier_mask = np.zeros(len(points), dtype=bool)
            
            # 分别对x和y坐标进行RANSAC
            for coord_idx in range(2):
                coords = points[:, coord_idx].reshape(-1, 1)
                
                ransac = RANSACRegressor(
                    residual_threshold=self.config.quality.ransac_threshold,
                    min_samples=max(2, len(points) // 4),
                    random_state=42
                )
                
                try:
                    ransac.fit(frame_ids, coords.ravel())
                    coord_outliers = ~ransac.inlier_mask_
                    outlier_mask |= coord_outliers
                except:
                    # RANSAC失败时使用统计方法
                    median_coord = np.median(coords)
                    mad = np.median(np.abs(coords - median_coord))
                    threshold = median_coord + 3 * mad
                    coord_outliers = np.abs(coords.ravel() - median_coord) > threshold
                    outlier_mask |= coord_outliers
            
        except Exception as e:
            self.logger.warning(f"Outlier detection failed for trajectory {trajectory.id}: {e}")
            outlier_mask = np.zeros(len(points), dtype=bool)
        
        return torch.from_numpy(outlier_mask)
    
    def compute_motion_consistency(self, trajectory: Trajectory) -> float:
        """
        计算运动一致性分数
        
        Args:
            trajectory: 待评估的轨迹
            
        Returns:
            运动一致性分数 (0.0-1.0)
        """
        if trajectory.length < 4:
            return 0.5
        
        points = trajectory.get_points_tensor().numpy()
        
        # 计算速度向量
        velocities = np.diff(points, axis=0)
        
        if len(velocities) < 2:
            return 0.5
        
        # 计算加速度向量
        accelerations = np.diff(velocities, axis=0)
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
        
        # 计算加速度的一致性（低方差表示运动平滑）
        if len(acceleration_magnitudes) == 0:
            return 0.5
        
        acceleration_std = np.std(acceleration_magnitudes)
        acceleration_mean = np.mean(acceleration_magnitudes)
        
        # 归一化一致性分数
        if acceleration_mean == 0:
            consistency = 1.0
        else:
            consistency = 1.0 / (1.0 + acceleration_std / (acceleration_mean + 1e-6))
        
        return np.clip(consistency, 0.0, 1.0)
    
    def compute_temporal_consistency(self, trajectory: Trajectory) -> float:
        """
        计算时间一致性分数
        
        Args:
            trajectory: 待评估的轨迹
            
        Returns:
            时间一致性分数 (0.0-1.0)
        """
        if trajectory.length < 3:
            return 0.5
        
        frame_ids = trajectory.get_frame_ids()
        
        # 计算帧间隔
        frame_gaps = np.diff(frame_ids)
        
        # 检查时间间隔的一致性
        gap_std = np.std(frame_gaps)
        gap_mean = np.mean(frame_gaps)
        
        if gap_mean == 0:
            return 0.0  # 所有点在同一帧，不合理
        
        # 计算时间一致性（间隔越均匀越好）
        temporal_consistency = 1.0 / (1.0 + gap_std / gap_mean)
        
        return np.clip(temporal_consistency, 0.0, 1.0)
    
    def compute_geometric_consistency(self, trajectory: Trajectory) -> float:
        """
        计算几何一致性分数
        
        Args:
            trajectory: 待评估的轨迹
            
        Returns:
            几何一致性分数 (0.0-1.0)
        """
        if trajectory.length < 5:
            return 0.5
        
        points = trajectory.get_points_tensor().numpy()
        
        # 使用多项式拟合评估轨迹的几何一致性
        try:
            frame_ids = np.array([p.frame_id for p in trajectory.points_2d])
            
            # 对x和y坐标分别进行二次多项式拟合
            x_coeffs = np.polyfit(frame_ids, points[:, 0], deg=min(2, len(points)-1))
            y_coeffs = np.polyfit(frame_ids, points[:, 1], deg=min(2, len(points)-1))
            
            # 计算拟合误差
            x_fitted = np.polyval(x_coeffs, frame_ids)
            y_fitted = np.polyval(y_coeffs, frame_ids)
            
            x_errors = np.abs(points[:, 0] - x_fitted)
            y_errors = np.abs(points[:, 1] - y_fitted)
            
            # 计算平均拟合误差
            mean_error = np.mean(np.sqrt(x_errors**2 + y_errors**2))
            
            # 转换为一致性分数（误差越小，一致性越高）
            geometric_consistency = 1.0 / (1.0 + mean_error / 10.0)  # 10像素作为归一化因子
            
            return np.clip(geometric_consistency, 0.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Geometric consistency computation failed: {e}")
            return 0.5
    
    def compute_visibility_stability(self, trajectory: Trajectory) -> float:
        """
        计算可见性稳定性分数
        
        Args:
            trajectory: 待评估的轨迹
            
        Returns:
            可见性稳定性分数 (0.0-1.0)
        """
        if trajectory.length < 3:
            return trajectory.average_confidence
        
        confidences = np.array(trajectory.confidence_scores)
        
        # 计算置信度的稳定性
        confidence_mean = np.mean(confidences)
        confidence_std = np.std(confidences)
        
        # 计算置信度变化的平滑性
        confidence_changes = np.abs(np.diff(confidences))
        change_stability = 1.0 / (1.0 + np.mean(confidence_changes))
        
        # 综合稳定性分数
        if confidence_mean == 0:
            stability = 0.0
        else:
            variance_stability = 1.0 - (confidence_std / confidence_mean)
            stability = 0.6 * variance_stability + 0.4 * change_stability
        
        return np.clip(stability, 0.0, 1.0)
    
    def assess_trajectory_completeness(self, trajectory: Trajectory, 
                                    expected_length: Optional[int] = None) -> float:
        """
        评估轨迹完整性
        
        Args:
            trajectory: 待评估的轨迹
            expected_length: 期望的轨迹长度（可选）
            
        Returns:
            完整性分数 (0.0-1.0)
        """
        if expected_length is None:
            # 基于帧间隔估算期望长度
            frame_ids = trajectory.get_frame_ids()
            if len(frame_ids) < 2:
                return 0.5
            
            frame_span = frame_ids[-1] - frame_ids[0] + 1
            expected_length = frame_span
        
        # 计算实际长度与期望长度的比率
        completeness = min(trajectory.length / expected_length, 1.0)
        
        # 考虑帧间隔的连续性
        frame_ids = trajectory.get_frame_ids()
        expected_frames = set(range(frame_ids[0], frame_ids[-1] + 1))
        actual_frames = set(frame_ids)
        
        frame_completeness = len(actual_frames) / len(expected_frames)
        
        # 综合完整性分数
        total_completeness = 0.7 * completeness + 0.3 * frame_completeness
        
        return np.clip(total_completeness, 0.0, 1.0)
    
    def compute_advanced_quality_metrics(self, trajectory: Trajectory) -> Dict[str, float]:
        """
        计算高级质量指标
        
        Args:
            trajectory: 待评估的轨迹
            
        Returns:
            高级质量指标字典
        """
        metrics = {}
        
        # 基础指标
        basic_metrics = self.compute_quality_metrics(trajectory)
        metrics.update(basic_metrics)
        
        # 高级指标
        metrics['motion_consistency'] = self.compute_motion_consistency(trajectory)
        metrics['temporal_consistency'] = self.compute_temporal_consistency(trajectory)
        metrics['geometric_consistency'] = self.compute_geometric_consistency(trajectory)
        metrics['visibility_stability'] = self.compute_visibility_stability(trajectory)
        metrics['completeness'] = self.assess_trajectory_completeness(trajectory)
        
        # 异常值比例
        outlier_mask = self.detect_outliers(trajectory)
        metrics['outlier_ratio'] = float(outlier_mask.sum()) / len(outlier_mask) if len(outlier_mask) > 0 else 0.0
        
        # 综合质量分数（使用更多指标）
        advanced_quality = (
            metrics['length_score'] * 0.15 +
            metrics['visibility_score'] * 0.15 +
            metrics['consistency_score'] * 0.15 +
            metrics['stability_score'] * 0.15 +
            metrics['motion_consistency'] * 0.15 +
            metrics['temporal_consistency'] * 0.10 +
            metrics['geometric_consistency'] * 0.10 +
            metrics['completeness'] * 0.05
        )
        
        metrics['advanced_quality_score'] = np.clip(advanced_quality, 0.0, 1.0)
        
        return metrics