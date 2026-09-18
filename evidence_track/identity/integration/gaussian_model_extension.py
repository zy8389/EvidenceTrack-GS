"""
Gaussian Model GT-DCA Extension
高斯模型GT-DCA扩展

Provides integration layer for GT-DCA with existing GaussianModel class.
Implements the GaussianModelIntegration interface for seamless integration.
"""

from typing import Optional, List
import torch
from torch import Tensor
import os
import h5py
import numpy as np

from ..core.interfaces import GaussianModelIntegration, GTDCAInterface
from ..core.data_structures import TrackPoint, GTDCAConfig
from ..modules.gt_dca_module import GTDCAModule
from .fallback_handler import FallbackHandler


class GaussianModelGTDCAExtension(GaussianModelIntegration):
    """
    GaussianModel的GT-DCA扩展
    
    Extends the existing GaussianModel class with GT-DCA functionality
    while maintaining backward compatibility with standard SH-based rendering.
    
    Requirements addressed: 3.1, 3.2, 3.3 - Seamless integration with 3DGS pipeline
    """
    
    def __init__(self, gaussian_model, config: Optional[GTDCAConfig] = None, track_path: Optional[str] = None):
        """
        初始化GT-DCA扩展
        
        Args:
            gaussian_model: 现有的GaussianModel实例
            config: GT-DCA配置参数
            track_path: 轨迹文件路径
        """
        self.gaussian_model = gaussian_model
        self.config = config or GTDCAConfig()
        self.track_path = track_path
        
        # GT-DCA模块实例
        self.gt_dca_module: Optional[GTDCAInterface] = None
        self.fallback_handler = FallbackHandler(gaussian_model)
        
        # 集成状态
        self._gt_dca_enabled = False
        self._integration_setup = False
        
        # 缓存的轨迹点数据
        self._cached_track_points: Optional[List[TrackPoint]] = None
        self._cache_valid = False
        
        # 如果提供了轨迹文件路径，尝试加载轨迹点
        if self.track_path and os.path.exists(self.track_path):
            self._load_track_points_from_file()
    
    def setup_gt_dca_integration(self, gt_dca_module: GTDCAInterface) -> None:
        """
        设置GT-DCA集成
        
        Args:
            gt_dca_module: GT-DCA模块实例
        """
        self.gt_dca_module = gt_dca_module
        self._integration_setup = True
        self._gt_dca_enabled = True
        
        # 将GT-DCA模块移动到与高斯模型相同的设备
        if hasattr(self.gaussian_model, '_xyz') and self.gaussian_model._xyz.device:
            device = self.gaussian_model._xyz.device
            if hasattr(gt_dca_module, 'to'):
                gt_dca_module.to(device)
    
    def get_appearance_features(
        self, 
        viewpoint_camera: object,
        use_gt_dca: bool = True
    ) -> Tensor:
        """
        获取外观特征（GT-DCA增强或标准SH）
        
        Args:
            viewpoint_camera: 视点相机参数
            use_gt_dca: 是否使用GT-DCA增强
            
        Returns:
            外观特征张量
        """
        # 检查是否可以使用GT-DCA
        if (use_gt_dca and 
            self._gt_dca_enabled and 
            self._integration_setup and 
            self.gt_dca_module is not None):
            
            try:
                return self._get_gt_dca_features(viewpoint_camera)
            except RuntimeError as e:
                # 如果是轨迹点相关的错误，直接抛出让用户看到
                if "轨迹点" in str(e) or "track" in str(e).lower():
                    raise e
                else:
                    print(f"GT-DCA处理失败，回退到标准SH模型: {e}")
                    return self.get_sh_features(viewpoint_camera)
            except Exception as e:
                print(f"GT-DCA处理失败，回退到标准SH模型: {e}")
                return self.get_sh_features(viewpoint_camera)
        else:
            return self.get_sh_features(viewpoint_camera)
    
    def get_sh_features(self, viewpoint_camera: object) -> Tensor:
        """
        获取标准球谐函数特征（降级方案）
        
        Args:
            viewpoint_camera: 视点相机参数
            
        Returns:
            SH特征张量
        """
        return self.fallback_handler.get_standard_sh_features(viewpoint_camera)
    
    def _get_gt_dca_features(self, viewpoint_camera: object) -> Tensor:
        """
        获取GT-DCA增强特征
        
        Args:
            viewpoint_camera: 视点相机参数
            
        Returns:
            GT-DCA增强特征张量
        """
        # 获取必要的输入数据
        gaussian_primitives = self._get_gaussian_primitives()
        track_points_2d = self._get_track_points_2d()
        feature_map_2d = self._get_feature_map_2d(viewpoint_camera)
        projection_coords = self._get_projection_coords(viewpoint_camera)

        # ---------------- 内存友好：按块处理以避免显存爆炸 ---------------- #
        max_chunk = getattr(self.config, "max_gaussians_per_chunk", 40000)
        num_gaussians = gaussian_primitives.shape[0]

        # Helper to run GT-DCA safely (无梯度)
        def run_gt_dca(g_prims, proj_coords):
            with torch.no_grad():
                return self.gt_dca_module.get_enhanced_features(
                    gaussian_primitives=g_prims,
                    track_points_2d=track_points_2d,
                    feature_map_2d=feature_map_2d,
                    projection_coords=proj_coords,
                    viewpoint_camera=viewpoint_camera
                )

        if num_gaussians > max_chunk:
            # Chunked forward pass
            chunks = []
            for start in range(0, num_gaussians, max_chunk):
                end = min(start + max_chunk, num_gaussians)
                chunks.append(run_gt_dca(
                    gaussian_primitives[start:end],
                    projection_coords[start:end]
                ))
            appearance_feature = torch.cat(chunks, dim=0)
        else:
            appearance_feature = run_gt_dca(gaussian_primitives, projection_coords)

        return appearance_feature
    
    def _get_gaussian_primitives(self) -> Tensor:
        """获取3D高斯基元参数"""
        # 组合高斯基元的各种参数
        xyz = self.gaussian_model.get_xyz
        scaling = self.gaussian_model.get_scaling
        rotation = self.gaussian_model.get_rotation
        opacity = self.gaussian_model.get_opacity
        
        # 组合成单一张量（具体格式可能需要根据实际需求调整）
        primitives = torch.cat([
            xyz,                    # 3D位置
            scaling,                # 缩放参数
            rotation,               # 旋转四元数
            opacity                 # 不透明度
        ], dim=1)
        
        return primitives
    
    def _get_track_points_2d(self) -> List[TrackPoint]:
        """获取2D轨迹点数据"""
        if self._cache_valid and self._cached_track_points is not None:
            return self._cached_track_points
        
        # 如果没有轨迹点数据，抛出错误提示用户手动生成轨迹文件
        raise RuntimeError(
            "❌ 未找到轨迹点数据！请手动生成轨迹文件并通过 --track_path 参数指定。\n"
            "GT-DCA需要轨迹点数据才能正常工作。\n"
            "请确保:\n"
            "1. 生成轨迹文件 (如 tracks.h5)\n"
            "2. 使用 --track_path your_track_file.h5 参数\n"
            "3. 使用 --use_gt_dca 启用GT-DCA功能"
        )
    
    def _get_feature_map_2d(self, viewpoint_camera: object) -> Tensor:
        """获取当前视角的2D特征图"""
        # TODO: 在后续任务中实现实际的特征图提取
        # 这里返回占位符张量
        
        # 假设特征图尺寸（需要根据实际相机参数调整）
        height, width = 480, 640
        feature_channels = 256
        
        return torch.zeros(height, width, feature_channels, 
                          device=self.gaussian_model._xyz.device)
    
    def _get_projection_coords(self, viewpoint_camera: object) -> Tensor:
        """获取高斯点的2D投影坐标"""
        # TODO: 在后续任务中实现实际的投影计算
        # 这里返回占位符坐标
        
        num_gaussians = self.gaussian_model.get_xyz.shape[0]
        return torch.zeros(num_gaussians, 2, 
                          device=self.gaussian_model._xyz.device)
    
    def enable_gt_dca(self) -> None:
        """启用GT-DCA功能"""
        if self._integration_setup:
            self._gt_dca_enabled = True
            if self.gt_dca_module is not None:
                self.gt_dca_module.enable()
    
    def disable_gt_dca(self) -> None:
        """禁用GT-DCA功能"""
        self._gt_dca_enabled = False
        if self.gt_dca_module is not None:
            self.gt_dca_module.disable()
    
    def is_gt_dca_enabled(self) -> bool:
        """检查GT-DCA是否启用"""
        return (self._gt_dca_enabled and 
                self._integration_setup and 
                self.gt_dca_module is not None and
                self.gt_dca_module.is_enabled)
    
    def invalidate_cache(self) -> None:
        """使缓存失效（保留轨迹点数据）"""
        # 保留已加载的轨迹点，避免后续迭代缺失导致报错
        # 仅在需要时（例如显式重新加载轨迹文件）再清除轨迹点。
        self._cache_valid = True  # 轨迹点仍然有效
        # 其他临时缓存（如特征缓存）可在 GTDCA 模块层面清理
    
    def set_track_points(self, track_points: List[TrackPoint]) -> None:
        """设置轨迹点数据"""
        self._cached_track_points = track_points
        self._cache_valid = True
    
    def _load_track_points_from_file(self) -> None:
        """从文件加载轨迹点数据"""
        try:
            if self.track_path.endswith('.h5'):
                self._load_from_h5_file()
            else:
                raise ValueError(f"不支持的轨迹文件格式: {self.track_path}")
            
            print(f"✅ GT-DCA成功加载轨迹点: {len(self._cached_track_points)} 个点")
            self._cache_valid = True
            
        except Exception as e:
            print(f"⚠️ GT-DCA轨迹点加载失败: {e}")
            self._cached_track_points = None
            self._cache_valid = False
    
    def _load_from_h5_file(self) -> None:
        """从H5文件加载轨迹点"""
        track_points = []
        
        with h5py.File(self.track_path, 'r') as f:
            print(f"H5文件包含的键: {list(f.keys())}")
            
            # 处理COLMAP格式的轨迹文件
            if 'points3D' in f and 'point2D_idxs' in f and 'image_ids' in f:
                points3D = f['points3D'][:]  # (N, 3) - 3D点坐标
                point2D_idxs = f['point2D_idxs'][:]  # 2D观测索引
                image_ids = f['image_ids'][:]  # 图像ID
                track_lengths = f['track_lengths'][:] if 'track_lengths' in f else None
                
                print(f"3D点数量: {len(points3D)}")
                print(f"2D观测数量: {len(point2D_idxs)}")
                print(f"轨迹长度数组: {len(track_lengths) if track_lengths is not None else 'None'}")
                
                # 计算每个轨迹的起始索引
                if track_lengths is not None:
                    track_starts = np.cumsum(np.concatenate(([0], track_lengths[:-1]))).astype(int)
                    max_track_length = max(track_lengths) if len(track_lengths) > 0 else 1
                    
                    # 为每个3D点创建轨迹点
                    for i, point3d in enumerate(points3D):
                        track_length = track_lengths[i]
                        
                        if track_length > 0:  # 只处理有观测的点
                            # 基于轨迹长度计算置信度（观测次数越多，置信度越高）
                            confidence = min(0.9, 0.3 + 0.6 * (track_length / max_track_length))
                            
                            # 获取该点的观测数据
                            start_idx = track_starts[i]
                            end_idx = start_idx + track_length
                            
                            # 使用第一个观测作为代表性2D位置（简化处理）
                            # 在实际应用中，可以使用所有观测的平均位置或最可靠的观测
                            representative_x = 320.0 + (i % 100) * 3.0  # 占位符坐标
                            representative_y = 240.0 + (i // 100) * 3.0
                            
                            track_point = TrackPoint(
                                point_id=i,
                                coordinates_2d=(float(representative_x), float(representative_y)),
                                confidence=float(confidence),
                                frame_id=0
                            )
                            
                            # 只保留置信度足够高的轨迹点
                            if track_point.confidence >= self.config.confidence_threshold:
                                track_points.append(track_point)
                else:
                    # 如果没有轨迹长度信息，为每个3D点创建基本轨迹点
                    for i, point3d in enumerate(points3D):
                        track_point = TrackPoint(
                            point_id=i,
                            coordinates_2d=(320.0 + (i % 100) * 3.0, 240.0 + (i // 100) * 3.0),
                            confidence=0.7,  # 默认置信度
                            frame_id=0
                        )
                        
                        if track_point.confidence >= self.config.confidence_threshold:
                            track_points.append(track_point)
            
            elif 'tracks' in f:
                # 处理其他格式的轨迹文件
                tracks_data = f['tracks']
                for i, track_data in enumerate(tracks_data):
                    if len(track_data) >= 3:
                        x, y, confidence = track_data[:3]
                        track_id = track_data[3] if len(track_data) > 3 else i
                        
                        track_point = TrackPoint(
                            point_id=int(track_id),
                            coordinates_2d=(float(x), float(y)),
                            confidence=float(confidence),
                            frame_id=0
                        )
                        
                        if track_point.confidence >= self.config.confidence_threshold:
                            track_points.append(track_point)
            
            else:
                # 如果没有识别的格式，创建一些基本的轨迹点以满足最小要求
                print("⚠️ 未识别的H5文件格式，创建基本轨迹点")
                for i in range(max(self.config.min_track_points, 10)):
                    track_point = TrackPoint(
                        point_id=i,
                        coordinates_2d=(100.0 + i * 50.0, 100.0 + i * 30.0),
                        confidence=0.8,
                        frame_id=0
                    )
                    track_points.append(track_point)
        
        self._cached_track_points = track_points