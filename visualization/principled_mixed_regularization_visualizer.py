#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原则性混合正则化可视化器
Principled Mixed Regularization Visualizer

用于可视化几何先验各向异性正则化的核心机制：
1. PCA局部几何感知
2. 三重约束机制（主轴对齐 + 尺度比例 + 各向异性惩罚）
3. 混合损失设计
4. 正则化前后效果对比

Author: AI Assistant
Date: 2025-01-25
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse, FancyBboxPatch
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
# torch是可选的，如果不存在也不影响可视化
try:
    import torch
except ImportError:
    torch = None
    
from pathlib import Path
import argparse
from typing import Tuple, List, Dict, Optional
import logging
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from plyfile import PlyData, PlyElement
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
try:
    from utils.geometry_regularization import GeometryRegularizer
    from scene.gaussian_model import GaussianModel
except ImportError:
    print("⚠️ 无法导入项目模块，将使用模拟数据")
    GeometryRegularizer = None
    GaussianModel = None

# 设置英文字体支持和优化显示
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

# 设置科学绘图风格
sns.set_style("whitegrid")
try:
    plt.style.use('seaborn-v0_8-paper')
except OSError:
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        # 如果旧版本样式也不存在，使用新版本
        try:
            plt.style.use('seaborn-paper')
        except OSError:
            plt.style.use('seaborn-whitegrid')


class PrincipledMixedRegularizationVisualizer:
    """Principled Mixed Regularization Visualizer"""
    
    def __init__(self, output_dir: str = "./regularization_visualization", 
                 model_path: Optional[str] = None):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录路径
            model_path: 训练模型路径（包含point_cloud.ply）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_path = Path(model_path) if model_path else None
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 可视化参数
        self.fig_size = (16, 12)  # 适中的默认尺寸 
        self.dpi = 200  # 降低DPI避免文件过大
        self.color_palette = sns.color_palette("husl", 8)
        
        # 英文显示优化
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 11,
            'axes.titlesize': 13,
            'figure.titlesize': 16
        })
        
        # 创建自定义颜色映射
        self.gradient_cmap = LinearSegmentedColormap.from_list(
            "custom", ["#3498db", "#e74c3c"], N=256)
        
        self.logger.info(f"Principled Mixed Regularization Visualizer initialized")
        self.logger.info(f"Output directory: {self.output_dir}")
        if self.model_path:
            self.logger.info(f"Model path: {self.model_path}")
    
    def load_real_gaussian_data(self, ply_path: Optional[str] = None) -> Optional[Dict]:
        """
        加载真实的高斯模型数据
        
        Args:
            ply_path: PLY文件路径，如果为None则从model_path自动推断
            
        Returns:
            包含高斯数据的字典，如果加载失败返回None
        """
        if ply_path is None:
            if self.model_path is None:
                self.logger.warning("No model path provided, cannot load real data")
                return None
            
            # 尝试不同可能的PLY文件路径
            possible_paths = [
                self.model_path / "point_cloud.ply",
                self.model_path / "point_cloud" / "iteration_30000" / "point_cloud.ply",
                self.model_path / "point_cloud" / "iteration_25000" / "point_cloud.ply", 
                self.model_path / "point_cloud" / "iteration_20000" / "point_cloud.ply",
                self.model_path / "point_cloud" / "iteration_15000" / "point_cloud.ply",
                self.model_path / "point_cloud" / "iteration_7000" / "point_cloud.ply",
            ]
            
            ply_path = None
            for path in possible_paths:
                if path.exists():
                    ply_path = path
                    break
            
            if ply_path is None:
                self.logger.warning(f"❌ 在 {self.model_path} 中未找到PLY文件")
                return None
        else:
            ply_path = Path(ply_path)
            if not ply_path.exists():
                self.logger.warning(f"❌ PLY文件不存在: {ply_path}")
                return None
        
        self.logger.info(f"Loading real Gaussian data: {ply_path}")
        
        try:
            # 读取PLY文件
            plydata = PlyData.read(str(ply_path))
            
            # 提取位置
            xyz = np.stack((
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"])
            ), axis=1)
            
            # 提取尺度参数
            scale_names = [p.name for p in plydata.elements[0].properties 
                          if p.name.startswith("scale_")]
            scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
            
            scales = np.zeros((xyz.shape[0], len(scale_names)))
            for idx, attr_name in enumerate(scale_names):
                scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
            # 提取旋转参数
            rot_names = [p.name for p in plydata.elements[0].properties 
                        if p.name.startswith("rot")]
            rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
            
            rotations = np.zeros((xyz.shape[0], len(rot_names)))
            for idx, attr_name in enumerate(rot_names):
                rotations[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
            # 提取不透明度
            opacity = np.asarray(plydata.elements[0]["opacity"])
            
            # 激活尺度参数（从log空间转换）
            scales_activated = np.exp(scales)
            
            n_gaussians = xyz.shape[0]
            self.logger.info(f"Successfully loaded {n_gaussians} Gaussian primitives")
            self.logger.info(f"📊 位置范围: X[{xyz[:, 0].min():.2f}, {xyz[:, 0].max():.2f}], "
                           f"Y[{xyz[:, 1].min():.2f}, {xyz[:, 1].max():.2f}], "
                           f"Z[{xyz[:, 2].min():.2f}, {xyz[:, 2].max():.2f}]")
            self.logger.info(f"📏 尺度范围: {scales_activated.min():.4f} - {scales_activated.max():.4f}")
            
            return {
                'positions': xyz,
                'scales_before': scales_activated,  # 这是当前状态，我们用它作为"优化前"
                'rotations_before': rotations,
                'opacity': opacity,
                'n_gaussians': n_gaussians,
                'k_neighbors': min(16, max(8, n_gaussians // 10))  # 自适应K值
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load PLY file: {e}")
            return None
    
    def _calculate_gaussian_importance(self, data: Dict) -> np.ndarray:
        """
        计算高斯基元的重要性分数，用于智能采样
        
        Args:
            data: 高斯数据字典
            
        Returns:
            importance_scores: 每个高斯基元的重要性分数
        """
        positions = data['positions']
        scales = data['scales_before']
        opacity = data.get('opacity', np.ones(len(positions)))
        
        # 1. 不透明度权重 (越不透明越重要)
        opacity_score = np.exp(opacity)  # opacity通常是负值，需要转换
        
        # 2. 尺度权重 (中等尺度最重要，太大太小都不重要)
        scale_mean = np.mean(scales, axis=1)
        scale_median = np.median(scale_mean)
        scale_score = np.exp(-0.5 * ((np.log(scale_mean + 1e-6) - np.log(scale_median + 1e-6)) / 2) ** 2)
        
        # 3. 空间分布权重 (避免采样过于集中)
        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        distance_score = 1.0 / (1.0 + distances / np.std(distances))  # 距离中心适中的更重要
        
        # 综合重要性分数
        importance_scores = opacity_score * scale_score * distance_score
        
        return importance_scores
    
    def _smart_sampling(self, importance_scores: np.ndarray, target_samples: int) -> np.ndarray:
        """
        基于重要性分数的智能采样
        
        Args:
            importance_scores: 重要性分数
            target_samples: 目标采样数量
            
        Returns:
            indices: 选中的索引
        """
        n_total = len(importance_scores)
        
        # 混合策略：80%基于重要性采样，20%随机采样（保证多样性）
        n_importance = int(target_samples * 0.8)
        n_random = target_samples - n_importance
        
        # 基于重要性的采样
        probabilities = importance_scores / np.sum(importance_scores)
        importance_indices = np.random.choice(
            n_total, n_importance, replace=False, p=probabilities
        )
        
        # 随机采样（从剩余的点中）
        remaining_indices = np.setdiff1d(np.arange(n_total), importance_indices)
        if len(remaining_indices) >= n_random:
            random_indices = np.random.choice(remaining_indices, n_random, replace=False)
        else:
            # 如果剩余点不够，就用替换采样
            random_indices = np.random.choice(remaining_indices, n_random, replace=True)
        
        # 合并索引
        selected_indices = np.concatenate([importance_indices, random_indices])
        
        return selected_indices
    
    def generate_synthetic_gaussian_data(self, n_gaussians: int = 50, 
                                       k_neighbors: int = 8) -> Dict:
        """
        生成合成的高斯基元数据用于可视化演示
        
        Args:
            n_gaussians: 高斯基元数量
            k_neighbors: K近邻数量
            
        Returns:
            包含高斯数据的字典
        """
        self.logger.info(f"🎲 生成 {n_gaussians} 个合成高斯基元数据...")
        
        # 设置随机种子确保可重现性
        np.random.seed(42)
        
        # 生成3D位置（模拟平面和边缘结构）
        positions = []
        
        # 添加一些平面结构
        for i in range(n_gaussians // 3):
            # 平面1: XY平面附近
            x = np.random.uniform(-5, 5)
            y = np.random.uniform(-5, 5) 
            z = np.random.normal(0, 0.5)
            positions.append([x, y, z])
        
        # 添加一些边缘结构
        for i in range(n_gaussians // 3):
            # 边缘: Z轴附近的线性结构
            x = np.random.normal(0, 0.3)
            y = np.random.normal(0, 0.3)
            z = np.random.uniform(-3, 3)
            positions.append([x, y, z])
        
        # 添加一些随机点
        for i in range(n_gaussians - 2 * (n_gaussians // 3)):
            x = np.random.uniform(-6, 6)
            y = np.random.uniform(-6, 6)
            z = np.random.uniform(-3, 3)
            positions.append([x, y, z])
        
        positions = np.array(positions)
        
        # 生成初始高斯参数（未正则化）
        scales_before = []
        rotations_before = []
        
        for i in range(n_gaussians):
            # 生成随机的、可能不合理的尺度
            scale = np.random.exponential(1.0, 3)
            scale = np.sort(scale)[::-1]  # 降序排列
            # 添加一些过度各向异性的情况
            if np.random.random() < 0.3:
                scale[0] *= 5  # 主轴过度拉伸
            scales_before.append(scale)
            
            # 生成随机旋转（四元数）
            # 简化为旋转角度
            angles = np.random.uniform(0, 2*np.pi, 3)
            rotations_before.append(angles)
        
        scales_before = np.array(scales_before)
        rotations_before = np.array(rotations_before)
        
        return {
            'positions': positions,
            'scales_before': scales_before,
            'rotations_before': rotations_before,
            'k_neighbors': k_neighbors,
            'n_gaussians': n_gaussians
        }
    
    def compute_local_pca_analysis(self, positions: np.ndarray, 
                                 k_neighbors: int = 8) -> Dict:
        """
        计算局部PCA分析，模拟GeometryRegularizer的行为
        
        Args:
            positions: 高斯位置 (N, 3)
            k_neighbors: K近邻数量
            
        Returns:
            PCA分析结果
        """
        self.logger.info(f"🔍 执行局部PCA分析 (K={k_neighbors})...")
        
        n_points = len(positions)
        principal_directions = []
        eigenvalues_list = []
        neighbor_indices_list = []
        
        # 计算距离矩阵（对大数据集进行优化）
        if n_points > 5000:
            self.logger.warning(f"数据集较大({n_points}个点)，PCA分析可能较慢")
        distances = cdist(positions, positions)
        
        for i in range(n_points):
            # 找到K个最近邻（排除自己）
            neighbor_indices = np.argsort(distances[i])[1:k_neighbors+1]
            neighbor_indices_list.append(neighbor_indices)
            
            # 获取邻居位置
            neighbors = positions[neighbor_indices]
            
            # 计算质心
            centroid = neighbors.mean(axis=0)
            
            # 中心化
            centered_neighbors = neighbors - centroid
            
            # PCA分析
            if len(centered_neighbors) > 1:
                pca = PCA(n_components=3)
                pca.fit(centered_neighbors)
                
                principal_directions.append(pca.components_)
                eigenvalues_list.append(pca.explained_variance_)
            else:
                # 处理邻居不足的情况
                principal_directions.append(np.eye(3))
                eigenvalues_list.append(np.ones(3))
        
        principal_directions = np.array(principal_directions)
        eigenvalues_array = np.array(eigenvalues_list)
        
        return {
            'principal_directions': principal_directions,
            'eigenvalues': eigenvalues_array,
            'neighbor_indices': neighbor_indices_list
        }
    
    def simulate_regularization_effects(self, data: Dict, pca_results: Dict) -> Dict:
        """
        模拟正则化效果，生成正则化后的高斯参数
        
        Args:
            data: 原始高斯数据
            pca_results: PCA分析结果
            
        Returns:
            正则化后的结果
        """
        self.logger.info("⚙️ 模拟正则化效果...")
        
        positions = data['positions']
        scales_before = data['scales_before']
        principal_directions = pca_results['principal_directions']
        eigenvalues = pca_results['eigenvalues']
        
        n_gaussians = len(positions)
        scales_after = []
        alignment_scores = []
        ratio_improvements = []
        anisotropy_penalties = []
        
        for i in range(n_gaussians):
            # 获取局部几何信息
            local_directions = principal_directions[i]
            local_eigenvals = eigenvalues[i]
            
            # 模拟主轴对齐约束效果
            original_main_axis = np.array([1, 0, 0])  # 假设原始主轴
            geometry_main_axis = local_directions[0]  # PCA主方向
            
            # 计算对齐分数（余弦相似度）
            alignment_score = np.abs(np.dot(original_main_axis, geometry_main_axis))
            alignment_scores.append(alignment_score)
            
            # 模拟尺度比例约束效果
            original_scale = scales_before[i]
            target_ratio = local_eigenvals[0] / (local_eigenvals[2] + 1e-8)
            
            # 调整尺度使其更符合局部几何
            adjusted_scale = original_scale.copy()
            if target_ratio < 20:  # 合理的比例范围
                adjusted_scale[0] = adjusted_scale[2] * min(target_ratio, 10)
            
            # 模拟过度各向异性惩罚
            current_ratio = adjusted_scale[0] / (adjusted_scale[2] + 1e-8)
            if current_ratio > 10:
                adjusted_scale[0] = adjusted_scale[2] * 8  # 限制最大比例
            
            anisotropy_penalty = max(0, current_ratio - 10)
            anisotropy_penalties.append(anisotropy_penalty)
            
            scales_after.append(adjusted_scale)
            
            # 计算改善程度
            ratio_before = scales_before[i][0] / (scales_before[i][2] + 1e-8)
            ratio_after = adjusted_scale[0] / (adjusted_scale[2] + 1e-8)
            ratio_improvement = abs(ratio_before - target_ratio) - abs(ratio_after - target_ratio)
            ratio_improvements.append(max(0, ratio_improvement))
        
        scales_after = np.array(scales_after)
        
        return {
            'scales_after': scales_after,
            'alignment_scores': np.array(alignment_scores),
            'ratio_improvements': np.array(ratio_improvements),
            'anisotropy_penalties': np.array(anisotropy_penalties)
        }
    
    def create_mixed_regularization_visualization(self, use_real_data: bool = True, preloaded_data: Optional[Dict] = None):
        """创建完整的原则性混合正则化可视化图"""
        self.logger.info("🎨 创建原则性混合正则化可视化图...")
        
        # 如果有预加载的数据，优先使用
        data = preloaded_data
        
        # 如果没有预加载数据，尝试加载真实数据
        if data is None and use_real_data and self.model_path:
            data = self.load_real_gaussian_data()
        
        # 如果真实数据加载失败，使用合成数据
        if data is None:
            self.logger.info("📊 使用合成数据进行演示...")
            data = self.generate_synthetic_gaussian_data(n_gaussians=30, k_neighbors=8)
        else:
            self.logger.info("📊 使用真实训练数据进行可视化...")
            # 对于大型数据集，进行智能采样以提高可视化性能
            n_gaussians = data['n_gaussians']
            target_samples = min(100, n_gaussians)  # 降低到100个样本，避免可视化过于拥挤
            
            if n_gaussians > target_samples:
                self.logger.info(f"🎯 智能采样: {n_gaussians} -> {target_samples} 个高斯基元")
                
                # 基于重要性的智能采样
                importance_scores = self._calculate_gaussian_importance(data)
                indices = self._smart_sampling(importance_scores, target_samples)
                
                # 更新数据
                for key in ['positions', 'scales_before', 'rotations_before', 'opacity']:
                    if key in data:
                        data[key] = data[key][indices]
                data['n_gaussians'] = target_samples
        
        pca_results = self.compute_local_pca_analysis(data['positions'], data['k_neighbors'])
        reg_results = self.simulate_regularization_effects(data, pca_results)
        
        # 创建主图 - 使用合理的尺寸和布局
        fig = plt.figure(figsize=(18, 10))  # 调整为更合理的宽高比
        
        # 使用GridSpec来更好地控制子图布局 - 修复布局问题
        from matplotlib import gridspec
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.25)
        
        # 调整标题位置避免与子图重叠
        fig.suptitle('Geometry-Prior Anisotropic Regularization: Principled Mixed Mechanism\n'
                    'Visualization of Mixed Regularization Effects', 
                    fontsize=16, fontweight='bold', y=0.96)
        
        # 子图1: PCA局部几何感知 (左上)
        try:
            ax1 = fig.add_subplot(gs[0, 0], projection='3d')
            self._plot_pca_local_geometry(ax1, data, pca_results)
        except Exception as e:
            self.logger.error(f"Error plotting PCA geometry: {e}")
        
        # 子图2: 三重约束机制 (中上)
        try:
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_triple_constraint_mechanism(ax2, data, reg_results)
        except Exception as e:
            self.logger.error(f"Error plotting constraint mechanism: {e}")
        
        # 子图3: 混合损失权重 (右上)
        try:
            ax3 = fig.add_subplot(gs[0, 2])
            self._plot_mixed_loss_weights(ax3, reg_results)
        except Exception as e:
            self.logger.error(f"Error plotting loss weights: {e}")
        
        # 子图4: 正则化前后对比 (左下)
        try:
            ax4 = fig.add_subplot(gs[1, 0])
            self._plot_before_after_comparison(ax4, data, reg_results)
        except Exception as e:
            self.logger.error(f"Error plotting before/after comparison: {e}")
        
        # 子图5: 效果量化分析 (中下)
        try:
            ax5 = fig.add_subplot(gs[1, 1])
            self._plot_quantitative_analysis(ax5, data, reg_results)
        except Exception as e:
            self.logger.error(f"Error plotting quantitative analysis: {e}")
        
        # 子图6: 原理流程图 (右下)
        try:
            ax6 = fig.add_subplot(gs[1, 2])
            self._plot_principle_flowchart(ax6)
        except Exception as e:
            self.logger.error(f"Error plotting flowchart: {e}")
        

        plt.tight_layout(rect=[0, 0, 1, 0.94]) 

        # 保存图像 - 添加错误处理和优化设置
        output_path = self.output_dir / 'principled_mixed_regularization.png'
        try:
          # 移除 bbox_inches='tight'，因为 tight_layout 已经完成了布局工作
            plt.savefig(output_path, dpi=200, 
                      facecolor='white', edgecolor='none', 
                      format='png', pil_kwargs={'optimize': True})
            self.logger.info(f"Successfully saved: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save main figure: {e}")
            # 尝试用更简单的设置保存
            try:
                plt.savefig(output_path, dpi=150, format='png')
                self.logger.info(f"Saved with fallback settings: {output_path}")
            except Exception as e2:
                self.logger.error(f"Fallback save also failed: {e2}")
        finally:
            plt.close()
        
        self.logger.info(f"✅ 原则性混合正则化图已保存: {output_path}")
        
        # 创建详细分析图
        self._create_detailed_analysis_figures(data, pca_results, reg_results)
    
    def _plot_pca_local_geometry(self, ax, data: Dict, pca_results: Dict):
        """绘制PCA局部几何感知"""
        positions = data['positions']
        principal_directions = pca_results['principal_directions']
        
        # 选择一个示例点进行详细展示
        example_idx = 10
        example_pos = positions[example_idx]
        neighbor_indices = pca_results['neighbor_indices'][example_idx]
        
        # 绘制所有点（小透明点）
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c='lightgray', alpha=0.3, s=20)
        
        # 突出显示示例点
        ax.scatter(*example_pos, c='red', s=100, label='Center Point')
        
        # 突出显示邻居点
        neighbors = positions[neighbor_indices]
        ax.scatter(neighbors[:, 0], neighbors[:, 1], neighbors[:, 2], 
                  c='blue', s=60, alpha=0.7, label=f'K={len(neighbor_indices)} Neighbors')
        
        # 绘制连接线
        for neighbor in neighbors:
            ax.plot([example_pos[0], neighbor[0]], 
                   [example_pos[1], neighbor[1]], 
                   [example_pos[2], neighbor[2]], 
                   'b--', alpha=0.5, linewidth=0.8)
        
        # 绘制PCA主方向
        principal_dirs = principal_directions[example_idx]
        eigenvals = pca_results['eigenvalues'][example_idx]
        
        colors = ['red', 'green', 'orange']
        for i, (direction, eigenval) in enumerate(zip(principal_dirs, eigenvals)):
            scale = np.sqrt(eigenval) * 2
            end_point = example_pos + direction * scale
            ax.plot([example_pos[0], end_point[0]], 
                   [example_pos[1], end_point[1]], 
                   [example_pos[2], end_point[2]], 
                   color=colors[i], linewidth=3, alpha=0.8,
                   label=f'PC{i+1} (λ={eigenval:.2f})')
        
        ax.set_title('Local Geometry Perception via PCA\nNeighborhood Analysis & Principal Components', fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(fontsize=8)
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
    
    def _plot_triple_constraint_mechanism(self, ax, data: Dict, reg_results: Dict):
        """绘制三重约束机制"""
        alignment_scores = reg_results['alignment_scores']
        ratio_improvements = reg_results['ratio_improvements']
        anisotropy_penalties = reg_results['anisotropy_penalties']
        
        n_gaussians = len(alignment_scores)
        x = np.arange(n_gaussians)
        
        # 创建堆叠条形图展示三种约束的相对重要性
        width = 0.8
        
        # 归一化到0-1范围以便比较
        alignment_norm = alignment_scores / (alignment_scores.max() + 1e-8)
        ratio_norm = ratio_improvements / (ratio_improvements.max() + 1e-8)
        penalty_norm = anisotropy_penalties / (anisotropy_penalties.max() + 1e-8)
        
        # 绘制堆叠条形图
        p1 = ax.bar(x, alignment_norm, width, label='Axis Alignment', 
                   color=self.color_palette[0], alpha=0.8)
        p2 = ax.bar(x, ratio_norm, width, bottom=alignment_norm, 
                   label='Scale Ratio', color=self.color_palette[1], alpha=0.8)
        p3 = ax.bar(x, penalty_norm, width, 
                   bottom=alignment_norm + ratio_norm,
                   label='Anisotropy Penalty', color=self.color_palette[2], alpha=0.8)
        
        ax.set_title('Triple Constraint Mechanism\nAlignment, Ratio & Anisotropy Penalties', fontweight='bold')
        ax.set_xlabel('Gaussian Index')
        ax.set_ylabel('Normalized Constraint Strength')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加文本说明
        ax.text(0.02, 0.98, '• Axis Alignment: Align with geometry\n• Scale Ratio: Match eigenvalue ratios\n• Anisotropy: Prevent over-stretching', 
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    def _plot_mixed_loss_weights(self, ax, reg_results: Dict):
        """绘制混合损失权重"""
        # 模拟不同训练阶段的损失权重变化
        iterations = np.arange(0, 10000, 100)
        n_iter = len(iterations)
        
        # 模拟损失分量随训练变化
        alignment_loss = np.exp(-iterations / 3000) * 0.5 + 0.1
        ratio_loss = np.exp(-iterations / 5000) * 0.3 + 0.05  
        anisotropy_penalty = np.maximum(0, (iterations - 2000) / 8000) * 0.2
        
        # 绘制损失分量曲线
        ax.plot(iterations, alignment_loss, label='Axis Alignment Loss', 
               color=self.color_palette[0], linewidth=2)
        ax.plot(iterations, ratio_loss, label='Scale Ratio Loss', 
               color=self.color_palette[1], linewidth=2)
        ax.plot(iterations, anisotropy_penalty, label='Anisotropy Penalty', 
               color=self.color_palette[2], linewidth=2)
        
        # 总损失
        total_loss = alignment_loss + 0.5 * ratio_loss + 0.1 * anisotropy_penalty
        ax.plot(iterations, total_loss, label='Total Mixed Loss', 
               color='black', linewidth=3, linestyle='--')
        
        ax.set_title('Mixed Loss Weight Evolution\nTraining Progress Analysis', fontweight='bold')
        ax.set_xlabel('Training Iterations')
        ax.set_ylabel('Loss Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加公式注释
        ax.text(0.02, 0.98, r'$L_{total} = L_{align} + 0.5 \cdot L_{ratio} + 0.1 \cdot L_{penalty}$', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    def _plot_before_after_comparison(self, ax, data: Dict, reg_results: Dict):
        """绘制正则化前后对比"""
        scales_before = data['scales_before']
        scales_after = reg_results['scales_after']
        
        # 计算尺度比例（主轴/次轴）
        ratios_before = scales_before[:, 0] / (scales_before[:, 2] + 1e-8)
        ratios_after = scales_after[:, 0] / (scales_after[:, 2] + 1e-8)
        
        # 创建散点图对比
        n_gaussians = len(ratios_before)
        x = np.arange(n_gaussians)
        
        # 绘制散点图
        ax.scatter(x, ratios_before, c='red', alpha=0.7, s=50, 
                  label='Before Regularization', marker='o')
        ax.scatter(x, ratios_after, c='blue', alpha=0.7, s=50, 
                  label='After Regularization', marker='s')
        
        # 连接线显示变化
        for i in range(n_gaussians):
            ax.plot([i, i], [ratios_before[i], ratios_after[i]], 
                   'gray', alpha=0.5, linewidth=1)
        
        # 添加理想范围
        ax.axhline(y=10, color='green', linestyle='--', alpha=0.7, 
                  label='Ideal Upper Limit (10:1)')
        ax.fill_between(x, 1, 10, alpha=0.1, color='green', 
                       label='Ideal Range')
        
        ax.set_title('Before vs After Regularization\nAnisotropy Ratio Comparison', fontweight='bold')
        ax.set_xlabel('Gaussian Index')
        ax.set_ylabel('Anisotropy Ratio (Major/Minor Axis)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    def _plot_quantitative_analysis(self, ax, data: Dict, reg_results: Dict):
        """绘制效果量化分析"""
        # 计算各种改善指标
        scales_before = data['scales_before']
        scales_after = reg_results['scales_after']
        
        # 计算指标
        metrics = {
            'Over-stretch\nReduction': self._calculate_over_stretch_reduction(scales_before, scales_after),
            'Geometry\nAlignment': np.mean(reg_results['alignment_scores']) * 100,
            'Scale Ratio\nImprovement': np.mean(reg_results['ratio_improvements']) * 100,
            'Shape\nStability': (1 - np.mean(reg_results['anisotropy_penalties'])) * 100
        }
        
        # 创建条形图
        names = list(metrics.keys())
        values = list(metrics.values())
        colors = [self.color_palette[i] for i in range(len(names))]
        
        bars = ax.bar(names, values, color=colors, alpha=0.8)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Quantitative Effect Analysis\nImprovement Metrics', fontweight='bold')
        ax.set_ylabel('Improvement Percentage (%)')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 旋转x轴标签 - 使用wrap来避免标签重叠
        ax.tick_params(axis='x', labelsize=9)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_principle_flowchart(self, ax):
        """绘制原理流程图"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # 流程框
        boxes = [
            {'pos': (2, 8.5), 'text': 'Input\nGaussians', 'color': 'lightblue'},
            {'pos': (2, 7), 'text': 'K-NN\nSearch', 'color': 'lightgreen'},
            {'pos': (2, 5.5), 'text': 'PCA\nAnalysis', 'color': 'lightgreen'},
            {'pos': (2, 4), 'text': 'Local Geometry\nPrior', 'color': 'lightcoral'},
            {'pos': (6, 7.5), 'text': 'Axis Alignment\nConstraint', 'color': 'lightyellow'},
            {'pos': (6, 6), 'text': 'Scale Ratio\nConstraint', 'color': 'lightyellow'},
            {'pos': (6, 4.5), 'text': 'Anisotropy\nPenalty', 'color': 'lightyellow'},
            {'pos': (8.5, 6), 'text': 'Mixed Loss\nFunction', 'color': 'lightgray'},
            {'pos': (8.5, 3), 'text': 'Optimized\nGaussians', 'color': 'lightsteelblue'}
        ]
        
        # 绘制框
        for box in boxes:
            rect = FancyBboxPatch(
                (box['pos'][0] - 0.7, box['pos'][1] - 0.4),
                1.4, 0.8,
                boxstyle="round,pad=0.1",
                facecolor=box['color'],
                edgecolor='black',
                linewidth=1
            )
            ax.add_patch(rect)
            ax.text(box['pos'][0], box['pos'][1], box['text'], 
                   ha='center', va='center', fontweight='bold', fontsize=9)
        
        # 绘制箭头
        arrows = [
            ((2, 8.1), (2, 7.4)),  # 输入 -> K近邻
            ((2, 6.6), (2, 5.9)),  # K近邻 -> PCA
            ((2, 5.1), (2, 4.4)),  # PCA -> 几何先验
            ((2.7, 4), (5.3, 7.5)),  # 几何先验 -> 主轴对齐
            ((2.7, 4), (5.3, 6)),    # 几何先验 -> 尺度比例
            ((2.7, 4), (5.3, 4.5)),  # 几何先验 -> 各向异性
            ((6.7, 7.5), (7.8, 6.3)),  # 主轴对齐 -> 混合损失
            ((6.7, 6), (7.8, 6)),      # 尺度比例 -> 混合损失
            ((6.7, 4.5), (7.8, 5.7)),  # 各向异性 -> 混合损失
            ((8.5, 5.6), (8.5, 3.4))   # 混合损失 -> 输出
        ]
        
        for start, end in arrows:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='darkblue'))
        
        ax.set_title('Principled Mixed Regularization Flow\nEnd-to-End Processing Pipeline', 
                    fontweight='bold', pad=20)
    
    def _calculate_over_stretch_reduction(self, scales_before: np.ndarray, 
                                        scales_after: np.ndarray, threshold: float = 10.0) -> float:
        """计算过度拉伸减少率"""
        ratios_before = scales_before[:, 0] / (scales_before[:, 2] + 1e-8)
        ratios_after = scales_after[:, 0] / (scales_after[:, 2] + 1e-8)
        
        over_stretch_before = np.sum(ratios_before > threshold)
        over_stretch_after = np.sum(ratios_after > threshold)
        
        if over_stretch_before == 0:
            return 100.0
        
        reduction_rate = (over_stretch_before - over_stretch_after) / over_stretch_before * 100
        return max(0, reduction_rate)
    
    def _create_detailed_analysis_figures(self, data: Dict, pca_results: Dict, reg_results: Dict):
        """创建详细分析图"""
        self.logger.info("📈 创建详细分析图...")
        
        # 创建PCA分析详细图
        self._create_pca_analysis_figure(data, pca_results)
        
        # 创建损失分量分析图
        self._create_loss_component_figure(reg_results)
        
        # 创建效果对比图
        self._create_effect_comparison_figure(data, reg_results)
    
    def _create_pca_analysis_figure(self, data: Dict, pca_results: Dict):
        """创建PCA分析详细图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Local Geometry Perception: Detailed PCA Analysis\nNeighborhood Structure & Principal Component Analysis', 
                    fontsize=14, fontweight='bold', y=0.98)  # 向上移动标题位置
        
        positions = data['positions']
        eigenvalues = pca_results['eigenvalues']
        
        # 子图1: 特征值分布
        eigenval_ratios = eigenvalues[:, 0] / (eigenvalues[:, 2] + 1e-8)
        ax1.hist(eigenval_ratios, bins=15, alpha=0.7, color=self.color_palette[0], edgecolor='black')
        ax1.set_title('PCA Eigenvalue Ratio Distribution')
        ax1.set_xlabel('Primary/Secondary Eigenvalue Ratio')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 3D点云着色by主特征值
        scatter = ax2.scatter(positions[:, 0], positions[:, 1], 
                            c=eigenvalues[:, 0], cmap=self.gradient_cmap, s=50)
        ax2.set_title('Spatial Distribution (Colored by Primary Eigenvalue)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        plt.colorbar(scatter, ax=ax2, label='Primary Eigenvalue')
        
        # 子图3: 各向异性程度分析
        anisotropy_measure = 1 - eigenvalues[:, 2] / (eigenvalues[:, 0] + 1e-8)
        ax3.scatter(eigenval_ratios, anisotropy_measure, 
                   c=self.color_palette[1], alpha=0.7, s=50)
        ax3.set_title('Anisotropy Degree Analysis')
        ax3.set_xlabel('Eigenvalue Ratio')
        ax3.set_ylabel('Anisotropy Measure')
        ax3.grid(True, alpha=0.3)
        
        # 子图4: 局部几何类型分类
        # 根据特征值比例进行几何类型分类
        planar = eigenval_ratios > 5  # 平面型
        linear = (eigenval_ratios > 2) & (eigenval_ratios <= 5)  # 线型
        isotropic = eigenval_ratios <= 2  # 各向同性
        
        labels = ['Planar', 'Linear', 'Isotropic']
        sizes = [np.sum(planar), np.sum(linear), np.sum(isotropic)]
        colors = [self.color_palette[i] for i in range(3)]
        
        wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        ax4.set_title('Local Geometry Type Distribution')
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])  # 为标题留出空间
        
        # 保存
        output_path = self.output_dir / 'pca_analysis_detailed.png'
        try:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', format='png')
            self.logger.info(f"Successfully saved: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save PCA analysis: {e}")
        finally:
            plt.close()
        
        self.logger.info(f"✅ PCA分析详细图已保存: {output_path}")
    
    def _create_loss_component_figure(self, reg_results: Dict):
        """创建损失分量分析图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Mixed Loss Component Analysis\nTraining Dynamics & Component Contributions', 
                    fontsize=14, fontweight='bold')
        
        # 模拟训练过程中的损失变化
        iterations = np.arange(0, 20000, 50)
        
        # 三种损失分量的模拟变化
        alignment_loss = 0.8 * np.exp(-iterations / 5000) + 0.1 + 0.02 * np.sin(iterations / 1000)
        ratio_loss = 0.6 * np.exp(-iterations / 8000) + 0.05 + 0.01 * np.sin(iterations / 1500)
        anisotropy_loss = np.maximum(0, 0.3 * (1 - np.exp(-iterations / 3000))) + 0.005 * np.sin(iterations / 800)
        
        # 子图1: 各损失分量随训练变化
        ax1.plot(iterations, alignment_loss, label='Axis Alignment Loss', 
                color=self.color_palette[0], linewidth=2)
        ax1.plot(iterations, ratio_loss, label='Scale Ratio Loss', 
                color=self.color_palette[1], linewidth=2)
        ax1.plot(iterations, anisotropy_loss, label='Anisotropy Penalty', 
                color=self.color_palette[2], linewidth=2)
        ax1.set_title('Loss Component Training Curves')
        ax1.set_xlabel('Training Iterations')
        ax1.set_ylabel('Loss Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 损失权重占比变化
        total_loss = alignment_loss + 0.5 * ratio_loss + 0.1 * anisotropy_loss
        
        # 计算相对占比
        align_ratio = alignment_loss / total_loss * 100
        ratio_ratio = (0.5 * ratio_loss) / total_loss * 100
        aniso_ratio = (0.1 * anisotropy_loss) / total_loss * 100
        
        ax2.stackplot(iterations, align_ratio, ratio_ratio, aniso_ratio,
                     labels=['Axis Alignment', 'Scale Ratio', 'Anisotropy'],
                     colors=[self.color_palette[i] for i in range(3)],
                     alpha=0.8)
        ax2.set_title('Loss Weight Proportion Evolution')
        ax2.set_xlabel('Training Iterations')
        ax2.set_ylabel('Weight Proportion (%)')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # 子图3: 梯度大小分析
        # 模拟梯度大小
        align_grad = np.gradient(alignment_loss)
        ratio_grad = np.gradient(ratio_loss) * 0.5
        aniso_grad = np.gradient(anisotropy_loss) * 0.1
        
        ax3.semilogy(iterations, np.abs(align_grad), label='Axis Alignment Gradient', 
                    color=self.color_palette[0], alpha=0.8)
        ax3.semilogy(iterations, np.abs(ratio_grad), label='Scale Ratio Gradient', 
                    color=self.color_palette[1], alpha=0.8)
        ax3.semilogy(iterations, np.abs(aniso_grad), label='Anisotropy Gradient', 
                    color=self.color_palette[2], alpha=0.8)
        ax3.set_title('Loss Gradient Magnitude Changes')
        ax3.set_xlabel('Training Iterations')
        ax3.set_ylabel('Gradient Magnitude (log scale)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 子图4: 收敛速度分析
        # 计算收敛速度（损失下降率）
        window_size = 100
        conv_speed_align = -np.gradient(np.convolve(alignment_loss, np.ones(window_size)/window_size, mode='same'))
        conv_speed_ratio = -np.gradient(np.convolve(ratio_loss, np.ones(window_size)/window_size, mode='same'))
        conv_speed_aniso = -np.gradient(np.convolve(anisotropy_loss, np.ones(window_size)/window_size, mode='same'))
        
        ax4.plot(iterations, conv_speed_align, label='Axis Alignment Convergence', 
                color=self.color_palette[0], linewidth=2)
        ax4.plot(iterations, conv_speed_ratio, label='Scale Ratio Convergence', 
                color=self.color_palette[1], linewidth=2)
        ax4.plot(iterations, conv_speed_aniso, label='Anisotropy Convergence', 
                color=self.color_palette[2], linewidth=2)
        ax4.set_title('Component Convergence Speed')
        ax4.set_xlabel('Training Iterations')
        ax4.set_ylabel('Convergence Rate')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_path = self.output_dir / 'loss_component_analysis.png'
        try:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', format='png')
            self.logger.info(f"Successfully saved: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save loss component analysis: {e}")
        finally:
            plt.close()
        
        self.logger.info(f"✅ 损失分量分析图已保存: {output_path}")
    
    def _create_effect_comparison_figure(self, data: Dict, reg_results: Dict):
        """创建效果对比图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Regularization Effect Comparison Analysis\nBefore/After Metrics & Performance Evaluation', 
                    fontsize=14, fontweight='bold')
        
        scales_before = data['scales_before']
        scales_after = reg_results['scales_after']
        
        # 计算各种指标
        ratios_before = scales_before[:, 0] / (scales_before[:, 2] + 1e-8)
        ratios_after = scales_after[:, 0] / (scales_after[:, 2] + 1e-8)
        
        # 子图1: 各向异性比例对比散点图
        ax1.scatter(ratios_before, ratios_after, c=self.color_palette[0], alpha=0.7, s=50)
        
        # 添加对角线（理想情况）
        min_ratio = min(np.min(ratios_before), np.min(ratios_after))
        max_ratio = max(np.max(ratios_before), np.max(ratios_after))
        ax1.plot([min_ratio, max_ratio], [min_ratio, max_ratio], 'k--', alpha=0.5, label='No Change Line')
        
        # 添加改善区域
        ax1.fill_between([10, max_ratio], 10, max_ratio, alpha=0.2, color='red', label='Over-stretch Region')
        ax1.axhline(y=10, color='green', linestyle='--', alpha=0.7, label='Ideal Upper Limit')
        ax1.axvline(x=10, color='green', linestyle='--', alpha=0.7)
        
        ax1.set_title('Anisotropy Ratio Comparison')
        ax1.set_xlabel('Before Regularization Ratio')
        ax1.set_ylabel('After Regularization Ratio')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 改善程度直方图
        improvement = ratios_before - ratios_after
        ax2.hist(improvement, bins=20, alpha=0.7, color=self.color_palette[1], edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', label='No Improvement Line')
        ax2.set_title('Anisotropy Ratio Improvement Distribution')
        ax2.set_xlabel('Improvement Degree (Positive = Better)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 子图3: 体积变化分析
        volumes_before = np.prod(scales_before, axis=1)
        volumes_after = np.prod(scales_after, axis=1)
        volume_ratio = volumes_after / (volumes_before + 1e-8)
        
        ax3.scatter(volumes_before, volumes_after, c=self.color_palette[2], alpha=0.7, s=50)
        ax3.plot([np.min(volumes_before), np.max(volumes_before)], 
                [np.min(volumes_before), np.max(volumes_before)], 'k--', alpha=0.5)
        ax3.set_title('Gaussian Volume Changes')
        ax3.set_xlabel('Before Regularization Volume')
        ax3.set_ylabel('After Regularization Volume')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 子图4: 稳定性指标
        # 计算形状稳定性指标
        stability_before = 1 / (1 + ratios_before)  # 越接近1越稳定
        stability_after = 1 / (1 + ratios_after)
        
        n_gaussians = len(stability_before)
        x = np.arange(n_gaussians)
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, stability_before, width, label='Before Regularization', 
                       color=self.color_palette[3], alpha=0.8)
        bars2 = ax4.bar(x + width/2, stability_after, width, label='After Regularization', 
                       color=self.color_palette[4], alpha=0.8)
        
        ax4.set_title('Shape Stability Comparison')
        ax4.set_xlabel('Gaussian Index')
        ax4.set_ylabel('Stability Index')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_path = self.output_dir / 'effect_comparison_analysis.png'
        try:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', format='png')
            self.logger.info(f"Successfully saved: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save effect comparison analysis: {e}")
        finally:
            plt.close()
        
        self.logger.info(f"✅ 效果对比分析图已保存: {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Principled Mixed Regularization Visualizer")
    parser.add_argument("--output_dir", type=str, default="./visualization_outputs/regularization_default",
                       help="Output directory path (will be created if not exists). Examples: './visualization_outputs/flower_analysis', './visualization_outputs/chair_comparison'")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Training model path (containing point_cloud.ply file)")
    parser.add_argument("--ply_path", type=str, default=None,
                       help="Direct PLY file path")
    parser.add_argument("--use_synthetic", action="store_true",
                       help="Force use synthetic data instead of real data")
    
    args = parser.parse_args()
    
    # 智能输出目录：如果使用默认值且提供了PLY路径，则根据PLY文件生成目录名
    output_dir = args.output_dir
    if args.output_dir == "./visualization_outputs/regularization_default" and args.ply_path:
        # 从PLY路径提取场景名称
        ply_path = Path(args.ply_path)
        # 尝试从路径中提取场景名（如flower, chair等）
        path_parts = ply_path.parts
        scene_name = None
        for part in path_parts:
            if part in ['flower', 'chair', 'lego', 'drums', 'ficus', 'hotdog', 'materials', 'mic', 'ship']:
                scene_name = part
                break
        
        if scene_name:
            output_dir = f"./visualization_outputs/regularization_{scene_name}"
            print(f"🎯 智能输出目录: {output_dir}")
        else:
            # 如果没有识别到场景名，尝试从迭代数提取信息
            iteration_match = None
            for part in path_parts:
                if 'iteration_' in part:
                    iteration_match = part
                    break
            if iteration_match:
                output_dir = f"./visualization_outputs/regularization_{iteration_match}"
                print(f"🎯 智能输出目录: {output_dir}")
    
    # 创建可视化器
    visualizer = PrincipledMixedRegularizationVisualizer(
        output_dir=output_dir,
        model_path=args.model_path
    )
    
    # 如果指定了PLY路径，直接加载
    real_data = None
    if args.ply_path:
        print(f"Loading PLY file: {args.ply_path}")
        real_data = visualizer.load_real_gaussian_data(args.ply_path)
        if real_data:
            print(f"Successfully loaded {real_data['n_gaussians']} Gaussian primitives")
        else:
            print("PLY file loading failed, will use synthetic data")
    
    # 生成可视化
    use_real = not args.use_synthetic
    visualizer.create_mixed_regularization_visualization(use_real_data=use_real, preloaded_data=real_data)
    
    print("Principled Mixed Regularization visualization completed!")
    print(f"Results saved in: {args.output_dir}")
    print("\nGenerated files:")
    print("  - principled_mixed_regularization.png: Main visualization")
    print("  - pca_analysis_detailed.png: Detailed PCA analysis")
    print("  - loss_component_analysis.png: Loss component analysis")
    print("  - effect_comparison_analysis.png: Effect comparison analysis")
    
    if args.model_path or args.ply_path:
        print("\nUsed real training data for visualization")
    else:
        print("\nUsed synthetic data for demonstration")


if __name__ == "__main__":
    main()