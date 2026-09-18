"""
Base Appearance Feature Generator
基础外观特征生成器

Generates base appearance feature vectors for 3D Gaussian primitives.
These serve as query vectors for the geometry-guided processing stage.
"""

import torch
import torch.nn as nn
from torch import Tensor

from ..core.interfaces import AppearanceFeatureGenerator
from ..core.data_structures import GTDCAConfig


class BaseAppearanceFeatureGenerator(nn.Module, AppearanceFeatureGenerator):
    """
    基础外观特征生成器实现
    
    Generates learnable base appearance feature vectors from 3D Gaussian primitives.
    These features serve as the initial query vectors for subsequent processing.
    
    Requirements addressed: 1.1 - Generate learnable feature vectors for 3D Gaussian primitives
    """
    
    def __init__(self, config: GTDCAConfig):
        super().__init__()
        
        self.config = config
        self._feature_dim = config.feature_dim
        
        # 3D高斯基元通常包含：位置(3) + 旋转(4) + 缩放(3) + 不透明度(1) + SH系数(N*3)
        # 为了通用性，我们使用一个可学习的投影层来处理任意维度的输入
        self.feature_projection = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate) if config.dropout_rate > 0 else nn.Identity(),
            nn.Linear(config.hidden_dim, config.feature_dim),
        )
        
        # 可学习的基础特征嵌入，为每个高斯基元提供初始查询向量
        self.base_embedding = nn.Parameter(
            torch.randn(1, config.feature_dim) * 0.02
        )
        
        # 位置编码层，用于编码3D位置信息
        self.position_encoder = nn.Sequential(
            nn.Linear(3, config.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_dim // 2, config.feature_dim // 2)
        )
        
        # 几何属性编码层，用于编码旋转、缩放等几何属性
        self.geometry_encoder = nn.Sequential(
            nn.Linear(8, config.hidden_dim // 2),  # 旋转(4) + 缩放(3) + 不透明度(1)
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_dim // 2, config.feature_dim // 2)
        )
        
        # 层归一化
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.feature_dim)
        else:
            self.layer_norm = nn.Identity()
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def _extract_gaussian_components(self, gaussian_primitives: Tensor) -> tuple:
        """
        从高斯基元参数中提取不同组件
        
        Args:
            gaussian_primitives: 3D高斯基元参数 (N, primitive_dim)
            
        Returns:
            (positions, rotations, scales, opacity, sh_coeffs)
        """
        # 假设高斯基元参数的标准格式：
        # 位置(3) + 旋转四元数(4) + 缩放(3) + 不透明度(1) + SH系数(剩余)
        
        positions = gaussian_primitives[:, :3]  # (N, 3)
        rotations = gaussian_primitives[:, 3:7]  # (N, 4) 四元数
        scales = gaussian_primitives[:, 7:10]  # (N, 3)
        opacity = gaussian_primitives[:, 10:11]  # (N, 1)
        
        # SH系数（如果存在）
        if gaussian_primitives.shape[1] > 11:
            sh_coeffs = gaussian_primitives[:, 11:]  # (N, remaining)
        else:
            sh_coeffs = None
            
        return positions, rotations, scales, opacity, sh_coeffs
        
    def generate_query_features(self, gaussian_primitives: Tensor) -> Tensor:
        """
        为3D高斯基元生成可学习的基础外观特征向量
        
        Args:
            gaussian_primitives: 3D高斯基元参数 (N, primitive_dim)
            
        Returns:
            基础外观特征向量 (N, feature_dim)
        """
        batch_size = gaussian_primitives.shape[0]
        device = gaussian_primitives.device
        
        # 提取高斯基元的不同组件
        positions, rotations, scales, opacity, sh_coeffs = self._extract_gaussian_components(
            gaussian_primitives
        )
        
        # 1. 基础嵌入向量（广播到所有高斯点）
        base_features = self.base_embedding.to(device).expand(batch_size, -1)
        
        # 2. 位置编码
        position_features = self.position_encoder(positions)  # (N, feature_dim//2)
        
        # 3. 几何属性编码（旋转 + 缩放 + 不透明度）
        geometry_attrs = torch.cat([rotations, scales, opacity], dim=1)  # (N, 8)
        geometry_features = self.geometry_encoder(geometry_attrs)  # (N, feature_dim//2)
        
        # 4. 组合位置和几何特征
        spatial_features = torch.cat([position_features, geometry_features], dim=1)  # (N, feature_dim)
        
        # 5. 与基础特征相加（残差连接）
        combined_features = base_features + spatial_features
        
        # 6. 通过特征投影层进一步处理
        enhanced_features = self.feature_projection(combined_features)
        
        # 7. 层归一化
        final_features = self.layer_norm(enhanced_features)
        
        return final_features
    
    @property
    def feature_dim(self) -> int:
        """返回特征向量维度"""
        return self._feature_dim