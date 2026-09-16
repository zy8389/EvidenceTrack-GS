"""
Deformable Sampling Module
可变形采样模块

Implements deformable sampling from 2D feature maps using predicted
offsets and attention weights for enhanced feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

from ..core.interfaces import DeformableSampler
from ..core.data_structures import GTDCAConfig


class OffsetPredictor(nn.Module):
    """
    偏移量预测器
    
    MLP network for predicting multiple sampling offsets relative to base coordinates.
    Ensures numerical stability through proper initialization and normalization.
    
    Requirements addressed: 2.1, 4.2 - Offset prediction with numerical stability
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int, num_sample_points: int, 
                 sampling_radius: float = 2.0, dropout_rate: float = 0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_sample_points = num_sample_points
        self.sampling_radius = sampling_radius
        
        # MLP layers for offset prediction
        self.offset_mlp = nn.Sequential(
            # First layer: feature_dim + 2 (coordinates) -> hidden_dim
            nn.Linear(feature_dim + 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Second layer: hidden_dim -> hidden_dim
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Output layer: hidden_dim -> num_sample_points * 2
            nn.Linear(hidden_dim, num_sample_points * 2)
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(feature_dim + 2)
        
        # Initialize weights for numerical stability
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for numerical stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for better gradient flow
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize final layer with smaller weights to prevent large offsets
        final_layer = self.offset_mlp[-1]
        nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
        nn.init.zeros_(final_layer.bias)
    
    def forward(self, guided_queries: Tensor, projection_coords: Tensor) -> Tensor:
        """
        Predict sampling offsets
        
        Args:
            guided_queries: Geometry-guided query vectors (N, feature_dim)
            projection_coords: 2D projection coordinates (N, 2)
            
        Returns:
            Sampling offsets (N, num_sample_points, 2)
        """
        batch_size = guided_queries.shape[0]
        
        # Concatenate features and coordinates
        input_features = torch.cat([guided_queries, projection_coords], dim=-1)  # (N, feature_dim + 2)
        
        # Apply layer normalization for stability
        input_features = self.layer_norm(input_features)
        
        # Predict raw offsets
        raw_offsets = self.offset_mlp(input_features)  # (N, num_sample_points * 2)
        
        # Reshape to (N, num_sample_points, 2)
        offsets = raw_offsets.view(batch_size, self.num_sample_points, 2)
        
        # Apply tanh activation and scale by sampling radius for bounded offsets
        # This ensures numerical stability and prevents extreme offset values
        offsets = torch.tanh(offsets) * self.sampling_radius
        
        return offsets


class WeightPredictor(nn.Module):
    """
    权重预测器
    
    MLP network for generating attention weights for each sampling position.
    Includes softmax normalization to ensure weights sum to 1.
    
    Requirements addressed: 2.2, 4.2 - Weight prediction with softmax normalization
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int, num_sample_points: int, 
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_sample_points = num_sample_points
        
        # MLP layers for weight prediction
        self.weight_mlp = nn.Sequential(
            # First layer: feature_dim + 2 (coordinates) -> hidden_dim
            nn.Linear(feature_dim + 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Second layer: hidden_dim -> hidden_dim
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Output layer: hidden_dim -> num_sample_points
            nn.Linear(hidden_dim, num_sample_points)
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(feature_dim + 2)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize final layer with small positive bias for balanced attention
        final_layer = self.weight_mlp[-1]
        nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
        nn.init.constant_(final_layer.bias, 0.1)  # Small positive bias
    
    def forward(self, guided_queries: Tensor, projection_coords: Tensor) -> Tensor:
        """
        Predict sampling weights
        
        Args:
            guided_queries: Geometry-guided query vectors (N, feature_dim)
            projection_coords: 2D projection coordinates (N, 2)
            
        Returns:
            Normalized sampling weights (N, num_sample_points)
        """
        # Concatenate features and coordinates
        input_features = torch.cat([guided_queries, projection_coords], dim=-1)  # (N, feature_dim + 2)
        
        # Apply layer normalization for stability
        input_features = self.layer_norm(input_features)
        
        # Predict raw weights
        raw_weights = self.weight_mlp(input_features)  # (N, num_sample_points)
        
        # Apply softmax normalization to ensure weights sum to 1
        # Use temperature scaling for more stable gradients
        temperature = 1.0
        normalized_weights = F.softmax(raw_weights / temperature, dim=-1)
        
        return normalized_weights


class FeatureSampler:
    """
    特征图采样器
    
    Handles safe sampling from 2D feature maps with boundary handling
    and weighted feature aggregation.
    
    Requirements addressed: 2.3, 6.2 - Feature map sampling with boundary handling
    """
    
    @staticmethod
    def safe_sample_feature_map(
        feature_map_2d: Tensor, 
        sampling_coords: Tensor,
        mode: str = 'bilinear',
        padding_mode: str = 'border'
    ) -> Tensor:
        """
        Safe sampling from 2D feature map with boundary handling
        
        Args:
            feature_map_2d: 2D feature map (H, W, C) or (1, C, H, W)
            sampling_coords: Sampling coordinates (N, num_points, 2)
            mode: Interpolation mode ('bilinear' or 'nearest')
            padding_mode: Padding mode ('zeros', 'border', 'reflection')
            
        Returns:
            Sampled features (N, num_points, C)
        """
        # Ensure feature map is in correct format (1, C, H, W)
        if feature_map_2d.dim() == 3:  # (H, W, C)
            feature_map_2d = feature_map_2d.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        elif feature_map_2d.dim() == 4 and feature_map_2d.shape[0] != 1:
            # If batch dimension > 1, take first sample
            feature_map_2d = feature_map_2d[:1]
        
        batch_size, num_points = sampling_coords.shape[:2]
        h, w = feature_map_2d.shape[2], feature_map_2d.shape[3]
        
        # Normalize coordinates to [-1, 1] range for grid_sample
        # sampling_coords is in pixel coordinates, need to convert to normalized coordinates
        normalized_coords = sampling_coords.clone()
        normalized_coords[..., 0] = 2.0 * sampling_coords[..., 0] / (w - 1) - 1.0  # x coordinate
        normalized_coords[..., 1] = 2.0 * sampling_coords[..., 1] / (h - 1) - 1.0  # y coordinate
        
        # Clamp coordinates to valid range to prevent extreme values
        normalized_coords = torch.clamp(normalized_coords, -1.1, 1.1)
        
        # Reshape for grid_sample: (N, num_points, 2) -> (N, 1, num_points, 2)
        grid = normalized_coords.unsqueeze(1)  # (N, 1, num_points, 2)
        
        # Expand feature map for batch sampling if needed
        if batch_size > 1:
            feature_map_2d = feature_map_2d.expand(batch_size, -1, -1, -1)
        
        # Sample features using grid_sample
        # Output shape: (N, C, 1, num_points)
        sampled_features = F.grid_sample(
            feature_map_2d, 
            grid, 
            mode=mode, 
            padding_mode=padding_mode, 
            align_corners=True
        )
        
        # Reshape to (N, num_points, C)
        sampled_features = sampled_features.squeeze(2).transpose(1, 2)
        
        return sampled_features
    
    @staticmethod
    def weighted_feature_aggregation(
        sampled_features: Tensor, 
        sampling_weights: Tensor
    ) -> Tensor:
        """
        Weighted aggregation of sampled features
        
        Args:
            sampled_features: Sampled features (N, num_points, C)
            sampling_weights: Sampling weights (N, num_points)
            
        Returns:
            Aggregated features (N, C)
        """
        # Ensure weights are normalized
        weights_sum = sampling_weights.sum(dim=-1, keepdim=True)
        weights_sum = torch.clamp(weights_sum, min=1e-8)  # Prevent division by zero
        normalized_weights = sampling_weights / weights_sum
        
        # Weighted sum: (N, num_points, 1) * (N, num_points, C) -> (N, C)
        weighted_features = torch.sum(
            normalized_weights.unsqueeze(-1) * sampled_features, 
            dim=1
        )
        
        return weighted_features
    
    @staticmethod
    def compute_sampling_coordinates(
        projection_coords: Tensor, 
        sampling_offsets: Tensor
    ) -> Tensor:
        """
        Compute final sampling coordinates from base coordinates and offsets
        
        Args:
            projection_coords: Base projection coordinates (N, 2)
            sampling_offsets: Predicted offsets (N, num_points, 2)
            
        Returns:
            Final sampling coordinates (N, num_points, 2)
        """
        # Expand base coordinates to match offset dimensions
        base_coords_expanded = projection_coords.unsqueeze(1)  # (N, 1, 2)
        
        # Add offsets to base coordinates
        sampling_coords = base_coords_expanded + sampling_offsets  # (N, num_points, 2)
        
        return sampling_coords


class DeformableSamplingModule(nn.Module, DeformableSampler):
    """
    可变形采样模块实现
    
    Predicts sampling offsets and weights for deformable feature sampling
    from 2D feature maps, enabling adaptive feature extraction.
    
    Requirements addressed: 2.1, 2.2, 2.3 - Deformable sampling with offset and weight prediction
    """
    
    def __init__(self, config: GTDCAConfig):
        super().__init__()
        
        self.config = config
        self.feature_dim = config.feature_dim
        self.hidden_dim = config.hidden_dim
        self._num_sample_points = config.num_sample_points
        
        # Task 4.1: Create offset predictor
        self.offset_predictor = OffsetPredictor(
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            num_sample_points=self._num_sample_points,
            sampling_radius=config.sampling_radius,
            dropout_rate=config.dropout_rate
        )
        
        # Task 4.2: Create weight predictor
        self.weight_predictor = WeightPredictor(
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            num_sample_points=self._num_sample_points,
            dropout_rate=config.dropout_rate
        )
        
    def predict_sampling_offsets(
        self, 
        guided_queries: Tensor, 
        projection_coords: Tensor
    ) -> Tensor:
        """
        预测采样偏移量
        
        Args:
            guided_queries: 几何引导后的查询向量 (N, feature_dim)
            projection_coords: 高斯点2D投影坐标 (N, 2)
            
        Returns:
            采样偏移量 (N, num_sample_points, 2)
        """
        # Task 4.1: Use offset predictor to predict sampling offsets
        return self.offset_predictor(guided_queries, projection_coords)
    
    def predict_sampling_weights(
        self, 
        guided_queries: Tensor, 
        projection_coords: Tensor
    ) -> Tensor:
        """
        预测采样权重
        
        Args:
            guided_queries: 几何引导后的查询向量 (N, feature_dim)
            projection_coords: 高斯点2D投影坐标 (N, 2)
            
        Returns:
            采样权重 (N, num_sample_points)，已归一化
        """
        # Task 4.2: Use weight predictor to predict normalized sampling weights
        return self.weight_predictor(guided_queries, projection_coords)
    
    def sample_features(
        self, 
        feature_map_2d: Tensor, 
        sampling_coords: Tensor, 
        sampling_weights: Tensor
    ) -> Tensor:
        """
        从2D特征图采样特征
        
        Args:
            feature_map_2d: 2D图像特征图 (H, W, C)
            sampling_coords: 采样坐标 (N, num_sample_points, 2)
            sampling_weights: 采样权重 (N, num_sample_points)
            
        Returns:
            加权采样特征 (N, C)
        """
        # Task 4.3: Implement feature map sampling with boundary handling
        
        # Step 1: Safe sampling from 2D feature map
        sampled_features = FeatureSampler.safe_sample_feature_map(
            feature_map_2d=feature_map_2d,
            sampling_coords=sampling_coords,
            mode='bilinear',
            padding_mode='border'
        )  # (N, num_sample_points, C)
        
        # Step 2: Weighted feature aggregation
        aggregated_features = FeatureSampler.weighted_feature_aggregation(
            sampled_features=sampled_features,
            sampling_weights=sampling_weights
        )  # (N, C)
        
        return aggregated_features
    
    def forward(
        self, 
        guided_queries: Tensor, 
        feature_map_2d: Tensor, 
        projection_coords: Tensor
    ) -> Tensor:
        """
        Complete forward pass for deformable sampling module
        
        Integrates offset prediction, weight prediction, and feature sampling
        to perform the complete deformable sampling operation.
        
        Args:
            guided_queries: Geometry-guided query vectors (N, feature_dim)
            feature_map_2d: 2D image feature map (H, W, C)
            projection_coords: Gaussian 2D projection coordinates (N, 2)
            
        Returns:
            Enhanced appearance features (N, C)
        """
        # Task 4.4: Complete deformable sampling module integration
        
        # Step 1: Predict sampling offsets
        sampling_offsets = self.predict_sampling_offsets(
            guided_queries, projection_coords
        )  # (N, num_sample_points, 2)
        
        # Step 2: Predict sampling weights
        sampling_weights = self.predict_sampling_weights(
            guided_queries, projection_coords
        )  # (N, num_sample_points)
        
        # Step 3: Compute final sampling coordinates
        sampling_coords = FeatureSampler.compute_sampling_coordinates(
            projection_coords, sampling_offsets
        )  # (N, num_sample_points, 2)
        
        # Step 4: Sample features from 2D feature map
        enhanced_features = self.sample_features(
            feature_map_2d, sampling_coords, sampling_weights
        )  # (N, C)
        
        return enhanced_features
    
    def get_sampling_metadata(
        self, 
        guided_queries: Tensor, 
        projection_coords: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Get detailed sampling metadata for analysis and debugging
        
        Args:
            guided_queries: Geometry-guided query vectors (N, feature_dim)
            projection_coords: Gaussian 2D projection coordinates (N, 2)
            
        Returns:
            Tuple of (sampling_offsets, sampling_weights, sampling_coords)
        """
        # Predict offsets and weights
        sampling_offsets = self.predict_sampling_offsets(guided_queries, projection_coords)
        sampling_weights = self.predict_sampling_weights(guided_queries, projection_coords)
        
        # Compute final coordinates
        sampling_coords = FeatureSampler.compute_sampling_coordinates(
            projection_coords, sampling_offsets
        )
        
        return sampling_offsets, sampling_weights, sampling_coords
    
    @property
    def num_sample_points(self) -> int:
        """返回采样点数量"""
        return self._num_sample_points