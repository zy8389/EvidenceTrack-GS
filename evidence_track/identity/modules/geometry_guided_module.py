"""
Geometry Guided Module
几何引导模块

Implements geometry-guided processing using cross-attention mechanism
to inject geometric context into appearance feature queries.
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.interfaces import GeometryGuidedProcessor
from ..core.data_structures import TrackPoint, GTDCAConfig


class MLPProjectionNetwork(nn.Module):
    """
    MLP投射网络
    
    Projects 2D trajectory point coordinates to feature space.
    Optimized structure for computational efficiency.
    
    Requirements addressed: 1.2, 5.1 - Efficient MLP projection of 2D coordinates
    """
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, output_dim: int = 256, dropout_rate: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Efficient MLP structure with residual connections for better gradient flow
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Layer normalization for training stability
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Initialize weights for better convergence
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, coordinates_2d: Tensor) -> Tensor:
        """
        Project 2D coordinates to feature space
        
        Args:
            coordinates_2d: 2D trajectory point coordinates (M, 2)
            
        Returns:
            Projected features (M, output_dim)
        """
        if coordinates_2d.numel() == 0:
            return torch.empty(0, self.output_dim, device=coordinates_2d.device)
        
        # Ensure input has correct shape
        if coordinates_2d.dim() == 1:
            coordinates_2d = coordinates_2d.unsqueeze(0)
        
        # Project coordinates through MLP
        features = self.layers(coordinates_2d)
        
        # Apply layer normalization
        features = self.layer_norm(features)
        
        return features


class CrossAttentionModule(nn.Module):
    """
    交叉注意力模块
    
    Implements cross-attention mechanism for fusing geometric context
    with appearance feature queries.
    
    Requirements addressed: 1.3, 4.1 - Cross-attention for geometric information fusion
    """
    
    def __init__(self, query_dim: int, key_dim: int, value_dim: int, 
                 num_heads: int = 8, dropout_rate: float = 0.1, use_layer_norm: bool = True):
        super().__init__()
        
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.query_projection = nn.Linear(query_dim, query_dim)
        self.key_projection = nn.Linear(key_dim, query_dim)
        self.value_projection = nn.Linear(value_dim, query_dim)
        
        # Output projection
        self.output_projection = nn.Linear(query_dim, query_dim)
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout_rate)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(query_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize attention weights using Xavier initialization"""
        for module in [self.query_projection, self.key_projection, 
                      self.value_projection, self.output_projection]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, 
                attention_mask: Tensor = None) -> Tensor:
        """
        Cross-attention forward pass
        
        Args:
            queries: Query features (N, query_dim) - appearance features
            keys: Key features (M, key_dim) - geometric context
            values: Value features (M, value_dim) - geometric context
            attention_mask: Optional attention mask (N, M)
            
        Returns:
            Attended features (N, query_dim)
        """
        batch_size_q, seq_len_q = queries.shape[0], queries.shape[0]
        batch_size_k, seq_len_k = keys.shape[0], keys.shape[0]
        
        # Handle empty geometric context
        if keys.numel() == 0 or values.numel() == 0:
            return queries.clone()
        
        # Project to Q, K, V
        Q = self.query_projection(queries)  # (N, query_dim)
        K = self.key_projection(keys)       # (M, query_dim)
        V = self.value_projection(values)   # (M, query_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(seq_len_q, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, N, head_dim)
        K = K.view(seq_len_k, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, M, head_dim)
        V = V.view(seq_len_k, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, M, head_dim)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (num_heads, N, M)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                attention_mask.unsqueeze(0) == 0, float('-inf')
            )
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (num_heads, N, M)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)  # (num_heads, N, head_dim)
        
        # Concatenate heads
        attended_values = attended_values.transpose(0, 1).contiguous().view(
            seq_len_q, self.query_dim
        )  # (N, query_dim)
        
        # Final output projection
        output = self.output_projection(attended_values)
        
        # Residual connection and layer normalization
        output = output + queries  # Residual connection
        if self.use_layer_norm:
            output = self.layer_norm(output)
        
        return output


class GeometryGuidedModule(nn.Module, GeometryGuidedProcessor):
    """
    几何引导模块实现
    
    Uses cross-attention mechanism to inject geometric context from 2D trajectory
    points into the base appearance feature queries.
    
    Requirements addressed: 1.2, 1.3 - Geometric guidance using 2D trajectory points
    """
    
    def __init__(self, config: GTDCAConfig):
        super().__init__()
        
        self.config = config
        self.feature_dim = config.feature_dim
        self.hidden_dim = config.hidden_dim
        
        # Task 3.1: MLP projection network for 2D coordinates
        self.projection_mlp = MLPProjectionNetwork(
            input_dim=2,
            hidden_dim=self.hidden_dim,
            output_dim=self.feature_dim,
            dropout_rate=config.dropout_rate
        )
        
        # Task 3.2: Cross attention mechanism
        self.cross_attention = CrossAttentionModule(
            query_dim=self.feature_dim,
            key_dim=self.feature_dim,
            value_dim=self.feature_dim,
            num_heads=config.attention_heads,
            dropout_rate=config.dropout_rate,
            use_layer_norm=config.use_layer_norm
        )
        
        # Task 3.3: Integration complete - all components integrated
        
    def process_geometry_guidance(
        self, 
        query_features: Tensor, 
        track_points_2d: List[TrackPoint]
    ) -> Tensor:
        """
        通过几何引导处理查询特征
        
        Args:
            query_features: 基础查询特征 (N, feature_dim)
            track_points_2d: 2D特征轨迹点列表
            
        Returns:
            几何引导后的特征 (N, feature_dim)
        """
        # Task 3.3: Complete geometry guidance integration
        
        # Step 1: Extract geometric context using MLP projection
        geometric_context = self.extract_geometric_context(track_points_2d)
        
        # Step 2: Handle case where no valid geometric context is available
        if geometric_context.numel() == 0:
            # Return original features if no geometric guidance available
            return query_features.clone()
        
        # Step 3: Apply cross-attention to fuse geometric context with queries
        # Queries: appearance features, Keys/Values: geometric context
        guided_features = self.cross_attention(
            queries=query_features,
            keys=geometric_context,
            values=geometric_context
        )
        
        return guided_features
    
    def forward(self, query_features: Tensor, track_points_2d: List[TrackPoint]) -> Tensor:
        """
        Complete forward pass for geometry guided module
        
        Args:
            query_features: Base appearance feature queries (N, feature_dim)
            track_points_2d: List of 2D trajectory points for geometric guidance
            
        Returns:
            Geometry-guided features (N, feature_dim)
        """
        return self.process_geometry_guidance(query_features, track_points_2d)
    
    def extract_geometric_context(self, track_points_2d: List[TrackPoint]) -> Tensor:
        """
        从2D轨迹点提取几何上下文
        
        Args:
            track_points_2d: 2D特征轨迹点列表
            
        Returns:
            几何上下文特征 (M, context_dim)
        """
        if not track_points_2d:
            # Get device from the first parameter of this module
            device = next(self.parameters()).device
            return torch.empty(0, self.feature_dim, device=device)
        
        # Filter valid track points based on confidence threshold
        valid_points = [
            point for point in track_points_2d 
            if point.is_valid() and point.confidence >= self.config.confidence_threshold
        ]
        
        if not valid_points:
            # Get device from the first parameter of this module
            device = next(self.parameters()).device
            return torch.empty(0, self.feature_dim, device=device)
        
        # Get device from the first parameter of this module
        device = next(self.parameters()).device
        
        # Convert track points to coordinate tensor with correct device
        coordinates = torch.stack([point.to_tensor(device) for point in valid_points])
        
        # Project 2D coordinates to feature space using MLP
        geometric_context = self.projection_mlp(coordinates)
        
        return geometric_context