"""
Geometry-based Anisotropic Gaussian Regularization

This module implements structure-aware anisotropic regularization for 3D Gaussian Splatting.
The method follows a two-step process:
1. Local Geometry Perception: Use PCA on K-nearest neighbors to understand local geometric structure
2. Anisotropic Constraint: Regularize Gaussian shape to align with local geometry
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

try:
    from simple_knn._C import distCUDA2
except ImportError:
    print("⚠️ simple_knn module not found, using fallback KNN implementation")
    distCUDA2 = None


class GeometryRegularizer:
    """
    Geometry-based regularization for 3D Gaussian Splatting
    """
    
    def __init__(self, 
                 k_neighbors: int = 16,
                 reg_weight: float = 0.01,
                 enable_threshold: int = 5000,
                 min_eigenvalue_ratio: float = 0.1):
        """
        Args:
            k_neighbors: Number of nearest neighbors for PCA analysis
            reg_weight: Weight for regularization loss
            enable_threshold: Training iteration to enable regularization
            min_eigenvalue_ratio: Minimum ratio between smallest and largest eigenvalue
        """
        self.k_neighbors = k_neighbors
        self.reg_weight = reg_weight
        self.enable_threshold = enable_threshold
        self.min_eigenvalue_ratio = min_eigenvalue_ratio
        
    def find_k_nearest_neighbors(self, xyz: torch.Tensor, k: int) -> torch.Tensor:
        """
        Find K nearest neighbors for each point
        
        Args:
            xyz: Point positions (N, 3)
            k: Number of neighbors
            
        Returns:
            neighbor_indices: Indices of K nearest neighbors (N, k)
        """
        if distCUDA2 is not None:
            # Use fast CUDA KNN if available
            _, neighbor_indices = distCUDA2(xyz.float())
            return neighbor_indices[:, :k]
        else:
            # Fallback to PyTorch implementation
            N = xyz.shape[0]
            # Compute pairwise distances
            distances = torch.cdist(xyz, xyz, p=2)  # (N, N)
            # Find k+1 nearest (including self) and exclude self
            _, indices = torch.topk(distances, k+1, largest=False, dim=1)
            return indices[:, 1:k+1]  # Exclude self (first column)
    
    def compute_local_pca(self, xyz: torch.Tensor, k: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute PCA for each point's local neighborhood
        
        Args:
            xyz: Point positions (N, 3)
            k: Number of neighbors (default: self.k_neighbors)
            
        Returns:
            principal_directions: Principal directions (N, 3, 3) - each row is [v1, v2, v3]
            eigenvalues: Eigenvalues (N, 3) - sorted in descending order
        """
        if k is None:
            k = self.k_neighbors
            
        N = xyz.shape[0]
        device = xyz.device
        
        # Find K nearest neighbors
        neighbor_indices = self.find_k_nearest_neighbors(xyz, k)  # (N, k)
        
        # Gather neighbor positions
        neighbors = xyz[neighbor_indices]  # (N, k, 3)
        
        # Center the neighborhoods
        centroids = neighbors.mean(dim=1, keepdim=True)  # (N, 1, 3)
        centered_neighbors = neighbors - centroids  # (N, k, 3)
        
        # Compute covariance matrices for each neighborhood
        # cov = (1/(k-1)) * X^T X where X is centered
        covariance_matrices = torch.bmm(centered_neighbors.transpose(1, 2), 
                                      centered_neighbors) / (k - 1)  # (N, 3, 3)
        
        # Compute eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrices)
        
        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = torch.argsort(eigenvalues, dim=1, descending=True)
        eigenvalues = torch.gather(eigenvalues, 1, sorted_indices)
        eigenvectors = torch.gather(eigenvectors, 2, sorted_indices.unsqueeze(1).expand(-1, 3, -1))
        
        # Ensure minimum eigenvalue ratio for numerical stability
        min_eigenval = eigenvalues[:, 0:1] * self.min_eigenvalue_ratio
        eigenvalues = torch.max(eigenvalues, min_eigenval)
        
        return eigenvectors, eigenvalues
    
    def gaussian_to_rotation_matrix(self, rotation_quaternion: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternion to rotation matrix
        
        Args:
            rotation_quaternion: Quaternion (N, 4) - normalized
            
        Returns:
            rotation_matrix: Rotation matrices (N, 3, 3)
        """
        q = rotation_quaternion
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # Compute rotation matrix elements
        rotation_matrix = torch.stack([
            torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], dim=1),
            torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], dim=1),
            torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], dim=1)
        ], dim=1)  # (N, 3, 3)
        
        return rotation_matrix
    
    def compute_anisotropic_regularization_loss(self,
                                              xyz: torch.Tensor,
                                              scaling: torch.Tensor,
                                              rotation: torch.Tensor,
                                              iteration: int) -> torch.Tensor:
        """
        Compute geometry-based anisotropic regularization loss
        
        Args:
            xyz: Gaussian positions (N, 3)
            scaling: Gaussian scales (N, 3) - already activated (exp applied)
            rotation: Gaussian rotations (N, 4) - normalized quaternions
            iteration: Current training iteration
            
        Returns:
            regularization_loss: Scalar loss value
        """
        # Only apply regularization after certain iterations
        if iteration < self.enable_threshold:
            return torch.tensor(0.0, device=xyz.device, requires_grad=True)
        
        # Compute local geometry via PCA
        principal_directions, eigenvalues = self.compute_local_pca(xyz)  # (N, 3, 3), (N, 3)
        
        # Convert Gaussian rotation to rotation matrix
        gaussian_rotation_matrices = self.gaussian_to_rotation_matrix(rotation)  # (N, 3, 3)
        
        # Compute Gaussian principal axes (scaled and rotated unit vectors)
        # The Gaussian ellipsoid axes are the columns of rotation matrix scaled by scaling factors
        unit_axes = torch.eye(3, device=xyz.device).unsqueeze(0).expand(xyz.shape[0], -1, -1)  # (N, 3, 3)
        gaussian_axes = torch.bmm(gaussian_rotation_matrices, unit_axes)  # (N, 3, 3)
        
        # Scale the axes
        gaussian_axes_scaled = gaussian_axes * scaling.unsqueeze(1)  # (N, 3, 3)
        
        # Regularization 1: Align Gaussian's main axis with local geometry's main axis
        # The main axis of Gaussian should align with the direction of largest variance
        gaussian_main_axis = gaussian_axes_scaled[:, :, 0]  # (N, 3) - first column
        geometry_main_axis = principal_directions[:, :, 0]  # (N, 3) - first principal direction
        
        # Compute alignment loss (1 - |cosine similarity|)
        cos_similarity = F.cosine_similarity(gaussian_main_axis, geometry_main_axis, dim=1)
        alignment_loss = (1.0 - torch.abs(cos_similarity)).mean()
        
        # Regularization 2: Scale ratio should match eigenvalue ratio
        # Prevent over-flattened or over-stretched Gaussians
        gaussian_scale_ratios = scaling[:, 0] / (scaling[:, 2] + 1e-8)  # max_scale / min_scale
        geometry_scale_ratios = eigenvalues[:, 0] / (eigenvalues[:, 2] + 1e-8)  # max_eigenval / min_eigenval
        
        # Use log space to handle large ratios
        log_gaussian_ratios = torch.log(gaussian_scale_ratios + 1e-8)
        log_geometry_ratios = torch.log(geometry_scale_ratios + 1e-8)
        ratio_loss = F.mse_loss(log_gaussian_ratios, log_geometry_ratios)
        
        # Regularization 3: Prevent excessive anisotropy
        # This helps maintain stability during training
        max_anisotropy_ratio = 10.0
        excessive_anisotropy = torch.clamp(gaussian_scale_ratios - max_anisotropy_ratio, min=0.0)
        anisotropy_penalty = excessive_anisotropy.mean()
        
        # Combine all losses
        total_loss = alignment_loss + 0.5 * ratio_loss + 0.1 * anisotropy_penalty
        
        return total_loss * self.reg_weight
    
    def adaptive_regularization_weight(self, iteration: int, base_weight: float = None) -> float:
        """
        Compute adaptive regularization weight based on training progress
        
        Args:
            iteration: Current training iteration
            base_weight: Base weight (default: self.reg_weight)
            
        Returns:
            adaptive_weight: Adapted weight value
        """
        if base_weight is None:
            base_weight = self.reg_weight
            
        if iteration < self.enable_threshold:
            return 0.0
        
        # Gradually increase weight in early stages, then stabilize
        warmup_iterations = 2000
        if iteration < self.enable_threshold + warmup_iterations:
            progress = (iteration - self.enable_threshold) / warmup_iterations
            return base_weight * progress
        else:
            return base_weight


def create_geometry_regularizer(args) -> GeometryRegularizer:
    """
    Factory function to create geometry regularizer from arguments
    
    Args:
        args: Arguments object containing regularization parameters
        
    Returns:
        GeometryRegularizer instance
    """
    k_neighbors = getattr(args, 'geometry_reg_k_neighbors', 16)
    reg_weight = getattr(args, 'geometry_reg_weight', 0.01)
    enable_threshold = getattr(args, 'geometry_reg_enable_threshold', 5000)
    min_eigenvalue_ratio = getattr(args, 'geometry_reg_min_eigenvalue_ratio', 0.1)
    
    return GeometryRegularizer(
        k_neighbors=k_neighbors,
        reg_weight=reg_weight,
        enable_threshold=enable_threshold,
        min_eigenvalue_ratio=min_eigenvalue_ratio
    )