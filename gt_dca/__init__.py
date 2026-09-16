"""
GT-DCA (Geometry-guided Deformable Cross-Attention) Module
几何引导可变形交叉注意力外观建模模块

This module provides enhanced appearance modeling for 3D Gaussian Splatting
by leveraging geometric trajectory information and deformable sampling.
"""

from .core.interfaces import (
    AppearanceFeatureGenerator,
    GeometryGuidedProcessor,
    DeformableSampler
)
from .core.data_structures import (
    TrackPoint,
    SamplingPoint,
    AppearanceFeature
)
from .modules.gt_dca_module import GTDCAModule
from .modules.base_appearance_generator import BaseAppearanceFeatureGenerator
from .modules.geometry_guided_module import GeometryGuidedModule
from .modules.deformable_sampling_module import DeformableSamplingModule

__version__ = "0.1.0"
__author__ = "GT-DCA Development Team"

__all__ = [
    # Core interfaces
    "AppearanceFeatureGenerator",
    "GeometryGuidedProcessor", 
    "DeformableSampler",
    # Data structures
    "TrackPoint",
    "SamplingPoint",
    "AppearanceFeature",
    # Main modules
    "GTDCAModule",
    "BaseAppearanceFeatureGenerator",
    "GeometryGuidedModule",
    "DeformableSamplingModule",
]