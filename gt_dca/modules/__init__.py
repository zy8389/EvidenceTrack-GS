"""
GT-DCA Modules
模块实现

Contains the concrete implementations of GT-DCA components.
"""

from .gt_dca_module import GTDCAModule
from .base_appearance_generator import BaseAppearanceFeatureGenerator
from .geometry_guided_module import GeometryGuidedModule
from .deformable_sampling_module import DeformableSamplingModule

__all__ = [
    "GTDCAModule",
    "BaseAppearanceFeatureGenerator", 
    "GeometryGuidedModule",
    "DeformableSamplingModule",
]