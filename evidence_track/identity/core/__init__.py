"""
GT-DCA Core Components
核心组件模块

Contains core interfaces, data structures, and utilities for the GT-DCA system.
"""

from .interfaces import (
    AppearanceFeatureGenerator,
    GeometryGuidedProcessor,
    DeformableSampler
)
from .data_structures import (
    TrackPoint,
    SamplingPoint,
    AppearanceFeature
)

__all__ = [
    "AppearanceFeatureGenerator",
    "GeometryGuidedProcessor",
    "DeformableSampler",
    "TrackPoint", 
    "SamplingPoint",
    "AppearanceFeature",
]