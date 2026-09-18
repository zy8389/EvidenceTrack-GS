"""
GT-DCA Integration Components
集成组件

Contains integration utilities and extensions for connecting GT-DCA
with existing 3D Gaussian Splatting systems.
"""

from .gaussian_model_extension import GaussianModelGTDCAExtension
from .fallback_handler import FallbackHandler

__all__ = [
    "GaussianModelGTDCAExtension",
    "FallbackHandler",
]