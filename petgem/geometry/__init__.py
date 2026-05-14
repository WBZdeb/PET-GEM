"""
Detector geometry models for geometric efficiency calculations.

This package provides detector geometry classes that implement
geometry-specific numerical integration methods used to compute
geometric detection efficiency for different emission models.

Classes:
    CylindricalDetector: Idealized cylindrical detector geometry.
    DualHeadDetector: Idealized dual-head planar detector geometry.
"""

from .cylindrical import CylindricalDetector
from .dualhead import DualHeadDetector

__all__ = [
    "CylindricalDetector",
    "DualHeadDetector",
]