# PETGEM - PET Geometric Efficiency Modelling
"""
Tools for calculating detector geometric efficiency.

This package provides detector geometry models and source emission
models for evaluating geometric detection efficiency.

Example:
    >>> from lib import CylindricalDetector, SingleGamma
    >>> detector = CylindricalDetector(inner_radius=20.0, afov=20.0)
    >>> emission = SingleGamma()
    >>> detector.geometric_efficiency((0.0, 0.0, 0.0), emission)
    0.447213595499958
"""

from .geometry import CylindricalDetector, DualHeadDetector
from .emission import SingleGamma, BackToBack

__all__ = [
    "CylindricalDetector",
    "DualHeadDetector",
    "SingleGamma",
    "BackToBack"
]