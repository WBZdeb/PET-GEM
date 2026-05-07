"""
Common type definitions used throughout the geometry and simulation library.
"""

from typing import Tuple
from numpy.typing import NDArray
import numpy as np

Point3D = NDArray[np.float_]
"""Three-dimensional Cartesian point."""

Vector3D  = NDArray[np.float_]
"""Three-dimensional Cartesian vector."""

CylinderCoords = Tuple[float, float]
"""
Cylindrical surface parameterization coordinates `(theta, z)`.
"""

PlaneCoords = Tuple[float, float, float]
"""
Planar surface parameterization coordinates `(x, z, y)`.
"""