"""
Common type definitions used throughout the geometry and simulation library.
"""

#from typing import TypeAlias
from numpy.typing import NDArray
import numpy as np

Point3D = NDArray[np.float_]
"""Three-dimensional Cartesian point."""

Vector3D  = NDArray[np.float_]
"""Three-dimensional Cartesian vector."""