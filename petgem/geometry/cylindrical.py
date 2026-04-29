"""
Cylindrical detector geometry definition.

This module provides a class describing idealized  cylindrical detector geometry
used in geometric efficiency modeling. Geometric parameters are represented in an
arbitrary but internally consistent length scale unless stated otherwise.
"""

import numpy as np

from .._types import Point3D, Vector3D

class CylindricalDetector:
    """
    Cylindrical detector geometry model.

    Represents an idealized cylindrical detector defined by a characteristic
    radius and an angular field-of-view (AFOV). Both parameters are expressed
    in a common, arbitrary length scale and must be provided consistently.

    Attributes:
        _inner_radius (float): Characteristic inner radius of the detector,
            in an arbitrary but consistent length unit.
        _afov (float): Angular field-of-view parameter,
            expressed in the same length unit as _inner_radius.
    """

    def __init__(self, inner_radius: float, afov: float):
        """
        Initialize a CylindricalDetector instance.

        Args:
            inner_radius (float): Inner radius of the cylindrical detector.
                Must use the same length unit as afov.
            afov (float): Angular field of view of the detector.
                Must use the same length unit as inner_radius.
        """

        self._inner_radius = inner_radius
        self._afov = afov

    def get_inner_radius(self) -> float:
        """
        Return the detector inner radius.

        Returns:
            float: Characteristic inner radius of the detector.
        """

        return self._inner_radius

    def set_inner_radius(self, inner_radius: float) -> None:
        """
        Set the detector inner radius.

        Args:
            inner_radius (float): New characteristic inner radius in the
                same length scale used by the detector model.
        """

        self._inner_radius = inner_radius

    def get_afov(self) -> float:
        """
        Return the detector axial field-of-view.

        Returns:
            float: Axial field-of-view parameter.
        """

        return self._afov

    def set_afov(self, afov: float) -> None:
        """
        Set the detector axial field-of-view.

        Args:
            afov (float): New axial field-of-view parameter in the same
                length scale used by the detector model.
        """

        self._afov = afov

    def _dist_vector(self, theta: float, z: float, source_pos: Point3D) -> Vector3D:
        """
        Compute the displacement vector from the source position to a point on the
        cylindrical detector surface.

        Evaluates the vector connecting source_pos to the cylindrical surface point
        defined by azimuthal coordinate theta and axial coordinate z for a cylinder
        centered at the origin and aligned with the z-axis.

        Args:
            theta (float): Azimuthal angle in radians.
            z (float): Axial coordinate on the cylindrical surface.
            source_pos (Point3D): Three-dimensional source position vector.

        Returns:
            Vector3D: Three-dimensional displacement vector from source_pos to
                the specified detector surface point.
        """

        return np.array([
            self._inner_radius * np.cos(theta) - source_pos[0],
            self._inner_radius * np.sin(theta) - source_pos[1],
            z - source_pos[2]
        ])

    def _surface_normal(self, theta: float) -> Vector3D:
        """
        Compute the outward radial surface vector on the cylindrical detector.

        Evaluates the outward-pointing radial vector associated with the
        azimuthal angle theta for a cylinder centered at the origin and aligned
        with the z-axis. The returned vector lies in the transverse xy-plane and
        has magnitude equal to the detector inner radius.

        Args:
            theta (float): Azimuthal angle in radians.

        Returns:
            Vector3D: Three-dimensional outward radial surface vector.
        """

        return np.array([
            self._inner_radius * np.cos(theta),
            self._inner_radius * np.sin(theta),
            0.0
        ])

    def _integrand(self, theta: float, z: float, source_pos: Point3D) -> float:
        """
        Evaluate the cylindrical surface integration kernel.

        Computes the geometric integrand associated with the detector cylindrical
        surface at angular coordinate theta and axial coordinate z for a source
        located at source_pos.

        Args:
            theta (float): Azimuthal angle in radians.
            z (float): Axial coordinate on the cylindrical surface.
            source_pos (np.ndarray): Three-dimensional source position vector.

        Returns:
            float: Value of the geometric integration kernel.
        """

        r = self._dist_vector(theta, z, source_pos)
        r_len = float(np.linalg.norm(r))
        n = self._surface_normal(theta)

        if r_len == 0.0:
            return 0.0

        return np.dot(r, n) / (r_len ** 3.0)