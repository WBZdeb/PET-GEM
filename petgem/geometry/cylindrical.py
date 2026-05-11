"""
Cylindrical detector geometry definition.

This module provides a class describing idealized cylindrical detector geometry
used in geometric efficiency modeling. Geometric parameters are represented in an
arbitrary but internally consistent length scale unless stated otherwise.
"""
from math import sqrt

import numpy as np
from scipy.integrate import dblquad

from ._base import _BaseDetector
from .._types import Point3D, Vector3D, CylinderCoords
from ..emission.types import EmissionTypes


class CylindricalDetector(_BaseDetector):
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

    Notes:
        The detector geometry is centered at the origin and aligned with the z-axis.
    """

    _supported_emissions = {
        EmissionTypes.SINGLE: "_single_solver",
        EmissionTypes.BACKTOBACK: "_backtoback_solver"
    }

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

    def _dist_vector(self, surface_coords: CylinderCoords, source_pos: Point3D) -> Vector3D:
        """
        Evaluate the displacement vector between the source position and a
        cylindrical detector surface point.
        """

        theta, z = surface_coords

        return np.array([
            self._inner_radius * np.cos(theta) - source_pos[0],
            self._inner_radius * np.sin(theta) - source_pos[1],
            z - source_pos[2]
        ])

    def _surface_normal(self, surface_coords: CylinderCoords) -> Vector3D:
        """
        Evaluate the outward radial vector on the cylindrical detector surface.
        """

        theta, _ = surface_coords

        return np.array([
            self._inner_radius * np.cos(theta),
            self._inner_radius * np.sin(theta),
            0.0
        ])

    def _integrand_wrapper(self, theta: float, z: float, source_pos: Point3D) -> float:
        """
        Wrapper adapting cylindrical surface coordinates to scipy integration
        conventions.
        """
        return self._integrand((theta, z), source_pos)

    def _projected_z(self, source_pos: Point3D, boundary_point: Point3D) -> float:
        """
        Compute the axial coordinate of the projected intersection between a
        source trajectory and the cylindrical detector boundary.

        Args:
            source_pos (Point3D): Three-dimensional source position vector.
            boundary_point (Point3D): Detector boundary reference point.

        Returns:
            float: Axial coordinate of the projected cylindrical intersection.
        """

        return (
                source_pos[2]
                + (-boundary_point[0] - source_pos[0])
                * (boundary_point[2] - source_pos[2])
                / (boundary_point[0] - source_pos[0])
        )

    def _compute_integration_bounds(self, source_pos: Point3D):
        """
        Compute integration bounds for cylindrical back-to-back geometry.

        Constructs axial projection limits and a geometry-dependent angular
        boundary function describing the visible detector region for a source
        located at `source_pos`.

        The returned angular constraint accounts for cylindrical curvature
        and axial field-of-view truncation.

        Args:
            source_pos (Point3D): Three-dimensional source position vector.

        Returns:
            tuple:
                - z_bottom (tuple[float, float]): projected lower axial bounds
                - z_top (tuple[float, float]): projected upper axial bounds
                - theta_max (Callable[[float], float]): angular सीमा function
        """

        r = float(self._inner_radius)
        afov = float(self._afov)

        # Axial limits of a cylinder projection
        z_bottom = (
            self._projected_z(source_pos, np.array((r, 0.0, afov / 2))),
            self._projected_z(source_pos, np.array((r, 0.0, -afov / 2)))
        )

        z_top = (
            self._projected_z(source_pos, np.array((-r, 0.0, afov / 2))),
            self._projected_z(source_pos, np.array((-r, 0.0, -afov / 2)))
        )

        # Radial integration boundaries as function of z coordinate
        def x_boundary(z: float) -> float:
            z_b0, z_b1 = z_bottom
            z_t0, z_t1 = z_top

            if z < z_t0:
                return -r + (z - z_b0) * (2 * r / (z_t0 - z_b0))

            elif z <= z_t1:
                return -r

            else:
                return -r + (z_b1 - z) * (2 * r / (z_b1 - z_t1))

        # Angular integration boundaries as function of z coordinate
        def theta_max(z: float) -> float:
            z_t0, z_t1 = z_top

            if z_t0 <= z <= z_t1:
                return np.pi

            x = x_boundary(z) / r
            x = np.clip(x, -1.0, 1.0)

            return np.arccos(-x)

        return z_bottom, z_top, theta_max

    def _single_solver(self, source_pos: Point3D) -> float:
        """
        Compute geometric efficiency for isotropic single-photon emission.

        Evaluates the normalized solid-angle contribution of the cylindrical
        detector surface for a point source located at `source_pos`.

        Args:
            source_pos (Point3D): Three-dimensional source position vector.

        Returns:
            float: Geometric detection efficiency for single-photon emission.
        """
        integration_result = dblquad(self._integrand_wrapper, -self._afov / 2, self._afov / 2,
                                0.0, 2 * np.pi, args=(source_pos,))[0]

        return integration_result / self._FULL_SOLID_ANGLE

    def _backtoback_solver(self, source_pos: Point3D) -> float:
        """
        Compute geometric efficiency for back-to-back photon emission.

        Evaluates the normalized solid-angle contribution associated with
        coincident back-to-back emission using cylindrical detector symmetry
        to reduce the integration domain.

        The source position is transformed into a symmetry-equivalent
        representation in the transverse plane to simplify integration
        bounds evaluation.

        Args:
            source_pos (Point3D): Three-dimensional source position vector.

        Returns:
            float: Geometric detection efficiency for back-to-back emission.
        """

        # Simplifying position using symmetry
        source_pos = np.array((
            sqrt(source_pos[0] ** 2 + source_pos[1] ** 2),
            0.0,
            np.abs(source_pos[2])
        ))

        z_bottom, z_top, theta_max = self._compute_integration_bounds(source_pos)

        z_min = max(-self._afov / 2, z_bottom[0])
        z_max = self._afov / 2

        integration_result = dblquad(self._integrand_wrapper, z_min, z_max, lambda z: np.pi - theta_max(z),
                                     lambda z: np.pi + theta_max(z), args=(source_pos,))[0]

        return integration_result / self._FULL_SOLID_ANGLE