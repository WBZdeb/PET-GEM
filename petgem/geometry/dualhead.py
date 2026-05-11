"""
Dual-head detector geometry definition.

This module provides a class describing idealized dual-head detector geometry
used in geometric efficiency modeling. Geometric parameters are represented in an
arbitrary but internally consistent length scale unless stated otherwise.
"""

import numpy as np
from scipy.integrate import dblquad

from ._base import _BaseDetector
from .._types import Point3D, Vector3D, PlaneCoords
from ..emission.types import EmissionTypes


class DualHeadDetector(_BaseDetector):
    """
    Dual-head detector geometry model.

    Represents an idealized dual-head detector. Both heads of the detector are assumed
    to be of the same size, given by axial and transaxial field-of-view parameters.
    All parameters are expressed in a common, arbitrary length scale and must be provided consistently.

    Attributes:
        _transaxial_fov (float): Characteristic transaxial field-of-view of the detector,
            in an arbitrary but consistent length unit. Assumed to be along the X-axis.
        _axial_fov (float): Characteristic axial field-of-view of the detector,
            in an arbitrary but consistent length unit. Assumed to be along the Z-axis.
        _head_separation (float): Characteristic distance between the heads of a dual-head detector,
            in an arbitrary but consistent length unit. Assumed to be along the Y-axis.

    Notes:
        The detector coordinate system is centered midway between detector heads. Detector planes
        are parallel to the xz-plane and positioned symmetrically along the y-axis.
    """

    _supported_emissions = {
        EmissionTypes.SINGLE: "_single_solver",
        EmissionTypes.BACKTOBACK: "_backtoback_solver"
    }

    def __init__(self, transaxial_fov: float, axial_fov: float, head_separation: float):
        """
        Initialize a DualHeadDetector instance.

        Args:
            transaxial_fov (float): Transaxial field-of-view of the dual-head detector.
                Must use the same length unit as axial_fov.
            axial_fov (float): Angular field-of-view of the dual-head detector.
                Must use the same length unit as transaxial_fov.
            head_separation (float): Distance between the heads of a dual-head detector.
                Must use the same length unit as transaxial_fov.
        """

        self._transaxial_fov = transaxial_fov
        self._axial_fov = axial_fov
        self._head_separation = head_separation

    def get_transaxial_fov(self) -> float:
        """
        Return the transaxial field-of-view of the detector.

        Returns:
            float: Transaxial field-of-view extent.
        """

        return self._transaxial_fov

    def set_transaxial_fov(self, transaxial_fov: float) -> None:
        """
        Set the transaxial field-of-view of the detector.

        Args:
            transaxial_fov (float): New transaxial field-of-view extent,
                expressed in the detector's internal length scale.
        """

        self._transaxial_fov = transaxial_fov

    def get_axial_fov(self) -> float:
        """
        Return the axial field-of-view of the detector.

        Returns:
            float: Axial field-of-view extent.
        """

        return self._axial_fov

    def set_axial_fov(self, axial_fov: float) -> None:
        """
        Set the axial field-of-view of the detector.

        Args:
            axial_fov (float): New axial field-of-view extent,
                expressed in the detector's internal length scale.
        """

        self._axial_fov = axial_fov

    def get_fov(self) -> tuple[float, float]:
        """
        Return the detector field-of-view dimensions.

        Returns:
            tuple[float, float]: Two-element tuple containing the transaxial and
                axial field-of-view parameter, respectively.
        """
        return self._transaxial_fov, self._axial_fov

    def set_fov(self, transaxial_fov: float, axial_fov: float) -> None:
        """
        Set the detector field-of-view dimensions.

        Args:
            transaxial_fov (float): Transaxial field-of-view parameter.
            axial_fov (float): Axial field-of-view parameter.
        """
        self._transaxial_fov = transaxial_fov
        self._axial_fov = axial_fov

    def get_head_separation(self) -> float:
        """
        Return the detector field-of-view dimensions.

        Returns:
            float: Distance between heads of the detector.
        """
        return self._head_separation

    def set_head_separation(self, head_separation: float) -> None:
        """
        Set the distance between heads of the detector.

        Args:
            head_separation (float): Distance between heads of the detector.
        """
        self._head_separation = head_separation

    def _dist_vector(self, surface_coords: PlaneCoords, source_pos: Point3D) -> Vector3D:
        """
        Evaluate the displacement vector between the source position and a planar detector surface point.
        """

        x_coord, z_coord, y_coord = surface_coords

        return np.array([
            x_coord - source_pos[0],
            y_coord - source_pos[1],
            z_coord - source_pos[2]
        ])

    def _surface_normal(self, surface_coords: PlaneCoords) -> Vector3D:
        """
        Evaluate the outward surface normal vector for a planar detector head.
        """

        _, _, y_coord = surface_coords

        return np.array([
            0.0,
            np.sign(y_coord),
            0.0
        ])

    def _integrand_wrapper(self, x: float, z: float, source_pos: Point3D, y_coord: float) -> float:
        """
        Wrapper adapting planar surface coordinates to scipy integration conventions.
        """

        return self._integrand(
            (x, z, y_coord),
            source_pos
        )

    def _compute_integration_bounds(self, source_pos: Point3D, y_coord: float, afov_x, afov_z):
        """
        Compute projected integration bounds for a planar detector head.

        Evaluates the effective visible detector region associated with a
        back-to-back emission geometry using rectangular projection onto
        the detector plane.

        Args:
            source_pos (Point3D): Three-dimensional source position vector.
            y_coord (float): Y-coordinate of the detector head plane.
            afov_x (float): Transaxial detector field-of-view extent.
            afov_z (float): Axial detector field-of-view extent.

        Returns:
            tuple[float, float, float, float]:
                Minimum and maximum integration bounds in x and z coordinates:
                (x_min, x_max, z_min, z_max).
        """
        abs_y = abs(y_coord)

        if y_coord >= 0:
            scale = np.abs((abs_y - source_pos[1]) / (abs_y + source_pos[1]))
        else:
            scale = np.abs((abs_y + source_pos[1]) / (abs_y - source_pos[1]))

        x_cen = source_pos[0] * (1 - scale)
        z_cen = source_pos[2] * (1 - scale)
        x_len = scale * afov_x
        z_len = scale * afov_z
        x_min = max(x_cen - x_len / 2, -afov_x / 2)
        z_min = max(z_cen - z_len / 2, -afov_z / 2)
        x_max = min(x_cen + x_len / 2, afov_x / 2)
        z_max = min(z_cen + z_len / 2, afov_z / 2)

        return x_min, x_max, z_min, z_max

    def _single_solver(self, source_pos: Point3D) -> float:
        """
        Compute geometric efficiency for isotropic single-photon emission.

        Evaluates the normalized solid-angle contribution of both planar
        detector heads for a point source located at `source_pos`.

        Args:
            source_pos (Point3D): Three-dimensional source position vector.

        Returns:
            float: Geometric detection efficiency for single-photon emission.
        """
        plane_up = dblquad(
            self._integrand,
            -self._axial_fov / 2,
            self._axial_fov / 2,
            -self._transaxial_fov / 2,
            self._transaxial_fov / 2,
            args=(source_pos, self._head_separation / 2,)
        )[0]
        plane_down = dblquad(
            self._integrand,
            -self._axial_fov / 2,
            self._axial_fov / 2,
            -self._transaxial_fov / 2,
            self._transaxial_fov / 2,
            args=(source_pos, -self._head_separation / 2,)
        )[0]

        return (plane_up + plane_down) / self._FULL_SOLID_ANGLE

    def _backtoback_solver(self, source_pos: Point3D) -> float:
        """
        Compute geometric efficiency for back-to-back photon emission.

        Evaluates the normalized solid-angle contribution associated with
        coincident back-to-back emission for opposing planar detector heads.

        Effective integration bounds are computed separately for each
        detector plane using geometric projection of the visible detector
        region.

        Args:
            source_pos (Point3D): Three-dimensional source position vector.

        Returns:
            float: Geometric detection efficiency for back-to-back emission.
        """

        # Calculate solid angle for upper plane
        x_min, x_max, z_min, z_max = self._compute_integration_bounds(
            source_pos,
            self._head_separation / 2,
            self._transaxial_fov,
            self._axial_fov
        )
        plane_up = dblquad(
            self._integrand,
            z_min,
            z_max,
            x_min,
            x_max,
            args=(source_pos, self._head_separation / 2,)
        )[0]

        # Calculate solid angle for lower plane
        x_min, x_max, z_min, z_max = self._compute_integration_bounds(
            source_pos,
            -self._head_separation / 2,
            self._transaxial_fov,
            self._axial_fov
        )
        plane_down = dblquad(
            self._integrand,
            z_min,
            z_max,
            x_min,
            x_max,
            args=(source_pos, -self._head_separation / 2,)
        )[0]

        return (plane_up + plane_down) / self._FULL_SOLID_ANGLE