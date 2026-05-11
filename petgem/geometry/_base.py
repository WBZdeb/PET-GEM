"""
Abstract detector geometry definitions.

This module defines internal abstract base classes shared by detector
geometry implementations used in geometric efficiency calculations.

The base detector interface establishes the geometric operations required
by numerical integration routines while allowing individual detector
geometries to provide geometry-specific parameterization.
"""

#import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from .._types import Point3D, Vector3D
from ..emission._base import EmissionModel


class _BaseDetector(ABC):
    """
    Abstract base class for detector geometry models.

    Defines the internal interface required by detector geometries used in
    geometric efficiency calculations. Concrete subclasses implement
    geometry-specific surface parameterization and integration kernels,
    while shared solver dispatch and numerical integration infrastructure
    are handled by the base class.

    Subclasses must implement:
        - surface displacement vector evaluation,
        - surface normal evaluation,
        - geometric integration kernel evaluation.

    Notes:
        This class is intended for internal library use only.
    """

    _FULL_SOLID_ANGLE = 4.0 * np.pi
    """Solid angle corresponding to full spherical coverage."""

    _supported_emissions = {}
    """Mapping between emission types and geometry-specific solver methods."""

    def _select_solver(self, emission: EmissionModel):
        """
        Select the numerical solver associated with a given emission model.

        Args:
            emission (EmissionModel): Emission model describing the source
                emission characteristics.

        Returns:
            Callable: Geometry-specific solver method associated with the
                requested emission type.

        Raises:
            NotImplementedError: If the detector geometry does not support
                the requested emission type.
        """
        emission_type = emission.type

        if emission_type not in self._supported_emissions:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support '{emission_type}' emission type."
            )

        solver_name = self._supported_emissions[emission_type]
        solver = getattr(self, solver_name)

        return solver

    @abstractmethod
    def _surface_normal(self, *args):
        """
        Evaluate the outward detector surface normal vector.

        Returns:
            Vector3D: Outward-oriented surface normal vector associated with
                the parameterized detector surface point.
        """

    @abstractmethod
    def _dist_vector(self, surface_coords: Tuple[float, ...], source_pos: Point3D):
        """
        Evaluate the displacement vector between a source position and a
        detector surface point.

        Args:
            surface_coords: Geometry-specific surface parameterization
                coordinates.
            source_pos (Point3D): Three-dimensional source position vector.

        Returns:
            Vector3D: Displacement vector connecting the source position
                and the detector surface point.
        """

    def _integrand(self, surface_coords: Tuple[float, ...], source_pos: Point3D):
        """
        Evaluate geometric surface integration kernel.

        Args:
            surface_coords (tuple[float, ...]): Geometry-specific surface
                parameterization coordinates.
            source_pos (Point3D): Three-dimensional source position vector.

        Returns:
            float: Value of the geometric integration kernel.
        """

        r = self._dist_vector(surface_coords, source_pos)
        r_len = np.linalg.norm(r)
        n = self._surface_normal(surface_coords)

        if r_len == 0.0:
            return 0.0

        return np.dot(r, n) / (r_len ** 3.0)

    def geometric_efficiency(self, source_pos: Point3D, emission: EmissionModel) -> float:
        """
        Compute detector geometric efficiency for a source position and
        emission model.

        Args:
            source_pos (Point3D): Three-dimensional source position vector.
            emission (EmissionModel): Emission model defining the source
                emission characteristics.

        Returns:
            float: Detector geometric efficiency expressed as a percentage of
                all emitted photons.
        """
        efficiency_solver = self._select_solver(emission)

        return efficiency_solver(source_pos)