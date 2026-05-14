"""
Emission topology models used in geometric efficiency calculations.

This package provides source emission models describing photon
emission characteristics supported by detector geometry solvers.

Classes:
    SingleEmission: Isotropic single-photon emission model.
    BackToBackEmission: Coincident back-to-back photon emission model.

Enums:
    EmissionTypes: Enumeration of supported emission model types.
"""

from .single import SingleGamma
from .backtoback import BackToBack

__all__ = [
    "SingleGamma",
    "BackToBack"
]