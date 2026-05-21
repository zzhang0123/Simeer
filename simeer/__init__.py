"""
Simeer: optimal MeerKLASS TOD simulator with native (l, m) beam interpolation.

Public API
----------

High-level entry points:

*   :class:`MeerKLASSBeam` -- load and query the holographic beam.
*   :class:`SimeerTODSim` -- drop-in replacement for
    :class:`limTOD.TODSim` whose sky-TOD step uses the (l, m) disc path
    instead of HEALPix spherical-harmonic rotation.

Lower-level building blocks (useful for custom pipelines / tests):

*   :func:`simeer.sky_integrator.integrate_sample`
*   :func:`simeer.sky_integrator.integrate_tod`
*   :func:`simeer.projection.direction_cosines`
*   :func:`simeer.interpolation.precompute_bilinear_weights`
*   :func:`simeer.disc.select_disc`
"""

from .beam import MeerKLASSBeam, synthetic_gaussian_beam
from .sky_integrator import (
    integrate_sample,
    integrate_tod,
    materialise_sky_cube,
    materialize_sky_cube,
)

__all__ = [
    "MeerKLASSBeam",
    "synthetic_gaussian_beam",
    "integrate_sample",
    "integrate_tod",
    "materialize_sky_cube",
    # ``materialise_sky_cube`` is the deprecated British spelling kept for
    # back-compat; will be removed in v0.2.
    "materialise_sky_cube",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazy import :class:`SimeerTODSim` to avoid hard dependency on limTOD.

    Allows ``from simeer import SimeerTODSim`` to give a clean
    ``ImportError`` mentioning limTOD only when the class is actually
    referenced.
    """
    if name == "SimeerTODSim":
        from .simulator import SimeerTODSim as _S

        return _S
    raise AttributeError(name)
