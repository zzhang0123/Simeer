"""Minimal end-to-end example: synthetic Gaussian beam + uniform sky.

Runs without the real MeerKLASS beam file (which is tens of GB) and
without ``limTOD``. Demonstrates the low-level integrator API of Simeer.

    python examples/quickstart.py
"""

from __future__ import annotations

import healpy as hp
import numpy as np

from simeer import integrate_tod, synthetic_gaussian_beam


def main() -> None:
    # Build a synthetic +/-6 degree beam with FWHM ~ 1.5 degrees, over a
    # few representative frequencies.
    freq_MHz = np.array([850.0, 900.0, 950.0])
    margin_deg = np.linspace(-6, 6, 121)
    beam = synthetic_gaussian_beam(freq_MHz=freq_MHz, margin_deg=margin_deg, fwhm_deg=1.5)

    # Uniform sky at 10 K (intentionally trivial -- the result should
    # come back at ~ 10 K for every sample at every frequency).
    nside = 128
    sky = np.full((len(freq_MHz), hp.nside2npix(nside)), 10.0, dtype=np.float64)

    # A small azimuth raster.
    ntime = 32
    lst_deg = np.linspace(100.0, 110.0, ntime)
    az_deg = np.linspace(170.0, 190.0, ntime)
    el_deg = np.full(ntime, 45.0)

    tod = integrate_tod(
        lst_deg_list=lst_deg,
        az_deg_list=az_deg,
        el_deg_list=el_deg,
        lat_deg=-30.7130,
        beam=beam,
        sky_maps=sky,
        freq_MHz=freq_MHz,
        disc_radius_deg=8.0,
        polarization="HH",
        n_jobs=1,
        progress=False,
    )

    print(f"tod.shape = {tod.shape}")
    print(f"tod[freq=900 MHz] = {tod[1]}")
    print(f"mean over all samples = {tod.mean():.4f} (expected ~ 10.0)")


if __name__ == "__main__":
    main()
