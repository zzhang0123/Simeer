# Simeer

**Sim**ulator for **Mee**rKLASS **R**adio TOD with native (l, m) primary
beam interpolation.

- **Author**: Zheng Zhang &lt;zheng.zhang@manchester.ac.uk&gt; (University of Manchester)
- **License**: MIT (see [LICENSE](LICENSE))
- **Citation**: see [CITATION.cff](CITATION.cff)
- **Version**: 0.1.0

Simeer is an optimal replacement for `limTOD`'s `simulate_sky_TOD` step
specialised for the MeerKLASS holographic primary beam. It avoids the
HEALPix spherical-harmonic rotation path -- which would require
representing the +/-6 degree beam at very high HEALPix resolution to
preserve precision -- by working directly on the beam's native
direction-cosine grid `(l, m)` for every pointing.

Everything else in a MeerKLASS TOD simulation (gain noise, flicker
noise, white noise, LST utilities, map-making) is unchanged and is
delegated to `limTOD`.

## Why a separate package

The current MeerKLASS holographic beam ships as an NPZ archive on a
regular `(n_freq, n_m, n_l)` grid with a beam extent of about +/-6
degrees. Two facts make the standard `limTOD` path expensive here:

1.  The beam is much narrower than the HEALPix pixel size at modest
    `nside`, so the spherical-harmonic rotation either loses precision
    (at low `nside`) or becomes very expensive (at high `nside`).
2.  The raw beam file is tens of GB on disk. Rotating the whole sky to
    the beam frame, or vice versa, is unnecessary -- only the sky
    pixels within the +/-6 degree disc around each pointing matter.

Simeer addresses both points with a **sky-to-beam, disc-restricted**
path:

1.  For each pointing, query a HEALPix disc around the antenna pointing
    direction (typically ~8 degrees of radius).
2.  Project the disc pixel directions back to the antenna-local tangent
    plane to get their direction cosines `(l, m)`.
3.  Precompute bilinear interpolation indices and weights against the
    beam's native `(m, l)` grid **once per pointing**, and apply them
    vectorially across the full frequency axis. The beam never has to
    be rotated.
4.  Multiply by the sky temperature, sum, and normalise by the beam
    solid angle $\Omega_b(\nu)$.

The result is a drop-in `SimeerTODSim` class that exposes the same
interface as `limTOD.TODSim`.

## Architecture at a glance

```
simeer/
  __init__.py            Public exports (lazy SimeerTODSim import)
  beam.py                MeerKLASSBeam (file loader + power cubes + Omega_b)
  projection.py          Horizon <-> equatorial; SIN direction cosines
  disc.py                HEALPix disc query with per-pointing LRU cache
  interpolation.py       Bilinear weights precomputed once, applied to all freqs
  stokes.py              Stokes-I integration (Q/U/V on the follow-up list)
  sky_integrator.py      integrate_sample + integrate_tod driver
  simulator.py           SimeerTODSim (subclass of limTOD.TODSim)
  _parallel.py           joblib (Loky process pool) wrapper
tests/
  test_projection.py
  test_interpolation.py
  test_beam.py
  test_sky_integrator.py
examples/
  quickstart.py
```

## Installation

```bash
git clone <repo>
cd Simeer
pip install -e .

# Optional: needed only for the SimeerTODSim drop-in class.
pip install -e ../limTOD
```

`limTOD` is an optional dependency. The lower-level
`simeer.sky_integrator.integrate_tod` works without it and is what most
research scripts will call directly.

### Dependencies

Required: `numpy`, `healpy`, `joblib`, `tqdm`.

Optional: `limTOD` (for `SimeerTODSim` and the Full-TOD path),
`matplotlib` and `jupyter` for the notebooks, `pytest` for tests.

## Quick start

This produces a **Sky TOD** -- the noiseless, beam-weighted sky signal
for each pointing. No external dependencies beyond the ones in
`pyproject.toml`; gain and noise injection are out of scope (see
*Production path* below).

```python
import numpy as np
import healpy as hp
from simeer import MeerKLASSBeam, integrate_tod

# 1. Load the holographic beam (array-average, HH+VV by default).
beam = MeerKLASSBeam("MeerKAT_U_band_primary_beam.npz")

# 2. Pick the frequency channels you want to simulate (must match the
#    beam's grid -- use beam.freq_MHz directly or a subset of it).
freq_MHz = beam.freq_MHz[::32]                        # 32 channels across the band

# 3. Build (or load) a sky model on the same frequency grid. Here we use
#    a uniform 8 K sky as a placeholder; substitute your own GSM/GDSM/
#    pyradiosky cube of shape (n_freq, n_pix_sky).
nside_sky = 256
sky_maps = np.full((len(freq_MHz), hp.nside2npix(nside_sky)), 8.0, dtype=np.float64)

# 4. Specify the pointing time series. (LST is in degrees; convert from
#    UTC + site coordinates yourself, or use whatever scheduler you have.)
ntime = 4000
lst_deg = np.linspace(0.0, 30.0, ntime)
az_deg  = 180.0 + 5.0 * np.sin(np.linspace(0, 4 * np.pi, ntime))
el_deg  = np.full(ntime, 41.5)

# 5. Generate the Sky TOD.
sky_tod = integrate_tod(
    lst_deg_list=lst_deg,
    az_deg_list=az_deg,
    el_deg_list=el_deg,
    lat_deg=-30.7130,                # MeerKAT
    beam=beam,
    sky_maps=sky_maps,
    freq_MHz=freq_MHz,
    disc_radius_deg=8.0,             # ~ +/-6 deg beam + safety margin
    polarization="HH",
    n_jobs=-1,                       # joblib: use all cores
    progress=True,
)
# sky_tod.shape == (n_freq, n_time)  -- antenna temperature in K
```

For the **Full TOD** (Sky TOD multiplied by receiver gain and with 1/f
and white noise injected), see the *Production path* section below and
`SimeerTODSim.generate_TOD` in `simeer/simulator.py`.

## Production path: Sky TOD vs Full TOD

### What is the Sky TOD?

The **Sky TOD** is the noiseless, gain-free sky signal that an ideal
telescope would measure at each pointing -- the beam-weighted integral
of the sky brightness temperature over the antenna's primary beam:

$$
T_{\text{sky}}(\nu, t) = \frac{1}{\Omega_b(\nu)} \int_{S^2} B(\hat{n}, \nu) T(\hat{n}, \nu) d\Omega
$$

where:

- $B(\hat{n}, \nu)$ is the primary beam *power* pattern at frequency $\nu$ — a function on the **2-sphere** $S^2$ in directions $\hat{n}$ relative to the pointing. Storage-wise it is parameterised by direction cosines $(l, m)$ for convenience, but the integrand is a density with respect to **sphere measure $d\Omega$**.
- $\Omega_b(\nu) = \int_{S^2} B \mathop{}\!d\Omega$ is the beam solid angle, with units of steradians, so the ratio is dimensionless and the result is in the same temperature units as the input sky.
- $T(\hat{n}, \nu)$ is the sky brightness temperature in Kelvin, evaluated at the disc pixels (Simeer projects HEALPix sky pixels into the antenna-local frame on the fly).

The Sky TOD is the "pure" signal that downstream calibration and
map-making algorithms ultimately try to recover from the measured TOD.
Simeer **only** computes the Sky TOD; gain, 1/f noise, and white-noise
injection live in `limTOD`.

### What is the Full TOD?

The **Full TOD** is the realistic instrument-modulated signal, the
quantity an actual receiver writes to disk:

$$
\text{TOD}(\nu, t) = G_{\rm bg}(\nu, t) \bigl[1 + G_{\rm noise}(\nu, t)\bigr] \bigl[T_{\rm sky}(\nu, t) + T_{\rm sys}^{\text{other}}(\nu, t)\bigr] \bigl[1 + \eta(t)\bigr]
$$

with $G_{\rm bg}$ the background gain pattern, $G_{\rm noise}$ the
multiplicative $1/f$ gain fluctuation, $T_{\rm sys}^{\text{other}}$
other system-temperature components (CMB, ground spill, receiver),
and $\eta$ additive white noise. The Full TOD assembly happens inside
`limTOD.TODSim.generate_TOD`; Simeer plugs into it by replacing only
the $T_{\rm sky}(\nu, t)$ step.

### How the Sky TOD is computed (production formula)

Each Sky TOD sample is a weighted sum of sky temperatures over a disc
of HEALPix pixels around the pointing. The exact production formula
implemented by `simeer.integrate_sample` is:

$$
\boxed{T_{\text{sky}}(f, t) = \frac{d\Omega_{\text{pix}}}{\Omega_b(f)} \sum_{i \in \text{disc}} B(l_i, m_i, f) T(\text{pix}_i, f)}
$$

> [!IMPORTANT]
> **Measure convention.** The discrete sum on the sky side is taken
> with the **HEALPix sphere-pixel solid angle** $d\Omega_{\text{pix}} = 4\pi / N_{\text{pix}}$,
> i.e., the natural measure of the 2-sphere $S^2$. This implicitly
> treats $B(l_i, m_i, f)$ as a **density on $S^2$** that is merely
> *parameterised* by the direction cosines $(l, m)$ via the SIN
> projection — not as a density on the flat $(l, m)$ tangent plane.
>
> If the stored beam were instead a density with respect to the
> Cartesian measure $dl \cdot dm$ on the tangent plane, the sum would need
> to be corrected by the SIN-projection Jacobian
>
> $$\frac{d\Omega}{dl \cdot dm} = \frac{1}{\sqrt{1 - l^2 - m^2}}$$
>
> before contracting against the sphere-measure sum. The MeerKLASS
> holographic beam is treated as the former (sphere density,
> direction-cosine parameterisation), so **no Jacobian factor is
> applied at the disc pixels** — `B(l_i, m_i, f)` enters the sum
> directly.
>
> The same logic applies on the normalisation side: $\Omega_b(f)$
> should *also* be the sphere integral $\int_{S^2} B d\Omega$.
> Currently the code computes
>
> $$\Omega_b(f) \approx (\Delta l)^2 \sum_{j,k} B(l_j, m_k, f)$$
>
> in the **flat $(l, m)$ measure** rather than including the Jacobian
> $1/\sqrt{1 - l^2 - m^2}$ inside the sum. This is a $\sim 0.55\%$
> bias at the $\pm 6^{\circ}$ beam edge (much smaller in practice
> because the beam mass is concentrated near $l = m = 0$ where the
> Jacobian is $\approx 1$). Behaviour inherited from `primary_beam.py`,
> tracked as follow-up #7.

#### Symbol $\to$ code correspondence

| Symbol | Meaning | Where it comes from |
| --- | --- | --- |
| $(\alpha_p, e_p)$ | pointing direction in degrees: azimuth and elevation | caller (`az_pointing_deg`, `el_pointing_deg`) |
| $(\text{ra}_p, \text{dec}_p)$ | equatorial direction of the pointing | `simeer.projection.horizon_to_equatorial(az_p, el_p, lst_deg, lat_deg)` |
| disc pixel set | HEALPix pixels within `disc_radius_deg` of $(\text{ra}_p, \text{dec}_p)$, minus pixels below the horizon ($e_i \leq 0$) and any pixels excluded by `horizontal_mask` | `simeer.disc.select_disc(...)` + keep mask in `integrate_sample` |
| $(\alpha_i, e_i)$ | horizontal direction (az, el) of each disc pixel at this LST | `simeer.projection.pixel_directions_to_az_el(nside_sky, pix_ids, lst, lat)` |
| $(l_i, m_i)$ | direction cosines of disc pixel $i$ in beam-local SIN projection: $l = \cos(e_i) \sin(\Delta\alpha)$, $m = \sin(e_i)\cos(e_p) - \cos(e_i)\sin(e_p)\cos(\Delta\alpha)$, $\Delta\alpha = \alpha_i - \alpha_p$ | `simeer.projection.direction_cosines(...)` (converted to degrees for the grid lookup) |
| $B(l_i, m_i, f)$ | beam **power** $\lvert\text{Jones}\rvert^2$ on $S^2$, bilinearly interpolated from the $(n_{\rm freq}, n_m, n_l)$ cube; treated as a density w.r.t. $d\Omega$ | `simeer.interpolation.precompute_bilinear_weights(...)` + `apply_bilinear(...)` on `beam.power_cube(pol)` |
| $T(\text{pix}_i, f)$ | sky brightness temperature [K] at disc HEALPix pixel $i$ (equatorial frame, same nside as the sum) | `sky_maps[sky_freq_indices, pix_ids]` |
| $d\Omega_{\text{pix}}$ | HEALPix pixel solid angle on $S^2$, constant by HEALPix's equal-area construction | $4\pi / N_{\text{pix}}$, computed inline |
| $\Omega_b(f)$ | beam solid angle (in steradians) | `beam.beam_solid_angle(pol)` $= (\Delta l_{\rm rad})^2 \sum_{j,k} B(l_j, m_k, f)$ — **flat-measure approximation**, see callout |
| product + weighted sum | the integration step itself | `simeer.stokes.integrate_stokes_I(beam_disc, sky_disc, omega_b, d_omega_pix)` |

#### What is and isn't applied

- ✅ **Beam normalisation by $\Omega_b$.** Result is in the same
  temperature units as `sky_maps`. For a uniform sky $T_0$ the formula
  returns $T_0$ modulo discretisation, provided the disc encloses the
  beam support.
- ⚠️ **SIN-projection Jacobian on $\Omega_b$** is omitted — flat-measure
  $(\Delta l)^2 \sum B$ rather than $(\Delta l)^2 \sum B / \sqrt{1 - l^2 - m^2}$. See the **Measure convention** callout above; tracked as follow-up #7.
- ✅ **HEALPix pixel-area Jacobian** not needed — HEALPix pixels all
  have the same solid angle, so the scalar $d\Omega_{\text{pix}}$ is
  exact.
- ✅ **Below-horizon and masked pixels** are dropped *before* the sum,
  so they contribute zero rather than being multiplied by a small but
  nonzero beam value.
- ❌ **Sky-frequency interpolation** not done — `sky_maps` must be
  aligned to the beam's output channels (follow-up #14).
- ❌ **Polarisation field rotation** not done — only Stokes I;
  `polarization='HH'` or `'VV'` selects the diagonal Jones-product
  cube. Sky Q/U would need a $2\chi$ rotation between sky and beam
  frames (follow-up #1); cross-pol $HV$/$VH$ leakage is not applied
  (follow-up #2).
- 🔁 **Beam rotation** never happens — only sky pixels are projected
  into the beam frame via the direction-cosine formula above. This is
  the key design difference from limTOD's spherical-harmonic rotation
  path.

#### Pseudocode of the actual call chain

```python
def integrate_sample(*, beam, sky_maps, lst_deg, az_pointing_deg, el_pointing_deg,
                    lat_deg, beam_freq_indices, sky_freq_indices, disc_radius_deg,
                    polarization, horizontal_mask):

    nside_sky = hp.npix2nside(sky_maps.shape[-1])

    # 1. Pointing -> equatorial.
    ra_p, dec_p = projection.horizon_to_equatorial(
        az_pointing_deg, el_pointing_deg, lst_deg, lat_deg)

    # 2. Disc query in equatorial coords.
    pix_ids = disc.select_disc(nside_sky, ra_p, dec_p, disc_radius_deg)

    # 3. Disc pixels back to horizontal at this LST; drop below-horizon and masked.
    az_s, el_s = projection.pixel_directions_to_az_el(
        nside_sky, pix_ids, lst_deg, lat_deg)
    keep = (el_s > 0)
    if horizontal_mask is not None:
        keep &= horizontal_mask[pix_ids]
    pix_ids, az_s, el_s = pix_ids[keep], az_s[keep], el_s[keep]

    # 4. Direction cosines in beam frame.
    l, m = projection.direction_cosines(
        az_pointing_deg, el_pointing_deg, az_s, el_s)

    # 5. Bilinear weights against (m, l) grid -- once per pointing, all freqs.
    weights = interpolation.precompute_bilinear_weights(
        np.rad2deg(l), np.rad2deg(m), beam.margin_deg, beam.margin_deg)

    # 6. Gather B at the disc points for the requested frequency channels.
    B_disc = interpolation.apply_bilinear(
        weights, beam.power_cube(polarization), beam_freq_indices)  # (n_freq, n_disc)

    # 7. Weighted sum, normalised by the beam solid angle.
    d_omega_pix = 4 * np.pi / hp.nside2npix(nside_sky)
    omega_b = beam.beam_solid_angle(polarization)[beam_freq_indices]
    sky_disc = sky_maps[sky_freq_indices[:, None], pix_ids[None, :]]
    return (d_omega_pix / omega_b) * np.sum(B_disc * sky_disc, axis=-1)
```

`integrate_tod` is a thin driver that loops the above over `lst_deg_list[i]`,
`az_deg_list[i]`, `el_deg_list[i]`, batches across joblib workers, and
hoists the per-batch invariants (the power cube, the omega_b slice, the
margin grid) so they are passed once as top-level ndarrays (joblib
auto-memmaps them).

### Key APIs

**Important: `integrate_tod` and `integrate_sample` both return the
Sky TOD only -- they do NOT add gain, 1/f, or white noise. Despite the
generic-sounding name, `integrate_tod` is a Sky-TOD generator.** The
only entry point that returns the Full TOD is
`SimeerTODSim.generate_TOD`.

| You want                                       | Call                                      | Returns                                          | Requires `limTOD`? |
| ---------------------------------------------- | ----------------------------------------- | ------------------------------------------------ | ------------------ |
| **Sky TOD** for the whole pointing list        | `simeer.integrate_tod(...)`               | `ndarray (n_freq, n_time)` -- the Sky TOD        | No                 |
| **Sky TOD** for a single pointing              | `simeer.integrate_sample(...)`            | `ndarray (n_freq,)` -- one Sky-TOD sample        | No                 |
| **Sky TOD** via the `limTOD`-style class       | `SimeerTODSim.simulate_sky_TOD(...)`      | `ndarray (n_freq, n_time)` -- Sky TOD            | Yes                |
| **Full TOD** (sky + gain + 1/f + white noise)  | `SimeerTODSim.generate_TOD(...)`          | `(overall_TOD, sky_TOD, gain_noise)` triple      | Yes                |

`SimeerTODSim.generate_TOD` is inherited unchanged from
`limTOD.TODSim.generate_TOD`; the only overridden step is the sky-TOD
computation, which now goes through Simeer's `integrate_tod` instead of
limTOD's spherical-harmonic rotation.

In short:

- If you only need Sky TOD (forward modelling, beam diagnostics, sky
  validation), call `simeer.integrate_tod` directly and skip the
  `limTOD` dependency.
- If you need a Full TOD with realistic gain and noise behaviour, use
  `SimeerTODSim.generate_TOD` -- the Sky TOD computed by Simeer is
  threaded straight into limTOD's noise/gain pipeline.

### Full TOD example

Requires `limTOD` to be installed (`pip install -e ../limTOD`). This
example produces the **Full TOD** -- the realistic instrument-modulated
signal that an actual receiver would write to disk.

```python
import numpy as np
from simeer import MeerKLASSBeam, SimeerTODSim

# Sky model and scan-pattern helpers come from limTOD.
from limTOD.sky_model import GDSM_sky_model
from limTOD.simulator import example_scan

beam = MeerKLASSBeam("MeerKAT_U_band_primary_beam.npz")

sim = SimeerTODSim(
    beam=beam,
    sky_func=GDSM_sky_model,                # callable: sky_func(freq=..., nside=...)
    sky_nside=512,
    disc_radius_deg=8.0,
    polarization="HH",
    ant_latitude_deg=-30.7130,              # MeerKAT
    ant_longitude_deg=21.4430,
    ant_height_m=1054.0,
    n_jobs=-1,
)

time_list, az_list = example_scan()         # simple raster scan from limTOD
overall_TOD, sky_TOD, gain_noise = sim.generate_TOD(
    freq_list=beam.freq_MHz[::32],          # 32 channels across U-band
    time_list=time_list,
    azimuth_deg_list=az_list,
    elevation_deg=41.5,
    start_time_utc="2019-04-23 20:41:56.397",
    # Optional: gain_noise_params=[f0, fc, alpha], white_noise_var=..., etc.
)
# overall_TOD : Full TOD with gain + 1/f + white noise injected
# sky_TOD     : the pure Sky TOD that integrate_tod would also return
# gain_noise  : the 1/f gain-noise realisation that was injected
```

The Sky-TOD step inside `generate_TOD` is Simeer's; everything else
(LST generation, background gain, 1/f noise, white noise, TOD assembly)
is inherited unchanged from `limTOD.TODSim`.

## Design choices

| Choice | Rationale |
| --- | --- |
| **sky-to-beam, disc-only** | Preserves the HEALPix sky parameterisation expected by downstream map-makers, while transforming only the small number of sky pixels actually convolved with the beam. |
| **Bilinear on native `(m, l)` grid** | Avoids forcing the beam onto a HEALPix grid; matches the data's intrinsic representation; trivially vectorisable across frequency. |
| **Precompute weights per pointing** | The interpolation indices and weights depend only on `(l, m)`, not frequency. Once per pointing we compute four scalars per pixel and reuse them across every channel. |
| **Float32 working dtype for the beam** | Cuts memory by 2x relative to float64; well below the precision of the holographic measurement. |
| **joblib (Loky) by default** | Avoids the launcher overhead of `mpi4py` on single-node runs. The hot loop is numpy-heavy, so a thread pool would be GIL-bound; a process pool is the right default. |
| **Stokes I only in v0.1** | The (Q, U) rotation between sky and beam frames is non-trivial and lands cleanly on top of TIBEC's validated rotation utilities; see follow-ups. |

## Parallelism

The per-sample TOD calculation is embarrassingly parallel over time.
`integrate_tod(..., n_jobs=-1)` uses `joblib.Parallel` with the Loky
process backend, dispatching one batch per worker (auto-tuned via the
`batch_size=` kwarg). The beam power cube, sky cube, omega_b slice and
`(l, m)` grid are passed as **top-level ndarrays** so that joblib's
auto-memmap (threshold lowered to 100 KB) takes over for the large
arrays, avoiding redundant per-batch pickling. Pass `n_jobs=1` for
serial execution (useful for debugging or for fine-grained MPI
partitioning at a higher level).

Measured scaling on a 28-core macOS host (`scripts/benchmark.py`):

| Problem size                         | n_jobs=1 | n_jobs=2 | n_jobs=4 | n_jobs=-1 |
| ------------------------------------ | -------: | -------: | -------: | --------: |
| ntime=4000, nside=128, nfreq=8       |  1.46 s  |  1.36 s  |  1.05 s  |   1.32 s  |
| ntime=20000, nside=256, nfreq=64     | 92.9 s   | 49.7 s   | 26.7 s   |   9.0 s   |

The small-problem row shows the regime where joblib pool startup
(~500 ms) dominates. The realistic-problem row -- which matches the
shape of a typical MeerKLASS observation -- shows clean near-linear
scaling (~10x at 28 cores).

For multi-node runs, partition the time list yourself and call
`integrate_tod(..., n_jobs=N)` on each node. A native MPI backend is on
the follow-up list.

## Testing

```bash
pytest                                  # unit + integration
pytest -m unit                          # fast unit tests only
pytest -m slow                          # tests that load the real beam file (skipped by default in CI)
```

The default suite uses the synthetic Gaussian beam via
`synthetic_gaussian_beam(...)` so it runs without the real ~30 GB beam
file (and without limTOD). The cross-check against limTOD's HEALPix-SH
path lives in `tests/test_against_limtod.py` (11 tests, gated behind
`pytest.importorskip("limTOD")` so the rest of the suite stays
limTOD-free). The cross-check also smoke-tests
`SimeerTODSim.simulate_sky_TOD` and `.generate_TOD` (the Full-TOD path),
which need limTOD's noise/gain machinery.

## Review iteration log

v0.1 went through one round of independent agent review covering
performance, architecture, Python idiom, and overall code quality. The
findings landed in code as follows (each item is now closed):

| # | Severity | Area              | Fix                                                                |
|---|----------|-------------------|--------------------------------------------------------------------|
| 1 | HIGH     | Correctness       | `synthetic_gaussian_beam` FWHM was sqrt(2) narrower than declared; formula corrected and a regression test added. |
| 2 | HIGH     | Robustness        | `horizontal_mask` wrong-shape now raises `ValueError` instead of silently disabling the mask. |
| 3 | HIGH     | Lint              | `dtype: np.dtype = np.float32` -> `DTypeLike`; local `ArrayLike = np.ndarray` -> `numpy.typing.ArrayLike`; unused `Mapping` import removed. |
| 4 | HIGH     | UX                | Progress bar wrapped the input list (flashed to 100% immediately); now wraps the work generator. |
| 5 | HIGH     | Deps              | `astropy` and `scipy` removed from required deps (never imported). |
| 6 | HIGH     | Performance       | Joblib was pickling the 80 MB beam cube per batch dispatch. Worker now receives bare ndarrays (cube + omega_b + margin_deg) at top level; joblib's `max_nbytes='100K'` auto-memmap takes over. Realistic scaling now ~10x on 28 cores. |
| 7 | MEDIUM   | API               | `sky_freq_indices` is now optional with sensible default. |
| 8 | MEDIUM   | API               | `integrate_sample` accepts `stokes_modes: tuple[str, ...] = ('I',)`; raises `NotImplementedError` for anything other than `('I',)`. Q/U addition will not be a signature break. |
| 9 | MEDIUM   | API               | `limTOD.TODSim` signature is now checked at `SimeerTODSim` construction. |
| 10| MEDIUM   | Robustness        | `integrate_stokes_I` raises if `omega_b <= 0`; `MeerKLASSBeam.from_arrays` rejects non-uniform `margin_deg`; `_BeamArrays.power` wrapped in `MappingProxyType`. |
| 11| MEDIUM   | Tests             | Added VV-polarisation test, empty/zeroed-mask test, beam-solid-angle cache test, FWHM regression test, mask-shape validation test, `Stokes != I` test, optional-`sky_freq_indices` test. 33 tests total (up from 20). |
| 12| LOW      | Style             | All modules: `from collections.abc import ...`, `X | None`, `tuple[X, Y]`; black + ruff clean. |
| 13| LOW      | Spelling          | `materialise_sky_cube` is now `materialize_sky_cube`; the British alias is retained for one release. |
| 14| MEDIUM   | Tests             | Cross-check vs limTOD landed in `tests/test_against_limtod.py` (11 tests, gated by `importorskip`). Median agreement 0.18-1.9% across el/az/FWHM corners. `SimeerTODSim.generate_TOD` (Full-TOD path) is now smoke-tested end-to-end. 44 tests total in TOD env (33 without limTOD). |

## Follow-ups

The roadmap below is **planned but not implemented** in v0.1. PRs
welcome.

### Correctness / scientific

1.  **Full-Stokes (Q, U, V) integration.** Implement
    `simeer.stokes.integrate_stokes_full` on top of TIBEC's validated
    sky-to-beam frame rotation. Needs a per-pixel parallactic-angle
    table for the disc, plus a 2x2 rotation by `2*chi` applied to the
    sky (Q, U) before contracting with the beam. The
    `stokes_modes` kwarg of `integrate_sample` is already wired up.
2.  **Cross-polarisation (HV, VH) leakage.** Materialise the off-diagonal
    Jones entries and propagate them into the antenna-temperature
    formula. Affects the linear polarisation fidelity at the few-percent
    level.
3.  ~~**Boundary-validation cross-check against limTOD.**~~ **Done.**
    See `tests/test_against_limtod.py` (11 tests, gated behind
    `pytest.importorskip("limTOD")`). Covers uniform sky and a smooth
    Dec-gradient sky, at boundary corners `el in {15, 45, 89}`, az
    wrap (0 / 180 / 359), and FWHM in {1.5, 2.5, 4.0} deg. Median
    relative agreement on the matched-Gaussian configurations: 0.18%
    at FWHM=4 deg, 0.65% at FWHM=2.5 deg, 1.9% at FWHM=1.5 deg --
    discrepancy is dominated by limTOD's HEALPix discretisation, not
    Simeer's bilinear interpolation. Also smoke-tests
    `SimeerTODSim.simulate_sky_TOD` and `.generate_TOD` (the Full-TOD
    path), which were previously zero-coverage.
4.  **Horizontal masking in horizontal coords (not equatorial).**
    Today `horizontal_mask` is indexed by equatorial pixel id (same
    frame as `sky_maps`). Translate to a genuine horizon-frame mask so
    callers can represent a permanent ground screen without rebuilding
    the mask per LST.
5.  **Beam interpolation accuracy study.** Quantify the bilinear-vs-
    bicubic-vs-spline error on the holographic beam; promote to a higher
    order if the few-percent residual at large radii matters for
    science.
6.  **Frequency-dependent disc radius.** The beam contracts with
    frequency. Currently `disc_radius_deg` is a single scalar; an array
    or callable would let us shrink the disc for higher-frequency
    channels and save compute.
7.  **Solid-angle cosine-projection correction.** `beam_solid_angle`
    uses the flat-tangent-plane sum `d_omega = (d_rad)^2`. At the
    +/-6 deg beam edge this is biased by ~0.6% versus the proper
    `1/sqrt(1 - l^2 - m^2)` factor.

### Performance / engineering

8.  **MPI backend.** A `simeer._parallel.map_samples_mpi` alongside the
    joblib one, partitioning the time axis just like
    `limTOD.mpiutil.partition_list_mpi`. Useful for cluster runs that
    span more than one node. The batch-construction logic should move
    from `integrate_tod` into `_parallel.map_samples` as a prerequisite.
9.  **Memory-mapped beam loading.** For multi-antenna beams that exceed
    RAM, build a streaming loader that mmaps each `(freq, m, l)` slab
    on demand. NPZ archives need to be extracted to plain `.npy` first;
    a helper `MeerKLASSBeam.extract_to_npy(...)` would automate this.
10. **GPU / JAX backend.** The bilinear + sum is a perfect fit for
    `jax.numpy`; a single GPU should run a full TOD in seconds.
11. **Caching across antennas.** When simulating per-antenna TODs (not
    just array_average), the `(l, m)` weights are shared across
    antennas at the same pointing -- only the beam cube changes.
    Multi-antenna driver that reuses weights would amortise the
    projection/disc work.

### Quality of life

12. **Cross-check with the existing `primary_beam.py`.** Verify that
    `MeerKLASSBeam.evaluate(...)` agrees with the original
    `PrimaryBeam.get_beam_gain(...)` to machine precision on a battery
    of (`l`, `m`, `freq`) test points, then deprecate the old class.
13. **Map-making forward operator.** Expose a `simeer.mapmaking` module
    that produces a sparse operator matching limTOD's
    `HPW_mapmaking` conventions, but built from the disc-based forward
    model.
14. **Sky-frequency interpolation.** Today the sky cube must be on the
    output frequency grid. Add a `sky_freq_MHz` argument that triggers
    linear or cubic interpolation to the requested channels. The
    `sky_freq_indices` kwarg of `integrate_sample` already supports
    this internally.
15. **Composition-based `SimeerTODSim`.** Replace inheritance with a
    thin wrapper around a `limTOD.TODSim` instance to fully decouple
    Simeer from limTOD's `__init__` contract.
16. **Self-rotation (`chi`) for Stokes I cross-check.** Stokes I is
    invariant under antenna self-rotation in the ideal case; add a test
    that confirms the expected behaviour.
17. **Type-checked public API.** A `py.typed` marker + `mypy --strict`
    pass on the public modules.
18. **Conda-forge / PyPI release** once v0.1 stabilises.

## License

MIT. See [LICENSE](LICENSE)
for the full text.

## Author

**Zheng Zhang** &lt;zheng.zhang@manchester.ac.uk&gt; -- University of
Manchester. Part of the MeerKLASS analysis toolchain; see also
[`limTOD`](https://github.com/zzhang0123/limTOD) and
[`TIBEC`](https://github.com/zzhang0123/TIBEC).
