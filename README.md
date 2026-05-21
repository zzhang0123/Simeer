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
    solid angle `Omega_b(freq)`.

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

Required: `numpy`, `scipy`, `healpy`, `astropy`, `joblib`, `tqdm`.

Optional: `limTOD` (for `SimeerTODSim`), `matplotlib` and `jupyter` for
the notebooks, `pytest` for tests.

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

```
                 1
  T_sky(nu, t) = ----- * integral over sky of  B(l, m, nu) * T(l, m, nu) dOmega
                Omega_b(nu)
```

where:

- `B(l, m, nu)` is the primary beam *power* pattern at frequency `nu`,
  defined in direction-cosine coordinates `(l, m)` centred on the
  pointing;
- `Omega_b(nu) = integral B dOmega` is the beam solid angle, so the
  result is in the same temperature units as the input sky;
- `T(l, m, nu)` is the sky brightness temperature evaluated at the disc
  pixels (Simeer projects HEALPix sky pixels into the antenna-local
  `(l, m)` frame on the fly).

The Sky TOD is the "pure" signal that downstream calibration and
map-making algorithms ultimately try to recover from the measured TOD.
Simeer **only** computes the Sky TOD; gain, 1/f noise, and white-noise
injection live in `limTOD`.

### What is the Full TOD?

The **Full TOD** is the realistic instrument-modulated signal, the
quantity an actual receiver writes to disk:

```
  TOD(nu, t) = G_bg(nu, t) * [1 + G_noise(nu, t)]
               * [T_sky(nu, t) + T_sys_other(nu, t)]
               * [1 + eta(t)]
```

with `G_bg` the background gain pattern, `G_noise` the multiplicative
1/f gain fluctuation, `T_sys_other` other system-temperature components
(CMB, ground spill, receiver), and `eta` additive white noise. The Full
TOD assembly happens inside `limTOD.TODSim.generate_TOD`; Simeer plugs
into it by replacing only the `T_sky(nu, t)` step.

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

The integration tests use the synthetic Gaussian beam constructed via
`synthetic_gaussian_beam(...)` so they run without the real ~30 GB beam
file. The cross-check against `limTOD`'s HEALPix path lives in
`tests/test_against_limtod.py` (planned, see follow-ups).

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
3.  **Boundary-validation cross-check against limTOD.** Add
    `tests/test_against_limtod.py` that takes a symmetric Gaussian beam,
    routes it through both the limTOD SH path and the Simeer disc path
    at matched `nside`, and asserts agreement on a parameter grid that
    includes extreme corners (`el in {15, 89}`, az wrap, disc radius
    vs beam extent ratios).
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

MIT, matching the rest of the MeerKLASS toolchain. See [LICENSE](LICENSE)
for the full text.

## Author

**Zheng Zhang** &lt;zheng.zhang@manchester.ac.uk&gt; -- University of
Manchester. Part of the MeerKLASS analysis toolchain; see also
[`limTOD`](https://github.com/zzhang0123/limTOD) and
[`TIBEC`](https://github.com/zzhang0123/TIBEC).

