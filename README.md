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

### Using the low-level integrator (no limTOD)

```python
import numpy as np
from simeer import MeerKLASSBeam, integrate_tod

beam = MeerKLASSBeam("MeerKAT_U_band_primary_beam_aa_highres.npz")

# Build the sky model on the beam's frequency grid.
freq_MHz = beam.freq_MHz[::10]      # every 10th channel for this demo
sky = my_sky_function(freq_MHz)     # shape (n_freq, n_pix_sky)

# Telescope pointing.
lst_deg = ...   # shape (n_time,)
az_deg  = ...   # shape (n_time,)
el_deg  = ...   # shape (n_time,)

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
    n_jobs=-1,            # joblib: use all cores
    progress=True,
)
# tod.shape == (n_freq, n_time)
```

### Drop-in replacement for `limTOD.TODSim`

```python
from limTOD.sky_model import GDSM_sky_model
from limTOD.simulator import example_scan
from simeer import MeerKLASSBeam, SimeerTODSim

beam = MeerKLASSBeam("MeerKAT_U_band_primary_beam_aa_highres.npz")

sim = SimeerTODSim(
    beam=beam,
    sky_func=GDSM_sky_model,
    sky_nside=512,
    disc_radius_deg=8.0,
    polarization="HH",
    n_jobs=-1,
)

time_list, az_list = example_scan()
tod, sky_tod, gain_noise = sim.generate_TOD(
    freq_list=beam.freq_MHz[::10],
    time_list=time_list,
    azimuth_deg_list=az_list,
    elevation_deg=41.5,
)
```

Everything outside `simulate_sky_TOD` (gain noise, flicker noise,
white-noise injection, TOD assembly) is inherited unchanged from
`limTOD.TODSim`.

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

## Citation

If you use Simeer in your research, please cite as:

```bibtex
@software{zhang_simeer_2026,
  author  = {Zhang, Zheng},
  title   = {Simeer: optimal MeerKLASS TOD simulator with native (l, m) beam interpolation},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/zzhang0123/Simeer},
  license = {MIT}
}
```

The CITATION.cff file in this repository provides the same metadata in
machine-readable form for GitHub's "Cite this repository" button.
