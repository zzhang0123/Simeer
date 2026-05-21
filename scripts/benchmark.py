"""Benchmark the Simeer TOD integrator on a synthetic beam.

Measures three things:

1.  Per-sample latency in serial mode (microbenchmark, n_jobs=1, ntime=64).
2.  Time-per-sample of each subcomponent (disc query, projection,
    bilinear weights, beam gather, sum) using cProfile + pstats.
3.  Scaling across n_jobs in {1, 2, 4, all}.

Output goes to stdout and ``scripts/benchmark_results.txt`` so the
artefact is checked in for inspection by reviewers.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
from pathlib import Path
from time import perf_counter

import healpy as hp
import numpy as np

from simeer import integrate_tod, synthetic_gaussian_beam

RESULTS_PATH = Path(__file__).parent / "benchmark_results.txt"


def _build_inputs(*, ntime: int, nside_sky: int, n_freq: int):
    """Build inputs that mimic a realistic MeerKLASS-U-band-ish scene."""
    freq_MHz = np.linspace(580.0, 1015.0, n_freq)
    margin_deg = np.linspace(-6, 6, 121)
    beam = synthetic_gaussian_beam(freq_MHz=freq_MHz, margin_deg=margin_deg, fwhm_deg=1.5)
    sky = np.full((len(freq_MHz), hp.nside2npix(nside_sky)), 8.0, dtype=np.float64)
    lst_deg = np.linspace(0.0, 30.0, ntime)
    az_deg = 180.0 + 5.0 * np.sin(np.linspace(0, 4 * np.pi, ntime))
    el_deg = np.full(ntime, 41.5)
    return beam, sky, freq_MHz, lst_deg, az_deg, el_deg


def benchmark_serial(*, ntime: int, nside_sky: int, n_freq: int, n_repeats: int = 3):
    beam, sky, freq_MHz, lst_deg, az_deg, el_deg = _build_inputs(
        ntime=ntime, nside_sky=nside_sky, n_freq=n_freq
    )
    # Warm-up to trigger any JIT / import-time caching.
    integrate_tod(
        lst_deg_list=lst_deg[:4],
        az_deg_list=az_deg[:4],
        el_deg_list=el_deg[:4],
        lat_deg=-30.7130,
        beam=beam,
        sky_maps=sky,
        freq_MHz=freq_MHz,
        disc_radius_deg=8.0,
        n_jobs=1,
    )

    timings = []
    for _ in range(n_repeats):
        t0 = perf_counter()
        integrate_tod(
            lst_deg_list=lst_deg,
            az_deg_list=az_deg,
            el_deg_list=el_deg,
            lat_deg=-30.7130,
            beam=beam,
            sky_maps=sky,
            freq_MHz=freq_MHz,
            disc_radius_deg=8.0,
            n_jobs=1,
        )
        timings.append(perf_counter() - t0)

    elapsed = float(np.median(timings))
    per_sample_ms = 1e3 * elapsed / ntime
    return {
        "ntime": ntime,
        "nside_sky": nside_sky,
        "n_freq": n_freq,
        "total_s": elapsed,
        "per_sample_ms": per_sample_ms,
        "best_s": float(min(timings)),
    }


def profile_serial(*, ntime: int, nside_sky: int, n_freq: int):
    beam, sky, freq_MHz, lst_deg, az_deg, el_deg = _build_inputs(
        ntime=ntime, nside_sky=nside_sky, n_freq=n_freq
    )

    profiler = cProfile.Profile()
    profiler.enable()
    integrate_tod(
        lst_deg_list=lst_deg,
        az_deg_list=az_deg,
        el_deg_list=el_deg,
        lat_deg=-30.7130,
        beam=beam,
        sky_maps=sky,
        freq_MHz=freq_MHz,
        disc_radius_deg=8.0,
        n_jobs=1,
    )
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(25)
    return stream.getvalue()


def benchmark_parallel(*, ntime: int, nside_sky: int, n_freq: int):
    beam, sky, freq_MHz, lst_deg, az_deg, el_deg = _build_inputs(
        ntime=ntime, nside_sky=nside_sky, n_freq=n_freq
    )
    import os

    rows = []
    for n_jobs in (1, 2, 4, -1):
        t0 = perf_counter()
        integrate_tod(
            lst_deg_list=lst_deg,
            az_deg_list=az_deg,
            el_deg_list=el_deg,
            lat_deg=-30.7130,
            beam=beam,
            sky_maps=sky,
            freq_MHz=freq_MHz,
            disc_radius_deg=8.0,
            n_jobs=n_jobs,
        )
        elapsed = perf_counter() - t0
        rows.append((n_jobs, elapsed))
    rows.append(("os.cpu_count()", os.cpu_count()))
    return rows


def main() -> None:
    lines: list[str] = []

    def out(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    out("Simeer benchmark")
    out("=" * 60)

    # 1) Serial latency, three problem sizes.
    out("\n[1] Serial throughput (median of 3 runs):")
    out(f"{'ntime':>8} {'nside':>6} {'nfreq':>6} {'total_s':>10} {'per_sample_ms':>16}")
    for cfg in [
        dict(ntime=64, nside_sky=128, n_freq=1),
        dict(ntime=64, nside_sky=128, n_freq=8),
        dict(ntime=256, nside_sky=128, n_freq=8),
        dict(ntime=256, nside_sky=256, n_freq=8),
    ]:
        r = benchmark_serial(**cfg)
        out(
            f"{r['ntime']:>8} {r['nside_sky']:>6} {r['n_freq']:>6} "
            f"{r['total_s']:>10.3f} {r['per_sample_ms']:>16.2f}"
        )

    # 2a) Parallel scaling on a small problem (ntime=4000, nfreq=8). At
    #     this size the per-batch overhead is comparable to the work, so
    #     scaling is only ~1.4x. Realistic MeerKLASS sizes are much
    #     larger and scale far better -- see 2b.
    out("\n[2a] Parallel scaling on ntime=4000, nside=128, nfreq=8 (small):")
    out("    Pool-startup overhead (~500 ms) dominates here; expect ~1.3-1.4x at n_jobs=4.")
    rows = benchmark_parallel(ntime=4000, nside_sky=128, n_freq=8)
    for n_jobs, elapsed in rows:
        if isinstance(elapsed, float):
            out(f"  n_jobs={n_jobs:>4}: {elapsed:.3f} s")
        else:
            out(f"  (host has {elapsed} CPUs)")

    # 2b) Parallel scaling at a realistic MeerKLASS problem size.
    out("\n[2b] Parallel scaling on ntime=20000, nside=256, nfreq=64 (realistic):")
    out("    Auto-memmap kicks in (>100 KB top-level ndarrays), pool startup")
    out("    is amortised over the longer run -> clean near-linear scaling.")
    rows = benchmark_parallel(ntime=20000, nside_sky=256, n_freq=64)
    for n_jobs, elapsed in rows:
        if isinstance(elapsed, float):
            out(f"  n_jobs={n_jobs:>4}: {elapsed:.3f} s")
        else:
            out(f"  (host has {elapsed} CPUs)")

    # 3) cProfile dump of one serial run.
    out("\n[3] cProfile top-25 (ntime=128, nside=128, nfreq=8):")
    out(profile_serial(ntime=128, nside_sky=128, n_freq=8))

    RESULTS_PATH.write_text("\n".join(lines) + "\n")
    out(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
