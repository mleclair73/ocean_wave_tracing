"""Benchmark and validity harness for the wave-ray solver.

Designed to be re-run across commits to compare throughput and to catch
regressions in the physics. The validity scenario is an idealised
deep-water, zero-current basin where the analytical answer is exact:
each ray travels in a straight line at the deep-water group velocity
cg = g*T / (4*pi).
"""
import time
import numpy as np
import pytest

from ocean_wave_tracing import Wave_tracing


# --- Scenario --------------------------------------------------------------

# Kept small enough to run fast in CI but large enough for the per-step
# work to dominate Python overhead. Bumping nb_wave_rays or nt is the
# right knob for stress-testing.
BENCH_NX = 64
BENCH_NY = 64
BENCH_NT = 200
BENCH_T = 600.0
BENCH_DX = 100.0
BENCH_DY = 100.0
BENCH_NB_RAYS = 200
BENCH_DEPTH = 1.0e5  # deep water
BENCH_WAVE_PERIOD = 10.0
G = 9.81


def _build_scenario(nb_wave_rays=BENCH_NB_RAYS, nt=BENCH_NT):
    nx, ny = BENCH_NX, BENCH_NY
    U = np.zeros((nx, ny))
    V = np.zeros((nx, ny))
    d = np.ones((ny, nx)) * BENCH_DEPTH

    X0, XN = 0.0, (nx - 1) * BENCH_DX
    Y0, YN = 0.0, (ny - 1) * BENCH_DY

    wt = Wave_tracing(
        U, V, nx, ny, nt, BENCH_T,
        BENCH_DX, BENCH_DY,
        nb_wave_rays, X0, XN, Y0, YN,
        temporal_evolution=False, T0=None, d=d,
    )
    wt.set_initial_condition(
        wave_period=BENCH_WAVE_PERIOD, theta0=0.0, incoming_wave_side='left'
    )
    return wt


def _analytical_deep_water_cg(T):
    return G * T / (4.0 * np.pi)


# --- Validity --------------------------------------------------------------

def test_solver_validity_deep_water_straight_line():
    """Deep-water, zero-current rays should travel in a straight line at
    the analytical group velocity to within a tight tolerance.
    """
    wt = _build_scenario()
    wt.solve()

    cg_expected = _analytical_deep_water_cg(BENCH_WAVE_PERIOD)
    # Final x-position predicted analytically.
    x_final_expected = wt.ray_x[:, 0] + cg_expected * BENCH_T

    x_final = wt.ray_x[:, -1]
    y_final = wt.ray_y[:, -1]
    y_initial = wt.ray_y[:, 0]

    # Theta0 = 0 -> y should be unchanged.
    np.testing.assert_allclose(y_final, y_initial, atol=1e-6)
    # x should advance by cg*T. Allow 0.5% for numerical integration error.
    np.testing.assert_allclose(x_final, x_final_expected, rtol=5e-3)


# --- Throughput ------------------------------------------------------------

def test_solver_throughput(capsys):
    """Times the solver. Not a pass/fail gate by default; prints a
    rays*steps/sec figure so we can compare across commits.

    Run with `-s` to see the print output:
        pytest tests/test_solver_benchmark.py::test_solver_throughput -s
    """
    wt = _build_scenario()

    t0 = time.perf_counter()
    wt.solve()
    elapsed = time.perf_counter() - t0

    work = BENCH_NB_RAYS * (BENCH_NT - 1)
    throughput = work / elapsed

    with capsys.disabled():
        print(
            f"\n[bench] rays={BENCH_NB_RAYS} nt={BENCH_NT} "
            f"elapsed={elapsed:.3f}s "
            f"throughput={throughput:,.0f} ray-steps/s"
        )

    # Soft sanity floor: an absurdly slow run (>60s) means something
    # structural broke, not a perf regression.
    assert elapsed < 60.0, f"Solver took {elapsed:.1f}s, far above any expected wall time"


@pytest.mark.parametrize("nb_rays,nt", [(50, 100), (200, 200), (500, 200)])
def test_solver_scaling(nb_rays, nt, capsys):
    """Throughput at a few problem sizes, useful for spotting per-step
    overheads that scale poorly with ray count."""
    wt = _build_scenario(nb_wave_rays=nb_rays, nt=nt)
    t0 = time.perf_counter()
    wt.solve()
    elapsed = time.perf_counter() - t0

    work = nb_rays * (nt - 1)
    with capsys.disabled():
        print(
            f"\n[bench-scale] rays={nb_rays} nt={nt} "
            f"elapsed={elapsed:.3f}s "
            f"throughput={work / elapsed:,.0f} ray-steps/s"
        )
