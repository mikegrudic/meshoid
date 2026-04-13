import numpy as np
from meshoid.grid_deposition import GridSurfaceDensity, GridAverage


def _setup(seed=0, N=5000):
    rng = np.random.default_rng(seed)
    x = rng.random((N, 3))
    h = 0.05 + 0.05 * rng.random(N)
    f = rng.random(N)
    center = np.array([0.5, 0.5, 0.5])
    size = 1.0
    res = 64
    return f, x, h, center, size, res


def test_surface_density_parallel_matches_serial():
    f, x, h, center, size, res = _setup()
    g_serial = GridSurfaceDensity(f, x, h, center, size, res, parallel=False)
    g_parallel = GridSurfaceDensity(f, x, h, center, size, res, parallel=True)
    assert np.allclose(g_serial, g_parallel)


def test_surface_density_conservative_parallel_matches_serial():
    f, x, h, center, size, res = _setup()
    g_serial = GridSurfaceDensity(f, x, h, center, size, res, parallel=False, conservative=True)
    g_parallel = GridSurfaceDensity(f, x, h, center, size, res, parallel=True, conservative=True)
    assert np.allclose(g_serial, g_parallel)


def test_surface_density_conservative_conserves_mass():
    f, x, h, center, size, res = _setup()
    dx = size / (res - 1)
    g = GridSurfaceDensity(f, x, h, center, size, res, conservative=True)
    assert np.isclose(g.sum() * dx * dx, f.sum())


def test_grid_average_constant_field():
    """For a constant field f=c, GridAverage should return c at every populated cell."""
    f, x, h, center, size, res = _setup()
    f_const = np.full_like(f, 3.5)
    ga = GridAverage(f_const, x, h, center, size, res)
    finite = np.isfinite(ga)
    assert finite.any()
    assert np.allclose(ga[finite], 3.5)
