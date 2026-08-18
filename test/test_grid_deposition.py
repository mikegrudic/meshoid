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


def test_multi_matches_single_serial():
    from meshoid.grid_deposition import GridSurfaceDensityMulti

    f, x, h, center, size, res = _setup()
    F = np.column_stack([f, 2 * f, f * f, 0 * f])
    multi = GridSurfaceDensityMulti(F, x, h, center, size, res, parallel=False)
    for c in range(F.shape[1]):
        single = GridSurfaceDensity(F[:, c], x, h, center, size, res, parallel=False)
        assert np.allclose(multi[:, :, c], single, rtol=1e-12, atol=0)
    assert np.all(multi[:, :, 3] == 0)  # zero channel stays exactly zero (no channel bleed)


def test_multi_parallel_matches_serial():
    from meshoid.grid_deposition import GridSurfaceDensityMulti

    f, x, h, center, size, res = _setup()
    F = np.column_stack([f, f * f])
    g_serial = GridSurfaceDensityMulti(F, x, h, center, size, res, parallel=False)
    g_parallel = GridSurfaceDensityMulti(F, x, h, center, size, res, parallel=True)
    assert np.allclose(g_serial, g_parallel)


def test_multi_single_channel_matches_1d():
    from meshoid.grid_deposition import GridSurfaceDensityMulti

    f, x, h, center, size, res = _setup()
    multi = GridSurfaceDensityMulti(f[:, None], x, h, center, size, res, parallel=False)
    assert multi.shape == (res, res, 1)
    single = GridSurfaceDensity(f, x, h, center, size, res, parallel=False)
    assert np.allclose(multi[:, :, 0], single, rtol=1e-12, atol=0)


def test_multi_periodic_matches_single():
    """Periodic multi-field deposition must match stacked single-field calls,
    for the generic core (k=2) and the k=3 specialization alike."""
    from meshoid.grid_deposition import GridSurfaceDensityMulti

    f, x, h, center, size, res = _setup()
    for k in (2, 3):
        F = np.column_stack([(c + 1) * f for c in range(k)])
        for parallel in (False, True):
            multi = GridSurfaceDensityMulti(F, x, h, center, size, res, 1.0, parallel=parallel)
            for c in range(k):
                single = GridSurfaceDensity(
                    F[:, c], x, h, center, size, res, box_size=1.0, parallel=False
                )
                assert np.allclose(multi[:, :, c], single, rtol=1e-12, atol=0)


def test_multi_periodic_differs_from_nonperiodic():
    """Guards against box_size being silently ignored: wrapping mass in from
    across the boundary must change the map."""
    from meshoid.grid_deposition import GridSurfaceDensityMulti

    f, x, h, center, size, res = _setup()
    F = np.column_stack([f, 2 * f])
    periodic = GridSurfaceDensityMulti(F, x, h, center, size, res, 1.0)
    free = GridSurfaceDensityMulti(F, x, h, center, size, res)
    assert periodic.sum() > free.sum() * 1.01


def test_multi_accepts_box_size_none():
    from meshoid.grid_deposition import GridSurfaceDensityMulti

    f, x, h, center, size, res = _setup()
    F = np.column_stack([f, 2 * f])
    assert np.allclose(
        GridSurfaceDensityMulti(F, x, h, center, size, res, None),
        GridSurfaceDensityMulti(F, x, h, center, size, res, -1),
    )


def test_multi_rejects_1d_weights():
    import pytest
    from meshoid.grid_deposition import GridSurfaceDensityMulti

    f, x, h, center, size, res = _setup()
    with pytest.raises(ValueError):
        GridSurfaceDensityMulti(f, x, h, center, size, res)


def test_multi_k3_specialized_matches_single():
    # k=3 dispatches to GridSurfaceDensity3_core, not the generic core
    from meshoid.grid_deposition import GridSurfaceDensityMulti

    f, x, h, center, size, res = _setup()
    F = np.column_stack([f, 2 * f, f * f])
    for parallel in (False, True):
        multi = GridSurfaceDensityMulti(F, x, h, center, size, res, parallel=parallel)
        assert multi.shape == (res, res, 3)
        for c in range(3):
            single = GridSurfaceDensity(F[:, c], x, h, center, size, res, parallel=False)
            assert np.allclose(multi[:, :, c], single, rtol=1e-12, atol=0)
