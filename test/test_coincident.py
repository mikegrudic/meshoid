"""Regression tests for issue #2: coincident (repeated) coordinates.

A particle having one or more neighbors at exactly zero distance must not break
the smoothing-length solve. Historically the neighbor-count root-find would
collapse the smoothing length to zero once ~des_ngb/norm points coincided
(as few as 3 in 3D), producing zero smoothing lengths and infinite densities.
"""

import numpy as np
import pytest
from meshoid import Meshoid
from meshoid.kernel_density import HsmlIter
from scipy.spatial import KDTree


def _hsml(x, des_ngb=32):
    d, _ = KDTree(x).query(x, min(round(des_ngb * 1.5), len(x)))
    return HsmlIter(d, des_ngb=des_ngb, dim=3, error_norm=1e-13)


@pytest.mark.parametrize("n_dup", [2, 3, 5, 11, 33, 47])
def test_coincident_cluster_gives_finite_positive_hsml(n_dup):
    """A cluster of n_dup exactly-coincident points (spanning below and above
    des_ngb) must yield finite, strictly-positive smoothing lengths."""
    rng = np.random.default_rng(0)
    x = rng.random((500, 3))
    x[10 : 10 + n_dup] = x[10]  # stack n_dup points on one location
    h = _hsml(x)
    assert np.all(np.isfinite(h)), "non-finite smoothing length"
    assert np.all(h > 0), f"zero smoothing length with {n_dup} coincident points"


def test_coincident_hsml_is_physically_sane():
    """The coincident particles should get a smoothing length comparable to the
    local interparticle spacing, not something tiny or huge."""
    rng = np.random.default_rng(1)
    x = rng.random((2000, 3))
    x[100:120] = x[100]  # 20 coincident (> des_ngb/norm, would have collapsed)
    h = _hsml(x)
    typical = np.median(h)
    hc = h[100]
    assert 0.1 * typical < hc < 10 * typical


def test_unique_data_unchanged():
    """For data with no exact duplicates the solve is untouched: it still
    reproduces des_ngb effective neighbors."""
    from meshoid.kernel_density import kernel

    rng = np.random.default_rng(2)
    x = rng.random((2000, 3))
    des_ngb, norm = 32, 32.0 / 3
    d, _ = KDTree(x).query(x, 48)
    h = _hsml(x, des_ngb)
    # effective neighbor number n_ngb = norm * sum kernel(d/h) should equal des_ngb
    n_eff = norm * np.array([np.sum([kernel(dij / hi) for dij in di]) for di, hi in zip(d[:50], h[:50])])
    assert np.allclose(n_eff, des_ngb, atol=1e-3)


def test_meshoid_density_finite_with_duplicates():
    """End-to-end: Meshoid.Density()/SmoothingLength() must be finite and
    positive even with several duplicate clusters."""
    rng = np.random.default_rng(3)
    x = rng.random((3000, 3))
    x[100:140] = x[0]      # 41 coincident
    x[200:205] = x[199]    # 5 coincident
    x[300] = x[301]        # a single pair
    m = np.full(len(x), 1.0 / len(x))
    M = Meshoid(x, m=m)
    assert np.all(np.isfinite(M.Density()))
    assert np.all(M.SmoothingLength() > 0)


def test_deposit_to_grid_with_duplicates_conserves_mass():
    """The downstream grid deposition must not blow up and must conserve mass."""
    rng = np.random.default_rng(4)
    x = rng.random((3000, 3))
    x[100:140] = x[0]  # heavy coincident cluster
    m = np.full(len(x), 1.0 / len(x))
    M = Meshoid(x, m=m)
    res = 32
    g = M.DepositToGrid(m, size=1.0, center=np.array([0.5, 0.5, 0.5]), res=res)
    assert np.all(np.isfinite(g))
    # GridDensity is per-particle mass-conserving; non-periodic cell size is size/(res-1)
    dx = 1.0 / (res - 1)
    assert np.isclose(g.sum() * dx**3, m.sum(), rtol=1e-9)
