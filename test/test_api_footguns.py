"""Tests for API least-surprise fixes:

  #1 DepositToGrid no longer advertises a silently-ignored `weights` argument
  #2 MassWeightedAverage(masses=None) defaults to the particle masses
  #3 particle_mask no longer double-indexes the masses (crashed for high indices)
  #4 multigrid resolution check raises a real ValueError, not a bare string
"""

import numpy as np
import pytest
from meshoid import Meshoid
from meshoid.grid_deposition import GridSurfaceDensityMultigrid


def _mesh(seed=0, N=2000):
    rng = np.random.default_rng(seed)
    x = rng.random((N, 3))
    m = np.full(N, 1.0 / N)
    return Meshoid(x, m=m), x, m


def test_deposit_to_grid_rejects_removed_weights_kwarg():
    """The dead `weights` parameter is gone: passing it errors loudly instead of
    being silently ignored."""
    M, x, m = _mesh()
    with pytest.raises(TypeError):
        M.DepositToGrid(m, weights=np.ones(len(m)), size=1.0, center=np.array([0.5, 0.5, 0.5]), res=16)
    # the normal call still works
    g = M.DepositToGrid(m, size=1.0, center=np.array([0.5, 0.5, 0.5]), res=16)
    assert np.all(np.isfinite(g))


def test_mass_weighted_average_defaults_masses_to_particle_mass():
    """With no `masses` given, MassWeightedAverage weights by particle mass
    instead of crashing on `f * None`. A constant field averages to itself."""
    M, x, m = _mesh()
    c = 3.5
    avg = M.MassWeightedAverage(np.full(len(m), c), size=1.0, center=np.array([0.5, 0.5, 0.5]), res=48)
    finite = np.isfinite(avg)
    assert finite.any()
    assert np.allclose(avg[finite], c)


def test_particle_mask_high_indices_does_not_crash():
    """A mask selecting high-index particles used to crash with an IndexError
    from double-masking self.m. It must now work and return subset-length data."""
    rng = np.random.default_rng(1)
    x = rng.random((300, 3))
    m = np.full(300, 1.0 / 300)
    mask = np.zeros(300, bool)
    mask[250:] = True  # 50 high-index particles
    M = Meshoid(x, m=m, particle_mask=mask)
    dens = M.Density()
    assert dens.shape == (50,)
    assert np.all(np.isfinite(dens)) and np.all(dens > 0)


def test_particle_mask_matches_unmasked_subset():
    """Density for masked particles equals the corresponding entries of the
    full unmasked computation (the mask only selects *which* particles are
    reported, using the full set as neighbors)."""
    rng = np.random.default_rng(2)
    x = rng.random((400, 3))
    m = np.full(400, 1.0 / 400)
    full = Meshoid(x, m=m).Density()
    idx = np.arange(300, 400)
    mask = np.zeros(400, bool)
    mask[idx] = True
    masked = Meshoid(x, m=m, particle_mask=mask).Density()
    assert np.allclose(masked, full[idx])


def test_multigrid_bad_resolution_raises_valueerror():
    """Non-power-of-2 multigrid resolution raises ValueError with the intended
    message, not `TypeError: exceptions must derive from BaseException`."""
    rng = np.random.default_rng(3)
    x = rng.random((200, 3))
    h = np.full(200, 0.1)
    f = np.full(200, 0.1)
    with pytest.raises(ValueError, match="power of 2"):
        GridSurfaceDensityMultigrid(f, x, h, np.zeros(3), 1.0, res=100)
