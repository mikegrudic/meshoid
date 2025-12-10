from meshoid import particle_glass, Meshoid
import numpy as np


def test_glass(N=32**3, tol=1e-2, seed=42):
    """Generates a glass and tests that its RMS density variation is less than specified tolerance"""
    np.random.seed(seed)
    x = particle_glass(N, tol=tol)
    assert Meshoid(x, boxsize=1.0).Density().std() < tol
