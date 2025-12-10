"""Test gradient estimators"""

import numpy as np
import os
from meshoid import Meshoid

test_glass_path = os.path.dirname(os.path.abspath(__file__)) + "/glass_32_tol0.001.npy"


def test_gradients():
    """Tests least-squares gradients by running them on polynomials for which they are supposed to get
    the machine-precision answer"""
    x = np.load(test_glass_path)
    N, dim = x.shape
    identity = np.array(N * [np.eye(dim)])
    zeros = np.zeros_like(identity)
    const = np.ones_like(x)

    # 1st derivatives
    assert np.all(np.isclose(Meshoid(x).D(x), identity))
    assert np.all(np.isclose(Meshoid(x).D(const), zeros))
    assert np.all(np.isclose(Meshoid(x).D(x, order=2), identity))
    assert np.all(np.isclose(Meshoid(x).D(const, order=2), zeros))
    assert np.all(
        np.isclose(Meshoid(x).D(x**2, order=2), 2 * x[:, :, None] * identity)
    )  # 2nd-order weights will get this right too

    # 2nd derivatives
    assert np.all(np.isclose(Meshoid(x).D2(x**2)[:, :, :3], 2 * identity))
    assert np.all(np.isclose(Meshoid(x).D2(x**2)[:, :, 3:], zeros))
    d2x = Meshoid(x).D2(x)
    assert np.all(np.isclose(d2x, np.zeros_like(d2x)))

    for k in range(3):  # cross-terms
        x1, x2 = x[:, k], x[:, (k + 1) % dim]
        assert np.all(np.isclose(Meshoid(x).D2(x1 * x2)[:, k + 3], 1.0))


# for periodic box: coordinate functions are discontinuous across edges, so instead do sinusoids and expect only truncation-error precision
def test_gradients_periodic():
    """Tests gradients in periodic boundary conditions by checking that sinusoids are differentiated to within
    expected truncation-error precision
    """
    x = np.load(test_glass_path)

    f = np.sin(2 * np.pi * x[:, 0])
    df = 2 * np.pi * np.cos(2 * np.pi * x[:, 0])
    assert np.all(np.isclose(Meshoid(x, boxsize=1.0).D(f)[:, 0], df, atol=0.3))
    assert np.all(np.isclose(Meshoid(x, boxsize=1.0).D(f, order=2)[:, 0], df, atol=0.3))
    assert np.all(np.isclose(Meshoid(x, boxsize=1.0).D2(f)[:, 0], -((2 * np.pi) ** 2) * f, atol=2 * np.pi * 0.3))
