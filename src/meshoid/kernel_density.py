from numba import (
    vectorize,
    float32,
    float64,
    njit,
    jit,
    prange,
    get_num_threads,
)
import numpy as np
from scipy.interpolate import RectBivariateSpline


@njit(fastmath=True, parallel=True)
def HsmlIter(neighbor_dists, dim=3, error_norm=1e-6):
    """
    Performs the iteration to get smoothing lengths, according to Eq. 26 in Hopkins 2015 MNRAS 450.

    Arguments:
    neighbor_dists: (N, Nngb) array of distances from particles to their Nngb nearest neighbors

    Keyword arguments:
    dim - Dimensionality (default: 3)
    error_norm - Tolerance in the particle number density to stop iterating at (default: 1e-6)
    """
    if dim == 3:
        norm = 32.0 / 3
    elif dim == 2:
        norm = 40.0 / 7
    else:
        norm = 8.0 / 3
    N, des_ngb = neighbor_dists.shape
    hsml = np.zeros(N)
    n_ngb = 0.0
    bound_coeff = 1.0 / (1 - (2 * norm) ** (-1.0 / 3))
    for i in prange(N):
        upper = neighbor_dists[i, des_ngb - 1] * bound_coeff
        lower = neighbor_dists[i, 1]
        error = 1e100
        while error > error_norm:
            h = (upper + lower) / 2
            n_ngb = 0.0
            q = 0.0
            for j in range(des_ngb):
                q = neighbor_dists[i, j] / h
                if q <= 0.5:
                    n_ngb += 1 - 6 * q**2 + 6 * q**3
                elif q <= 1.0:
                    n_ngb += 2 * (1 - q) ** 3
            n_ngb *= norm
            if n_ngb > des_ngb:
                upper = h
            else:
                lower = h
            error = np.fabs(n_ngb - des_ngb)
        hsml[i] = h
    return hsml


@njit(fastmath=True, error_model="numpy")
def kernel2d(q):
    """Returns the normalized 2D spline kernel evaluated at radius q"""
    if q <= 0.5:
        kernel = 1 - 6 * q * q * (1 - q)
    elif q <= 1.0:
        a = 1 - q
        kernel = 2 * a * a * a
    return kernel * 1.8189136353359467


@njit(fastmath=True, error_model="numpy")
def kernel3d(q):
    """Returns the normalized 3D spline kernel evaluated at radius q"""
    if q <= 0.5:
        kernel = 1 - 6 * q * q * (1 - q)
    elif q <= 1.0:
        a = 1 - q
        kernel = 2 * a * a * a
    return kernel * 2.546479089470325


@vectorize([float32(float32), float64(float64)])
def Kernel(q):
    """
    Un-normalized cubic-spline kernel function

    Arguments:
        q - array containing radii at which to evaluate the kernel,
        scaled to the kernel support radius (between 0 and 1)
    """
    if q <= 0.5:
        return 1 - 6 * q**2 + 6 * q**3
    elif q <= 1.0:
        return 2 * (1 - q) ** 3
    else:
        return 0.0
