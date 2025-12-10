from numba import vectorize, float32, float64, njit, jit, prange, cfunc
import numpy as np
from scipy.integrate import quad
from .periodic import nearest_image


@cfunc("float64(float64)", fastmath=True)
def kernel(q):
    """Cubic-spline kernel function"""
    if q <= 0.5:
        return 1 - 6 * q**2 + 6 * q**3
    elif q <= 1.0:
        return 2 * (1 - q) ** 3
    else:
        return 0


kernel_norm = {dim: 1.0 / quad(lambda x: kernel(x) * np.pi * (2 * x) ** (dim - 1), 0, 1)[0] for dim in range(1, 4)}
kernel_norm[1] *= 0.5


@jit(fastmath=True)
def dkernel(dx, h):
    """Spatial derivative of cubic spline kernel function"""
    dim = dx.shape[-1]
    r = 0
    for k in range(dim):
        r += dx[k] * dx[k]
    r = np.sqrt(r)
    q = r / h
    if q == 0 or q >= 1:
        return np.zeros(dim)
    # chain rule: dk/dx = dk/dq dq/dr dr/dx
    if q <= 0.5:
        dk = -12 * q + 18 * q**2
    elif q <= 1.0:
        dk = -6 * (1 - q) ** 2
    dk /= h
    result = np.empty(dim)
    for k in range(dim):
        result[k] = -dk * dx[k] / r
    return result


@njit(parallel=True, fastmath=True)
def dkernel_ngb_sum(x, h, ngb, boxsize):
    """Neighbor sum of spatial derivative of cubic spline kernel function"""
    N, dim = x.shape
    result = np.zeros_like(x)
    dx = np.empty(dim)
    for i in prange(N):
        x0, h0 = x[i], h[i]
        for n in ngb[i]:
            for k in range(dim):
                dx[k] = x[n, k] - x0[k]
                if boxsize is not None:
                    dx[k] = nearest_image(dx[k], boxsize)
            dk = dkernel(dx, h0)
            for k in range(dim):
                result[i, k] += dk[k]
    return result


@njit(parallel=True, fastmath=True)
def kernel_gradient(f, x, h, ngb, boxsize):
    """SPH-style kernel gradient estimator"""
    N, dim = x.shape
    result = np.zeros_like(x)
    dx = np.empty(dim)
    for i in prange(N):
        x0, h0 = x[i], h[i]
        for n in ngb[i]:
            for k in range(dim):
                dx[k] = x[n, k] - x0[k]
                if boxsize is not None:
                    dx[k] = nearest_image(dx[k], boxsize)
            dk = dkernel(dx, h0)
            for k in range(dim):
                result[i, k] += f[n] * dk[k]
    return result


@jit(fastmath=True, parallel=True)
def HsmlIter(neighbor_dists, des_ngb, dim=3, error_norm=1e-6):
    """
    Performs the iteration to get smoothing lengths, according to Eq. 26 in Hopkins 2015 MNRAS 450.

    Arguments
    ---------
    neighbor_dists: (N, Nngb) array of distances from particles to their Nngb nearest neighbors

    Keyword arguments
    -----------------
    dim - Dimensionality (default: 3)
    error_norm - Tolerance in the particle number density to stop iterating at (default: 1e-6)
    """
    if dim == 3:
        norm = 32.0 / 3
    elif dim == 2:
        norm = 40.0 / 7
    else:
        norm = 8.0 / 3
    N, num_ngb = neighbor_dists.shape
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
            for j in range(num_ngb):
                q = neighbor_dists[i, j] / h
                n_ngb += kernel(q)
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
def Kernel_v(q):
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
