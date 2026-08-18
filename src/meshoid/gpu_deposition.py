"""CUDA implementation of GridSurfaceDensity.

Mirrors the njit core in grid_deposition.py: scatter deposition, one thread per
particle, atomic adds into a device grid. Arithmetic is done in the same order
as the CPU core, so the two agree to summation-order roundoff.

Optional backend -- nothing else in meshoid imports this module, so meshoid
works normally without a CUDA toolchain.
"""

import math

import numpy as np
from numba import cuda, float32, float64

SUPPORTED_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))


def cuda_available():
    """Whether a usable CUDA device is present."""
    try:
        return cuda.is_available()
    except Exception:  # driver/toolkit mismatch, no device, etc.
        return False


def check_dtype(dtype):
    """Normalize and validate a deposition precision."""
    dtype = np.dtype(dtype)
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"dtype must be float32 or float64, got {dtype}")
    return dtype


def _make_gsd_kernel(T):
    """Compile a deposition kernel with all arithmetic in precision T
    (numba.float32 or numba.float64). Literals are cast through T so the
    float32 variant does not silently promote to float64."""
    ONE = T(1.0)
    HALF = T(0.5)
    TWO = T(2.0)
    SIX = T(6.0)
    NORM2D = T(1.8189136353359467)  # 40/(7*pi), unit-integral 2D cubic spline

    @cuda.jit(fastmath=True)
    def gsd_kernel(f, x2d, h, offsets, dx, res, grid):
        i = cuda.grid(1)
        if i >= f.shape[0]:
            return
        hs = h[i]
        hs_sqr = hs * hs
        hinv = ONE / hs
        mh2 = f[i] * hinv * hinv
        n_off = offsets.shape[0]

        for a in range(n_off):
            xs0 = x2d[i, 0] + offsets[a]
            gxmin = max(int((xs0 - hs) / dx + ONE), 0)
            gxmax = min(int((xs0 + hs) / dx), res - 1)
            if gxmin > gxmax:
                continue
            for b in range(n_off):
                xs1 = x2d[i, 1] + offsets[b]
                gymin = max(int((xs1 - hs) / dx + ONE), 0)
                gymax = min(int((xs1 + hs) / dx), res - 1)
                if gymin > gymax:
                    continue
                for gx in range(gxmin, gxmax + 1):
                    delta_x = xs0 - T(gx) * dx
                    delta_x_sqr = delta_x * delta_x
                    for gy in range(gymin, gymax + 1):
                        delta_y = xs1 - T(gy) * dx
                        r_sqr = delta_x_sqr + delta_y * delta_y
                        if r_sqr > hs_sqr:
                            continue
                        q = math.sqrt(r_sqr) * hinv
                        if q <= HALF:
                            w = ONE - SIX * q * q * (ONE - q)
                        elif q <= ONE:
                            aa = ONE - q
                            w = TWO * aa * aa * aa
                        else:
                            continue
                        # (w * NORM2D) * mh2 matches kernel2d(q) * mh2 on the
                        # CPU term for term, keeping the two within roundoff
                        cuda.atomic.add(grid, (gx, gy), w * NORM2D * mh2)

    return gsd_kernel


_gsd_kernels = {}


def get_gsd_kernel(dtype):
    """Return the compiled deposition kernel for float32 or float64."""
    dtype = check_dtype(dtype)
    if dtype not in _gsd_kernels:
        T = float32 if dtype == np.dtype(np.float32) else float64
        _gsd_kernels[dtype] = _make_gsd_kernel(T)
    return _gsd_kernels[dtype]


def prepare_gsd_inputs(f, x, h, center, size, res, box_size=None, dtype=np.float64):
    """Host-side preparation shared by the wrapper and the benchmarks.

    Returns (f, x2d, h, offsets, dx) as contiguous arrays of dtype. The center
    subtraction happens in the input precision before any downcast, so float32
    mode does not lose precision to large absolute coordinates.
    """
    dtype = check_dtype(dtype)
    center = np.asarray(center)
    # grid conventions of the CPU core: periodic tiles res cells of size/res,
    # non-periodic uses the node convention size/(res-1)
    if box_size is not None and box_size > 0:
        dx = size / res
        offsets = np.array([-box_size, 0.0, box_size])
    else:
        dx = size / (res - 1)
        offsets = np.array([0.0])
    x2d = x[:, :2] - center[:2] + size / 2
    return (
        np.ascontiguousarray(f, dtype=dtype),
        np.ascontiguousarray(x2d, dtype=dtype),
        np.ascontiguousarray(h, dtype=dtype),
        offsets.astype(dtype),
        dtype.type(dx),
    )


def GridSurfaceDensityMulti_cuda(
    f, x, h, center, size, res=128, box_size=None, dtype=np.float64, threads_per_block=256
):
    """Deposit k fields, one launch of the single-field kernel per field, with
    the positions and smoothing lengths uploaded once and reused.

    Unlike the CPU path there is no fused k-field kernel: fusing amortizes the
    kernel arithmetic, which is not what this kernel is bound by, and it
    measured no faster than separate launches.

    f is (N,k); returns (res,res,k), where [:,:,c] is the surface density of
    f[:,c]. Other arguments are as GridSurfaceDensity_cuda.
    """
    f = np.asarray(f)
    if f.ndim != 2:
        raise ValueError(f"f must be an (N,k) weights array, got shape {f.shape}")
    if len(f) != len(x) or len(f) != len(h):
        raise ValueError(f"f, x, h must have matching lengths, got {len(f)}, {len(x)}, {len(h)}")
    dtype = check_dtype(dtype)
    _, x2d, hd, offsets, dx = prepare_gsd_inputs(f[:, 0], x, h, center, size, res, box_size, dtype)
    kernel = get_gsd_kernel(dtype)

    x_dev = cuda.to_device(x2d)
    h_dev = cuda.to_device(hd)
    off_dev = cuda.to_device(offsets)
    blocks = (len(hd) + threads_per_block - 1) // threads_per_block

    zeros = np.zeros((res, res), dtype=dtype)
    out = np.empty((res, res, f.shape[1]), dtype=dtype)
    for c in range(f.shape[1]):
        f_dev = cuda.to_device(np.ascontiguousarray(f[:, c], dtype=dtype))
        grid_dev = cuda.to_device(zeros)
        kernel[blocks, threads_per_block](f_dev, x_dev, h_dev, off_dev, dx, res, grid_dev)
        out[:, :, c] = grid_dev.copy_to_host()
    return out


def GridSurfaceDensity_cuda(
    f, x, h, center, size, res=128, box_size=None, dtype=np.float64, threads_per_block=256
):
    """CUDA equivalent of GridSurfaceDensity (non-conservative deposition).

    Same arguments and conventions as grid_deposition.GridSurfaceDensity, plus:

    dtype: np.float64 (default) or np.float32, setting the precision of the
        device arrays, the kernel arithmetic and the accumulator. float32 is
        ~10x faster on consumer GPUs (which run FP64 at 1/64 rate) and about
        the same speed on datacenter parts; its deposition error is ~1e-6
        relative, well below plotting accuracy.

    Returns a (res,res) array of the chosen dtype.
    """
    if len(f) != len(x) or len(f) != len(h):
        raise ValueError(f"f, x, h must have matching lengths, got {len(f)}, {len(x)}, {len(h)}")
    fd, x2d, hd, offsets, dx = prepare_gsd_inputs(f, x, h, center, size, res, box_size, dtype)
    kernel = get_gsd_kernel(dtype)

    f_dev = cuda.to_device(fd)
    x_dev = cuda.to_device(x2d)
    h_dev = cuda.to_device(hd)
    off_dev = cuda.to_device(offsets)
    grid_dev = cuda.to_device(np.zeros((res, res), dtype=np.dtype(dtype)))

    n = len(fd)
    blocks = (n + threads_per_block - 1) // threads_per_block
    kernel[blocks, threads_per_block](f_dev, x_dev, h_dev, off_dev, dx, res, grid_dev)
    return grid_dev.copy_to_host()
