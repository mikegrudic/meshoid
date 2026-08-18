"""Agreement between the CUDA and CPU surface-density deposition.

The two run the same algorithm in the same operation order, so they differ only
by atomic summation order. With ~90 deposits per pixel in this setup that bounds
the relative difference at ~90*eps: ~2e-14 in float64 and ~1e-5 in float32.
Tolerances below sit an order of magnitude above the measured worst case
(4e-15 and 7e-6) and far below any algorithmic discrepancy.
"""

import numpy as np
import pytest

from meshoid.grid_deposition import GridSurfaceDensity

try:
    from meshoid.gpu_deposition import cuda_available

    HAVE_CUDA = cuda_available()
except ImportError:  # numba.cuda not installed
    HAVE_CUDA = False

requires_cuda = pytest.mark.skipif(not HAVE_CUDA, reason="no CUDA device available")

RTOL_F64 = 1e-12
RTOL_F32 = 1e-4


def _setup(seed=0, N=5000):
    rng = np.random.default_rng(seed)
    x = rng.random((N, 3))
    h = 0.05 + 0.05 * rng.random(N)
    f = rng.random(N)
    return f, x, h, np.array([0.5, 0.5, 0.5]), 1.0, 64


@requires_cuda
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("box_size", [None, 1.0])
def test_cuda_f64_matches_cpu_pixelwise(seed, box_size):
    f, x, h, center, size, res = _setup(seed)
    cpu = GridSurfaceDensity(f, x, h, center, size, res, box_size=box_size)
    gpu = GridSurfaceDensity(f, x, h, center, size, res, box_size=box_size, backend="cuda")
    assert gpu.shape == cpu.shape
    assert np.allclose(gpu, cpu, rtol=RTOL_F64, atol=RTOL_F64 * cpu.max())


@requires_cuda
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("box_size", [None, 1.0])
def test_cuda_f32_matches_cpu_pixelwise(seed, box_size):
    f, x, h, center, size, res = _setup(seed)
    cpu = GridSurfaceDensity(f, x, h, center, size, res, box_size=box_size)
    gpu = GridSurfaceDensity(
        f, x, h, center, size, res, box_size=box_size, backend="cuda", dtype=np.float32
    )
    assert np.allclose(gpu, cpu, rtol=RTOL_F32, atol=1e-5 * cpu.max())


@requires_cuda
@pytest.mark.parametrize("box_size", [None, 1.0])
def test_cuda_support_footprint_is_identical(box_size):
    """Same pixels are touched -- catches off-by-one in the bounding box, which
    a tolerance check on populated pixels alone would miss."""
    f, x, h, center, size, res = _setup()
    cpu = GridSurfaceDensity(f, x, h, center, size, res, box_size=box_size)
    for dtype in (np.float64, np.float32):
        gpu = GridSurfaceDensity(
            f, x, h, center, size, res, box_size=box_size, backend="cuda", dtype=dtype
        )
        assert np.array_equal(gpu > 0, cpu > 0)


@requires_cuda
def test_cuda_conserves_total_within_roundoff():
    f, x, h, center, size, res = _setup()
    cpu = GridSurfaceDensity(f, x, h, center, size, res)
    for dtype, rtol in ((np.float64, 1e-12), (np.float32, 1e-5)):
        gpu = GridSurfaceDensity(f, x, h, center, size, res, backend="cuda", dtype=dtype)
        assert np.isclose(gpu.sum(), cpu.sum(), rtol=rtol)


@requires_cuda
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cuda_returns_requested_precision(dtype):
    f, x, h, center, size, res = _setup()
    gpu = GridSurfaceDensity(f, x, h, center, size, res, backend="cuda", dtype=dtype)
    assert gpu.dtype == np.dtype(dtype)


@requires_cuda
@pytest.mark.parametrize(
    "dtype, rtol, atol_frac", [(np.float64, RTOL_F64, RTOL_F64), (np.float32, RTOL_F32, 1e-5)]
)
def test_meshoid_surface_density_cuda_matches_cpu(dtype, rtol, atol_frac):
    """The Meshoid method plumbs backend/dtype through, including its own
    smoothing-length clipping."""
    from meshoid import Meshoid

    f, x, h, center, size, res = _setup()
    M = Meshoid(x, f, h)
    cpu = M.SurfaceDensity(f, center=center, size=size, res=res)
    gpu = M.SurfaceDensity(f, center=center, size=size, res=res, backend="cuda", dtype=dtype)
    assert gpu.dtype == np.dtype(dtype)
    assert np.allclose(gpu, cpu, rtol=rtol, atol=atol_frac * cpu.max())


@requires_cuda
def test_meshoid_surface_density_rejects_cpu_float32():
    from meshoid import Meshoid

    f, x, h, center, size, res = _setup()
    M = Meshoid(x, f, h)
    with pytest.raises(ValueError, match="float64"):
        M.SurfaceDensity(f, center=center, size=size, res=res, dtype=np.float32)


@requires_cuda
def test_cuda_rejects_conservative():
    f, x, h, center, size, res = _setup()
    with pytest.raises(NotImplementedError):
        GridSurfaceDensity(f, x, h, center, size, res, backend="cuda", conservative=True)


def test_cpu_rejects_float32():
    """float32 is cuda-only; the cpu cores always accumulate in float64."""
    f, x, h, center, size, res = _setup()
    with pytest.raises(ValueError, match="float64"):
        GridSurfaceDensity(f, x, h, center, size, res, dtype=np.float32)


def test_unknown_backend_raises():
    f, x, h, center, size, res = _setup()
    with pytest.raises(ValueError, match="backend"):
        GridSurfaceDensity(f, x, h, center, size, res, backend="opencl")


def test_multigrid_rejects_non_cpu_backend():
    from meshoid import Meshoid

    f, x, h, center, size, res = _setup()
    M = Meshoid(x, f, h)
    with pytest.raises(NotImplementedError):
        M.SurfaceDensity(f, center=center, size=size, res=res, multigrid=True, backend="cuda")


@pytest.mark.skipif(not HAVE_CUDA, reason="no CUDA device available")
def test_invalid_dtype_raises():
    f, x, h, center, size, res = _setup()
    with pytest.raises(ValueError, match="float32 or float64"):
        GridSurfaceDensity(f, x, h, center, size, res, backend="cuda", dtype=np.int32)
