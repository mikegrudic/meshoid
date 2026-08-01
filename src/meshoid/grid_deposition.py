from numba import njit, get_num_threads
import numpy as np
from scipy.interpolate import RectBivariateSpline
from .kernel_density import *


# Internal convention for the box_size argument of the njit deposition cores:
# a positive value enables periodic boundary conditions with coordinates assumed
# to run from [0, box_size); any non-positive value (or None normalized to -1 by
# the Python wrappers) disables periodicity.


@njit(fastmath=True, error_model="numpy", cache=True)
def periodic_offsets(box_size):
    """Return the set of periodic image shifts to apply along each periodic axis.

    For a periodic domain we deposit each particle plus its two nearest periodic
    images (shifted by +/- box_size) so that kernels straddling a boundary wrap
    correctly.  For a non-periodic domain (box_size <= 0) only the identity shift
    is returned, so the deposition loops reduce to the ordinary single-image case.
    """
    if box_size > 0:
        return np.array([-box_size, 0.0, box_size])
    return np.array([0.0])


def _normalize_box_size(box_size):
    """Map the user-facing box_size (None or a positive float) to the numeric
    sentinel convention used by the njit cores."""
    if box_size is None:
        return -1.0
    return float(box_size)


def GridSurfaceDensityMultigrid(
    f, x, h, center, size, res=128, box_size=None, N_grid_kernel=8, parallel=True, conservative=False
):
    box_size = _normalize_box_size(box_size)
    if not ((res != 0) and (res & (res - 1) == 0)):
        raise ValueError("Multigrid resolution must be a power of 2")
    res_bins = size / 2 ** np.arange(0, round(np.log2(res) + 1))
    res_bins[0] = np.inf
    res_bins[-1] = 0

    grid = np.zeros((4, 4))
    for i in range(2, len(res_bins) - 1):
        grid = UpsampleGrid(grid)
        Ni = grid.shape[0]
        # bin particles by smoothing length to decide which resolution level they get deposited at
        idx = (h / N_grid_kernel < res_bins[i]) & (h / N_grid_kernel >= res_bins[i + 1])
        if np.any(idx):
            grid += GridSurfaceDensity(
                f[idx],
                x[idx],
                h[idx],
                center,
                size,
                res=Ni,
                box_size=box_size,
                parallel=parallel,
                conservative=conservative,
            )
    return grid


def UpsampleGrid(grid):
    N = grid.shape[0]
    x1 = np.linspace(0.5 / N, 1 - 0.5 / N, N)  # original coords
    x2 = np.linspace(0.25 / N, 1 - 0.25 / N, 2 * N)  # new coords
    return RectBivariateSpline(x1, x1, grid)(x2, x2)  # RectBivariateSpline(x1,


def GridSurfaceDensity(f, x, h, center, size, res=128, box_size=None, parallel=True, conservative=False):
    """
    Computes the surface density of conserved quantity f colocated at positions
    x with smoothing lengths h. E.g. plugging in particle masses would return
    mass surface density. The result is on a Cartesian grid of sightlines, the
    result being the density of quantity f integrated along those sightlines.

    Parameters
    ----------
    f: array_like
        Shape (N,) array of the conserved quantity that you want the surface density of (e.g. particle masses)
    x: array_like
        Shape (N,3) array of particle positions
    h: array_like
        Shape (N,) array of particle smoothing lengths
    center: array_like
        Shape (2,) array containing the coorindates of the center of the map
    size: float
        Side-length of the map
    res: int, optional
        Resolution of the map
    box_size: float, optional
        Size of the periodic domain. If set, periodic boundary conditions are
        applied and coordinates are assumed to run from [0, box_size).
    parallel: boolean, optional
        Whether to run in parallel (default True)
    conservative: boolean, optional
        Whether to do a a perfectly-conservative deposition to the grid (default False)
    """
    box_size = _normalize_box_size(box_size)

    if conservative:
        core = GridSurfaceDensity_conservative_core
    else:
        core = GridSurfaceDensity_core

    if parallel:
        return _run_parallel_splatting(core, f, x, h, center, size, res, box_size, ndim=2)
    else:
        return core(f, x, h, center, size, res, box_size)


def _run_parallel_splatting(splatting_func, f, x, h, center, size, res, box_size, ndim=2):
    """Parallelize a cached splatting function using Python threads.

    Each thread runs the numba-compiled (and cached) core function on a chunk
    of particles.  The GIL is released inside numba, so threads run truly in
    parallel.  This avoids the numba parallel-function caching limitation.
    """
    nthreads = get_num_threads()
    fc = np.array_split(f, nthreads, 0)
    xc = np.array_split(x, nthreads, 0)
    hc = np.array_split(h, nthreads, 0)
    shape = (res,) * ndim
    results = [None] * nthreads

    def _worker(i):
        results[i] = splatting_func(fc[i], xc[i], hc[i], center, size, res, box_size)

    threads = []
    from threading import Thread
    for i in range(nthreads):
        t = Thread(target=_worker, args=(i,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    out = np.zeros(shape)
    for r in results:
        out += r
    return out




@njit(fastmath=True, error_model="numpy", cache=True)
def GridSurfaceDensity_core(f, x, h, center, size, res=128, box_size=-1.0):
    """
    Computes the surface density of conserved quantity f colocated at positions x with smoothing lengths h. E.g.
    plugging in particle masses would return mass surface density. The result is on a Cartesian grid of sightlines,
    the result being the density of quantity f integrated along those sightlines.

    Arguments:
    f - (N,) array of the conserved quantity that you want the surface density of (e.g. particle masses)
    x - (N,3) array of particle positions
    h - (N,) array of particle smoothing lengths
    center - (2,) array containing the coorindates of the center of the map
    size - side-length of the map
    res - resolution of the grid
    box_size - size of the periodic domain (<= 0 disables periodicity)
    """
    # periodic mode tiles the box with res *distinct* cells (spacing size/res);
    # non-periodic uses the node convention spanning corner-to-corner (size/(res-1))
    dx = size / res if box_size > 0 else size / (res - 1)

    x2d = x[:, :2] - center[:2] + size / 2

    grid = np.zeros((res, res))
    offsets = periodic_offsets(box_size)
    n_off = offsets.shape[0]

    N = len(x)
    for i in range(N):
        hs = h[i]
        hs_sqr = hs * hs
        hinv = 1 / hs
        mh2 = f[i] * hinv * hinv

        for a in range(n_off):
            xs0 = x2d[i, 0] + offsets[a]
            gxmin = max(int((xs0 - hs) / dx + 1), 0)
            gxmax = min(int((xs0 + hs) / dx), res - 1)
            if gxmin > gxmax:
                continue
            for b in range(n_off):
                xs1 = x2d[i, 1] + offsets[b]
                gymin = max(int((xs1 - hs) / dx + 1), 0)
                gymax = min(int((xs1 + hs) / dx), res - 1)
                if gymin > gymax:
                    continue
                for gx in range(gxmin, gxmax + 1):
                    delta_x_sqr = xs0 - gx * dx
                    delta_x_sqr *= delta_x_sqr
                    for gy in range(gymin, gymax + 1):
                        delta_y_sqr = xs1 - gy * dx
                        delta_y_sqr *= delta_y_sqr
                        r_sqr = delta_x_sqr + delta_y_sqr
                        if r_sqr > hs_sqr:
                            continue
                        q = np.sqrt(r_sqr) * hinv
                        grid[gx, gy] += kernel2d(q) * mh2
    return grid


@njit(fastmath=True, cache=True)
def GridSurfaceDensity_conservative_core(f, x, h, center, size, res=128, box_size=-1.0):
    """
    Computes the surface density of conserved quantity f colocated at positions x with smoothing lengths h.
    E.g. plugging in particle masses would return mass surface density.

    This method performs a kernel-weighted deposition so that the quantity deposited to the grid is conserved to machine precision.

    Arguments:
    f - (N,) array of the conserved quantity that you want the surface density of (e.g. particle masses)
    x - (N,3) array of particle positions
    h - (N,) array of particle smoothing lengths
    center - (2,) array containing the coorindates of the center of the map
    size - side-length of the map
    res - resolution of the grid
    box_size - size of the periodic domain (<= 0 disables periodicity)
    """
    # periodic mode tiles the box with res *distinct* cells (spacing size/res);
    # non-periodic uses the node convention spanning corner-to-corner (size/(res-1))
    dx = size / res if box_size > 0 else size / (res - 1)
    dx2inv = 1 / (dx * dx)

    x2d = x[:, :2] - center[:2] + size / 2

    grid = np.zeros((res, res))
    offsets = periodic_offsets(box_size)
    n_off = offsets.shape[0]

    N = len(x)
    for i in range(N):
        hs = h[i]
        hs = max(hs, dx)
        hs_sqr = hs * hs
        hinv = 1 / hs

        # cache kernel values across both passes; the two passes recompute the
        # exact same (image, cell) iteration order so the flat index j stays aligned
        nper = int(2.0 * hs / dx) + 2
        kval = np.empty(n_off * n_off * nper * nper)

        # first pass: cache kernel values and accumulate total weight over images
        total_wt = 0.0
        j = 0
        for a in range(n_off):
            xs0 = x2d[i, 0] + offsets[a]
            gxmin = max(int((xs0 - hs) / dx + 1), 0)
            gxmax = min(int((xs0 + hs) / dx), res - 1)
            if gxmin > gxmax:
                continue
            for b in range(n_off):
                xs1 = x2d[i, 1] + offsets[b]
                gymin = max(int((xs1 - hs) / dx + 1), 0)
                gymax = min(int((xs1 + hs) / dx), res - 1)
                if gymin > gymax:
                    continue
                for gx in range(gxmin, gxmax + 1):
                    delta_x_sqr = xs0 - gx * dx
                    delta_x_sqr *= delta_x_sqr
                    for gy in range(gymin, gymax + 1):
                        delta_y_sqr = xs1 - gy * dx
                        delta_y_sqr *= delta_y_sqr
                        r_sqr = delta_x_sqr + delta_y_sqr
                        if r_sqr > hs_sqr:
                            kval[j] = 0.0
                        else:
                            k = kernel2d(np.sqrt(r_sqr) * hinv)
                            kval[j] = k
                            total_wt += k
                        j += 1

        if total_wt == 0:
            continue

        # second pass: deposit using cached, normalized kernel weights
        fac = f[i] / total_wt
        j = 0
        for a in range(n_off):
            xs0 = x2d[i, 0] + offsets[a]
            gxmin = max(int((xs0 - hs) / dx + 1), 0)
            gxmax = min(int((xs0 + hs) / dx), res - 1)
            if gxmin > gxmax:
                continue
            for b in range(n_off):
                xs1 = x2d[i, 1] + offsets[b]
                gymin = max(int((xs1 - hs) / dx + 1), 0)
                gymax = min(int((xs1 + hs) / dx), res - 1)
                if gymin > gymax:
                    continue
                for gx in range(gxmin, gxmax + 1):
                    for gy in range(gymin, gymax + 1):
                        k = kval[j]
                        j += 1
                        if k > 0:
                            grid[gx, gy] += k * fac

    return grid * dx2inv


@njit(fastmath=True, cache=True)
def UpsampleGrid_PPV(grid):
    newgrid = np.empty((grid.shape[0] * 2, grid.shape[1] * 2, grid.shape[2]))
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            newgrid[2 * i, 2 * j, :] = grid[i, j, :]
            newgrid[2 * i + 1, 2 * j, :] = grid[i, j, :]
            newgrid[2 * i, 2 * j + 1, :] = grid[i, j, :]
            newgrid[2 * i + 1, 2 * j + 1, :] = grid[i, j, :]
    return newgrid


def Grid_PPZ_DataCube_Multigrid(f, x, h, center, size, z, h_z, res, box_size=None, N_grid_kernel=8):
    """Faster, multigrid version of Grid_PPZ_DataCube. Since the third dimension is separate from the spatial ones, we only do the multigrid approach on the spatial grid. See Grid_PPZ_DataCube for desription of inputs"""
    box_size = _normalize_box_size(box_size)
    if not ((res[0] != 0) and (res[0] & (res[0] - 1) == 0)):
        raise ValueError("Multigrid resolution must be a power of 2")
    res_bins = size[0] / 2 ** np.arange(0, round(np.log2(res[0]) + 1))
    res_bins[0] = np.inf
    res_bins[-1] = 0
    grid = np.zeros((1, 1, res[1]))
    for i in range(len(res_bins) - 1):
        grid = UpsampleGrid_PPV(grid)
        Ni = grid.shape[0]
        # bin particles by smoothing length to decide which resolution level they get deposited at
        idx = (h / N_grid_kernel < res_bins[i]) & (h / N_grid_kernel >= res_bins[i + 1])
        print(Ni, np.sum(idx))
        if np.any(idx):
            grid += Grid_PPZ_DataCube(
                f[idx],
                x[idx],
                h[idx],
                center,
                size,
                z[idx],
                h_z[idx],
                np.array([Ni, res[1]], dtype=np.int32),
                box_size=box_size,
            )
    return grid


@njit(fastmath=True, cache=True)
def Grid_PPZ_DataCube(f, x, h, center, size, z, h_z, res, box_size=-1.0):
    """
    A modified version of the GridSurfaceDensity script, it computes the PPZ datacube of conserved quantity f, where Z is an arbitrary data dimension (e.g. using line-of-sight velocity as Z gives the usual astro PPV datacubes, using position gives a PPP cube, which is just the density).

    Periodic boundary conditions (if box_size > 0) are applied to the two spatial
    dimensions only; the Z dimension is always treated as non-periodic.

    Arguments:
    f - (N,) array of the conserved quantity that you want the surface density of (e.g. particle masses)
    x - (N,3) array of particle positions
    h - (N,) array of particle smoothing lengths
    z - (N,) array of particle positions in the Z dimension (e.g. line of sight velocity values)
    h_z - (N,) array of uncertainties ("smoothing lengths") in the Z dimension (e.g. thermal velocity)
    center - (3,) array containing the coordinates of the center of the PPZ map
    size - (2) side-length of the map, first value for the PP map, second value is for Z (e.g. max velocity - min velocity)
    res - (2) resolution of the PPX grid, first value is for the PP map, second is for Z (e.g. [128, 16] means a 128x128x16 PPX cube)
    box_size - size of the periodic domain in the spatial dimensions (<= 0 disables periodicity)
    """
    # periodic BCs apply to the two spatial dimensions only (see periodic_offsets)
    dx = size[0] / res[0] if box_size > 0 else size[0] / (res[0] - 1)
    dz = size[1] / (res[1] - 1)

    x2d = x[:, :2] - center[:2] + size[0] / 2
    z1d = z - center[2] + size[1] / 2

    grid = np.zeros((res[0], res[0], res[1]))
    offsets = periodic_offsets(box_size)
    n_off = offsets.shape[0]

    N = len(x)
    for i in range(N):
        zs = z1d[i]
        hs = h[i]
        h_z_s = h_z[i]
        hinvsq = 1 / (hs * hs)
        h_z_invsq = 1 / (h_z_s * h_z_s)
        f_density = f[i] / (hs * hs * h_z_s)

        gzmin = max(int((zs - h_z_s) / dz + 1), 0)
        gzmax = min(int((zs + h_z_s) / dz), res[1] - 1)
        if gzmin > gzmax:
            continue

        for a in range(n_off):
            xs0 = x2d[i, 0] + offsets[a]
            gxmin = max(int((xs0 - hs) / dx + 1), 0)
            gxmax = min(int((xs0 + hs) / dx), res[0] - 1)
            if gxmin > gxmax:
                continue
            for b in range(n_off):
                xs1 = x2d[i, 1] + offsets[b]
                gymin = max(int((xs1 - hs) / dx + 1), 0)
                gymax = min(int((xs1 + hs) / dx), res[0] - 1)
                if gymin > gymax:
                    continue
                for gx in range(gxmin, gxmax + 1):
                    delta_x_sqr = xs0 - gx * dx
                    delta_x_sqr *= delta_x_sqr
                    for gy in range(gymin, gymax + 1):
                        delta_y_sqr = xs1 - gy * dx
                        delta_y_sqr *= delta_y_sqr
                        q2dsq = (delta_x_sqr + delta_y_sqr) * hinvsq
                        for gz in range(gzmin, gzmax + 1):
                            delta_z_sqr = zs - gz * dz
                            delta_z_sqr *= delta_z_sqr
                            q = np.sqrt(q2dsq + delta_z_sqr * h_z_invsq)
                            # Using 3D normalization
                            grid[gx, gy, gz] += kernel3d(q) * f_density
    return grid


@njit(fastmath=True, cache=True)
def GridAverage(f, x, h, center, size, res=128, box_size=-1.0):
    """
    Computes the number density-weighted average of a function f, integrated along sightlines on a Cartesian grid. ie. integral(n f dz)/integral(n dz) where n is the number density and z is the direction of the sightline.

    Arguments:
    f - (N,) array of the conserved quantity that you want the surface density of (e.g. particle masses)
    x - (N,3) array of particle positions
    h - (N,) array of particle smoothing lengths
    center - (2,) array containing the coordinates of the center of the map
    size - side-length of the map
    res - resolution of the grid
    box_size - size of the periodic domain (<= 0 disables periodicity)
    """
    # periodic mode tiles the box with res *distinct* cells (spacing size/res);
    # non-periodic uses the node convention spanning corner-to-corner (size/(res-1))
    dx = size / res if box_size > 0 else size / (res - 1)

    x2d = x[:, :2] - center[:2] + size / 2

    grid1 = np.zeros((res, res))
    grid2 = np.zeros((res, res))
    offsets = periodic_offsets(box_size)
    n_off = offsets.shape[0]

    N = len(x)
    for i in range(N):
        hs = h[i]
        hs_sqr = hs * hs
        hinv = 1 / hs
        h2 = hinv * hinv
        fh2 = f[i] * h2

        for a in range(n_off):
            xs0 = x2d[i, 0] + offsets[a]
            gxmin = max(int((xs0 - hs) / dx + 1), 0)
            gxmax = min(int((xs0 + hs) / dx), res - 1)
            if gxmin > gxmax:
                continue
            for b in range(n_off):
                xs1 = x2d[i, 1] + offsets[b]
                gymin = max(int((xs1 - hs) / dx + 1), 0)
                gymax = min(int((xs1 + hs) / dx), res - 1)
                if gymin > gymax:
                    continue
                for gx in range(gxmin, gxmax + 1):
                    delta_x_sqr = xs0 - gx * dx
                    delta_x_sqr *= delta_x_sqr
                    for gy in range(gymin, gymax + 1):
                        delta_y_sqr = xs1 - gy * dx
                        delta_y_sqr *= delta_y_sqr
                        r_sqr = delta_x_sqr + delta_y_sqr
                        if r_sqr > hs_sqr:
                            continue
                        q = np.sqrt(r_sqr) * hinv
                        kernel = kernel2d(q)
                        grid1[gx, gy] += kernel * h2
                        grid2[gx, gy] += fh2 * kernel
    return grid2 / grid1


@njit(fastmath=True, cache=True)
def WeightedGridInterp3D(f, wt, x, h, center, size, res=128, box_size=-1.0):
    """
    Peforms a weighted grid interpolation of quantity f onto a 3D grid

    Arguments:
    f - (N,) array of the function defined on the point set that you want to interpolate to the grid
    wt - (N,) array of weights
    x - (N,3) array of particle positions
    h - (N,) array of particle smoothing lengths
    center - (3,) array containing the coorindates of the center of the map
    size - side-length of the map
    res - resolution of the grid
    box_size - size of the periodic domain (<= 0 disables periodicity)
    """
    # periodic mode tiles the box with res *distinct* cells (spacing size/res);
    # non-periodic uses the node convention spanning corner-to-corner (size/(res-1))
    dx = size / res if box_size > 0 else size / (res - 1)

    x3d = x - center  # coordinates in the grid frame, such that the origin is at the corner of the grid
    gridcoords = np.linspace(dx / 2 - size / 2, size / 2 - dx / 2, res)

    grid = np.zeros((res, res, res))
    gridwt = np.zeros_like(grid)
    offsets = periodic_offsets(box_size)
    n_off = offsets.shape[0]

    N = len(x)
    for i in range(N):
        hs = h[i]
        hinv = 1 / hs
        hs_dx = hs / dx

        # cache kernel values across both passes; the two passes recompute the
        # exact same (image, cell) iteration order so the flat index j stays aligned
        nper = int(2.0 * hs_dx) + 2
        kval = np.empty(n_off * n_off * n_off * nper * nper * nper)

        # first pass: cache kernel values and accumulate total weight over images
        total_wt = 0.0
        j = 0
        for a in range(n_off):
            xs0 = x3d[i, 0] + offsets[a]
            gxmin = max(int((xs0 + size / 2) / dx - 0.5 - hs_dx + 1), 0)
            gxmax = min(int((xs0 + size / 2) / dx - 0.5 + hs_dx), res - 1)
            if gxmin > gxmax:
                continue
            for b in range(n_off):
                xs1 = x3d[i, 1] + offsets[b]
                gymin = max(int((xs1 + size / 2) / dx - 0.5 - hs_dx + 1), 0)
                gymax = min(int((xs1 + size / 2) / dx - 0.5 + hs_dx), res - 1)
                if gymin > gymax:
                    continue
                for c in range(n_off):
                    xs2 = x3d[i, 2] + offsets[c]
                    gzmin = max(int((xs2 + size / 2) / dx - 0.5 - hs_dx + 1), 0)
                    gzmax = min(int((xs2 + size / 2) / dx - 0.5 + hs_dx), res - 1)
                    if gzmin > gzmax:
                        continue
                    for gx in range(gxmin, gxmax + 1):
                        delta_x_sqr = xs0 - gridcoords[gx]
                        delta_x_sqr *= delta_x_sqr
                        for gy in range(gymin, gymax + 1):
                            delta_y_sqr = xs1 - gridcoords[gy]
                            delta_y_sqr *= delta_y_sqr
                            for gz in range(gzmin, gzmax + 1):
                                delta_z_sqr = xs2 - gridcoords[gz]
                                delta_z_sqr *= delta_z_sqr
                                q = np.sqrt(delta_x_sqr + delta_y_sqr + delta_z_sqr) * hinv
                                k = kernel3d(q)
                                kval[j] = k
                                total_wt += k
                                j += 1

        if total_wt == 0:
            continue

        # second pass: deposit using cached, normalized kernel weights
        finv = wt[i] / total_wt
        j = 0
        for a in range(n_off):
            xs0 = x3d[i, 0] + offsets[a]
            gxmin = max(int((xs0 + size / 2) / dx - 0.5 - hs_dx + 1), 0)
            gxmax = min(int((xs0 + size / 2) / dx - 0.5 + hs_dx), res - 1)
            if gxmin > gxmax:
                continue
            for b in range(n_off):
                xs1 = x3d[i, 1] + offsets[b]
                gymin = max(int((xs1 + size / 2) / dx - 0.5 - hs_dx + 1), 0)
                gymax = min(int((xs1 + size / 2) / dx - 0.5 + hs_dx), res - 1)
                if gymin > gymax:
                    continue
                for c in range(n_off):
                    xs2 = x3d[i, 2] + offsets[c]
                    gzmin = max(int((xs2 + size / 2) / dx - 0.5 - hs_dx + 1), 0)
                    gzmax = min(int((xs2 + size / 2) / dx - 0.5 + hs_dx), res - 1)
                    if gzmin > gzmax:
                        continue
                    for gx in range(gxmin, gxmax + 1):
                        for gy in range(gymin, gymax + 1):
                            for gz in range(gzmin, gzmax + 1):
                                kernel = kval[j]
                                j += 1
                                grid[gx, gy, gz] += f[i] * kernel * finv
                                gridwt[gx, gy, gz] += kernel * finv

    result = grid / gridwt
    #    do a loop through the grid to check for nans and address if needed
    pos = np.zeros(3)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for k in range(grid.shape[2]):
                # we have to search for the nearest neighbor and take its value - this is usually a rare edge case
                if gridwt[i, j, k] == 0:
                    pos[0] = gridcoords[i]
                    pos[1] = gridcoords[j]
                    pos[2] = gridcoords[k]
                    min_dist = 1e100
                    min_dist_idx = -1
                    for p in range(x3d.shape[0]):  # brute force search lol
                        dist = 0
                        for d in range(3):
                            ddx = x3d[p, d] - pos[d]
                            if box_size > 0:
                                if ddx > 0.5 * box_size:
                                    ddx -= box_size
                                elif ddx < -0.5 * box_size:
                                    ddx += box_size
                            dist += ddx * ddx
                        if dist < min_dist:
                            min_dist = dist
                            min_dist_idx = p
                    result[i, j, k] = f[min_dist_idx]
    return result


def GridDensity(f, x, h, center, size, res=128, box_size=None, parallel=True):
    box_size = _normalize_box_size(box_size)
    if parallel:
        return _run_parallel_splatting(GridDensity_core, f, x, h, center, size, res, box_size, ndim=3)
    else:
        return GridDensity_core(f, x, h, center, size, res, box_size)


@njit(fastmath=True, cache=True)
def GridDensity_core(f, x, h, center, size, res=128, box_size=-1.0):
    """
    Estimates the density of the conserved quantity f possessed by each particle (e.g. mass, momentum, energy) on a 3D grid

    Arguments:
    f - (N,) array of the conserved quantity that you want the surface density of (e.g. particle masses)
    x - (N,3) array of particle positions
    h - (N,) array of particle smoothing lengths
    center - (2,) array containing the coorindates of the center of the map
    size - side-length of the map
    res - resolution of the grid
    box_size - size of the periodic domain (<= 0 disables periodicity)
    """
    # periodic mode tiles the box with res *distinct* cells (spacing size/res);
    # non-periodic uses the node convention spanning corner-to-corner (size/(res-1))
    dx = size / res if box_size > 0 else size / (res - 1)

    # coordinates in the grid frame, such that the origin is at the corner of the grid
    x3d = x - center + size / 2 - dx / 2

    grid = np.zeros((res, res, res))
    offsets = periodic_offsets(box_size)
    n_off = offsets.shape[0]

    N = len(x)
    for i in range(N):
        hs = h[i]
        hinv = 1 / hs

        # cache kernel values across both passes; the two passes recompute the
        # exact same (image, cell) iteration order so the flat index j stays aligned
        nper = int(2.0 * hs / dx) + 2
        kval = np.empty(n_off * n_off * n_off * nper * nper * nper)

        # first pass: cache kernel values and accumulate total weight over images
        total_wt = 0.0
        j = 0
        for a in range(n_off):
            xs0 = x3d[i, 0] + offsets[a]
            gxmin = max(int((xs0 - hs) / dx + 1), 0)
            gxmax = min(int((xs0 + hs) / dx), res - 1)
            if gxmin > gxmax:
                continue
            for b in range(n_off):
                xs1 = x3d[i, 1] + offsets[b]
                gymin = max(int((xs1 - hs) / dx + 1), 0)
                gymax = min(int((xs1 + hs) / dx), res - 1)
                if gymin > gymax:
                    continue
                for c in range(n_off):
                    xs2 = x3d[i, 2] + offsets[c]
                    gzmin = max(int((xs2 - hs) / dx + 1), 0)
                    gzmax = min(int((xs2 + hs) / dx), res - 1)
                    if gzmin > gzmax:
                        continue
                    for gx in range(gxmin, gxmax + 1):
                        delta_x_sqr = xs0 - gx * dx
                        delta_x_sqr *= delta_x_sqr
                        for gy in range(gymin, gymax + 1):
                            delta_y_sqr = xs1 - gy * dx
                            delta_y_sqr *= delta_y_sqr
                            for gz in range(gzmin, gzmax + 1):
                                delta_z_sqr = xs2 - gz * dx
                                delta_z_sqr *= delta_z_sqr
                                q = np.sqrt(delta_x_sqr + delta_y_sqr + delta_z_sqr) * hinv
                                k = kernel3d(q)
                                kval[j] = k
                                total_wt += k
                                j += 1

        if total_wt == 0:
            continue

        # second pass: deposit using cached, normalized kernel weights
        fac = f[i] / total_wt
        j = 0
        for a in range(n_off):
            xs0 = x3d[i, 0] + offsets[a]
            gxmin = max(int((xs0 - hs) / dx + 1), 0)
            gxmax = min(int((xs0 + hs) / dx), res - 1)
            if gxmin > gxmax:
                continue
            for b in range(n_off):
                xs1 = x3d[i, 1] + offsets[b]
                gymin = max(int((xs1 - hs) / dx + 1), 0)
                gymax = min(int((xs1 + hs) / dx), res - 1)
                if gymin > gymax:
                    continue
                for c in range(n_off):
                    xs2 = x3d[i, 2] + offsets[c]
                    gzmin = max(int((xs2 - hs) / dx + 1), 0)
                    gzmax = min(int((xs2 + hs) / dx), res - 1)
                    if gzmin > gzmax:
                        continue
                    for gx in range(gxmin, gxmax + 1):
                        for gy in range(gymin, gymax + 1):
                            for gz in range(gzmin, gzmax + 1):
                                grid[gx, gy, gz] += kval[j] * fac
                                j += 1
    return grid / (dx * dx * dx)
