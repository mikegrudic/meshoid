from numba import njit, prange, get_num_threads
import numpy as np
from scipy.interpolate import RectBivariateSpline
from .derivatives import nearest_image
from .kernel_density import *


@njit(fastmath=True, error_model="numpy")
def grid_index_to_coordinate(index: int, grid_length: float, grid_center: float, N_grid: int, box_size=None):
    """Convert the index of a grid to the *cell-centered* coordinate of that
    grid cell, in a given dimension

    Parameters
    ----------
    index: int
        Grid index, running from 0 to N_grid-1
    grid_length: float
        Side-length of the grid
    grid_center: float
        Coordinate of the grid center
    N_grid: int
        Number of cells per grid dimension
    box_size:
        Size of the periodic domain - if not None, we assume the domain coordinates
        run from [0,box_size)

    Returns
    -------
    x: float
        Coordinate of the center of the grid cell
    """

    x = grid_center - 0.5 * grid_length + grid_length / N_grid * (0.5 + index)
    if box_size is not None:
        return x % box_size
    return x


@njit(fastmath=True, error_model="numpy")
def grid_dx_from_coordinate(x: float, index: int, grid_length: float, grid_center: float, N_grid: int, box_size=None):
    """
    Returns the *nearest image* coordinate difference from a given coordinate x
    to the cell-centered grid-point residing at index

    Parameters
    ----------
    x: float
        The original coordinate, from which to compute coordinate difference
    index: int
        Grid index, running from 0 to N_grid-1
    grid_length: float
        Side-length of the grid
    N_grid: int
        Number of cells per grid dimension
    grid_center: float
        Coordinate of the grid center
    box_size:
        Size of the periodic domain - if not None, we assume the domain coordinates
        run from [0,box_size)

    Returns
    -------
    dx: float
        The nearest-image Cartesian coordinate difference from x to grid cell i
    """

    x_grid = grid_index_to_coordinate(index, grid_length, grid_center, N_grid, box_size)
    dx = x_grid - x
    if box_size is not None:
        return nearest_image(dx, box_size)
    return dx


def GridSurfaceDensityMultigrid(f, x, h, center, size, res=128, box_size=-1, N_grid_kernel=8, parallel=True):
    if not ((res != 0) and (res & (res - 1) == 0)):
        raise ("Multigrid resolution must be a power of 2")
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
                f[idx], x[idx], h[idx], center, size, res=Ni, box_size=box_size, parallel=parallel
            )
    return grid


def UpsampleGrid(grid):
    N = grid.shape[0]
    x1 = np.linspace(0.5 / N, 1 - 0.5 / N, N)  # original coords
    x2 = np.linspace(0.25 / N, 1 - 0.25 / N, 2 * N)  # new coords
    return RectBivariateSpline(x1, x1, grid)(x2, x2)  # RectBivariateSpline(x1,


# @njit  # (parallel=True, fastmath=True)
def GridSurfaceDensity(f, x, h, center, size, res=128, box_size=-1, parallel=True, conservative=False):
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
    parallel: boolean, optional
        Whether to run in parallel (default True)
    conservative: boolean, optional
        Whether to do a a perfectly-conservative deposition to the grid (default False)
    """

    if conservative:
        splatting_func = GridSurfaceDensity_conservative_core
    else:
        splatting_func = GridSurfaceDensity_core

    if parallel:
        return parallelize_generic_splatting_2d(splatting_func, f, x, h, center, size, res, box_size)
    else:
        return splatting_func(f, x, h, center, size, res, box_size)


@njit(parallel=True)
def parallelize_generic_splatting_3d(splatting_function, f, x, h, center, size, res=128, box_size=-1):
    """
    Generic driver for performing splatting/deposition operations in parallel by chunking the particle list
    and summing the end result.

    Parameters
    ----------
    splatting_function: function
        Function that performs the splatting on a subset of particles
    f: array_like
        Shape (N,) array of the conserved quantity that you want the surface density of (e.g. particle masses)
    x: array_like
        Shape (N,3) array of particle positions
    h: array_like
        Shape (N,) array of particle smoothing lengths
    center: array_like
        Shape (3,) array containing the coorindates of the center of the map
    size: float
        Side-length of the map
    res: int, optional
        Resolution of the map
    """

    Nthreads = get_num_threads()
    NUM_CHUNKS_PER_THREAD = 1
    Nchunks = Nthreads * NUM_CHUNKS_PER_THREAD
    # chunk the particles among the threads
    splatted_values = np.zeros((res, res, res))  # numba will automatically perform parallel reduction onto this
    f, x, h = np.array_split(f, Nchunks, 0), np.array_split(x, Nchunks, 0), np.array_split(h, Nchunks, 0)
    for i in prange(Nchunks):
        splatted_values += splatting_function(f[i], x[i], h[i], center, size, res, box_size)
    return splatted_values


@njit(parallel=True)
def parallelize_generic_splatting_2d(splatting_function, f, x, h, center, size, res=128, box_size=-1):
    """
    Generic driver for performing splatting/deposition operations in parallel by chunking the particle list
    and summing the end result.

    Parameters
    ----------
    splatting_function: function
        Function that performs the splatting on a subset of particles
    f: array_like
        Shape (N,) array of the conserved quantity that you want the surface density of (e.g. particle masses)
    x: array_like
        Shape (N,3) array of particle positions
    h: array_like
        Shape (N,) array of particle smoothing lengths
    center: array_like
        Shape (3,) or (2,) array containing the coordinates of the center of the map
    size: float
        Side-length of the map
    res: int, optional
        Resolution of the map
    """

    Nthreads = get_num_threads()
    NUM_CHUNKS_PER_THREAD = 1
    Nchunks = Nthreads * NUM_CHUNKS_PER_THREAD
    # chunk the particles among the threads
    splatted_values = np.zeros((res, res))  # numba will automatically perform parallel reduction onto this
    f, x, h = np.array_split(f, Nchunks, 0), np.array_split(x, Nchunks, 0), np.array_split(h, Nchunks, 0)
    for i in prange(Nchunks):
        splatted_values += splatting_function(f[i], x[i], h[i], center, size, res, box_size)
    return splatted_values


@njit(fastmath=True, error_model="numpy")
def GridSurfaceDensity_core(f, x, h, center, size, res=128, box_size=-1):
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
    """
    dx = size / (res - 1)

    x2d = x[:, :2] - center[:2] + size / 2

    grid = np.zeros((res, res))

    N = len(x)
    for i in range(N):
        xs = x2d[i]
        hs = h[i]
        hinv = 1 / hs
        mh2 = f[i] * hinv * hinv

        gxmin = max(int((xs[0] - hs) / dx + 1), 0)
        gxmax = min(int((xs[0] + hs) / dx), res - 1)
        gymin = max(int((xs[1] - hs) / dx + 1), 0)
        gymax = min(int((xs[1] + hs) / dx), res - 1)

        for gx in range(gxmin, gxmax + 1):
            delta_x_sqr = xs[0] - gx * dx
            delta_x_sqr *= delta_x_sqr
            for gy in range(gymin, gymax + 1):
                delta_y_sqr = xs[1] - gy * dx
                delta_y_sqr *= delta_y_sqr
                r = np.sqrt(delta_x_sqr + delta_y_sqr)
                if r > hs:
                    continue
                q = r * hinv
                grid[gx, gy] += kernel2d(q) * mh2
    return grid


@njit(fastmath=True)
def GridSurfaceDensity_conservative_core(f, x, h, center, size, res=128, box_size=-1):
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
    """
    dx = size / (res - 1)
    dx2inv = 1 / (dx * dx)

    x2d = x[:, :2] - center[:2] + size / 2

    grid = np.zeros((res, res))

    N = len(x)
    for i in range(N):
        xs = x2d[i]
        hs = h[i]
        hs = max(hs, dx)
        hinv = 1 / hs

        gxmin = max(int((xs[0] - hs) / dx + 1), 0)
        gxmax = min(int((xs[0] + hs) / dx), res - 1)
        gymin = max(int((xs[1] - hs) / dx + 1), 0)
        gymax = min(int((xs[1] + hs) / dx), res - 1)

        total_wt = 0
        for gx in range(gxmin, gxmax + 1):
            delta_x_sqr = xs[0] - gx * dx
            delta_x_sqr *= delta_x_sqr
            for gy in range(gymin, gymax + 1):
                delta_y_sqr = xs[1] - gy * dx
                delta_y_sqr *= delta_y_sqr
                r = np.sqrt(delta_x_sqr + delta_y_sqr)
                if r > hs:
                    continue
                q = r * hinv
                total_wt += kernel2d(q)

        if total_wt == 0:
            continue

        for gx in range(gxmin, gxmax + 1):
            delta_x_sqr = xs[0] - gx * dx
            delta_x_sqr *= delta_x_sqr
            for gy in range(gymin, gymax + 1):
                delta_y_sqr = xs[1] - gy * dx
                delta_y_sqr *= delta_y_sqr
                r = np.sqrt(delta_x_sqr + delta_y_sqr)
                if r > hs:
                    continue
                q = r * hinv
                grid[gx, gy] += kernel2d(q) * f[i] / total_wt

    return grid * dx2inv


@njit(fastmath=True)
def UpsampleGrid_PPV(grid):
    newgrid = np.empty((grid.shape[0] * 2, grid.shape[1] * 2, grid.shape[2]))
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            newgrid[2 * i, 2 * j, :] = grid[i, j, :]
            newgrid[2 * i + 1, 2 * j, :] = grid[i, j, :]
            newgrid[2 * i, 2 * j + 1, :] = grid[i, j, :]
            newgrid[2 * i + 1, 2 * j + 1, :] = grid[i, j, :]
    return newgrid


def Grid_PPZ_DataCube_Multigrid(f, x, h, center, size, z, h_z, res, box_size=-1, N_grid_kernel=8):
    """Faster, multigrid version of Grid_PPZ_DataCube. Since the third dimension is separate from the spatial ones, we only do the multigrid approach on the spatial grid. See Grid_PPZ_DataCube for desription of inputs"""
    if not ((res[0] != 0) and (res[0] & (res[0] - 1) == 0)):
        raise ("Multigrid resolution must be a power of 2")
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


@njit(fastmath=True)
def Grid_PPZ_DataCube(f, x, h, center, size, z, h_z, res, box_size=-1):
    """
    A modified version of the GridSurfaceDensity script, it computes the PPZ datacube of conserved quantity f, where Z is an arbitrary data dimension (e.g. using line-of-sight velocity as Z gives the usual astro PPV datacubes, using position gives a PPP cube, which is just the density).

    Arguments:
    f - (N,) array of the conserved quantity that you want the surface density of (e.g. particle masses)
    x - (N,3) array of particle positions
    h - (N,) array of particle smoothing lengths
    z - (N,) array of particle positions in the Z dimension (e.g. line of sight velocity values)
    h_z - (N,) array of uncertainties ("smoothing lengths") in the Z dimension (e.g. thermal velocity)
    center - (3,) array containing the coordinates of the center of the PPZ map
    size - (2) side-length of the map, first value for the PP map, second value is for Z (e.g. max velocity - min velocity)
    res - (2) resolution of the PPX grid, first value is for the PP map, second is for Z (e.g. [128, 16] means a 128x128x16 PPX cube)
    """
    dx = size[0] / (res[0] - 1)
    dz = size[1] / (res[1] - 1)

    x2d = x[:, :2] - center[:2] + size[0] / 2
    z1d = z - center[2] + size[1] / 2

    grid = np.zeros((res[0], res[0], res[1]))

    N = len(x)
    for i in range(N):
        xs = x2d[i]
        zs = z1d[i]
        hs = h[i]
        h_z_s = h_z[i]
        hinvsq = 1 / (hs * hs)
        h_z_invsq = 1 / (h_z_s * h_z_s)
        f_density = f[i] / (hs * hs * h_z_s)

        gxmin = max(int((xs[0] - hs) / dx + 1), 0)
        gxmax = min(int((xs[0] + hs) / dx), res[0] - 1)
        gymin = max(int((xs[1] - hs) / dx + 1), 0)
        gymax = min(int((xs[1] + hs) / dx), res[0] - 1)
        gzmin = max(int((zs - h_z_s) / dz + 1), 0)
        gzmax = min(int((zs + h_z_s) / dz), res[1] - 1)

        for gx in range(gxmin, gxmax + 1):
            delta_x_sqr = xs[0] - gx * dx
            delta_x_sqr *= delta_x_sqr
            for gy in range(gymin, gymax + 1):
                delta_y_sqr = xs[1] - gy * dx
                delta_y_sqr *= delta_y_sqr
                q2dsq = (delta_x_sqr + delta_y_sqr) * hinvsq
                for gz in range(gzmin, gzmax + 1):
                    delta_z_sqr = zs - gz * dz
                    delta_z_sqr *= delta_z_sqr
                    q = np.sqrt(q2dsq + delta_z_sqr * h_z_invsq)
                    # Using 3D normalization
                    grid[gx, gy, gz] += kernel3d(q) * f_density
    return grid


@njit(fastmath=True)
def GridAverage(f, x, h, center, size, res=128, box_size=-1):
    """
    Computes the number density-weighted average of a function f, integrated along sightlines on a Cartesian grid. ie. integral(n f dz)/integral(n dz) where n is the number density and z is the direction of the sightline.

    Arguments:
    f - (N,) array of the conserved quantity that you want the surface density of (e.g. particle masses)
    x - (N,3) array of particle positions
    h - (N,) array of particle smoothing lengths
    center - (2,) array containing the coordinates of the center of the map
    size - side-length of the map
    res - resolution of the grid
    """
    dx = size / (res - 1)

    x2d = x[:, :2] - center[:2] + size / 2

    grid1 = np.zeros((res, res))
    grid2 = np.zeros((res, res))
    N = len(x)
    for i in range(N):
        xs = x2d[i]
        hs = h[i]
        hinv = 1 / hs
        h2 = hinv * hinv

        gxmin = max(int((xs[0] - hs) / dx + 1), 0)
        gxmax = min(int((xs[0] + hs) / dx), res - 1)
        gymin = max(int((xs[1] - hs) / dx + 1), 0)
        gymax = min(int((xs[1] + hs) / dx), res - 1)

        for gx in range(gxmin, gxmax + 1):
            delta_x_sqr = xs[0] - gx * dx
            delta_x_sqr *= delta_x_sqr
            for gy in range(gymin, gymax + 1):
                delta_y_sqr = xs[1] - gy * dx
                delta_y_sqr *= delta_y_sqr
                q = np.sqrt(delta_x_sqr + delta_y_sqr) * hinv
                kernel = kernel2d(q)
                grid1[gx, gy] += kernel * h2
                grid2[gx, gy] += f[i] * kernel * h2
    return grid2 / grid1


@njit(fastmath=True)
def WeightedGridInterp3D(f, wt, x, h, center, size, res=128, box_size=-1):
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
    """
    dx = size / (res - 1)

    x3d = x - center  # coordinates in the grid frame, such that the origin is at the corner of the grid
    gridcoords = np.linspace(dx / 2 - size / 2, size / 2 - dx / 2, res)

    grid = np.zeros((res, res, res))
    gridwt = np.zeros_like(grid)

    x_to_grid_idx = np.empty(3)
    N = len(x)
    for i in range(N):
        xs = x3d[i]
        hs = h[i]
        hinv = 1 / hs
        for k in range(3):
            x_to_grid_idx[k] = (xs[k] + size / 2) / dx - 0.5
        hs_dx = hs / dx
        gxmin = max(int(x_to_grid_idx[0] - hs_dx + 1), 0)
        gxmax = min(int(x_to_grid_idx[0] + hs_dx), res - 1)
        gymin = max(int(x_to_grid_idx[1] - hs_dx + 1), 0)
        gymax = min(int(x_to_grid_idx[1] + hs_dx), res - 1)
        gzmin = max(int(x_to_grid_idx[2] - hs_dx + 1), 0)
        gzmax = min(int(x_to_grid_idx[2] + hs_dx), res - 1)

        # first have to do a prepass to get the weight
        kval = np.empty(int(2 * hs / dx + 1) ** 3)  # save kernel values so don't have to recompute
        total_wt = 0
        j = 0
        for gx in range(gxmin, gxmax + 1):
            #            delta_x_sqr = xs[0] - gx*dx
            delta_x_sqr = xs[0] - gridcoords[gx]
            delta_x_sqr *= delta_x_sqr
            for gy in range(gymin, gymax + 1):
                #                delta_y_sqr = xs[1] - gy*dx
                delta_y_sqr = xs[1] - gridcoords[gy]
                delta_y_sqr *= delta_y_sqr
                for gz in range(gzmin, gzmax + 1):
                    delta_z_sqr = xs[2] - gridcoords[gz]
                    #                    delta_z_sqr = xs[2] - gz*dx
                    delta_z_sqr *= delta_z_sqr
                    q = np.sqrt(delta_x_sqr + delta_y_sqr + delta_z_sqr) * hinv
                    kernel = kernel3d(q)
                    total_wt += kernel  # to normalize out the kernel weights
                    kval[j] = kernel
                    j += 1

        # OK now do the actual deposition
        j = 0
        for gx in range(gxmin, gxmax + 1):
            for gy in range(gymin, gymax + 1):
                for gz in range(gzmin, gzmax + 1):
                    kernel = kval[j]
                    j += 1
                    if total_wt > 0:
                        grid[gx % res, gy % res, gz % res] += f[i] * kernel * wt[i] / total_wt
                        gridwt[gx % res, gy % res, gz % res] += kernel * wt[i] / total_wt

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
                            dist += (x3d[p, d] - pos[d]) * (x3d[p, d] - pos[d])
                        if dist < min_dist:
                            min_dist = dist
                            min_dist_idx = p
                    result[i, j, k] = f[min_dist_idx]
    return result


def GridDensity(f, x, h, center, size, res=128, box_size=-1.0, parallel=True):
    if parallel:
        return parallelize_generic_splatting_3d(GridDensity_core, f, x, h, center, size, res, box_size)
    else:
        return GridDensity_core(f, x, h, center, size, res, box_size)


@njit(fastmath=True)
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
    """
    dx = size / (res - 1)

    # coordinates in the grid frame, such that the origin is at the corner of the grid
    x3d = x - center + size / 2 - dx / 2

    grid = np.zeros((res, res, res))

    N = len(x)
    for i in range(N):
        xs = x3d[i]
        hs = h[i]
        hinv = 1 / hs
        if box_size < 0:
            gxmin = max(int((xs[0] - hs) / dx + 1), 0)
            gxmax = min(int((xs[0] + hs) / dx), res - 1)
            gymin = max(int((xs[1] - hs) / dx + 1), 0)
            gymax = min(int((xs[1] + hs) / dx), res - 1)
            gzmin = max(int((xs[2] - hs) / dx + 1), 0)
            gzmax = min(int((xs[2] + hs) / dx), res - 1)
        else:
            gxmin = int((xs[0] - hs) / dx + 1)
            gxmax = int((xs[0] + hs) / dx)
            gymin = int((xs[1] - hs) / dx + 1)
            gymax = int((xs[1] + hs) / dx)
            gzmin = int((xs[2] - hs) / dx + 1)
            gzmax = int((xs[2] + hs) / dx)

        # first have to do a prepass to get the weight
        kval = np.empty(int(2 * hs / dx + 1) ** 3)  # save kernel values so don't have to recompute
        total_wt = 0
        j = 0
        for gx in range(gxmin, gxmax + 1):
            delta_x_sqr = xs[0] - gx * dx
            delta_x_sqr *= delta_x_sqr
            for gy in range(gymin, gymax + 1):
                delta_y_sqr = xs[1] - gy * dx
                delta_y_sqr *= delta_y_sqr
                for gz in range(gzmin, gzmax + 1):
                    delta_z_sqr = xs[2] - gz * dx
                    delta_z_sqr *= delta_z_sqr
                    q = np.sqrt(delta_x_sqr + delta_y_sqr + delta_z_sqr) * hinv
                    kernel = kernel3d(q)
                    total_wt += kernel  # to normalize out the kernel weights
                    kval[j] = kernel
                    j += 1

        # OK now do the actual deposition
        j = 0
        for gx in range(gxmin, gxmax + 1):
            for gy in range(gymin, gymax + 1):
                for gz in range(gzmin, gzmax + 1):
                    kernel = kval[j]
                    j += 1
                    if total_wt > 0:
                        grid[gx % res, gy % res, gz % res] += f[i] * kernel / total_wt
    return grid / (dx * dx * dx)
