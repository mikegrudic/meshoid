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
from .derivatives import nearest_image


@njit(fastmath=True, error_model="numpy")
def grid_index_to_coordinate(
    index: int, grid_length: float, grid_center: float, grid_res: int, box_size=None
):
    """Convert the index of a grid to the *cell-centered* coordinate of that
    grid cell, in a given dimension

    Parameters
    ----------
    index: int
        Grid index, running from 0 to grid_res-1
    grid_length: float
        Side-length of the grid
    grid_center: float
        Coordinate of the grid center
    grid_res: int
        Number of cells per grid dimension
    box_size:
        Size of the periodic domain - if not None, we assume the domain coordinates
        run from [0,box_size)

    Returns
    -------
    x: float
        Coordinate of the center of the grid cell
    """

    x = grid_center - 0.5 * grid_length + grid_length / grid_res * (0.5 + index)
    if box_size is not None:
        return x % box_size
    return x


# candidate for vectorization over dimensions
@njit(fastmath=True, error_model="numpy")
def coordinate_to_grid_index(
    x: float,
    grid_length: float,
    grid_center: float,
    grid_res: int,  # , box_size=None
):
    """Convert coordinate to the integer coordinate on a grid, where 0 is
    corresponds to the position of the first grid cell center and grid_res-1
    corresponds to the last. Assumes grid size is less than box_size (so the
    mapping is bijective with no repeat images.

    Parameters
    ----------
    x: float
        Cartesian coordinate value
    grid_length: float
        Side-length of the grid
    grid_center: float
        Coordinate of the grid center
    grid_res: int
        Number of cells per grid dimension

    Returns
    -------
    grid_coord: float
        Grid coordinate - still float, must be rounded to int as appropriate.
    """
    dx = x - grid_center
    grid_spacing = grid_length / grid_res
    return dx / grid_spacing + 0.5 * (grid_res - 1)


@njit(fastmath=True, error_model="numpy")
def grid_index_bounds(
    x: np.ndarray,
    r: float,
    grid_length: float,
    grid_center: float,
    grid_res: int,
    box_size=None,
):
    """
    Returns the lower-left corner on the grid of the square of grid points that
    a particle will overlap, and the width of the square in grid points

    Parameters
    ----------
    x: np.ndarray
        Shape (N_dimensions,) array containing coordinates
    r: float
        Radius of the particle
    grid_length: float
        Side-length of the grid
    grid_center: float
        Coordinate of the grid center
    grid_res: int
        Number of cells per grid dimension
    box_size: optional
        Size of the periodic domain - if not None, we assume the domain
        coordinates run from [0,box_size)
    """
    N_dim = x.shape[0]
    grid_dx = grid_length / (grid_res - 1)
    width = int(2 * r / grid_dx + 1)
    corner = np.empty(2, dtype=np.int64)
    for dim in range(N_dim):
        corner[dim] = int(
            coordinate_to_grid_index(
                x[dim] - r, grid_length, grid_center[dim], grid_res
            )
            + 1
        )

    return corner, width


@njit
def wrapped_index(gx, size, res, box_size):
    if box_size is None:
        return gx
    return gx % (res * box_size / size)


@njit
def coordinate_lies_on_grid(x, size, center, res):
    if x > center - 0.5 * size:
        if x <= center + 0.5 * size:
            return True
    return False


@njit(fastmath=True, error_model="numpy")
def grid_dx_from_coordinate(
    x: float,
    index: int,
    grid_length: float,
    grid_center: float,
    grid_res: int,
    box_size=None,
):
    """
    Returns the *nearest image* coordinate difference from a given coordinate x
    to the cell-centered grid-point residing at index, ASSUMING grid_length <=
    box_size (no repeat images, so mapping from coordaintes to grid is unique)

    Parameters
    ----------
    x: float
        The original coordinate, from which to compute coordinate difference
    index: int
        Grid index, running from 0 to grid_res-1
    grid_length: float
        Side-length of the grid
    grid_res: int
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

    x_grid = grid_index_to_coordinate(
        index, grid_length, grid_center, grid_res, box_size
    )
    if box_size is not None:
        x_grid = x_grid % box_size
    dx = x_grid - x
    if box_size is not None:
        return nearest_image(dx, box_size)
    return dx


def GridSurfaceDensityMultigrid(
    f, x, h, center, size, res=128, box_size=-1, grid_res_kernel=8, parallel=False
):
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
        idx = (h / grid_res_kernel < res_bins[i]) & (
            h / grid_res_kernel >= res_bins[i + 1]
        )
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
            )
    return grid


def UpsampleGrid(grid):
    N = grid.shape[0]
    x1 = np.linspace(0.5 / N, 1 - 0.5 / N, N)  # original coords
    x2 = np.linspace(0.25 / N, 1 - 0.25 / N, 2 * N)  # new coords
    return RectBivariateSpline(x1, x1, grid)(x2, x2)  # RectBivariateSpline(x1,


@njit(parallel=True, fastmath=True)
def GridSurfaceDensity(
    f, x, h, center, size, res=100, box_size=-1, parallel=False, conservative=False
):
    """
    Computes the surface density of conserved quantity f colocated at positions
    x with smoothing lengths h. E.g. plugging in particle masses would return
    mass surface density. The result is on a Cartesian grid of sightlines, the
    result being the density of quantity f integrated along those sightlines.

    Parameters
    ----------
    f - (N,) array of the conserved quantity that you want the surface density of (e.g. particle masses)
    x - (N,3) array of particle positions
    h - (N,) array of particle smoothing lengths
    center - (2,) array containing the coorindates of the center of the map
    size - side-length of the map
    res - resolution of the grid
    parallel - whether to run in parallel, if numeric then how many cores

    """

    if parallel:
        Nthreads = get_num_threads()
        # chunk the particles among the threads
        chunksize = max(len(f) // Nthreads, 1)
        sigmas = np.empty(
            (Nthreads, res, res)
        )  # will store separate grids and sum them at the end

        for i in prange(Nthreads):
            # for i in range(Nthreads):
            if conservative:
                sigmas[i] = GridSurfaceDensity_conservative_core(
                    f[i * chunksize : (i + 1) * chunksize],
                    x[i * chunksize : (i + 1) * chunksize],
                    h[i * chunksize : (i + 1) * chunksize],
                    center,
                    size,
                    res,
                    box_size,
                )
            else:
                sigmas[i] = GridSurfaceDensity_core(
                    f[i * chunksize : (i + 1) * chunksize],
                    x[i * chunksize : (i + 1) * chunksize],
                    h[i * chunksize : (i + 1) * chunksize],
                    center,
                    size,
                    res,
                    box_size,
                )
        return sigmas.sum(0)
    else:
        if conservative:
            return GridSurfaceDensity_conservative_core(
                f, x, h, center, size, res, box_size
            )
        else:
            return GridSurfaceDensity_core(f, x, h, center, size, res, box_size)

def overlapping_grid_coordinates(x,h,center,length,res,box_size):
    num_points = 2*h * (res - 1) / length
    x0 = 

# @njit(fastmath=True)
def GridSurfaceDensity_core(f, x, h, center, size, res=100, box_size=None):
    """
    Computes the surface density of conserved quantity f colocated at positions
    x with smoothing lengths h. E.g. plugging in particle masses would return
    mass surface density. The result is on a Cartesian grid of sightlines, the
    result being the density of quantity f integrated along those sightlines.

    Parameters
    ----------
    f - (N,) array of the conserved quantity that you want the surface density of (e.g. particle masses)
    x - (N,3) array of particle positions
    h - (N,) array of particle smoothing lengths
    center - (2,) array containing the coorindates of the center of the map
    size - side-length of the map
    res - resolution of the grid

    Returns
    -------
    grid - np.ndarray
        Shape (res,res) grid of estimated surface density of the quantity f
    """
    #grid_dx = size / (res - 1)
    #    x2d = x[:, :2] - center[:2] + size / 2

    grid = np.zeros((res, res))

    N = len(x)
    for i in range(N):
        xs = x[i]
        hs = h[i]
        hs2 = hs * hs
        hinv = 1 / hs
        mh2 = f[i] * hinv * hinv



        Nx = len(grid_index_x)
        Ny = len(grid_index_y)

        for i in range(Nx):
            ix = grid_index_x[i]
            x = grid_coordinate_x[i]
            for j in range(Ny):
                iy = grid_index_y[j]
                grid_coordinate_y[j]
                
                #y = grid_index_to_coordinate(gy, size, center[1], res, box_size)
                #print(y)
                if not coordinate_lies_on_grid(y, size, center[1], res):                
                    continue
                dy = nearest_image(
                    y - xs[1], box_size
                )  # grid_dx_from_coordinate(xs[1], gy, size, center[1], res, box_size)
                r2 = dy * dy + dx * dx
                print(
                    gx,
                    gy,
                    x,
                    y,
                    dx,
                    dy,
                    np.sqrt(r2),
                )
                if r2 > hs2:
                    continue
                r = np.sqrt(r2)
                q = r * hinv
                if q <= 0.5:
                    kernel = 1 - 6 * q * q * (1 - q)
                else:
                    a = 1 - q
                    kernel = 2 * a * a * a

                grid[gx, gy] += 1.8189136353359467 * kernel * mh2
    return grid


@njit(fastmath=True)
def GridSurfaceDensity_conservative_core(f, x, h, center, size, res=100, box_size=-1):
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
        if hs < dx:
            hs = dx
        hinv = 1 / hs

        gxmin = max(int((xs[0] - hs) / dx + 1), 0)
        gxmax = min(int((xs[0] + hs) / dx), res - 1)
        gymin = max(int((xs[1] - hs) / dx + 1), 0)
        gymax = min(int((xs[1] + hs) / dx), res - 1)

        total_wt = 0
        for gx in range(gxmin, gxmax + 1):
            delta_x_Sqr = xs[0] - gx * dx
            delta_x_Sqr *= delta_x_Sqr
            for gy in range(gymin, gymax + 1):
                delta_y_Sqr = xs[1] - gy * dx
                delta_y_Sqr *= delta_y_Sqr
                r = np.sqrt(delta_x_Sqr + delta_y_Sqr)
                if r > hs:
                    continue
                q = r * hinv
                if q <= 0.5:
                    kernel = 1 - 6 * q * q * (1 - q)
                elif q <= 1.0:
                    a = 1 - q
                    kernel = 2 * a * a * a
                else:
                    continue
                total_wt += kernel

        if total_wt == 0:
            continue

        for gx in range(gxmin, gxmax + 1):
            delta_x_Sqr = xs[0] - gx * dx
            delta_x_Sqr *= delta_x_Sqr
            for gy in range(gymin, gymax + 1):
                delta_y_Sqr = xs[1] - gy * dx
                delta_y_Sqr *= delta_y_Sqr
                r = np.sqrt(delta_x_Sqr + delta_y_Sqr)
                if r > hs:
                    continue
                q = r * hinv
                if q <= 0.5:
                    kernel = 1 - 6 * q * q * (1 - q)
                elif q <= 1.0:
                    a = 1 - q
                    kernel = 2 * a * a * a
                else:
                    continue
                grid[gx, gy] += kernel * f[i] / total_wt

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


def Grid_PPZ_DataCube_Multigrid(
    f, x, h, center, size, z, h_z, res, box_size=-1, grid_res_kernel=8
):
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
        idx = (h / grid_res_kernel < res_bins[i]) & (
            h / grid_res_kernel >= res_bins[i + 1]
        )
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
            delta_x_Sqr = xs[0] - gx * dx
            delta_x_Sqr *= delta_x_Sqr
            for gy in range(gymin, gymax + 1):
                delta_y_Sqr = xs[1] - gy * dx
                delta_y_Sqr *= delta_y_Sqr
                q2dsq = (delta_x_Sqr + delta_y_Sqr) * hinvsq
                for gz in range(gzmin, gzmax + 1):
                    delta_z_Sqr = zs - gz * dz
                    delta_z_Sqr *= delta_z_Sqr
                    q = np.sqrt(q2dsq + delta_z_Sqr * h_z_invsq)
                    if q <= 0.5:
                        kernel = 1 - 6 * q * q + 6 * q * q * q
                    elif q <= 1.0:
                        kernel = 2 * (1 - q) * (1 - q) * (1 - q)
                    else:
                        continue
                    grid[gx, gy, gz] += (
                        2.546479089470325 * kernel * f_density
                    )  # Using 3D normalization
    return grid


@njit(fastmath=True)
def GridAverage(f, x, h, center, size, res=100, box_size=-1):
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
            delta_x_Sqr = xs[0] - gx * dx
            delta_x_Sqr *= delta_x_Sqr
            for gy in range(gymin, gymax + 1):
                delta_y_Sqr = xs[1] - gy * dx
                delta_y_Sqr *= delta_y_Sqr
                q = np.sqrt(delta_x_Sqr + delta_y_Sqr) * hinv
                if q <= 0.5:
                    kernel = 1 - 6 * q * q + 6 * q * q * q
                elif q <= 1.0:
                    kernel = 2 * (1 - q) * (1 - q) * (1 - q)
                else:
                    continue
                grid1[gx, gy] += kernel * h2
                grid2[gx, gy] += f[i] * kernel * h2
    return grid2 / grid1


@njit(fastmath=True)
def WeightedGridInterp3D(f, wt, x, h, center, size, res=100, box_size=-1):
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

    x3d = (
        x - center
    )  # coordinates in the grid frame, such that the origin is at the corner of the grid
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
        kval = np.empty(
            int(2 * hs / dx + 1) ** 3
        )  # save kernel values so don't have to recompute
        total_wt = 0
        j = 0
        for gx in range(gxmin, gxmax + 1):
            #            delta_x_Sqr = xs[0] - gx*dx
            delta_x_Sqr = xs[0] - gridcoords[gx]
            delta_x_Sqr *= delta_x_Sqr
            for gy in range(gymin, gymax + 1):
                #                delta_y_Sqr = xs[1] - gy*dx
                delta_y_Sqr = xs[1] - gridcoords[gy]
                delta_y_Sqr *= delta_y_Sqr
                for gz in range(gzmin, gzmax + 1):
                    delta_z_Sqr = xs[2] - gridcoords[gz]
                    #                    delta_z_Sqr = xs[2] - gz*dx
                    delta_z_Sqr *= delta_z_Sqr
                    q = np.sqrt(delta_x_Sqr + delta_y_Sqr + delta_z_Sqr) * hinv
                    if q > 1:
                        kernel = 0
                    elif q <= 0.5:
                        kernel = 1 - 6 * q * q + 6 * q * q * q
                    else:
                        kernel = 2 * (1 - q) * (1 - q) * (1 - q)
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
                        grid[gx % res, gy % res, gz % res] += (
                            f[i] * kernel * wt[i] / total_wt
                        )
                        gridwt[gx % res, gy % res, gz % res] += (
                            kernel * wt[i] / total_wt
                        )

    result = grid / gridwt
    #    do a loop through the grid to check for nans and address if needed
    pos = np.zeros(3)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for k in range(grid.shape[2]):
                if (
                    gridwt[i, j, k] == 0
                ):  # we have to search for the nearest neighbor and take its value - this is usually a rare edge case
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


@njit(fastmath=True)
def GridDensity(f, x, h, center, size, res=100, box_size=-1.0):
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

    x3d = (
        x - center + size / 2 - dx / 2
    )  # + dx/2 # coordinates in the grid frame, such that the origin is at the corner of the grid

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
        kval = np.empty(
            int(2 * hs / dx + 1) ** 3
        )  # save kernel values so don't have to recompute
        total_wt = 0
        j = 0
        for gx in range(gxmin, gxmax + 1):
            delta_x_Sqr = xs[0] - gx * dx
            delta_x_Sqr *= delta_x_Sqr
            for gy in range(gymin, gymax + 1):
                delta_y_Sqr = xs[1] - gy * dx
                delta_y_Sqr *= delta_y_Sqr
                for gz in range(gzmin, gzmax + 1):
                    delta_z_Sqr = xs[2] - gz * dx
                    delta_z_Sqr *= delta_z_Sqr
                    q = np.sqrt(delta_x_Sqr + delta_y_Sqr + delta_z_Sqr) * hinv
                    if q > 1:
                        kernel = 0
                    elif q <= 0.5:
                        kernel = 1 - 6 * q * q + 6 * q * q * q
                    else:
                        kernel = 2 * (1 - q) * (1 - q) * (1 - q)
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


@njit(fastmath=True)
def GridRadTransfer(lum, m, kappa, x, h, gridres, L, center=0, i0=0):
    """Simple radiative transfer solver

    Solves the radiative transfer equation with emission and absorption along a grid of sightlines, in multiple bands

    Parameters
    ----------
    lum: array_like
        shape (N, Nbands) array of particle luminosities in the different bands
    m: array_like
        shape (N,) array of particle masses
    kappa: array_like
        shape (N,) array of particle opacities (dimensions: length^2 / mass)
    x: array_like
        shape (N,3) array of particle positions)
    h: array_like
        shape (N,) array containing kernel radii of the particles
    center: array_like
        shape (3,) array containing the center coordinates of the image
    gridres: int
        image resolution
    L: float
        size of the image window in length units
    i0: array_like, optional
        shape (Nbands,) or (gridres,greidres,Nbands) array of background intensities

    Returns
    -------
    image: array_like
        shape (res,res) array of integrated intensities, in the units of your luminosity units / length units^2 / sr
    """

    #   don't have parallel working yet - trickier than simple surface density map because the order of extinctions and emissions matters
    # if ncores = -1:
    #     Nchunks = get_num_threads()
    # else:
    #     set_num_threads(ncores)
    #     Nchunks = ncores

    x -= center
    order = (
        -x[:, 2]
    ).argsort()  # get order for sorting by distance from observer - farthest to nearest

    lum, m, kappa, x, h = (
        np.copy(lum)[order],
        np.copy(m)[order],
        np.copy(kappa)[order],
        np.copy(x)[order],
        np.copy(h)[order],
    )

    Nbands = lum.shape[1]

    image = np.zeros((gridres, gridres, Nbands))
    image += i0 * 4 * np.pi  # factor of 4pi because we divide by that at the end

    dx = L / (gridres - 1)
    N = len(x)

    lh2 = np.empty(Nbands)
    k = np.empty(Nbands)
    for i in range(N):
        # unpack particle properties ##################
        xs = x[i] + L / 2
        hs = h[i]
        if hs == 0:
            continue
        for b in range(Nbands):  # unpack the brightness and opacity
            lh2[b] = lum[i, b] / (hs * hs)
            k[b] = kappa[i, b]
            if lh2[b] > 0 or k[b] > 0:
                skip = False
        if skip:
            continue
        if m[i] == 0:
            continue
        mh2 = m[i] / hs**2

        # done unpacking particle properties ##########

        gxmin = max(int((xs[0] - hs) / dx + 1), 0)
        gxmax = min(int((xs[0] + hs) / dx), gridres - 1)
        gymin = max(int((xs[1] - hs) / dx + 1), 0)
        gymax = min(int((xs[1] + hs) / dx), gridres - 1)

        for gx in range(gxmin, gxmax + 1):
            delta_x_Sqr = xs[0] - gx * dx
            delta_x_Sqr *= delta_x_Sqr
            for gy in range(gymin, gymax + 1):
                delta_y_Sqr = xs[1] - gy * dx
                delta_y_Sqr *= delta_y_Sqr
                r = delta_x_Sqr + delta_y_Sqr
                if r > hs * hs:
                    continue

                q = np.sqrt(r) / hs
                if q <= 0.5:
                    kernel = 1 - 6 * q * q * (1 - q)
                elif q <= 1.0:
                    a = 1 - q
                    kernel = 2 * a * a * a
                kernel *= 1.8189136353359467

                for b in range(Nbands):
                    image[gx, gy, b] += kernel * lh2[b]  # emission
                    tau = (
                        k[b] * kernel * mh2
                    )  # optical depth through the sightline through the particle
                    if tau == 0:
                        continue
                    if tau < 0.3:  # if optically thin use approximation
                        image[gx, gy, b] *= (1 - 0.5 * tau) / (1 + 0.5 * tau)
                    else:
                        image[gx, gy, b] *= np.exp(
                            -tau
                        )  # otherwise use the full (more expensive) solution

    return image / (4 * np.pi)
