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


def GridSurfaceDensityMultigrid(
    f, x, h, center, size, res=128, box_size=-1, N_grid_kernel=8, parallel=False
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


@njit(fastmath=True)
def GridSurfaceDensity_core(f, x, h, center, size, res=100, box_size=-1):
    """
    Computes the surface density of conserved quantity f colocated at positions x with smoothing lengths h. E.g. plugging in particle masses would return mass surface density. The result is on a Cartesian grid of sightlines, the result being the density of quantity f integrated along those sightlines.

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
    f, x, h, center, size, z, h_z, res, box_size=-1, N_grid_kernel=8
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
