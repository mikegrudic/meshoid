"""Routines for radiative transfer calculations"""

from numba import njit, prange, get_num_threads
import numpy as np
from astropy import constants
import astropy.units as u
from ..kernel_density import *
import os


@njit(fastmath=True, error_model="numpy", parallel=True)
def radtransfer(j, m, kappa, x, h, gridres, L, center=0, i0=0):
    """Simple radiative transfer solver

    Solves the radiative transfer equation with emission and absorption along a
    grid of sightlines, at multiple frequencies. Raytraces in parallel, with number
    of threads determined by numba set_num_threads

    Parameters
    ----------
    j: array_like
        shape (N, num_freqs) array of particle emissivities per unit mass (e.g.
        erg/s/g/Hz)
    m: array_like
        shape (N,) array of particle masses
    kappa: array_like
        shape (N,) array of particle opacities (dimensions: length^2 / mass)
    x: array_like
        shape (N,3) array of particle positions
    h: array_like
        shape (N,) array containing kernel radii of the particles
    gridres: int
        image resolution
    center: array_like
        shape (3,) array containing the center coordinates of the image
    L: float
        size of the image window in length units
    i0: array_like, optional
        shape (num_freqs,) or (gridres,greidres,num_freqs) array of background
        intensities

    Returns
    -------
    image: array_like
        shape (res,res) array of integrated intensities, in your units of
        power / length^2 / sr / frequency (this is the quantity I_ν in RT)
    """

    Nchunks = get_num_threads()
    num_bands = j.shape[1]

    # sort by z then split into chunks
    # print("Sorting...")
    order = (x[:, 2]).argsort()
    # print("Chunking...")
    j, m, kappa, x, h = (
        np.array_split(j[order], Nchunks, 0),
        np.array_split(m[order], Nchunks, 0),
        np.array_split(kappa[order], Nchunks, 0),
        np.array_split(x[order], Nchunks, 0),
        np.array_split(h[order], Nchunks, 0),
    )
    # j, m, kappa, x, h = (
    #     np.split(j,Nchunks,0)
    #     np.split(m,Nchunks,0),
    # np.split(kappa,Nchunks,0), np.split(x,Nchunks,0), np.split(h,Nchunks,0)

    intensity = np.empty((Nchunks, gridres, gridres, num_bands))
    tau = np.empty_like(intensity)
    # parallel part, raytracing the individual slabs
    # print("Raytracing...")
    for c in prange(Nchunks):
        intensity[c], tau[c] = radtransfer_singlethread(j[c], m[c], kappa[c], x[c], h[c], gridres, L, center, i0)

    tau_total = np.zeros((gridres, gridres, num_bands))
    I_total = np.zeros((gridres, gridres, num_bands))

    # now we add up total extinction and total integrated intensity from last
    # chunk to first, attenuating each successive slab by the running extinction
    # print("summing slabs...")
    for c in range(Nchunks - 1, -1, -1):
        I_total += intensity[c] * np.exp(-tau_total)
        tau_total += tau[c]
    return I_total


@njit(fastmath=True, error_model="numpy")
def radtransfer_singlethread(j, m, kappa, x, h, gridres, L, center=0, i0=0):
    """Simple radiative transfer solver

    Solves the radiative transfer equation with emission and absorption along a
    grid of sightlines, at multiple frequencies

    Parameters
    ----------
    j: array_like
        shape (N, num_freqs) array of particle emissivities per unit mass (e.g.
        erg/s/g/Hz)
    m: array_like
        shape (N,) array of particle masses
    kappa: array_like
        shape (N,) array of particle opacities (dimensions: length^2 / mass)
    x: array_like
        shape (N,3) array of particle positions
    h: array_like
        shape (N,) array containing kernel radii of the particles
    gridres: int
        image resolution
    center: array_like
        shape (3,) array containing the center coordinates of the image
    L: float
        size of the image window in length units
    i0: array_like, optional
        shape (num_freqs,) or (gridres,greidres,num_freqs) array of background
        intensities
    return_taumap: boolean, optional
        Whether to return the optical depth map alongside the intensity map
        (default False)

    Returns
    -------
    intensity, tau: array_like, array_like
        shape (res,res,num_bands) array of integrated intensity, in your units of
        power / length^2 / sr / frequency (this is the quantity I_ν in RT), and
        of optical depth in the different bands along the lines of sight
    """

    x -= center
    # get order for sorting by distance from observer - farthest to nearest
    order = (-x[:, 2]).argsort()
    j, m, kappa, x, h = (
        np.copy(j)[order],
        np.copy(m)[order],
        np.copy(kappa)[order],
        np.copy(x)[order],
        np.copy(h)[order],
    )

    num_freqs = j.shape[1]

    intensity = np.zeros((gridres, gridres, num_freqs))
    intensity += i0  # * 4 * np.pi  # factor of 4pi because we divide by that at the end
    taumap = np.zeros_like(intensity)
    dx = L / (gridres - 1)
    N = len(x)

    j_over_4pi_kappa_i = np.empty(num_freqs)
    kappa_i = np.empty(num_freqs)
    for i in range(N):
        # unpack particle properties that will be the same for each grid point
        xs = x[i] + L / 2
        hs = max(h[i], dx)
        if hs == 0 or m[i] == 0:
            continue

        for b in range(num_freqs):  # unpack the brightness and opacity
            kappa_i[b] = kappa[i, b]
            j_over_4pi_kappa_i[b] = j[i, b] / (4 * np.pi * kappa_i[b] + 1e-100)

        mh2 = m[i] / hs**2

        # done unpacking particle properties ##########

        # determine bounds of grid indices
        gxmin = max(int((xs[0] - hs) / dx + 1), 0)
        gxmax = min(int((xs[0] + hs) / dx), gridres - 1)
        gymin = max(int((xs[1] - hs) / dx + 1), 0)
        gymax = min(int((xs[1] + hs) / dx), gridres - 1)

        for gx in range(gxmin, gxmax + 1):
            delta_x_sqr = xs[0] - gx * dx
            delta_x_sqr *= delta_x_sqr
            for gy in range(gymin, gymax + 1):
                delta_y_sqr = xs[1] - gy * dx
                delta_y_sqr *= delta_y_sqr
                r = delta_x_sqr + delta_y_sqr
                if r > hs * hs:
                    continue

                q = np.sqrt(r) / hs
                kernel = kernel2d(q)
                for b in range(num_freqs):
                    # optical depth through the sightline through the particle
                    tau = kappa_i[b] * kernel * mh2
                    fac1 = np.exp(-tau)
                    fac2 = -np.expm1(-tau)
                    taumap[gx, gy, b] += tau
                    intensity[gx, gy, b] = fac1 * intensity[gx, gy, b] + fac2 * j_over_4pi_kappa_i[b]
    return intensity, taumap
