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
    order = (-x[:, 2]).argsort()
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
        intensity[c], tau[c] = radtransfer_singlethread(
            j[c], m[c], kappa[c], x[c], h[c], gridres, L, center, i0
        )

    tau_total = np.zeros((gridres, gridres, num_bands))
    I_total = np.zeros((gridres, gridres, num_bands))

    # now we add up total extinction and total integrated intensity from last
    # chunk to first, attenuating each successive slab by the running extinction
    # print("summing slabs...")
    for c in range(Nchunks-1,-1,-1):
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
            j_over_4pi_kappa_i[b] = j[i, b] / (4 * np.pi * kappa_i[b])
            if j_over_4pi_kappa_i[b] > 0 or kappa_i[b] > 0:
                skip = False
        if skip:
            continue

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
                    intensity[gx, gy, b] = (
                        fac1 * intensity[gx, gy, b] + fac2 * j_over_4pi_kappa_i[b]
                    )
    return intensity, taumap


HERSCHEL_DEFAULT_WAVELENGTHS = np.array([150, 250, 350, 500])


def dust_abs_opacity(
    wavelength_um: np.ndarray = HERSCHEL_DEFAULT_WAVELENGTHS, XH=0.71, Zd=1.0
) -> np.ndarray:
    """Returns the dust+PAH absorption opacity in cm^2/g at a set of wavelengths
    in micron

    Parameters
    ----------
    wavelength_um: array_like
        Shape (num_bands) array of wavelengths in micron
    XH: float, optional
        Mass fraction of hydrogen (needed to convert from per H to per g)
    Zd: float, optional
        Dust-to-gas ratio normalized to Solar neighborhood

    Returns
    -------
    kappa: array_like
        shape (num_bands,) array of opacities in cm^2/g
    """
    data = np.loadtxt(
        os.path.dirname(os.path.abspath(__file__))
        + "/hensley_draine_2022_astrodust_opacity.dat"
    )
    wavelength_grid, kappa_abs_PAH, kappa_abs_astrodust = data[:, :3].T
    kappa_abs = kappa_abs_PAH + kappa_abs_astrodust
    kappa_per_H = np.interp(wavelength_um, wavelength_grid, kappa_abs)
    return kappa_per_H * (XH / (constants.m_p)).cgs.value * Zd


def thermal_emissivity(kappa, T, wavelengths_um=HERSCHEL_DEFAULT_WAVELENGTHS):
    """
    Returns the thermal emissivity j_ν = 4 pi kappa_ν B_ν(T) in erg/s/g for a
    specified list of wavelengths, temperatures, and opacities defined at those
    wavelengths

    Parameters
    ----------
    kappa: array_like
        shape (N,num_bands) array of opacities
    T: array_like
        shape (N,) array of temperatures
    wavelengths: array_like
        shape (num_bands,) array of wavelengths

    Returns
    -------
    j: array_like
        shape (N,num_bands) array of thermal emissivities
    """
    h, c, k_B = constants.h, constants.c, constants.k_B
    freqs = c / (wavelengths_um * u.si.micron)
    B = (
        2
        * h
        * freqs[None, :] ** 3
        * c**-2
        / np.expm1(h * freqs[None, :] / (k_B * T[:, None] * u.K))
    )
    return (4 * np.pi * kappa * u.cm**2 / u.g * B).cgs.value


def dust_emission_map(
    x_pc,
    m_msun,
    h_pc,
    Tdust,
    size_pc,
    res,
    wavelengths_um=HERSCHEL_DEFAULT_WAVELENGTHS,
    center_pc=0,
    parallel=True,
) -> np.ndarray:
    """Generates a map of dust emission in cgs units for specified wavelengths,
    neglecting scattering (OK for FIR/submm wavelengths)

    Parameters
    ----------
    x_pc: array_like
        Shape (N,3) array of coordinates in pc
    m_msun: array_like
        Shape (N,) array of masses in msun
    h_pc: array_like
        Shape (N,) array of kernel radii in pc
    Tdust: array_like
        Shape (N,) array of dust temperatures in K
    wavelengths_um: array_like
        Shape (num_bands,) array of wavelengths in micron
    size_pc: float
        Size of the image in pc
    res: int
        Image resolution
    center_pc: array_like, optional
        Shape (3,) array providing the coordinate center of the image

    Returns
    -------
    intensity: array_like
        shape (res,res,num_bands) datacube of dust emission intensity
        in erg/s/cm^2/sr
    """
    kappa = dust_abs_opacity(wavelengths_um)
    kappa = np.array(len(x_pc) * [kappa])
    j = thermal_emissivity(kappa, Tdust, wavelengths_um)
    m_cgs = m_msun * (constants.M_sun.cgs.value)
    pc_to_cm = constants.pc.cgs.value
    intensity = radtransfer(
        j,
        m_cgs,
        kappa,
        x_pc * pc_to_cm,
        h_pc * pc_to_cm,
        res,
        size_pc * pc_to_cm,
        center_pc * pc_to_cm,
    )
    return intensity
