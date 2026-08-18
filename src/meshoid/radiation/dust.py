import numpy as np
from astropy import constants
import astropy.units as u
from .radtransfer import radtransfer
from .blackbody import planck_function, planck_function_dT
import os

HERSCHEL_DEFAULT_WAVELENGTHS = np.array([150, 250, 350, 500]) * u.um


def dust_abs_opacity(
    wavelength,
    Zd=1.0,
    model="Semenov03",
    Tdust=0,
    hydrogen_massfrac=0.71,
    submodel="Multishell_spheres",
) -> u.quantity.Quantity:
    """
    Returns the dust absorption opacity in cm^2/g at a set of wavelengths

    Parameters
    ----------
    wavelength: array_like
        Shape (num_bands) array-like of wavelengths; if a raw float array, microns are assumed, but will take an astropy quantity.
    hydrogen_massfrac: float, optional
        Mass fraction of hydrogen (needed to convert from per H to per g)
    Zd: float, optional
        Dust-to-gas ratio normalized to Solar neighborhood
    model: str, optional
        Which dust model to use: choose from astrodust (Hensley & Draine 2022) or Semenov03 (Semenov et al 2003). Semenov03
        will account for sublimation if supplied with Tdust. If Tdust is an array, will return
    Tdust: float or array_like, optional
        Scalar or shape (N,) array of dust temperatures used to determine dust composition; if array, output will be
        broadcast to shape (N, num_bands). Defaults to 0, which assumes no sublimation has taken place.
    submodel: string, optional
        When specifying Semenov03 model, choose between Comp_aggregates, Comp_spheres, Hom_aggregates, Hom_spheres,
        Multishell_spheres, Porous_comp_spheres, Porous_multishell_spheres

    Returns
    -------
    kappa: array_like
        shape (num_bands,) array of opacities in cm^2/g if Tdust is None or a scalar, or shape (len(Tdust),num_bands)
        if Tdust is an array (broadcasting over Tdust and wavelength)
    """

    if isinstance(wavelength, u.quantity.Quantity):
        wavelength = wavelength.to(u.um).value
    if isinstance(Tdust, u.quantity.Quantity):
        Tdust = Tdust.to(u.K).value

    path0 = os.path.dirname(os.path.abspath(__file__)) + "/opacity_tables/"
    if model.lower() == "astrodust":
        data = np.loadtxt(path0 + "hensley_draine_2022_astrodust_opacity.dat")
        wavelength_grid, kappa_abs_PAH, kappa_abs_astrodust = data[:, :3].T
        kappa_abs = kappa_abs_PAH + kappa_abs_astrodust
        kappa_per_H = np.interp(wavelength, wavelength_grid, kappa_abs)
        return kappa_per_H * (hydrogen_massfrac / (constants.m_p)).cgs.value * Zd * u.cm**2 / u.g
    elif model.lower() == "semenov03":
        data = np.loadtxt(path0 + f"semenov2003/{submodel}/1.dat")
        if Tdust is None:
            wavelength_grid, kappa_grid = data[::-1].T
        else:
            wavelength_grid = data[::-1, 0]  # note reversal because interp arguments must be sorted
            kappa_grid = [np.loadtxt(path0 + f"semenov2003/{submodel}/{i}.dat")[::-1, 1] for i in range(1, 6)]
            kappa_grid.append(np.zeros_like(kappa_grid[0]))
            kappa_grid = np.array(kappa_grid)
        # Nx5 table of opacities, need to choose based on temperature range

        # establish bins for sublimation temperatures of different components at 10^-10 g cm^-3; could extend to density-dependent temps
        Tbin_edges = [0, 160, 275, 425, 680, 1500, np.inf]
        Tbin = np.clip(np.digitize(Tdust, Tbin_edges) - 1, 0, len(Tbin_edges) + 1)

        if Tdust is None:
            kappa = np.interp(wavelength, wavelength_grid, kappa_grid)
        elif np.isscalar(Tdust):  # only need to lookup for one dust temp
            kappa_grid = kappa_grid[Tbin]
            kappa = np.interp(wavelength, wavelength_grid, kappa_grid)
        else:  # need to used Tdust-binned
            kappa = np.zeros((len(Tdust), len(wavelength)))
            for i in range(len(Tbin_edges) - 1):
                idx = Tbin == i
                if not np.any(idx):
                    continue
                kappa_interp = np.interp(wavelength, wavelength_grid, kappa_grid[i])
                kappa[idx] = kappa_interp[None, :]
        kappa *= u.cm**2 / u.g
        return kappa

    else:
        raise NotImplementedError("Specified dust model not implemented.")


def dust_mean_opacity(
    Trad,
    Tdust,
    which="planck",
    Zd=1.0,
    model="Semenov03",
    submodel="Multishell_spheres",
    hydrogen_massfrac=0.71,
    num_wavelengths=10**4,
):
    """
    Returns the mean absorption opacity in cm^2/g at a set of radiation and dust temperatures

    Parameters
    ----------
    Trad: float or array_like
        Scalar or shape (N,) array of radiation temepratures; if array, output will be broadcast to shape (N, num_bands).
    Tdust: float or array_like
        Scalar or shape (N,) array of dust temperatures used to determine dust composition. Should match Trad or
        be broadcast to it.
    which: string, optional
        Which mean to compute out of 'planck' (rosseland to be implemented)
    Zd: float, optional
        Dust-to-gas ratio normalized to Solar neighborhood value (default 1.)
    hydrogen_massfrac: float, optional
        Mass fraction of hydrogen (needed to convert from per H to per g) (default 0.71)
    model: str, optional
        Which dust model to use: choose from 'astrodust' (Hensley & Draine 2022) or 'Semenov03' (Semenov et al 2003).
        Semenov03 will account for sublimation of volatiles if supplied with Tdust.
    submodel: string, optional
        When specifying Semenov03 model, choose between Comp_aggregates, Comp_spheres, Hom_aggregates, Hom_spheres,
        Multishell_spheres, Porous_comp_spheres, Porous_multishell_spheres

    Returns
    -------
    kappa: array_like

    """

    Trad = np.atleast_1d(np.float64(Trad))
    Tdust = np.atleast_1d(np.float64(Tdust))
    if len(Tdust) == 1:
        Tdust = np.repeat(Tdust, len(Trad))

    if len(Tdust) > 1 and len(Tdust) != len(Trad):
        raise ValueError("Tdust could not be broadcast to Trad")

    wavelength_grid = np.logspace(4, -1, num_wavelengths)

    # temperature regions for dust opacity - must do these separately
    Tbin_edges = [0, 160, 275, 425, 680, 1500, np.inf]
    Tbin = np.clip(np.digitize(Tdust, Tbin_edges) - 1, 0, len(Tbin_edges) + 1)
    kappa_mean = np.empty_like(Trad)

    for bin in np.unique(Tbin):
        idx = Tbin == bin
        monochrome_opacity = dust_abs_opacity(
            wavelength_grid, Zd, model, Tdust[idx][0], hydrogen_massfrac, submodel
        ).value
        freq = (constants.c / (wavelength_grid * u.um)).to(u.Hz)  # .value
        if which == "planck":

            def mean(T):
                return np.trapezoid(planck_function(freq, T) * monochrome_opacity, freq) / np.trapezoid(
                    planck_function(freq, T), freq
                )

        elif which == "rosseland":

            def mean(T):
                #                print(planck_function_dT(freq, T))
                return (
                    np.trapezoid(planck_function_dT(freq, T) / monochrome_opacity, freq)
                    / np.trapezoid(planck_function_dT(freq, T), freq)
                ) ** -1.0

        else:
            raise NotImplementedError(f"Opacity mean {which} not implemented.")

        kappa_mean[idx] = np.array([mean(T) for T in Trad[idx]])
    return kappa_mean


def dust_ext_opacity(
    wavelength_um: np.ndarray = HERSCHEL_DEFAULT_WAVELENGTHS, hydrogen_massfrac=0.71, Zd=1.0
) -> u.quantity.Quantity:
    """
    Returns the dust extinction opacity in cm^2/g at a set of wavelengths
    in micron

    Parameters
    ----------
    wavelength_um: array_like
        Shape (num_bands) array of wavelengths in micron
    hydrogen_massfrac: float, optional
        Mass fraction of hydrogen (needed to convert from per H to per g)
    Zd: float, optional
        Dust-to-gas ratio normalized to Solar neighborhood

    Returns
    -------
    kappa: array_like
        shape (num_bands,) array of opacities in cm^2/g
    """
    data = np.loadtxt(
        os.path.dirname(os.path.abspath(__file__)) + "/opacity_tables/hensley_draine_2022_astrodust_opacity.dat"
    )
    wavelength_grid, kappa_ext = data[:, 0], data[:, -2]
    kappa_per_H = np.interp(wavelength_um, wavelength_grid, kappa_ext)
    return kappa_per_H * (hydrogen_massfrac / (constants.m_p)).cgs.value * Zd * u.cm**2 / u.g


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
    wavelengths_um: array_like
        shape (num_bands,) array of wavelengths in microns

    Returns
    -------
    j: array_like
        shape (N,num_bands) array of thermal emissivities
    """
    h, c, k_B = constants.h, constants.c, constants.k_B
    freqs = c / (wavelengths_um * u.si.micron)
    B = 2 * h * freqs[None, :] ** 3 * c**-2 / np.expm1(h * freqs[None, :] / (k_B * T[:, None] * u.K))
    return (4 * np.pi * kappa * u.cm**2 / u.g * B).cgs.value


def dust_emission_map(
    x_pc, m_msun, h_pc, Tdust, size_pc, res, wavelengths_um=HERSCHEL_DEFAULT_WAVELENGTHS, center_pc=0, model="Semenov03"
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
        in erg/s/cm^2/sr/Hz
    """
    kappa = dust_abs_opacity(wavelengths_um, Tdust=Tdust, model=model)
    # kappa = np.array(len(x_pc) * [kappa])
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
