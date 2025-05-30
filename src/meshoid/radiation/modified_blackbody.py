import numpy as np
from numpy.linalg import pinv
from numba import njit, prange


@njit(fastmath=True, error_model="numpy")
def planck_function(freqs, T):
    """
    Modified blackbody function:

    I = (1-exp(-tau(f))) B(T)

    where

    B(T) = 2 h f^3/c^2 / (exp(h f/(k_B T) -1 ))

    and tau(f) = tau0 (f/f0)^beta

    Parameters
    ----------
    freqs: array_like
        Photon frequencies in Hz
    logtau: float
        log10 of optical depth at 500um
    beta: float
        Dust spectral index beta
    T: float
        Temperature in K

    Returns
    -------
    I: array_like
        Intensity of modified blackbody in erg/s/cm^2/sr
    """
    h_over_kB, h, c = 4.799243073366221e-11, 6.62607015e-27, 29979245800.0
    I_thick = 2 * h / c**2 * freqs**3 / np.expm1(freqs * h_over_kB / T)
    return I_thick


@njit(fastmath=True, error_model="numpy")
def modified_planck_function(freqs, logtau, beta, T):
    """
    Modified blackbody function:

    I = (1-exp(-tau(f))) B(T)

    where

    B(T) = 2 h f^3/c^2 / (exp(h f/(k_B T) -1 ))

    and tau(f) = tau0 (f/f0)^beta

    Parameters
    ----------
    freqs: array_like
        Photon frequencies in Hz
    logtau: float
        log10 of optical depth at 500um
    beta: float
        Dust spectral index beta
    T: float
        Temperature in K

    Returns
    -------
    I: array_like
        Intensity of modified blackbody in erg/s/cm^2/sr
    """
    h_over_kB, h, c, f0 = (
        4.799243073366221e-11,
        6.62607015e-27,
        29979245800.0,
        599584916000.0,
    )
    tau0 = 10**logtau
    tauf = tau0 * (freqs / f0) ** beta
    taufac = -np.expm1(-tauf)
    I_thick = 2 * h / c**2 * freqs**3 / np.expm1(freqs * h_over_kB / T)
    return I_thick * taufac


@njit(fastmath=True, error_model="numpy")
def blackbody_residual_jacobian(freqs, sed_error, logtau, beta, logT):
    """Jacobian for least-squares fit to modified blackbody

    Parameters
    ----------
    freqs: array_like
        Photon frequencies in Hz
    sed_error: array_like
        SED errors
    logtau: float
        log10 of optical depth at 500um
    beta: float
        Dust spectral index beta
    logT: float
        log10 of temperature in K

    Returns
    -------
    jac: np.ndarray
        Shape (N,3) matrix of the gradient of the modified blackbody function
        with respect to logtau, beta, and logT
    """

    h = 6.62607015e-27
    c = 29979245800.0
    f0 = 599584916000.0
    k = 1.380649e-16
    T = 10**logT

    jac = np.empty((freqs.shape[0], 3))
    tau0 = 10**logtau
    for i in range(freqs.shape[0]):
        f = freqs[i]
        e_over_kT = f * h / (k * T)
        expfac = np.expm1(e_over_kT)
        tauf = tau0 * (f / f0) ** beta
        taufac = -np.expm1(-tauf)
        modbb = 2 * h / c**2 * f**3 * taufac / expfac
        jac[i, 2] = (np.log(10) * taufac * f**4 * h**2 / (c**2 * k * T * (np.cosh(e_over_kT) - 1))) / sed_error[i]
        jac[i, 1] = np.log(f / f0) * modbb / sed_error[i]
        jac[i, 0] = np.log(10) * modbb / sed_error[i]
    return jac


@njit(error_model="numpy", parallel=True)
def modified_blackbody_fit_image(image, image_error, wavelengths, fixed_beta=0):
    """Fit each pixel in a datacube to a modified blackbody

    Parameters
    ----------
    image: array_like
        Shape (N,N,num_bands) datacube of dust emission intensities
    image_error: array_like
        Shape (N,N,num_bands) datacube of dust emission intensity errors
    wavelengths: array_like
        Shape (num_bands,) array of wavelengths at which the SEDs are computed
    fixed_beta: float, optional
        If 0 (the default), the spectral index beta is allowed to vary freely as a fit parameter.
    Otherwise, fits are performed assuming this fixed value of beta.

    Returns
    -------
    params: np.ndarray
        Shape (N,N,3) array of best-fitting modified blackbody parameters:
            - tau: optical depth at 500 micron
            - beta: spectral index
            - T: temperature
    """
    res = (image.shape[0], image.shape[1])
    if fixed_beta:
        params = np.empty((res[0], res[1], 2))
        p0 = np.array([1e-3, 30.0])
    else:
        params = np.empty((res[0], res[1], 3))
        p0 = np.array([1e-3, 1.8, 30.0])

    for i in prange(res[0]):
        guess = np.copy(p0)
        for j in range(res[1]):
            params[i, j] = modified_blackbody_fit_gaussnewton(
                image[i, j], image_error[i, j], wavelengths, p0=guess, fixed_beta=fixed_beta
            )
            if not np.any(np.isnan(params[i, j])):
                guess = params[i, j]  # use previous pixel as next guess
            else:
                guess = p0
                params[i, j] = modified_blackbody_fit_gaussnewton(
                    image[i, j], image_error[i, j], wavelengths, p0=guess, fixed_beta=fixed_beta
                )
    return params


@njit(error_model="numpy")
def modified_blackbody_fit_gaussnewton(sed, sed_error, wavelengths, p0=(1.0, 1.5, 30.0), fixed_beta=0.0):
    """
    Fits a single SED to a modified blackbody using Gauss-Newton method

    Parameters
    ----------
    sed: array_like
        Shape (num_bands,) array of dust emission intensities
    sed: array_like
        Shape (num_bands,) array of dust emission intensity errors
    wavelengths:
        Shape (num_bands,) array of wavelengths at which the SEDs are computed
    p0: tuple, optional
        Shape (3,) tuple of initial guesses for tau(500um), beta, T
    fixed_beta: float, optional
        If 0 (the default), the spectral index beta is allowed to vary freely as a fit parameter.
    Otherwise, fits are performed assuming this fixed value of beta.

    Returns
    -------
    params: np.ndarray
        Shape (3,) array of best-fitting modified blackbody parameters:
            - tau(500um): optical depth at 500 micron
            - beta: spectral index
            - T: temperature
    """
    freqs = 299792458000000.0 / wavelengths

    max_iter = 30

    if fixed_beta:
        num_params = 2
        logtau, logT = np.log10(p0[0]), np.log10(p0[1])
        beta = fixed_beta
        params = np.array([logtau, logT])
    else:
        num_params = 3
        logtau, beta, logT = np.log10(p0[0]), p0[1], np.log10(p0[2])
        params = np.array([logtau, beta, logT])
    tol, i, error = 1e-15, 0, 1e100
    residual = (sed - modified_planck_function(freqs, logtau, beta, 10**logT)) / sed_error

    while error > tol and i < max_iter:
        if fixed_beta:
            beta = fixed_beta
            logtau, logT = params
        else:
            logtau, beta, logT = params

        try:
            jac = blackbody_residual_jacobian(freqs, sed_error, logtau, beta, logT)
            if fixed_beta:
                jac = jac[:, ::2]
            jac_inv = pinv(jac)
        except:
            return np.nan * np.ones(num_params)

        dx = jac_inv @ residual

        res_new = 1e100
        res_old = (residual * residual).sum()
        alpha = 1.0
        while res_new > res_old:
            x_new = params + alpha * dx
            if fixed_beta:
                I = modified_planck_function(freqs, x_new[0], fixed_beta, 10 ** x_new[1])
            else:
                I = modified_planck_function(freqs, x_new[0], x_new[1], 10 ** x_new[2])
            residual = sed - I
            residual /= sed_error
            res_new = (residual * residual).sum()
            alpha *= 0.5
        params = x_new
        error = np.abs(params[0] - logtau)
        i += 1
        if i > max_iter:
            return np.nan * np.ones(num_params)
    params[0] = 10 ** params[0]
    params[num_params - 1] = 10 ** params[num_params - 1]
    if not fixed_beta:
        if params[1] < 0:
            params[1] = np.nan
    return params
