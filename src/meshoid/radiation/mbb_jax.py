"""JAX implementation of modified blackbody model"""

import jax.numpy as jnp


def modified_planck_function_jax(freqs, logtau, beta, T):
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
    tau0 = 10.0**logtau
    tauf = tau0 * (freqs / f0) ** beta
    taufac = -jnp.expm1(-tauf)
    I_thick = 2 * h / c**2 * freqs * freqs * freqs / jnp.expm1(freqs * h_over_kB / T)
    return I_thick * taufac
