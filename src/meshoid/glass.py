"""Routine for generating a uniform-density glass configuration consistent with meshoid's density estimator."""

import numpy as np
from .Meshoid import Meshoid
from scipy.optimize import minimize_scalar


def particle_glass(N: int = 64**3, L: float = 1.0, dim: int = 3, tol: float = 1e-2, optimize=False) -> np.ndarray:
    """Returns coordinate positions of particles in a uniform-density glass

    Parameters
    ----------
    N: int, optional
        Number of particles (default 64^3)
    L: float, optional
        Box side length (default 1.)c
    dim: int, optional
        Dimensionality of the box (default 3)
    tol: float, optional
        Tolerance for the RMS density variation (default 1e-2)

    Returns
    -------
    coords: ndarray
        Shape (N,dim) array of coordinate positions of the particle glass arrangement
    """
    x = np.random.rand(N, dim)
    relax_particles(x, tol)
    return x * L


def relax_particles(coordinates: np.ndarray, tol: float = 1e-2, optimize=False) -> np.ndarray:
    """
    Iteratively relax an input particle arrangement towards a uniform-density glass configuration
    in the unit cube.

    Parameters
    ----------
    coordinates: ndarray
        Initial array of coordinates within the unit cube. This is overwritten in place.
    tol: float, optional
        Tolerance for RMS density variation (default 1e-2)
    optimize: boolean, optional
        Force every iteration to choose the optimal displacement against the density gradient that locally
        minimizes the RMS density variation (default False)
    """
    if np.any(coordinates > 1):
        raise ValueError("input coordinates to relaxation routine are not inside the unit cube.")

    N = len(coordinates)
    hfrac = 0.5
    i = 0
    wt = np.repeat(1.0 / N, N)
    tree_new = Meshoid(coordinates)
    while True:
        M = tree_new  # Meshoid(coordinates, wt, boxsize=1)
        rho = M.Density()
        rho_std = rho.std()
        print(hfrac, rho.std(), rho.mean())
        dx = (wt / rho) ** (1.0 / 3)
        gradrho = M.D(rho)
        gradrho_norm = np.sum(gradrho * gradrho, 1) ** 0.5

        def new_coords(hfrac):
            return (coordinates - (hfrac * dx / gradrho_norm)[:, None] * gradrho) % 1.0

        def new_density_std(hfrac):
            return Meshoid(new_coords(hfrac), boxsize=1).Density().std()

        coords_new = new_coords(hfrac)
        tree_new = Meshoid(coords_new, boxsize=1.0)
        if not optimize and (tree_new.Density().std() < rho_std * (1 - 0.3 * rho_std / rho.mean())):
            hfrac *= 0.9
        else:
            sol = minimize_scalar(new_density_std, [hfrac / 10, hfrac * 10], tol=1e-1)
            print(sol.x, rho.std())
            hfrac = sol.x
            coords_new = new_coords(hfrac)
            tree_new = Meshoid(coords_new, boxsize=1.0)

        coordinates = coords_new  # (hfrac * dx / gradrho_norm)[:, None] * gradrho

        if rho.std() < tol * rho.mean():
            break

        i += 1

    coordinates -= coordinates[0] - 0.5  # center on 0'th particle
    coordinates %= 1.0
