"""Routine for generating a uniform-density glass configuration consistent with meshoid's density estimator."""

import numpy as np
from .Meshoid import Meshoid
from scipy.optimize import minimize_scalar
from scipy.stats import qmc


def tesselate_glass_coords(x):
    """Tesselates 8 rescaled copies of glass coordinates to make a glass with 8x more particles"""
    x = np.concatenate(
        [
            x / 2 + i * np.array([0.5, 0, 0]) + j * np.array([0, 0.5, 0]) + k * np.array([0, 0, 0.5])
            for i in range(2)
            for j in range(2)
            for k in range(2)
        ]
    )
    return x


def particle_glass(
    N: int = 64**3,
    L: float = 1.0,
    dim: int = 3,
    tol: float = 1e-3,
    optimize=False,
    gradient="hybrid",
    num_steps=np.inf,
    sobol=False,
) -> np.ndarray:
    """Returns coordinate positions of particles in a uniform-density glass

    Parameters
    ----------
    N: int, optional
        Number of particles (default 64^3)
    L: float, optional
        Box side length (default 1.)
    dim: int, optional
        Dimensionality of the box (default 3)
    tol: float, optional
        Tolerance for the RMS density variation (default 1e-3)

    Returns
    -------
    coords: ndarray
        Shape (N,dim) array of coordinate positions of the particle glass arrangement
    """
    if sobol and (N & (N - 1)) == 0:  # checks if we have a power-of-two and can do a sobol sequence for starters
        sampler = qmc.Sobol(d=3, scramble=False)
        x = sampler.random_base2(m=int(np.log2(N)))
    elif gradient == "hybrid":
        x = particle_glass(N, 1.0, dim, max(1e-2, 2 * tol), optimize=optimize, gradient="sph", sobol=sobol)
        gradient = "leastsquares"
    else:
        x = np.random.rand(N, dim)
    relax_particles(x, tol, optimize=optimize, gradient=gradient, num_steps=num_steps)
    return x * L


def relax_particles(
    coordinates: np.ndarray, tol: float = 1e-2, optimize=False, gradient="sph", num_steps=np.inf
) -> np.ndarray:
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
    if np.any(coordinates > 1) or np.any(coordinates < 0):
        raise ValueError(
            f"input coordinates to relaxation routine are not inside the unit cube. max={coordinates.max()} min={coordinates.min()}"
        )

    hfrac = 0.5
    tree_new = Meshoid(coordinates, boxsize=1.0)
    wt = 1 / len(coordinates)
    n_iter = 0
    while n_iter < num_steps:
        rho = tree_new.Density()
        rho_std = rho.std()
        print(f"iter={n_iter} RMS density variation: {rho_std}", end="\r")
        cell_size = (wt / rho) ** (1.0 / 3)
        match gradient:
            case "leastsquares":
                gradrho = tree_new.D(rho)
            case "sph":
                gradrho = tree_new.SPH_Density_Gradient()
            case "random":
                gradrho = tree_new.D(rho) if np.random.rand() > 0.5 else tree_new.SPH_Density_Gradient()
            case "alternate":
                if n_iter % 2:
                    gradrho = tree_new.SPH_Density_Gradient()
                else:
                    gradrho = tree_new.D(rho)
            case _:
                raise ValueError(f"Gradient {gradient} not implemented")

        gradrho_norm = np.sum(gradrho * gradrho, 1) ** 0.5

        def new_coords(hfrac):
            dx = -(hfrac * cell_size / gradrho_norm)[:, None] * gradrho
            return (coordinates + dx) % 1.0

        def new_density_std(hfrac):
            return Meshoid(new_coords(hfrac), boxsize=1).Density().std()

        coords_new = new_coords(hfrac)
        tree_new = Meshoid(coords_new, boxsize=1.0)
        if not optimize and (tree_new.Density().std() < rho_std * (1 - 0.3 * rho_std / rho.mean())):
            hfrac *= 0.9
        else:
            sol = minimize_scalar(new_density_std, [hfrac / 10, hfrac * 10], tol=1e-1)
            hfrac = sol.x
            coords_new = new_coords(hfrac)
            tree_new = Meshoid(coords_new, boxsize=1.0)

        coordinates[:] = coords_new

        if rho.std() < tol * rho.mean():
            break
        n_iter += 1

    coordinates[:] = coordinates - (coordinates[0] - 0.5)  # center on 0'th particle
    coordinates[:] = coordinates % 1.0
