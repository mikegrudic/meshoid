"""Functions for computing the matrices weights for least-squares derivative operators"""

from numba import njit, prange, vectorize
import numpy as np
from .kernel_density import Kernel


@njit(fastmath=True)
def nearest_image(dx_coord, boxsize):
    """Returns separation vector for nearest image, given the coordinate
    difference dx_coord and assuming coordinates run from 0 to boxsize"""
    if np.abs(dx_coord) > boxsize / 2:
        return -np.copysign(boxsize - np.abs(dx_coord), dx_coord)
    return dx_coord


@vectorize
def nearest_image_v(dx_coord, boxsize):
    """Returns separation vector for nearest image, given the coordinate
    difference dx_coord and assuming coordinates run from 0 to boxsize"""
    if np.abs(dx_coord) > boxsize / 2:
        return -np.copysign(boxsize - np.abs(dx_coord), dx_coord)
    return dx_coord


@njit(fastmath=True)
def kernel_dx_and_weights(
    i: int,
    pos: np.ndarray,
    ngb: np.ndarray,
    kernel_radius: float,
    boxsize=None,
    weighted: bool = True,
):
    """Computes coordinate differences and weights for the neighbors in the
    kernel of particle i.
    """
    num_ngb = ngb.shape[0]
    dim = pos.shape[1]
    dx = np.zeros((num_ngb, dim))
    weights = np.ones(num_ngb)
    for j in range(num_ngb):
        r = 0
        n = ngb[j]
        for k in range(dim):
            dx[j, k] = pos[n, k] - pos[i, k]
            if boxsize is not None:
                dx[j, k] = nearest_image(dx[j, k], boxsize)
            r += dx[j, k] * dx[j, k]
        if weighted:
            weights[j] = Kernel(np.sqrt(r) / kernel_radius)
    return dx, weights


@njit(fastmath=True)
def polyfit_leastsq_matrices(dx: np.ndarray, weights: np.ndarray, order: int = 1):
    """
    Return the left-hand side and right-hand side matrices for the system of
    equations for a weighted least-squares fit to a polynomial needed to
    compute the least-squares gradient matrices:

    (A^T W A) X = (A^T W) Y

    where A is the polynomial Vandermonde matrix, W are the weights, X
    Y are the differences in function values, and X are the unknown derivative
    components we wish to solve for.
    """
    num_ngb, dim = dx.shape
    if order == 1:
        lhs_matrix = np.zeros((dim, dim))
        rhs_matrix = np.empty((dim, num_ngb))
        for j in range(num_ngb):
            for k in range(dim):
                rhs_matrix[k, j] = weights[j] * dx[j, k]
        for j in range(num_ngb):
            for k in range(dim):
                for l in range(dim):
                    lhs_matrix[k, l] += rhs_matrix[k, j] * dx[j, l]
    else:  # 2nd order system
        num_derivs = get_num_derivs(dim, order)
        lhs_matrix = np.zeros((num_derivs, num_derivs))
        rhs_matrix = np.empty((num_derivs, num_ngb))
        for j in range(num_ngb):
            for k in range(num_derivs):
                if k < dim:  # 1st derivatives
                    rhs_matrix[k, j] = dx[j, k]
                elif k < 2 * dim:  # pure 2nd derivatives
                    d = k - dim
                    rhs_matrix[k, j] = 0.5 * dx[j, d] * dx[j, d]
                else:  # this does the cross-terms, e.g. xy, xz, yz
                    d = k - 2 * dim
                    rhs_matrix[k, j] = dx[j, d % dim] * dx[j, (d + 1) % dim]

        for j in range(num_ngb):
            for k in range(num_derivs):
                for l in range(num_derivs):
                    lhs_matrix[k, l] += rhs_matrix[k, j] * rhs_matrix[l, j] * weights[j]

        for j in range(num_ngb):
            for k in range(num_derivs):
                rhs_matrix[k, j] *= weights[j]

    return lhs_matrix, rhs_matrix


@njit
def get_num_derivs(dim: int, order: int) -> int:
    """Returns number of unique derivatives to compute for a given matrix order
    and dimension
    """
    if order == 1:
        return dim
    else:
        return {1: 2, 2: 5, 3: 9}[dim]


@njit(parallel=True, fastmath=True, error_model="numpy")
def gradient_weights(pos, ngb, kernel_radius, indices, boxsize=None, weighted=True, order=1):
    """Computes the N_particles (dim x N_ngb) matrices that encode the least-
    squares gradient operators
    """
    dim = pos.shape[1]
    N = indices.shape[0]
    num_ngb = ngb.shape[1]
    num_derivs = get_num_derivs(dim, order)
    result = np.zeros((N, num_derivs, num_ngb))

    for i in prange(N):
        index = indices[i]
        dx, weights = kernel_dx_and_weights(index, pos, ngb[i], kernel_radius[i], boxsize, weighted)
        lhs_matrix, rhs_matrix = polyfit_leastsq_matrices(dx, weights, order)
        lhs_matrix_inv = np.linalg.inv(lhs_matrix)
        for k in range(num_derivs):
            for l in range(num_derivs):
                for j in range(num_ngb):
                    result[i, k, j] += lhs_matrix_inv[k, l] * rhs_matrix[l, j]

    return result
