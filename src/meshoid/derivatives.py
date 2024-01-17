"""Functions for computing the matrices weights for least-squares derivative operators"""
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
from .kernel_density import Kernel


@njit(fastmath=True)
def nearest_image(dx_coord, boxsize):
    """Returns separation vector for nearest image, given the coordinate
    difference dx_coord and assuming coordinates run from 0 to boxsize"""
    if np.abs(dx_coord) > boxsize / 2:
        return -np.copysign(boxsize - np.abs(dx_coord), dx_coord)
    return dx_coord


@njit(parallel=True, fastmath=True, error_model="numpy")
def derivative_weights1(pos, ngb, kernel_radius, boxsize=None, weighted=True):
    """Computes the N x N_ngb x dim matrix that encodes the 1st derivative
    operator, accurate to 2nd order
    """
    N, dim = pos.shape
    num_ngb = ngb.shape[1]
    result = np.zeros((N, num_ngb, 3))
    for i in prange(N):
        # get coordinate differences
        dx = np.zeros((num_ngb, 3))
        weights = np.ones(num_ngb)
        for j in range(num_ngb):
            r = 0
            for k in range(dim):
                dx[j, k] = pos[ngb[i, j], k] - pos[i, k]
                if boxsize is not None:
                    dx[j, k] = nearest_image(dx[j, k], boxsize)
                r += dx[j, k] * dx[j, k]
            if weighted:
                weights[j] = Kernel(np.sqrt(r) / kernel_radius[i])

        # compute the matrix of weights * outer product of dx * dx
        dx2_matrix = np.zeros((3, 3))
        for j in range(num_ngb):
            for k in range(dim):
                for l in range(dim):
                    dx2_matrix[k, l] += weights[j] * dx[j, k] * dx[j, l]
        # invert outer product matrix
        dx2_matrix_inv = np.linalg.inv(dx2_matrix)

        # multiply matrix by dx to get derivative operator
        for j in range(num_ngb):
            for k in range(dim):
                for l in range(dim):
                    result[i, j, k] += weights[j] * dx2_matrix_inv[k, l] * dx[j, l]

    return result


@njit(fastmath=True, parallel=True)
def d2matrix(dx):
    """
    Generates the Vandermonde matrix to solve if you want the weights for the
    least-squares Jacobian estimator

    Arguments:
        dx - (N, Nngb, dim) array of coordinate differences between particle N
        and its nearest neighbours

    """
    N, Nngb, dim = dx.shape
    N_derivs = {1: 2, 2: 5, 3: 9}[
        dim
    ]  # in 3D: 3 first derivatives + 6 unique second derivatives
    A = np.empty((N, Nngb, N_derivs), dtype=np.float64)
    for k in prange(N):
        for i in range(Nngb):
            for j in range(N_derivs):
                if j < dim:
                    A[k, i, j] = dx[k, i, j]
                elif j < 2 * dim:
                    A[k, i, j] = dx[k, i, j - dim] * dx[k, i, j - dim] / 2
                else:
                    A[k, i, j] = (
                        dx[k, i, (j + 1) % dim] * dx[k, i, (j + 2) % dim]
                    )  # this does the cross-terms, e.g. xy, xz, yz
    return A


@njit(fastmath=True, parallel=True)
def d2weights(d2_matrix2, d2_matrix, w):
    N, Nngb, Nderiv = d2_matrix.shape
    result = np.zeros((N, Nngb, Nderiv), dtype=np.float64)
    for i in prange(N):
        for j in range(Nngb):
            for k in range(Nderiv):
                for l in range(Nderiv):
                    result[i, j, k] += (
                        d2_matrix2[i, k, l] * d2_matrix[i, j, l] * w[i, j]
                    )
    return result


@njit(parallel=True, fastmath=True, error_model="numpy")
def derivative_weights2(pos, ngb, kernel_radius, boxsize=None, weighted=True):
    """Computes the N x N_ngb x dim matrix that encodes the matrix operators for
    2nd derivatives AND 1st derivatives, accurate to 3rd order
    """
    N, dim = pos.shape
    num_ngb = ngb.shape[1]
    N_derivs = {1: 2, 2: 5, 3: 9}[
        dim
    ]  # in 3D: 3 first derivatives + 6 unique second derivatives
    result = np.zeros((N, num_ngb, N_derivs))
    for i in prange(N):
        # get coordinate differences
        dx = np.zeros((num_ngb, 3))
        weights = np.ones(num_ngb)
        for j in range(num_ngb):
            r = 0
            for k in range(dim):
                dx[j, k] = pos[ngb[i, j], k] - pos[i, k]
                if boxsize is not None:
                    dx[j, k] = nearest_image(dx[j, k], boxsize)
                r += dx[j, k] * dx[j, k]
            if weighted:
                weights[j] = Kernel(np.sqrt(r) / kernel_radius[i])

        # vandermonde matrix for fitting to N-dimensional quadratic
        A = np.empty((num_ngb, N_derivs))
        for j in range(num_ngb):
            for k in range(N_derivs):
                if k < dim:
                    A[j, k] = dx[j, k]
                elif k < 2 * dim:
                    A[j, k] = 0.5 * dx[j, k - dim] * dx[j, k - dim]
                else:
                    A[j, k] = (
                        dx[j, k % dim] * dx[j, (k + 1) % dim]
                    )  # this does the cross-terms, e.g. xy, xz, yz

        A2 = np.zeros((N_derivs, N_derivs))
        for j in range(num_ngb):
            for k in range(N_derivs):
                for l in range(N_derivs):
                    A2[k, l] += weights[j] * A[j, k] * A[j, l]
        A2_inv = np.linalg.inv(A2)

        for j in range(num_ngb):
            for k in range(N_derivs):
                for l in range(N_derivs):
                    result[i, j, k] += weights[j] * A2_inv[k, l] * A[j, l]

    return result
