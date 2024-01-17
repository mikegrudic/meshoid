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

    # dx = self.pos[self.ngb] - self.pos[self.particle_mask][:, None, :]
    #  if order == 1:
    # dx_matrix = np.einsum(
    #     "ij,ijk,ijl->ikl", weights, dx, dx, optimize="optimal"
    # )  # matrix for least-squares fit to a linear function

    # dx_matrix = np.linalg.inv(dx_matrix)  # invert the matrices
    # self.dweights = np.einsum(
    #     "ikl,ijl,ij->ijk", dx_matrix, dx, weights, optimize="optimal"
    # )  # gradient estimator is sum over j of dweight_ij (f_j - f_i)


@njit(parallel=True, fastmath=True)
def compute_dweights(pos, ngb, kernel_radius, boxsize=None):
    N, dim = pos.shape
    num_ngb = ngb.shape[1]
    dx = np.zeros(num_ngb, 3)
    weights = np.zeros(num_ngb)
    for i in prange(N):
        # compute coordinate offsets,
        x = pos[i]
        r = 0
        for j in range(num_ngb):
            for k in range(dim):
                dx[j, k] = pos[ngb[i, j], k] - x[k]
                if boxsize is not None:
                    dx[j, k] = nearest_image(dx[j, k], boxsize)
                r += dx[j, k] * dx[j, k]
            weights[j] = Kernel(r / kernel_radius[i])

        dx_matrix = np.zeros(3, 3)
        for j in range(num_ngb):
            for k in range(dim):
                for l in range(dim):
                    dx_matrix[k, l] += weights[j] * dx[j, k] * dx[j, l]
        #         for j in range(num_ngb):
        #             dx_matrix[k, l] += Kernel()
