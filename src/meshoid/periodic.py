"""Routines for dealing with periodic boundary conditions"""

from numba import njit, vectorize
import numpy as np


@njit(fastmath=True, cache=True)
def nearest_image(dx_coord, boxsize):
    """Returns separation vector for nearest image, given the coordinate
    difference dx_coord and assuming coordinates run from 0 to boxsize"""
    if np.abs(dx_coord) > 0.5 * boxsize:
        return -np.copysign(boxsize - np.abs(dx_coord), dx_coord)
    return dx_coord


@vectorize(cache=True)
def nearest_image_v(dx_coord, boxsize):
    """Returns separation vector for nearest image, given the coordinate
    difference dx_coord and assuming coordinates run from 0 to boxsize"""
    if np.abs(dx_coord) > 0.5 * boxsize:
        return -np.copysign(boxsize - np.abs(dx_coord), dx_coord)
    return dx_coord
