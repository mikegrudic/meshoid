"""
Implementation of simple 1D grid class, containing the parameters of an evenly-
spaced grid methods for converting between grid integer coordinates and global
coordinates and vice versa.
"""

from numba.experimental import jitclass
from numba import float64, int64, boolean
import numpy as np

spec = [
    ("length", float64),  # side length of grid per dimension
    ("length_centers", float64),  # distance between centers of left and right cells
    ("dx", float64),  # grid spacing per dimension
    ("res", int64),  # number of grid points per dimension
    ("center", float64),  # location of the grid center in global coordinates
    ("left_edge", float64),  # left edge in global coordinates
    ("left_cell_center", float64),  # center of left-most cell
    ("boxsize", float64),  # size of periodic box, assumes coordinates run [0,boxsize]
    ("periodic", boolean),  # whether grid lives in a periodic domain
]


@jitclass(spec)
class Grid1D:
    """Stores attributes of a 1D grid, and implements methods needed for conversion
    between grid indices and global coordinates.
    """

    def __init__(self, center, length, res, boxsize=None):
        self.center = center
        if boxsize is None:
            self.periodic = False
        else:
            self.periodic = True
            self.boxsize = boxsize
        self.res = res
        self.length = length
        self.length_centers = length * (res - 1) / res
        self.dx = self.length / self.res
        self.left_edge = self.center - 0.5 * self.length
        self.left_cell_center = self.center - (0.5 * (self.length - self.dx))

    def overlap_indices_and_coordinates(self, x, radius):
        """
        Given a finite interval specified by a center and a max distance, returns
        the list of indices of grid centers overlapped by that interval
        """

        width_on_grid = 2 * radius / self.dx
        indices = np.empty(width_on_grid, dtype=np.int32)
        coordinates = np.empty(width_on_grid, dtype=np.float64)

        left_edge = x - radius

        left_edge_index = self.global_to_grid(left_edge)

        if self.periodic:
            left_edge %= self.boxsize
            # to be continued...

    def global_to_grid(self, coordinate: float):
        """Converts a global coordinate to grid coordinate:
        grid = (global - left cell center) / (grid length - grid spacing)

        so that 0 corresponds to the position of the leftmost cell center,
        and res-1 corresponds to position of the rightmost cell center.

        Parameters
        ----------
        coordinate : float
            The global coordinate to be converted to grid coordinates
        """
        return (coordinate - self.left_cell_center) / self.length_centers
