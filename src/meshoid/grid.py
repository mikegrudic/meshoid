"""
Implementation of simple square Grid class, containing the parameters of a Cartesian
grid and methods for converting between grid integer coordinates and global
coordinates and vice versa.
"""

from numba.experimental import jitclass
from numba import float64, int32, int64, boolean
import numpy as np

spec = [
    ("length", float64[:]),  # side length of grid per dimension
    ("dx", float64[:]),  # grid spacing per dimension
    ("res", int64[:]),  # number of grid points per dimension
    ("center", float64[:]),  # location of the grid center in global coordinates
    ("corner", float64[:]),  # location of the grid lower corner in global coordinates
    ("dim", int32),  # dimensionality
    ("boxsize", float64[:]),
    ("is_periodic", boolean),  # whether grid lives in a periodic domain
]


@jitclass(spec)
class Grid:
    def __init__(self, center, length, res, boxsize=None):
        self.center = center
        if boxsize is None:
            self.is_periodic = False
        else:
            self.is_periodic = True
            self.boxsize = boxsize

        self.dim = center.shape[0]
        self.res = res
        self.length = length
        self.dx = self.length / self.res
        self.corner = self.center - 0.5 * self.length

    # # @njit(fastmath=True, error_model="numpy")
    def index_to_global_coord(self, index: np.ndarray):
        """Convert the index of a grid to the *cell-centered* coordinate of that
        grid cell, in a given dimension

        Parameters
        ----------
        index: nd.ndarray
            Shape (dim,) integer array of grid indices

        Returns
        -------
        x: np.ndarray
            Shape (dim,) array of global coordinates
        """

        coord = np.empty(self.dim)
        for d in range(self.dim):
            coord[d] = self.corner[d] + (0.5 + index[d]) * self.dx[d]
            if self.is_periodic:
                coord[d] = coord[d] % self.boxsize[d]
        return coord


@njit(fastmath=True, error_model="numpy")
def grid_index_to_coordinate(
    index: int, grid_length: float, grid_center: float, grid_res: int, box_size=None
):
    """Convert the index of a grid to the *cell-centered* coordinate of that
    grid cell, in a given dimension

    Parameters
    ----------
    index: int
        Grid index, running from 0 to grid_res-1
    grid_length: float
        Side-length of the grid
    grid_center: float
        Coordinate of the grid center
    grid_res: int
        Number of cells per grid dimension
    box_size:
        Size of the periodic domain - if not None, we assume the domain coordinates
        run from [0,box_size)

    Returns
    -------
    x: float
        Coordinate of the center of the grid cell
    """

    x = grid_center - 0.5 * grid_length + grid_length / grid_res * (0.5 + index)
    if box_size is not None:
        return x % box_size
    return x


# candidate for vectorization over dimensions
@njit(fastmath=True, error_model="numpy")
def coordinate_to_grid_index(
    x: float,
    grid_length: float,
    grid_center: float,
    grid_res: int,  # , box_size=None
):
    """Convert coordinate to the integer coordinate on a grid, where 0 is
    corresponds to the position of the first grid cell center and grid_res-1
    corresponds to the last. Assumes grid size is less than box_size (so the
    mapping is bijective with no repeat images.

    Parameters
    ----------
    x: float
        Cartesian coordinate value
    grid_length: float
        Side-length of the grid
    grid_center: float
        Coordinate of the grid center
    grid_res: int
        Number of cells per grid dimension

    Returns
    -------
    grid_coord: float
        Grid coordinate - still float, must be rounded to int as appropriate.
    """
    dx = x - grid_center
    grid_spacing = grid_length / grid_res
    return dx / grid_spacing + 0.5 * (grid_res - 1)


@njit(fastmath=True, error_model="numpy")
def grid_index_bounds(
    x: np.ndarray,
    r: float,
    grid_length: float,
    grid_center: float,
    grid_res: int,
    box_size=None,
):
    """
    Returns the lower-left corner on the grid of the square of grid points that
    a particle will overlap, and the width of the square in grid points

    Parameters
    ----------
    x: np.ndarray
        Shape (N_dimensions,) array containing coordinates
    r: float
        Radius of the particle
    grid_length: float
        Side-length of the grid
    grid_center: float
        Coordinate of the grid center
    grid_res: int
        Number of cells per grid dimension
    box_size: optional
        Size of the periodic domain - if not None, we assume the domain
        coordinates run from [0,box_size)
    """
    N_dim = x.shape[0]
    grid_dx = grid_length / (grid_res - 1)
    width = int(2 * r / grid_dx + 1)
    corner = np.empty(2, dtype=np.int64)
    for dim in range(N_dim):
        corner[dim] = int(
            coordinate_to_grid_index(
                x[dim] - r, grid_length, grid_center[dim], grid_res
            )
            + 1
        )

    return corner, width


@njit
def wrapped_index(gx, size, res, box_size):
    if box_size is None:
        return gx
    return gx % (res * box_size / size)


@njit
def coordinate_lies_on_grid(x, size, center, res):
    if x > center - 0.5 * size:
        if x <= center + 0.5 * size:
            return True
    return False


@njit(fastmath=True, error_model="numpy")
def grid_dx_from_coordinate(
    x: float,
    index: int,
    grid_length: float,
    grid_center: float,
    grid_res: int,
    box_size=None,
):
    """
    Returns the *nearest image* coordinate difference from a given coordinate x
    to the cell-centered grid-point residing at index, ASSUMING grid_length <=
    box_size (no repeat images, so mapping from coordaintes to grid is unique)

    Parameters
    ----------
    x: float
        The original coordinate, from which to compute coordinate difference
    index: int
        Grid index, running from 0 to grid_res-1
    grid_length: float
        Side-length of the grid
    grid_res: int
        Number of cells per grid dimension
    grid_center: float
        Coordinate of the grid center
    box_size:
        Size of the periodic domain - if not None, we assume the domain coordinates
        run from [0,box_size)

    Returns
    -------
    dx: float
        The nearest-image Cartesian coordinate difference from x to grid cell i
    """

    x_grid = grid_index_to_coordinate(
        index, grid_length, grid_center, grid_res, box_size
    )
    if box_size is not None:
        x_grid = x_grid % box_size
    dx = x_grid - x
    if box_size is not None:
        return nearest_image(dx, box_size)
    return dx

    def global_coord_to_index(self, coord):
        """Convert global coordinates to the integer coordinates on the grid,
        where 0 is to the position of the first grid cell center and self.res-1
        corresponds to the last. Assumes grid size is less than box_size, so the
        mapping is bijective with no repeat images.

        Parameters
        ----------
        coord: nd.ndarray
            Shape (dim,) array of global coordinates.

        Returns
        -------
        grid_coord: np.ndarray
            Shape (dim,) float array of grid coordinates
        """
        grid_coord = np.empty(self.dim)
        for d in range(self.dim):
            grid_coord[d] = (coord[d] - self.corner[d]) / self.dx[d] - 0.5
        return grid_coord

        return dx / grid_spacing + 0.5 * (grid_res - 1)

    # #    @njit(fastmath=True, error_model="numpy")
    # def grid_index_bounds(
    #     x: np.ndarray,
    #     r: float,
    #     grid_length: float,
    #     grid_center: float,
    #     grid_res: int,
    #     box_size=None,
    # ):
    #     """
    #     Returns the lower-left corner on the grid of the square of grid points that
    #     a particle will overlap, and the width of the square in grid points

    #     Parameters
    #     ----------
    #     x: np.ndarray
    #         Shape (N_dimensions,) array containing coordinates
    #     r: float
    #         Radius of the particle
    #     grid_length: float
    #         Side-length of the grid
    #     grid_center: float
    #         Coordinate of the grid center
    #     grid_res: int
    #         Number of cells per grid dimension
    #     box_size: optional
    #         Size of the periodic domain - if not None, we assume the domain
    #         coordinates run from [0,box_size)
    #     """
    #     N_dim = x.shape[0]
    #     grid_dx = grid_length / (grid_res - 1)
    #     width = int(2 * r / grid_dx + 1)
    #     corner = np.empty(2, dtype=np.int64)
    #     for dim in range(N_dim):
    #         corner[dim] = int(
    #             coordinate_to_grid_index(
    #                 x[dim] - r, grid_length, grid_center[dim], grid_res, box_size
    #             )
    #             + 1
    #         )
    #     # bounds = np.empty((N_dim, 2), dtype=np.int64)
    #     # for dim in range(N_dim):
    #     #     print(
    #     #         coordinate_to_grid_index(
    #     #             x[dim] - r, grid_length, grid_center[dim], grid_res, box_size
    #     #         )
    #     #     )
    #     #     print(
    #     #         coordinate_to_grid_index(
    #     #             x[dim] + r, grid_length, grid_center[dim], grid_res, box_size
    #     #         )
    #     #     )
    #     #     bounds[dim, 0] = int(
    #     #         coordinate_to_grid_index(
    #     #             x[dim] - r, grid_length, grid_center[dim], grid_res, box_size
    #     #         )
    #     #         + 1
    #     #     )  # round the lower-bound up
    #     #     bounds[dim, 1] = int(
    #     #         coordinate_to_grid_index(
    #     #             x[dim] + r, grid_length, grid_center[dim], grid_res, box_size
    #     #         )
    #     #     )  # round the upper bound down

    #     #        if box_size is not None:

    #     return corner, width
