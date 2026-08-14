# MESHOID: MESHless Operations including Integrals and Derivatives
# "It's not a mesh; it's a meshoid!" - Alfred H. Einstimes

import numpy as np
from scipy.spatial import KDTree
from .grid_deposition import *
from .kernel_density import *
from .derivatives import *
from .periodic import nearest_image_v


class Meshoid:
    """Meshoid object that stores particle data and kernel-weighting quantities, and implements various methods for integral and derivative operations."""

    def __init__(
        self,
        pos: np.ndarray,
        m=None,
        kernel_radius=None,
        des_ngb=None,
        boxsize=None,
        verbose=False,
        particle_mask=None,
        n_jobs=-1,
    ):
        """Construct a meshoid instance.

        Parameters
        ----------
        pos : array_like, shape (n,k)
            The coordinates of the particle data
        m : array_like, shape (n,), optional
            Masses of the particles for computing e.g. mass densities,
        kernel_radius : array_like, shape (n,), optional
            Kernel support radii (e.g. SmoothingLength") of the particles. Can
            be computed adaptively where needed, if not provided.
        des_ngb : positive int, optional
            Number of neighbors to search for kernel density and weighting
            calculations  (defaults to 4/20/32 for 1D/2D/3D)
        boxsize: positive float, optional
            Side-length of box if periodic topology is to be assumed
        verbose: boolean, optinal, default: False
            Whether to print what meshoid is doing to stdout
        particle_mask: array_like, optional
            array-like of integer indices OR boolean mask of the particles you
            want to compute things for
        n_jobs: int, optional
            number of cores to use for parallelization (default: -1, uses all
            cores available)

        Returns
        -------
        Meshoid
            Meshoid instance created from particle data
        """
        self._tree = None
        if len(pos.shape) == 1:
            pos = pos[:, None]

        self.verbose = verbose
        self.N, self.dim = pos.shape
        if particle_mask is None:
            self.particle_mask = np.arange(self.N)
        else:
            if np.array(particle_mask).dtype == np.dtype("bool"):  # boolean mask
                self.particle_mask = np.arange(len(particle_mask))[particle_mask]
            else:
                self.particle_mask = particle_mask
        self.Nmask = len(self.particle_mask)

        if des_ngb is None:
            des_ngb = {1: 4, 2: 20, 3: 32}[self.dim]

        self.des_ngb = des_ngb
        self.n_jobs = n_jobs

        self.volnorm = {1: 2.0, 2: np.pi, 3: 4 * np.pi / 3}[self.dim]
        self.boxsize = boxsize
        self.pos = pos

        if self.boxsize is None:
            # self.boxsize = -1.0
            self.center = np.median(self.pos, axis=0)
            self.L = 2 * np.percentile(np.sum((self.pos - self.center) ** 2, axis=1), 90) ** 0.5
        else:
            self.center = np.ones(3) * self.boxsize / 2
            self.L = self.boxsize

        if m is None:
            # assume unit total mass
            m = np.repeat(1.0 / self.N, self.N)
        # keep masses aligned with the full position array (like self.pos); the
        # particle_mask is applied where masked quantities are computed (see
        # TreeUpdate). Masking here as well would double-index self.m.
        self.m = m

        self._ngb = None
        self._ngbdist = None
        self._kernel_radius = kernel_radius
        self._density = None
        self._vol = None
        self.dweights = None
        self.d2weights = None
        self.dweights_3rdorder = None
        self.sliceweights = None

        if self._kernel_radius is not None:
            self._vol = self.volnorm * self._kernel_radius**self.dim / self.des_ngb
            self._density = self.m / self._vol

    @property
    def _grid_boxsize(self):
        """Returns boxsize as a float for numba grid functions; -1.0 signals no periodic wrapping."""
        return self.boxsize if self.boxsize is not None else -1.0

    @property
    def tree(self):
        if self._tree is None:
            self.BuildTree()
        return self._tree

    @tree.setter
    def tree(self, value):
        self._tree = value

    @property
    def ngb(self):
        if self._ngb is None:
            self.TreeUpdate()
        return self._ngb

    @ngb.setter
    def ngb(self, value):
        self._ngb = value

    @property
    def ngbdist(self):
        if self._ngbdist is None:
            self.TreeUpdate()
        return self._ngbdist

    @ngbdist.setter
    def ngbdist(self, value):
        self._ngbdist = value

    @property
    def kernel_radius(self):
        if self._kernel_radius is None:
            self.TreeUpdate()
        return self._kernel_radius

    @kernel_radius.setter
    def kernel_radius(self, value):
        self._kernel_radius = value

    @property
    def density(self):
        if self._density is None:
            self.TreeUpdate()
        return self._density

    @density.setter
    def density(self, value):
        self._density = value

    @property
    def vol(self):
        if self._vol is None:
            self.TreeUpdate()
        return self._vol

    @vol.setter
    def vol(self, value):
        self._vol = value

    def printv(self, *a, **k):
        if self.verbose:
            print(*a, **k)

    def reset_dweights(self):
        self.dweights = self.d2weights = self.dweights_3rdorder = None

    def ComputeDWeights(self, order=1, weighted=True):
        """
        Computes the weights required to compute least-squares gradient estimators on data colocated on the meshoid

        Parameters
        ----------
        order: int, optional
            1 to compute weights for first derivatives, 2 for the Hessian matrix (default: 1)
        weighted: boolean, optional
            whether to kernel-weight the least-squares gradient solutions (default: True)
        """

        self.printv(f"Computing weights for derivatives of order {order}...")

        weights = gradient_weights(
            self.pos,
            self.ngb,
            self.kernel_radius,
            self.particle_mask,
            boxsize=self.boxsize,
            weighted=weighted,
            order=order,
        )

        if order == 1:
            self.dweights = weights
        elif order == 2:
            self.d2weights, self.dweights_3rdorder = (weights[:, self.dim :, :], weights[:, : self.dim, :])

    def BuildTree(self):
        self.printv("Building tree...")
        self.tree = KDTree(self.pos, boxsize=self.boxsize)

    def TreeUpdate(self):
        """
        Computes or updates the neighbor lists, smoothing lengths, and densities of particles.
        """

        self.printv("Finding neighbors...")
        self.ngbdist, self.ngb = self.tree.query(
            self.pos[self.particle_mask], round(self.des_ngb * 1.5), workers=self.n_jobs
        )

        self.printv("Neighbours found!")

        self.printv("Iterating for smoothing lengths...")

        self.kernel_radius = HsmlIter(self.ngbdist, des_ngb=self.des_ngb, error_norm=1e-13, dim=self.dim)

        self.printv("Smoothing lengths found!")

        self.density = self.des_ngb * self.m[self.particle_mask] / (self.volnorm * self.kernel_radius**self.dim)
        self.vol = self.m[self.particle_mask] / self.density

    def get_kernel_weights(self):
        q = self.ngbdist / self.kernel_radius[:, None]
        K = Kernel_v(q)
        return K / np.sum(K, axis=1)[:, None]

    def Volume(self):
        """
        Returns the effective particle volume = (mass / density)^(1/3)

        Returns
        -------
        self.vol - (N,) array of particle volumes
        """
        return self.vol

    def NearestNeighbors(self):
        """
        Returns the indices of the N_ngb nearest neighbors of each particle in a (N, N_ngb) array

        Returns
        -------
        self.ngb - (N,ngb) array of the indices of the each particle's Nngb nearest neighbors
        """
        return self.ngb

    def NeighborDistance(self):
        """
        Returns the distances of the N_ngb nearest neighbors of each particle in a (N, N_ngb) array

        Returns
        -------
        self.ngbdist - (N,Nngb) array of distances to nearest neighbors of each particle.
        """
        return self.ngbdist

    def SmoothingLength(self):
        """
        Returns the neighbor kernel radii of of each particle in a (N,) array

        Returns
        -------
        (N,) array of particle smoothing lengths
        """
        return self.kernel_radius

    def Density(self):
        """
        Returns the mass density (or number density, if mass not provided) of each particle in a (N,) array

        Returns
        -------
        self.density - (N,) array of particle densities
        """
        return self.density

    def D(self, f, order=1, weighted=True):
        """
        Computes the kernel-weighted least-squares gradient estimator of the function f.

        Parameters
        ----------
        f : array_like
            shape (N,...) array of (possibly vector- or tensor-valued) function
            values (N is the total number of particles)
        order : int, optional
            desired order of precision, either 1 or 2

        Returns
        -------
        (Nmask, ..., dim) array of partial derivatives, evaluated at the
        positions of the particles in the particle mask
        """

        df = np.take(f, self.ngb, axis=0) - f[self.particle_mask, None]

        if order == 1:
            if self.dweights is None:
                self.ComputeDWeights(1, weighted=weighted)
            weights = self.dweights
        else:
            if self.dweights_3rdorder is None:
                self.ComputeDWeights(2, weighted=weighted)
            weights = self.dweights_3rdorder
        return np.einsum("ikj,ij...->i...k", weights, df, optimize="optimal")

    def D2(self, f, weighted=True):
        """
        Computes the kernel-weighted least-squares Hessian estimator of the function f.

        Parameters
        ----------
        f : array_like
          shape (N,...) array of (possibly vector- or tensor-valued) function values (N is the total number of particles)

        Returns
        -------
        (Nmask, ..., N_derivs) array of partial second derivatives, evaluated at the positions of the particles in the particle mask

        Here N_derivs is the number of unique second derivatives in the given number of dimensions: 1 for 1D, 3 for 2D,
        6 for 3D etc.
        For 2D, the order is [xx,yy,xy]
        for 3D, the order is [xx,yy,zz,xy,yz,zx]
        """
        df = np.take(f, self.ngb, axis=0) - f[self.particle_mask, None]

        if self.d2weights is None:
            self.ComputeDWeights(2, weighted=weighted)
        return np.einsum("ikj,ij...->i...k", self.d2weights, df, optimize="optimal")

    def Curl(self, v):
        """
        Computes the curl of a vector field.

        Parameters
        ----------
        v : array_like
          shape (N,3) array containing a vector field colocated on the meshoid

        Returns
        -------
        shape (N,3) array containing the curl of vector field v
        """
        dv = self.D(v[self.particle_mask])
        return np.c_[dv[:, 2, 1] - dv[:, 1, 2], dv[:, 0, 2] - dv[:, 2, 0], dv[:, 1, 0] - dv[:, 0, 1]]

    def Div(self, v):
        """
        Computes the divergence of a vector field.

        Parameters
        ----------
        v : array_like
          shape (N,3) array containing a vector field colocated on the meshoid

        Returns
        -------
        shape (N,) array containing the divergence of v
        """
        dv = self.D(v[self.particle_mask])
        return dv[:, 0, 0] + dv[:, 1, 1] + dv[:, 2, 2]

    def Integrate(self, f):
        """
        Computes the volume integral of a quantity over the volume partition of the domain

        Parameters
        ----------
        f : array_like
          Shape (Nmask, ...) function colocated on the meshoid

        Returns:
        integral of f over the domain
        """
        return np.einsum("i,i...->...", self.vol, f[self.particle_mask], optimize="optimal")

    def KernelVariance(self, f):
        """
        Computes the standard deviation of a quantity over all nearest neighbours

        Parameters
        ----------
        f: array_like, shape (N ,...)
            Shape (N, ...) function colocated on the meshoid

        Returns
        -------
        Shape (Nmask,...) array of standard deviations of f over the kernel
        """
        return np.std(f[self.ngb], axis=1)

    def KernelAverage(self, f):
        """
        Computes the kernel-weighted average of a function

        Parameters
        ----------
        f : array_like, shape (N ,...)
            Shape (N, ...) function colocated on the meshoid

        Returns
        -------
        Shape (N, ...) array of kernel-averaged values of f
        """
        return np.einsum("ij,ij->i", self.get_kernel_weights(), f[self.ngb])

    def Reconstruct(
        self,
        f: np.ndarray,
        target_points: np.ndarray,
        order: int = 1,
        slope_limiter: bool = False,
    ):
        """
        Gives the value of a function f colocated on the meshoid points
        reconstructed at an arbitrary set of points

        Parameters
        ----------
        f : array_like
            The quantity defined on the set of meshoid points that we wish to
            reconstruct at the target points, first dimension should be N_mask
        target_points : array_like
            The shape (N_target,dim) array of points where you would like to
            reconstruct the function
        order: int, optional
            The order of the reconstruction (default 1):
            0 - nearest-neighbor value
            1 - linear reconstruction from the nearest neighbor
            2 - quadratic reconstruction from the nearest neighbor
        slope_limiter: bool, optional
            Whether to apply the Barth-Jespersen slope limiter to the gradient
            (default False). Limits the gradient so that the reconstructed
            value at any neighbor does not exceed the local min/max of f,
            consistent with the default slope limiter in GIZMO.

        Returns
        -------
        f_target : ndarray
            Values of f reconstructed at the target points
        """
        if order < 0 or order > 2:
            raise NotImplementedError("Reconstruction of order > 2 not supported.")

        # get nearest neighbor of each target point
        _, target_neighbors = self.tree.query(target_points, workers=self.n_jobs)

        # get value of f at each nearest neighbor
        f_target = np.take(f, target_neighbors, axis=0)
        if order == 0:  # 0'th order reconstruction
            return f_target

        # 1st-order reconstruction
        dx = target_points - self.pos.take(target_neighbors, axis=0)
        if self.boxsize is not None:
            dx = nearest_image_v(dx, self.boxsize)

        # determine whether we can reuse cached weights or need to compute
        # for only the unique target neighbors
        weights_key = self.dweights if order == 1 else self.dweights_3rdorder
        if weights_key is not None:
            # weights already cached for all particles — use D() directly
            gradf = self.D(f, order=order)
            gradf_neighbors = gradf.take(target_neighbors, axis=0)
        else:
            # compute gradient weights only for unique target neighbors
            unique_nb, inv = np.unique(target_neighbors, return_inverse=True)
            subset_weights = gradient_weights(
                self.pos,
                self.ngb[unique_nb],
                self.kernel_radius[unique_nb],
                unique_nb,
                boxsize=self.boxsize,
                order=order,
            )
            if order == 2:
                subset_d1weights = subset_weights[:, : self.dim, :]
                subset_d2weights = subset_weights[:, self.dim :, :]
            else:
                subset_d1weights = subset_weights
            df = np.take(f, self.ngb[unique_nb], axis=0) - f[unique_nb, None]
            gradf_unique = np.einsum(
                "ikj,ij...->i...k", subset_d1weights, df, optimize="optimal"
            )
            gradf_neighbors = gradf_unique[inv]

        if slope_limiter:
            # in the cached path, unique_nb/inv/gradf_unique aren't set yet
            if weights_key is not None:
                unique_nb, inv = np.unique(target_neighbors, return_inverse=True)
                gradf_unique = gradf[unique_nb]
            f_ngb = np.take(f, self.ngb[unique_nb], axis=0)
            f_i = f[unique_nb]
            f_max = np.maximum(f_i, np.max(f_ngb, axis=1))
            f_min = np.minimum(f_i, np.min(f_ngb, axis=1))
            dx_ngb = self.pos[self.ngb[unique_nb]] - self.pos[unique_nb, None]
            if self.boxsize is not None:
                orig_shape = dx_ngb.shape
                dx_ngb = nearest_image_v(
                    dx_ngb.reshape(-1, self.dim), self.boxsize
                ).reshape(orig_shape)
            delta = np.einsum("ijk,i...k->ij...", dx_ngb, gradf_unique)
            d_plus = np.expand_dims(f_max - f_i, axis=1)
            d_minus = np.expand_dims(f_min - f_i, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                alpha = np.where(
                    delta > 0,
                    np.minimum(1.0, d_plus / delta),
                    np.where(delta < 0, np.minimum(1.0, d_minus / delta), 1.0),
                )
            alpha = np.where(np.isfinite(alpha), alpha, 1.0)
            alpha = np.clip(np.min(alpha, axis=1), 0.0, 1.0)
            gradf_neighbors = (gradf_unique * alpha[..., None])[inv]

        f_target += np.einsum("ij,i...j->i...", dx, gradf_neighbors)
        if order == 1:
            return f_target

        # 2nd order reconstruction
        if self.d2weights is not None:
            d2f = self.D2(f)
            d2f_neighbors = d2f.take(target_neighbors, axis=0)
        else:
            # subset_d2weights and df already computed above for order==2
            d2f_unique = np.einsum(
                "ikj,ij...->i...k", subset_d2weights, df, optimize="optimal"
            )
            d2f_neighbors = d2f_unique[inv]

        f_target += 0.5 * np.einsum("ij,ij...->i...", dx * dx, d2f_neighbors[..., :3])  # pure 2nd derivative terms
        for dim in range(self.dim):
            f_target += np.einsum(
                "i,i...->i...",
                dx[:, dim] * dx[:, (dim + 1) % (self.dim)],
                d2f_neighbors[..., self.dim + dim],
            )  # mixed terms
        return f_target

    def Slice(self, f, size=None, plane="z", center=None, res: int = 128, order: int = 0, return_grid: bool = False, slope_limiter: bool = False):
        """
        Gives the value of a function f deposited on a 2D Cartesian grid slicing
        through the 3D domain.

        Parameters
        ----------
        f : array_like
            Quantity to be reconstructed on the slice grid
        size : optional
            Side length of the grid (default: None, will use the meshoid's predefined side length').
        plane : optional
            The direction of the normal of the slicing plane, one of x, y, or z,
            OR can give an iterable containing 2 orthogonal basis vectors that
            specify the plane (default: 'z')
        center : array_like, optional
            (2,) or (3,) array containing the coordaintes of the center of the grid (default: None, will use the meshoid's predefined center)
        res : int, optional
            the resolution of the grid (default: 128)
        order : int, optional
            Order of the reconstruction on the slice: 0, 1, or 2 (default: 1)
        return_grid : bool, optional
            Also return the grid coordinates corresponding to the slice
        slope_limiter : bool, optional
            Whether to apply the Barth-Jespersen slope limiter (default False)

        Returns
        -------
        slice : array_like
            Shape (res,res,...) grid of values reconstructed on the slice
        """
        if center is None:
            center = self.center
        if size is None:
            size = self.L

        x = np.linspace(-size / 2, size / 2, res + 1)
        x = (x[1:] + x[:-1]) / 2  # cell centers
        y = np.copy(x)
        x, y = np.meshgrid(x, y, indexing="ij")

        slicegrid = np.c_[x.flatten(), y.flatten(), np.zeros(res * res)] + center
        if plane == "x":
            slicegrid = np.c_[np.zeros(res * res), x.flatten(), y.flatten()] + center
        elif plane == "y":
            slicegrid = np.c_[x.flatten(), np.zeros(res * res), y.flatten()] + center

        f_grid = self.Reconstruct(f, slicegrid, order, slope_limiter=slope_limiter)
        shape = res, res
        if len(f.shape) > 1:
            shape += f.shape[1:]
        if return_grid:
            return x, y, f_grid.reshape(shape)
        return f_grid.reshape(shape)

    def InterpToGrid(self, f, size=None, center=None, res: int = 128, order: int = 1, return_grid: bool = False, slope_limiter: bool = False):
        """
        Interpolates the quantity f defined on the meshoid to the cell centers
        of a 3D Cartesian grid

        Parameters
        ----------
        f : array_like
            Shape (N,) function colocated at the particle coordinates
        size : float, optional
            Side length of the grid - defaults to the self.L value: either 2
            times the 90th percentile radius from the center, or the specified
            boxsize
        center: array_like, optional
            Center of the grid - defaults to self.center value, either the box
            center if boxsize given, or average particle position
        res: integer, optional
            Resolution of the grid - default 128
        order : int, optional
            Order of the reconstruction on the slice: 0, 1, or 2 (default: 1)
        slope_limiter : bool, optional
            Whether to apply the Barth-Jespersen slope limiter (default False)

        Returns
        -------
        Shape (res,res,res) grid of cell-centered interpolated values
        """
        if center is None:
            center = self.center
        if size is None:
            size = self.L

        # generate the grid cells' *center points* whoses edges line up with
        # desired domain bounds
        x = np.linspace(-size / 2, size / 2, res + 1)
        x = (x[1:] + x[:-1]) / 2
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        gridcoords = np.c_[X.flatten(), Y.flatten(), Z.flatten()] + center
        f_grid = self.Reconstruct(f, gridcoords, order, slope_limiter=slope_limiter)
        shape = (res, res, res)
        if len(f.shape) > 1:
            shape += f.shape[1:]
        if return_grid:
            return X, Y, Z, f_grid.reshape(shape)
        else:
            return f_grid.reshape(shape)

    def DepositToGrid(self, f, size=None, center=None, res=128):
        """
        Deposits a conserved quantity (e.g. mass, momentum, energy) to a 3D
        grid and returns the density of that quantity on that grid

        Parameters
        ----------
        f : array_like
            Shape (N,) array of conserved quantity values colocated at the particle coordinates
        size : float, optional
            Side length of the grid - defaults to the self.L value: either 2
            times the 90th percentile radius from the center, or the specified
            boxsize
        center: array_like, optional
            Center of the grid - defaults to self.center value, either the box
            center if boxsize given, or average particle position
        res: integer, optional
            Resolution of the grid - default 128

        Returns
        -------
        Shape (res,res,res) grid of the density of the conserved quantity
        """
        if center is None:
            center = self.center
        if size is None:
            size = self.L

        h = np.clip(self.kernel_radius, size / (res - 1), 1e100)

        f_grid = GridDensity(f, self.pos, h, center, size, res=res, box_size=self._grid_boxsize)

        return f_grid

    def SurfaceDensity(
        self, f=None, size=None, center=None, res=128, smooth_fac=1.0, conservative=False, multigrid=False
    ):
        """
        Computes the surface density of a quantity f defined on the meshoid on a grid of sightlines. e.g. if f is the particle masses, you will get mass surface density.

        Parameters
        ----------
        f: array_like, optional
            the quantity you want the surface density of (default: particle mass)
        size: float, optional
            the side length of the window of sightlines (default: None, will use the meshoid's predefined side length')
        center: array_like, optional
            (2,) or (3,) array containing the coordaintes of the center of the grid (default: None, will use the meshoid's predefined center)
        res: int, optional
            the resolution of the grid of sightlines (default: 128)
        smooth_fac: float, optional
            smoothing lengths are increased by this factor (default: 1.)

        Returns
        -------
        (res,res) array containing the column densities integrated along sightlines
        """
        if f is None:
            f = self.m
        if center is None:
            center = self.center
        if size is None:
            size = self.L
        if multigrid:
            renderer = GridSurfaceDensityMultigrid
        else:
            renderer = GridSurfaceDensity
        return renderer(
            f,
            self.pos,
            np.clip(smooth_fac * self.kernel_radius, 2 * size / res, 1e100),
            center,
            size,
            res,
            self._grid_boxsize,
            parallel=(False if self.n_jobs == 1 else True),
            conservative=conservative,
        )

    def MassWeightedAverage(self, f, masses=None, size=None, center=None, res=128, smooth_fac=1.0, conservative=False):
        """
        Computes the surface density of a quantity f defined on the meshoid on a grid of sightlines. e.g. if f is the particle masses, you will get mass surface density.

        Parameters
        ----------
        f: array_like
            The quantity you want the mass-weighted average of
        masses: array_like, optional
            Shape (N,) array of masses, or in general arbitrary weights
        size: float, optional
            the side length of the window of sightlines (default: None, will use the meshoid's predefined side length')
        center: array_like, optional
            (2,) or (3,) array containing the coordaintes of the center of the grid (default: None, will use the meshoid's predefined center)
        res: int, optional
            the resolution of the grid of sightlines (default: 128)
        smooth_fac: float, optional
            smoothing lengths are increased by this factor (default: 1.)

        Returns
        -------
        (res,res) array containing the column densities integrated along sightlines
        """
        if f is None:
            f = self.m
        if masses is None:
            masses = self.m
        if center is None:
            center = self.center
        if size is None:
            size = self.L

        args = (
            self.pos,
            np.clip(smooth_fac * self.kernel_radius, 2 * size / res, 1e100),
            center,
            size,
            res,
            self._grid_boxsize,
            (False if self.n_jobs == 1 else True),
            conservative,
        )

        return GridSurfaceDensity(f * masses, *args) / GridSurfaceDensity(masses, *args)

    def ProjectedAverage(self, f, size=None, center=None, res=128, smooth_fac=1.0):  # plane="z",
        """
        Computes the average value of a quantity f along a Cartesian grid of sightlines from +/- infinity.

        Parameters
        ----------
        f: array_like, optional
            (N,) array containing the quantity you want the average of
        size: float, optional
            the side length of the window of sightlines (default: None, will use the meshoid's predefined side length')
        center: array_like, optional
            (2,) or (3,) array containing the coordaintes of the center of the grid (default: None, will use the meshoid's predefined center)
        res: int, optional
            the resolution of the grid of sightlines (default: 128)
        smooth_fac: float, optional
            smoothing lengths are increased by this factor (default: 1.)

        Returns
        -------
        (res,res) array containing the averages along sightlines
        """
        if center is None:
            center = self.center
        if size is None:
            size = self.L
        return GridAverage(
            f,
            self.pos,
            np.clip(smooth_fac * self.kernel_radius, 2 * size / res, 1e100),
            center,
            size,
            res,
            self._grid_boxsize,
        )

    def Projection(self, f, size=None, center=None, res=128, smooth_fac=1.0):
        """
        Computes the integral of quantity f along a Cartesian grid of sightlines
        from +/- infinity. e.g. plugging in 3D density for f will return
        surface density.

        Parameters
        ----------
        f: array_like, optional
            (N,) array containing the quantity you want the average of
        size: float, optional
            the side length of the window of sightlines (default: None, will use the meshoid's predefined side length')
        center: array_like, optional
            (2,) or (3,) array containing the coordaintes of the center of the grid (default: None, will use the meshoid's predefined center)
        res: int, optional
            the resolution of the grid of sightlines (default: 128)
        smooth_fac: float, optional
            smoothing lengths are increased by this factor (default: 1.)

        Returns
        -------
        (res, res) array containing the projected values
        """
        if center is None:
            center = self.center
        if size is None:
            size = self.L
        return GridSurfaceDensity(
            f * self.vol,
            self.pos,
            np.clip(smooth_fac * self.kernel_radius, 2 * size / res, 1e100),
            center,
            size,
            res,
            self._grid_boxsize,
        )

    def KDE(self, grid, bandwidth=None):
        """
        Computes the kernel density estimate of the meshoid points on a 1D grid

        Parameters
        ----------
        grid : array_like
          1D array of coordintes upon which to compute the KDE
        bandwidth : float, optional
          constant bandwidth of the kernel, defined as the radius of support of the cubic spline kernel.

        Returns
        -------
        array containing the density of particles defined at the grid points
        """

        if self.dim != 1:
            raise Exception("KDE only implemented in 1D")
        if bandwidth is None:
            bandwidth = self.SmoothingLength()

        f = np.zeros_like(grid)
        gtree = KDTree(np.c_[grid,])
        for d, bw in zip(self.pos, bandwidth):
            ngb = gtree.query_ball_point(d, bw)
            ngbdist = np.abs(grid[ngb] - d)
            f[ngb] += Kernel_v(ngbdist / bw) / bw * 4.0 / 3

        return f

    def SPH_Density_Gradient(self):
        return dkernel_ngb_sum(self.pos, self.kernel_radius, self.ngb, self.boxsize) * kernel_norm[self.dim]

    def SPH_gradient(self, f):
        return kernel_gradient(f, self.pos, self.kernel_radius, self.ngb, self.boxsize)
