#MESHOID: MESHless Operations including Integrals and Derivatives
# "It's not a mesh; it's a meshoid!" - Alfred H. Einstimes

import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import inv
from scipy.special import comb
from numba import jit, vectorize, float32, float64, njit, guvectorize
from .backend import *
import h5py

class Meshoid(object):
    def __init__(self, pos, m=None, hsml=None, des_ngb=None, boxsize=-1, verbose=False, particle_mask=None, n_jobs=1):
        """
        Creates a Meshoid object instance given at least the positions of the particles.
        
        Arguments:
        x -- shape (N,3) array of particle positions

        Keyword arguments:
        m -- shape (N,) array of particle masses - defaults to 1    
        h -- shape (N,) array of particle kernel lengths
        des_ngb -- integer number of nearest neighbors (defaults to 4/20/32 for 1D/2D/3D)
        boxsize -- size of box if periodic boundary conditions
        verbose -- bool whether to print everything the code is doing to stdout
        particle_mask -- array-like of indices of the particles you want to compute things for (defaults to all)
        n_jobs -- number of logical cores available (default 1)
        """
    
        self.tree=None
        if len(pos.shape)==1:
            pos = pos[:,None]

        self.verbose = verbose
        self.N, self.dim = pos.shape
        if particle_mask is None:
            self.particle_mask = np.arange(self.N)
        else:
            self.particle_mask = particle_mask
        self.Nmask = len(self.particle_mask)
        
        if des_ngb==None:
            des_ngb = {1: 4, 2: 20, 3:32}[self.dim]
                
        self.des_ngb = des_ngb
        self.n_jobs = n_jobs    

        self.volnorm = {1: 2.0, 2: np.pi, 3: 4*np.pi/3}[self.dim]
        self.boxsize = boxsize
        self.pos = pos
        
        if self.boxsize is None:
            self.center = np.average(self.pos, axis=0)
            self.L = 2*np.percentile(np.sum((self.pos-self.center)**2,axis=1),90)**0.5
        else:
            self.center = np.ones(3) * self.boxsize / 2
            self.L = self.boxsize
            
        if m is None:
            m = np.repeat(1.,self.N) # assume unit masses so that density corresponds to number-density
        self.m = m[self.particle_mask]

        self.ngb = None
        self.hsml = hsml
        self.weights = None
        self.dweights = None
        self.d2weights = None
        self.sliceweights = None
        self.slicegrid = None

        if self.hsml is None:
#            self.hsml = -np.ones(self.N)
            self.TreeUpdate()
        else:
            self.vol = self.volnorm * self.hsml**self.dim / self.des_ngb
            self.density = self.m / self.vol


    def ComputeDWeights(self, order=1, weighted=True):
        """
        Computes the weights required to compute least-squares gradient estimators on data colocated on the meshoid

        Keyword arguments:
        order - whether to compute weights for first derivatives or for the Jacobian (default: 1)
        weighted - whether to kernel-weight the least-squares gradient solutions (default: True)
        """
        if self.weights is None: self.TreeUpdate()
        
        dx = self.pos[self.ngb] - self.pos[self.particle_mask][:,None,:]
        self.dx = dx

        if order == 1:
            dx_matrix = np.einsum('ij,ijk,ijl->ikl', self.weights, dx, dx, optimize='optimal') # matrix for least-squares fit to a linear function
        
            dx_matrix = np.linalg.inv(dx_matrix) # invert the matrices 
            self.dweights = np.einsum('ikl,ijl,ij->ijk',dx_matrix, dx, self.weights, optimize='optimal') # gradient estimator is sum over j of dweight_ij (f_j - f_i)
        elif order == 2:
            if weighted: w = self.weights
            else: w = np.ones_like(self.weights)
            Nngb = self.des_ngb
            N_derivs = 2*self.dim + comb(self.dim, 2, exact=True)            
            dx_matrix = d2matrix(dx)
            dx_matrix2 = np.einsum('ij,ijk,ijl->ikl', w, dx_matrix, dx_matrix, optimize='optimal')
            dx_matrix2 = np.linalg.inv(dx_matrix2)
            self.d2_condition_number = np.linalg.cond(dx_matrix2)

            self.d2weights = d2weights(dx_matrix2, dx_matrix, w)

            self.d2weights, self.dweights_3rdorder = self.d2weights[:,:,self.dim:], self.d2weights[:,:,:self.dim]
            # gradient estimator is sum over j of dweight_ij (f_j - f_i)

    def TreeUpdate(self):
        """
        Computes or updates the neighbor lists, smoothing lengths, and densities of particles.        
        """
        if self.verbose: print("Finding nearest neighbours...")
                
        self.tree = cKDTree(self.pos, boxsize=self.boxsize)
        self.ngbdist, self.ngb = self.tree.query(self.pos[self.particle_mask], self.des_ngb, workers=self.n_jobs)
                
        if self.verbose: print("Neighbours found!")

        if self.verbose: print("Iterating for smoothing lengths...")

        self.hsml = HsmlIter(self.ngbdist, error_norm=1e-13,dim=self.dim)
        if self.verbose: print("Smoothing lengths found!")

        q = self.ngbdist / self.hsml[:,None]
        K = Kernel(q)
        self.weights = K / np.sum(K, axis=1)[:,None]
        self.density = self.des_ngb * self.m / (self.volnorm * self.hsml**self.dim)
        self.vol = self.m / self.density

    def Volume(self):
        """
        Returns the effective particle volume

        Returns:
        self.vol - (N,) array of particle volumes
        """
        return self.vol

    def NearestNeighbors(self):
        """
        Returns the indices of the N_ngb nearest neighbors of each particle in a (N, N_ngb) array

        Returns:
        self.ngb - (N,ngb) array of the indices of the each particle's Nngb nearest neighbors
        """
        if self.ngb is None: self.TreeUpdate()
        return self.ngb

    def NeighborDistance(self):
        """
        Returns the distances of the N_ngb nearest neighbors of each particle in a (N, N_ngb) array

        Returns:
        self.ngbdist - (N,Nngb) array of distances to nearest neighbors of each particle.
        """
        if self.ngbdist is None: self.TreeUpdate()
        return self.ngbdist

    def SmoothingLength(self):
        """
        Returns the neighbor kernel radii of of each particle in a (N,) array

        Returns:
        self.hsml - (N,) array of particle smoothing lengths
        """
        return self.hsml

    def Density(self):
        """
        Returns the mass density (or number density, if mass not provided) of each particle in a (N,) array

        Returns:
        self.density - (N,) array of particle densities
        """
        return self.density

    def D(self, f):
        """
        Computes the kernel-weighted least-squares gradient estimator of the function f.

        Arguments:
        f -- shape (N,...) array of (possibly vector- or tensor-valued) function values (N is the total number of particles)

        Returns:
        gradf - (Nmask, ..., dim) array of partial derivatives, evaluated at the positions of the particles in the particle mask
        """
            
        if self.ngb is None: self.TreeUpdate()
            
        df = f[self.ngb] - f[self.particle_mask,None]
        
        if self.dweights is None:
            self.ComputeDWeights()
        return np.einsum('ijk,ij...->i...k',self.dweights,df, optimize='optimal')

    def D2(self, f, weighted=True):
        """
        Computes the kernel-weighted least-squares Jacobian estimator of the function f.

        Arguments:
        f -- shape (N,...) array of (possibly vector- or tensor-valued) function values (N is the total number of particles)

        Returns:
        jacf - (Nmask, ..., dim,dim) array of partial derivatives, evaluated at the positions of the particles in the particle mask
        """
        if self.ngb is None: self.TreeUpdate()
            
        df = f[self.ngb] - f[self.particle_mask,None]

        if self.d2weights is None:
            self.ComputeDWeights(2, weighted=weighted)
        return np.einsum('ijk,ij...->i...k',self.d2weights,df, optimize='optimal')
        

    def Curl(self, v):
        """
        Computes the curl of a vector field.

        Arguments:
        v -- shape (N,3) array containing a vector field colocated on the meshoid

        Returns:
        curlv - shape (N,3) array containing the curl of vector field v
        """
        dv = self.D(v[self.particle_mask])
        return np.c_[dv[:,1,2]-dv[:,2,1], dv[:,0,2]-dv[:,2,0], dv[:,0,1] - dv[:,1,0]]
        
    def Div(self, v):
        """
        Computes the divergence of a vector field.

        Arguments:
        v -- shape (N,3) array containing a vector field colocated on the meshoid

        Returns:
        divv -- shape (N,) array containing the divergence of v
        """        
        dv = self.D(v[self.particle_mask])
        return dv[:,0,0]+ dv[:,1,1] + dv[:,2,2]
    
    def Integrate(self, f):
        """
        Computes the volume integral of a quantity over the volume partition of the meshoid

        Arguments:
        f -- Shape (N, ...) function colocated on the meshoid

        Returns:
        integral of f over the meshoid
        """
        if self.hsml is None: self.TreeUpdate()
        elif self.vol is None: self.vol = self.volnorm * self.hsml**self.dim
        return np.einsum('i,i...->...', self.vol,f[self.particle_mask], optimize='optimal')

    def KernelVariance(self, f):
        """
        Computes the standard deviation of a quantity over all nearest neighbours

        Arguments:
        f -- Shape (N, ...) function colocated on the meshoid

        Returns:
        sigma_f -- shape (N,...) array of standard deviations of f over the kernel
        """
        if self.ngb is None: self.TreeUpdate()
#        return np.einsum('ij,ij->i', (f[self.ngb] - self.KernelAverage(f)[:,np.newaxis])**2, self.weights)
        return np.std(f[self.ngb], axis=1)
    
    def KernelAverage(self, f):
        """
        Computes the kernel-weighted average of a function

        Arguments:
        f -- Shape (N, ...) function colocated on the meshoid

        returns:
        Shape (N, ...) array of kernel-averaged values of f
        """            
        if self.weights is None: self.TreeUpdate()        
        return np.einsum('ij,ij->i',self.weights, f[self.ngb])

    def Slice(self, f, size=None, plane='z', center=None, res=100, gridngb=32):
        """
        Gives the kernel-weighted value of a function f deposited on a Cartesion grid slicing through the meshoid.

        Keyword arguments:
        f - the quantity you want the surface density of (default: particle mass)
        size - the side length of the window of sightlines (default: None, will use the meshoid's predefined side length')
        plane - the direction of the normal of the slicing plane, one of x, y, or z (default: 'z'')
        center - (2,) or (3,) array containing the coordaintes of the center of the grid (default: None, will use the meshoid's predefined center)
        res - the resolution of the grid of sightlines (default: 128)
        gridngb - how many nearest neighbors the gridpoints should search for to construct their neighbor kernel(default: 32)        
        """
        if center is None: center = self.center
        if size is None: size = self.L
        if self.tree is None: self.TreeUpdate()
        
        x, y = np.linspace(-size/2,size/2,res), np.linspace(-size/2, size/2,res)
        x, y = np.meshgrid(x, y)

        self.slicegrid = np.c_[x.flatten(), y.flatten(), np.zeros(res*res)] + center
        if plane=='x':
            self.slicegrid = np.c_[np.zeros(res*res), x.flatten(), y.flatten()] + center
        elif plane=='y':
            self.slicegrid = np.c_[x.flatten(), np.zeros(res*res), y.flatten()] + center
        
        ngbdist, ngb = self.tree.query(self.slicegrid,gridngb)

        if gridngb > 1:
            hgrid = HsmlIter(ngbdist,dim=3,error_norm=1e-3)
            self.sliceweights = Kernel(ngbdist/hgrid[:,None])
            self.sliceweights /= np.sum(self.sliceweights,axis=1)[:,None]
        else:
            self.sliceweights = np.ones(ngbdist.shape)

        if len(f.shape)>1:
            return np.einsum('ij,ij...->i...', self.sliceweights, f[ngb]).reshape((res,res,f.shape[-1]))
        else:
            return np.einsum('ij,ij...->i...', self.sliceweights, f[ngb]).reshape((res,res))

    def InterpToGrid(self, f, weights=None, size=None, center=None, res=128, method='kernel'):
        """
        Interpolates the quantity f defined on the meshoid to a Cartesian grid

        """
        if center is None: center = self.center
        if size is None: size = self.L
        if weights is None: weights = self.m

        x = np.linspace(-size/2,size/2,res+1)
        x = (x[1:] + x[:-1])/2
        X, Y, Z = np.meshgrid(x,x,x,indexing='ij')
        gridcoords = np.c_[X.flatten(),Y.flatten(),Z.flatten()] + center
        if method=='nearest':
            ngbdist, ngb = self.tree.query(gridcoords, workers=self.n_jobs)
            f_interp = f[ngb].reshape((res,res,res))
        elif method=='kernel':
            h = np.clip(self.hsml, 2 * size/(res-1), 1e100)
            f_interp = WeightedGridInterp3D(f, weights, self.pos, h, center, size, res=res,box_size=self.boxsize)
                   
        return f_interp

    def DepositToGrid(self, f, weights=None, size=None, center=None, res=128):
        """
        Deposits a conserved quantity (e.g. mass, momentum, energy) to a 3D grid and returns the density of that quantity that grid
        """
        if center is None: center = self.center
        if size is None: size = self.L
        if weights is None: weights = self.m

        h = np.clip(self.hsml, 2 * size/(res-1), 1e100)
        f_grid = GridDensity(f, self.pos, h, center, size, res=res,box_size=self.boxsize)
                   
        return f_grid

        
        

    def SurfaceDensity(self, f=None, size=None, plane='z', center=None, res=128, smooth_fac=1.):
        """
        Computes the surface density of a quantity f defined on the meshoid on a grid of sightlines. e.g. if f is the particle masses, you will get mass surface density.

        Keyword arguments:
        f - the quantity you want the surface density of (default: particle mass)
        size - the side length of the window of sightlines (default: None, will use the meshoid's predefined side length')
        plane - the direction you want to project along, of x, y, or z (default: 'z')
        center - (2,) or (3,) array containing the coordaintes of the center of the grid (default: None, will use the meshoid's predefined center)
        res - the resolution of the grid of sightlines (default: 128)
        smooth_fac - smoothing lengths are increased by this factor (default: 1.)

        Returns:
        (res,res) array containing the surface densities along sightlines
        """
        if f is None: f = self.m
        if center is None: center = self.center
        if size is None: size = self.L
        return GridSurfaceDensity(f, self.pos, np.clip(smooth_fac*self.hsml, 2*size/res,1e100), center, size, res, self.boxsize, parallel=(False if self.n_jobs == 1 else True))

    def ProjectedAverage(self, f, size=None, plane='z', center=None, res=128, smooth_fac=1.):
        """
        Computes the average value of a quantity f along a Cartesian grid of sightlines from +/- infinity.

        Arguments:
        f - (N,) array containing the quantity you want the average of
        
        Keyword arguments:
        size - the side length of the window of sightlines (default: None, will use the meshoid's predefined side length')
        plane - the direction you want to project along, of x, y, or z (default: 'z')
        center - (2,) or (3,) array containing the coordaintes of the center of the grid (default: None, will use the meshoid's predefined center)
        res - the resolution of the grid of sightlines (default: 128)
        smooth_fac - smoothing lengths are increased by this factor (default: 1.)

        Returns:
        (res,res) array containing the averages along sightlines        
        """
        if center is None: center = self.center
        if size is None: size = self.L
        return GridAverage(f, self.pos, np.clip(smooth_fac*self.hsml, 2*size/res,1e100), center, size, res, self.boxsize)

    def Projection(self, f, size=None, plane='z', center=None, res=128, smooth_fac=1.):
        """
        Computes the integral of quantity f along a Cartesian grid of sightlines from +/- infinity. e.g. plugging in 3D density for f will return surface density.

        Arguments:
        f - (N,) array containing the quantity you want the projected integral of
        
        Keyword arguments:
        size - the side length of the window of sightlines (default: None, will use the meshoid's predefined side length')
        plane - the direction you want to project along, of x, y, or z (default: 'z')
        center - (2,) or (3,) array containing the coordaintes of the center of the grid (default: None, will use the meshoid's predefined center)
        res - the resolution of the grid of sightlines (default: 128)
        smooth_fac - smoothing lengths are increased by this factor (default: 1.)
        
        Returns:
        (res, res) array containing the projected values
        """
        if center is None: center = self.center
        if size is None: size = self.L
        return GridAverage(f * self.vol, self.pos, np.clip(smooth_fac*self.hsml, 2*size/res,1e100), center, size, res, self.boxsize)        

    def KDE(self, grid, bandwidth=None):
        """
        Computes the kernel density estimate of the meshoid points on a 1D grid

        Arguments:
        grid - 1D array of coordintes upon which to compute the KDE

        Keyword arguments:
        bandwidth - constant bandwidth of the kernel, defined as the radius of support of the cubic spline kernel.

        Returns:
        array containing the density of particles defined at the grid points
        """

        if self.dim != 1: raise Exception("KDE only implemented in 1D")
        if bandwidth is None:
            bandwidth = self.SmoothingLength()

        f = np.zeros_like(grid)
        gtree = cKDTree(np.c_[grid,])
        for d, bw in zip(self.pos, bandwidth):
            ngb = gtree.query_ball_point(d, bw)
            ngbdist = np.abs(grid[ngb] - d)
            f[ngb] += Kernel(ngbdist/bw) / bw * 4./3
            
        return f


