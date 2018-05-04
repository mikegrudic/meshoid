#MESHOID: MESHless Operations including Integrals and Derivatives
# "It's not a mesh; it's a meshoid!" - Alfred H. Einstimes

import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import inv
from numba import jit, vectorize, float32, float64, njit
import h5py

class meshoid(object):
    def __init__(self, x, m=None, h=None, des_ngb=None, boxsize=None, verbose=False):
        self.tree=None
        if len(x.shape)==1:
            x = x[:,None]

        self.verbose = verbose
        self.N, self.dim = x.shape
        if des_ngb==None:
            des_ngb = {1: 4, 2: 20, 3:32}[self.dim]
                
        self.des_ngb = des_ngb    

        self.volnorm = {1: 2.0, 2: np.pi, 3: 4*np.pi/3}[self.dim]
        self.boxsize = boxsize
        self.x = x
        if self.boxsize is None:
            self.center = np.average(self.x, axis=0)
            self.L = 2*np.percentile(np.sum((x-self.center)**2,axis=1),90)**0.5
        else:
            self.center = np.ones(3) * self.boxsize / 2
            self.L = self.boxsize
            
        if m is None:
            m = np.repeat(1./len(x),len(x))
        self.m = m

        self.ngb = None
        self.h = h
        self.weights = None
        self.dweights = None
        self.sliceweights = None
        self.slicegrid = None

        if self.h is None:
            self.TreeUpdate()
        else:
            self.vol = self.volnorm * self.h**self.dim / self.des_ngb
            self.density = self.m / self.vol


    def ComputeDWeights(self):
        if self.weights is None: self.TreeUpdate()
        
        dx = self.x[self.ngb] - self.x[:,None,:]
        self.dx = dx
        if self.boxsize != None:
            PeriodicizeDX(dx.ravel(), self.boxsize)
    
        dx_matrix = np.einsum('ij,ijk,ijl->ikl', self.weights, dx, dx)
        
        dx_matrix = np.linalg.inv(dx_matrix)
        self.dweights = np.einsum('ikl,ijl,ij->ijk',dx_matrix, dx, self.weights)
        #        self.d2weights = d2weights(dx, self.weights)        
        self.A = ComputeFaces(self.ngb,self.invngb, self.vol, self.dweights)

    def TreeUpdate(self):
        if self.verbose: print("Finding nearest neighbours...")
                
        self.tree = cKDTree(self.x, boxsize=self.boxsize)
        self.ngbdist, self.ngb = self.tree.query(self.x, self.des_ngb)
                
        if self.verbose: print("Neighbours found!")

        self.invngb = invngb(self.ngb)

        if self.verbose: print("Iterating for smoothing lengths...")
        self.h = HsmlIter(self.ngbdist, error_norm=1e-13,dim=self.dim)
        if self.verbose: print("Smoothing lengths found!")

        q = np.einsum('i,ij->ij', 1/self.h, self.ngbdist)
        K = Kernel(q)
        self.weights = np.einsum('ij,i->ij',K, 1/np.sum(K,axis=1))
        self.density = self.des_ngb * self.m / (self.volnorm * self.h**self.dim)
        self.vol = self.m / self.density
        
#        self.A = ComputeFaces(self.ngb, self.vol, self.dweights)

    def Volume(self):
        return self.vol

    def NearestNeighbors(self):
        if self.ngb is None: self.TreeUpdate()
        return self.ngb

    def NeighborDistance(self):
        if self.ngbdist is None: self.TreeUpdate()
        return self.ngbdist

    def SmoothingLength(self):
        return self.h

    def Density(self):
        return self.density

    def D(self, f):
        if self.ngb is None: self.TreeUpdate()
        df = DF(f, self.ngb)
        if self.dweights is None:
            self.ComputeDWeights()
        return np.einsum('ijk,ij...->i...k',self.dweights,df)

    def Curl(self, v):
        dv = self.D(v)
        return np.c_[dv[:,1,2]-dv[:,2,1], dv[:,0,2]-dv[:,2,0], dv[:,0,1] - dv[:,1,0]]
    def Div(self, v):
        dv = self.D(v)
        return dv[:,0,0]+dv[:,1,1] + dv[:,2,2]
    
    def Integrate(self, f):
        if self.h is None: self.TreeUpdate()
        elif self.vol is None: self.vol = self.volnorm * self.h**self.dim
        return np.einsum('i,i...->...', self.vol,f)

    def KernelVariance(self, f):
        if self.ngb is None: self.TreeUpdate()
#        return np.einsum('ij,ij->i', (f[self.ngb] - self.KernelAverage(f)[:,np.newaxis])**2, self.weights)
        return np.std(f[self.ngb], axis=1)
    
    def KernelAverage(self, f):
        if self.weights is None: self.TreeUpdate()        
        return np.einsum('ij,ij->i',self.weights, f[self.ngb])

    def Slice(self, f, size=None, plane='z', center=None, res=100, gridngb=32):
        
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
            self.sliceweights = Kernel(np.einsum('ij,i->ij',ngbdist, hgrid**-1))
            self.sliceweights = np.einsum('ij,i->ij', self.sliceweights, 1/np.sum(self.sliceweights,axis=1))
        else:
            self.sliceweights = np.ones(ngbdist.shape)

        #return f[ngb].reshape(res,res)
            
#        hgrid = HsmlIter(ngbdist,dim=3,error_norm=1e-3)
#        self.sliceweights = Kernel(np.einsum('ij,i->ij',ngbdist, hgrid**-1))
#        self.sliceweights = np.einsum('ij,i->ij', self.sliceweights, 1/np.sum(self.sliceweights,axis=1))

        if len(f.shape)>1:
            return np.einsum('ij,ij...->i...', self.sliceweights, f[ngb]).reshape((res,res,f.shape[-1]))
        else:
            return np.einsum('ij,ij...->i...', self.sliceweights, f[ngb]).reshape((res,res))

    def SurfaceDensity(self, f=None, size=None, plane='z', center=None, res=128, smooth_fac=1.):
        if f is None: f = self.m
        if center is None: center = self.center
        if size is None: size = self.L
#        if self.boxsize is None:
        return GridSurfaceDensity(f, self.x-center, np.clip(smooth_fac*self.h, 2*size/res,1e100), res, size)
        #else:
#            return GridSurfaceDensityPeriodic(f, (self.x-center) % self.boxsize, np.clip(self.h, size/res,1e100), res, size, self.boxsize)

    def ProjectedAverage(self, f, size=None, plane='z', center=None, res=128):
        if size is None: size = self.L
        if center is None: center = self.center
        return GridAverage(f, self.x-center, np.clip(self.h, size/res,1e100), res, size)

    def Projection(self, f, size=None, plane='z', center=None, res=128, smooth_fac=1.):
        if size is None: size = self.L
        if center is None: center = self.center
        return GridSurfaceDensity(f * self.vol, self.x-center, np.clip(smooth_fac*self.h, 2*size/res,1e100), res, size) #GridSurfaceDensity(f * self.vol, size, plane=plane, center=center,res=res)

    def KDE(self, grid, bandwidth=None):
        if bandwidth is None:
            bandwidth = self.SmoothingLength()

        f = np.zeros_like(grid)
        gtree = cKDTree(np.c_[grid,])
        for d, bw in zip(self.x, bandwidth):
            ngb = gtree.query_ball_point(d, bw)
            ngbdist = np.abs(grid[ngb] - d)
            f[ngb] += Kernel(ngbdist/bw) / bw * 4./3
            
        return f

def FromSnapshot(filename, ptype=None):
    F = h5py.File(filename)
    meshoids = {}
    if ptype is None: types = list(F.keys())[1:]
    else: types = ["PartType%d"%ptype,]
    for k in types:
        x = np.array(F[k]["Coordinates"])
        m = np.array(F[k]["Masses"])
        if "SmoothingLength" in list(F[k].keys()):
            h = np.array(F[k]["SmoothingLength"])
        elif "AGS-Softening" in list(F[k].keys()):
            h = np.array(F[k]["AGS-Softening"])
        else:
            h = None
        boxsize = F["Header"].attrs["BoxSize"]
        if np.any(x<0): boxsize=None    
        if ptype is None:
            meshoids[k] = meshoid(x, m, h,boxsize=boxsize)
        else: return meshoid(x,m,h,boxsize=boxsize)
    F.close()
    return meshoids
        
@jit
def d2weights(dx, w):
    N = w.shape[0]
    Nngb = w.shape[1]
    dim = dx.shape[-1]
    
    N2 = dim*(dim+1)/2
    M = np.zeros((N,N2,N2))

    dx2 = np.empty((N, Nngb,N2))
    d2weights = np.zeros((N,Nngb, N2))
    for i in range(N):
        for j in range(Nngb):
            weight = w[i,j]
            for k in range(dim):
                for l in range(dim):
                    if l < k: continue
                    n = (dim-1)*k + l
                    #if l==k:
                    dx2[i, j,n] = 0.5*dx[i,j,k]*dx[i,j,l]
                    #else:
                     #   dx2[i, j,n] = dx[i,j,k]*dx[i,j,l]
            
            for p in range(N2):
                for q in range(N2):
                    M[i,p,q] += weight * dx2[i,j,p] * dx2[i,j,q]
                    
    M = np.linalg.inv(M)
    
    for i in range(N):
        for j in range(Nngb):
            weight = w[i,j]
            for p in range(N2):
                for q in range(N2):
                    d2weights[i,j,p] += M[i,p,q] * dx2[i,j,q] * weight

    return d2weights
                    


@jit
def HsmlIter(neighbor_dists,  dim=3, error_norm=1e-6):
    if dim==3:
        norm = 32./3
    elif dim==2:
        norm = 40./7
    else:
        norm = 8./3
    N, des_ngb = neighbor_dists.shape
    hsml = np.zeros(N)
    n_ngb = 0.0
    bound_coeff = (1./(1-(2*norm)**(-1./3)))
    for i in range(N):
        upper = neighbor_dists[i,des_ngb-1] * bound_coeff
        lower = neighbor_dists[i,1]
        error = 1e100
        count = 0
        while error > error_norm:
            h = (upper + lower)/2
            n_ngb=0.0
            dngb=0.0
            q = 0.0
            for j in range(des_ngb):
                q = neighbor_dists[i, j]/h
                if q <= 0.5:
                    n_ngb += (1 - 6*q**2 + 6*q**3)
                elif q <= 1.0:
                    n_ngb += 2*(1-q)**3
            n_ngb *= norm
            if n_ngb > des_ngb:
                upper = h
            else:
                lower = h
            error = np.fabs(n_ngb-des_ngb)
        hsml[i] = h
    return hsml
        
@vectorize([float32(float32), float64(float64)])
def Kernel(q):
    if q <= 0.5:
        return 1 - 6*q**2 + 6*q**3
    elif q <= 1.0:
        return 2 * (1-q)**3
    else: return 0.0
        
@jit
def DF(f, ngb):
    if len(f.shape) > 1:
        df = np.empty((ngb.shape[0],ngb.shape[1], f.shape[1]))
    else:
        df = np.empty(ngb.shape)
    for i in range(ngb.shape[0]):
        for j in range(ngb.shape[1]):
            df[i,j] = f[ngb[i,j]] - f[i]
    return df
    
@jit
def PeriodicizeDX(dx, boxsize):
    for i in range(dx.size):
        if np.abs(dx[i]) > boxsize/2:
            dx[i] = -np.sign(dx[i])*(boxsize - np.abs(dx[i]))

@jit
def invngb(ngb):
    result = np.empty_like(ngb)
    for i in range(len(ngb)):
        ngbi = ngb[i]
        for j in range(ngb.shape[1]):
            for k in range(ngb.shape[1]):
                if ngb[ngbi[j],k]==i:
                    result[i,j]=k
                    break
                if k==ngb.shape[1]-1: result[i,j]=-1
    return result


@jit
def NearestNeighbors1D(x, des_ngb):
    N = len(x)
    neighbor_dists = np.empty((N,des_ngb))
    neighbors = np.empty((N,des_ngb),dtype=np.int64)
    for i in range(N):
        x0 = x[i]
        left = 0
#        if i == N-1:
#            right = 0
#        else:
#            right = 1
        right = 1
        total_ngb = 0
        while total_ngb < des_ngb:
            lpos = i - left
            rpos = i + right
            if lpos < 0:
                dl = 1e100
            else:
                dl = np.fabs(x0 - x[lpos])
            if rpos > N-1:
                dr = 1e100
            else:
                dr = np.fabs(x0 - x[rpos])

            if dl < dr:
                neighbors[i,total_ngb] = lpos
                neighbor_dists[i, total_ngb] = dl
                left += 1
            else:
                neighbors[i,total_ngb] = rpos
                neighbor_dists[i, total_ngb] = dr
                right += 1
            total_ngb += 1
    return neighbor_dists, neighbors

@jit
def invsort(index):
    out = np.empty_like(index)
    for i in range(len(index)):
        out[index[i]] = i

@jit
def GridSurfaceDensity(mass, x, h, gridres, L):
#    count = 0
    grid = np.zeros((gridres,gridres))
    dx = L/(gridres-1)
    N = len(x)
    for i in range(N):
        xs = x[i] + L/2
        hs = h[i]
        mh2 = mass[i]/hs**2

        gxmin = max(int((xs[0] - hs)/dx+1),0)
        gxmax = min(int((xs[0] + hs)/dx),gridres-1)
        gymin = max(int((xs[1] - hs)/dx+1), 0)
        gymax = min(int((xs[1] + hs)/dx), gridres-1)
        
        for gx in range(gxmin, gxmax+1):
            for gy in range(gymin,gymax+1):
                kernel = 1.8189136353359467 * Kernel(((xs[0] - gx*dx)**2 + (xs[1] - gy*dx)**2)**0.5 / hs)
                grid[gx,gy] +=  kernel * mh2
#                count += 1
                
    return grid

@jit
def GridAverage(f, x, h, gridres, L):
#    count = 0
    grid1 = np.zeros((gridres,gridres))
    grid2 = np.zeros((gridres,gridres))
    dx = L/(gridres-1)
    N = len(x)
    for i in range(N):
        xs = x[i] + L/2
        hs = h[i]
        mh2 = hs**-2
        fi = f[i]

        gxmin = max(int((xs[0] - hs)/dx+1),0)
        gxmax = min(int((xs[0] + hs)/dx),gridres-1)
        gymin = max(int((xs[1] - hs)/dx+1), 0)
        gymax = min(int((xs[1] + hs)/dx), gridres-1)
        
        for gx in range(gxmin, gxmax+1):
            for gy in range(gymin,gymax+1):
                kernel = 1.8189136353359467 * Kernel(np.sqrt((xs[0] - gx*dx)**2 + (xs[1] - gy*dx)**2) / hs)
                grid1[gx,gy] +=  kernel * mh2
                grid2[gx,gy] +=  fi * kernel * mh2
#                count += 1

    return grid2/grid1
   
   
@jit
def GridSurfaceDensityPeriodic(mass, x, h, gridres, L, boxsize): # need to fix this
    x = (x+L/2)%boxsize
    grid = np.zeros((gridres,gridres))
    dx = L/(gridres-1)
    N = len(x)

    b2 = boxsize/2
    for i in range(N):
        xs = x[i]
        hs = h[i]
        mh2 = mass[i]/hs**2

        gxmin = int((xs[0] - hs)/dx + 1)
        gxmax = int((xs[0] + hs)/dx)
        gymin = int((xs[1] - hs)/dx + 1)
        gymax = int((xs[1] + hs)/dx)
        
        for gx in range(gxmin, gxmax+1):
            ix = gx%gridres
            for gy in range(gymin,gymax+1):
                iy = gy%gridres
                delta_x = np.abs(xs[0] - ix*dx)
                if b2 < delta_x: delta_x -= boxsize
                delta_y = np.abs(xs[1] - iy*dx)
                if b2 < delta_y: delta_y -= boxsize
                kernel = 1.8189136353359467 * Kernel((delta_x**2 + delta_y**2)**0.5 / hs)
                grid[ix,iy] +=  kernel * mh2
                 
    return grid

@jit
def ComputeFaces(ngb, ingb, vol, dweights):
    N, Nngb, dim = dweights.shape
    result = np.zeros_like(dweights)
    for i in range(N):
        for j in range(Nngb):
            result[i,j] += vol[i] * dweights[i,j]
            if ingb[i,j] > -1: result[ngb[i,j],ingb[i,j]] -= vol[i] * dweights[i,j]
    return result
