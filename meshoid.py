#MESHOID: MESHless Operations including Integrals and Derivatives
# "It's not a mesh; it's a meshoid!" - Alfred H. Einstimes

import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import inv
from numba import jit, vectorize, float32, float64

class meshoid(object):
    def __init__(self, x, m=None, h=None, des_ngb=None, boxsize=None, verbose=False):
        if len(x.shape)==1:
            x = x[:,None]

        self.verbose = verbose
        self.N, self.dim = x.shape
        if des_ngb==None:
            des_ngb = {1: 4, 2: 20, 3:32}[self.dim]
                
        self.des_ngb = des_ngb    

        self.volnorm = {1: 2.0, 2: np.pi, 3: 4*np.pi/3}[self.dim]
        self.boxsize = boxsize

        if m==None:
            m = np.repeat(1./len(x),len(x))
        self.m = m
        self.x = x
        self.ngb = None
        self.h = h
        self.weights = None
        self.dweights = None
        self.sliceweights = None
        self.slicegrid = None
        self.vol = None
        self.density = None

        if self.h is None: self.TreeUpdate()


    def ComputeDWeights(self):
        if self.weights is None: self.TreeUpdate()
        
        dx = self.x[self.ngb] - self.x[:,None,:]
        self.dx = dx
        if self.boxsize != None:
            Periodicize(dx.ravel(), self.boxsize)
    
        dx_matrix = np.einsum('ij,ijk,ijl->ikl', self.weights, dx, dx)
        
        dx_matrix = np.linalg.inv(dx_matrix)
        self.dweights = np.einsum('ikl,ijl,ij->ijk',dx_matrix, dx, self.weights)
        #        self.d2weights = d2weights(dx, self.weights)
        
        
    
    def TreeUpdate(self):
        if self.dim == 1:
            sort_order = self.x[:,0].argsort()
            self.ngbdist, self.ngb = NearestNeighbors1D(self.x[:,0][sort_order], self.des_ngb)
            sort_order = invsort(sort_order)
            self.ngbdist, self.ngb = self.ngbdist[sort_order][0], self.ngb[sort_order][0]
        else:
            if self.verbose: print "Finding nearest neighbours..."
                
            self.tree = cKDTree(self.x, boxsize=self.boxsize)
            self.ngbdist, self.ngb = self.tree.query(self.x, self.des_ngb)
                
            if self.verbose: print "Neighbours found!"

        if self.verbose: print "Iterating for smoothing lengths..."
        self.h = HsmlIter(self.ngbdist, error_norm=1e-13,dim=self.dim)
        if self.verbose: print "Smoothing lengths found!"

        q = np.einsum('i,ij->ij', 1/self.h, self.ngbdist)
        K = Kernel(q)
        self.weights = np.einsum('ij,i->ij',K, 1/np.sum(K,axis=1))
        self.density = self.des_ngb * self.m / (self.volnorm * self.h**self.dim)
        self.vol = self.m / self.density

    def D(self, f):
        df = DF(f, self.ngb)
        if self.dweights is None:
            self.ComputeDWeights()
        return np.einsum('ijk,ij->ik',self.dweights,df)

#    def D2(self ,f):
#        df = DF(f, self.ngb)
#        if self.d2weights==None:
#            self.ComputeWeights()
#        return np.einsum('ij,ij->i',self.d2weights[:,:,2],df-np.einsum('ik,ijk->ij',self.D(f),self.dx))
    
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

    def Slice(self, f, size, plane='z', center=np.array([0,0,0]), res=(100,100)):
        if self.tree is None: self.TreeUpdate()
        
        if np.array([size]).size ==1:
            size = (size,size)
        x, y = np.linspace(-size[0]/2,size[0]/2,res[0]), np.linspace(-size[1]/2, size[1]/2,res[1])
        x, y = np.meshgrid(x, y)

        self.slicegrid = np.c_[x.flatten(), y.flatten(), np.zeros(res[0]*res[1])] + center
        if plane=='x':
            self.slicegrid = np.c_[np.zeros(res[0]*res[1]), x.flatten(), y.flatten()] + center
        elif plane=='y':
            self.slicegrid = np.c_[x.flatten(), np.zeros(res[0]*res[1]), y.flatten()] + center
        
        ngbdist, ngb = self.tree.query(self.slicegrid,32)
        hgrid = HsmlIter(ngbdist,dim=3,error_norm=1e-3)
        self.sliceweights = Kernel(np.einsum('ij,i->ij',ngbdist, hgrid**-1))
        self.sliceweights = np.einsum('ij,i->ij', self.sliceweights, 1/np.sum(self.sliceweights,axis=1))
        if len(f.shape)>1:
            return np.einsum('ij,ij...->i...', self.sliceweights, f[ngb]).reshape((res[0],res[1],f.shape[-1]))
        else:
            return np.einsum('ij,ij...->i...', self.sliceweights, f[ngb]).reshape((res[0],res[1]))

    def SurfaceDensity(self, f, size, plane='z', center=np.array([0,0,0]), res=(100,100)):
        if self.h is None: self.TreeUpdate()
        return GridSurfaceDensity(f, self.x-center, np.clip(self.h, size/res[0],1e100), res[0], size)

    def Projection(self, f, size, plane='z', center=np.array([0,0,0]), res=(100,100)):
        if self.h is None: self.TreeUpdate()
        elif self.vol is None: self.vol = self.volnorm * self.h**self.dim
        return SurfaceDensity(f* self.vol, size, plane=plane, center=center,res=res)
        
        
@jit
def d2weights(dx, w):
    N = w.shape[0]
    Nngb = w.shape[1]
    dim = dx.shape[-1]
    
    N2 = dim*(dim+1)/2
    M = np.zeros((N,N2,N2))

    dx2 = np.empty((N, Nngb,N2))
    d2weights = np.zeros((N,Nngb, N2))
    for i in xrange(N):
        for j in xrange(Nngb):
            weight = w[i,j]
            for k in xrange(dim):
                for l in xrange(dim):
                    if l < k: continue
                    n = (dim-1)*k + l
                    #if l==k:
                    dx2[i, j,n] = 0.5*dx[i,j,k]*dx[i,j,l]
                    #else:
                     #   dx2[i, j,n] = dx[i,j,k]*dx[i,j,l]
            
            for p in xrange(N2):
                for q in xrange(N2):
                    M[i,p,q] += weight * dx2[i,j,p] * dx2[i,j,q]
                    
    M = np.linalg.inv(M)
    
    for i in xrange(N):
        for j in xrange(Nngb):
            weight = w[i,j]
            for p in xrange(N2):
                for q in xrange(N2):
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
    for i in xrange(N):
        upper = neighbor_dists[i,des_ngb-1] * bound_coeff
        lower = neighbor_dists[i,1]
        error = 1e100
        count = 0
        while error > error_norm:
            h = (upper + lower)/2
            n_ngb=0.0
            dngb=0.0
            q = 0.0
            for j in xrange(des_ngb):
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
    df = np.empty(ngb.shape)
    for i in xrange(ngb.shape[0]):
        for j in xrange(ngb.shape[1]):
            df[i,j] = f[ngb[i,j]] - f[i]
    return df
    
@jit
def Periodicize(dx, boxsize):
    for i in xrange(dx.size):
        if np.abs(dx[i]) > boxsize/2:
            dx[i] = -np.sign(dx[i])*(boxsize - np.abs(dx[i]))

@jit
def NearestNeighbors1D(x, des_ngb):
    N = len(x)
    neighbor_dists = np.empty((N,des_ngb))
    neighbors = np.empty((N,des_ngb),dtype=np.int64)
    for i in xrange(N):
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
    for i in xrange(len(index)):
        out[index[i]] = i

@jit
def GridSurfaceDensity(mass, x, h, gridres, L):
    count = 0
    grid = np.zeros((gridres,gridres))
    dx = L/(gridres-1)
    N = len(x)
    for i in xrange(N):
        xs = x[i] + L/2
        hs = h[i]
        mh2 = mass[i]/hs**2

        gxmin = max(int((xs[0] - hs)/dx+1),0)
        gxmax = min(int((xs[0] + hs)/dx),gridres-1)
        gymin = max(int((xs[1] - hs)/dx+1), 0)
        gymax = min(int((xs[1] + hs)/dx), gridres-1)
        
        for gx in xrange(gxmin, gxmax+1):
            for gy in xrange(gymin,gymax+1):
                kernel = 1.8189136353359467 * Kernel(((xs[0] - gx*dx)**2 + (xs[1] - gy*dx)**2)**0.5 / hs)
                grid[gx,gy] +=  kernel * mh2
                count +=1
                
    return grid, count
