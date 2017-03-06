#MESHOID: MESHless Operations including Integrals and Derivatives
# "It's not a mesh; it's a meshoid!" - Alfred H. Einstimes

import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import inv
from numba import jit, vectorize, float32, float64
from backend import *

class meshoid(object):
    def __init__(self, x, masses=None, des_ngb=None,boxsize=None, fixed_h = None): 
        if len(x.shape)==1:
            x = x[:,None]

        self.N, self.dim = x.shape
        if des_ngb==None:
            if fixed_h == None: 
                des_ngb = {1: 4, 2: 20, 3:32}[self.dim]
            

        self.fixed_h = fixed_h
        self.des_ngb = des_ngb    
        print self.des_ngb    
        self.volnorm = {1: 2.0, 2: np.pi, 3: 4*np.pi/3}[self.dim]
        self.boxsize = boxsize
        
        if masses==None:
            masses = np.repeat(1./len(x),len(x))
        self.m = masses
        self.x = x
        
        self.TreeUpdate()
        
        self.dweights = None

        self.sliceweights = None
        self.slicegrid = None

    def ComputeWeights(self):
        dx = self.x[self.ngb] - self.x[:,None,:]
        self.dx = dx
        if self.boxsize != None:
            Periodicize(dx.ravel(), self.boxsize)
    
        dx_matrix = np.einsum('ij,ijk,ijl->ikl', self.weights, dx, dx)
        
        dx_matrix = np.linalg.inv(dx_matrix)
        self.dweights = np.einsum('ikl,ijl,ij->ijk',dx_matrix, dx, self.weights)
        
#        self.d2weights = d2weights(dx, self.weights)
        
        
    
    def TreeUpdate(self):
        if self.fixed_h == None:
            if self.dim == 1:
                sort_order = self.x[:,0].argsort()
                self.ngbdist, self.ngb = NearestNeighbors1D(self.x[:,0][sort_order], self.des_ngb)
                sort_order = invsort(sort_order)
                self.ngbdist, self.ngb = self.ngbdist[sort_order][0], self.ngb[sort_order][0]
            else:                
                self.tree = cKDTree(self.x, boxsize=self.boxsize)
                self.ngbdist, self.ngb = self.tree.query(self.x, self.des_ngb)
            self.h = HsmlIter(self.ngbdist, error_norm=1e-13,dim=self.dim)
#        else:
#            self.h = np.repeat(self.fixed_h, len(self.x))
#            self.ngb = 
        q = np.einsum('i,ij->ij', 1/self.h, self.ngbdist)
        K = Kernel(q)
        self.weights = np.einsum('ij,i->ij',K, 1/np.sum(K,axis=1))
        self.density = self.des_ngb * self.m / (self.volnorm * self.h**self.dim)
        self.vol = self.m / self.density

    def D(self, f):
        df = DF(f, self.ngb)
        if self.dweights == None:
            self.ComputeWeights()
        return np.einsum('ijk,ij->ik',self.dweights,df)

#    def D2(self ,f):
#        df = DF(f, self.ngb)
#        if self.d2weights==None:
#            self.ComputeWeights()
#        return np.einsum('ij,ij->i',self.d2weights[:,:,2],df-np.einsum('ik,ijk->ij',self.D(f),self.dx))
    
    def Integrate(self, f):
        return np.einsum('i,i...->...', self.vol,f)

    def KernelVariance(self, f):
        return np.std(DF(f,self.ngb)*self.weights, axis=1)
    
    def KernelAverage(self, f):
        return np.einsum('ij,ij->i',self.weights, f[self.ngb])

    def Slice(self, f, res=(100,100), limits=((0,1),(0,1))):
        x, y = np.linspace(limits[0][0],limits[0][1],res[0]), np.linspace(limits[1][0],limits[1][1],res[1])
        x, y = np.meshgrid(x, y)
        self.slicegrid = np.c_[x.flatten(), y.flatten(), np.zeros(res[0]*res[1])]
        ngbdist, ngb = self.tree.query(self.slicegrid,32)
        hgrid = HsmlIter(ngbdist,dim=3,error_norm=1e-3)
        self.sliceweights = Kernel(np.einsum('ij,i->ij',ngbdist, hgrid**-1))
        self.sliceweights = np.einsum('ij,i->ij', self.sliceweights, 1/np.sum(self.sliceweights,axis=1))
        return np.einsum('ij,ij->i', self.sliceweights, f[ngb]).reshape(res)
