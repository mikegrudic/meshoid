from numba import jit, vectorize, float32, float64, cfunc, njit
import numpy as np
from scipy.special import comb

@jit(fastmath=True)
def d2matrix(dx):
    """
    Generates the Vandermonde matrix to solve if you want the weights for the least-squares Jacobian estimator

    Arguments:
    dx - (N, Nngb, dim) array of coordinate differences between particle N and its nearest neighbours

    """
    N, Nngb, dim = dx.shape
    N_derivs = 2*dim + comb(dim, 2, exact=True)
    A = np.empty((N, Nngb, N_derivs), dtype=np.float64)    
    for k in range(N):
        for i in range(Nngb):
            for j in range(N_derivs):
                if j < dim:
                    A[k,i,j] = dx[k,i,j] 
                elif j < 2*dim:
                    A[k,i,j] = dx[k,i,j-dim] * dx[k,i,j-dim] / 2
                else:
                    A[k,i,j] = dx[k,i,(j+1)%dim] * dx[k,i,(j+2)%dim]  # this does the cross-terms, e.g. xy, xz, yz
    return A

@njit
def d2weights(d2_matrix2, d2_matrix, w):
    N, Nngb, Nderiv = d2_matrix.shape
#    print(d2_matrix2.shape, d2_matrix.shape, w.shape)
    result = np.zeros((N,Nngb, Nderiv), dtype=np.float64)
    for i in range(N):
        for j in range(Nngb):
            for k in range(Nderiv):
                for l in range(Nderiv):
                    result[i,j,k] += d2_matrix2[i,k,l] * d2_matrix[i,j,l] * w[i,j]
    return result
    
@jit
def HsmlIter(neighbor_dists,  dim=3, error_norm=1e-6):
    """
    Performs the iteration to get smoothing lengths, according to Eq. 26 in Hopkins 2015 MNRAS 450.

    Arguments:
    neighbor_dists: (N, Nngb) array of distances from particles to their Nngb nearest neighbors

    Keyword arguments:
    dim - Dimensionality (default: 3)
    error_norm - Tolerance in the particle number density to stop iterating at (default: 1e-6)
    """
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
    """
    Un-normalized cubic-spline kernel function

    Arguments:
    q - array containing radii at which to evaluate the kernel, scaled to the kernel support radius (between 0 and 1)
    """
    if q <= 0.5:
        return 1 - 6*q**2 + 6*q**3
    elif q <= 1.0:
        return 2 * (1-q)**3
    else: return 0.0

@njit(fastmath=True)
def GridSurfaceDensity(f, x, h, center, size, res=100, box_size=-1):
    """
    Computes the surface density of conserved quantity f colocated at positions x with smoothing lengths h. E.g. plugging in particle masses would return mass surface density. The result is on a Cartesian grid of SIGHTLINES (not cells), the result being the density of quantity f integrated along those sightlines.

    Arguments:
    f - (N,) array of the conserved quantity that you want the surface density of (e.g. particle masses)
    x - (N,3) array of particle positions
    h - (N,) array of particle smoothing lengths
    center - (2,) array containing the coorindates of the center of the map
    size - side-length of the map
    res - resolution of the grid
    """
    dx = size/(res-1)

    x2d = x[:,:2] - center[:2] + size/2
    
    grid = np.zeros((res,res))
    
    N = len(x)
    for i in range(N):
        xs = x2d[i]
        hs = h[i]
        hinv = 1/hs
        mh2 = f[i]*hinv*hinv

        gxmin = max(int((xs[0] - hs)/dx+1),0)
        gxmax = min(int((xs[0] + hs)/dx),res-1)
        gymin = max(int((xs[1] - hs)/dx+1), 0)
        gymax = min(int((xs[1] + hs)/dx),res-1)

        for gx in range(gxmin, gxmax+1):            
            delta_x_Sqr = xs[0] - gx*dx
            delta_x_Sqr *= delta_x_Sqr
            for gy in range(gymin,gymax+1):
                delta_y_Sqr = xs[1] - gy*dx
                delta_y_Sqr *= delta_y_Sqr
                q = np.sqrt(delta_x_Sqr + delta_y_Sqr) * hinv
                if q <= 0.5:
                    kernel = 1 - 6*q*q + 6*q*q*q
                elif q <= 1.0:
                    kernel = 2 * (1-q)*(1-q)*(1-q)
                else:
                    continue
                grid[gx,gy] += 1.8189136353359467 * kernel * mh2
    return grid

@njit(fastmath=True)
def GridAverage(f, x, h, center, size, res=100, box_size=-1):
    """
    Computes the number density-weighted average of a function f, integrated along sightlines on a Cartesian grid. ie. integral(n f dz)/integral(n dz) where n is the number density and z is the direction of the sightline.

    Arguments:
    f - (N,) array of the conserved quantity that you want the surface density of (e.g. particle masses)
    x - (N,3) array of particle positions
    h - (N,) array of particle smoothing lengths
    center - (2,) array containing the coorindates of the center of the map
    size - side-length of the map
    res - resolution of the grid
    """
    dx = size/(res-1)

    x2d = x[:,:2] - center[:2] + size/2
    
    grid1 = np.zeros((res,res))
    grid2 = np.zeros((res,res))
    N = len(x)
    for i in range(N):
        xs = x2d[i]
        hs = h[i]
        hinv = 1/hs
        h2 = hinv*hinv

        gxmin = max(int((xs[0] - hs)/dx+1),0)
        gxmax = min(int((xs[0] + hs)/dx),res-1)
        gymin = max(int((xs[1] - hs)/dx+1), 0)
        gymax = min(int((xs[1] + hs)/dx),res-1)

        for gx in range(gxmin, gxmax+1):            
            delta_x_Sqr = xs[0] - gx*dx
            delta_x_Sqr *= delta_x_Sqr
            for gy in range(gymin,gymax+1):
                delta_y_Sqr = xs[1] - gy*dx
                delta_y_Sqr *= delta_y_Sqr
                q = np.sqrt(delta_x_Sqr + delta_y_Sqr) * hinv
                if q <= 0.5:
                    kernel = 1 - 6*q*q + 6*q*q*q
                elif q <= 1.0:
                    kernel = 2 * (1-q)*(1-q)*(1-q)
                else:
                    continue
                grid1[gx,gy] += kernel * h2
                grid2[gx,gy] += f[i] * kernel * h2
    return grid2/grid1
   
@jit
def ComputeFaces(ngb, ingb, vol, dweights):
    N, Nngb, dim = dweights.shape
    result = np.zeros_like(dweights)
    for i in range(N):
        for j in range(Nngb):
            result[i,j] += vol[i] * dweights[i,j]
            if ingb[i,j] > -1: result[ngb[i,j],ingb[i,j]] -= vol[i] * dweights[i,j]
    return result
