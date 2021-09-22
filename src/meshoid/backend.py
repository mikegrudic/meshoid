from numba import jit, vectorize, float32, float64, cfunc, njit, prange, get_num_threads
import numpy as np
from scipy.special import comb
from scipy.interpolate import interp2d, RectBivariateSpline

@njit(fastmath=True)
def NearestImage(x,boxsize):
    if abs(x) > boxsize/2: return -copysign(boxsize-abs(x),x)
    else: return x

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
    
def GridSurfaceDensityMultigrid(f, x, h, center, size, res=128, box_size=-1,N_grid_kernel=8,parallel=False):
    if not ((res != 0) and (res & (res-1) == 0)): raise("Multigrid resolution must be a power of 2")
    res_bins = size / 2**np.arange(0,round(np.log2(res)+1))
    res_bins[0] = np.inf
    res_bins[-1] = 0
    
    grid = np.zeros((4,4))
    for i in range(2,len(res_bins)-1):
#        print("Upsampling...")
        grid = UpsampleGrid(grid)
#        print("Done!")        
        Ni = grid.shape[0]
        # bin particles by smoothing length to decide which resolution level they get deposited at                   
        idx = (h/N_grid_kernel < res_bins[i]) & (h/N_grid_kernel >= res_bins[i+1])
        if np.any(idx): 
            grid += GridSurfaceDensity(f[idx], x[idx], h[idx], center, size, res=Ni, box_size=box_size,parallel=parallel)
    return grid



#@njit(fastmath=True)
def UpsampleGrid(grid):
    N = grid.shape[0]
    x1 = np.linspace(0.5/N, 1-0.5/N, N) # original coords
    x2 = np.linspace(0.25/N, 1-0.25/N, 2*N) # new coords
    return RectBivariateSpline(x1,x1,grid)(x2,x2) #RectBivariateSpline(x1,
#    newgrid = np.empty((grid.shape[0]*2, grid.shape[1]*2))
#    for i in range(grid.shape[0]):
#        for j in range(grid.shape[1]):
#            newgrid[2*i,2*j] = grid[i,j]
#            newgrid[2*i+1,2*j] = grid[i,j]
#            newgrid[2*i,2*j+1] = grid[i,j]
#            newgrid[2*i+1,2*j+1] = grid[i,j]
#    return newgrid

@njit(parallel=True,fastmath=True)
def GridSurfaceDensity(f, x, h, center, size, res=100, box_size=-1,parallel=False):
    """
    Computes the surface density of conserved quantity f colocated at positions x with smoothing lengths h. E.g. plugging in particle masses would return mass surface density. The result is on a Cartesian grid of sightlines, the result being the density of quantity f integrated along those sightlines.

    Arguments:
    f - (N,) array of the conserved quantity that you want the surface density of (e.g. particle masses)
    x - (N,3) array of particle positions
    h - (N,) array of particle smoothing lengths
    center - (2,) array containing the coorindates of the center of the map
    size - side-length of the map
    res - resolution of the grid
    """
    
    if parallel:
        Nthreads = get_num_threads()
        # chunk the particles among the threads
        chunksize = max(len(f) // Nthreads, 1)
        sigmas = np.empty((Nthreads,res,res)) # will store separate grids and sum them at the end
    
        for i in prange(Nthreads):        
            sigmas[i] = GridSurfaceDensity_core(f[i*chunksize:(i+1)*chunksize], x[i*chunksize:(i+1)*chunksize], h[i*chunksize:(i+1)*chunksize], center, size, res, box_size)
        return sigmas.sum(0)
    else:
        return GridSurfaceDensity_core(f, x, h, center, size, res=res, box_size=box_size)

@njit(fastmath=True)
def GridSurfaceDensity_core(f, x, h, center, size, res=100, box_size=-1):
    """
    Computes the surface density of conserved quantity f colocated at positions x with smoothing lengths h. E.g. plugging in particle masses would return mass surface density. The result is on a Cartesian grid of sightlines, the result being the density of quantity f integrated along those sightlines.

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
                r = delta_x_Sqr + delta_y_Sqr
                if r > hs: continue
                q = np.sqrt(r) * hinv                
                if q <= 0.5:
                    kernel = 1 - 6*q*q * (1-q)
                elif q <= 1.0:
                    a = 1-q
                    kernel = 2 * a * a * a
                else:
                    continue
                grid[gx,gy] += 1.8189136353359467 * kernel * mh2
    return grid

#@njit(fastmath=True, parallel=True)
#def GridSurfaceDensity_parallel(f, x, h, center, size, res=100, box_size=-1):

#    return sigmas.sum(0)
    
    
    
@njit(fastmath=True)
def UpsampleGrid_PPV(grid):
    newgrid = np.empty((grid.shape[0]*2, grid.shape[1]*2, grid.shape[2]))
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            newgrid[2*i,2*j,:] = grid[i,j,:]
            newgrid[2*i+1,2*j,:] = grid[i,j,:]
            newgrid[2*i,2*j+1,:] = grid[i,j,:]
            newgrid[2*i+1,2*j+1,:] = grid[i,j,:]
    return newgrid


def Grid_PPZ_DataCube_Multigrid(f, x, h, center, size, z, h_z, res, box_size=-1,N_grid_kernel=8):
    """ Faster, multigrid version of Grid_PPZ_DataCube. Since the third dimension is separate from the spatial ones, we only do the multigrid approach on the spatial grid. See Grid_PPZ_DataCube for desription of inputs """
    if not ((res[0] != 0) and (res[0] & (res[0]-1) == 0)): raise("Multigrid resolution must be a power of 2")
    res_bins = size[0] / 2**np.arange(0,round(np.log2(res[0])+1))
    res_bins[0] = np.inf
    res_bins[-1] = 0
    grid = np.zeros((1,1,res[1]))
    for i in range(len(res_bins)-1):
        grid = UpsampleGrid_PPV(grid)
        Ni = grid.shape[0]
        # bin particles by smoothing length to decide which resolution level they get deposited at                                             
        idx = (h/N_grid_kernel < res_bins[i]) & (h/N_grid_kernel >= res_bins[i+1])
        print(Ni, np.sum(idx))
        if np.any(idx): 
            grid += Grid_PPZ_DataCube(f[idx], x[idx], h[idx], center, size, z[idx], h_z[idx], np.array([Ni,res[1]],dtype=np.int32), box_size=box_size)
    return grid

@njit(fastmath=True)
def Grid_PPZ_DataCube(f, x, h, center, size, z, h_z, res, box_size=-1):
    """
    A modified version of the GridSurfaceDensity script, it computes the PPZ datacube of conserved quantity f, where Z is an arbitrary data dimension (e.g. using line-of-sight velocity as Z gives the usual astro PPV datacubes, using position gives a PPP cube, which is just the density).

    Arguments:
    f - (N,) array of the conserved quantity that you want the surface density of (e.g. particle masses)
    x - (N,3) array of particle positions
    h - (N,) array of particle smoothing lengths
    z - (N,) array of particle positions in the Z dimension (e.g. line of sight velocity values)
    h_z - (N,) array of uncertainties ("smoothing lengths") in the Z dimension (e.g. thermal velocity)
    center - (3,) array containing the coordinates of the center of the PPZ map
    size - (2) side-length of the map, first value for the PP map, second value is for Z (e.g. max velocity - min velocity)
    res - (2) resolution of the PPX grid, first value is for the PP map, second is for Z (e.g. [128, 16] means a 128x128x16 PPX cube)
    """
    dx = size[0]/(res[0]-1)
    dz = size[1]/(res[1]-1)

    x2d = x[:,:2] - center[:2] + size[0]/2
    z1d = z - center[2] + size[1]/2
    
    grid = np.zeros((res[0],res[0],res[1]))
    
    N = len(x)
    for i in range(N):
        xs = x2d[i]
        zs = z1d[i]
        hs = h[i]
        h_z_s = h_z[i]
        hinvsq = 1/(hs*hs);
        h_z_invsq = 1/(h_z_s*h_z_s)
        f_density = f[i]/(hs*hs*h_z_s)

        gxmin = max(int((xs[0] - hs)/dx+1),0)
        gxmax = min(int((xs[0] + hs)/dx),res[0]-1)
        gymin = max(int((xs[1] - hs)/dx+1), 0)
        gymax = min(int((xs[1] + hs)/dx),res[0]-1)
        gzmin = max(int((zs - h_z_s)/dz+1), 0)
        gzmax = min(int((zs + h_z_s)/dz),res[1]-1)

        for gx in range(gxmin, gxmax+1):            
            delta_x_Sqr = xs[0] - gx*dx
            delta_x_Sqr *= delta_x_Sqr
            for gy in range(gymin,gymax+1):
                delta_y_Sqr = xs[1] - gy*dx
                delta_y_Sqr *= delta_y_Sqr
                q2dsq = (delta_x_Sqr + delta_y_Sqr)*hinvsq
                for gz in range(gzmin,gzmax+1):
                    delta_z_Sqr = zs - gz*dz
                    delta_z_Sqr *= delta_z_Sqr
                    q = np.sqrt( q2dsq + delta_z_Sqr*h_z_invsq ) 
                    if q <= 0.5:
                        kernel = 1 - 6*q*q + 6*q*q*q
                    elif q <= 1.0:
                        kernel = 2 * (1-q)*(1-q)*(1-q)
                    else:
                        continue
                    grid[gx,gy, gz] += 2.546479089470325 * kernel * f_density #Using 3D normalization
    return grid

@njit(fastmath=True)
def GridAverage(f, x, h, center, size, res=100, box_size=-1):
    """
    Computes the number density-weighted average of a function f, integrated along sightlines on a Cartesian grid. ie. integral(n f dz)/integral(n dz) where n is the number density and z is the direction of the sightline.

    Arguments:
    f - (N,) array of the conserved quantity that you want the surface density of (e.g. particle masses)
    x - (N,3) array of particle positions
    h - (N,) array of particle smoothing lengths
    center - (2,) array containing the coordinates of the center of the map
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


@njit(fastmath=True)
def WeightedGridInterp3D(f, wt, x, h, center, size, res=100, box_size=-1):
    """
    Peforms a weighted grid interpolation of quantity f onto a 3D grid
    
    Arguments:
    f - (N,) array of the function defined on the point set that you want to interpolate to the grid
    wt - (N,) array of weights
    x - (N,3) array of particle positions
    h - (N,) array of particle smoothing lengths
    center - (2,) array containing the coorindates of the center of the map
    size - side-length of the map
    res - resolution of the grid
    """
    dx = size/(res-1)

    x3d = x - center + size/2 - dx/2 # + dx/2 # coordinates in the grid frame, such that the origin is at the corner of the grid
    
    grid = np.zeros((res,res,res))
    gridwt = np.zeros_like(grid)
    
    N = len(x)
    for i in range(N):
        xs = x3d[i]        
        hs = h[i]        
        hinv = 1/hs
#        if True:
        if box_size < 0:
            gxmin = max(int((xs[0] - hs)/dx+1),0)
            gxmax = min(int((xs[0] + hs)/dx),res-1)
            gymin = max(int((xs[1] - hs)/dx+1), 0)
            gymax = min(int((xs[1] + hs)/dx),res-1)
            gzmin = max(int((xs[2] - hs)/dx+1), 0)
            gzmax = min(int((xs[2] + hs)/dx),res-1)
        else:
            gxmin = int((xs[0] - hs)/dx+1)
            gxmax = int((xs[0] + hs)/dx)
            gymin = int((xs[1] - hs)/dx+1)
            gymax = int((xs[1] + hs)/dx)
            gzmin = int((xs[2] - hs)/dx+1)
            gzmax = int((xs[2] + hs)/dx)

        # first have to do a prepass to get the weight
        kval = np.empty(int(2*hs/dx+1)**3) # save kernel values so don't have to recompute
        total_wt = 0
        j = 0
        for gx in range(gxmin, gxmax+1):            
            delta_x_Sqr = xs[0] - gx*dx
            delta_x_Sqr *= delta_x_Sqr
            for gy in range(gymin,gymax+1):
                delta_y_Sqr = xs[1] - gy*dx
                delta_y_Sqr *= delta_y_Sqr
                for gz in range(gzmin,gzmax+1):
                    delta_z_Sqr = xs[2] - gz*dx
                    delta_z_Sqr *= delta_z_Sqr
                    q = np.sqrt(delta_x_Sqr + delta_y_Sqr + delta_z_Sqr) * hinv
                    if q > 1:
                        kernel = 0
                    elif q <= 0.5:
                        kernel = 1 - 6*q*q + 6*q*q*q
                    else:
                        kernel = 2 * (1-q)*(1-q)*(1-q)
                    total_wt += kernel # to normalize out the kernel weights
                    kval[j] = kernel; j += 1

        # OK now do the actual deposition
        j = 0
        for gx in range(gxmin, gxmax+1):            
#            delta_x_Sqr = xs[0] - gx*dx
#            delta_x_Sqr *= delta_x_Sqr
            for gy in range(gymin,gymax+1):
#                delta_y_Sqr = xs[1] - gy*dx
#                delta_y_Sqr *= delta_y_Sqr
                for gz in range(gzmin,gzmax+1):
#                    delta_z_Sqr = xs[2] - gz*dx
#                    delta_z_Sqr *= delta_z_Sqr
#                    q = np.sqrt(delta_x_Sqr + delta_y_Sqr + delta_z_Sqr) * hinv
#                    if q > 1:
#                        kernel = 0
#                    elif q <= 0.5:
#                        kernel = 1 - 6*q*q + 6*q*q*q
#                    else:
#                        kernel = 2 * (1-q)*(1-q)*(1-q)
                    kernel = kval[j]; j+=1
                    total_wt += kernel
                    grid[gx%res,gy%res,gz%res] += f[i] * kernel * wt[i] / total_wt
                    gridwt[gx%res,gy%res,gz%res] += kernel*wt[i]/total_wt
    return grid/gridwt

@njit(fastmath=True)
def GridDensity(f, x, h, center, size, res=100, box_size=-1):
    """
    Estimates the density of the conserved quantity f possessed by each particle (e.g. mass, momentum, energy) on a 3D grid
    
    Arguments:
    f - (N,) array of the conserved quantity that you want the surface density of (e.g. particle masses)
    x - (N,3) array of particle positions
    h - (N,) array of particle smoothing lengths
    center - (2,) array containing the coorindates of the center of the map
    size - side-length of the map
    res - resolution of the grid
    """
    dx = size/(res-1)

    x3d = x - center + size/2 - dx/2 # + dx/2 # coordinates in the grid frame, such that the origin is at the corner of the grid
    
    grid = np.zeros((res,res,res))
    
    N = len(x)
    for i in range(N):
        xs = x3d[i]        
        hs = h[i]        
        hinv = 1/hs
#        if True:
        if box_size < 0:
            gxmin = max(int((xs[0] - hs)/dx+1),0)
            gxmax = min(int((xs[0] + hs)/dx),res-1)
            gymin = max(int((xs[1] - hs)/dx+1), 0)
            gymax = min(int((xs[1] + hs)/dx),res-1)
            gzmin = max(int((xs[2] - hs)/dx+1), 0)
            gzmax = min(int((xs[2] + hs)/dx),res-1)
        else:
            gxmin = int((xs[0] - hs)/dx+1)
            gxmax = int((xs[0] + hs)/dx)
            gymin = int((xs[1] - hs)/dx+1)
            gymax = int((xs[1] + hs)/dx)
            gzmin = int((xs[2] - hs)/dx+1)
            gzmax = int((xs[2] + hs)/dx)

        # first have to do a prepass to get the weight
        kval = np.empty(int(2*hs/dx+1)**3) # save kernel values so don't have to recompute
        total_wt = 0
        j = 0                            
        for gx in range(gxmin, gxmax+1):            
            delta_x_Sqr = xs[0] - gx*dx
            delta_x_Sqr *= delta_x_Sqr
            for gy in range(gymin,gymax+1):
                delta_y_Sqr = xs[1] - gy*dx
                delta_y_Sqr *= delta_y_Sqr
                for gz in range(gzmin,gzmax+1):
                    delta_z_Sqr = xs[2] - gz*dx
                    delta_z_Sqr *= delta_z_Sqr
                    q = np.sqrt(delta_x_Sqr + delta_y_Sqr + delta_z_Sqr) * hinv
                    if q > 1:
                        kernel = 0
                    elif q <= 0.5:
                        kernel = 1 - 6*q*q + 6*q*q*q
                    else:
                        kernel = 2 * (1-q)*(1-q)*(1-q)
                    total_wt += kernel # to normalize out the kernel weights
                    kval[j] = kernel; j+=1

        # OK now do the actual deposition
        j = 0
        for gx in range(gxmin, gxmax+1):          
#            delta_x_Sqr = xs[0] - gx*dx
#            delta_x_Sqr *= delta_x_Sqr
            for gy in range(gymin,gymax+1):
#                delta_y_Sqr = xs[1] - gy*dx
#                delta_y_Sqr *= delta_y_Sqr
                for gz in range(gzmin,gzmax+1):
#                    delta_z_Sqr = xs[2] - gz*dx
#                    delta_z_Sqr *= delta_z_Sqr
#                    q = np.sqrt(delta_x_Sqr + delta_y_Sqr + delta_z_Sqr) * hinv
#                    if q > 1:
#                        kernel = 0
#                    elif q <= 0.5:
#                        kernel = 1 - 6*q*q + 6*q*q*q
#                    else:
#                        kernel = 2 * (1-q)*(1-q)*(1-q)
                    kernel = kval[j]; j+=1
                    total_wt += kernel
                    grid[gx%res,gy%res,gz%res] += f[i] * kernel / total_wt
    return grid / (dx*dx*dx)

   
# @jit
# def ComputeFaces(ngb, ingb, vol, dweights):
#     N, Nngb, dim = dweights.shape
#     result = np.zeros_like(dweights)
#     for i in range(N):
#         for j in range(Nngb):
#             result[i,j] += vol[i] * dweights[i,j]
#             if ingb[i,j] > -1: result[ngb[i,j],ingb[i,j]] -= vol[i] * dweights[i,j]
#     return result
