from numba import jit, vectorize, float32, float64

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
def GridSurfaceDensity(mass, x, h, gridres, rmax):
    L = rmax*2
    grid = np.zeros((gridres,gridres))
    dx = L/(gridres-1)
    N = len(x)
    for i in xrange(N):
        xs = x[i] + rmax
        hs = h[i]
        mh2 = mass[i]/hs**2

        gxmin = max(int((xs[0] - hs)/dx+1),0)
        gxmax = min(int((xs[0] + hs)/dx),gridres-1)
        gymin = max(int((xs[1] - hs)/dx+1), 0)
        gymax = min(int((xs[1] + hs)/dx), gridres-1)
        
        for gx in xrange(gxmin, gxmax+1):
            for gy in xrange(gymin,gymax+1):
                kernel = Kernel2D(((xs[0] - gx*dx)**2 + (xs[1] - gy*dx)**2)**0.5 / hs)
                grid[gx,gy] +=  kernel * mh2
                
    return grid
