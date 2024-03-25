import numpy as np
from numba import jit, prange, get_num_threads
import scipy

# Benchmarks run on 2x 32-core Intel Xeon 6448Y "Sapphire Rapids"
benchmarks_numa = {'load': 121, 'store':  91, 'copy':  95, 'triad':  98}
benchmarks_node = {'load': 480, 'store': 348, 'copy': 384, 'triad': 388}

def memory_benchmarks():
    nthreads = get_num_threads()
    if nthreads == 16:
        return benchmarks_numa
    elif nthreads == 64:
        return benchmarks_node
    else:
        print('Warning: Memory benchmarks only available for 16 or 64 cores of DelftBlue Phase 2 nodes.')
        return benchmarks_node


@jit(nopython=True, parallel=True)
def copy_vector(x):
    y = np.empty_like(x)
    for i in prange(x.size):
        y[i] = x[i]
    return y

@jit(nopython=True, parallel=True)
def copy_csr_arrays(Adata, Aindptr, Aindices):
    data = np.empty_like(Adata)
    indices = np.empty_like(Aindices)
    indptr = np.empty_like(Aindptr)
    nrows = len(indptr)-1
    nnz   = len(Adata)
    for i in prange(nrows):
        indptr[i] = Aindptr[i]
        indptr[i+1] = Aindptr[i+1]
        for j in range(indptr[i], indptr[i+1]):
            data[j] = Adata[j]
            indices[j] = Aindices[j]
    return data, indices, indptr

@jit(nopython=True, parallel=True)
def csr_spmv(valA,rptrA,colA, x, y):
    '''
    Usage:
      - if A is a scipy.sparse.csr_matrix,
      you can get the components by valA = A.data; rptrA = cptrA; colA = A.indices.
      - x and y numpy arrays of size A.shape[0] and A.shape[1], respectively.

      Then this function returns y = A*x
    '''
    for row in prange(len(rptrA)-1):
        y[row] = 0
        for j in range(rptrA[row], rptrA[row+1]):
            y[row] += valA[j] * x[colA[j]]

@jit(nopython=True, parallel=True)
def sell_spmv(valA, cptrA, colA, C, x, y):
    '''
    Usage: sell_spmv(valA, cptrA, colA, C, x, y) computes y=A*x for a sellcs.sellcs_matrix A,
           where the members are extracted like this:
           valA, cptrA, colA, C = A.data, cptrA, A.indices, A.C
           Sorting of in and/or output vectors is **not performed** by this function:
           If A.sigma>1, the user is responsible to provided x in permuted form and
           "unpermute" y if desired.
    '''
    nchunks = len(cptrA)-1
    nrows = x.size
    for chunk in prange(nchunks):
        offs = cptrA[chunk]
        row0 = chunk*C
        row1 = min(row0+C, nrows)
        c    = row1-row0
        w    = (cptrA[chunk+1]-offs)//c
        #print('rows %d:%d, c=%d, w=%d'%(row0,row1,c,w))
        y[row0:row1] = 0
        for j in range(w):
                y[row0:row1] += valA[offs+j*c:offs+(j+1)*c] * x[colA[offs+j*c:offs+(j+1)*c]]


@jit(nopython=True, parallel=True)
def init(x, val):
    for i in prange(x.size):
        x[i]=val

@jit(nopython=True, parallel=True)
def axpby(a,x,b,y):
    for i in prange(x.size):
        y[i]=a*x[i]+b*y[i]

@jit(nopython=True, parallel=True)
def dot(x,y):
    s=0.0
    for i in prange(x.size):
        s += x[i]*y[i]
    return s

