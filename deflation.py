#/*******************************************************************************************/
#/* This file is part of the training material available at                                 */
#/* https://github.com/jthies/PELS                                                          */
#/* You may redistribute it and/or modify it under the terms of the BSD-style licence       */
#/* included in this software.                                                              */
#/*                                                                                         */
#/* Contact: Jonas Thies (j.thies@tudelft.nl)                                               */
#/*                                                                                         */
#/*******************************************************************************************/

import numpy as np
import scipy
import scipy.sparse.linalg as spla
from time import perf_counter
from numba import cuda, jit, int32, float64
import pymetis

from kernels import to_device, to_host, clone, axpby, spmv
from cuda_deflation import cu_restrict, cu_prolongate, cu_sell_restrict

from sellcs import sellcs_matrix

# total number of calls
calls = {'setup': 0, 'apply': 0}
# total elapsed time in seconds
time = {'setup': 0, 'apply': 0.0}

def reset_counters():
    r''' Reset performance counters for deflation setup and application. '''
    calls['setup'] = 0
    calls['apply'] = 0
    time['setup'] = 0.0
    time['apply'] = 0.0

def perf_report():
    r''' Print a performance report for the deflation operator. '''
    print('Deflation report:')
    print('  setup calls: %d, time: %10.4g s'%(calls['setup'], time['setup']))
    print('  apply calls: %d, time: %10.4g s'%(calls['apply'], time['apply']))

def compile_all():
    r''' 
    Pre-compile JIT kernels and class methods by running a small problem. 
    This removes compilation overhead from subsequent benchmark runs.
    '''
    n = 128
    nc = 4
    # Create a small SPD matrix for compilation
    A_csr = scipy.sparse.diags([[-1]*n, [2]*n, [-1]*n], [-1, 0, 1]).tocsr()
    A_gpu = to_device(A_csr)
    x = to_device(np.ones(n, dtype='float64'))
    y = to_device(np.zeros(n, dtype='float64'))

    A_defl = DeflatedOperator(A_csr, A_gpu, nc)
    A_defl.applyQ(x, y)
    A_defl.proj(x, y)
    reset_counters()

def partition_csr_matrix(A_csr, nparts):
    r'''
    Partition a sparse matrix's graph into nparts using METIS.

    Args:
        A_csr: A scipy.sparse.csr_matrix.
        nparts (int): Number of partitions.

    Returns:
        np.array (int32): Partition ID for each row/node.
    '''
    # Metis wants a "list of lists" of column indices,
    # so we construct it from the CSR format:
    adjacency_list = [
        A_csr.indices[A_csr.indptr[i] : A_csr.indptr[i+1]]
        for i in range(A_csr.shape[0])
    ]

    # Perform the partitioning
    # ncuts is the number of edges that span across different partitions
    ncuts, part = pymetis.part_graph(nparts, adjacency=adjacency_list)
    return np.array(part, dtype='int32')

@jit(nopython=True)
def invert_partitioning(nparts: int32, part: int32[:]) -> (int32[:], int32[:]):
    r'''
    ipart, nmembers = invert_partitioning(nparts, part)

    If part contains integers in the range 0 to nparts-1, inclusive,
    creates a row-major 2D array ipart[npart, max_i(part==i)] s.t.
    ipart[p, j] == i <-> part[i] == p and count(part[0:i]==p)=j.
    The second array returned indicates the actual number of global indicesj assigned to partition
    p, nmembers[p]<=ipart.shape[1]. Elements ipart[p,nmembers[p]:] are invalid (undefined).
    '''
    N = part.size
    # First count the partition sizes
    nmembers = np.zeros(nparts,dtype='int32')
    for i in range(N):
        nmembers[part[i]] += 1
    # Then construct the inverse mapping per partition:
    max_k = np.max(nmembers)
    ipart = -np.ones((nparts, max_k), dtype='int32')
    nmembers[:] = 0

    for i in range(N):
        pi = part[i]
        ni = nmembers[pi]
        ipart[pi, ni] = i
        nmembers[pi] += 1
    return ipart, nmembers

class DeflatedOperator:
    r'''
    Implements the Deflated Conjugate Gradient (DPCG) operator.
    The subspace is defined by a partition of the domain (coarse grid).
    Q = V (V^T A V)^{-1} V^T is the projection into the deflation space.
    '''
    def __init__(self, A_csr, A_gpu, nc):
        r'''
        Initialize the deflated operator by partitioning the matrix and 
        factorizing the coarse-grid operator E = V^T A V.

        Args:
            A_csr (scipy.sparse.csr_matrix): System matrix on CPU for setup.
            A_gpu: System matrix on GPU for projection operations (must be SELL-C-sigma i.e. sellcs_matrix).
            nc (int): Number of coarse partitions (size of coarse problem).
        '''
        t0 = perf_counter()
        if (type(A_csr) is not scipy.sparse.csr_matrix or
            type(A_gpu) is not sellcs_matrix or
            hasattr(A_gpu, 'cu_data') == False):
            raise Exception(f'DeflatedOperator requires a CSR (host) and SELL-C-sigma (device) matrix.\n'+
                            f'Found A_csr[{type(A_csr)}] and A_gpu[{type(A_gpu)}].')
        self.shape = A_csr.shape
        self.dtype = A_csr.dtype
        self.A = A_gpu
        self.nparts = nc
        self.n = A_csr.shape[0]

        # 1. Partitioning using METIS
        self.part = partition_csr_matrix(A_csr, nc)
        self.cu_part = to_device(self.part)

        #print('part:')
        #for i in range(self.n):
        #    print(f'{i}\t{self.part[i]}')

        # determine the size of each partition and the inverse mapping 'ipart'
        self.ipart, self.nmembers = invert_partitioning(self.nparts, self.part)

        self.cu_ipart = to_device(self.ipart)
        self.cu_nmembers = to_device(self.nmembers)

        self.valV = 1.0/np.sqrt(self.nmembers.astype(np.float64))
        self.cu_valV = to_device(self.valV)

        nchunks = len(self.A.indptr)-1
        C = self.A.C
        n = self.A.shape[0]

        self.cu_A_c = cuda.device_array((self.nparts, self.nparts), dtype=self.dtype)

        cu_sell_restrict[nchunks, C](self.A.cu_data, self.A.cu_indptr, self.A.cu_indices, n, 
                                     self.cu_part, self.cu_valV,self.cu_A_c)
        cuda.synchronize()

        self.A_c = to_host(self.cu_A_c)

        self.L_c = np.linalg.cholesky(self.A_c)

        t1 = perf_counter()

        time['setup'] += t1 - t0
        calls['setup'] += 1

    def applyQ(self, x, y):
        r'''
        computes y = Q*x
        with Q = V(V'AV)\V'
        '''
        # coarsen ("restrict"): x_c = V^T@x
        x_c = cuda.device_array((self.nparts,),dtype='float64')
        threadsperblock = 128
        blockspergrid = self.nparts
        cu_restrict[blockspergrid, threadsperblock](self.cu_ipart, self.cu_nmembers, self.cu_valV, x, x_c)

        # solve the projected linear system, y_c = (V^TAV) \ x_c
        y_c = np.linalg.solve(self.L_c.T, np.linalg.solve(self.L_c, to_host(x_c)))

        # interpolate y = V*y_c
        threadsperblock = 256
        blockspergrid = (self.n + threadsperblock - 1) // threadsperblock
        cu_prolongate[blockspergrid, threadsperblock](self.cu_part, self.valV, to_device(y_c), y)

    def proj(self, x, y, tmp):
        r'''
        Compute the projection orthogonal to the deflation space: y = (I - QA)x.

        Args:
            x: Input vector on GPU.
            y: Output vector on GPU.
            tmp: a temporary work vector of the same type and length as x, y
        '''
        # We need a temporary for A@x
        spmv(self.A, x, tmp)

        # y = Q (A x)
        self.applyQ(tmp, y)

        # y = x - y
        axpby(1.0, x, -1.0, y)


    def projT(self, x, y, tmp):
        r'''
        computes y = (I - AQ)x
        with Q = V(V'AV)\V' (note that A is symmetric).
        '''
        return x - self.A @ self.applyQ(x)
        self.applyQ(x, tmp)
        spmv(self.A, tmp, y)

        # y = x - y
        axpby(1.0, x, -1.0, y)

    def apply(self, x, y):
        r'''
        apply deflated operator A,
          y = (I - AQ)A(I - QA)x,
          Q = V (V'AV)\(V'x)
        '''
        # temporary vectors
        v1 = cuda.device_array((self.n,), x.dtype)
        v2 = cuda.device_array((self.n,), x.dtype)
        self.proj(x, v1, v2) # compute v1, use v2 as workspace
        spmv(self.A, v1, v2) # compute v2, use v1 as input
        self.projT(v2, y, v1) # compute y, use v2 as input and v1 as workspace
        return

