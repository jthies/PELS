#/*******************************************************************************************/
#/* This file is part of the training material available at                                 */
#/* https://github.com/jthies/PELS                                                          */
#/* You may redistribute it and/or modify it under the terms of the BSD-style licence       */
#/* included in this software.                                                              */
#/*                                                                                         */
#/* Contact: Jonas Thies (j.thies@tudelft.nl)                                               */
#/*                                                                                         */
#/*******************************************************************************************/

import sys
import unittest
import pytest
import numpy as np
from numpy.linalg import norm
import scipy.sparse
from kernels import to_device, to_host, clone, available_gpus, spmv
from deflation import DeflatedOperator, partition_csr_matrix
from cuda_deflation import *
from matrix_generator import create_matrix
from sellcs import sellcs_matrix

from test_kernels import diff_norm


def build_numpy_V(A_defl):
        n = A_defl.shape[0]
        nc = A_defl.nparts
        V = np.zeros((n, nc),dtype='float64')
        for i in range(n):
            V[i,A_defl.part[i]] = 1.0
        nmembers = np.sum(V, axis=0, dtype='int32')
        V = V @ np.diag(1.0/np.sqrt(nmembers.astype('float64')))
        return V

class DeflationTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(12345678)
        self.n = 200
        self.nc = 10
        # Create a simple 1D Laplace matrix
        self.A_csr = scipy.sparse.diags([[-1.0]*(self.n-1), [2.0]*self.n, [-1.0]*(self.n-1)], [-1, 0, 1]).tocsr()
        self.A_gpu = to_device(sellcs_matrix(self.A_csr, C=32, sigma=1))

        # Exact solution and RHS
        self.x_ex = np.random.rand(self.n)
        self.b_gpu = to_device(self.A_csr @ self.x_ex)
        self.v_tmp = clone(self.b_gpu)

        # Deflated Operator
        self.A_defl = DeflatedOperator(self.A_csr, self.A_gpu, self.nc)

        # explicitly build the V operator using numpy
        self.V = build_numpy_V(self.A_defl)
        self.A_c = self.V.T @ self.A_csr @ self.V

        self.eps = 1e-12

    def test_partitioning(self):
        part = partition_csr_matrix(self.A_csr, self.nc)
        assert len(part) == self.n
        assert np.max(part) == self.nc - 1
        assert np.min(part) == 0
        # Check that it returns an array of int32 as expected by CUDA kernels
        assert part.dtype == np.int32


    def test_restrict_ones(self):
        ''' Test that V^Te_n = e_nc, with e_k=ones(k,1)
        '''
        v = np.ones(self.n)
        v_c_ref = self.V.T @ v

        assert(np.all(np.abs(v_c_ref-1) < self.eps))

        v_c = to_device(np.zeros(self.nc))
        threadsperblock = 128
        blockspergrid = self.nc
        cu_restrict[blockspergrid, threadsperblock](self.A_defl.cu_ipart, self.A_defl.cu_nmembers,
                                                    self.A_defl.cu_valV, to_device(v), v_c)
        cuda.synchronize()
        assert diff_norm(v_c, v_c_ref) < self.eps

    def test_prolongate_ones(self):
        ''' Test that V e_nc = e_n, with e_k=ones(k,1)
        '''
        v_c = self.A_defl.nmembers.astype('float64')
        v_ref = self.V @ v_c

        assert(np.all(np.abs(v_ref-1.0) < self.eps))

        v = to_device(np.zeros(self.n))
        threadsperblock = 256
        blockspergrid = (self.n + threadsperblock - 1) // threadsperblock
        cu_prolongate[blockspergrid, threadsperblock](self.A_defl.cu_part, self.A_defl.cu_valV,
                                                    to_device(v_c), v)
        cuda.synchronize()
        assert diff_norm(v, v_ref) < self.eps

    def test_coarsen_compare_numpy(self):
        '''
        Compute A_c = V^TAV on host (numpy) and device, and compare the two
        '''
        A_c = to_device(np.zeros((self.nc,self.nc), dtype='float64'))
        nchunks = len(self.A_gpu.indptr)-1
        C = self.A_gpu.C
        n = self.n

        cu_sell_restrict[nchunks, C](self.A_gpu.cu_data, self.A_gpu.cu_indptr, self.A_gpu.cu_indices, n,
                                     self.A_defl.cu_part, self.A_defl.cu_valV, A_c)
        cuda.synchronize()
        assert diff_norm(A_c, self.A_c) < self.eps

    def test_applyQ_compare_numpy(self):
        ''' Test that Q = V(V'AV)^{-1}V' satisfies V'A Q = V' '''
        x = to_device(np.random.rand(self.n))
        y = clone(x)
        self.A_defl.applyQ(x, y)

        y_host = to_host(y)
        x_host = to_host(x)

        # We now have y = Qx = V(V'AV)^{-1} V'x
        # A_c*V' y should match V' x, where A_c = V'AV
        lhs = self.A_c @ (self.V.T @ y_host)
        rhs = self.V.T @ x_host

        error = diff_norm(lhs, rhs)
        assert error < self.eps

    def test_applyQ_idempotent(self):
        ''' Test that Q A Q = Q '''
        x = to_device(np.random.rand(self.n))
        y1 = clone(x)
        y2 = clone(x)
        temp = clone(x)

        # y1 = Q x
        self.A_defl.applyQ(x, y1)

        # temp = A (Q x)
        spmv(self.A_gpu, y1, temp)

        # y2 = Q (A Q x)
        self.A_defl.applyQ(temp, y2)

        error = diff_norm(y1, y2)
        assert error < self.eps

    def test_proj_orthogonality(self):
        ''' Test that P = I - QA satisfies V' A P = 0 '''
        x = to_device(np.random.rand(self.n))
        y = clone(x)

        # y = (I - QA) x
        self.A_defl.proj(x, y, self.v_tmp)

        y_host = to_host(y)

        # V' A y should be zero
        res = self.V.T @ (self.A_csr @ y_host)
        error = norm(res) / norm(to_host(x))
        assert error < self.eps

    def test_proj_idempotent(self):
        ''' Test that P = I - QA is idempotent: P^2 = P '''
        x = to_device(np.random.rand(self.n))
        y1 = clone(x)
        y2 = clone(x)

        # y1 = P x
        self.A_defl.proj(x, y1, self.v_tmp)

        # y2 = P (P x)
        self.A_defl.proj(y1, y2, self.v_tmp)

        error = diff_norm(y1,y2)
        assert error < self.eps

#@pytest.mark.parametrize('Matrix', ['Laplace16x16','Ddiag13', 'Dsprandn388'])
#def test_deflation_parametrized(Matrix):
#    import scipy.io
#    from matrix_generator import create_matrix
#    try:
#        A_csr = scipy.sparse.csr_matrix(scipy.io.mmread('matrices/'+Matrix+'.mm.gz'))
#    except:
#        A_csr = create_matrix(Matrix)
#
#    n = A_csr.shape[0]
#    nc = min(n // 2, 5) # Small nc for small matrices
#
#    A_sell = sellcs_matrix(A_csr, C=32, sigma=1)
#
#    A_gpu = to_device(A_sell)
#    A_defl = DeflatedOperator(A_csr, A_gpu, nc)
#    V = build_numpy_V(A_defl)
#
#    x = to_device(np.random.rand(n))
#    y = clone(x)
#    v_tmp = clone(x)
#    A_defl.proj(x, y, v_tmp)
#
#    # Orthogonality check: V' A P x = 0
#    y_host = to_host(y)
#    res = V.T @ (A_csr @ y_host)
#    assert norm(res) / norm(to_host(x)) < 1e-10
#

if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    unittest.main()
