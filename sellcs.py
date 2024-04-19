#/*******************************************************************************************/
#/* This file is part of the training material available at                                 */
#/* https://github.com/jthies/PELS                                                          */
#/* You may redistribute it and/or modify it under the terms of the BSD-style licence       */
#/* included in this software.                                                              */
#/*                                                                                         */
#/* Contact: Jonas Thies (j.thies@tudelft.nl)                                               */
#/*                                                                                         */
#/*******************************************************************************************/

from numba import jit, prange
import numpy as np
from scipy.sparse import csr_matrix

class sellcs_matrix:

    def __init__(self, A_csr=None, A_arrays=None, C=32, sigma=1, shape=None, dtype=None):
        '''
        The sellcss_matrix is a simple object without any methods or operators. To perform a
        sparse matrix-vector multiplication, use kernels.spmv(A, x, y) instead of y=A*x.
        There are two ways to construct this matrix class:

        1. from a scipy.sparse.csr_matrix: A = sellcs_matrix(A_csr, C, sigma)
        2. from existing SELL-C-sigma arrays: A = sellcs_matrix((data,indices,indptr, permute, unpermute, nnz), C, shape)
        '''
        if (A_csr is None and A_arrays is None) or (A_csr!=None and A_arrays!=None):
            raise ValueError('Invalid input to sellcs_matrix: need either A_csr or (data, indices, indptr)')
        if C<1 or sigma<0 or (sigma!=1 and sigma%C!=0):
            raise ValueError('Invalid parameters C and/or sigma: sigma should be 1 or a multiple of C.')

        self.C = C
        self.sigma = sigma

        if A_csr != None:
            self.from_csr(A_csr)
        elif A_arrays != None:
            self.data, self.indices, self.indptr, self.permute, self.unpermute, self.nnz = A_arrays
            self.nchunks = self.indptr.size-1
            self.shape = shape
            self.dtype = self.data.dtype

    def from_csr(self, A_csr):

        self.shape = A_csr.shape
        self.dtype = A_csr.dtype

        C = self.C
        sigma = self.sigma

        nrows = self.shape[0]
        nchunks = (nrows+C-1)//C
        self.nchunks = nchunks
        self.indptr  = np.zeros(nchunks+1, dtype='int64')
        self.permute = np.arange(nrows)
        self.unpermute = np.empty_like(self.permute)
        self.nnz = A_csr.nnz
        scope = max(C,sigma)

        chunk = 0

        for row_offset in range(0,nrows, scope):
            last = min(row_offset+scope, nrows)
            rng0=range(row_offset, last)
            rng1=range(row_offset+1, last+1)
            row_len = A_csr.indptr[rng1] - A_csr.indptr[rng0]
            # Note: the only way to make numpy sort in descending order is to
            #       give it the negative values and sort ascending :)
            if sigma>1:
                idx = np.argsort(-row_len)
                row_len = row_len[idx]
                self.permute[rng0] = row_offset+idx
            for i in range(row_offset, last, C):
                c = min(C, nrows-i)
                w = max(row_len[i-row_offset:i-row_offset+c])
                self.indptr[chunk+1] = self.indptr[chunk] + c*w
                chunk += 1

        nnz_stored = self.indptr[nchunks]
        self.unpermute[self.permute] = np.arange(nrows)
        self.data = np.zeros(nnz_stored, dtype=A_csr.data.dtype)
        self.indices = np.zeros(nnz_stored, dtype=A_csr.indices.dtype)

        for chunk in range(nchunks):
            c = min(C*(chunk+1), nrows)-C*chunk
            i0 = C*chunk
            i1 = i0 + c
            for i in range(i0, i1):
                rowA = self.permute[i]
                k0 = A_csr.indptr[rowA]
                k1 = A_csr.indptr[rowA+1]
                for k in range(k0, k1):
                    pos  = self.indptr[chunk] + (k-k0)*c + (i-i0)
                    self.data[pos] = A_csr.data[k]
                    self.indices[pos] = self.unpermute[A_csr.indices[k]]

    def __str__(self):
        out = 'SELL-%d-%d matrix, shape=[%d %d]\n'%(self.C,self.sigma, self.shape[0],self.shape[1])
        if self.sigma>1:
            out += 'permute='+str(self.permute)+'\n'
            out += 'unpermute='+str(self.permute)+'\n'
        out += 'indptr='+str(self.indptr)+'\n'
        out += 'indices='+str(self.indices)+'\n'
        out += 'data='+str(self.data)+'\n'
        return out
