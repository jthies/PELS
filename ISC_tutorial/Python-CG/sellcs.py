import numpy as np
from scipy.sparse import csr_matrix

class sellcs_matrix:

    def __init__(self, A_csr, C=32, sigma=1):

        if C<1 or sigma<0 or (sigma!=1 and sigma%C!=0):
            raise ValueError('Invalid parameters C and/or sigma: sigma should be 1 or a multiple of C.')

        self.shape = A_csr.shape
        self.dtype = A_csr.dtype
        self.C = C
        self.sigma = sigma

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
