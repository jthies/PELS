import numpy as np
from scipy.sparse import *
import kernels

class poly_op:
    '''
    Given a matrix A and an integer k, this operator constructs the splitting
    (cf. Jacobi iteration):

        D^{-1/2}AD^{-1/2} = I - (L+U)

    The 'apply' function implements a preconditioned matrix-vector product

        w = M1 A M2 v

    where M1 and M2 approximately solve

    (I-L)v=w    and    (I-U)v=w, respectively,

    using the truncated Neumann series:

        v_0 = 0
        v \approx \sum_{i=0}^k L^{i} w.

    For symmetric and positive (spd) matrices A, this preconditioned operator is spd,
    which makes it suitable for CG.

    '''

    def __init__(self, A, k):

        self.k = k
        # store the inverse of the square-root of the diagonal
        # s.t. isqD*A*isqD has ones on the diagonal.
        self.isqD = spdiags([1.0/np.sqrt(A.diagonal())], [0])
        self.A1 = self.isqD*A*self.isqD
        self.L = tril(self.A1,-1).tocsr()
        self.U = triu(self.A1,1).tocsr()

        # in case A has CUDA arrays, also copy over our compnents:
        self.A1 = to_device(self.A1)
        self.L = to_device(self.L)
        self.U = to_device(self.U)

    def prec_rhs(b, prec_b):
        '''
        Given the right-hand side b of a linear system
        Ax=b, computes prec_b=M1\b s.t.(this op)(M2x)=M1b solves the
        equivalent preconditioned linear system for the preconditioned
        solution vector M1x
        '''
        diag_spmv(self.isqD, b, self.t1)
        neumann(self.L, self.k, self.t1, prec_b)

    def unprec_sol(prec_x, x):
        '''
        Given the left-preconditioned solution vector prec_x = M2\x,
        returns x.
        '''
        _neumann(self.U, self.k, prec_x, x)
        diag_spmv(self.isqD, x, x)

    def apply(self, w, v):
        '''
        Apply the complete (preconditioned) operator to a vector.

           v = M1 A M2 w.

        See class description for details.
        '''
        _neumann(self.U, self.k, w, self.tmp)
        spmv(self.A1, self.t1, self.t2)
        _neumann(self.L, self.k, self.t2, v)

    def _neumann(self, M, k, w, v):
        '''
        Apply truncated Neumann-series preconditioner to w, returning v.

        If A = I-M, the Neumann series to approximate v=A^{-1}w is defined as

        inv(I-M) = sum_{k=0}^{\inf} M^k.

        It converges if ||M||<1
        (analogous to geometric series for scalar x: 1/(1-x) = sum_{k=0}^\inf x^k)
        '''
        # This is the naive implementation with 'back-to-back' spmv's.
        # Every intermediate vector M^jy is computed explicitly.
        copy(x,y)
        for j in range(k):
            spmv(M,y,self.t1)
            axpby(1.0,self.t1,1.0,y)
