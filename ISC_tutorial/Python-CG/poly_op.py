import numpy as np
from scipy.sparse import *
from kernels import *

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
        self.shape = A.shape
        self.dtype = A.dtype
        # store the inverse of the square-root of the diagonal
        # s.t. isqD*A*isqD has ones on the diagonal.
        self.isqD = spdiags([1.0/np.sqrt(A.diagonal())], [0])
        self.A1 = self.isqD*A*self.isqD
        self.L = -tril(self.A1,-1).tocsr()
        self.U = -triu(self.A1,1).tocsr()

        self.t1 = np.empty(self.shape[0], dtype=self.dtype)
        self.t2 = np.empty(self.shape[0], dtype=self.dtype)
        self.t3 = np.empty(self.shape[0], dtype=self.dtype)

        # in case A has CUDA arrays, also copy over our compnents:
        self.A1 = to_device(self.A1)
        self.L = to_device(self.L)
        self.U = to_device(self.U)

        self.t1 = to_device(self.t1)
        self.t2 = to_device(self.t2)
        self.t3 = to_device(self.t3)

    def prec_rhs(self, b, prec_b):
        '''
        Given the right-hand side b of a linear system
        Ax=b, computes prec_b=M1\b s.t.(this op)(M2x)=M1b solves the
        equivalent preconditioned linear system for the preconditioned
        solution vector M1x
        '''
        diag_spmv(self.isqD, b, self.t1)
        self._neumann(self.L, self.k, self.t1, prec_b)

    def unprec_sol(self, prec_x, x):
        '''
        Given the right-preconditioned solution vector prec_x = M2^{-1}x,
        returns x.
        '''
        self._neumann(self.U, self.k, prec_x, x)
        diag_spmv(self.isqD, x, x)

    def apply(self, w, v):
        '''
        Apply the complete (preconditioned) operator to a vector.

           v = M1 A M2 w.

        See class description for details.
        '''
        self._neumann(self.U, self.k, w, self.t1)
        spmv(self.A1, self.t1, self.t2)
        self._neumann(self.L, self.k, self.t2, v)


# protected
    def _neumann(self, M, k, rhs, sol):
        '''
        Apply truncated Neumann-series preconditioner to rhs, computing sol.

        If A = I-M, the Neumann series to approximate v=A^{-1}rhs is defined as

        inv(I-M) = sum_{k=0}^{\inf} M^k.

        It converges if ||M||<1
        (analogous to geometric series for scalar x: 1/(1-x) = sum_{k=0}^\inf x^k)
        '''
        # This is the naive implementation with 'back-to-back' spmv's.
        # Every intermediate vector M^jy is computed explicitly.

        axpby(1.0, rhs, 0.0, sol)
        for j in range(k):
            spmv(M,sol,self.t3)
            axpby(1.0,self.t3,1.0,sol)


