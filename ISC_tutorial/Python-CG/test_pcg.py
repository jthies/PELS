import unittest
import pytest
from parameterized import parameterized_class
from test_kernels import diff_norm

import numpy as np
from kernels import *
from poly_op import *
from pcg import *
from matrix_generator import create_matrix

@parameterized_class(('Matrix', 'maxit', 'poly_k'),[
    ['Laplace10x10', 30, -1],
    ['Laplace10x10', 30, 0],
    ['Laplace20x20', 60, 0],
    ['Laplace40x40',120, 0],
    ['Laplace40x40',80, 1],
    ['Laplace40x40', 66, 2] ])
class CgTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(12345678)
        self.tol = 1.0e-6
        if self.Matrix.endswith('.mm'):
            self.A = scipy.io.mmread(self.Matrix).tocsr()
        else:
            self.A = create_matrix(self.Matrix)
        self.x_ex =np.random.rand(self.A.shape[0])
        self.b = self.A*self.x_ex
        self.x0 = np.zeros(self.A.shape[0], self.A.dtype)
        self.r = np.empty_like(self.x0)

        if available_gpus()>0:
            self.A = to_device(self.A)
            self.x0 = to_device(self.x0)
            self.b = to_device(self.b)
            self.r = to_device(self.r)
        self.norm_b = np.sqrt(dot(self.b,self.b))

        self.A_op = None
        if  self.poly_k>0:
            self.A_op = poly_op(self.A, self.poly_k)

    def test_poly_op(self):
        if self.A_op != None:
            n = self.A.shape[0]
            x_host = np.random.rand(n)
            y_host = x_host.copy()
            x = to_device(x_host)
            y = to_device(y_host)

            self.A_op.apply(x, y)

            for i in range(self.poly_k):
                y_host += self.A_op.U*y_host
            y_host = self.A_op.A1*y_host
            for i in range(self.poly_k):
                y_host += self.A_op.L*y_host
            assert(diff_norm(y,y_host)<1.0e-12)

    def test_poly_op_symmetric(self):
        if self.A_op != None:
            n = self.A.shape[0]
            k=30
            X = np.random.rand(n,k)
            Y = X.copy()
            for j in range(k):
                self.A_op.apply(X[:,j], Y[:,j])
            Z = np.matmul(X.transpose(),Y)
            assert(np.linalg.norm(Z-Z.transpose())<1.0e-14)

    def test_cg(self):

        rhs = self.b
        A = self.A
        if self.A_op != None:
            rhs = copy(self.b)
            A   = self.A_op
            self.A_op.prec_rhs(self.b, rhs)

        sol, relres, iter = cg_solve(A, rhs, self.x0, self.tol, self.maxit)

        x = copy(sol)
        if self.A_op != None:
            self.A_op.unprec_sol(sol, x)

        assert(diff_norm(x, self.x_ex)/self.norm_b<self.tol)
        r = copy(x)
        spmv(self.A, x, r)
        axpby(1.0,self.b, -1.0, r)
        norm_r = np.sqrt(dot(r,r))
        assert(norm_r/self.norm_b<self.tol)
