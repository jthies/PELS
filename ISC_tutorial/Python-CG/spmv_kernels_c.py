#/*******************************************************************************************/
#/* This file is part of the DHPC training material available at                            */
#/* https://gitlab.tudelft.nl/dhpc/training                                                 */
#/* You may redistribute it and/or modify it under the terms of the BSD-style licence       */
#/* included in this software.                                                              */
#/*                                                                                         */
#/* Contact: Jonas Thies (j.thies@tudelft.nl)                                               */
#/*                                                                                         */
#/*******************************************************************************************/
import numpy as np
from ctypes import *
from numpy.ctypeslib import as_ctypes
import os

from scipy.io import mmread
from matrix_generator import create_matrix
import argparse

# compile the C code into a shared library

# CLANG -- doesn't vectorize well
#os.system('clang -frtlib-add-rpath -fPIC -O3 -fopenmp -fvectorize -ffast-math -march=native -shared -o vector_ops.so vector_ops.c')

# Intel LLVM-based (icx)
os.system('icx -fPIC -O3 -qopenmp -xHOST -fimf-use-svml=true -Xclang -target-feature -Xclang +prefer-no-gather -o spmv_kernels.s -S spmv_kernels.c')
os.system('icx -fPIC -O3 -qopenmp -xHOST -fimf-use-svml=true -shared -Xclang -target-feature -Xclang +prefer-no-gather -o spmv_kernels.so spmv_kernels.c')

#Intel traditional (icc)
#os.system('icc -fPIC -O3 -qopenmp -xHOST -fimf-use-svml=true -o spmv_kernels.s -S spmv_kernels.c')
#os.system('icc -fPIC -O3 -qopenmp -xHOST -fimf-use-svml=true -shared -o spmv_kernels.so spmv_kernels.c')


# GCC
#os.system('gcc -fPIC -O3 -fopenmp -ffast-math -march=native -shared -o spmv_kernels.s -S spmv_kernels.c')
#os.system('gcc -fPIC -O3 -fopenmp -ffast-math -march=native -shared -o spmv_kernels.so spmv_kernels.c')

# import the C library -> c_functions object
so_file = "./spmv_kernels.so"
c_functions = CDLL(so_file)

# convenient alias
c_double_p = POINTER(c_double)
c_int_p = POINTER(c_int)

# need to explicitly set the argument types to avoid runtime errors for some reason...
c_functions.csr_spmv.argtypes = [c_size_t, c_double_p, c_int_p, c_int_p, c_double_p, c_double_p]

def csr_spmv(A, x, y):
    N=y.shape[0]
    c_functions.csr_spmv(N, as_ctypes(A.data), as_ctypes(A.indptr), as_ctypes(A.indices), as_ctypes(x), as_ctypes(y))
    return y

def get_argparser():
    parser = argparse.ArgumentParser(description='Run a CG benchmark.')
    parser.add_argument('-matfile', type=str, default='None',
                    help='MatrixMarket filename for matrix A')
    parser.add_argument('-matgen', type=str, default='None',
                    help='Matrix generator string  for matrix A (e.g., "Laplace128x64"')
    parser.add_argument('-rhsfile', type=str, default='None',
                    help='MatrixMarket filename for right-hand side vector b')
    parser.add_argument('-solfile', type=str, default='None',
                    help='MatrixMarket filename for exact solution x')
    parser.add_argument('-maxit', type=int, default=1000,
                    help='Maximum number of CG iterations allowed.')
    parser.add_argument('-tol', type=float, default=1e-6,
                    help='Convergence criterion: ||b-A*x||_2/||b||_2<tol')
    parser.add_argument('-fmt', type=str, default='CSR',
                    help='Sparse matrix format to be used [CSR, SELL]')
    parser.add_argument('-C', type=int, default=1,
                    help='Chunk size C for SELL-C-sigma format.')
    parser.add_argument('-sigma', type=int, default=1,
                    help='Sorting scope sigma for SELL-C-sigma format.')
    parser.add_argument('-seed', type=int, default=None,
                    help='Random seed to make runs reproducible')
    return parser


#if __name__=='__main__':
#    parser = get_argparser()
#    args = parser.parse_args()
#
#    if args.seed is not None:
#        np.random.seed(args.seed)
#
#    #if args.matfile != 'None':
#    #    if args.matgen!='None':
#    #        print('got both -matfile and -matgen, the latter will be ignored.')
#    #    if not args.matfile.endswith(('.mm','.mtx')):
#    #        raise(ValueError('Expecting MatrixMarket file with extension .mm or .mtx'))
#    #    A = csr_matrix(mmread(args.matfile))
#    #elif args.matgen != 'None':
#    A = create_matrix(args.matgen)
#    N = A.shape[0]
#
#    x=np.random.rand(N)
#    y=np.random.rand(N)
#    csr_spmv(A, x, y)
