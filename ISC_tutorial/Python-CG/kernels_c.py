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

# some stuff is missing here up to now:
from kernels_cpu import copy_vector, copy_csr_arrays, multiple_axpbys, memory_benchmarks, sell_spmv, vscale

# compile the C code into a shared library
os.system("make -j")


# import the C library -> c_functions object
so_file = "./libkernels.so"
c_functions = CDLL(so_file)

# convenient alias
c_double_p = POINTER(c_double)
c_int_p = POINTER(c_int)

# need to explicitly set the argument types to avoid runtime errors for some reason...
c_functions.csr_spmv.argtypes = [c_size_t, c_double_p, c_int_p, c_int_p, c_double_p, c_double_p]
c_functions.axpby.argtypes = [c_size_t, c_double, c_double_p, c_double, c_double_p]
c_functions.dot.argtypes = [c_size_t, c_double_p, c_double_p]
c_functions.dot.restype = c_double
c_functions.init.argtypes = [c_size_t, c_double_p, c_double]


def csr_spmv(valA, rptrA, colA, x, y):
    N=y.shape[0]
    c_functions.csr_spmv(N, as_ctypes(valA), as_ctypes(rptrA), as_ctypes(colA), as_ctypes(x), as_ctypes(y))

def init(x, val):
    N = x.size
    c_functions.init(N, as_ctypes(x), val)

def axpby(a,x,b,y):
    N = x.size
    c_functions.axpby(N, a, as_ctypes(x), b, as_ctypes(y))

def dot(x,y):
    N = x.size
    s = c_functions.dot(N, as_ctypes(x), as_ctypes(y))
    return s

def multiple_axpbys(a, x, b, y, ntimes):
    for it in range(ntimes):
        axpby(a,x,b,y)

