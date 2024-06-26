#/*******************************************************************************************/
#/* This file is part of the training material available at                                 */
#/* https://github.com/jthies/PELS                                                          */
#/* You may redistribute it and/or modify it under the terms of the BSD-style licence       */
#/* included in this software.                                                              */
#/*                                                                                         */
#/* Contact: Jonas Thies (j.thies@tudelft.nl)                                               */
#/*                                                                                         */
#/*******************************************************************************************/

from kernels import *
from kernels_cpu import csr_spmv as csr_spmv_numba
from kernels_c import csr_spmv as csr_spmv_c
from matrix_generator import create_matrix
import argparse
from timeit import timeit



parser = argparse.ArgumentParser(description='Run CSR Spmv in two variants (numba.jit and C)')

parser.add_argument('-t', type=str, default='cpu',
                    help='Select "cpu" or "gpu"')
parser.add_argument('-matgen', type=str, default='Laplace1000x1000',
                    help='number of elements in x, y and z.')
parser.add_argument('-k', type=int, default=50,
                    help='number of runs')

args = parser.parse_args()

if args.matgen != 'None':
    A = create_matrix(args.matgen)
    N = A.shape[0]
    x=np.random.rand(N)
    y=np.random.rand(N)
    # get NUMA placement right by doing a parallel copy
    x = copy(x)
    y = copy(y)
    A = copy(A)

    # copile the two spmv variants:
    spmv(A,x,y)
    csr_spmv_numba(A.data,A.indptr,A.indices,x,y)
    csr_spmv_c(A.data,A.indptr,A.indices,x,y)

    print('Numba csr_spmv, matrix='+args.matgen)
    t_tot = timeit('csr_spmv_numba(A.data,A.indptr,A.indices,x,y)', globals=globals(), number=args.k)
    print('Total time for %d SpMVs: %e seconds.'%(args.k, t_tot))

    print('C csr_spmv, matrix='+args.matgen)
    t_tot = timeit('csr_spmv_c(A.data,A.indptr,A.indices,x,y)', globals=globals(), number=args.k)
    print('Total time for %d SpMVs: %e seconds.'%(args.k, t_tot))

    print('')
    print('Run our own loop, C csr_spmv')
    t0 = perf_counter()
    for i in range(args.k):
        csr_spmv_c(A.data,A.indptr,A.indices,x,y)
    t1 = perf_counter()
    t_tot = t1-t0
    print('Total time for %d SpMVs: %e seconds.'%(args.k, t_tot))
