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
from numpy.linalg import norm
from scipy.sparse import *
from scipy.io import mmread
from pcg import cg_solve
from kernels import *
from poly_op import *
from sellcs import sellcs_matrix

from matrix_generator import create_matrix
from pels_args import *

import numba

import gc

if __name__ == '__main__':

    # Note: The Python garbage collector (gc)
    # kills the performance of the C kernels
    # for some obscure reason. For the pure
    # Python/numba/cuda kernels, this is not
    # the case, but to make things run with
    # C kernels and RACE, we disable it here.
    #gc.disable()

    parser = get_argparser()

    # add driver-specific command-line arguments for polynomial preconditioning with or without RACE:
    parser.add_argument('-printerr', action=BooleanOptionalAction,
                    help='Besides the residual norm, also compute and print the error norm.')
    parser.add_argument('-rhsfile', type=str, default='None',
                    help='MatrixMarket filename for right-hand side vector b')
    parser.add_argument('-solfile', type=str, default='None',
                    help='MatrixMarket filename for exact solution x')
    parser.add_argument('-poly_k', type=int, default=0,
                    help='Use a degree-k polynomial preconditioner based on the Neumann series.')
    parser.add_argument('-use_RACE', action=BooleanOptionalAction,
                    help='Use RACE for cache blocking.')
    parser.add_argument('-cache_size', type=float, default=30,
                    help='Cache size used to perform RACE\'s cache blocking')


    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    if args.matfile != 'None':
        if args.matgen!='None':
            print('got both -matfile and -matgen, the latter will be ignored.')
        if not args.matfile.endswith(('.mm','.mtx')):
            raise(ValueError('Expecting MatrixMarket file with extension .mm or .mtx'))
        A = csr_matrix(mmread(args.matfile))
    elif args.matgen != 'None':
        A = create_matrix(args.matgen)
    N = A.shape[0]

    if args.solfile!='None':
        x_ex=mmread(args.solfile).reshape(N)
    else:
        x_ex=np.random.rand(N)

    if args.rhsfile!='None':
        b=mmread(args.rhsfile).reshape(N)
    else:
        b=A*x_ex

    x0 = np.zeros(N,dtype='float64')

    print('norm of rhs: %e'%(norm(b)))
    print('rel. residual of given solution: %e'%(norm(A*x_ex-b)/norm(b)))

    tol = args.tol
    maxit = args.maxit

    sigma=1

    A_csr = A # we may need it for creating the preconditioner
              # in case the user wants a SELL-C-sigma matrix.

    if args.fmt=='SELL':
        C=args.C
        sigma=args.sigma
        A = sellcs_matrix(A_csr=A_csr, C=C, sigma=sigma)
        b = b[A.permute]
        print('Matrix format: SELL-%d-%d'%(C,sigma))
        A_csr = A_csr[A.permute[:,None], A.permute]
    else:
        print('Matrix format: CSR')

    if available_gpus()>0:
        type = 'gpu'
    else:
        type = 'cpu'

    print('Will run on '+type)

    if type=='gpu':
        x0 = to_device(x0)
        b  = to_device(b)
        A  = to_device(A)
    elif type=='cpu':
        # First-touch (re-)initialize all arrays.
        # We can't do it consistently beforehand because
        # we're calling library functions like numpy 'rand'
        # and scipy 'mmread'.
        x0 = copy(x0)
        b  = copy(b)
        A  = copy(A)
    # take compilation time out of the balance:
    compile_all()

    A_prec = A
    b_prec = b

    # we want to make sure what we measure during CG in total
    # is consistent with the sum of the kernel calls and their
    # runtime as predicted by the roofline model, so reset all
    # counters and timers:
    reset_counters()

    t0 = perf_counter()

    t0_pre = perf_counter()
    if args.poly_k>0:
        # building preconditioners typically requires a certain format,
        # in our case, the poly_op class uses scipy functions tril and triu,
        # which are not implemented by the sellcs_matrix class.
        A_prec = poly_op(A_csr, args.poly_k)
        if A_prec.mpkHandle != None:
            b = b[A_prec.permute]
            A_csr = A_csr[A_prec.permute[:,None], A_prec.permute]
        if args.fmt == 'SELL':
            # note: If A was originally sorted by row-length (sigma>1), use the same
            # sorting for L and U to avoid intermittent permutation by setting sigma=1.
            # There still seems to be some kind of bug, though, because the number of
            # iterations will increase with poly_k>0 and sigma>1. Hence this warning.
            A_prec.L = to_device(sellcs_matrix(A_csr=A_prec.L, C=args.C, sigma=1))
            A_prec.U = to_device(sellcs_matrix(A_csr=A_prec.U, C=args.C, sigma=1))
        b_prec = copy(b)
        A_prec.prec_rhs(b, b_prec)

    x_ex_in = None
    if args.printerr:
        x_ex_in = x_ex

    t1_pre = perf_counter()

    t0_soln = perf_counter()
    x_prec, relres, iter = cg_solve(A_prec,b_prec,x0,tol,maxit, x_ex=x_ex_in)
    t1_soln = perf_counter()

    if args.poly_k>0:
        x = clone(x_prec)
        A_prec.unprec_sol(x_prec, x)
    else:
        x = x_prec

    t1 = perf_counter()
    t_pre = t1_pre-t0_pre
    t_soln = t1_soln-t0_soln
    t_CG = t1-t0
    gc.enable()

    x = to_host(x)

    print('number of CG iterations: %d'%(iter))
    res = np.empty_like(x)
    spmv(A_csr,x,res)
    res=b-res
    print('relative residual of computed solution: %e'%(norm(res)/norm(b)))

    if args.fmt=='SELL' and sigma>1:
        x = x[A.unpermute]

    if args.poly_k>0:
        if A_prec.mpkHandle != None:
            x = x[A_prec.unpermute]

    print('relative error of computed solution: %e'%(norm(x-x_ex)/norm(x_ex)))

    hw_string = type
    if type=='cpu':
        hw_string+=' ('+str(numba.get_num_threads())+' cores)'
    print('Hardware: '+hw_string)
    perf_report(type)
    print('Total time for constructing precon: %g seconds.'%(t_pre))
    print('Total time for solving: %g seconds.'%(t_soln))
    print('Total time for CG: %g seconds.'%(t_CG))
