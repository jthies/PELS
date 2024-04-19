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
from kernels import *

def cg_solve(A, b, x0, tol, maxit, verbose=True, x_ex=None):
    '''
    x, tol, iter = cg_solve(A, b, x0, tol, maxit)
    Where A is an spd scipy.sparse.csr_matrix, b and x0 are numpy.array's of size A.shpae[0],
    tol is the convergence tolerance and maxit the maximum number of iterations.
    '''
    x = clone(b)
    r = clone(b)
    p = clone(b)
    q = clone(b)

    if x_ex is not None:
        print('PerfWarning: providing the exact solution x_ex results in additional operations to calculate and print the error norm.')
        err = clone(b)

    tol2 = tol*tol


    axpby(1.0,x0,0.0,x)

    #r = A*x
    if hasattr(A, 'apply'):
        A.apply(x, r)
    else:
        spmv(A, x, r)
    #r = b - r
    axpby(1.0, b, -1.0, r)
    #p = r
    axpby(1.0, r, 0.0, p)

    # rho = <r, r>
    rho = dot(r,r);
    rho_old = 1.0
    if verbose:
        print('%d\t%e'%(0, np.sqrt(rho)))


    for iter in range(maxit+1):

        # check stop criteria
        if rho < tol2:
            break;

        # q = A*p
        if hasattr(A, 'apply'):
            A.apply(p, q)
        else:
            spmv(A, p, q)

        pq = dot(p,q)
        alpha = rho / pq
        #print('%16.12e = %16.12e / %16.12e'%(alpha, rho, pq))
        # x = x+alpha*p
        axpby(alpha, p, 1.0, x)
        # r = r - alpha*q
        axpby(-alpha, q, 1.0, r)

        rho_old = rho
        rho = dot(r, r)

        if verbose:
            if x_ex is not None:
                if hasattr(A, 'unprec_sol'):
                    A.unprec_sol(x, err)
                else:
                    axpby(1.0, x, 0.0, err)
                axpby(-1.0, x_ex, 1.0, err)
                err_norm = np.sqrt(dot(err, err))
                print('%d\t%e\t%e'%(iter, np.sqrt(rho), err_norm))
            else:
                print('%d\t%e'%(iter, np.sqrt(rho)))

        beta = rho / rho_old
        # p = r+beta*p
        axpby (1.0, r, beta, p)

    iter_count = iter
    final_residual = np.sqrt(rho)

    return x, final_residual, iter_count



