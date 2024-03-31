import numpy as np
from numpy.linalg import norm
from scipy.sparse import *
from scipy.io import mmread
from pcg import cg_solve
from kernels import *
from poly_op import *
from sellcs import sellcs_matrix

from matrix_generator import create_matrix

import numba
import argparse

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
    parser.add_argument('-poly_k', type=int, default=0,
                    help='Use a degree-k polynomial preconditioner based on the Neumann series.')
    return parser

if __name__ == '__main__':

    parser = get_argparser()
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

    if args.fmt=='SELL':
        C=args.C
        sigma=args.sigma
        A = sellcs_matrix(A, C, sigma)
        b = b[A.permute]

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
    if args.poly_k>0:
        A_prec = poly_op(A, args.poly_k)
        b_prec = copy(b)
        A_prec.prec_rhs(b, b_prec)

    t0 = perf_counter()
    x_prec, relres, iter = cg_solve(A_prec,b_prec,x0,tol,maxit)
    t1 = perf_counter()

    if args.poly_k>0:
        x = copy(x)
        A_prec.prec_rhs(x_prec, x)
    else:
        x = x_prec

    t_CG = t1-t0

    x = to_host(x)

    print('number of CG iterations: %d'%(iter))
    res = np.empty_like(x)
    spmv(A,x,res)
    res=b-res
    print('relative residual of computed solution: %e'%(norm(res)/norm(b)))

    if args.fmt=='SELL' and sigma>1:
        x = x[A.unpermute]

    print('relative error of computed solution: %e'%(norm(x-x_ex)/norm(x_ex)))

    hw_string = type
    if type=='cpu':
        hw_string+=' ('+str(numba.get_num_threads())+' cores)'
    print('Hardware: '+hw_string)
    perf_report(type)
    print('Total time for CG: %g seconds.'%(t_CG))
