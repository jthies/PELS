import numpy as np
import scipy
import scipy.sparse.linalg as spla

from kernels_cpu import *
from pcg import *

def create_matrix(nx,ny, density_ratio):

    N=nx*ny
    ex=numpy.ones([nx])
    ey=numpy.ones([ny])
    Ix=scipy.sparse.eye(nx)
    Iy=scipy.sparse.eye(ny)
    Dx=scipy.sparse.spdiags([-ex,2*ex,-ex],[-1,0,1],nx,nx)
    Dy=scipy.sparse.spdiags([-ey,2*ey,-ey],[-1,0,1],ny,ny)
    A=scipy.sparse.kron(Dx,Iy) + scipy.sparse.kron(Ix,Dy)
    return A


def main(nx, density_ratio):

    ny = nx
    N = nx*ny
    print('\nnx=%d, ny=%d\n'%(nx,ny))
    print('case    \titer  \terror\n')

    A = lap2_matrix(nx,ny)
    xex=numpy.random.randn(N)
    rhs=A*xex

    x0=numpy.zeros(N)
    tol=1e-10
    maxit=500

    report = lambda label,x,iter: print('%10s\t%d\t%5.3e'%(label,iter,norm(x-xex)))


    def count_iter(xj):
        nonlocal it
        it += 1

    it = 0
    x, flag = spla.cg(A,rhs,x0,tol,maxit,callback=count_iter)
    report('Ax=b',x,it)

    it = 0
    sx=4
    sy=4
    M_bj = BlockJacobi(A, sx*sy)
    x, flag = spla.cg(A,rhs,x0,tol,maxit,M=M_bj,callback=count_iter)
    report('M\\Ax=M\\b',x,it)


    # deflation
    it = 0
    A_d = DeflatedOperator(A, sx*sy)
    xtil = A_d.applyQ(rhs)
    btil = rhs - A @ xtil
    xbar, flag = spla.cg(A_d, btil, x0, tol, maxit, callback=count_iter)
    x=A_d.proj(xbar) + xtil
    report('PAx=Pb',x,it);

#  % deflated and preconditioned method
    it = 0
    xbar, flag = spla.cg(A_d, btil, x0, tol, maxit, M=M_bj, callback=count_iter)
    x=A_d.proj(xbar) + xtil
    report('PAx=Pb',x,it);

if __name__ == '__main__':

    for nx in [16, 32, 64]:
        main(nx, False)
