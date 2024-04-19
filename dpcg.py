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

def axpy(a, x, y):
    return a*x+y

def dot(x, t)
    return np.dot(x,y)

def matvec(A, x):
    return A*x

def precon(M, x):
    return M.solve(x)

def proj(x, v):
    if v is not None:
        s = np.dot(v,x)
        y = x - s*x
    else:
        y = x

def dpcg(A, b, tol, maxit, x0, M, v):

    x_hat  = proj(x0, v)
    r0 = b − Ax0
    r_hat = proj(r0, v)
    y = precon(M, r)hat)
    p = y

    for it in range(maxit):
        w_hat = precon(M, matvec(A,p))
        alpha = dot(r_hat,y)/dot(p, w_hat)
        x_hat += alpha*p
        r_hat -= alpha*what
4:
αi := (p
i ,ŵi )
5:
x̂i+1 := x̂i + αi pi
6:
r̂i+1 := r̂i − αi ŵi
7:
Solve M yi+1 = r̂i+1
(r̂
,yi+1 )
8:
βi := i+1
(r̂i ,yi )
9:
pi+1 := yi+1 + βi pi
10: end for
11: xit := Qb + P T xi+1
