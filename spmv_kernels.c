/*******************************************************************************************/
/* This file is part of the training material available at                                 */
/* https://github.com/jthies/PELS                                                          */
/* You may redistribute it and/or modify it under the terms of the BSD-style licence       */
/* included in this software.                                                              */
/*                                                                                         */
/* Contact: Jonas Thies (j.thies@tudelft.nl)                                               */
/*                                                                                         */
/*******************************************************************************************/

#include <sys/types.h>
#include <stdio.h>
#include <math.h>

// Operation that computes y=Ax.
void csr_spmv(size_t N, const double *restrict val, const int *restrict rowPtr, const int *restrict col, const double *restrict x, double *restrict y)
{
#pragma omp parallel for schedule(runtime)
    for(int row=0; row<N; ++row)
    {
        double tmp=0;
        for(int idx=rowPtr[row]; idx<rowPtr[row+1]; ++idx)
        {
            tmp += val[idx]*x[col[idx]];
        }
        y[row]=tmp;
    }
}
