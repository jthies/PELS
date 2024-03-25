/*******************************************************************************************/
/* This file is part of the DHPC training material available at                            */
/* https://gitlab.tudelft.nl/dhpc/training                                                 */
/* You may redistribute it and/or modify it under the terms of the BSD-style licence       */
/* included in this software.                                                              */
/*                                                                                         */
/* Contact: Jonas Thies (j.thies@tudelft.nl)                                               */
/*                                                                                         */
/*******************************************************************************************/
#include <sys/types.h>
#include <stdio.h>
#include <math.h>

// Basic C implemenattions of axpy and dot using OpenMP.

// Operation that loads two and writes one vector (z=a*x+y).
void axpy(size_t N, double a, const double *restrict X, const double *restrict Y, double *restrict Z)
{
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < N; i++)
  {
    Z[i]=a*X[i]+Y[i];
  }
}

double dot(size_t N, const double* X, const double* Y)
{
  double dot=0.0;
#pragma omp parallel for schedule(static) reduction(+:dot)
  for (size_t i=0; i<N; i++) dot+=X[i]*Y[i];
  return dot;
}

void foo(size_t N, double a, const double *restrict X, const double *restrict Y, double *restrict Z)
{
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < N; i++)
  {
    Z[i] = log(atan(a*X[i]+Y[i]))/log(sqrt(X[i]*X[i] + Y[i]*Y[i]));
  }
}
