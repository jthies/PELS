#/*******************************************************************************************/
#/* This file is part of the training material available at                                 */
#/* https://github.com/jthies/PELS                                                          */
#/* You may redistribute it and/or modify it under the terms of the BSD-style licence       */
#/* included in this software.                                                              */
#/*                                                                                         */
#/* Contact: Jonas Thies (j.thies@tudelft.nl)                                               */
#/*                                                                                         */
#/*******************************************************************************************/

import argparse
from timeit import timeit
import numpy as np
from numba import get_num_threads
from kernels import *

parser = argparse.ArgumentParser(description='Run a selected benchmark from the kernels module.')
parser.add_argument('-t', type=str, default='cpu',
                    help='Select "cpu" or "gpu"')
parser.add_argument('-n', type=int, default=int(2**20),
                    help='number of elements in x, y and z.')
parser.add_argument('-k', type=int, default=50,
                    help='number of runs')
parser.add_argument('-f', type=str, default='axpby',
                    help='benchmark function: [init, axpby, dot]')

args = parser.parse_args()

compile_all()

N = args.n
ntimes = args.k
foo = args.f
bar = {'axpby': '(a,x,b,y)', 'dot': '(x,y)', 'init': '(x,a)'}[foo]
type= args.t

a = 42.0
b = 19.3
x = np.empty(N)
init(x,0.0)
y = np.empty(N)
init(y,0.0)

if type=='gpu':
    if available_gpus()<1:
        print('Warning: you gave -t "gpu" but no device was found.')
    x = to_device(x)
    y = to_device(y)

t = timeit(stmt='zz='+foo+bar, globals=globals(), number=ntimes)
perf_report(type)

