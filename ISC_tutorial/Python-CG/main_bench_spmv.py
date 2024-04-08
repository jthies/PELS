from kernels import *
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
    spmv_c(A,x,y)

    print('Numba csr_spmv, matrix='+args.matgen)
    reset_counters()
    t_tot = timeit('spmv(A,x,y)', globals=globals(), number=args.k)
    perf_report('cpu')
    print('Total time for %d SpMVs: %e seconds.'%(args.k, t_tot))

    print('C csr_spmv, matrix='+args.matgen)
    reset_counters()
    t_tot = timeit('spmv_c(A,x,y)', globals=globals(), number=args.k)
    perf_report('cpu')
    print('Total time for %d SpMVs: %e seconds.'%(args.k, t_tot))

    print('')
    print('Run our own loop...')
    reset_counters()
    t0 = perf_counter()
    for i in range(args.k):
        spmv(A,x,y)
    t1 = perf_counter()
    t_tot = t1-t0
    perf_report('cpu')
    print('Total time for %d SpMVs: %e seconds.'%(args.k, t_tot))
