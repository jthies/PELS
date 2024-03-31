from time import perf_counter
import numpy as np
import scipy
import numba

import sellcs

import kernels_cpu as cpu
# for benchmarking numpy/scipy implementations
#import kernels_numpy as cpu

try:
    from numba import cuda
    from numba.cuda import is_cuda_array
    import kernels_gpu as gpu
except:
    print('Could not load cuda module and/or kernels')
    gpu = cpu
    cuda = None

def available_gpus():
    if cuda is None:
        return 0
    if cuda.is_available()==False:
        return 0
    return len(cuda.gpus)

def compile_all():
    n=10
    x=np.ones(n,dtype='float64')
    y=np.ones(n,dtype='float64')
    a=numba.float64(1.0)
    b=numba.float64(1.0)
    A1=scipy.sparse.csr_matrix(scipy.sparse.rand(n,n,0.6))
    A2=sellcs.sellcs_matrix(A1, C=1, sigma=1)

    # compile CPU kernels:
    init(x,a)
    z = clone(x)
    s=dot(x,y)
    axpby(a,x,b,y)
    spmv(A1,x,y)
    spmv(A2,x,y)
    # compile GPU kernels:
    if available_gpus()>0:
        x = to_device(x)
        y = to_device(x)
        A1 = to_device(A1)
        A2 = to_device(A2)
        init(x,a)
        z = clone(x)
        s=dot(x,y)
        axpby(a,x,b,y)
        spmv(A1,x,y)
        spmv(A2,x,y)

def memory_benchmarks(type):
    if type=='cpu':
        return cpu.memory_benchmarks()
    elif type=='gpu':
        return gpu.memory_benchmarks()
    else:
        raise('type should be "cpu" or "gpu"')

# total number of calls
# note: we do not measure the first call because that will involve compilation time
calls = {'spmv': -1, 'axpby': -1, 'dot': -1, 'init':-1}
# total elapsed time in seconds
time = {'spmv': 0.0, 'axpby': 0.0, 'dot': 0.0, 'init':0.0}
# total loaded data in GB
load = {'spmv': 0.0, 'axpby': 0.0, 'dot': 0.0, 'init':0.0}
# total stored data in GB
store = {'spmv': 0.0, 'axpby': 0.0, 'dot': 0.0, 'init':0.0}
# total floating point operations [GFlop]
flop = {'spmv': 0.0, 'axpby': 0.0, 'dot': 0.0, 'init':0.0}

# which benchmark to use for predicting memory bandwidth achievable by an operation.
# Benchmark values are currently hard-coded into kernels_cpu.py and kernels_gpu.py for Sapphire Rapids and A100, resp.
bench_map = {'spmv': 'triad', 'axpby': 'triad', 'dot': 'load', 'init': 'store'}

def to_device(A):
    if available_gpus()>0:
        return gpu.to_device(A)
    else:
        return A

def to_host(A):
    if cuda and is_cuda_array(A):
        return A.copy_to_host()
    elif type(A)==scipy.sparse.csr_matrix or type(A)==sellcs.sellcs_matrix:
        if available_gpus()>0:
            A.indptr = A.cu_indptr.copy_to_host()
            A.data = A.cu_data.copy_to_host()
            A.indices = A.cu_indices.copy_to_host()
    return A

def spmv(A, x, y):
    t0 = perf_counter()
    if cuda and is_cuda_array(x):
        if not hasattr(A, 'cu_data'):
            print('PerfWarning: copying matrix data to device in spmv call. Manually call kernels.to_device(A) to avoid this.')
            A = to_device(A)
        run_on = gpu
        data = A.cu_data
        indptr = A.cu_indptr
        indices = A.cu_indices
    else:
        run_on = cpu
        data = A.data
        indptr = A.indptr
        indices = A.indices
    if type(A)==scipy.sparse.csr_matrix:
            run_on.csr_spmv(data, indptr, indices, x, y)
    elif type(A)==sellcs.sellcs_matrix:
        run_on.sell_spmv(data, indptr, indices, A.C, x, y)
    else:
        raise TypeError('spmv wrapper only implemented for scipy.sparse.csr_matrix or sellcs.sellcs_matrix')
    t1 = perf_counter()
    time['spmv']  += t1-t0
    calls['spmv'] += 1
    if calls['spmv']>0:
        load['spmv']  += 12*A.nnz+8*(A.shape[0]+A.shape[1])
        store['spmv'] += 8*A.shape[0]
        flop['spmv'] += 2*A.nnz

def diag_spmv(A, x, y):
    if cuda and is_cuda_array(x):
        gpu.vscale(A.cu_data, x, y)
    else:
        print(type(A))
        print(type(A.data))
        print(A.data.shape)
        cpu.vscale(A.data.reshape(x.size), x, y)

def clone(v):
    w = None
    if cuda and is_cuda_array(v):
        w = cuda.device_array(shape=v.shape,dtype=v.dtype)
    else:
        w = np.empty_like(v)
        # first-touch initialization
        cpu.init(w,0.0)
    return w

def copy(X):
    '''
    Copy a vector or matrix (csr_matrix or sellcs_matrix)
    that may live on a GPU, and assure first-touch initialization
    on the CPU.
    '''
    if cuda and is_cuda_array(X):
        return v.copy()
    elif type(X) == np.ndarray:
        return cpu.copy_vector(X)
    elif type(X) == scipy.sparse.csr_matrix or type(X) == sellcs.sellcs_matrix:
        data, indices, indptr = cpu.copy_csr_arrays(X.data, X.indptr, X.indices)
        if type(X) == scipy.sparse.csr_matrix:
            A = scipy.sparse.csr_matrix((data, indices, indptr), shape=X.shape)
        elif type(A) == sellcs.sellcs_matrix:
            A = sellcs.sellcs_matrix((data, indices, indptr), shape=X.shape, C=X.C, sigma=X.sigma)
        if hasattr(X, 'cu_data'):
            A.cu_data = X.cu_data.copy()
        if hasattr(X, 'cu_indices'):
            A.cu_indices = X.cu_indices.copy()
        if hasattr(X, 'cu_indptr'):
            A.cu_indptr = X.cu_indptr.copy()
        return A

def init(v, val):
    t0 = perf_counter()
    if cuda and is_cuda_array(v):
        gpu.init(v,val)
    else:
        cpu.init(v,val)
    t1 = perf_counter()
    calls['init'] += 1
    if calls['init']>0:
        time['init']  += t1-t0
        store['init'] += 8*v.size

def axpby(a,x,b,y):
    t0 = perf_counter()
    if cuda and is_cuda_array(y):
        gpu.axpby(a,x,b,y)
    else:
        cpu.axpby(a,x,b,y)
    t1 = perf_counter()
    time['axpby']  += t1-t0
    calls['axpby'] += 1
    if calls['axpby']==0:
        return
    load['axpby']  += 16*x.size
    store['axpby'] += 8*x.size
    flop['axpby'] += 2*x.size

def dot(x,y):
    t0 = perf_counter()
    if cuda and is_cuda_array(y):
        s = gpu.dot(x,y)
    else:
        s = cpu.dot(x,y)
    t1 = perf_counter()
    time['dot']  += t1-t0
    calls['dot'] += 1
    load['dot']  += 16*x.size
    flop['spmv'] += 2*x.size
    return s

def perf_report(type):
    bench = memory_benchmarks(type)

    t_tot  = 0
    t_mod  = 0

    print('kernel\tcalls\tbw_meas\tbw_expected\tt_meas/call\tt_expected/call\n')
    for kern in ('dot', 'axpby', 'spmv'):
        if calls[kern]>0:
            print('%s\t%d\t%g GB/s\t%g GB/s\t%g s \t%g s '%
                (kern, calls[kern], (load[kern]+store[kern])*1e-9/time[kern], bench[bench_map[kern]],
                time[kern]/calls[kern], (load[kern]+store[kern])*1e-9/bench[bench_map[kern]]/calls[kern]))
            t_tot += time[kern]
            t_mod += (load[kern]+store[kern])*1e-9/bench[bench_map[kern]]
    print('Total\t \t \t \t \t %g \t %g'%(t_tot, t_mod))

def perf_report_plot(type, filename=None):
    from matplotlib import pyplot as plt
    bench = memory_benchmarks(type)
    x = [1, 2, 3]
    xlabels = ['dot', 'axpby', 'spmv']
    plt.plot(x, [ bench['load'], bench['triad'], bench['triad']], 's')
    plt.plot(x, [ load['dot']/time['dot']*1e-9, (load['axpby']+store['axpby'])/time['axpby']*1e-9, (load['spmv']+store['spmv'])/time['spmv']*1e-9], 'x')
    plt.xticks([1,2,3],xlabels)
    plt.ylabel('GB/s')
    if filename==None:
        plt.show()
    else:
        plt.savefig(filename)