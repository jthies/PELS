
# PELS: Performance Engineering for (sparse) Linear Solvers Demo

(Intro/overview)

## Computational backends and performance reporting

- There are both CPU and GPU implementations, using ``numba`` and ``numba.cuda``, respectively.
  The drivers will detect if there is a GPU available and switch to using it automatically.
- Use the ``-c_kernels`` flag to the driver routine to use C/OpenMP implementations of
  basic operations. The rules to comple the kernels are defined in ``Makefile``. By default,
  GCC is used, if you want to use another compiler, adapt it accordingly.


# Setup

The Python code in this repository requires only standard libraries like numpy, numba and scipy.
To make sure they are available, you can use the following command:

```bash
pip install --user -r requirements.txt
```
## Optional requirements

In order to use an NVidia GPU, you also need cuda.
For using the C backend (in particular in combination with RACE, see below), you need a C compiler
like GCC or Intel.

## Testing the installation

The command ``pytest`` will execute a few simple tests. If a GPU and cuda are available, they will be run on the GPU.
In that case, please ignore warnings about under-utilization of the device.
To specifically test the C backend, use ``env USE_C_KERNELS=1 pytest``. To see all output from the tests (even those that pass), you can use the '-s' flag.

# Example usage

## Example 1: Sparse Matrix Formats on the GPU

On a node with an Nvidia A100 GPU, we compute the smallest eigenvalue (ground state energy) of a spin-chain matrix:

```bash
$ python3 lanczos.py -matfile spinSZ26.mm.gz
```

Which may result in output like this:

```bash
Smallest eigenvalue computed: -4.621451e+01
Hardware: NVIDIA A100 80GB PCIe
--------        -----   --------------- --------------- --------------- ---------------
kernel          calls    bw_meas         bw_roofline     t_meas/call    t_roofline/call
========        =====   =============== =============== =============== ===============
     dot          119      734.8 GB/s       1560 GB/s   0.0002265 s     0.0001067 s 
   axpby          178      986.8 GB/s       1690 GB/s   0.0002529 s     0.0001477 s 
    spmv           59      347.4 GB/s       1690 GB/s   0.005936 s       0.00122 s 
--------        -----   --------------- --------------- --------------- ---------------
   Total                                                   0.4222 s         0.111 s
--------        -----   --------------- --------------- --------------- ---------------

```

This matrix is irregular with about 10M rows and 1-27 nonzeros per row.
A smaller variant, SpinSZ22 is shown below:

SpinSZ22 sparsity pattern | SpinSZ22 compressed view                 
--------------------------|---------------------------------------------
<img src="spinSZ22.png" alt="SpinSZ22 pattern" width="400"/> | <img src="spinSZ22_ELL.png" alt="SpinSZ22 compressed form" width="400"/> |

Better SpMV performance on the GPU can be achieved by using the SELL-C-sigma format:

```bash
$ python3 lanczos.py -matfile spinSZ26.mm.gz -fmt SELL -C 128 -sigma 1024
[...]
Smallest eigenvalue computed: -4.621451e+01
Hardware: NVIDIA A100 80GB PCIe
--------        -----   --------------- --------------- --------------- ---------------
kernel          calls    bw_meas         bw_roofline     t_meas/call    t_roofline/call
========        =====   =============== =============== =============== ===============
     dot          109      808.8 GB/s       1560 GB/s   0.0002057 s     0.0001067 s 
   axpby          163      983.8 GB/s       1690 GB/s   0.0002537 s     0.0001477 s 
    spmv           54       1260 GB/s       1690 GB/s   0.001636 s       0.00122 s 
--------        -----   --------------- --------------- --------------- ---------------
   Total                                                   0.1521 s        0.1016 s
--------        -----   --------------- --------------- --------------- ---------------
```

For the larger and very regular 5-point Laplace matrix with 25M rows and columns, the CSR format works just fine,
and kernel launch latency has less impact, so that the performance of all kernels is quite good:
```bash
$ python lanczos.py -matgen Laplace5000x5000 -tol 1e-3
[...]
Smallest eigenvalue computed: 1.640077e-06
Hardware: NVIDIA A100 80GB PCIe
--------        -----   --------------- --------------- --------------- ---------------
kernel          calls    bw_meas         bw_roofline     t_meas/call    t_roofline/call
========        =====   =============== =============== =============== ===============
     dot         1653       1185 GB/s       1560 GB/s   0.0003375 s     0.0002564 s 
   axpby         2479       1302 GB/s       1690 GB/s   0.0004607 s     0.000355 s 
    spmv          826       1301 GB/s       1690 GB/s   0.001614 s      0.001242 s 
--------        -----   --------------- --------------- --------------- ---------------
   Total                                                    3.033 s          2.33 s
--------        -----   --------------- --------------- --------------- ---------------
```

# Example 2: CG solver on the CPU

The hardware used is a node with 2 32-core Intel Sapphire Rapids CPU's and sub-NUMA clustering, giving four NUMA domains with
16 cores each. We start by setting some environment variables to disable using the GPUs (if any) and using the OpenMP backend 
of Numba:
````bash
export NUMBA_THREADING_LAYER=omp
export CUDA_VISIBLE_DEVICES=""
```

Depending on your system, you may want to manually pin threads to cores, e.g., to run on all 16 threads of one NUMA domain:

```bash
export LAUNCH="likwid-pin -C E:M0:0-15"
export NUMBA_NUM_THREADS=16
```

## Plain CG method 

To solve a linear system with the Laplace operator above:

```bash
${LAUNCH} python3 pcg.py -matgen Laplace5000x5000 -tol 1e-3
```
