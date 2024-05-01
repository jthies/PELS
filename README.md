
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
pip install -r requirements.txt
```
## Optional requirements

In order to use an NVidia GPU, you also need cuda.
For using the C backend (in particular in combination with RACE, see below), you need a C compiler
like GCC or Intel.

## Testing the installation

The command ``pytest`` will execute a few simple tests. If a GPU and cuda are available, they will be run on the GPU.
In that case, please ignore warnings about under-utilization of the device.
To specifically test the C backend, use ``pytest -c_kernels``.

# Example usage
