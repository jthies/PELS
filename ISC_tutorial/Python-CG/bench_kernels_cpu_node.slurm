#!/bin/bash
#SBATCH --partition=compute-p2
#SBATCH --account=projects-dhpctestproj
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --exclusive
#SBATCH --output="bench_kernels_cpu-np64.out"
#SBATCH --mem=0
#SBATCH -t 00:10:00

module load 2023r1
module load py-numba

export NUMBA_THREADING_LAYER=omp
export OMP_SCHEDULE=static
export NUMBA_NUM_THREADS=64
export OMP_NUM_THREADS=64

SRUN_FLAGS=--cpu-bind=ldoms

N=100000000

srun $SRUN_FLAGS ython3 main_bench.py -n $N -k 50 -f init -t cpu
srun $SRUN_FLAGS ython3 main_bench.py -n $N -k 50 -f axpby -t cpu
srun $SRUN_FLAGS ython3 main_bench.py -n $N -k 50 -f dot -t cpu

