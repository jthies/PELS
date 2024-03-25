#!/bin/bash
#SBATCH --partition=compute-p2
#SBATCH --account=projects-dhpctestproj
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --exclusive
#SBATCH --output="bench_cg_cpu-np64.out"
#SBATCH --mem=0
#SBATCH -t 00:10:00

module load 2023r1
module load py-numba

export NUMBA_THREADING_LAYER=omp
export OMP_SCHEDULE=static
export NUMBA_NUM_THREADS=64
export OMP_NUM_THREADS=64

SRUN_FLAGS=--cpu-bind=ldoms

srun $SRUN_FLAGS python3 main_cg.py -matgen 'Laplace500x500'
srun $SRUN_FLAGS python3 main_cg.py -matgen 'Laplace1000x1000'
srun $SRUN_FLAGS python3 main_cg.py -matgen 'Laplace2000x2000'
srun $SRUN_FLAGS python3 main_cg.py -matgen 'Laplace4000x4000'
srun $SRUN_FLAGS python3 main_cg.py -matgen 'Laplace8000x8000'
srun $SRUN_FLAGS python3 main_cg.py -matgen 'Laplace16000x16000'
