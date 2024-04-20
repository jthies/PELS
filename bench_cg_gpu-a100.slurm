#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --account=projects-dhpctestproj
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gpus-per-task=1
#SBATCH --output="bench_cg_gpu-a100.out"
#SBATCH --mem=32GB
#SBATCH -t 00:10:00

module load 2023r1
module load py-numba
module load cuda

export NUMBA_THREADING_LAYER=omp
export OMP_SCHEDULE=static
export NUMBA_NUM_THREADS=8
export OMP_NUM_THREADS=8

SRUN_FLAGS=

CG_FLAGS="-fmt SELL\ -C 128 -sigma 1 -poly_k=0 -maxit 10000 -tol 1.0e-6"
echo "CG_FLAGS='$CG_FLAGS'"
srun $SRUN_FLAGS python3 main_cg.py -matgen 'Laplace5000x5000' ${CG_FLAGS}

CG_FLAGS="-fmt SELL -C 128 -sigma 1 -poly_k=1 -maxit 10000 -tol 1.0e-6"
echo "CG_FLAGS='$CG_FLAGS'"
srun $SRUN_FLAGS python3 main_cg.py -matgen 'Laplace5000x5000' ${CG_FLAGS}

CG_FLAGS="-fmt SELL -C 128 -sigma 1 -poly_k=2 -maxit 10000 -tol 1.0e-6"
echo "CG_FLAGS='$CG_FLAGS'"
srun $SRUN_FLAGS python3 main_cg.py -matgen 'Laplace5000x5000' ${CG_FLAGS}

