#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --account=projects-dhpctestproj
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus-per-task=1
#SBATCH --output="bench_kernels_gpu.out"
#SBATCH --mem=64GB
#SBATCH -t 00:15:00

module load 2023r1
module load py-numba
module load cuda

SRUN_FLAGS=--cpu-bind=ldoms

N=100000000

srun $SRUN_FLAGS ython3 main_bench.py -n $N -k 50 -f init -t gpu
srun $SRUN_FLAGS ython3 main_bench.py -n $N -k 50 -f axpby -t gpu
srun $SRUN_FLAGS ython3 main_bench.py -n $N -k 50 -f dot -t gpu

