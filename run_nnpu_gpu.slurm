#!/bin/bash
#SBATCH --job-name=nnpu3d_gpu
#SBATCH --output=logs/%j_nnpu3d_gpu.out
#SBATCH --error=logs/%j_nnpu3d_gpu.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# GPU variant. The installed torch is built for CUDA 13 (cu130), which supports
# the A100s on this cluster's "gpu" partition (the older head-node V100S is not
# targeted by cu130). Note the gpu partition is typically heavily queued here.

cd /scratch/gpfs/nc1514/NN-enrichment3D

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

echo "host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L
srun .venv/bin/python nnpu_torch.py --device cuda
