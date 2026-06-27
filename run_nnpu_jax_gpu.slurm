#!/bin/bash
#SBATCH --job-name=nnpu3d_jax_gpu
#SBATCH --output=logs/%j_nnpu3d_jax_gpu.out
#SBATCH --error=logs/%j_nnpu3d_jax_gpu.err
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# JAX backend, single GPU (A100/H100, sm_80+).
# One-time setup on a GPU node: uv sync --extra jax-cuda   (installs jax[cuda12] alongside torch)
cd /scratch/gpfs/nc1514/NN-enrichment3D

export PYTHONUNBUFFERED=1
# Let JAX grow GPU memory on demand rather than pre-grabbing 75% (plays nicer when sharing a node).
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

echo "host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L
srun .venv/bin/python nnpu_jax.py --device cuda
