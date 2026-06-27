#!/bin/bash
#SBATCH --job-name=nnpu3d_jax
#SBATCH --output=logs/%j_nnpu3d_jax.out
#SBATCH --error=logs/%j_nnpu3d_jax.err
#SBATCH --partition=all
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G

# JAX backend, CPU (general partition -- schedules fastest; the demo mesh runs fine on CPU).
# One-time setup: uv sync --extra jax
cd /scratch/gpfs/nc1514/NN-enrichment3D

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} job=${SLURM_JOB_ID}"
srun .venv/bin/python nnpu_jax.py --device cpu
