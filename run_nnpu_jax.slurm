#!/bin/bash
#SBATCH --job-name=nnpu3d_jax
#SBATCH --output=logs/%j_nnpu3d_jax.out
#SBATCH --error=logs/%j_nnpu3d_jax.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --exclusive

# JAX backend, CPU, on Cosmos (SDSC). Nodes are exclusive (single "cluster" partition),
# so the job gets all 96 Zen4 cores; the demo mesh runs fine on CPU.
# One-time setup: uv sync --extra jax
cd /cosmos/vast/scratch/nchen9/NN-enrichment3D

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} job=${SLURM_JOB_ID}"
srun .venv/bin/python nnpu_jax.py --device cpu
