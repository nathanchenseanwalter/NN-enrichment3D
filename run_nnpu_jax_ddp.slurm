#!/bin/bash
#SBATCH --job-name=nnpu3d_jax_ddp
#SBATCH --output=logs/%j_nnpu3d_jax_ddp.out
#SBATCH --error=logs/%j_nnpu3d_jax_ddp.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --exclusive

# *** NOT YET SUPPORTED ON COSMOS -- JAX-on-ROCm is a FUTURE TASK ***
#
# JAX backend, multi-GPU DOMAIN DECOMPOSITION (single node). Unlike the torch DDP script
# (torchrun, one process per GPU), the JAX path runs ONE process that sees all GPUs:
# nnpu_jax shards the integration cells across jax.devices() with shard_map and
# SUM-combines per-device energies with jax.lax.psum. On Cosmos that would be the node's
# 4 MI300A APUs -- but it needs AMD's jax-rocm plugin wheels (https://github.com/ROCm/jax)
# first; see run_nnpu_jax_gpu.slurm. Use the torch DDP script (run_nnpu_ddp.slurm), which
# fully supports ROCm, in the meantime. This script fails fast rather than falling back to CPU.
cd /cosmos/vast/scratch/nchen9/NN-enrichment3D

export PYTHONUNBUFFERED=1

echo "host=$(hostname) job=${SLURM_JOB_ID}"
echo "ERROR: JAX GPU on Cosmos (ROCm) is not set up yet -- see header comment." >&2
exit 1

# Once jax-rocm is installed, the run itself is unchanged (single process, all 4 APUs
# visible -> jax shards cells across them):
# srun .venv/bin/python nnpu_jax.py --device cuda
#     # add --smoke for a quick validation run
