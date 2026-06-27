#!/bin/bash
#SBATCH --job-name=nnpu3d_jax_ddp
#SBATCH --output=logs/%j_nnpu3d_jax_ddp.out
#SBATCH --error=logs/%j_nnpu3d_jax_ddp.err
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G

# JAX backend, multi-GPU DOMAIN DECOMPOSITION (single node).
#
# Unlike the torch DDP script (torchrun --nproc_per_node=N, one process per GPU), the JAX path
# runs ONE process that sees all N allocated GPUs: nnpu_jax shards the integration cells across
# jax.devices() with shard_map and SUM-combines the per-device energies with jax.lax.psum (the
# loss is a spatial integral). Keeping the collective inside the jitted energy is what lets the
# L-BFGS line search stay globally consistent. No torchrun, no rendezvous.
#
# Each GPU owns ~n_cell/N cells. The bundled 2662-cell demo is FASTER single-GPU
# (run_nnpu_jax_gpu.slurm); this path is for meshes too large for one device.
# One-time setup on a GPU node: uv sync --extra jax-cuda
cd /scratch/gpfs/nc1514/NN-enrichment3D

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

echo "host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L 2>/dev/null || true

# Single process, all GPUs on the node visible -> jax shards cells across them.
srun .venv/bin/python nnpu_jax.py --device cuda
    # add --smoke for a quick validation run

# --- multi-NODE variant: one process per GPU across nodes, wired by jax.distributed.initialize() ---
# Replace the srun above with the block below and set --nodes / --ntasks-per-node / --gres to match
# (jax.distributed.initialize() auto-detects the SLURM topology; jax.devices() then spans all nodes):
#
# #SBATCH --nodes=2
# #SBATCH --ntasks-per-node=2     # one task per GPU
# #SBATCH --gres=gpu:2
# srun --ntasks-per-node=2 .venv/bin/python nnpu_jax.py --device cuda
