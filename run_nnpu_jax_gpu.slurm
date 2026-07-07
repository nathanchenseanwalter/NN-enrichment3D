#!/bin/bash
#SBATCH --job-name=nnpu3d_jax_gpu
#SBATCH --output=logs/%j_nnpu3d_jax_gpu.out
#SBATCH --error=logs/%j_nnpu3d_jax_gpu.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --exclusive

# *** NOT YET SUPPORTED ON COSMOS -- JAX-on-ROCm is a FUTURE TASK ***
#
# JAX on the MI300A GPUs needs AMD's jax-rocm plugin wheels (https://github.com/ROCm/jax),
# which are versioned separately from upstream jax and not yet wired into this repo's
# uv setup (the old jax-cuda extra was removed in the Cosmos port). Until that is done,
# use the CPU JAX script (run_nnpu_jax.slurm) or the torch GPU scripts, which fully
# support ROCm. This script fails fast rather than silently falling back to CPU.
cd /cosmos/vast/scratch/nchen9/NN-enrichment3D

export PYTHONUNBUFFERED=1

echo "host=$(hostname) job=${SLURM_JOB_ID}"
echo "ERROR: JAX GPU on Cosmos (ROCm) is not set up yet -- see header comment." >&2
exit 1

# Once jax-rocm is installed, the run itself is unchanged:
# srun .venv/bin/python nnpu_jax.py --device cuda
