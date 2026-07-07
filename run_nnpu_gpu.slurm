#!/bin/bash
#SBATCH --job-name=nnpu3d_gpu
#SBATCH --output=logs/%j_nnpu3d_gpu.out
#SBATCH --error=logs/%j_nnpu3d_gpu.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --exclusive

# Single-GPU run on Cosmos (SDSC). Nodes are exclusive with 4 MI300A APUs (gfx942,
# SPX mode = 4 visible GPUs); a single-process run uses cuda:0. The installed torch
# is a ROCm 6.3 wheel -- HIP is exposed through the torch.cuda API, so --device cuda
# is correct on AMD. To use all 4 APUs see run_nnpu_ddp.slurm.
#
# Future optimization (not enabled): HSA_XNACK=1 turns on MI300A unified-memory page
# migration, letting CPU and GPU share allocations without explicit copies.

cd /cosmos/vast/scratch/nchen9/NN-enrichment3D

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

echo "host=$(hostname) job=${SLURM_JOB_ID}"
/opt/rocm/bin/rocm-smi --showproductname 2>/dev/null || true
srun .venv/bin/python nnpu_torch.py --device cuda
