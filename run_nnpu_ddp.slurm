#!/bin/bash
#SBATCH --job-name=nnpu3d_ddp
#SBATCH --output=logs/%j_nnpu3d_ddp.out
#SBATCH --error=logs/%j_nnpu3d_ddp.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Multi-GPU (distributed) NN-RKPM training.
#
# This is domain-decomposition data-parallelism, not sample-parallel DDP: the integration
# cells are split across ranks and gradients are SUM all-reduced (see nnpu_torch.py /
# distributed.py / partition.py). Each rank == one GPU. Keep --gres=gpu:N == --nproc_per_node=N.
#
# Worth it only for large meshes (millions of integration points); for the bundled 2662-cell
# demo a single process is faster. Use it to validate scaling and as the template to grow on.

cd /scratch/gpfs/nc1514/NN-enrichment3D

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1
NGPU=$(echo "$SLURM_GPUS_ON_NODE" | grep -oE '[0-9]+' | head -1); NGPU=${NGPU:-2}

echo "host=$(hostname) gpus=${NGPU} job=${SLURM_JOB_ID}"
nvidia-smi -L 2>/dev/null || true

# --- single node, NGPU GPUs (NCCL chosen automatically on GPU) ---
srun .venv/bin/torchrun --standalone --nproc_per_node="${NGPU}" \
    nnpu_torch.py --device cuda
    # add --smoke for a quick validation run

# --- multi-node variant: replace the srun above with the block below ---
# nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
# head_ip=$(srun --nodes=1 --ntasks=1 -w "${nodes[0]}" hostname --ip-address)
# srun .venv/bin/torchrun \
#     --nnodes="$SLURM_NNODES" --nproc_per_node="${NGPU}" \
#     --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d --rdzv_endpoint="${head_ip}:29500" \
#     nnpu_torch.py --device cuda

# --- SDSC Expanse notes ---
#   #SBATCH --partition=gpu-shared      (or gpu);  #SBATCH --account=<alloc>
#   module load gpu; module load slurm
#   Expanse GPUs are V100 (sm_70). The repo's torch 2.12+cu130 wheel DROPPED sm_70, so build
#   the venv there against a cu12x wheel, e.g.:
#     uv pip install torch --index-url https://download.pytorch.org/whl/cu124
# --- Princeton Della notes ---
#   #SBATCH --account=kolemen ;  A100/H200 (sm_80/sm_90) work with the default cu130 wheel.
