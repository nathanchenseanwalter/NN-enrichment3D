#!/bin/bash
#SBATCH --job-name=nnpu3d_ddp
#SBATCH --output=logs/%j_nnpu3d_ddp.out
#SBATCH --error=logs/%j_nnpu3d_ddp.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --exclusive

# Multi-GPU (distributed) NN-RKPM training on Cosmos (SDSC).
#
# This is domain-decomposition data-parallelism, not sample-parallel DDP: the integration
# cells are split across ranks and gradients are SUM all-reduced (see nnpu_torch.py /
# distributed.py / partition.py). Each rank == one MI300A APU. Nodes are exclusive with
# 4 APUs (SPX mode = 4 visible GPUs), so --nproc_per_node=4. The "nccl" backend string
# maps to RCCL on ROCm -- no code change.
#
# Worth it only for large meshes (millions of integration points); for the bundled
# 2662-cell demo a single process is faster. Use it to validate scaling and as the
# template to grow on.

cd /cosmos/vast/scratch/nchen9/NN-enrichment3D

export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-96} / 4 ))
export PYTHONUNBUFFERED=1
NGPU=4

echo "host=$(hostname) gpus=${NGPU} job=${SLURM_JOB_ID}"
/opt/rocm/bin/rocm-smi --showproductname 2>/dev/null || true

# --- single node, 4 APUs (RCCL chosen automatically via the "nccl" backend) ---
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

# --- Cosmos notes ---
#   * TPX-mode nodes (#SBATCH --res=tpx) split each APU into 3 logical GPUs, giving
#     12 devices/node -- bump NGPU/--nproc_per_node to 12 if you use that reservation.
#   * HSA_XNACK=1 (unified-memory page migration) is a future optimization to explore;
#     the MI300A shares physical memory between CPU and GPU.
