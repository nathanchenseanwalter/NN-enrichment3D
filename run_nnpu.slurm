#!/bin/bash
#SBATCH --job-name=nnpu3d
#SBATCH --output=logs/%j_nnpu3d.out
#SBATCH --error=logs/%j_nnpu3d.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --exclusive

# NN-RKPM 3D bimaterial elasticity (PyTorch port of NNPU_Modified2026_SCNI3D.ipynb).
#
# CPU run on Cosmos (SDSC). There is a single "cluster" partition and compute nodes
# are allocated exclusively, so every job gets a whole node: 4x MI300A APUs =
# 96 Zen4 cores. The bundled demo (n_cell=2662, n_node=396, tiny MLP) runs fine on
# CPU; for a GPU run see run_nnpu_gpu.slurm.

cd /cosmos/vast/scratch/nchen9/NN-enrichment3D

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} job=${SLURM_JOB_ID}"
srun .venv/bin/python nnpu_torch.py --device cpu
