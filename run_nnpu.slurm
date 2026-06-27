#!/bin/bash
#SBATCH --job-name=nnpu3d
#SBATCH --output=logs/%j_nnpu3d.out
#SBATCH --error=logs/%j_nnpu3d.err
#SBATCH --partition=all
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G

# NN-RKPM 3D bimaterial elasticity (PyTorch port of NNPU_Modified2026_SCNI3D.ipynb).
#
# Runs on CPU on the general "all" partition: the problem is small (n_cell=2662,
# n_node=396, tiny MLP) and the GPU queue on this cluster is very deep, so a
# multi-core CPU node schedules immediately and finishes in well under the limit.
# For a GPU run instead, see run_nnpu_gpu.slurm.

cd /scratch/gpfs/nc1514/NN-enrichment3D

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} job=${SLURM_JOB_ID}"
srun .venv/bin/python nnpu_torch.py --device cpu
