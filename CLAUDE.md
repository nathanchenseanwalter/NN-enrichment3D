# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Neural-network-enriched Reproducing Kernel Particle Method (NN-RKPM / NN-PU) that solves 3D bimaterial linear elasticity by the **Deep Energy Method** with SCNI (Stabilized Conforming Nodal Integration). It is a PyTorch port of `NNPU_Modified2026_SCNI3D.ipynb` (originally TensorFlow/Keras + TFP on Colab) and runs on Princeton HPC via SLURM.

`README.md` is the authoritative reference — read it for the physics, the `.dat` I/O column layouts, and the full distributed-training rationale. This file captures what's needed to work in the code without re-deriving it.

## Commands

Environment is managed by **uv** (Python 3.13, PyTorch with bundled CUDA):

```bash
uv sync                                                        # create .venv, install deps
```

Run and validate. There is **no unit-test suite, linter, or formatter** configured (no `[tool.*]` in `pyproject.toml`) — the smoke run is the fast end-to-end check:

```bash
uv run python nnpu_torch.py --smoke --out-dir /tmp/nnpu_smoke  # ~1 min, 200/200/50 iters
uv run python nnpu_torch.py                                    # full run, auto-selects device
uv run python nnpu_torch.py --device cpu                       # force CPU
uv run python plot_results.py                                  # plot a finished run (no retrain)
```

Cluster (SLURM):

```bash
sbatch run_nnpu.slurm        # CPU, general partition — schedules fastest
sbatch run_nnpu_gpu.slurm    # single GPU (A100)
sbatch run_nnpu_ddp.slurm    # multi-GPU domain decomposition
```

**JAX backend** (`nnpu_jax.py`, a function-for-function port; same flags + identical output layout). Install with `uv sync --extra jax` (CPU) or `--extra jax-cuda` (GPU); SLURM scripts are `run_nnpu_jax{,_gpu,_ddp}.slurm`. Compare backends with `uv run python compare_backends.py`.

```bash
uv run --extra jax python nnpu_jax.py --smoke --device cpu --out-dir /tmp/jx_smoke  # ~30 s
```

**Distributed correctness check** — the de-facto regression test; distributed output must match single-process:

```bash
python   nnpu_torch.py --smoke --device cpu --out-dir /tmp/r1
torchrun --standalone --nproc_per_node=2 nnpu_torch.py --smoke --device cpu --out-dir /tmp/r2
# Adam stages match to ~1e-15; L-BFGS to ~1e-6 (float reduction order). Diff results_nnrk_*.txt.
```

## Architecture

`main.py` and the SLURM scripts both call `nnpu_torch.main()` (`nnpu_torch.py` has its own `__main__` guard, so running it directly works). **`nnpu_torch.py` is the source of truth** — both notebooks (`*_torch.ipynb`) and `main.py` delegate to it; don't fork logic back into a notebook. `nnpu_jax.py` is a parallel JAX port that mirrors it function-for-function and is validated against it (the RNG-independent Adam-RK trajectory matches to ~`1e-16`); when changing the physics, change `nnpu_torch.py` first, then port. `partition.py`'s `build_local_triplets` is the shared NumPy slicing core used by both backends.

```
main.py / *.slurm → nnpu_torch.main()
  ├─ Distributed()          (distributed.py)  rank/device, SUM all-reduce
  ├─ NNRK(...)              loads .dat mesh + sparse operators (float64)
  │    └─ partition.*       (distributed only) slice cells/operators per rank
  ├─ train_adam_rk()        Stage 1: Adam on RK coeffs only
  ├─ train_adam_nnrk()      Stage 2: Adam on RK + NN coeffs + MLP weights
  └─ train_lbfgs()          Stage 3: L-BFGS on all params
plot_results.py — standalone post-processing of results_nnrk_LBFGS.txt
```

Concepts that span files (the parts that aren't obvious from any single file):

- **Loss = total potential energy** (strain-energy integral over cells), not a supervised target. Forward path inside `NNRK`: `rk_approx + nn_approx → smoothed_strain` (sparse `P1/P2/P3`) `→ stress` (per-cell bimaterial μ, λ) `→ energy_nnrk`.
- **Three-stage training** warms up the RK background field, then adds the MLP enrichment, then refines everything with L-BFGS. The enrichment MLP (`EnrichmentMLP`, coords `[3] → [40,40,40,40,5]`) is spatially gated to a y-band around y = 0.
- **`float64` everywhere**, matching the original `tf.keras.backend.set_floatx("float64")`. Keep any new tensors float64 or results diverge.
- **Distributed = domain decomposition, NOT data-parallel DDP.** The model (tiny MLP + global nodal DOFs) is *replicated* on every rank; the integration cells are partitioned (`partition.py`); gradients combine with a **SUM all-reduce** (`distributed.py`) because the loss is an integral, not a mean — so no `DistributedSampler`, no mean-reduce, no halo exchange. Identical seeds keep ranks in sync, and the L-BFGS closure uses the *global* energy so every rank takes the same line-search step. The bundled 2662-cell demo is **faster single-process**; distributed only pays off at millions of integration points. The JAX backend keeps these *semantics* but a different *mechanism*: one process shards cells across devices with `jax.shard_map` + a SUM `jax.lax.psum` inside the jitted energy (so the L-BFGS line search stays globally consistent); per-device operators are zero-padded to a common shape (`build_global_energy` in `nnpu_jax.py`).

## Data

`Input_3Dbimat_Node396_Cell2662/` holds the mesh (396 nodes, 2662 integration cells). Dense `.dat` files are CSV; operator `.dat` files are sparse **COO triplets** (`row col value` per line) loaded into `torch.sparse_coo_tensor`. Results default to `<input-dir>/Results_40NR_4hidden_5bases/` (override with `--out-dir`). See README "Outputs" for the column layout of `results_nnrk_*.txt`.

## Cluster gotchas (Princeton Stellar/Della)

- `--partition=all` reroutes to the `serial` queue, and the GPU queue is deep — **prefer CPU for small jobs**. The demo mesh runs fine on CPU and CPU schedules immediately.
- GPU wheels are version-sensitive: a `torch …+cu130` build targets A100/H100 (sm_80+); older cards such as V100 (sm_70) need a cu12x wheel. See README "Multi-GPU".
