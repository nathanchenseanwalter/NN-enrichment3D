# NN-enrichment 3D — NN-RKPM / NN-PU for 3D bimaterial elasticity

PyTorch port of `NNPU_Modified2026_SCNI3D.ipynb` (originally TensorFlow/Keras + TFP on Colab).

A **neural-network-enriched Reproducing Kernel Particle Method** solved by the **Deep Energy
Method** with **SCNI** (Stabilized Conforming Nodal Integration):

- An RK background displacement field (coefficients `d*_int`).
- A neural-network enrichment — an MLP of nodal coordinates `[3] → [40,40,40,40,5]` (ELU) produces
  `n_NC = 5` basis functions ζ, combined via enrichment coefficients `d*_NN` and gated by a spatial
  activation mask (a y-band of width 0.1 around y = 0).
- Smoothed strains from precomputed sparse derivative operators `P1/P2/P3`; isotropic linear-elastic
  bimaterial stresses (matrix `E=10400`, inclusion `E=52000` MPa).
- Loss = total potential energy = `scaling · Σ(strain_energy_density · WT)`.
- Boundary condition: top face (`y = y_max`) pulled by `gy_ebc = 0.01`, a uniaxial tension test.
- Three optimization stages: **Adam (RK only)** → **Adam (RK + NN + network)** → **L-BFGS (all)**.

## Layout

| Path | Purpose |
|------|---------|
| `nnpu_torch.py` | The PyTorch port: data loading, model, physics, training, output. Runnable script; **the numerical oracle**. |
| `nnpu_jax.py` | The JAX port (function-for-function mirror of `nnpu_torch.py`; same outputs). Runnable script. |
| `distributed.py` / `distributed_jax.py` | Domain-decomposition helpers (rank/device + SUM combine) for torch / jax. |
| `partition.py` | Integration-cell domain partitioning + per-rank operator slicing (shared NumPy core for both backends). |
| `plot_results.py` | Render the converged solution — backend-agnostic (reads `results_nnrk_LBFGS.txt`). |
| `compare_backends.py` | Head-to-head comparison (converged energy, per-field diffs, per-stage wall time). |
| `NNPU_Modified2026_SCNI3D_torch.ipynb` | Interactive PyTorch notebook (mirrors the original cells, reuses `nnpu_torch.py`). |
| `NNPU_Modified2026_SCNI3D.ipynb` | Original TensorFlow/Keras notebook (reference). |
| `run_nnpu{,_gpu,_ddp}.slurm` | SLURM submit scripts (torch): CPU / single-GPU / multi-GPU. |
| `run_nnpu_jax{,_gpu,_ddp}.slurm` | SLURM submit scripts (jax): CPU / single-GPU / multi-GPU. |
| `Input_3Dbimat_Node396_Cell2662/` | Mesh / shape-function / operator inputs (`.dat`). |
| `Input_3Dbimat_Node396_Cell2662/Results_40NR_4hidden_5bases{,_jax}/` | Outputs (created on run; `_jax` suffix for the JAX backend). |

## Setup

The environment is managed by [uv](https://docs.astral.sh/uv/) (Python 3.13). On Cosmos (SDSC,
AMD MI300A APUs) torch resolves from the ROCm 6.3 wheel index (`[tool.uv.index]` in
`pyproject.toml`); ROCm torch exposes HIP through the `torch.cuda` API, so `--device cuda` and
the `nccl` backend string (→ RCCL) work unchanged on AMD.

```bash
uv sync                  # torch backend: torch+rocm6.3, numpy, pandas, matplotlib, scipy
uv sync --extra jax      # + JAX backend on CPU (validation / backend comparison)
# JAX on the MI300A GPUs is a FUTURE TASK (needs AMD's jax-rocm plugin wheels; see pyproject.toml)
```

## Run

```bash
# quick end-to-end validation (200 / 200 / 50 iterations, ~1 min on CPU)
uv run python nnpu_torch.py --smoke --out-dir /tmp/nnpu_smoke

# full production run (20000 / 30000 / 2000 iterations)
uv run python nnpu_torch.py                 # auto-selects CUDA if available, else CPU
uv run python nnpu_torch.py --device cpu    # force CPU

# on the cluster (whole exclusive node: 4x MI300A APUs, 96 CPU cores)
sbatch run_nnpu.slurm       # CPU
sbatch run_nnpu_gpu.slurm   # single MI300A
```

Key flags: `--epochs-rk`, `--epochs-nnrk`, `--lbfgs-iters`, `--lr-rk`, `--lr-nnrk`,
`--device {auto,cpu,cuda}`, `--dist-backend {auto,nccl,gloo}`, `--out-dir`, `--no-plots`,
`--smoke`. See `--help`.

Plot a finished run's results (reads `results_nnrk_LBFGS.txt`, no retraining):

```bash
uv run python plot_results.py     # writes results_{displacement,strain,stress,enrichment,3d,...}.png
```

## JAX backend

`nnpu_jax.py` is a function-for-function JAX port of `nnpu_torch.py` (the oracle), written so the
three backends (TensorFlow / PyTorch / JAX) can be benchmarked head to head. Same flags
(`--smoke`, `--device`, `--epochs-*`, `--lbfgs-iters`, `--out-dir`, ...), and it writes the
**identical** artifact layout (so `plot_results.py` works on either backend's output). Defaults to
`<input-dir>/Results_40NR_4hidden_5bases_jax` so it never clobbers the torch run.

```bash
uv run --extra jax python nnpu_jax.py --smoke --device cpu --out-dir /tmp/jx_smoke   # ~30 s
uv run --extra jax python nnpu_jax.py --device cpu                                   # full CPU run
sbatch run_nnpu_jax.slurm                                                            # CPU, on the cluster
```

JAX on the Cosmos GPUs is not wired up yet (future task; needs AMD's
[jax-rocm](https://github.com/ROCm/jax) plugin wheels) — `run_nnpu_jax_{gpu,ddp}.slurm` fail fast
with a pointer until then.

JAX specifics: `float64` via `jax_enable_x64`; parameters are a plain pytree driven by Optax;
sparse operators are `jax.experimental.sparse` BCOO (`--dense` switches to dense for a perf
comparison); L-BFGS uses `optax.lbfgs` with a strong-Wolfe **zoom line search** (the analog of
torch's `strong_wolfe`). The enrichment MLP is seeded by JAX's RNG, so trajectories are not
bit-identical to torch, but the **RNG-independent quantities match the oracle to machine precision**
(e.g. the entire Adam-RK loss trajectory agrees to ~`1e-16`) and the converged physics matches.

**Compare backends** (after running ≥2 of them):

```bash
uv run python compare_backends.py            # default torch + jax out-dirs
uv run python compare_backends.py torch:/tmp/tx jax:/tmp/jx   # explicit dirs
# reports converged energy, per-field max/mean |diff|, and per-stage wall time for each backend
```

## Multi-GPU (distributed training)

This problem is **not** sample-parallel DDP. The loss is the *total potential energy = a sum
over integration cells*, and the model (a tiny MLP + global nodal DOFs) is **replicated** on
every rank. So the strategy is **domain decomposition**: the integration cells are split across
ranks (`partition.py`), each rank computes its local energy over a sliced set of operators, and
gradients are combined with a **SUM all-reduce** (`distributed.py`) — DDP's communication pattern
without its `DistributedSampler`/mean-reduce data path. FSDP is not applicable (the model is ~k
parameters; there is nothing to shard). The nodal DOFs are replicated, so there is **no halo
exchange** — the only inter-rank communication is the gradient all-reduce.

Launch with `torchrun`; the script auto-detects the distributed environment (single-process when
run without `torchrun`). Identical seeds across ranks keep the replicas in sync.

```bash
# all 4 MI300A APUs on one node
torchrun --standalone --nproc_per_node=4 nnpu_torch.py --device cuda

# correctness check on CPU (no GPUs needed): must match the single-process result
python  nnpu_torch.py --smoke --device cpu --out-dir /tmp/r1
torchrun --standalone --nproc_per_node=2 nnpu_torch.py --smoke --device cpu --out-dir /tmp/r2
# the Adam stages match to ~1e-15 (machine precision); L-BFGS to ~1e-6 (float reduction order)

# on the cluster
sbatch run_nnpu_ddp.slurm
```

It pays off only at large meshes (millions of integration points), where per-GPU compute (~1/N)
dwarfs the tiny fixed-size gradient all-reduce; for the bundled 2662-cell demo a single process is
faster. Note: the repo's torch is a ROCm 6.3 wheel targeting the MI300A (gfx942); the `nccl`
backend string maps to RCCL on ROCm, so no code changes vs. NVIDIA.

**JAX (`run_nnpu_jax_ddp.slurm`)** uses the same domain-decomposition *semantics* (cells
partitioned, energies SUM-combined) but a different *mechanism*: a **single process** sees all GPUs
on the node and shards the cells across them with `jax.shard_map`, combining per-device energies
with `jax.lax.psum` (no `torchrun`). Keeping the collective inside the jitted energy is what lets
the L-BFGS line search stay globally consistent. Per-device operators are zero-padded to a common
shape (required for the SPMD `psum`). Verified to reproduce the single-device result (Adam
bit-identical; L-BFGS to ~`1e-6`). Multi-node uses `jax.distributed.initialize()` (one process per
GPU; auto-detects SLURM) — see the commented block in the script.

## Outputs (in `--out-dir`)

- `results_nnrk_adam.txt`, `results_nnrk_LBFGS.txt` — per-cell `[x y z | ux uy uz | uy_nn | e11 e22 e33 g23 g13 g12]`.
- `Enrichment_{cell,smooth,node}_{ADAM,LBFGS}.txt` — enrichment basis values.
- `final_{cell,smooth,node}_LBFGS_d{x,y,z}.txt` — spatial Jacobians dζ/dx of the enrichment.
- `layer_*_weights.txt`, `layer_*_biases.txt`, `Final/LBFGS/NNRK_checkpoint.pt` — network parameters.
- `loss_{rk,nnrk,lbfgs}.txt`, `loss_history.png`, and z=0 slice PNGs of displacement/stress/strain/enrichment.
- `timings_{torch,jax}.txt` — per-stage + total wall time `[adam_rk, adam_nnrk, lbfgs, total]` (consumed by `compare_backends.py`).
- The JAX backend writes the same files; its checkpoint is `Final/LBFGS/NNRK_checkpoint.pkl` (pickled NumPy) rather than `.pt`.

## Cosmos cluster (SDSC)

The repo runs on [Cosmos](https://www.sdsc.edu) — HPE Cray EX2500, 42 nodes × 4 AMD MI300A APUs
(gfx942), ROCm 6.3 at `/opt/rocm`, SLURM. Practicalities:

- Single `cluster` partition, **exclusive whole-node allocation** — every job gets 4 APUs +
  96 Zen4 cores, so there is no CPU-vs-GPU queue tradeoff; use whichever fits the run.
- Repo + runs live on VAST scratch (`/cosmos/vast/scratch/$USER/...`) per cluster policy; the
  NFS home has a 100 GB quota and is for source/small files only.
- The login nodes are compute-restricted — validate with `sbatch` smoke jobs, not login-node runs.
- The default APU mode is SPX (4 visible GPUs/node); 6 reserved nodes run TPX
  (`#SBATCH --res=tpx`, 12 logical GPUs/node) — see `run_nnpu_ddp.slurm`.

**Future work on Cosmos** (deliberately not done in the compatibility port):

- **Unified CPU–GPU memory**: the MI300A shares physical memory between CPU and GPU.
  `HSA_XNACK=1` enables page migration so host/device copies (e.g. the `.dat` load → `to(device)`
  path, and the CPU-side result assembly) could be skipped entirely. Needs profiling before use.
- **TPX partitioning**: 12 logical GPUs/node for finer domain decomposition on large meshes.
- **JAX on ROCm**: wire AMD's [jax-rocm](https://github.com/ROCm/jax) plugin wheels into
  `pyproject.toml` and re-enable `run_nnpu_jax_{gpu,ddp}.slurm`.
- **Singularity containerization**: package the environment as a `.sif` image (Cosmos supports
  Singularity via `module load singularitypro`; AMD publishes ROCm PyTorch base images on the
  [Infinity Hub](https://www.amd.com/en/developer/resources/infinity-hub.html)). Run with
  `singularity exec --rocm --bind /cosmos/vast/scratch/$USER:/workspace <image>.sif ...` —
  makes the venv reproducible and portable across ROCm upgrades.

## Notes on the port

- `float64` throughout, matching the original `tf.keras.backend.set_floatx("float64")`.
- TensorFlow `SparseTensor` / `sparse_dense_matmul` → `torch.sparse_coo_tensor` / `torch.sparse.mm`.
- TFP `lbfgs_minimize` → `torch.optim.LBFGS` (strong-Wolfe line search).
- The orthonormalization diagnostic reproduces the notebook's ~`1e-8` RK-orthogonality check
  (observed ~`1e-13`).
- Random initialization differs from TensorFlow's RNG, so absolute loss trajectories will not be
  bit-identical, but the physics (displacements, stresses, energy) match.
