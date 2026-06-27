"""Device/topology helper for the JAX NN-RKPM trainer (``nnpu_jax.py``).

Same *semantics* as ``distributed.py`` (the torch port) -- domain decomposition with a SUM
combine, because the loss is a spatial integral over cells, not a mean -- but a different
*mechanism*, idiomatic to JAX:

  * Single node, multi-GPU: ONE process sees all local GPUs. The integration cells are sharded
    across ``jax.devices()`` with ``jax.shard_map`` and the per-device energies are combined with
    a SUM ``jax.lax.psum`` (see ``nnpu_jax.build_global_energy``). No ``torchrun``, no
    ``jax.distributed.initialize()`` -- the collective lives inside the jitted energy, which is
    exactly what lets the L-BFGS line search stay globally consistent (a host-side all-reduce
    could not run inside the line-search ``lax`` loop).
  * Multi node: launched with >1 SLURM task; ``jax.distributed.initialize()`` (auto-detects SLURM)
    wires the processes and ``jax.devices()`` then spans all of them, so the same shard_map+psum
    energy works unchanged. See ``run_nnpu_jax_ddp.slurm``.

The bundled 2662-cell demo is faster on a single device; the sharded path is for large meshes.
"""
from __future__ import annotations

import os

import jax


class Distributed:
    def __init__(self, force_cpu: bool = False):
        # Multi-process (multi-node) only when launched with >1 SLURM task; otherwise a single
        # process drives all visible devices. (``--device cpu`` forces the CPU platform via
        # JAX_PLATFORMS, set at the top of nnpu_jax.py before jax is imported.)
        ntasks = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", "1")))
        self.multiprocess = ntasks > 1
        if self.multiprocess:
            jax.distributed.initialize()          # auto-detects SLURM coordinator / process id
            self.process_id = jax.process_index()
            self.num_processes = jax.process_count()
        else:
            self.process_id, self.num_processes = 0, 1

        self.devices = jax.devices()              # all *global* devices (sharding axis for cells)
        self.n_devices = len(self.devices)
        # mirror distributed.py's API (output is written by the main process / device 0 owner)
        self.rank, self.world_size = self.process_id, self.num_processes

    @property
    def is_main(self) -> bool:
        return self.process_id == 0

    @property
    def sharded(self) -> bool:
        """True iff cells should be domain-decomposed across >1 device."""
        return self.n_devices > 1

    def make_mesh(self, axis_name: str = "cells"):
        """1-D device mesh over all global devices, for shard_map cell decomposition."""
        from jax.sharding import Mesh
        import numpy as np
        return Mesh(np.asarray(self.devices), (axis_name,))

    def barrier(self):
        if self.multiprocess:
            from jax.experimental import multihost_utils
            multihost_utils.sync_global_devices("barrier")

    def all_reduce_scalar(self, value: float) -> float:
        """Global SUM of a per-process scalar. Only needed across *processes*; within one process
        the shard_map ``psum`` already yields a global value, so this is the identity."""
        if not self.multiprocess:
            return float(value)
        import jax.numpy as jnp
        from jax.experimental import multihost_utils
        g = multihost_utils.process_allgather(jnp.asarray(float(value)))
        return float(g.sum())

    def finalize(self):
        if self.multiprocess:
            jax.distributed.shutdown()
