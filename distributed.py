"""Minimal torch.distributed helper for the NN-RKPM domain-decomposition trainer.

This is *not* standard data-parallel DDP: the model (a tiny MLP + global nodal DOFs) is
replicated identically on every rank, the integration domain is partitioned across ranks,
and gradients are combined with a **SUM** all-reduce (the loss is a spatial integral, i.e. a
sum over cells -- so the global gradient is the sum, not the mean, of per-rank gradients).

Convention follows the project's other repos (e.g. audioflow's ``Distributed``): rank/world
come from the ``torchrun`` environment, NCCL on GPU and gloo on CPU, and a transparent
single-process fallback when not launched under torchrun.
"""
from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist


class Distributed:
    def __init__(self, backend: str = "auto", force_cpu: bool = False, timeout_min: int = 30):
        ws = int(os.environ.get("WORLD_SIZE", "1"))
        self.distributed = ("RANK" in os.environ) and ws > 1
        if self.distributed:
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = ws
        else:
            self.rank, self.local_rank, self.world_size = 0, 0, 1

        if force_cpu or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)

        if self.distributed:
            if backend == "auto":
                backend = "gloo" if self.device.type == "cpu" else "nccl"
            kwargs = {}
            if self.device.type == "cuda":
                # pin the rank -> GPU mapping (avoids NCCL "devices unknown" barrier warning)
                kwargs["device_id"] = self.device
            dist.init_process_group(
                backend, init_method="env://",
                timeout=datetime.timedelta(minutes=timeout_min), **kwargs,
            )
            self.barrier()

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self):
        if self.distributed:
            dist.barrier()

    def all_reduce_grads_sum_(self, params):
        """In-place SUM all-reduce of the .grad of each trained parameter (global gradient)."""
        if not self.distributed:
            return
        for p in params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

    def all_reduce_scalar(self, value: float) -> float:
        """Return the global SUM of a per-rank scalar (e.g. local energy)."""
        if not self.distributed:
            return value
        t = torch.tensor([value], dtype=torch.float64, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item())

    def finalize(self):
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()
