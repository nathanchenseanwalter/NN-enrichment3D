"""Domain (integration-cell) partitioning for distributed NN-RKPM training.

The total potential energy is a sum over integration cells, so each rank owns a disjoint
subset of cells C_r and only needs:
  * the rows of the smoothed-derivative operators P1/P2/P3 for its cells, with columns
    restricted to the smoothing points S_r those rows reference (remapped to 0..|S_r|);
  * the rows of SHP_smooth for S_r (columns = RK nodes, kept full -- the nodal DOFs are
    replicated on every rank, so no halo exchange is needed);
  * per-cell dense arrays (WT, mu, lam) sliced to C_r and per-smoothing-point arrays
    (X_smooth) sliced to S_r.

Operators are sliced at the NumPy/CSV level so a rank never materializes the full operator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def partition_cells(n_cell: int, rank: int, world_size: int) -> np.ndarray:
    """Balanced contiguous split of cell indices across ranks (sizes differ by <= 1)."""
    return np.array_split(np.arange(n_cell, dtype=np.int64), world_size)[rank]


def _read_triplet(path):
    df = pd.read_csv(path, header=None, names=["i", "j", "v"])
    return (df["i"].to_numpy(np.int64), df["j"].to_numpy(np.int64),
            df["v"].to_numpy(np.float64))


def _coo(i, j, v, shape, device, dtype):
    idx = torch.as_tensor(np.ascontiguousarray(np.stack([i, j])), device=device)
    val = torch.as_tensor(np.ascontiguousarray(v), dtype=dtype, device=device)
    return torch.sparse_coo_tensor(idx, val, shape, device=device).coalesce()


def build_local_triplets(paths, cell_ids, n_cell, n_smooth, n_node):
    """Slice P1/P2/P3 (cell x smooth) and SHP_smooth (smooth x node) to a rank's cells.

    Framework-agnostic core (NumPy only, no torch/jax): both the PyTorch and JAX backends build
    their own sparse types from this identical slicing logic. ``paths`` maps
    {'P1','P2','P3','SHP_smooth'} -> filenames. Returns a dict with, per operator key, a tuple
    ``(i, j, v, shape)`` whose rows/cols are already remapped to the local index space, plus the
    global ``smooth_ids`` (S_r) and ``cell_ids`` (C_r) so the caller can slice dense per-cell /
    per-smooth arrays consistently.
    """
    cell_ids = np.asarray(cell_ids, np.int64)
    nC = len(cell_ids)
    row_keep = np.zeros(n_cell, dtype=bool); row_keep[cell_ids] = True
    row_remap = -np.ones(n_cell, dtype=np.int64); row_remap[cell_ids] = np.arange(nC)

    # pass 1: keep local cell rows of each P; collect the smoothing columns they reference
    filtered, cols = [], []
    for key in ("P1", "P2", "P3"):
        i, j, v = _read_triplet(paths[key])
        m = row_keep[i]
        filtered.append((row_remap[i[m]], j[m], v[m]))
        cols.append(j[m])
    smooth_ids = np.unique(np.concatenate(cols)) if cols else np.zeros(0, np.int64)
    nS = len(smooth_ids)
    col_remap = -np.ones(n_smooth, dtype=np.int64); col_remap[smooth_ids] = np.arange(nS)

    out = {"smooth_ids": smooth_ids, "cell_ids": cell_ids}
    for key, (i, j, v) in zip(("P1", "P2", "P3"), filtered):
        out[key] = (i, col_remap[j], v, (nC, nS))

    # SHP_smooth rows for S_r; node columns kept full (nodal DOFs are replicated)
    si, sj, sv = _read_triplet(paths["SHP_smooth"])
    sm = np.zeros(n_smooth, dtype=bool); sm[smooth_ids] = True
    keep = sm[si]
    out["SHP_smooth"] = (col_remap[si[keep]], sj[keep], sv[keep], (nS, n_node))
    return out


def build_local_operators(paths, cell_ids, n_cell, n_smooth, n_node, device, dtype):
    """Torch wrapper: build per-rank coalesced sparse COO operators from the shared triplets."""
    tri = build_local_triplets(paths, cell_ids, n_cell, n_smooth, n_node)
    out = {"smooth_ids": tri["smooth_ids"], "cell_ids": tri["cell_ids"]}
    for key in ("P1", "P2", "P3", "SHP_smooth"):
        i, j, v, shape = tri[key]
        out[key] = _coo(i, j, v, shape, device, dtype)
    return out
