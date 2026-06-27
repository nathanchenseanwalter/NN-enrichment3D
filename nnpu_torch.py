#!/usr/bin/env python
"""
NN-PU / NN-RKPM for 3D bimaterial elasticity -- PyTorch port of
``NNPU_Modified2026_SCNI3D.ipynb`` (originally TensorFlow/Keras + TFP, Colab).

Method (Deep Energy Method with SCNI -- Stabilized Conforming Nodal Integration):

  * RK (Reproducing Kernel) background displacement field, coefficients d*_int
  * Neural-network enrichment: an MLP of nodal coordinates produces n_NC basis
    functions zeta; enrichment coefficients d*_NN combine them, gated by a
    spatial activation mask ``is_activated``.
  * The displacement is u = u_RK + u_NN in x/y/z.
  * Smoothed (SCNI) strains are obtained from precomputed sparse derivative
    operators P1/P2/P3; stresses from isotropic linear elasticity with
    per-cell (bimaterial) moduli.
  * Loss = total potential energy = scaling * sum( strain_energy_density * WT ).
  * Three training stages:
        1. Adam, RK coefficients only
        2. Adam, RK + NN coefficients + network weights
        3. L-BFGS, all of the above

Outputs (displacements, strains, stresses, enrichment functions and their
spatial Jacobians, network weights, loss histories) are written to ``--out-dir``,
mirroring the original notebook's artifacts.
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import partition
from distributed import Distributed


# =====================================================================
# I/O helpers
# =====================================================================
def read_dense(fname, device, dtype):
    """Read a dense CSV (.dat) into a (rows, cols) tensor."""
    arr = np.array(pd.read_csv(fname, header=None).to_numpy(dtype=np.float64))
    return torch.as_tensor(arr, dtype=dtype, device=device)


def read_sparse(fname, size, device, dtype):
    """Read a COO triplet CSV (idx_i, idx_j, val) into a coalesced sparse tensor."""
    df = pd.read_csv(fname, header=None, names=["i", "j", "v"])
    idx = torch.as_tensor(np.array(df[["i", "j"]].to_numpy(dtype=np.int64).T), device=device)
    val = torch.as_tensor(np.array(df["v"].to_numpy(dtype=np.float64)), dtype=dtype, device=device)
    return torch.sparse_coo_tensor(idx, val, size=tuple(size), device=device).coalesce()


def sp_transpose(A):
    """Transpose of a 2-D sparse COO tensor (returns a coalesced tensor)."""
    idx = A._indices()
    val = A._values()
    return torch.sparse_coo_tensor(
        torch.stack([idx[1], idx[0]]), val, (A.shape[1], A.shape[0]), device=A.device
    ).coalesce()


def get_region(shp):
    """Binary support map: set all nonzeros of ``shp`` to 1 and transpose.

    Mirrors the notebook's ``get_region`` -> shape (n_points_cols, n_points_rows).
    """
    idx = shp._indices()
    ones = torch.ones_like(shp._values())
    mask = torch.sparse_coo_tensor(idx, ones, shp.shape, device=shp.device).coalesce()
    return sp_transpose(mask)


def spmm(A_sparse, B_dense):
    """Sparse-dense matmul that backpropagates through the dense argument."""
    return torch.sparse.mm(A_sparse, B_dense)


# =====================================================================
# Neural-network enrichment basis (model_NC)
# =====================================================================
class EnrichmentMLP(nn.Module):
    """MLP mapping 3D coordinates -> n_NC enrichment basis functions.

    Faithful to the Keras build: every layer (including the last) is followed by
    its activation, the first layer's weights are initialized to U(-0.1, 0.1)
    (a = 2/20) and the remaining layers to U(-1, 1); all biases start at zero.
    """

    def __init__(self, nHL, nNR, actv):
        super().__init__()
        assert len(nNR) == nHL and len(actv) == nHL
        self.acts = list(actv)
        self.layers = nn.ModuleList()
        in_dim = 3
        for i in range(nHL):
            lin = nn.Linear(in_dim, nNR[i])
            a = 2.0 / 20.0 if i == 0 else 1.0
            nn.init.uniform_(lin.weight, -a, a)
            nn.init.zeros_(lin.bias)
            self.layers.append(lin)
            in_dim = nNR[i]

    @staticmethod
    def _act(name, x):
        if name == "elu":
            return F.elu(x)
        if name in ("tanh",):
            return torch.tanh(x)
        if name in ("relu",):
            return F.relu(x)
        raise ValueError(f"unsupported activation {name!r}")

    def forward(self, x):
        for name, lin in zip(self.acts, self.layers):
            x = self._act(name, lin(x))
        return x


# =====================================================================
# The NN-RKPM problem
# =====================================================================
class NNRK:
    def __init__(self, data_dir, device, dtype, nHL, nNR, actv, gy_ebc=0.01, seed=42, dist=None):
        self.device = device
        self.dtype = dtype
        self.data_dir = data_dir
        self.dist = dist
        self.world_size = getattr(dist, "world_size", 1)
        self.rank = getattr(dist, "rank", 0)
        self.is_main = self.rank == 0
        # Identical seed on every rank -> identical replicated parameters (NOT 42 + rank).
        torch.manual_seed(seed)
        np.random.seed(seed)

        rd = lambda f: read_dense(os.path.join(data_dir, f), device, dtype)
        rs = lambda f, size: read_sparse(os.path.join(data_dir, f), size, device, dtype)

        # ---- point sets / per-cell data (full on every rank; small dense arrays) ----
        self.X_cell = rd("x.dat")            # integration points (cell centers)
        self.X_smooth = rd("x_smoothing.dat")  # smoothing (cell-edge) points (full coords)
        self.X_node = rd("x_node.dat")       # RK nodal points
        self.Vlabel = rd("Vlabel.dat")       # per-cell material labels
        self.WT = rd("WT.dat")               # cell integration weights (volumes)
        self.WT_smooth = rd("WT_smooth.dat")

        self.n_cell = self.X_cell.shape[0]
        self.n_smooth = self.X_smooth.shape[0]
        self.n_node = self.X_node.shape[0]

        # cell shape functions + support map (every rank: needed for the activation mask)
        self.SHP_cell = rs("shp.dat", [self.n_cell, self.n_node])
        self.region_cell = get_region(self.SHP_cell)

        # ---- boundary-condition DOF maps (replicated) ----
        ns = read_dense(os.path.join(data_dir, "bc_info.dat"), device, torch.int64)
        self.n_ebc_x = int(ns[0, 0]); self.n_ebc_y = int(ns[0, 1]); self.n_ebc_z = int(ns[0, 2])
        nn_ = self.n_node
        self.CVC_int_x = rs("CVC_int_x.dat", [nn_, nn_ - self.n_ebc_x])
        self.CVC_ebc_x = rs("CVC_ebc_x.dat", [nn_, self.n_ebc_x])
        self.CVC_int_y = rs("CVC_int_y.dat", [nn_, nn_ - self.n_ebc_y])
        self.CVC_ebc_y = rs("CVC_ebc_y.dat", [nn_, self.n_ebc_y])
        self.CVC_int_z = rs("CVC_int_z.dat", [nn_, nn_ - self.n_ebc_z])
        self.CVC_ebc_z = rs("CVC_ebc_z.dat", [nn_, self.n_ebc_z])

        # ---- geometry extents (global) ----
        Xs = self.X_smooth
        self.x_min, self.y_min, self.z_min = (Xs[:, k].min() for k in range(3))
        self.x_max, self.y_max, self.z_max = (Xs[:, k].max() for k in range(3))

        # ---- material properties (bimaterial, per cell; global) ----
        Em, num_ = 10400.0, 0.3      # matrix  (label 0)
        Ei, nui = 52000.0, 0.3       # inclusion (label != 0)
        is_matrix = self.Vlabel == 0
        E = torch.where(is_matrix, torch.full_like(self.Vlabel, Em), torch.full_like(self.Vlabel, Ei))
        nu = torch.where(is_matrix, torch.full_like(self.Vlabel, num_), torch.full_like(self.Vlabel, nui))
        self.mat_mu = E / 2.0 / (1.0 + nu)
        self.mat_lam = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu)

        vol = (self.x_max - self.x_min) * (self.y_max - self.y_min) * (self.z_max - self.z_min)
        mat_energy_scaling_ref = vol / E.max()

        # ---- enrichment network and parameters (replicated, identical on all ranks) ----
        self.nNR = list(nNR)
        self.n_NC = nNR[nHL - 1]
        self.model = EnrichmentMLP(nHL, nNR, actv).to(device=device, dtype=dtype)
        self.sum_NC = torch.ones((self.n_NC, 1), dtype=dtype, device=device)

        z = lambda r, c: nn.Parameter(torch.zeros((r, c), dtype=dtype, device=device))
        self.dx_int = z(nn_ - self.n_ebc_x, 1)
        self.dy_int = z(nn_ - self.n_ebc_y, 1)
        self.dz_int = z(nn_ - self.n_ebc_z, 1)
        self.dx_NN = z(nn_ - self.n_ebc_x, self.n_NC)
        self.dy_NN = z(nn_ - self.n_ebc_y, self.n_NC)
        self.dz_NN = z(nn_ - self.n_ebc_z, self.n_NC)

        # ---- essential boundary conditions ----
        self.gy_ebc = gy_ebc
        self._setup_ebc()

        # ---- scalings (use the GLOBAL total weight, not a partition) ----
        self.mat_energy_scaling = float(mat_energy_scaling_ref / (gy_ebc ** 2) / self.WT.sum())
        self.d_scaling = float(gy_ebc)

        # ---- activation mask (manual: a y-band of width 0.1 around y = 0) ----
        self.is_activated = self._initial_activation([0.0, 0.0, 0.0], -0.05, 0.05)

        # ---- operators: full (rank 0 / single-process) + per-rank training slice ----
        self._build_operators(rs)

    # ----------------------------------------------------------------
    def _build_operators(self, rs):
        """Assemble the operator bundles used by the forward model.

        ``self.tr`` (training): per-rank domain-decomposed slice in distributed mode, full
        otherwise. ``self.full`` (output/diagnostics, rank 0 and single-process only): the
        complete operators over all cells/smoothing points. ``M_c``/``M_s`` are rank-0-only.
        """
        full_owner = self.is_main or self.world_size == 1
        if full_owner:
            self.full = SimpleNamespace(
                X_smooth=self.X_smooth,
                SHP_smooth=rs("shp_smoothing.dat", [self.n_smooth, self.n_node]),
                P1=rs("P1.dat", [self.n_cell, self.n_smooth]),
                P2=rs("P2.dat", [self.n_cell, self.n_smooth]),
                P3=rs("P3.dat", [self.n_cell, self.n_smooth]),
                WT=self.WT, mu=self.mat_mu, lam=self.mat_lam,
            )
            self.M_c = rs("M_c.dat", [self.n_node, self.n_node])
            self.M_s = rs("M_s.dat", [self.n_node, self.n_node])
        else:
            self.full = None
            self.M_c = self.M_s = None

        if self.world_size == 1:
            self.tr = self.full
            self.cell_ids = np.arange(self.n_cell)
            self.smooth_ids = np.arange(self.n_smooth)
        else:
            cell_ids = partition.partition_cells(self.n_cell, self.rank, self.world_size)
            paths = {k: os.path.join(self.data_dir, f) for k, f in
                     (("P1", "P1.dat"), ("P2", "P2.dat"), ("P3", "P3.dat"),
                      ("SHP_smooth", "shp_smoothing.dat"))}
            loc = partition.build_local_operators(
                paths, cell_ids, self.n_cell, self.n_smooth, self.n_node, self.device, self.dtype)
            self.cell_ids, self.smooth_ids = loc["cell_ids"], loc["smooth_ids"]
            cids = torch.as_tensor(self.cell_ids, device=self.device)
            sids = torch.as_tensor(self.smooth_ids, device=self.device)
            self.tr = SimpleNamespace(
                X_smooth=self.X_smooth.index_select(0, sids),
                SHP_smooth=loc["SHP_smooth"], P1=loc["P1"], P2=loc["P2"], P3=loc["P3"],
                WT=self.WT.index_select(0, cids),
                mu=self.mat_mu.index_select(0, cids),
                lam=self.mat_lam.index_select(0, cids),
            )

    # distributed reduction helpers (no-ops without a Distributed instance)
    def _allreduce_grads(self, params):
        if self.dist is not None:
            self.dist.all_reduce_grads_sum_(params)

    def _global(self, value):
        return self.dist.all_reduce_scalar(value) if self.dist is not None else value

    # ----------------------------------------------------------------
    def _setup_ebc(self):
        # Apply unit (scaled by gy_ebc) y-displacement on the top (y = y_max) boundary.
        dx_ref = torch.zeros((self.n_ebc_x, 1), dtype=self.dtype, device=self.device)
        dy_ref = torch.zeros((self.n_ebc_y, 1), dtype=self.dtype, device=self.device)
        dz_ref = torch.zeros((self.n_ebc_z, 1), dtype=self.dtype, device=self.device)
        idx = read_dense(os.path.join(self.data_dir, "CVC_ebc_y.dat"), self.device, self.dtype)
        for i in range(idx.shape[0]):
            node_id = int(idx[i, 0].item())
            y = self.X_node[node_id, 1]
            if y > self.y_max - 1e-6:
                dy_ref[int(idx[i, 1].item()), 0] = 1.0
        self.dx_ebc = dx_ref
        self.dy_ebc = dy_ref * self.gy_ebc
        self.dz_ebc = dz_ref

    def _initial_activation(self, center, r1, r2):
        # Cells whose y-offset from center lies in (r1, r2); mapped to supporting nodes.
        dy = self.X_cell[:, 1] - center[1]
        cell_mask = ((dy > r1) & (dy < r2)).to(self.dtype).reshape(-1, 1)
        node_val = spmm(self.region_cell, cell_mask)
        node_val = node_val / (node_val + 1e-16)  # -> {0, 1}
        return node_val.detach()

    # ----------------------------------------------------------------
    # forward model
    # ----------------------------------------------------------------
    def rk_approx(self, shp, d, CVC=None, d_ebc=None, CVC_ebc=None, act=None):
        if CVC is None:
            d_full = d
        else:
            d_full = spmm(CVC, d)
            if d_ebc is not None:
                d_full = d_full + spmm(CVC_ebc, d_ebc)
        if act is not None:
            d_full = d_full * act
        return spmm(shp, d_full)

    def nn_approx(self, zeta, shp, d_NN, CVC_int):
        v = self.rk_approx(shp, d_NN, CVC=CVC_int, act=self.is_activated)
        return (zeta * v) @ self.sum_NC

    def total_approx(self, x, shp):
        s = self.d_scaling
        ux = self.rk_approx(shp, s * self.dx_int, CVC=self.CVC_int_x, d_ebc=self.dx_ebc, CVC_ebc=self.CVC_ebc_x)
        uy = self.rk_approx(shp, s * self.dy_int, CVC=self.CVC_int_y, d_ebc=self.dy_ebc, CVC_ebc=self.CVC_ebc_y)
        uz = self.rk_approx(shp, s * self.dz_int, CVC=self.CVC_int_z, d_ebc=self.dz_ebc, CVC_ebc=self.CVC_ebc_z)
        zeta = self.model(x)
        ux = ux + self.nn_approx(zeta, shp, s * self.dx_NN, self.CVC_int_x)
        uy = uy + self.nn_approx(zeta, shp, s * self.dy_NN, self.CVC_int_y)
        uz = uz + self.nn_approx(zeta, shp, s * self.dz_NN, self.CVC_int_z)
        return ux, uy, uz

    def total_approx_nn(self, x, shp):
        s = self.d_scaling
        zeta = self.model(x)
        ux = self.nn_approx(zeta, shp, s * self.dx_NN, self.CVC_int_x)
        uy = self.nn_approx(zeta, shp, s * self.dy_NN, self.CVC_int_y)
        uz = self.nn_approx(zeta, shp, s * self.dz_NN, self.CVC_int_z)
        return ux, uy, uz, zeta

    def rk_only(self, shp):
        s = self.d_scaling
        ux = self.rk_approx(shp, s * self.dx_int, CVC=self.CVC_int_x, d_ebc=self.dx_ebc, CVC_ebc=self.CVC_ebc_x)
        uy = self.rk_approx(shp, s * self.dy_int, CVC=self.CVC_int_y, d_ebc=self.dy_ebc, CVC_ebc=self.CVC_ebc_y)
        uz = self.rk_approx(shp, s * self.dz_int, CVC=self.CVC_int_z, d_ebc=self.dz_ebc, CVC_ebc=self.CVC_ebc_z)
        return ux, uy, uz

    def smoothed_strain(self, ux, uy, uz, ops):
        P1, P2, P3 = ops.P1, ops.P2, ops.P3
        exx = spmm(P1, ux); eyy = spmm(P2, uy); ezz = spmm(P3, uz)
        gxy = spmm(P2, ux) + spmm(P1, uy)
        gyz = spmm(P3, uy) + spmm(P2, uz)
        gxz = spmm(P3, ux) + spmm(P1, uz)
        return exx, eyy, ezz, gxy, gyz, gxz

    def stress(self, exx, eyy, ezz, gxy, gyz, gxz, mu=None, lam=None):
        # default to the full per-cell moduli (correct for single-process / full-domain use)
        if mu is None:
            mu, lam = self.mat_mu, self.mat_lam
        M = 2.0 * mu + lam
        sxx = M * exx + lam * (eyy + ezz)
        syy = M * eyy + lam * (exx + ezz)
        szz = M * ezz + lam * (exx + eyy)
        return sxx, syy, szz, mu * gxy, mu * gyz, mu * gxz

    @staticmethod
    def energy_density(exx, eyy, ezz, gxy, gyz, gxz, sxx, syy, szz, sxy, syz, sxz):
        return 0.5 * (exx * sxx + eyy * syy + ezz * szz + gxy * sxy + gyz * syz + gxz * sxz)

    # composed strain fields (ops defaults to the per-rank training slice)
    def strain_rk(self, ops=None):
        ops = ops or self.tr
        return self.smoothed_strain(*self.rk_only(ops.SHP_smooth), ops)

    def strain_nnrk(self, ops=None):
        ops = ops or self.tr
        return self.smoothed_strain(*self.total_approx(ops.X_smooth, ops.SHP_smooth), ops)

    def strain_nn(self, ops=None):
        ops = ops or self.tr
        ux, uy, uz, _ = self.total_approx_nn(ops.X_smooth, ops.SHP_smooth)
        return self.smoothed_strain(ux, uy, uz, ops)

    def _energy(self, strains, ops):
        s = self.stress(*strains, ops.mu, ops.lam)
        psi = self.energy_density(*strains, *s)
        # local (per-rank) potential energy; the global energy is its all-reduce SUM
        return self.mat_energy_scaling * (psi * ops.WT).sum()

    def energy_rk(self, ops=None):
        ops = ops or self.tr
        return self._energy(self.strain_rk(ops), ops)

    def energy_nnrk(self, ops=None):
        ops = ops or self.tr
        return self._energy(self.strain_nnrk(ops), ops)

    # ----------------------------------------------------------------
    # orthonormalization diagnostic (cell 12 of the notebook)
    # ----------------------------------------------------------------
    @torch.no_grad()
    def orthonormalize_nn_basis(self, x, weight, shp, M):
        """Project the raw NN basis onto the RK-orthogonal complement and unit-normalize."""
        zeta = self.model(x)
        zeta = zeta / (torch.sqrt((weight * zeta ** 2).sum(0)) + 1e-16)
        bk = spmm(sp_transpose(shp), weight * zeta)          # (n_node, n_NC)
        ck = torch.linalg.solve(M.to_dense(), bk)            # M ck = bk
        zeta_ortho = zeta - spmm(shp, ck)
        return zeta_ortho / (torch.sqrt((weight * zeta_ortho ** 2).sum(0)) + 1e-16)

    @torch.no_grad()
    def orthogonality_check(self):
        """Max RK-inner-product of the orthonormalized basis (should be ~0)."""
        out = {}
        for name, x, w, shp, M in (("cell", self.X_cell, self.WT, self.SHP_cell, self.M_c),
                                    ("smooth", self.X_smooth, self.WT_smooth, self.full.SHP_smooth, self.M_s)):
            zo = self.orthonormalize_nn_basis(x, w, shp, M)
            inner = spmm(sp_transpose(shp), w * zo)
            out[name] = float(inner.abs().max())
        return out

    # ----------------------------------------------------------------
    # training stages
    # ----------------------------------------------------------------
    def train_adam_rk(self, num_epochs, lr=1e-3, log_every=100,
                       es_threshold=1e-12, es_patience=5):
        params = [self.dx_int, self.dy_int, self.dz_int]
        opt = torch.optim.Adam(params, lr=lr, eps=1e-6)
        history = np.zeros(num_epochs)
        prev, counter = float("inf"), 0
        if self.is_main:
            print("Starting RK (background) optimization ...", flush=True)
        for epoch in range(num_epochs):
            opt.zero_grad(set_to_none=True)
            loss = self.energy_rk()           # local (per-rank) energy
            loss.backward()
            self._allreduce_grads(params)      # global gradient = SUM of per-rank gradients
            opt.step()
            lv = self._global(loss.item())     # global energy (for logging / early stop)
            history[epoch] = lv
            if epoch > 0:
                counter = counter + 1 if abs(prev - lv) < es_threshold else 0
                if counter >= es_patience:
                    if self.is_main:
                        print(f"  early stop @ epoch {epoch} (|dloss|<{es_threshold:.1e})", flush=True)
                    history = history[: epoch + 1]
                    break
            prev = lv
            if self.is_main and epoch % log_every == 0:
                print(f"  [RK] epoch {epoch:6d} | loss {lv:.6e}", flush=True)
        if self.is_main:
            print(f"RK optimization done. final loss {history[-1]:.6e}", flush=True)
        return history

    def _all_nnrk_params(self):
        return [self.dx_int, self.dy_int, self.dz_int,
                self.dx_NN, self.dy_NN, self.dz_NN] + list(self.model.parameters())

    def _snapshot(self):
        return [p.detach().clone() for p in self._all_nnrk_params()]

    def _restore(self, snap):
        with torch.no_grad():
            for p, s in zip(self._all_nnrk_params(), snap):
                p.copy_(s)

    def train_adam_nnrk(self, num_epochs, lr=1e-4, log_every=100,
                        es_threshold=1e-11, es_patience=5):
        params = self._all_nnrk_params()
        opt = torch.optim.Adam(params, lr=lr, eps=1e-6)
        history = np.zeros(num_epochs)
        prev, counter = float("inf"), 0
        best_loss, best_snap, best_epoch = float("inf"), self._snapshot(), 0
        if self.is_main:
            print("Starting NN-RK optimization ...", flush=True)
        for epoch in range(num_epochs):
            opt.zero_grad(set_to_none=True)
            loss = self.energy_nnrk()          # local (per-rank) energy
            loss.backward()
            self._allreduce_grads(params)      # global gradient = SUM of per-rank gradients
            opt.step()
            lv = self._global(loss.item())     # global energy (identical on all ranks)
            history[epoch] = lv
            if epoch > 100:
                counter = counter + 1 if abs(prev - lv) < es_threshold else 0
                if counter >= es_patience:
                    if self.is_main:
                        print(f"  early stop @ epoch {epoch} (|dloss|<{es_threshold:.1e})", flush=True)
                    history = history[: epoch + 1]
                    break
            prev = lv
            if lv < best_loss:   # global loss -> all ranks snapshot consistently
                best_loss, best_snap, best_epoch = lv, self._snapshot(), epoch
            if self.is_main and epoch % log_every == 0:
                print(f"  [NNRK] epoch {epoch:6d} | loss {lv:.6e}", flush=True)
        self._restore(best_snap)
        if self.is_main:
            print(f"NN-RK optimization done. best loss {best_loss:.6e} @ epoch {best_epoch}", flush=True)
        return history

    def train_lbfgs(self, max_iter=2000, history_size=20, log_every=100):
        params = self._all_nnrk_params()
        opt = torch.optim.LBFGS(
            params, max_iter=max_iter, history_size=history_size,
            line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-16,
            max_eval=max_iter * 5,
        )
        history = []

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = self.energy_nnrk()          # local energy
            loss.backward()
            self._allreduce_grads(params)      # global gradient in .grad
            gl = self._global(loss.item())     # global energy (identical on all ranks ->
            history.append(gl)                 # identical line-search decisions, no deadlock)
            if self.is_main and len(history) % log_every == 0:
                print(f"  [LBFGS] eval {len(history):6d} | loss {gl:.6e}", flush=True)
            return torch.as_tensor(gl, dtype=self.dtype, device=self.device)

        if self.is_main:
            print("Starting L-BFGS optimization ...", flush=True)
        tic = time.time()
        opt.step(closure)
        if self.is_main:
            print(f"L-BFGS done in {time.time() - tic:.1f}s, {len(history)} evals, "
                  f"final loss {history[-1]:.6e}", flush=True)
        return np.array(history)

    # ----------------------------------------------------------------
    # outputs
    # ----------------------------------------------------------------
    @torch.no_grad()
    def _np(self, t):
        return t.detach().cpu().numpy()

    @torch.no_grad()
    def write_solution(self, fname):
        ux, uy, uz = self.total_approx(self.X_cell, self.SHP_cell)
        _, uy_nn, _, _ = self.total_approx_nn(self.X_cell, self.SHP_cell)
        e11, e22, e33, g12, g23, g13 = self.strain_nnrk(self.full)
        cols = [self.X_cell, ux, uy, uz, uy_nn, e11, e22, e33, g23, g13, g12]
        out = np.concatenate([self._np(c).reshape(self.n_cell, -1) for c in cols], axis=1)
        np.savetxt(fname, out)

    @torch.no_grad()
    def save_enrichment(self, out_dir, tag):
        for name, x in (("cell", self.X_cell), ("smooth", self.X_smooth), ("node", self.X_node)):
            np.savetxt(os.path.join(out_dir, f"Enrichment_{name}_{tag}.txt"), self._np(self.model(x)))

    def save_enrichment_jacobian(self, out_dir, tag):
        """d zeta / d x at cell/smooth/node points (mirrors notebook cells 35-37)."""
        for name, X in (("cell", self.X_cell), ("smooth", self.X_smooth), ("node", self.X_node)):
            x = X.detach().clone().requires_grad_(True)
            z = self.model(x)                       # (N, n_NC)
            grads = []
            for k in range(self.n_NC):
                g = torch.autograd.grad(z[:, k].sum(), x, retain_graph=True)[0]  # (N, 3)
                grads.append(g)
            G = torch.stack(grads, dim=1)           # (N, n_NC, 3)
            np.savetxt(os.path.join(out_dir, f"Enrichment_{name}_{tag}.txt"), self._np(z))
            for d, comp in enumerate(("dx", "dy", "dz")):
                np.savetxt(os.path.join(out_dir, f"final_{name}_{tag}_{comp}.txt"), self._np(G[:, :, d]))

    @torch.no_grad()
    def save_layer_weights(self, out_dir):
        # weight saved transposed so the on-disk shape matches Keras (in_dim, units).
        for i, lin in enumerate(self.model.layers):
            np.savetxt(os.path.join(out_dir, f"layer_{i}_weights.txt"), self._np(lin.weight.t()), delimiter=" ")
            np.savetxt(os.path.join(out_dir, f"layer_{i}_biases.txt"), self._np(lin.bias), delimiter=" ")

    def save_checkpoint(self, out_dir, tag="NNRK"):
        os.makedirs(out_dir, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "dx_int": self.dx_int.detach().cpu(), "dy_int": self.dy_int.detach().cpu(),
                "dz_int": self.dz_int.detach().cpu(), "dx_NN": self.dx_NN.detach().cpu(),
                "dy_NN": self.dy_NN.detach().cpu(), "dz_NN": self.dz_NN.detach().cpu(),
                "is_activated": self.is_activated.detach().cpu(),
                "meta": dict(n_cell=self.n_cell, n_smooth=self.n_smooth, n_node=self.n_node,
                             n_NC=self.n_NC, d_scaling=self.d_scaling,
                             mat_energy_scaling=self.mat_energy_scaling, gy_ebc=self.gy_ebc),
            },
            os.path.join(out_dir, f"{tag}_checkpoint.pt"),
        )
        with open(os.path.join(out_dir, f"_0_{tag}_model_NC.pkl"), "wb") as fh:
            pickle.dump([self._np(lin.weight.t()) for lin in self.model.layers]
                        + [self._np(lin.bias) for lin in self.model.layers], fh)


# =====================================================================
# plotting (headless): z = 0 slice, mirrors Plot_3D_Slice
# =====================================================================
def plot_slice(prob, values, fname, z_slice=0.0, tol=1e-3, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X = prob._np(prob.X_cell)
    v = np.asarray(values).reshape(-1)
    mask = np.abs(X[:, 2] - z_slice) < tol
    if mask.sum() < 3:
        return
    fig = plt.figure(figsize=(4, 4))
    sc = plt.tripcolor(X[mask, 0], X[mask, 1], v[mask], cmap="jet")
    plt.colorbar(sc)
    plt.title(title or f"slice z={z_slice}")
    plt.axis("equal")
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_loss(histories, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 5))
    offset = 0
    for label, h in histories:
        if len(h) == 0:
            continue
        xs = np.arange(offset, offset + len(h))
        plt.semilogy(xs, np.clip(h, 1e-30, None), label=f"{label} ({len(h)})")
        offset += len(h)
    plt.xlabel("iteration"); plt.ylabel("total potential energy (loss)")
    plt.grid(True, which="both"); plt.legend()
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)


# =====================================================================
# main
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input-dir", default="Input_3Dbimat_Node396_Cell2662")
    ap.add_argument("--out-dir", default=None,
                    help="default: <input-dir>/Results_40NR_4hidden_5bases")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--epochs-rk", type=int, default=20000)
    ap.add_argument("--epochs-nnrk", type=int, default=30000)
    ap.add_argument("--lbfgs-iters", type=int, default=2000)
    ap.add_argument("--lr-rk", type=float, default=1e-3)
    ap.add_argument("--lr-nnrk", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gy-ebc", type=float, default=0.01)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--dist-backend", default="auto", choices=["auto", "nccl", "gloo"],
                    help="torch.distributed backend (auto: nccl on GPU, gloo on CPU)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run (200/200/50 iters) to validate the pipeline end-to-end")
    args = ap.parse_args()

    if args.smoke:
        args.epochs_rk, args.epochs_nnrk, args.lbfgs_iters = 200, 200, 50

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    dist = Distributed(backend=args.dist_backend, force_cpu=(args.device == "cpu"))
    device = dist.device

    out_dir = args.out_dir or os.path.join(args.input_dir, "Results_40NR_4hidden_5bases")
    final_dir = os.path.join(out_dir, "Final")
    if dist.is_main:
        os.makedirs(final_dir, exist_ok=True)
    dist.barrier()

    if dist.is_main:
        tag = f"distributed (world_size={dist.world_size})" if dist.distributed else "single process"
        print(f"device={device} dtype={dtype} | {tag}", flush=True)
        if device.type == "cuda":
            print("GPU:", torch.cuda.get_device_name(dist.local_rank), flush=True)

    nHL, nNR, actv = 5, [40, 40, 40, 40, 5], ["elu"] * 5

    t0 = time.time()
    prob = NNRK(args.input_dir, device, dtype, nHL, nNR, actv,
                gy_ebc=args.gy_ebc, seed=args.seed, dist=dist)
    if dist.distributed:
        print(f"[rank {prob.rank}/{prob.world_size}] local cells={len(prob.cell_ids)} "
              f"smooth pts={len(prob.smooth_ids)} (global {prob.n_cell}/{prob.n_smooth})", flush=True)
    if dist.is_main:
        print(f"loaded data in {time.time() - t0:.1f}s | "
              f"n_cell={prob.n_cell} n_smooth={prob.n_smooth} n_node={prob.n_node} "
              f"n_ebc=[{prob.n_ebc_x},{prob.n_ebc_y},{prob.n_ebc_z}] n_NC={prob.n_NC}", flush=True)
        print(f"mat_energy_scaling={prob.mat_energy_scaling:.6e} d_scaling={prob.d_scaling:.3e} "
              f"activated nodes={int(prob.is_activated.sum().item())}", flush=True)
        print("orthogonality (max inner product, pre-training):", prob.orthogonality_check(), flush=True)

    # ---- stage 1: Adam, RK only ----
    ts = time.time()
    h_rk = prob.train_adam_rk(args.epochs_rk, lr=args.lr_rk)
    t_rk = time.time() - ts
    # ---- stage 2: Adam, RK + NN ----
    ts = time.time()
    h_nnrk = prob.train_adam_nnrk(args.epochs_nnrk, lr=args.lr_nnrk)
    t_nnrk = time.time() - ts
    # ---- stage 3: L-BFGS ----
    ts = time.time()
    h_lbfgs = prob.train_lbfgs(max_iter=args.lbfgs_iters)
    t_lbfgs = time.time() - ts

    # ---- outputs / diagnostics (rank 0; uses the full-domain operators) ----
    if dist.is_main:
        prob.write_solution(os.path.join(out_dir, "results_nnrk_adam.txt"))  # post-LBFGS state
        prob.write_solution(os.path.join(out_dir, "results_nnrk_LBFGS.txt"))
        prob.save_enrichment_jacobian(out_dir, "LBFGS")
        prob.save_enrichment(out_dir, "ADAM")
        prob.save_layer_weights(out_dir)
        prob.save_layer_weights(final_dir)
        prob.save_checkpoint(os.path.join(final_dir, "LBFGS"), tag="NNRK")
        np.savetxt(os.path.join(out_dir, "loss_rk.txt"), h_rk)
        np.savetxt(os.path.join(out_dir, "loss_nnrk.txt"), h_nnrk)
        np.savetxt(os.path.join(out_dir, "loss_lbfgs.txt"), h_lbfgs)
        np.savetxt(os.path.join(out_dir, "timings_torch.txt"),
                   np.array([t_rk, t_nnrk, t_lbfgs, time.time() - t0]),
                   header="adam_rk adam_nnrk lbfgs total (seconds)")

        with torch.no_grad():
            ux, uy, uz = prob.total_approx(prob.X_cell, prob.SHP_cell)
            e = prob.strain_nnrk(prob.full)
            s = prob.stress(*e, prob.full.mu, prob.full.lam)
            final_energy = float(prob.energy_nnrk(prob.full))
        print("\n=== FINAL RESULTS ===", flush=True)
        print(f"final total potential energy (loss): {final_energy:.6e}", flush=True)
        print(f"uy range: [{prob._np(uy).min():.4e}, {prob._np(uy).max():.4e}]  "
              f"(applied top EBC = {args.gy_ebc})", flush=True)
        print(f"ux range: [{prob._np(ux).min():.4e}, {prob._np(ux).max():.4e}]", flush=True)
        print(f"s22 range: [{prob._np(s[1]).min():.4e}, {prob._np(s[1]).max():.4e}] MPa", flush=True)
        print(f"orthogonality (post-training): {prob.orthogonality_check()}", flush=True)
        print(f"total wall time: {time.time() - t0:.1f}s", flush=True)

        if not args.no_plots:
            plot_loss([("Adam-RK", h_rk), ("Adam-NNRK", h_nnrk), ("L-BFGS", h_lbfgs)],
                      os.path.join(out_dir, "loss_history.png"))
            plot_slice(prob, prob._np(ux), os.path.join(out_dir, "ux_slice.png"), title="ux (z=0)")
            plot_slice(prob, prob._np(uy), os.path.join(out_dir, "uy_slice.png"), title="uy (z=0)")
            plot_slice(prob, prob._np(s[1]), os.path.join(out_dir, "s22_slice.png"), title="s22 (z=0)")
            plot_slice(prob, prob._np(e[1]), os.path.join(out_dir, "e22_slice.png"), title="e22 (z=0)")
            with torch.no_grad():
                zeta = prob.model(prob.X_cell)
            for k in range(prob.n_NC):
                plot_slice(prob, prob._np(zeta[:, k]), os.path.join(out_dir, f"enrichment_{k}_slice.png"),
                           title=f"enrichment {k} (z=0)")
        print(f"\nartifacts written to: {out_dir}", flush=True)

    dist.barrier()
    dist.finalize()


if __name__ == "__main__":
    main()
