#!/usr/bin/env python
"""
NN-PU / NN-RKPM for 3D bimaterial elasticity -- JAX port of ``nnpu_torch.py``
(itself a port of ``NNPU_Modified2026_SCNI3D.ipynb``, TensorFlow/Keras + TFP).

Same Deep-Energy-Method-with-SCNI physics as the torch version (read its docstring / the
README for the method); this file mirrors it function-for-function in JAX so the three
backends (TensorFlow, PyTorch, JAX) can be benchmarked head to head. The torch port is the
numerical oracle: outputs are written in the *identical* column layout (so ``plot_results.py``
and ``compare_backends.py`` work unchanged), and the converged energy / fields match to a few
significant figures (the per-framework RNG differs, so initial weights -- and hence the exact
trajectory -- are not bit-identical, but the minimizer is).

Design (see the JAX-specific choices in comments below):
  * float64 everywhere (``jax_enable_x64``), set before any array is created.
  * Parameters are a plain pytree (dict); no Flax/Equinox. Optax drives all three stages.
  * Sparse operators are ``jax.experimental.sparse`` BCOO; ``A @ x`` is differentiable in the
    dense operand (the operators are constants -- only the d-coefficients / MLP carry grad).
  * Three stages: Adam (RK only) -> Adam (RK + NN + MLP) -> L-BFGS (all), with a strong-Wolfe
    zoom line search (``optax.scale_by_zoom_linesearch``), the analog of torch's strong_wolfe.
  * Multi-GPU is domain decomposition with a SUM combine (the loss is an integral): the cells
    are sharded across devices and energies combined with ``jax.lax.psum`` -- see
    ``build_global_energy`` and ``distributed_jax.py``.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from typing import NamedTuple


# --- force the CPU platform BEFORE importing jax if --device cpu was requested ---------------
# (Setting JAX_PLATFORMS after jax initializes its backend has no effect; this is the reliable
# way to keep local CPU smoke/validation runs off any visible GPU.)
def _maybe_force_cpu():
    argv = sys.argv
    dev = None
    for i, a in enumerate(argv):
        if a == "--device" and i + 1 < len(argv):
            dev = argv[i + 1]
        elif a.startswith("--device="):
            dev = a.split("=", 1)[1]
    if dev == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")


_maybe_force_cpu()

import numpy as np
import pandas as pd
import scipy.sparse as sp

import jax

jax.config.update("jax_enable_x64", True)  # MUST precede any array creation (else silent f32)

import jax.numpy as jnp
from jax.experimental import sparse as jsparse
import optax
import optax.tree_utils as otu

import partition
from distributed_jax import Distributed


# =====================================================================
# I/O helpers (NumPy / SciPy only -- no jax arrays created here)
# =====================================================================
def read_dense_np(fname):
    """Read a dense CSV (.dat) into a float64 (rows, cols) ndarray."""
    return np.asarray(pd.read_csv(fname, header=None).to_numpy(dtype=np.float64))


def coo_to_bcoo(i, j, v, shape):
    """Build a *coalesced* BCOO from COO triplets (duplicate (i,j) summed, indices sorted).

    scipy's COO->CSR->COO round-trip sums duplicate entries and sorts by (row, col) -- exactly
    what torch's ``.coalesce()`` does -- so the JAX operator equals the torch operator entrywise.
    """
    M = sp.coo_matrix((v, (i, j)), shape=tuple(shape)).tocsr().tocoo()
    idx = np.stack([M.row, M.col], axis=1).astype(np.int32)
    return jsparse.BCOO((jnp.asarray(M.data), jnp.asarray(idx)),
                        shape=tuple(shape), indices_sorted=True, unique_indices=True)


def read_sparse_bcoo(fname, shape):
    i, j, v = partition._read_triplet(fname)
    return coo_to_bcoo(i, j, v, shape)


def load_region_cell(data_dir, n_cell, n_node):
    """Binary support map transposed to (n_node, n_cell) -- the torch ``get_region(SHP_cell)``."""
    i, j, v = partition._read_triplet(os.path.join(data_dir, "shp.dat"))
    mask = sp.coo_matrix((np.ones_like(v), (i, j)), shape=(n_cell, n_node))
    return mask.transpose().tocsr()


# =====================================================================
# Static problem data carried into the (pure) forward model
# =====================================================================
class Glob(NamedTuple):
    """Replicated constants used by the forward path (BCOO operators + dense BC data)."""
    CVC_int_x: object; CVC_int_y: object; CVC_int_z: object
    CVC_ebc_x: object; CVC_ebc_y: object; CVC_ebc_z: object
    dx_ebc: object; dy_ebc: object; dz_ebc: object
    is_activated: object; sum_NC: object
    d_scaling: float


class Ops(NamedTuple):
    """Per-integration-domain operators/arrays (full domain, or a per-device shard)."""
    SHP_smooth: object; P1: object; P2: object; P3: object
    WT: object; mu: object; lam: object; X_smooth: object


# =====================================================================
# Forward model (pure functions -- mirror of NNRK's methods; jit/grad friendly)
# =====================================================================
def mlp_apply(mlp, x):
    """EnrichmentMLP: ELU after *every* layer (incl. the last), faithful to the Keras build."""
    for layer in mlp:
        x = jax.nn.elu(x @ layer["W"] + layer["b"])
    return x


def rk_approx(shp, d, CVC=None, d_ebc=None, CVC_ebc=None, act=None):
    if CVC is None:
        d_full = d
    else:
        d_full = CVC @ d
        if d_ebc is not None:
            d_full = d_full + CVC_ebc @ d_ebc
    if act is not None:
        d_full = d_full * act
    return shp @ d_full


def nn_approx(zeta, shp, d_NN, CVC_int, is_activated, sum_NC):
    v = rk_approx(shp, d_NN, CVC=CVC_int, act=is_activated)
    return (zeta * v) @ sum_NC


def rk_only(params, glob, shp):
    s, d = glob.d_scaling, params["d"]
    ux = rk_approx(shp, s * d["dx_int"], CVC=glob.CVC_int_x, d_ebc=glob.dx_ebc, CVC_ebc=glob.CVC_ebc_x)
    uy = rk_approx(shp, s * d["dy_int"], CVC=glob.CVC_int_y, d_ebc=glob.dy_ebc, CVC_ebc=glob.CVC_ebc_y)
    uz = rk_approx(shp, s * d["dz_int"], CVC=glob.CVC_int_z, d_ebc=glob.dz_ebc, CVC_ebc=glob.CVC_ebc_z)
    return ux, uy, uz


def total_approx(params, glob, x, shp):
    s, d, mlp = glob.d_scaling, params["d"], params["mlp"]
    ux, uy, uz = rk_only(params, glob, shp)
    zeta = mlp_apply(mlp, x)
    ux = ux + nn_approx(zeta, shp, s * d["dx_NN"], glob.CVC_int_x, glob.is_activated, glob.sum_NC)
    uy = uy + nn_approx(zeta, shp, s * d["dy_NN"], glob.CVC_int_y, glob.is_activated, glob.sum_NC)
    uz = uz + nn_approx(zeta, shp, s * d["dz_NN"], glob.CVC_int_z, glob.is_activated, glob.sum_NC)
    return ux, uy, uz


def total_approx_nn(params, glob, x, shp):
    s, d, mlp = glob.d_scaling, params["d"], params["mlp"]
    zeta = mlp_apply(mlp, x)
    ux = nn_approx(zeta, shp, s * d["dx_NN"], glob.CVC_int_x, glob.is_activated, glob.sum_NC)
    uy = nn_approx(zeta, shp, s * d["dy_NN"], glob.CVC_int_y, glob.is_activated, glob.sum_NC)
    uz = nn_approx(zeta, shp, s * d["dz_NN"], glob.CVC_int_z, glob.is_activated, glob.sum_NC)
    return ux, uy, uz, zeta


def smoothed_strain(ux, uy, uz, ops):
    P1, P2, P3 = ops.P1, ops.P2, ops.P3
    exx = P1 @ ux; eyy = P2 @ uy; ezz = P3 @ uz
    gxy = P2 @ ux + P1 @ uy
    gyz = P3 @ uy + P2 @ uz
    gxz = P3 @ ux + P1 @ uz
    return exx, eyy, ezz, gxy, gyz, gxz


def stress(exx, eyy, ezz, gxy, gyz, gxz, mu, lam):
    M = 2.0 * mu + lam
    sxx = M * exx + lam * (eyy + ezz)
    syy = M * eyy + lam * (exx + ezz)
    szz = M * ezz + lam * (exx + eyy)
    return sxx, syy, szz, mu * gxy, mu * gyz, mu * gxz


def energy_density(exx, eyy, ezz, gxy, gyz, gxz, sxx, syy, szz, sxy, syz, sxz):
    return 0.5 * (exx * sxx + eyy * syy + ezz * szz + gxy * sxy + gyz * syz + gxz * sxz)


def _energy(strains, ops, scaling):
    s = stress(*strains, ops.mu, ops.lam)
    psi = energy_density(*strains, *s)
    # local (per-shard) potential energy; the global energy is the SUM over shards/devices.
    return scaling * (psi * ops.WT).sum()


def energy_rk(params, glob, ops, scaling):
    return _energy(smoothed_strain(*rk_only(params, glob, ops.SHP_smooth), ops), ops, scaling)


def energy_nnrk(params, glob, ops, scaling):
    u = total_approx(params, glob, ops.X_smooth, ops.SHP_smooth)
    return _energy(smoothed_strain(*u, ops), ops, scaling)


# =====================================================================
# Parameter initialization (pytree)
# =====================================================================
def init_params(key, n_node, n_ebc, nNR, nHL):
    """RK/enrichment coeffs init to zero; MLP weights U(-0.1,0.1) first layer, U(-1,1) rest,
    biases zero -- the same scheme as ``EnrichmentMLP`` (the RNG draw differs from torch's)."""
    n_NC = nNR[nHL - 1]
    z = lambda r, c: jnp.zeros((r, c), jnp.float64)
    d = {
        "dx_int": z(n_node - n_ebc["x"], 1),
        "dy_int": z(n_node - n_ebc["y"], 1),
        "dz_int": z(n_node - n_ebc["z"], 1),
        "dx_NN": z(n_node - n_ebc["x"], n_NC),
        "dy_NN": z(n_node - n_ebc["y"], n_NC),
        "dz_NN": z(n_node - n_ebc["z"], n_NC),
    }
    mlp, in_dim = [], 3
    for i in range(nHL):
        key, k = jax.random.split(key)
        a = 2.0 / 20.0 if i == 0 else 1.0
        W = jax.random.uniform(k, (in_dim, nNR[i]), jnp.float64, -a, a)
        mlp.append({"W": W, "b": jnp.zeros((nNR[i],), jnp.float64)})
        in_dim = nNR[i]
    return {"d": d, "mlp": mlp}


# =====================================================================
# Problem assembly
# =====================================================================
class Problem:
    """Container for everything ``main`` needs (data, params, energy closures, output sets)."""


def build_problem(data_dir, nHL, nNR, actv, gy_ebc, seed, dist, dense=False):
    P = Problem()
    P.data_dir, P.gy_ebc = data_dir, gy_ebc
    P.nNR, P.n_NC = list(nNR), nNR[nHL - 1]

    # ---- point sets / per-cell data (NumPy) ----
    X_cell = read_dense_np(os.path.join(data_dir, "x.dat"))
    X_smooth = read_dense_np(os.path.join(data_dir, "x_smoothing.dat"))
    X_node = read_dense_np(os.path.join(data_dir, "x_node.dat"))
    Vlabel = read_dense_np(os.path.join(data_dir, "Vlabel.dat"))
    WT = read_dense_np(os.path.join(data_dir, "WT.dat"))
    WT_smooth = read_dense_np(os.path.join(data_dir, "WT_smooth.dat"))
    n_cell, n_smooth, n_node = X_cell.shape[0], X_smooth.shape[0], X_node.shape[0]
    P.n_cell, P.n_smooth, P.n_node = n_cell, n_smooth, n_node

    ns = read_dense_np(os.path.join(data_dir, "bc_info.dat")).astype(np.int64)
    n_ebc = {"x": int(ns[0, 0]), "y": int(ns[0, 1]), "z": int(ns[0, 2])}
    P.n_ebc = n_ebc

    # ---- geometry extents ----
    y_max = float(X_smooth[:, 1].max())
    x_min, x_max = float(X_smooth[:, 0].min()), float(X_smooth[:, 0].max())
    y_min = float(X_smooth[:, 1].min())
    z_min, z_max = float(X_smooth[:, 2].min()), float(X_smooth[:, 2].max())

    # ---- bimaterial moduli (per cell) ----
    Em, num_, Ei, nui = 10400.0, 0.3, 52000.0, 0.3
    is_matrix = Vlabel == 0
    E = np.where(is_matrix, Em, Ei)
    nu = np.where(is_matrix, num_, nui)
    mat_mu = E / 2.0 / (1.0 + nu)
    mat_lam = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu)
    vol = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    mat_energy_scaling_ref = vol / E.max()
    P.scaling = float(mat_energy_scaling_ref / (gy_ebc ** 2) / WT.sum())
    d_scaling = float(gy_ebc)

    # ---- essential BCs (the dual dense read of CVC_ebc_y.dat: [node, ebc_col, val] rows) ----
    dx_ebc = np.zeros((n_ebc["x"], 1)); dy_ebc = np.zeros((n_ebc["y"], 1)); dz_ebc = np.zeros((n_ebc["z"], 1))
    ey = read_dense_np(os.path.join(data_dir, "CVC_ebc_y.dat"))
    for r in range(ey.shape[0]):
        node_id = int(ey[r, 0])
        if X_node[node_id, 1] > y_max - 1e-6:
            dy_ebc[int(ey[r, 1]), 0] = 1.0
    dy_ebc = dy_ebc * gy_ebc

    # ---- activation mask (y-band of width 0.1 around y=0, mapped to supporting nodes) ----
    region_cell = load_region_cell(data_dir, n_cell, n_node)
    dy = X_cell[:, 1] - 0.0
    cell_mask = ((dy > -0.05) & (dy < 0.05)).astype(np.float64).reshape(-1, 1)
    node_val = region_cell @ cell_mask
    is_activated = node_val / (node_val + 1e-16)  # -> {0, 1}

    # ---- to JAX (float64) ----
    def to_sparse(fname, shape):
        b = read_sparse_bcoo(os.path.join(data_dir, fname), shape)
        return b.todense() if dense else b

    glob = Glob(
        CVC_int_x=to_sparse("CVC_int_x.dat", (n_node, n_node - n_ebc["x"])),
        CVC_int_y=to_sparse("CVC_int_y.dat", (n_node, n_node - n_ebc["y"])),
        CVC_int_z=to_sparse("CVC_int_z.dat", (n_node, n_node - n_ebc["z"])),
        CVC_ebc_x=to_sparse("CVC_ebc_x.dat", (n_node, n_ebc["x"])),
        CVC_ebc_y=to_sparse("CVC_ebc_y.dat", (n_node, n_ebc["y"])),
        CVC_ebc_z=to_sparse("CVC_ebc_z.dat", (n_node, n_ebc["z"])),
        dx_ebc=jnp.asarray(dx_ebc), dy_ebc=jnp.asarray(dy_ebc), dz_ebc=jnp.asarray(dz_ebc),
        is_activated=jnp.asarray(is_activated), sum_NC=jnp.ones((P.n_NC, 1), jnp.float64),
        d_scaling=d_scaling,
    )
    P.glob = glob

    # full-domain operators (used for the energy on a single device, and always for output)
    full = Ops(
        SHP_smooth=to_sparse("shp_smoothing.dat", (n_smooth, n_node)),
        P1=to_sparse("P1.dat", (n_cell, n_smooth)),
        P2=to_sparse("P2.dat", (n_cell, n_smooth)),
        P3=to_sparse("P3.dat", (n_cell, n_smooth)),
        WT=jnp.asarray(WT), mu=jnp.asarray(mat_mu), lam=jnp.asarray(mat_lam),
        X_smooth=jnp.asarray(X_smooth),
    )
    P.full = full
    P.SHP_cell = to_sparse("shp.dat", (n_cell, n_node))
    P.X_cell = jnp.asarray(X_cell); P.X_node = jnp.asarray(X_node)
    P.Vlabel = Vlabel
    # mass matrices + smooth weights for the orthogonality diagnostic (densified; 396x396)
    P.WT_smooth = jnp.asarray(WT_smooth)
    P.M_c = read_sparse_bcoo(os.path.join(data_dir, "M_c.dat"), (n_node, n_node)).todense()
    P.M_s = read_sparse_bcoo(os.path.join(data_dir, "M_s.dat"), (n_node, n_node)).todense()

    # ---- parameters (identical seed on every process -> replicated init) ----
    P.params = init_params(jax.random.PRNGKey(seed), n_node, n_ebc, nNR, nHL)

    # ---- energy closures (params -> scalar). Single device: full domain. ----
    P.cell_ids = np.arange(n_cell)
    P.smooth_ids = np.arange(n_smooth)
    if dist is not None and dist.sharded:
        P.energy_rk_fn, P.energy_nnrk_fn = build_global_energy(P, dist, dense)
    else:
        P.energy_rk_fn = lambda params: energy_rk(params, P.glob, P.full, P.scaling)
        P.energy_nnrk_fn = lambda params: energy_nnrk(params, P.glob, P.full, P.scaling)
    return P


# =====================================================================
# Multi-device domain decomposition: shard cells, SUM-combine energies with psum
# =====================================================================
def build_global_energy(P, dist, dense):
    """Return (energy_rk_fn, energy_nnrk_fn) that compute the GLOBAL energy as a SUM over a
    device-sharded cell partition, via ``shard_map`` + ``jax.lax.psum``.

    Each device owns a contiguous slice of cells (``partition.partition_cells``) and only its
    rows of P1/P2/P3 plus the smoothing points they reference (``partition.build_local_triplets``).
    Operators are **zero-padded to a common shape** so the program is SPMD-identical across devices
    -- required for the in-jit ``psum``, which is what lets the L-BFGS line search stay globally
    consistent (a host-side all-reduce cannot run inside the line-search ``lax`` loop). Zero rows
    (WT=0) and zero columns are energy-neutral. The loss is a spatial integral, so the cross-device
    combine is a SUM (``psum``), not a mean -- matching distributed.py's torch semantics.

    The padded per-device operators are kept **dense**: at ~n_cell/n_devices rows the memory is
    small, dense arrays shard cleanly (no batched-sparse machinery), and the energy code's ``A @ x``
    is agnostic to dense vs BCOO. The small CVC operators are densified too so the shard_map body
    is pure-dense.
    """
    if dense:
        raise SystemExit("--dense is single-device only; drop it for the multi-GPU (sharded) path")
    from functools import partial
    from jax.sharding import NamedSharding, PartitionSpec as Pspec

    nd = dist.n_devices
    mesh = dist.make_mesh("cells")
    paths = {k: os.path.join(P.data_dir, f) for k, f in
             (("P1", "P1.dat"), ("P2", "P2.dat"), ("P3", "P3.dat"), ("SHP_smooth", "shp_smoothing.dat"))}

    # 1) per-device local triplets (NumPy) + max shard sizes
    locs = [partition.build_local_triplets(paths, partition.partition_cells(P.n_cell, r, nd),
                                           P.n_cell, P.n_smooth, P.n_node) for r in range(nd)]
    maxC = max(len(t["cell_ids"]) for t in locs)
    maxS = max(len(t["smooth_ids"]) for t in locs)
    WT_np = np.asarray(P.full.WT); mu_np = np.asarray(P.full.mu); lam_np = np.asarray(P.full.lam)
    Xs_np = np.asarray(P.full.X_smooth)

    # 2) stack per-device zero-padded shards along a leading device axis
    def stack_rows(per_dev, width):  # list of (n,k) -> (nd, width, k)
        out = np.zeros((nd, width, per_dev[0].shape[1]), np.float64)
        for r, a in enumerate(per_dev):
            out[r, :a.shape[0]] = a
        return out

    def stack_op(key, shape2d):      # local triplets -> dense (nd, *shape2d), zero-padded
        out = np.zeros((nd, shape2d[0], shape2d[1]), np.float64)
        for r, t in enumerate(locs):
            i, j, v, _ = t[key]
            out[r] = sp.coo_matrix((v, (i, j)), shape=shape2d).toarray()
        return out

    WT_s = stack_rows([WT_np[t["cell_ids"]] for t in locs], maxC)
    mu_s = stack_rows([mu_np[t["cell_ids"]] for t in locs], maxC)
    lam_s = stack_rows([lam_np[t["cell_ids"]] for t in locs], maxC)
    Xs_s = stack_rows([Xs_np[t["smooth_ids"]] for t in locs], maxS)
    P1_s = stack_op("P1", (maxC, maxS)); P2_s = stack_op("P2", (maxC, maxS))
    P3_s = stack_op("P3", (maxC, maxS)); SHP_s = stack_op("SHP_smooth", (maxS, P.n_node))

    # 3) shard each stacked array along the device (cells) axis
    shard0 = NamedSharding(mesh, Pspec("cells"))
    P1_s, P2_s, P3_s, SHP_s, WT_s, mu_s, lam_s, Xs_s = (
        jax.device_put(jnp.asarray(a), shard0)
        for a in (P1_s, P2_s, P3_s, SHP_s, WT_s, mu_s, lam_s, Xs_s))

    # densify the small replicated CVC operators so the shard_map body is pure-dense
    densify = lambda b: b.todense() if isinstance(b, jsparse.BCOO) else b
    gd = P.glob._replace(
        CVC_int_x=densify(P.glob.CVC_int_x), CVC_int_y=densify(P.glob.CVC_int_y),
        CVC_int_z=densify(P.glob.CVC_int_z), CVC_ebc_x=densify(P.glob.CVC_ebc_x),
        CVC_ebc_y=densify(P.glob.CVC_ebc_y), CVC_ebc_z=densify(P.glob.CVC_ebc_z))
    scaling = P.scaling

    def per_device(params, P1, P2, P3, SHP, WT, mu, lam, Xs, nnrk):
        ops = Ops(SHP_smooth=SHP[0], P1=P1[0], P2=P2[0], P3=P3[0],
                  WT=WT[0], mu=mu[0], lam=lam[0], X_smooth=Xs[0])
        e = energy_nnrk(params, gd, ops, scaling) if nnrk else energy_rk(params, gd, ops, scaling)
        return jax.lax.psum(e, "cells")

    def make(nnrk):
        smap = jax.shard_map(
            partial(per_device, nnrk=nnrk), mesh=mesh,
            in_specs=(Pspec(),) + (Pspec("cells"),) * 8, out_specs=Pspec(), check_vma=False)
        return lambda params: smap(params, P1_s, P2_s, P3_s, SHP_s, WT_s, mu_s, lam_s, Xs_s)

    return make(False), make(True)


# =====================================================================
# Training stages
# =====================================================================
def _adam_step(energy_fn, opt):
    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(energy_fn)(params)  # grads already global (psum energy)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss
    return step


def train_adam(energy_fn, params, num_epochs, lr, *, label, is_main, log_every=100,
               es_threshold=1e-12, es_patience=5, es_start=0, track_best=False):
    opt = optax.adam(lr, eps=1e-6)            # eps=1e-6 matches torch (Optax default is 1e-8)
    opt_state = opt.init(params)
    step = _adam_step(energy_fn, opt)
    history = np.zeros(num_epochs)
    prev, counter = float("inf"), 0
    best_loss, best_params = float("inf"), params
    if is_main:
        print(f"Starting {label} optimization ...", flush=True)
    epoch = 0
    for epoch in range(num_epochs):
        params, opt_state, loss = step(params, opt_state)
        lv = float(loss)
        history[epoch] = lv
        if epoch > es_start:
            counter = counter + 1 if abs(prev - lv) < es_threshold else 0
            if counter >= es_patience:
                if is_main:
                    print(f"  early stop @ epoch {epoch} (|dloss|<{es_threshold:.1e})", flush=True)
                history = history[: epoch + 1]
                break
        prev = lv
        if track_best and lv < best_loss:
            best_loss, best_params = lv, params
        if is_main and epoch % log_every == 0:
            print(f"  [{label}] epoch {epoch:6d} | loss {lv:.6e}", flush=True)
    else:
        history = history[: epoch + 1]
    if track_best:
        params = best_params
        if is_main:
            print(f"{label} done. best loss {best_loss:.6e}", flush=True)
    elif is_main:
        print(f"{label} done. final loss {history[-1]:.6e}", flush=True)
    return params, history


def train_lbfgs(energy_fn, params, max_iter, *, history_size=20, is_main, log_every=100,
                tol_grad=1e-12, tol_change=1e-16, linesearch_steps=50):
    """L-BFGS with a strong-Wolfe zoom line search. ``max_iter`` is the number of *outer*
    L-BFGS iterations (one ``opt.update`` each) -- the analog of torch's LBFGS ``max_iter``.
    The recorded history is one entry per outer iteration (torch records per function eval), so
    compare the *final* energy across backends, not the history arrays."""
    linesearch = optax.scale_by_zoom_linesearch(max_linesearch_steps=linesearch_steps)
    opt = optax.lbfgs(memory_size=history_size, linesearch=linesearch)
    value_and_grad = optax.value_and_grad_from_state(energy_fn)

    @jax.jit
    def step(params, state):
        value, grad = value_and_grad(params, state=state)
        updates, state = opt.update(grad, state, params, value=value, grad=grad, value_fn=energy_fn)
        params = optax.apply_updates(params, updates)
        return params, state, value, otu.tree_norm(grad)

    state = opt.init(params)
    history, prev = [], float("inf")
    if is_main:
        print("Starting L-BFGS optimization ...", flush=True)
    tic = time.time()
    it = 0
    for it in range(max_iter):
        params, state, value, gnorm = step(params, state)
        v = float(value)
        history.append(v)
        if is_main and (it + 1) % log_every == 0:
            print(f"  [LBFGS] iter {it + 1:6d} | loss {v:.6e}", flush=True)
        if float(gnorm) < tol_grad or abs(prev - v) < tol_change:
            break
        prev = v
    final = float(energy_fn(params))   # converged energy after the last update
    history.append(final)
    if is_main:
        print(f"L-BFGS done in {time.time() - tic:.1f}s, {it + 1} iters, final loss {final:.6e}", flush=True)
    return params, np.array(history)


# =====================================================================
# Orthogonality diagnostic (RNG-independent sanity check; mirrors notebook cell 12)
# =====================================================================
def orthonormalize_nn_basis(mlp, x, weight, shp_dense, M_dense):
    zeta = mlp_apply(mlp, x)
    zeta = zeta / (jnp.sqrt((weight * zeta ** 2).sum(0)) + 1e-16)
    bk = shp_dense.T @ (weight * zeta)
    ck = jnp.linalg.solve(M_dense, bk)
    zeta_ortho = zeta - shp_dense @ ck
    return zeta_ortho / (jnp.sqrt((weight * zeta_ortho ** 2).sum(0)) + 1e-16)


def orthogonality_check(P, params):
    mlp = params["mlp"]
    shp_cell = P.SHP_cell.todense() if isinstance(P.SHP_cell, jsparse.BCOO) else P.SHP_cell
    shp_smooth = P.full.SHP_smooth.todense() if isinstance(P.full.SHP_smooth, jsparse.BCOO) else P.full.SHP_smooth
    WT = read_dense_np(os.path.join(P.data_dir, "WT.dat"))
    out = {}
    for name, x, w, shp, M in (("cell", P.X_cell, jnp.asarray(WT), shp_cell, P.M_c),
                               ("smooth", P.full.X_smooth, P.WT_smooth, shp_smooth, P.M_s)):
        zo = orthonormalize_nn_basis(mlp, x, w, shp, M)
        inner = shp.T @ (w * zo)
        out[name] = float(jnp.abs(inner).max())
    return out


# =====================================================================
# Outputs (NumPy; identical filenames / column layout to nnpu_torch.py)
# =====================================================================
def _np(a):
    return np.asarray(a)


def write_solution(P, params, fname):
    ux, uy, uz = total_approx(params, P.glob, P.X_cell, P.SHP_cell)
    _, uy_nn, _, _ = total_approx_nn(params, P.glob, P.X_cell, P.SHP_cell)
    e11, e22, e33, g12, g23, g13 = smoothed_strain(
        *total_approx(params, P.glob, P.full.X_smooth, P.full.SHP_smooth), P.full)
    cols = [P.X_cell, ux, uy, uz, uy_nn, e11, e22, e33, g23, g13, g12]
    out = np.concatenate([_np(c).reshape(P.n_cell, -1) for c in cols], axis=1)
    np.savetxt(fname, out)


def save_enrichment(P, params, out_dir, tag):
    for name, x in (("cell", P.X_cell), ("smooth", P.full.X_smooth), ("node", P.X_node)):
        np.savetxt(os.path.join(out_dir, f"Enrichment_{name}_{tag}.txt"), _np(mlp_apply(params["mlp"], x)))


def save_enrichment_jacobian(P, params, out_dir, tag):
    """zeta and d zeta/dx at cell/smooth/node points (mirrors notebook cells 35-37)."""
    mlp = params["mlp"]
    jac = jax.jit(jax.vmap(jax.jacrev(lambda xr: mlp_apply(mlp, xr))))
    for name, X in (("cell", P.X_cell), ("smooth", P.full.X_smooth), ("node", P.X_node)):
        G = jac(X)  # (N, n_NC, 3)
        np.savetxt(os.path.join(out_dir, f"Enrichment_{name}_{tag}.txt"), _np(mlp_apply(mlp, X)))
        for dcomp, comp in enumerate(("dx", "dy", "dz")):
            np.savetxt(os.path.join(out_dir, f"final_{name}_{tag}_{comp}.txt"), _np(G[:, :, dcomp]))


def save_layer_weights(P, params, out_dir):
    # W is stored (in_dim, units) and applied x@W, so it is written directly (matches the torch
    # on-disk layout, which transposes nn.Linear's (units, in_dim) weight to (in_dim, units)).
    for i, layer in enumerate(params["mlp"]):
        np.savetxt(os.path.join(out_dir, f"layer_{i}_weights.txt"), _np(layer["W"]), delimiter=" ")
        np.savetxt(os.path.join(out_dir, f"layer_{i}_biases.txt"), _np(layer["b"]), delimiter=" ")


def save_checkpoint(P, params, out_dir, tag="NNRK"):
    os.makedirs(out_dir, exist_ok=True)
    mlp = params["mlp"]
    ckpt = {
        "mlp": [{"W": _np(l["W"]), "b": _np(l["b"])} for l in mlp],
        **{k: _np(v) for k, v in params["d"].items()},
        "is_activated": _np(P.glob.is_activated),
        "meta": dict(n_cell=P.n_cell, n_smooth=P.n_smooth, n_node=P.n_node, n_NC=P.n_NC,
                     d_scaling=P.glob.d_scaling, mat_energy_scaling=P.scaling, gy_ebc=P.gy_ebc),
    }
    with open(os.path.join(out_dir, f"{tag}_checkpoint.pkl"), "wb") as fh:
        pickle.dump(ckpt, fh)
    with open(os.path.join(out_dir, f"_0_{tag}_model_NC.pkl"), "wb") as fh:
        pickle.dump([_np(l["W"]) for l in mlp] + [_np(l["b"]) for l in mlp], fh)


# =====================================================================
# plotting (headless), mirrors nnpu_torch.plot_* for artifact parity
# =====================================================================
def plot_slice(P, values, fname, z_slice=0.0, tol=1e-3, title=""):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    X = _np(P.X_cell); v = np.asarray(values).reshape(-1)
    mask = np.abs(X[:, 2] - z_slice) < tol
    if mask.sum() < 3:
        return
    fig = plt.figure(figsize=(4, 4))
    sc = plt.tripcolor(X[mask, 0], X[mask, 1], v[mask], cmap="jet")
    plt.colorbar(sc); plt.title(title or f"slice z={z_slice}"); plt.axis("equal")
    fig.savefig(fname, bbox_inches="tight", dpi=150); plt.close(fig)


def plot_loss(histories, fname):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9, 5)); offset = 0
    for label, h in histories:
        if len(h) == 0:
            continue
        xs = np.arange(offset, offset + len(h))
        plt.semilogy(xs, np.clip(h, 1e-30, None), label=f"{label} ({len(h)})")
        offset += len(h)
    plt.xlabel("iteration"); plt.ylabel("total potential energy (loss)")
    plt.grid(True, which="both"); plt.legend()
    fig.savefig(fname, bbox_inches="tight", dpi=150); plt.close(fig)


# =====================================================================
# main
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input-dir", default="Input_3Dbimat_Node396_Cell2662")
    ap.add_argument("--out-dir", default=None,
                    help="default: <input-dir>/Results_40NR_4hidden_5bases_jax")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--epochs-rk", type=int, default=20000)
    ap.add_argument("--epochs-nnrk", type=int, default=30000)
    ap.add_argument("--lbfgs-iters", type=int, default=2000)
    ap.add_argument("--lr-rk", type=float, default=1e-3)
    ap.add_argument("--lr-nnrk", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gy-ebc", type=float, default=0.01)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--dense", action="store_true",
                    help="use dense operators instead of BCOO (single device only; perf comparison)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run (200/200/50 iters) to validate the pipeline end-to-end")
    args = ap.parse_args()

    if args.smoke:
        args.epochs_rk, args.epochs_nnrk, args.lbfgs_iters = 200, 200, 50

    dist = Distributed(force_cpu=(args.device == "cpu"))
    is_main = dist.is_main

    out_dir = args.out_dir or os.path.join(args.input_dir, "Results_40NR_4hidden_5bases_jax")
    final_dir = os.path.join(out_dir, "Final")
    if is_main:
        os.makedirs(final_dir, exist_ok=True)
    dist.barrier()

    if is_main:
        plat = jax.devices()[0].platform
        tag = f"sharded (n_devices={dist.n_devices})" if dist.sharded else "single device"
        proc = f", {dist.num_processes} processes" if dist.multiprocess else ""
        x64 = jnp.zeros(()).dtype == jnp.float64
        print(f"backend={plat} float64={x64} | {tag}{proc}", flush=True)

    nHL, nNR, actv = 5, [40, 40, 40, 40, 5], ["elu"] * 5
    t0 = time.time()
    P = build_problem(args.input_dir, nHL, nNR, actv, gy_ebc=args.gy_ebc, seed=args.seed,
                      dist=dist, dense=args.dense)
    if is_main:
        print(f"loaded data in {time.time() - t0:.1f}s | n_cell={P.n_cell} n_smooth={P.n_smooth} "
              f"n_node={P.n_node} n_ebc=[{P.n_ebc['x']},{P.n_ebc['y']},{P.n_ebc['z']}] n_NC={P.n_NC}",
              flush=True)
        print(f"mat_energy_scaling={P.scaling:.6e} d_scaling={P.glob.d_scaling:.3e} "
              f"activated nodes={int(_np(P.glob.is_activated).sum())}", flush=True)
        print("orthogonality (pre-training):", orthogonality_check(P, P.params), flush=True)

    params = P.params
    timings = {}
    t = time.time()
    params, h_rk = train_adam(P.energy_rk_fn, params, args.epochs_rk, args.lr_rk,
                              label="RK", is_main=is_main, es_threshold=1e-12, es_start=0)
    timings["adam_rk"] = time.time() - t
    t = time.time()
    params, h_nnrk = train_adam(P.energy_nnrk_fn, params, args.epochs_nnrk, args.lr_nnrk,
                                label="NNRK", is_main=is_main, es_threshold=1e-11, es_start=100,
                                track_best=True)
    timings["adam_nnrk"] = time.time() - t
    t = time.time()
    params, h_lbfgs = train_lbfgs(P.energy_nnrk_fn, params, args.lbfgs_iters, is_main=is_main)
    timings["lbfgs"] = time.time() - t

    if is_main:
        write_solution(P, params, os.path.join(out_dir, "results_nnrk_adam.txt"))
        write_solution(P, params, os.path.join(out_dir, "results_nnrk_LBFGS.txt"))
        save_enrichment_jacobian(P, params, out_dir, "LBFGS")
        save_enrichment(P, params, out_dir, "ADAM")
        save_layer_weights(P, params, out_dir)
        save_layer_weights(P, params, final_dir)
        save_checkpoint(P, params, os.path.join(final_dir, "LBFGS"), tag="NNRK")
        np.savetxt(os.path.join(out_dir, "loss_rk.txt"), h_rk)
        np.savetxt(os.path.join(out_dir, "loss_nnrk.txt"), h_nnrk)
        np.savetxt(os.path.join(out_dir, "loss_lbfgs.txt"), h_lbfgs)
        np.savetxt(os.path.join(out_dir, "timings_jax.txt"),
                   np.array([timings["adam_rk"], timings["adam_nnrk"], timings["lbfgs"],
                             time.time() - t0]),
                   header="adam_rk adam_nnrk lbfgs total (seconds)")

        ux, uy, uz = total_approx(params, P.glob, P.X_cell, P.SHP_cell)
        e = smoothed_strain(*total_approx(params, P.glob, P.full.X_smooth, P.full.SHP_smooth), P.full)
        s = stress(*e, P.full.mu, P.full.lam)
        final_energy = float(energy_nnrk(params, P.glob, P.full, P.scaling))
        print("\n=== FINAL RESULTS ===", flush=True)
        print(f"final total potential energy (loss): {final_energy:.6e}", flush=True)
        print(f"uy range: [{_np(uy).min():.4e}, {_np(uy).max():.4e}]  (applied top EBC = {args.gy_ebc})", flush=True)
        print(f"ux range: [{_np(ux).min():.4e}, {_np(ux).max():.4e}]", flush=True)
        print(f"s22 range: [{_np(s[1]).min():.4e}, {_np(s[1]).max():.4e}] MPa", flush=True)
        print(f"orthogonality (post-training): {orthogonality_check(P, params)}", flush=True)
        print(f"stage timings (s): RK={timings['adam_rk']:.1f} NNRK={timings['adam_nnrk']:.1f} "
              f"LBFGS={timings['lbfgs']:.1f} | total wall {time.time() - t0:.1f}s", flush=True)

        if not args.no_plots:
            plot_loss([("Adam-RK", h_rk), ("Adam-NNRK", h_nnrk), ("L-BFGS", h_lbfgs)],
                      os.path.join(out_dir, "loss_history.png"))
            plot_slice(P, _np(ux), os.path.join(out_dir, "ux_slice.png"), title="ux (z=0)")
            plot_slice(P, _np(uy), os.path.join(out_dir, "uy_slice.png"), title="uy (z=0)")
            plot_slice(P, _np(s[1]), os.path.join(out_dir, "s22_slice.png"), title="s22 (z=0)")
            plot_slice(P, _np(e[1]), os.path.join(out_dir, "e22_slice.png"), title="e22 (z=0)")
            zeta = mlp_apply(params["mlp"], P.X_cell)
            for k in range(P.n_NC):
                plot_slice(P, _np(zeta[:, k]), os.path.join(out_dir, f"enrichment_{k}_slice.png"),
                           title=f"enrichment {k} (z=0)")
        print(f"\nartifacts written to: {out_dir}", flush=True)

    dist.barrier()
    dist.finalize()


if __name__ == "__main__":
    main()
